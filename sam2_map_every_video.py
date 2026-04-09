import argparse
import contextlib
import io
import logging
import math
from collections import OrderedDict
from pathlib import Path
import sys

import cv2  # Load before torch in this env.
import numpy as np
import torch
from tqdm.auto import tqdm

from build_rgb_map import GTInstanceMaskExtractor, SAMMaskExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent / "thirdParty" / "segment-anything-2"))

from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import _load_img_as_tensor

INPUT_DIR = Path("data/input/ScanNet")
OUTPUT_DIR = Path("data/output/sam2_map_every_video")
SAM2_MAX_NUM_OBJECTS = 16
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (20, 20, 20)
SAM2_LEVELS = {
    21: ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    22: ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    23: ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    24: ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
SAM2_MODE = "eval"
SAM2_HYDRA_OVERRIDES: tuple[str, ...] = ()
SAM2_APPLY_POSTPROCESSING = True


def parse_map_every(value: str) -> float:
    if value.lower() in {"inf", "infinity"}:
        return math.inf
    return float(int(value))


def list_frames(scene_dir: Path, frame_limit: int | None) -> list[int]:
    color_dir = scene_dir / "color"
    frame_ids = sorted(int(path.stem) for path in color_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if frame_limit is not None:
        frame_ids = [frame_id for frame_id in frame_ids if frame_id < frame_limit]
    return frame_ids


def load_color_frame(scene_dir: Path, frame_id: int) -> np.ndarray:
    for suffix in (".jpg", ".png"):
        path = scene_dir / "color" / f"{frame_id}{suffix}"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    raise FileNotFoundError(scene_dir / "color" / f"{frame_id}.jpg")


def color_for_id(label: int) -> np.ndarray:
    if label <= 0:
        return np.zeros(3, dtype=np.uint8)
    x = (int(label) * 2654435761) & 0xFFFFFFFF
    return np.array(
        [
            np.uint8((x >> 0) & 255),
            np.uint8((x >> 8) & 255),
            np.uint8((x >> 16) & 255),
        ],
        dtype=np.uint8,
    )


def colorize_instance_map(labels: np.ndarray) -> np.ndarray:
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    unique_labels = np.unique(labels)
    for label in unique_labels.tolist():
        if label <= 0:
            continue
        colored[labels == label] = color_for_id(int(label))
    return colored


def overlay_header(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), TEXT_BG, thickness=-1)
    cv2.putText(canvas, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_COLOR, 2, cv2.LINE_AA)
    return canvas


def load_gt_labels(scene_dir: Path, frame_id: int) -> np.ndarray:
    label_path = scene_dir / "instance-filt" / f"{frame_id}.png"
    labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise FileNotFoundError(label_path)
    return labels.astype(np.int32, copy=False)


def extract_seed_labels(
    scene_dir: Path,
    frame_id: int,
    image: np.ndarray,
    use_inst_gt: bool,
    gt_extractor: GTInstanceMaskExtractor | None,
    sam_extractor: SAMMaskExtractor | None,
) -> tuple[np.ndarray, dict[int, int]]:
    if use_inst_gt:
        if gt_extractor is None:
            raise RuntimeError("GT instance extractor was not initialized.")
        labels = gt_extractor.extract_labels(frame_id, image.shape[:2])
        valid = labels >= 0
        if valid.any():
            raw_labels = load_gt_labels(scene_dir, frame_id)
            if raw_labels.shape[:2] != image.shape[:2]:
                raw_labels = cv2.resize(raw_labels, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            raw_ids = np.unique(raw_labels[raw_labels > 0])
            display_map = {int(local_id): int(raw_id) for local_id, raw_id in enumerate(raw_ids.tolist())}
        else:
            display_map = {}
        return labels, display_map
    if sam_extractor is None:
        raise RuntimeError("SAM extractor was not initialized.")
    labels = sam_extractor.extract_labels(image)
    valid = labels >= 0
    display_map = {int(local_id): int(local_id) + 1 for local_id in np.unique(labels[valid]).tolist()}
    return labels, display_map


def colorize_with_display_map(labels: np.ndarray, display_map: dict[int, int]) -> np.ndarray:
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for local_id, display_id in display_map.items():
        colored[labels == int(local_id)] = color_for_id(int(display_id))
    return colored


def logits_to_mask(mask_logits: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_logits > 0.0)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {mask.shape}")
    return mask.astype(bool, copy=False)


def build_seed_objects(labels: np.ndarray) -> list[tuple[int, np.ndarray]]:
    pairs = []
    for obj_id in np.unique(labels).tolist():
        if obj_id < 0:
            continue
        mask = labels == int(obj_id)
        area = int(mask.sum())
        if area <= 0:
            continue
        pairs.append((int(obj_id), mask, area))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(obj_id, mask) for obj_id, mask, _ in pairs[:SAM2_MAX_NUM_OBJECTS]]


class LazyFrameLoader:
    def __init__(self, first_frame_path: Path, image_size: int) -> None:
        self.frame_paths = [first_frame_path]
        self.image_size = image_size
        self.img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        self._cache_index = None
        self._cache_image = None
        first_img, self.video_height, self.video_width = _load_img_as_tensor(str(first_frame_path), image_size)
        first_img -= self.img_mean
        first_img /= self.img_std
        self._cache_index = 0
        self._cache_image = first_img

    def append(self, frame_path: Path) -> None:
        self.frame_paths.append(frame_path)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._cache_index == index and self._cache_image is not None:
            return self._cache_image
        img, _, _ = _load_img_as_tensor(str(self.frame_paths[index]), self.image_size)
        img -= self.img_mean
        img /= self.img_std
        self._cache_index = index
        self._cache_image = img
        return img

    def __len__(self) -> int:
        return len(self.frame_paths)


class SAM2VideoTracker:
    def __init__(self, first_frame_path: Path, model_level: int) -> None:
        if int(model_level) not in SAM2_LEVELS:
            raise ValueError(f"Unsupported SAM2.1 level {model_level}. Expected one of {sorted(SAM2_LEVELS)}.")
        checkpoint_name, config_path = SAM2_LEVELS[int(model_level)]
        checkpoint_path = Path("data/input/sam_ckpts") / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        logging.getLogger("sam2").setLevel(logging.ERROR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_level = int(model_level)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.predictor = build_sam2_video_predictor(
                config_path,
                str(checkpoint_path),
                device=self.device,
                mode=SAM2_MODE,
                hydra_overrides_extra=list(SAM2_HYDRA_OVERRIDES),
                apply_postprocessing=SAM2_APPLY_POSTPROCESSING,
            )
        self.images = LazyFrameLoader(first_frame_path, self.predictor.image_size)
        self.inference_state = {
            "images": self.images,
            "num_frames": len(self.images),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
            "video_height": self.images.video_height,
            "video_width": self.images.video_width,
            "device": self.device,
            "storage_device": torch.device("cpu"),
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
            "tracking_has_started": False,
            "frames_already_tracked": {},
        }
        with self._inference_context():
            self.predictor._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)

    def append_frame(self, frame_idx: int, frame_path: Path) -> None:
        if frame_idx < len(self.images):
            return
        if frame_idx != len(self.images):
            raise ValueError(f"Expected next frame_idx {len(self.images)}, got {frame_idx}")
        self.images.append(frame_path)
        self.inference_state["num_frames"] = len(self.images)

    @contextlib.contextmanager
    def _inference_context(self):
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode():
            with autocast_ctx:
                yield

    def close(self) -> None:
        del self.inference_state
        del self.predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _request_add_mask(self, frame_idx: int, obj_id: int, mask: np.ndarray) -> tuple[int, list[int], np.ndarray]:
        with self._inference_context():
            frame_idx_out, obj_ids, mask_logits = self.predictor.add_new_mask(
                self.inference_state,
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                mask=np.asarray(mask, dtype=bool),
            )
        return int(frame_idx_out), list(obj_ids), mask_logits.detach().cpu().numpy()

    def _finalize_seed(self, frame_idx: int) -> dict[int, np.ndarray]:
        with self._inference_context():
            self.predictor.propagate_in_video_preflight(self.inference_state)
            current_out = self.inference_state["output_dict"]["cond_frame_outputs"][int(frame_idx)]
            obj_ids = list(self.inference_state["obj_ids"])
            _, video_res_masks = self.predictor._get_orig_video_res_output(
                self.inference_state,
                current_out["pred_masks"],
            )
        self.inference_state["frames_already_tracked"][int(frame_idx)] = {"reverse": False}
        return {
            int(obj_id): logits_to_mask(video_res_masks[out_idx].detach().cpu().numpy())
            for out_idx, obj_id in enumerate(obj_ids)
        }

    def reset_and_seed_masks(self, frame_idx: int, seed_masks: list[tuple[int, np.ndarray]]) -> tuple[dict[int, int], dict[int, np.ndarray]]:
        self.predictor.reset_state(self.inference_state)
        obj_to_gt: dict[int, int] = {}
        for obj_id, (gt_id, mask) in enumerate(seed_masks):
            self._request_add_mask(frame_idx, obj_id, mask)
            obj_to_gt[int(obj_id)] = int(gt_id)
        return obj_to_gt, self._finalize_seed(frame_idx)

    def track_frame(self, frame_idx: int) -> dict[int, np.ndarray]:
        batch_size = self.predictor._get_obj_num(self.inference_state)
        with self._inference_context():
            current_out, pred_masks = self.predictor._run_single_frame_inference(
                inference_state=self.inference_state,
                output_dict=self.inference_state["output_dict"],
                frame_idx=int(frame_idx),
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
        self.inference_state["output_dict"]["non_cond_frame_outputs"][int(frame_idx)] = current_out
        with self._inference_context():
            self.predictor._add_output_per_object(
                self.inference_state,
                int(frame_idx),
                current_out,
                "non_cond_frame_outputs",
            )
        self.inference_state["frames_already_tracked"][int(frame_idx)] = {"reverse": False}
        obj_ids = list(self.inference_state["obj_ids"])
        with self._inference_context():
            _, video_res_masks = self.predictor._get_orig_video_res_output(self.inference_state, pred_masks)
        return {
            int(obj_id): logits_to_mask(video_res_masks[out_idx].detach().cpu().numpy())
            for out_idx, obj_id in enumerate(obj_ids)
        }


def write_video(
    scene_dir: Path,
    output_path: Path,
    frame_ids: list[int],
    map_every: float,
    fps: float,
    use_inst_gt: bool,
    sam_model_level_inst: int,
    sam_model_level_tr: int,
) -> None:
    color_dir = scene_dir / "color"
    frame_paths = [
        color_dir / f"{frame_id}.jpg" if (color_dir / f"{frame_id}.jpg").exists() else color_dir / f"{frame_id}.png"
        for frame_id in frame_ids
    ]
    gt_extractor = GTInstanceMaskExtractor("ScanNet", scene_dir.name) if use_inst_gt else None
    sam_extractor = None if use_inst_gt else SAMMaskExtractor("cuda" if torch.cuda.is_available() else "cpu", sam_model_level_inst)
    first_image = load_color_frame(scene_dir, frame_ids[0])
    h, w = first_image.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 3, h))

    pbar = None
    try:
        pbar = tqdm(total=len(frame_ids), desc=scene_dir.name, unit="frame")
        seg_start = 0
        while seg_start < len(frame_ids):
            seed_frame_id = frame_ids[seg_start]
            if math.isinf(map_every):
                seg_end = len(frame_ids)
            else:
                seg_end = min(len(frame_ids), seg_start + int(map_every))
            segment_frame_paths = frame_paths[seg_start:seg_end]
            mask_tracker = SAM2VideoTracker(segment_frame_paths[0], sam_model_level_tr)
            seed_rgb = load_color_frame(scene_dir, seed_frame_id)
            seed_labels, display_map = extract_seed_labels(
                scene_dir,
                seed_frame_id,
                seed_rgb,
                use_inst_gt,
                gt_extractor,
                sam_extractor,
            )
            if use_inst_gt:
                panel2_vis = colorize_instance_map(load_gt_labels(scene_dir, seed_frame_id))
                panel2_title = "GT Instances"
            else:
                panel2_vis = colorize_with_display_map(seed_labels, display_map)
                panel2_title = "SAM AMG"
            seed_masks = build_seed_objects(seed_labels)
            try:
                mask_obj_to_local, mask_seed_outputs = mask_tracker.reset_and_seed_masks(0, seed_masks)
                mask_obj_to_display = {obj_id: display_map[local_id] for obj_id, local_id in mask_obj_to_local.items()}
                for seg_local_idx, frame_id in enumerate(frame_ids[seg_start:seg_end]):
                    rgb = load_color_frame(scene_dir, frame_id)
                    mask_vis = np.zeros_like(panel2_vis)
                    mask_frame_masks = mask_seed_outputs if seg_local_idx == 0 else mask_tracker.track_frame(seg_local_idx)
                    for obj_id, mask in mask_frame_masks.items():
                        display_id = mask_obj_to_display.get(int(obj_id), 0)
                        if display_id <= 0:
                            continue
                        mask_vis[mask] = color_for_id(display_id)
                    if math.isinf(map_every):
                        cycle_text = f"map_every=inf  age={seg_local_idx}"
                    else:
                        cycle_text = f"map_every={int(map_every)}  age={seg_local_idx}/{max(1, seg_end - seg_start) - 1}"
                    rgb_panel = overlay_header(rgb, "RGB", f"frame={frame_id}")
                    ref_panel = overlay_header(panel2_vis, panel2_title, f"seed={seed_frame_id}")
                    track_panel = overlay_header(
                        mask_vis,
                        "SAM2.1 Masks",
                        f"{cycle_text}  seed={seed_frame_id}  tracked={len(mask_obj_to_display)}/{SAM2_MAX_NUM_OBJECTS}  lvl={sam_model_level_tr}",
                    )
                    writer.write(np.hstack((rgb_panel, ref_panel, track_panel)))
                    pbar.update(1)
                    pbar.set_postfix(seed=seed_frame_id, tracked=len(mask_obj_to_display))
                    next_idx = seg_local_idx + 1
                    if next_idx < len(segment_frame_paths):
                        mask_tracker.append_frame(next_idx, segment_frame_paths[next_idx])
            finally:
                mask_tracker.close()
            seg_start = seg_end
    finally:
        if pbar is not None:
            pbar.close()
        writer.release()


def main(args) -> None:
    scene_dir = INPUT_DIR / args.scene_name
    if not scene_dir.exists():
        raise FileNotFoundError(scene_dir)
    frame_ids = list_frames(scene_dir, args.frame_limit)
    if not frame_ids:
        raise RuntimeError("No frames found.")
    map_every = parse_map_every(args.map_every)
    out_name = f"{args.scene_name}_mapevery_{args.map_every}.mp4"
    output_path = Path(args.output_root) / args.scene_name / out_name
    write_video(
        scene_dir,
        output_path,
        frame_ids,
        map_every,
        args.fps,
        args.use_inst_gt,
        args.sam_model_level_inst,
        args.sam_model_level_tr,
    )
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", default="scene0011_00")
    parser.add_argument("--output_root", default=str(OUTPUT_DIR))
    parser.add_argument("--map_every", default="10000")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--sam-model-level-inst", type=int, choices=[11, 12, 13], default=13)
    parser.add_argument("--sam-model-level-tr", type=int, choices=sorted(SAM2_LEVELS), default=24)
    parser.add_argument("--use-inst-gt", action="store_true")
    main(parser.parse_args())
