from __future__ import annotations

import contextlib
import io
import logging
from collections import OrderedDict
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdParty" / "segment-anything-2"))


INPUT_DIR = Path("data/input")
SAM2_MAX_NUM_OBJECTS = 16
DEFAULT_SAM2_TRACK_MODEL_LEVEL = 24
SAM2_LEVELS = {
    21: ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    22: ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    23: ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    24: ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
SAM2_MODE = "eval"
SAM2_HYDRA_OVERRIDES: tuple[str, ...] = ()
SAM2_APPLY_POSTPROCESSING = True


def logits_to_mask(mask_logits: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_logits > 0.0)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after squeeze, got shape {mask.shape}")
    return mask.astype(bool, copy=False)


def build_label_masks(labels: np.ndarray, max_objects: int | None = None) -> list[tuple[int, np.ndarray]]:
    pairs = []
    for obj_id in np.unique(labels).tolist():
        if obj_id < 0:
            continue
        mask = labels == int(obj_id)
        area = int(mask.sum())
        if area <= 0:
            continue
        pairs.append((int(obj_id), mask, area))
    pairs.sort(key=lambda item: item[2], reverse=True)
    if max_objects is not None:
        pairs = pairs[: int(max_objects)]
    return [(obj_id, mask) for obj_id, mask, _ in pairs]


def load_frame_source(frame_source: Path | str | np.ndarray, image_size: int) -> tuple[torch.Tensor, int, int]:
    if isinstance(frame_source, (str, Path)):
        from sam2.utils.misc import _load_img_as_tensor

        return _load_img_as_tensor(str(frame_source), image_size)
    image = np.asarray(frame_source)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 frame source, got shape {image.shape}")
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor, int(image.shape[0]), int(image.shape[1])


class LazyFrameLoader:
    def __init__(self, first_frame_source: Path | str | np.ndarray, image_size: int) -> None:
        self.frame_sources = [first_frame_source]
        self.image_size = image_size
        self.img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        self._cache_index = None
        self._cache_image = None
        first_img, self.video_height, self.video_width = load_frame_source(first_frame_source, image_size)
        first_img -= self.img_mean
        first_img /= self.img_std
        self._cache_index = 0
        self._cache_image = first_img

    def append(self, frame_source: Path | str | np.ndarray) -> None:
        self.frame_sources.append(frame_source)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._cache_index == index and self._cache_image is not None:
            return self._cache_image
        img, _, _ = load_frame_source(self.frame_sources[index], self.image_size)
        img -= self.img_mean
        img /= self.img_std
        self._cache_index = index
        self._cache_image = img
        return img

    def __len__(self) -> int:
        return len(self.frame_sources)


class SAM2VideoTracker:
    def __init__(
        self,
        first_frame_source: Path | str | np.ndarray,
    ) -> None:
        model_level = DEFAULT_SAM2_TRACK_MODEL_LEVEL
        if int(model_level) not in SAM2_LEVELS:
            raise ValueError(f"Unsupported SAM2.1 level {model_level}. Expected one of {sorted(SAM2_LEVELS)}.")
        checkpoint_name, config_path = SAM2_LEVELS[int(model_level)]
        checkpoint_path = INPUT_DIR / "sam_ckpts" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        logging.getLogger("sam2").setLevel(logging.ERROR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_level = int(model_level)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.max_num_objects = SAM2_MAX_NUM_OBJECTS
        from sam2.build_sam import build_sam2_video_predictor

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.predictor = build_sam2_video_predictor(
                config_path,
                str(checkpoint_path),
                device=self.device,
                mode=SAM2_MODE,
                hydra_overrides_extra=list(SAM2_HYDRA_OVERRIDES),
                apply_postprocessing=SAM2_APPLY_POSTPROCESSING,
            )
        self.images = None
        self.inference_state = None
        self._restart(first_frame_source)

    def _build_inference_state(self, first_frame_source: Path | str | np.ndarray) -> dict:
        images = LazyFrameLoader(first_frame_source, self.predictor.image_size)
        self.images = images
        return {
            "images": self.images,
            "num_frames": len(images),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
            "video_height": images.video_height,
            "video_width": images.video_width,
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

    def _restart(self, first_frame_source: Path | str | np.ndarray) -> None:
        self.inference_state = self._build_inference_state(first_frame_source)
        with self._inference_context():
            self.predictor._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)

    def append_frame(self, frame_idx: int, frame_source: Path | str | np.ndarray) -> None:
        if frame_idx < len(self.images):
            return
        if frame_idx != len(self.images):
            raise ValueError(f"Expected next frame_idx {len(self.images)}, got {frame_idx}")
        self.images.append(frame_source)
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

    def reset_and_seed_masks(self, seed_masks: list[tuple[int, np.ndarray]]) -> dict[int, np.ndarray]:
        self.predictor.reset_state(self.inference_state)
        if not seed_masks:
            return {}
        for obj_id, mask in seed_masks:
            self._request_add_mask(frame_idx=0, obj_id=int(obj_id), mask=mask)
        return self._finalize_seed(frame_idx=0)

    def restart_and_seed_masks(
        self,
        first_frame_source: Path | str | np.ndarray,
        seed_masks: list[tuple[int, np.ndarray]],
    ) -> dict[int, np.ndarray]:
        self._restart(first_frame_source)
        return self.reset_and_seed_masks(seed_masks)

    def track_frame(self, frame_idx: int) -> dict[int, np.ndarray]:
        batch_size = self.predictor._get_obj_num(self.inference_state)
        if batch_size == 0:
            return {}
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
