import argparse
import math
from pathlib import Path

import cv2  # Load before torch in this env.
import numpy as np
import torch
from tqdm.auto import tqdm

from map_runtime.sam_masks import GTInstanceMaskExtractor, SAMMaskExtractor
from map_runtime.sam2_tracking import SAM2_LEVELS, SAM2_MAX_NUM_OBJECTS, SAM2VideoTracker, build_seed_objects
from map_runtime.scene import INPUT_DIR, canonical_dataset_name

OUTPUT_DIR = Path("data/output/sam2_map_every_video")
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (20, 20, 20)


def parse_map_every(value: str) -> float:
    if value.lower() in {"inf", "infinity"}:
        return math.inf
    return float(int(value))


def frame_stem_to_id(stem: str) -> int:
    if stem.startswith("frame"):
        return int(stem[len("frame") :])
    return int(stem)


def build_frame_lookup(scene_dir: Path, dataset_name: str, frame_limit: int | None) -> dict[int, Path]:
    if dataset_name == "ScanNet":
        candidates = [
            path
            for path in (scene_dir / "color").iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    else:
        candidates = list((scene_dir / "results").glob("frame*.jpg")) + list((scene_dir / "results").glob("frame*.png"))
    frame_lookup = {frame_stem_to_id(path.stem): path for path in candidates}
    if frame_limit is not None:
        frame_lookup = {frame_id: path for frame_id, path in frame_lookup.items() if frame_id < frame_limit}
    return dict(sorted(frame_lookup.items()))


def load_color_frame(frame_path: Path) -> np.ndarray:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame_path)
    return image


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
    labels = sam_extractor.extract_labels(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    valid = labels >= 0
    display_map = {int(local_id): int(local_id) + 1 for local_id in np.unique(labels[valid]).tolist()}
    return labels, display_map


def colorize_with_display_map(labels: np.ndarray, display_map: dict[int, int]) -> np.ndarray:
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for local_id, display_id in display_map.items():
        colored[labels == int(local_id)] = color_for_id(int(display_id))
    return colored


def write_video(
    dataset_name: str,
    scene_dir: Path,
    output_path: Path,
    frame_ids: list[int],
    frame_lookup: dict[int, Path],
    map_every: float,
    fps: float,
    use_inst_gt: bool,
    sam_model_level_inst: int,
    sam2_model_level_track: int,
) -> None:
    frame_paths = [frame_lookup[frame_id] for frame_id in frame_ids]
    gt_extractor = GTInstanceMaskExtractor(dataset_name, scene_dir.name) if use_inst_gt else None
    sam_extractor = None if use_inst_gt else SAMMaskExtractor("cuda" if torch.cuda.is_available() else "cpu", sam_model_level_inst)
    first_image = load_color_frame(frame_lookup[frame_ids[0]])
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
            mask_tracker = SAM2VideoTracker(segment_frame_paths[0], sam2_model_level_track)
            seed_rgb = load_color_frame(frame_lookup[seed_frame_id])
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
                mask_seed_outputs = mask_tracker.reset_and_seed_masks(seed_masks)
                mask_obj_to_display = {int(obj_id): display_map[int(obj_id)] for obj_id, _ in seed_masks}
                for seg_local_idx, frame_id in enumerate(frame_ids[seg_start:seg_end]):
                    rgb = load_color_frame(frame_lookup[frame_id])
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
                        f"{cycle_text}  seed={seed_frame_id}  tracked={len(mask_obj_to_display)}/{SAM2_MAX_NUM_OBJECTS}  lvl={sam2_model_level_track}",
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
    dataset_name = args.dataset_name
    scene_dir = INPUT_DIR / canonical_dataset_name(dataset_name) / args.scene_name
    if not scene_dir.exists():
        raise FileNotFoundError(scene_dir)
    frame_lookup = build_frame_lookup(scene_dir, dataset_name, args.frame_limit)
    frame_ids = list(frame_lookup)
    if not frame_ids:
        raise RuntimeError("No frames found.")
    map_every = parse_map_every(args.map_every)
    out_name = f"{args.scene_name}_mapevery_{args.map_every}.mp4"
    output_path = Path(args.output_root) / canonical_dataset_name(dataset_name) / args.scene_name / out_name
    write_video(
        dataset_name,
        scene_dir,
        output_path,
        frame_ids,
        frame_lookup,
        map_every,
        args.fps,
        args.use_inst_gt,
        args.sam_model_level_inst,
        args.sam2_model_level_track,
    )
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="ScanNet", choices=["Replica", "ScanNet"])
    parser.add_argument("--scene_name", default="scene0011_00")
    parser.add_argument("--output_root", default=str(OUTPUT_DIR))
    parser.add_argument("--map_every", default="10000")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--sam-model-level-inst", type=int, choices=[11, 12, 13], default=13)
    parser.add_argument("--sam2-model-level-track", type=int, choices=sorted(SAM2_LEVELS), default=24)
    parser.add_argument("--use-inst-gt", action="store_true")
    main(parser.parse_args())
