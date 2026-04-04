from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


GRID_COLOR = (225, 225, 225)
AXIS_COLOR = (170, 170, 170)
TEXT_COLOR = (35, 35, 35)
GT_COLOR = (60, 180, 75)
PRED_COLOR = (40, 70, 220)
GT_NOW_COLOR = (0, 120, 0)
PRED_NOW_COLOR = (0, 0, 200)


def load_pose_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32).reshape(4, 4)


def load_predicted_poses(path: Path) -> dict[int, np.ndarray]:
    poses = torch.load(path, map_location="cpu", weights_only=False)
    return {
        int(frame_id): pose.detach().cpu().numpy() if isinstance(pose, torch.Tensor) else np.asarray(pose)
        for frame_id, pose in poses.items()
    }


def build_scene_inputs(scene_dir: Path, predicted_poses: dict[int, np.ndarray]) -> tuple[list[int], list[Path], dict[int, np.ndarray], dict[int, np.ndarray]]:
    color_paths = {
        int(path.stem): path
        for path in scene_dir.joinpath("color").glob("*.jpg")
    }
    gt_poses = {
        int(path.stem): load_pose_txt(path)
        for path in scene_dir.joinpath("pose").glob("*.txt")
    }
    frame_ids = sorted(
        frame_id
        for frame_id in (set(color_paths) & set(gt_poses) & set(predicted_poses))
        if np.isfinite(gt_poses[frame_id]).all() and np.isfinite(predicted_poses[frame_id]).all()
    )
    if not frame_ids:
        raise ValueError(f"No overlapping RGB, GT pose, and predicted pose frames found in {scene_dir}")
    return frame_ids, [color_paths[frame_id] for frame_id in frame_ids], gt_poses, predicted_poses


def extract_topdown_xy(poses: list[np.ndarray]) -> np.ndarray:
    xyz = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float32)
    return xyz[:, [0, 2]]


def compute_canvas_transform(gt_xz: np.ndarray, pred_xz: np.ndarray, panel_size: int, margin: int) -> tuple[float, float, float]:
    all_points = np.concatenate([gt_xz, pred_xz], axis=0)
    min_x, min_z = np.floor(all_points.min(axis=0) - 0.5)
    max_x, max_z = np.ceil(all_points.max(axis=0) + 0.5)
    span_x = max(max_x - min_x, 1.0)
    span_z = max(max_z - min_z, 1.0)
    scale = min((panel_size - 2 * margin) / span_x, (panel_size - 2 * margin) / span_z)
    return float(min_x), float(max_z), float(scale)


def world_to_canvas(points_xz: np.ndarray, min_x: float, max_z: float, scale: float, margin: int) -> np.ndarray:
    pixels = np.empty((len(points_xz), 2), dtype=np.int32)
    pixels[:, 0] = np.round((points_xz[:, 0] - min_x) * scale + margin).astype(np.int32)
    pixels[:, 1] = np.round((max_z - points_xz[:, 1]) * scale + margin).astype(np.int32)
    return pixels


def draw_metric_grid(panel: np.ndarray, min_x: float, max_z: float, scale: float, margin: int) -> None:
    height, width = panel.shape[:2]
    min_z = max_z - (height - 2 * margin) / scale
    max_x = min_x + (width - 2 * margin) / scale

    start_x = int(np.floor(min_x))
    end_x = int(np.ceil(max_x))
    start_z = int(np.floor(min_z))
    end_z = int(np.ceil(max_z))

    for meter_x in range(start_x, end_x + 1):
        x = int(round((meter_x - min_x) * scale + margin))
        color = AXIS_COLOR if meter_x == 0 else GRID_COLOR
        cv2.line(panel, (x, margin), (x, height - margin), color, 1, cv2.LINE_AA)
        cv2.putText(panel, f"{meter_x}m", (x + 4, height - margin + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    for meter_z in range(start_z, end_z + 1):
        y = int(round((max_z - meter_z) * scale + margin))
        color = AXIS_COLOR if meter_z == 0 else GRID_COLOR
        cv2.line(panel, (margin, y), (width - margin, y), color, 1, cv2.LINE_AA)
        if margin - 48 > 0:
            cv2.putText(panel, f"{meter_z}m", (6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_legend(panel: np.ndarray, pred_label: str) -> None:
    y = 36
    x = 24
    for color, label in [(GT_COLOR, "GT"), (PRED_COLOR, pred_label)]:
        cv2.line(panel, (x, y), (x + 40, y), color, 4, cv2.LINE_AA)
        cv2.putText(panel, label, (x + 52, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 2, cv2.LINE_AA)
        y += 32


def draw_stats(panel: np.ndarray, frame_id: int, error_m: float, gt_len_m: float, pred_len_m: float) -> None:
    lines = [
        f"frame: {frame_id}",
        f"pos err: {error_m:.2f} m",
        f"gt len: {gt_len_m:.1f} m",
        f"pred len: {pred_len_m:.1f} m",
    ]
    x = 24
    y = panel.shape[0] - 96
    for line in lines:
        cv2.putText(panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 2, cv2.LINE_AA)
        y += 26


def build_panel(
    scene_name: str,
    pred_label: str,
    frame_id: int,
    gt_pixels: np.ndarray,
    pred_pixels: np.ndarray,
    gt_xz: np.ndarray,
    pred_xz: np.ndarray,
    panel_size: int,
    margin: int,
    min_x: float,
    max_z: float,
    scale: float,
) -> np.ndarray:
    panel = np.full((panel_size, panel_size, 3), 255, dtype=np.uint8)
    draw_metric_grid(panel, min_x, max_z, scale, margin)
    draw_legend(panel, pred_label)

    if len(gt_pixels) > 1:
        cv2.polylines(panel, [gt_pixels], False, GT_COLOR, 3, cv2.LINE_AA)
    if len(pred_pixels) > 1:
        cv2.polylines(panel, [pred_pixels], False, PRED_COLOR, 3, cv2.LINE_AA)

    cv2.circle(panel, tuple(gt_pixels[-1]), 6, GT_NOW_COLOR, -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(pred_pixels[-1]), 6, PRED_NOW_COLOR, -1, cv2.LINE_AA)

    error_m = float(np.linalg.norm(gt_xz[-1] - pred_xz[-1]))
    gt_len_m = float(np.linalg.norm(np.diff(gt_xz, axis=0), axis=1).sum()) if len(gt_xz) > 1 else 0.0
    pred_len_m = float(np.linalg.norm(np.diff(pred_xz, axis=0), axis=1).sum()) if len(pred_xz) > 1 else 0.0

    cv2.putText(panel, f"{scene_name} top-down trajectory", (24, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(panel, "1 grid cell = 1 m", (24, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 2, cv2.LINE_AA)
    draw_stats(panel, frame_id, error_m, gt_len_m, pred_len_m)
    return panel


def resize_frame(frame_bgr: np.ndarray, target_height: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    target_width = int(round(w * target_height / h))
    return cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)


def render_scene_video(scene_dir: Path, predicted_path: Path, output_path: Path, fps: int, panel_size: int, margin: int, pred_label: str) -> None:
    predicted_poses = load_predicted_poses(predicted_path)
    frame_ids, color_paths, gt_poses, _ = build_scene_inputs(scene_dir, predicted_poses)

    gt_traj = [gt_poses[frame_id] for frame_id in frame_ids]
    pred_traj = [predicted_poses[frame_id] for frame_id in frame_ids]
    gt_xz_all = extract_topdown_xy(gt_traj)
    pred_xz_all = extract_topdown_xy(pred_traj)
    min_x, max_z, scale = compute_canvas_transform(gt_xz_all, pred_xz_all, panel_size, margin)

    first_frame = cv2.imread(str(color_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read {color_paths[0]}")
    rgb_panel = resize_frame(first_frame, panel_size)
    combined_size = (rgb_panel.shape[1] + panel_size, panel_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, combined_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for idx, (frame_id, color_path) in enumerate(tqdm(list(zip(frame_ids, color_paths)), total=len(frame_ids), desc=scene_dir.name)):
            frame_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Failed to read {color_path}")
            frame_bgr = resize_frame(frame_bgr, panel_size)
            cv2.putText(frame_bgr, f"{scene_dir.name} RGB", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"frame: {frame_id}", (24, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            gt_xz = gt_xz_all[: idx + 1]
            pred_xz = pred_xz_all[: idx + 1]
            gt_pixels = world_to_canvas(gt_xz, min_x, max_z, scale, margin)
            pred_pixels = world_to_canvas(pred_xz, min_x, max_z, scale, margin)
            panel = build_panel(scene_dir.name, pred_label, frame_id, gt_pixels, pred_pixels, gt_xz, pred_xz, panel_size, margin, min_x, max_z, scale)

            writer.write(np.concatenate([frame_bgr, panel], axis=1))
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ScanNet RGB plus GT-vs-predicted trajectory debug videos.")
    parser.add_argument("run_root", type=Path, help="Run directory that contains per-scene ORB outputs.")
    parser.add_argument("--data_root", type=Path, default=Path("data/input/ScanNet"), help="Decoded ScanNet scene root.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for generated MP4 files.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene list. Defaults to all scene subdirs in run_root.")
    parser.add_argument("--fps", type=int, default=4, help="Output video FPS.")
    parser.add_argument("--panel_size", type=int, default=900, help="Height of the RGB pane and size of the trajectory pane.")
    parser.add_argument("--margin", type=int, default=70, help="Pixel margin around the metric grid.")
    parser.add_argument("--pred_label", type=str, default="Prediction", help="Legend label for the predicted trajectory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    output_dir = args.output_dir or run_root / "debug_videos"

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = sorted(path.name for path in run_root.iterdir() if path.is_dir() and path.name.startswith("scene"))

    for scene_name in scenes:
        scene_dir = args.data_root / scene_name
        predicted_path = run_root / scene_name / "estimated_c2w.npy"
        if not scene_dir.exists():
            raise FileNotFoundError(f"Missing decoded scene directory: {scene_dir}")
        if not predicted_path.exists():
            raise FileNotFoundError(f"Missing predicted trajectory: {predicted_path}")

        render_scene_video(
            scene_dir=scene_dir,
            predicted_path=predicted_path,
            output_path=output_dir / f"{scene_name}.mp4",
            fps=args.fps,
            panel_size=args.panel_size,
            margin=args.margin,
            pred_label=args.pred_label,
        )


if __name__ == "__main__":
    main()
