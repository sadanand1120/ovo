from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from map_runtime.defaults import CACHE_MANIFEST_FILE, FRAME_CACHE_DIR, STATS_PATH
from . import geometry
from .debug_panels import overlay_header, render_sorted_label_map
from .rgb_map_utils import invert_rigid_transform
from .scene import get_tracked_pose, load_dataset_and_slam


def _project_point_labels_to_image(point_ids_image: np.ndarray, point_labels: np.ndarray) -> np.ndarray:
    labels = np.full(point_ids_image.shape, -1, dtype=np.int32)
    valid_mask = point_ids_image >= 0
    if not valid_mask.any():
        return labels
    point_ids = point_ids_image[valid_mask].astype(np.int64, copy=False)
    labels[valid_mask] = np.asarray(point_labels, dtype=np.int32)[point_ids]
    return labels


def _to_device_float32_tensor(array_like: np.ndarray | torch.Tensor, device: str) -> torch.Tensor:
    if torch.is_tensor(array_like):
        return array_like.to(device=device, dtype=torch.float32)
    return torch.from_numpy(np.asarray(array_like, dtype=np.float32)).to(device)


def _write_video(frames: list[np.ndarray], output_path: Path) -> Path:
    if not frames:
        raise ValueError("No frames were provided for instance-label video export.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (int(frames[0].shape[1]), int(frames[0].shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return output_path


def _annotate_pane_header(image: np.ndarray, label: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    header_h = 32
    canvas = np.zeros((header_h + image.shape[0], image.shape[1], 3), dtype=np.uint8)
    canvas[header_h:] = image
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x = max((image.shape[1] - text_size[0]) // 2, 8)
    y = 22
    cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _render_side_by_side_labels(
    pred_projected: np.ndarray,
    gt_projected: np.ndarray,
    *,
    title: str,
    subtitle: str,
) -> np.ndarray:
    pred_panel = _annotate_pane_header(render_sorted_label_map(pred_projected), "Predicted")
    gt_panel = _annotate_pane_header(render_sorted_label_map(gt_projected), "GT on predicted geometry")
    gap = 8
    spacer = np.full((pred_panel.shape[0], gap, 3), 255, dtype=np.uint8)
    combined = np.concatenate([pred_panel, spacer, gt_panel], axis=1)
    return overlay_header(combined, title, subtitle)


def write_instance_label_video_from_cache(
    cache_dir: str | Path,
    pred_point_labels: np.ndarray,
    *,
    gt_point_labels: np.ndarray,
    output_path: str | Path,
    title: str,
    subtitle_prefix: str = "frame",
    upto_frame: int | None = None,
) -> Path:
    cache_dir = Path(cache_dir)
    with open(cache_dir / CACHE_MANIFEST_FILE, "r") as handle:
        manifest = json.load(handle)
    frame_cache_dir = cache_dir / manifest.get("frame_cache_dir", FRAME_CACHE_DIR)
    total_frames = int(manifest["n_frames"])
    if upto_frame is None:
        target_frame = total_frames - 1
    else:
        target_frame = int(upto_frame)
    if target_frame < 0 or target_frame >= total_frames:
        raise ValueError(f"upto_frame must be in [0, {total_frames - 1}]")

    frames: list[np.ndarray] = []
    progress = tqdm(
        range(target_frame + 1),
        desc="instance label video",
        unit="frame",
        leave=False,
        dynamic_ncols=True,
    )
    try:
        for frame_id in progress:
            cache_path = frame_cache_dir / f"{int(frame_id):06d}.npz"
            with np.load(cache_path) as data:
                point_ids_after = np.array(data["point_ids_after"], copy=True)
                pred_projected = _project_point_labels_to_image(point_ids_after, pred_point_labels)
                gt_projected = _project_point_labels_to_image(point_ids_after, gt_point_labels)
            panel = _render_side_by_side_labels(
                pred_projected,
                gt_projected,
                title=title,
                subtitle=f"{subtitle_prefix} frame={int(frame_id)}",
            )
            frames.append(panel)
    finally:
        progress.close()
    return _write_video(frames, Path(output_path))


def write_instance_label_video_from_scene_output(
    output_dir: str | Path,
    pred_points: np.ndarray,
    pred_point_labels: np.ndarray,
    *,
    gt_point_labels: np.ndarray,
    output_path: str | Path,
    title: str,
    device: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    with open(output_dir / STATS_PATH, "r") as handle:
        stats = json.load(handle)

    dataset_name = str(stats["dataset_name"])
    scene_name = str(stats["scene_name"])
    frame_limit = int(stats["n_frames"])
    slam_module = stats.get("slam_module")
    disable_loop_closure = not bool(stats.get("slam_close_loops", True))
    match_distance_th = float(stats.get("match_distance_th", 0.03))
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    _, dataset, slam_backbone = load_dataset_and_slam(
        dataset_name=dataset_name,
        scene_name=scene_name,
        device=run_device,
        frame_limit=frame_limit,
        slam_module=slam_module,
        disable_loop_closure=disable_loop_closure,
    )
    intrinsics = torch.tensor(dataset.intrinsics.astype(np.float32), device=run_device)
    pred_points_t = torch.from_numpy(np.asarray(pred_points, dtype=np.float32)).to(run_device)
    pred_point_labels = np.asarray(pred_point_labels, dtype=np.int32)
    gt_point_labels = np.asarray(gt_point_labels, dtype=np.int32)

    frames: list[np.ndarray] = []
    progress = tqdm(
        range(len(dataset)),
        desc="instance label video",
        unit="frame",
        leave=False,
        dynamic_ncols=True,
    )
    try:
        for frame_idx in progress:
            frame_data = dataset[frame_idx]
            depth_np = np.asarray(frame_data[2], dtype=np.float32)
            pred_projected = np.full(depth_np.shape, -1, dtype=np.int32)
            gt_projected = np.full(depth_np.shape, -1, dtype=np.int32)
            c2w_np = get_tracked_pose(slam_backbone, frame_data)
            if c2w_np is not None and np.any(depth_np > 0):
                depth = torch.from_numpy(depth_np).to(run_device)
                c2w = _to_device_float32_tensor(c2w_np, run_device)
                frustum_corners = geometry.compute_camera_frustum_corners(depth, c2w, intrinsics)
                frustum_mask = geometry.compute_frustum_point_ids(pred_points_t, frustum_corners, device=run_device)
                if frustum_mask.numel() > 0:
                    matched_ids, matches = geometry.match_3d_points_to_2d_pixels(
                        depth,
                        invert_rigid_transform(c2w),
                        pred_points_t[frustum_mask],
                        intrinsics,
                        match_distance_th,
                    )
                    if matches.numel() > 0:
                        global_ids = frustum_mask[matched_ids].cpu().numpy().astype(np.int64, copy=False)
                        match_pixels = matches.cpu().numpy().astype(np.int64, copy=False)
                        pred_projected[match_pixels[:, 1], match_pixels[:, 0]] = pred_point_labels[global_ids]
                        gt_projected[match_pixels[:, 1], match_pixels[:, 0]] = gt_point_labels[global_ids]
            panel = _render_side_by_side_labels(
                pred_projected,
                gt_projected,
                title=title,
                subtitle=f"projected frame={int(frame_idx)}",
            )
            frames.append(panel)
    finally:
        progress.close()
        del slam_backbone
    return _write_video(frames, Path(output_path))
