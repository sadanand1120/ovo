import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build_rgb_map import (
    DEFAULT_MAP_EVERY,
    add_build_args,
    canonical_dataset_name,
    run_scene_build,
)
from get_metrics_map import load_pred_map
from visualize_rgb_map import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PCA_SAMPLE_SIZE,
    apply_pca_colormap_chunked,
    colorize_instance_labels,
    resolve_instance_labels,
)


VIDEO_FPS = 8
POINT_DILATE = 3
VIDEO_DIR_NAME = "topdown_videos"
FRUSTUM_DEPTH = 0.6
FRUSTUM_COLOR = (40, 40, 220)
FRUSTUM_THICKNESS = 2

def load_view(view_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    view = json.loads(view_path.read_text())
    intrinsic = np.asarray(view["intrinsic_matrix"], dtype=np.float32)
    extrinsic = np.asarray(view["extrinsic"], dtype=np.float32)
    return intrinsic, extrinsic, int(view["width"]), int(view["height"])


def project_points(points: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam = points_h @ extrinsic.T
    depth = cam[:, 2]
    valid = depth > 1e-6
    cam = cam[valid]
    depth = depth[valid]
    u = np.round(cam[:, 0] * intrinsic[0, 0] / depth + intrinsic[0, 2]).astype(np.int32)
    v = np.round(cam[:, 1] * intrinsic[1, 1] / depth + intrinsic[1, 2]).astype(np.int32)
    in_view = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack((u[in_view], v[in_view]), axis=1), depth[in_view].astype(np.float32), np.flatnonzero(valid)[in_view]


def project_vertices(points: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam = points_h @ extrinsic.T
    depth = cam[:, 2]
    valid = depth > 1e-6
    uv = np.zeros((points.shape[0], 2), dtype=np.float32)
    if valid.any():
        cam_valid = cam[valid]
        uv_valid = np.empty((cam_valid.shape[0], 2), dtype=np.float32)
        uv_valid[:, 0] = cam_valid[:, 0] * intrinsic[0, 0] / cam_valid[:, 2] + intrinsic[0, 2]
        uv_valid[:, 1] = cam_valid[:, 1] * intrinsic[1, 1] / cam_valid[:, 2] + intrinsic[1, 2]
        valid[valid] &= (
            (uv_valid[:, 0] >= 0)
            & (uv_valid[:, 0] < width)
            & (uv_valid[:, 1] >= 0)
            & (uv_valid[:, 1] < height)
        )
        uv[valid] = uv_valid[
            (uv_valid[:, 0] >= 0)
            & (uv_valid[:, 0] < width)
            & (uv_valid[:, 1] >= 0)
            & (uv_valid[:, 1] < height)
        ]
    return uv, valid


def reduce_chunk_to_visible(pixel_ids: np.ndarray, depths: np.ndarray, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((depths, pixel_ids))
    pixel_ids = pixel_ids[order]
    depths = depths[order]
    colors = colors[order]
    keep = np.empty(pixel_ids.shape[0], dtype=bool)
    keep[0] = True
    keep[1:] = pixel_ids[1:] != pixel_ids[:-1]
    return pixel_ids[keep], depths[keep], colors[keep]


def build_camera_frustum(
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    depth: float,
) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [(-cx) * depth / fx, (-cy) * depth / fy, depth],
            [((image_width - 1) - cx) * depth / fx, (-cy) * depth / fy, depth],
            [((image_width - 1) - cx) * depth / fx, ((image_height - 1) - cy) * depth / fy, depth],
            [(-cx) * depth / fx, ((image_height - 1) - cy) * depth / fy, depth],
        ],
        dtype=np.float32,
    )
    corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)
    return (corners_h @ c2w.T)[:, :3]


def draw_camera_frustum(
    frame: np.ndarray,
    c2w: np.ndarray,
    source_intrinsic: np.ndarray,
    source_width: int,
    source_height: int,
    render_intrinsic: np.ndarray,
    render_extrinsic: np.ndarray,
    render_width: int,
    render_height: int,
    depth: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    frustum = build_camera_frustum(c2w, source_intrinsic, source_width, source_height, depth)
    uv, valid = project_vertices(frustum, render_intrinsic, render_extrinsic, render_width, render_height)
    edges = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1))
    for a, b in edges:
        if valid[a] and valid[b]:
            p0 = tuple(np.round(uv[a]).astype(int))
            p1 = tuple(np.round(uv[b]).astype(int))
            cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)
    if valid[0]:
        center = tuple(np.round(uv[0]).astype(int))
        cv2.circle(frame, center, max(thickness + 1, 3), color, -1, cv2.LINE_AA)


def render_incremental_video(
    output_path: Path,
    mode_name: str,
    points: np.ndarray,
    point_colors: np.ndarray,
    snapshot_counts: list[int],
    snapshot_frame_ids: list[int],
    snapshot_c2w: list[np.ndarray],
    source_intrinsic: np.ndarray,
    source_width: int,
    source_height: int,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    width: int,
    height: int,
    fps: int,
    dilate: int,
) -> None:
    pixels, depths, point_ids = project_points(points, intrinsic, extrinsic, width, height)
    point_colors = point_colors[point_ids][:, ::-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas_flat = canvas.reshape(-1, 3)
    front_depth = np.full(height * width, np.inf, dtype=np.float32)
    prev_count = 0

    try:
        for frame_id, count, c2w in tqdm(
            list(zip(snapshot_frame_ids, snapshot_counts, snapshot_c2w)),
            desc=f"{mode_name} video",
            unit="frame",
        ):
            start_idx = np.searchsorted(point_ids, prev_count, side="left")
            end_idx = np.searchsorted(point_ids, count, side="left")
            if end_idx > start_idx:
                chunk_pixels = pixels[start_idx:end_idx]
                chunk_depths = depths[start_idx:end_idx]
                chunk_colors = point_colors[start_idx:end_idx]
                pixel_ids = chunk_pixels[:, 1] * width + chunk_pixels[:, 0]
                pixel_ids, chunk_depths, chunk_colors = reduce_chunk_to_visible(pixel_ids, chunk_depths, chunk_colors)
                update = chunk_depths < front_depth[pixel_ids]
                pixel_ids = pixel_ids[update]
                front_depth[pixel_ids] = chunk_depths[update]
                canvas_flat[pixel_ids] = chunk_colors[update]
            prev_count = count

            frame = canvas if dilate <= 1 else cv2.dilate(canvas, np.ones((dilate, dilate), dtype=np.uint8))
            frame = frame.copy()
            draw_camera_frustum(
                frame,
                c2w,
                source_intrinsic,
                source_width,
                source_height,
                intrinsic,
                extrinsic,
                width,
                height,
                FRUSTUM_DEPTH,
                FRUSTUM_COLOR,
                FRUSTUM_THICKNESS,
            )
            cv2.putText(frame, f"{mode_name} view build", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(frame, f"frame: {frame_id}", (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(frame, f"points: {count:,}", (24, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()


def main(args: argparse.Namespace) -> None:
    dataset_name = args.dataset_name
    scene_name = args.scene_name
    output_dir = Path(args.output_root) / canonical_dataset_name(dataset_name) / scene_name
    intrinsic, extrinsic, width, height = load_view(Path(args.load_view))

    snapshot_counts: list[int] = []
    snapshot_frame_ids: list[int] = []
    snapshot_c2w: list[np.ndarray] = []
    def snapshot_hook(frame_id: int, prev_n: int, cur_n: int, estimated_c2w) -> None:
        if cur_n <= prev_n or estimated_c2w is None:
            return
        snapshot_counts.append(int(cur_n))
        snapshot_frame_ids.append(int(frame_id))
        pose_np = np.asarray(estimated_c2w.detach().cpu().numpy() if hasattr(estimated_c2w, "detach") else estimated_c2w, dtype=np.float32)
        snapshot_c2w.append(pose_np.astype(np.float32, copy=True))

    output_dir, timing_summary, build_meta = run_scene_build(
        dataset_name=dataset_name,
        scene_name=scene_name,
        output_root=args.output_root,
        frame_limit=args.frame_limit,
        slam_module=args.slam_module,
        disable_loop_closure=args.disable_loop_closure,
        config_path=args.config_path,
        map_every=args.map_every,
        point_sample_stride=args.point_sample_stride,
        max_frame_points=args.max_frame_points,
        match_distance_th=args.match_distance_th,
        snapshot_hook=snapshot_hook,
    )
    stats_path = output_dir / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["snapshot_counts"] = snapshot_counts
    stats["snapshot_frame_ids"] = snapshot_frame_ids
    stats_path.write_text(json.dumps(stats, indent=2))

    pred = load_pred_map(output_dir / "rgb_map.ply")
    points = pred["points"]
    rgb_colors = pred["colors"]
    normal_colors = np.clip((pred["normals"] + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    feat_colors = (
        apply_pca_colormap_chunked(
            pred["clip_features"],
            args.pca_sample_size,
            args.chunk_size,
        )
        * 255.0
    ).astype(np.uint8)
    instance_colors = (colorize_instance_labels(resolve_instance_labels(output_dir, points.shape[0], args.min_component_size)) * 255.0).astype(np.uint8)

    video_dir = output_dir / VIDEO_DIR_NAME
    source_intrinsic = build_meta["dataset_intrinsics"]
    source_width = build_meta["source_width"]
    source_height = build_meta["source_height"]
    render_incremental_video(video_dir / "rgb.mp4", "RGB", points, rgb_colors, snapshot_counts, snapshot_frame_ids, snapshot_c2w, source_intrinsic, source_width, source_height, intrinsic, extrinsic, width, height, args.fps, args.dilate)
    render_incremental_video(video_dir / "normals.mp4", "Normals", points, normal_colors, snapshot_counts, snapshot_frame_ids, snapshot_c2w, source_intrinsic, source_width, source_height, intrinsic, extrinsic, width, height, args.fps, args.dilate)
    render_incremental_video(video_dir / "feat_pca.mp4", "Feature PCA", points, feat_colors, snapshot_counts, snapshot_frame_ids, snapshot_c2w, source_intrinsic, source_width, source_height, intrinsic, extrinsic, width, height, args.fps, args.dilate)
    instance_title = "SAM Instances"
    instance_name = "sam_instances.mp4"
    render_incremental_video(video_dir / instance_name, instance_title, points, instance_colors, snapshot_counts, snapshot_frame_ids, snapshot_c2w, source_intrinsic, source_width, source_height, intrinsic, extrinsic, width, height, args.fps, args.dilate)

    summary = {
        "output_dir": str(output_dir),
        "videos": {
            "rgb": str(video_dir / "rgb.mp4"),
            "normals": str(video_dir / "normals.mp4"),
            "feat_pca": str(video_dir / "feat_pca.mp4"),
            "instances": str(video_dir / instance_name),
        },
        "n_snapshots": len(snapshot_counts),
        "n_points": int(points.shape[0]),
        "timing": timing_summary,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RGB+normal+CLIP maps and render incremental view videos.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet"])
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--load-view", required=True, help="Path to a saved view JSON dumped from visualize_rgb_map.py.")
    add_build_args(parser, default_output_root="data/output/topdown_vis", default_map_every=DEFAULT_MAP_EVERY)
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--dilate", type=int, default=POINT_DILATE)
    parser.add_argument("--min_component_size", type=int, default=2000)
    parser.add_argument("--pca_sample_size", type=int, default=DEFAULT_PCA_SAMPLE_SIZE)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parsed = parser.parse_args()
    main(parsed)
