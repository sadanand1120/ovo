import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from build_rgb_map import (
    CLIP_FEATURE_FILE,
    CLIP_LOAD_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DEFAULT_DOWNSCALE_RES,
    DEFAULT_K_POOLING,
    DEFAULT_MAP_EVERY,
    DEFAULT_MATCH_DISTANCE_TH,
    DEFAULT_MAX_FRAME_POINTS,
    RGBMapper,
    canonical_dataset_name,
    load_dataset,
)
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
    dataset_name = args.dataset_name.lower()
    scene_name = args.scene_name
    output_dir = Path(args.output_root) / canonical_dataset_name(dataset_name) / scene_name
    dataset = load_dataset(dataset_name, scene_name, args.frame_limit)
    intrinsic, extrinsic, width, height = load_view(Path(args.load_view))

    mapper = RGBMapper(
        intrinsics=dataset.intrinsics,
        device="cuda" if torch.cuda.is_available() else "cpu",
        map_every=args.map_every,
        downscale_res=args.downscale_res,
        k_pooling=args.k_pooling,
        max_frame_points=args.max_frame_points,
        match_distance_th=args.match_distance_th,
        dataset_name=dataset_name,
        scene_name=scene_name,
        use_inst_gt=args.use_inst_gt,
        sam_model_level_inst=args.sam_model_level_inst,
        sam_model_level_tr=args.sam_model_level_tr,
    )

    snapshot_counts: list[int] = []
    snapshot_frame_ids: list[int] = []
    snapshot_c2w: list[np.ndarray] = []
    sample_image = dataset[0][1]
    source_height, source_width = sample_image.shape[:2]
    progress = tqdm(range(len(dataset)), desc=scene_name, unit="frame")
    for frame_id in progress:
        frame_data = dataset[frame_id]
        prev_n = mapper.n_points
        mapper.add_frame(frame_data)
        cur_n = mapper.n_points
        progress.set_postfix(points=cur_n, refresh=False)
        if cur_n > prev_n:
            snapshot_counts.append(cur_n)
            snapshot_frame_ids.append(frame_id)
            snapshot_c2w.append(np.asarray(frame_data[3], dtype=np.float32).copy())

    mapper.save(
        output_dir,
        {
            "dataset_name": canonical_dataset_name(dataset_name),
            "scene_name": scene_name,
            "n_frames": len(dataset),
            "n_points": mapper.n_points,
            "has_normals": True,
            "device": mapper.device,
            "map_every": mapper.map_every,
            "downscale_res": args.downscale_res,
            "k_pooling": args.k_pooling,
            "max_frame_points": mapper.max_frame_points,
            "match_distance_th": mapper.match_distance_th,
            "clip_model_name": CLIP_MODEL_NAME,
            "clip_pretrained": CLIP_PRETRAINED,
            "clip_load_size": CLIP_LOAD_SIZE,
            "clip_skip_center_crop": True,
            "clip_feature_dim": mapper.clip_extractor.feature_dim,
            "clip_feature_dtype": "float16",
            "clip_feature_path": CLIP_FEATURE_FILE,
            "clip_feature_bytes": mapper.n_points * mapper.clip_extractor.feature_dim * 2,
            "clip_feature_gib": mapper.n_points * mapper.clip_extractor.feature_dim * 2 / 1024**3,
            "snapshot_counts": snapshot_counts,
            "snapshot_frame_ids": snapshot_frame_ids,
        },
    )

    points = mapper.points[: mapper.n_points].cpu().numpy()
    rgb_colors = mapper.colors[: mapper.n_points].cpu().numpy()
    normal_colors = np.clip((mapper.normals[: mapper.n_points].cpu().numpy() + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    feat_colors = (
        apply_pca_colormap_chunked(
            np.load(output_dir / CLIP_FEATURE_FILE, mmap_mode="r"),
            args.pca_sample_size,
            args.chunk_size,
        )
        * 255.0
    ).astype(np.uint8)
    instance_colors = (colorize_instance_labels(resolve_instance_labels(output_dir, points.shape[0], args.min_component_size)) * 255.0).astype(np.uint8)

    video_dir = output_dir / VIDEO_DIR_NAME
    render_incremental_video(video_dir / "rgb.mp4", "RGB", points, rgb_colors, snapshot_counts, snapshot_frame_ids, snapshot_c2w, dataset.intrinsics.astype(np.float32), source_width, source_height, intrinsic, extrinsic, width, height, args.fps, args.dilate)
    render_incremental_video(video_dir / "normals.mp4", "Normals", points, normal_colors, snapshot_counts, snapshot_frame_ids, snapshot_c2w, dataset.intrinsics.astype(np.float32), source_width, source_height, intrinsic, extrinsic, width, height, args.fps, args.dilate)
    render_incremental_video(video_dir / "feat_pca.mp4", "Feature PCA", points, feat_colors, snapshot_counts, snapshot_frame_ids, snapshot_c2w, dataset.intrinsics.astype(np.float32), source_width, source_height, intrinsic, extrinsic, width, height, args.fps, args.dilate)
    instance_title = "GT Instances" if args.use_inst_gt else "SAM Instances"
    instance_name = "gt_instances.mp4" if args.use_inst_gt else "sam_instances.mp4"
    render_incremental_video(video_dir / instance_name, instance_title, points, instance_colors, snapshot_counts, snapshot_frame_ids, snapshot_c2w, dataset.intrinsics.astype(np.float32), source_width, source_height, intrinsic, extrinsic, width, height, args.fps, args.dilate)

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
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RGB+normal+CLIP maps and render incremental view videos.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet", "replica", "scannet"])
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--output_root", default="data/output/topdown_vis")
    parser.add_argument("--load-view", required=True, help="Path to a saved view JSON dumped from visualize_rgb_map.py.")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--map_every", type=int, default=DEFAULT_MAP_EVERY)
    parser.add_argument("--downscale_res", type=int, default=DEFAULT_DOWNSCALE_RES)
    parser.add_argument("--k_pooling", type=int, default=DEFAULT_K_POOLING)
    parser.add_argument("--max_frame_points", type=int, default=DEFAULT_MAX_FRAME_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)
    parser.add_argument("--sam-model-level-inst", type=int, choices=[11, 12, 13], default=13)
    parser.add_argument("--sam-model-level-tr", type=int, choices=[11, 12, 13], default=13)
    parser.add_argument("--use-inst-gt", action="store_true")
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--dilate", type=int, default=POINT_DILATE)
    parser.add_argument("--min_component_size", type=int, default=2000)
    parser.add_argument("--pca_sample_size", type=int, default=DEFAULT_PCA_SAMPLE_SIZE)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    main(parser.parse_args())
