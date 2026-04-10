import argparse
import json
import sys
from pathlib import Path

import cv2  # Keep OpenCV loaded before torch/build_rgb_map in the container env.
import numpy as np
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_rgb_map import (  # noqa: E402
    CLIP_FEATURE_FILE,
    CLIP_LOAD_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DEFAULT_DOWNSCALE_RES,
    DEFAULT_K_POOLING,
    DEFAULT_MAP_EVERY,
    DEFAULT_MATCH_DISTANCE_TH,
    DEFAULT_MAX_FRAME_POINTS,
    canonical_dataset_name,
    load_dataset,
)
from topdown_vis import (  # noqa: E402
    POINT_DILATE,
    VIDEO_DIR_NAME,
    VIDEO_FPS,
    load_view,
    render_incremental_video,
)
from visualize_rgb_map import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PCA_SAMPLE_SIZE,
    apply_pca_colormap_chunked,
    colorize_instance_labels,
    resolve_instance_labels,
)
from gt_equiv.build_rgb_map import (  # noqa: E402
    GTEquivalentRGBMapper,
)
from ovo.sam_mask_utils import SAM1_LEVELS  # noqa: E402
from ovo.sam2_utils import SAM2_LEVELS  # noqa: E402


def main(args: argparse.Namespace) -> None:
    dataset_name = args.dataset_name.lower()
    scene_name = args.scene_name
    output_dir = Path(args.output_root) / canonical_dataset_name(dataset_name) / scene_name
    dataset = load_dataset(dataset_name, scene_name, args.frame_limit)
    intrinsic, extrinsic, width, height = load_view(Path(args.load_view))

    mapper = GTEquivalentRGBMapper(
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
        sam_model_level_textregion=args.sam_model_level_textregion,
        sam2_model_level_track=args.sam2_model_level_track,
        scannet_raw_root=args.scannet_raw_root,
        normals_noise_deg=args.normals_noise_deg,
        instance_num_instances=args.instance_num_instances,
        instance_noise_frac=args.instance_noise_frac,
        instance_seed=args.instance_seed,
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
    instance_colors = (
        colorize_instance_labels(resolve_instance_labels(output_dir, points.shape[0], args.min_component_size)) * 255.0
    ).astype(np.uint8)

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
    parser = argparse.ArgumentParser(description="Build GT-equivalent RGB+normal+CLIP maps and render incremental view videos.")
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
    parser.add_argument("--sam-model-level-inst", type=int, choices=sorted(SAM1_LEVELS), default=13)
    parser.add_argument("--sam-model-level-textregion", type=int, choices=sorted(SAM1_LEVELS), default=13)
    parser.add_argument("--sam2-model-level-track", type=int, choices=sorted(SAM2_LEVELS), default=24)
    parser.add_argument("--use-inst-gt", action="store_true")
    parser.add_argument("--scannet-raw-root", default="../../dataset/scannet_v2/scans")
    parser.add_argument("--normals-noise-deg", type=float, default=0.0)
    parser.add_argument("--instance-num-instances", type=int, default=0)
    parser.add_argument("--instance-noise-frac", type=float, default=0.0)
    parser.add_argument("--instance-seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--dilate", type=int, default=POINT_DILATE)
    parser.add_argument("--min_component_size", type=int, default=2000)
    parser.add_argument("--pca_sample_size", type=int, default=DEFAULT_PCA_SAMPLE_SIZE)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    main(parser.parse_args())
