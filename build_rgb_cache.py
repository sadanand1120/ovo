import argparse
import json
from pathlib import Path

import cv2  # Keep OpenCV loaded before torch in the container env.

from map_runtime.rgb_scene_cache import run_scene_cache_build


OUTPUT_DIR = Path("data/output/rgb_caches")
DEFAULT_MAP_EVERY = 8
DEFAULT_POINT_SAMPLE_STRIDE = 2
DEFAULT_MAX_FRAME_POINTS = 5_000_000
DEFAULT_MATCH_DISTANCE_TH = 0.03


def main(args) -> None:
    output_dir, timing_summary = run_scene_cache_build(
        dataset_name=args.dataset_name,
        scene_name=args.scene_name,
        output_root=args.output_root,
        frame_limit=args.frame_limit,
        slam_module=args.slam_module,
        disable_loop_closure=args.disable_loop_closure,
        config_path=args.config_path,
        map_every=args.map_every,
        point_sample_stride=args.point_sample_stride,
        max_frame_points=args.max_frame_points,
        match_distance_th=args.match_distance_th,
    )
    print(json.dumps({"timing": timing_summary}, indent=2))
    print(output_dir / "cache_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an RGB-map cache with geometry/CLIP outputs and per-frame point-id caches.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet"])
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--output_root", default=str(OUTPUT_DIR))
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--slam_module", type=str, default=None, help="Override slam backend, e.g. vanilla, orbslam, or cuvslam.")
    parser.add_argument("--disable_loop_closure", action="store_true", help="Disable ORB-SLAM loop closure/global BA updates by forcing slam.close_loops=false.")
    parser.add_argument("--config_path", type=str, default="configs/ovo.yaml", help="Base runtime config file to load.")
    parser.add_argument("--map_every", type=int, default=DEFAULT_MAP_EVERY)
    parser.add_argument("--point_sample_stride", type=int, default=DEFAULT_POINT_SAMPLE_STRIDE, help="Seed-frame point-sampling stride used for geometry/normal fusion before point birth.")
    parser.add_argument("--max_frame_points", type=int, default=DEFAULT_MAX_FRAME_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)
    main(parser.parse_args())
