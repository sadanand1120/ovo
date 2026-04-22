import argparse
import json
from pathlib import Path

import cv2  # Keep OpenCV loaded before torch in the container env.

from map_runtime.defaults import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MAP_EVERY,
    DEFAULT_MATCH_DISTANCE_TH,
    DEFAULT_MAX_TOTAL_POINTS,
    DEFAULT_RGB_CACHE_OUTPUT_ROOT,
)
from map_runtime.rgb_scene_cache import run_scene_cache_build

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
        max_total_points=args.max_total_points,
        match_distance_th=args.match_distance_th,
    )
    print(json.dumps({"timing": timing_summary}, indent=2))
    print(output_dir / "cache_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an RGB-map cache with geometry/CLIP outputs and per-frame point-id caches.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet"])
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--output_root", default=str(DEFAULT_RGB_CACHE_OUTPUT_ROOT))
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--slam_module", type=str, default=None, help="Override slam backend, e.g. vanilla, orbslam, or cuvslam.")
    parser.add_argument("--disable_loop_closure", action="store_true", help="Disable ORB-SLAM loop closure/global BA updates by forcing slam.close_loops=false.")
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH), help="Base runtime config file to load.")
    parser.add_argument("--map_every", type=int, default=DEFAULT_MAP_EVERY)
    parser.add_argument("--max_total_points", type=int, default=DEFAULT_MAX_TOTAL_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)
    main(parser.parse_args())
