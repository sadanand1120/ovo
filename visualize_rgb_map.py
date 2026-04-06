import argparse
import json
from pathlib import Path

import open3d as o3d


def resolve_ply_path(input_path: str) -> Path:
    path = Path(input_path)
    return path if path.suffix == ".ply" else path / "rgb_map.ply"


def main(args):
    ply_path = resolve_ply_path(args.input_path)
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"Loaded {ply_path} with {len(pcd.points)} points")
    stats_path = ply_path.with_name("stats.json")
    if stats_path.exists():
        print(json.dumps(json.loads(stats_path.read_text()), indent=2))
    if not args.no_window:
        o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a saved RGB pointcloud map.")
    parser.add_argument("input_path", help="Path to rgb_map.ply or its containing directory.")
    parser.add_argument("--no_window", action="store_true", help="Only load and print map info.")
    main(parser.parse_args())
