import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


DEFAULT_POINT_SIZE = 2.5


def load_ply_vertices(ply_path: Path):
    vertex = PlyData.read(str(ply_path))["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    labels = np.asarray(vertex["label"], dtype=np.int32) if "label" in vertex.dtype.names else None
    return points, colors, labels


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    valid = labels >= 0
    if not valid.any():
        return colors
    unique_labels, inverse = np.unique(labels[valid], return_inverse=True)
    palette = np.random.default_rng(0).random((unique_labels.shape[0], 3), dtype=np.float32)
    colors[valid] = palette[inverse]
    return colors


def load_rgb(scene_dir: Path):
    points, colors, _ = load_ply_vertices(scene_dir / f"{scene_dir.name}_vh_clean_2.ply")
    return points, colors.astype(np.float32) / 255.0


def load_normals(scene_dir: Path):
    mesh = o3d.io.read_triangle_mesh(str(scene_dir / f"{scene_dir.name}_vh_clean_2.ply"))
    mesh.compute_vertex_normals()
    points = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    return points, colors


def load_semantics(scene_dir: Path):
    points, _, labels = load_ply_vertices(scene_dir / f"{scene_dir.name}_vh_clean_2.labels.ply")
    if labels is None:
        raise ValueError("Semantic labels not present in labels PLY.")
    return points, colorize_labels(labels), labels


def load_instances(scene_dir: Path):
    scene_name = scene_dir.name
    points, _, _ = load_ply_vertices(scene_dir / f"{scene_name}_vh_clean_2.labels.ply")
    seg_indices = np.asarray(json.loads((scene_dir / f"{scene_name}_vh_clean_2.0.010000.segs.json").read_text())["segIndices"], dtype=np.int32)
    agg = json.loads((scene_dir / f"{scene_name}.aggregation.json").read_text())
    labels = np.full(points.shape[0], -1, dtype=np.int32)
    for inst_id, group in enumerate(agg["segGroups"]):
        labels[np.isin(seg_indices, group["segments"])] = inst_id
    return points, colorize_labels(labels), labels


def show_point_cloud(points: np.ndarray, colors: np.ndarray, point_size: float) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()


def main(args: argparse.Namespace) -> None:
    scene_dir = Path(args.raw_root) / args.scene_name
    if not scene_dir.exists():
        raise ValueError(f"Scene not found: {scene_dir}")

    loaders = {
        "rgb": load_rgb,
        "normals": load_normals,
        "semantics": load_semantics,
        "instances": load_instances,
    }
    loaded = loaders[args.mode](scene_dir)
    if len(loaded) == 3:
        points, colors, labels = loaded
    else:
        points, colors = loaded
        labels = None

    summary = {
        "scene_name": args.scene_name,
        "mode": args.mode,
        "n_points": int(points.shape[0]),
        "bbox_min": np.round(points.min(axis=0), 3).tolist(),
        "bbox_max": np.round(points.max(axis=0), 3).tolist(),
    }
    if labels is not None:
        valid = labels >= 0
        summary["n_labeled_points"] = int(valid.sum())
        summary["n_unique_labels"] = int(np.unique(labels[valid]).shape[0]) if valid.any() else 0
    print(json.dumps(summary, indent=2))
    if not args.no_window:
        show_point_cloud(points, colors, args.point_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ScanNet raw ground-truth scene data.")
    parser.add_argument("scene_name", help="ScanNet scene name, e.g. scene0011_00")
    parser.add_argument("--mode", choices=["rgb", "normals", "semantics", "instances"], default="rgb")
    parser.add_argument("--raw_root", required=True, help="Path to ScanNet raw scans root.")
    parser.add_argument("--point_size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--no_window", action="store_true")
    main(parser.parse_args())
