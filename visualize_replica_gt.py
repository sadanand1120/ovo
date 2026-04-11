import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


DEFAULT_REPLICA_ROOT = Path("data/input/Replica")
DEFAULT_POINT_SIZE = 6.0


def load_ply_vertices(ply_path: Path):
    vertex = PlyData.read(str(ply_path))["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    normals = None
    if {"nx", "ny", "nz"}.issubset(vertex.dtype.names):
        normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
    return points, colors, normals


def read_label_txt(path: Path) -> np.ndarray:
    return np.array(path.read_text().splitlines(), dtype=np.int32)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
    valid = labels >= 0
    if not valid.any():
        return colors
    unique_labels, inverse = np.unique(labels[valid], return_inverse=True)
    palette = np.random.default_rng(0).random((unique_labels.shape[0], 3), dtype=np.float32)
    colors[valid] = palette[inverse]
    return colors


def resolve_mesh_path(replica_root: Path, scene_name: str) -> Path:
    mesh_path = replica_root / f"{scene_name}_mesh.ply"
    if not mesh_path.exists():
        raise ValueError(f"Missing Replica mesh: {mesh_path}")
    return mesh_path


def resolve_semantic_path(replica_root: Path, scene_name: str) -> Path:
    semantic_path = replica_root / "semantic_gt" / f"{scene_name}.txt"
    if not semantic_path.exists():
        raise ValueError(f"Missing Replica semantic GT labels: {semantic_path}")
    return semantic_path


def load_rgb(replica_root: Path, scene_name: str):
    mesh_path = resolve_mesh_path(replica_root, scene_name)
    points, colors, _ = load_ply_vertices(mesh_path)
    return points, colors.astype(np.float32) / 255.0


def load_normals(replica_root: Path, scene_name: str):
    mesh_path = resolve_mesh_path(replica_root, scene_name)
    points, _, normals = load_ply_vertices(mesh_path)
    if normals is None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        points = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    return points, colors


def load_semantics(replica_root: Path, scene_name: str):
    mesh_path = resolve_mesh_path(replica_root, scene_name)
    semantic_path = resolve_semantic_path(replica_root, scene_name)
    points, _, _ = load_ply_vertices(mesh_path)
    labels = read_label_txt(semantic_path)
    if labels.shape[0] != points.shape[0]:
        raise ValueError(
            f"Replica semantic GT size mismatch for {scene_name}: "
            f"{labels.shape[0]} labels vs {points.shape[0]} mesh vertices."
        )
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
    replica_root = Path(args.replica_root)
    loaders = {
        "rgb": load_rgb,
        "normals": load_normals,
        "semantics": load_semantics,
    }
    loaded = loaders[args.mode](replica_root, args.scene_name)
    if len(loaded) == 3:
        points, colors, labels = loaded
    else:
        points, colors = loaded
        labels = None

    summary = {
        "scene_name": args.scene_name,
        "mode": args.mode,
        "replica_root": str(replica_root),
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
    parser = argparse.ArgumentParser(description="Visualize Replica ground-truth scene data.")
    parser.add_argument("scene_name", help="Replica scene name, e.g. office0")
    parser.add_argument("--mode", choices=["rgb", "normals", "semantics"], default="rgb")
    parser.add_argument("--replica_root", default=str(DEFAULT_REPLICA_ROOT))
    parser.add_argument("--point_size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--no_window", action="store_true")
    main(parser.parse_args())
