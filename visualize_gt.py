import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from get_metrics_map import (
    load_ply_vertices,
    load_replica_gt,
    load_replica_vertex_instance_labels,
    load_scannet_gt,
    resolve_replica_gt_paths,
)
from visualize_rgb_map import colorize_instance_labels, show_point_cloud


DEFAULT_REPLICA_ROOT = Path("data/input/Replica")
DEFAULT_POINT_SIZE = 3.0
SCANNET_MODES = {"rgb", "normals", "semantics", "instances"}
REPLICA_MODES = {"rgb", "normals", "semantics", "ovo-semantics", "instances"}


def build_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def load_scannet_rgb(scene_name: str, raw_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mesh_path = raw_root / scene_name / f"{scene_name}_vh_clean_2.ply"
    points, colors, _, _ = load_ply_vertices(mesh_path)
    return points, colors.astype(np.float32) / 255.0, None


def load_scannet_normals(scene_name: str, raw_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mesh_path = raw_root / scene_name / f"{scene_name}_vh_clean_2.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    points = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    return points, np.clip((normals + 1.0) * 0.5, 0.0, 1.0), None


def load_scannet_mode(scene_name: str, raw_root: Path, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if mode == "rgb":
        return load_scannet_rgb(scene_name, raw_root)
    if mode == "normals":
        return load_scannet_normals(scene_name, raw_root)
    gt = load_scannet_gt(scene_name, raw_root)
    if mode == "semantics":
        labels = gt["semantic_raw"]
    elif mode == "instances":
        labels = gt["instance_labels"]
    else:
        raise ValueError(f"Unsupported ScanNet mode: {mode}")
    return gt["points"], colorize_instance_labels(labels), labels


def load_replica_rgb(scene_name: str, replica_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    _, mesh_path, _ = resolve_replica_gt_paths(scene_name, replica_root)
    points, colors, _, _ = load_ply_vertices(mesh_path)
    return points, colors.astype(np.float32) / 255.0, None


def load_replica_normals(scene_name: str, replica_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    gt = load_replica_gt(scene_name, replica_root)
    normals = gt["normals"]
    return gt["points"], np.clip((normals + 1.0) * 0.5, 0.0, 1.0), None


def load_replica_habitat_semantics(scene_name: str, replica_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, habitat_mesh_path = resolve_replica_gt_paths(scene_name, replica_root)
    points, instance_labels = load_replica_vertex_instance_labels(habitat_mesh_path)
    info = json.loads((habitat_mesh_path.parent / "info_semantic.json").read_text())
    object_id_to_class_id = {int(obj["id"]): int(obj["class_id"]) for obj in info.get("objects", [])}
    labels = np.full(instance_labels.shape, -1, dtype=np.int32)
    valid = instance_labels >= 0
    if valid.any():
        labels[valid] = np.array([object_id_to_class_id.get(int(obj_id), -1) for obj_id in instance_labels[valid]], dtype=np.int32)
    return points, colorize_instance_labels(labels), labels


def load_replica_mode(scene_name: str, replica_root: Path, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if mode == "rgb":
        return load_replica_rgb(scene_name, replica_root)
    if mode == "normals":
        return load_replica_normals(scene_name, replica_root)
    gt = load_replica_gt(scene_name, replica_root)
    if mode == "ovo-semantics":
        labels = gt["semantic_raw"]
        return gt["points"], colorize_instance_labels(labels), labels
    if mode == "instances":
        labels = gt["instance_labels"]
        return gt["points"], colorize_instance_labels(labels), labels
    if mode == "semantics":
        return load_replica_habitat_semantics(scene_name, replica_root)
    raise ValueError(f"Unsupported Replica mode: {mode}")


def summarize_points(dataset_name: str, scene_name: str, mode: str, points: np.ndarray, labels: np.ndarray | None) -> dict:
    summary = {
        "dataset_name": dataset_name,
        "scene_name": scene_name,
        "mode": mode,
        "n_points": int(points.shape[0]),
        "bbox_min": np.round(points.min(axis=0), 3).tolist(),
        "bbox_max": np.round(points.max(axis=0), 3).tolist(),
    }
    if labels is not None:
        valid = labels >= 0
        summary["n_labeled_points"] = int(valid.sum())
        summary["n_unique_labels"] = int(np.unique(labels[valid]).shape[0]) if valid.any() else 0
    return summary


def main(args: argparse.Namespace) -> None:
    dataset_name = args.dataset_name
    if dataset_name == "ScanNet":
        if args.mode not in SCANNET_MODES:
            raise ValueError(f"ScanNet mode must be one of {sorted(SCANNET_MODES)}")
        if args.scannet_raw_root is None:
            raise ValueError("--scannet_raw_root is required for ScanNet GT visualization.")
        points, colors, labels = load_scannet_mode(args.scene_name, Path(args.scannet_raw_root), args.mode)
    elif dataset_name == "Replica":
        if args.mode not in REPLICA_MODES:
            raise ValueError(f"Replica mode must be one of {sorted(REPLICA_MODES)}")
        points, colors, labels = load_replica_mode(args.scene_name, Path(args.replica_root), args.mode)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(json.dumps(summarize_points(dataset_name, args.scene_name, args.mode, points, labels), indent=2))
    if not args.no_window:
        show_point_cloud(build_point_cloud(points, colors), args.point_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ScanNet or Replica GT scene data.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet"])
    parser.add_argument("scene_name")
    parser.add_argument("--mode", default="rgb")
    parser.add_argument("--scannet_raw_root", default=None, help="Required for ScanNet.")
    parser.add_argument("--replica_root", default=str(DEFAULT_REPLICA_ROOT), help="Replica root containing semantic_gt/ and scene meshes.")
    parser.add_argument("--point_size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--no_window", action="store_true")
    main(parser.parse_args())
