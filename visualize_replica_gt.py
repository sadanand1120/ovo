import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


DEFAULT_REPLICA_ROOT = Path("data/input/Replica")
DEFAULT_POINT_SIZE = 6.0
INSTANCE_FACE_FIELD = "object_id"


def load_ply_vertices(ply_path: Path):
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    normals = None
    if {"nx", "ny", "nz"}.issubset(vertex.dtype.names):
        normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
    return points, colors, normals, ply


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


def resolve_ovo_semantic_path(replica_root: Path, scene_name: str) -> Path:
    semantic_path = replica_root / "semantic_gt" / f"{scene_name}.txt"
    if not semantic_path.exists():
        raise ValueError(f"Missing Replica ovo-semantics labels: {semantic_path}")
    return semantic_path


def resolve_habitat_semantic_mesh_path(replica_root: Path, scene_name: str) -> Path:
    mesh_path = replica_root / scene_name / "habitat" / "mesh_semantic.ply"
    if not mesh_path.exists():
        raise ValueError(f"Missing Replica habitat semantic mesh: {mesh_path}")
    return mesh_path


def resolve_habitat_info_path(replica_root: Path, scene_name: str) -> Path:
    info_path = replica_root / scene_name / "habitat" / "info_semantic.json"
    if not info_path.exists():
        raise ValueError(f"Missing Replica habitat semantic info: {info_path}")
    return info_path


def project_face_labels_to_vertices(face_vertices, face_labels: np.ndarray, n_vertices: int) -> np.ndarray:
    lengths = np.fromiter((len(v) for v in face_vertices), dtype=np.int32, count=len(face_vertices))
    vertex_indices = np.concatenate(face_vertices).astype(np.int64)
    repeated_labels = np.repeat(face_labels.astype(np.int64), lengths)
    pairs = np.stack([vertex_indices, repeated_labels], axis=1)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    order = np.lexsort((-counts, unique_pairs[:, 0]))
    ordered_pairs = unique_pairs[order]
    first_per_vertex = np.ones(ordered_pairs.shape[0], dtype=bool)
    first_per_vertex[1:] = ordered_pairs[1:, 0] != ordered_pairs[:-1, 0]
    selected = ordered_pairs[first_per_vertex]
    labels = np.full(n_vertices, -1, dtype=np.int32)
    labels[selected[:, 0].astype(np.int64)] = selected[:, 1].astype(np.int32)
    return labels


def load_habitat_vertex_instance_labels(replica_root: Path, scene_name: str):
    mesh_path = resolve_habitat_semantic_mesh_path(replica_root, scene_name)
    points, _, _, ply = load_ply_vertices(mesh_path)
    face = ply["face"].data
    if INSTANCE_FACE_FIELD not in face.dtype.names:
        raise ValueError(f"Replica habitat semantic mesh missing face field '{INSTANCE_FACE_FIELD}': {mesh_path}")
    face_vertices = face["vertex_indices"]
    face_labels = np.asarray(face[INSTANCE_FACE_FIELD], dtype=np.int32)
    vertex_labels = project_face_labels_to_vertices(face_vertices, face_labels, points.shape[0])
    return points, vertex_labels


def load_habitat_info(replica_root: Path, scene_name: str) -> dict:
    return json.loads(resolve_habitat_info_path(replica_root, scene_name).read_text())


def load_rgb(replica_root: Path, scene_name: str):
    mesh_path = resolve_mesh_path(replica_root, scene_name)
    points, colors, _, _ = load_ply_vertices(mesh_path)
    return points, colors.astype(np.float32) / 255.0


def load_normals(replica_root: Path, scene_name: str):
    mesh_path = resolve_mesh_path(replica_root, scene_name)
    points, _, normals, _ = load_ply_vertices(mesh_path)
    if normals is None:
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        points = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0)
    return points, colors


def load_ovo_semantics(replica_root: Path, scene_name: str):
    mesh_path = resolve_mesh_path(replica_root, scene_name)
    semantic_path = resolve_ovo_semantic_path(replica_root, scene_name)
    points, _, _, _ = load_ply_vertices(mesh_path)
    labels = read_label_txt(semantic_path)
    if labels.shape[0] != points.shape[0]:
        raise ValueError(
            f"Replica ovo-semantics size mismatch for {scene_name}: "
            f"{labels.shape[0]} labels vs {points.shape[0]} mesh vertices."
        )
    return points, colorize_labels(labels), labels


def load_instances(replica_root: Path, scene_name: str):
    points, labels = load_habitat_vertex_instance_labels(replica_root, scene_name)
    return points, colorize_labels(labels), labels


def load_semantics(replica_root: Path, scene_name: str):
    points, instance_labels = load_habitat_vertex_instance_labels(replica_root, scene_name)
    info = load_habitat_info(replica_root, scene_name)
    object_id_to_class_id = {int(obj["id"]): int(obj["class_id"]) for obj in info.get("objects", [])}
    labels = np.full(instance_labels.shape, -1, dtype=np.int32)
    valid = instance_labels >= 0
    if valid.any():
        mapped = np.array([object_id_to_class_id.get(int(obj_id), -1) for obj_id in instance_labels[valid]], dtype=np.int32)
        labels[valid] = mapped
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
        "ovo-semantics": load_ovo_semantics,
        "instances": load_instances,
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
    parser = argparse.ArgumentParser(description="Visualize Replica GT scene data. 'semantics' and 'instances' use Habitat assets; 'ovo-semantics' uses the tracked txt labels.")
    parser.add_argument("scene_name", help="Replica scene name, e.g. office0")
    parser.add_argument("--mode", choices=["rgb", "normals", "semantics", "ovo-semantics", "instances"], default="rgb")
    parser.add_argument("--replica_root", default=str(DEFAULT_REPLICA_ROOT))
    parser.add_argument("--point_size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--no_window", action="store_true")
    main(parser.parse_args())
