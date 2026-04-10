import json
from pathlib import Path

import numpy as np
import open3d as o3d


ORIENTATION_FILE = "instance_orientations.json"
ORIENTATION_FORMAT = "instance_orientations_v1"
ORIENTATION_AXIS_LENGTH = 0.4
MIN_RENDER_AXIS_LENGTH = 1e-3


def orientation_path_for_map_dir(map_dir: Path) -> Path:
    return map_dir / ORIENTATION_FILE


def orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rotation, dtype=np.float32), full_matrices=False)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation.astype(np.float32, copy=False)


def rotation_to_axis_vectors(rotation: np.ndarray, axis_length: float) -> np.ndarray:
    return (orthonormalize_rotation(rotation) * float(axis_length)).T.astype(np.float32, copy=False)


def axis_vectors_to_rotation_and_scale(axis_vectors: np.ndarray) -> tuple[np.ndarray, float]:
    axis_vectors = np.asarray(axis_vectors, dtype=np.float32)
    if axis_vectors.shape != (3, 3):
        raise ValueError(f"Expected axis vectors with shape (3, 3), found {axis_vectors.shape}")
    lengths = np.linalg.norm(axis_vectors, axis=1)
    scale = float(lengths.mean()) if lengths.size else 0.0
    if scale <= 0.0:
        return np.eye(3, dtype=np.float32), 0.0
    rotation = orthonormalize_rotation((axis_vectors / scale).T)
    return rotation, scale


def build_orientation_axes_mesh(centroid: np.ndarray, axis_vectors: np.ndarray) -> o3d.geometry.TriangleMesh:
    centroid = np.asarray(centroid, dtype=np.float32)
    rotation, _ = axis_vectors_to_rotation_and_scale(axis_vectors)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0.0, 0.0, 0.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (rotation * max(ORIENTATION_AXIS_LENGTH, MIN_RENDER_AXIS_LENGTH)).astype(np.float64, copy=False)
    transform[:3, 3] = centroid.astype(np.float64, copy=False)
    mesh.transform(transform)
    return mesh


def dump_instance_orientations(
    output_path: Path,
    input_ply: Path,
    min_component_size: int,
    instance_ids: np.ndarray,
    centroids: np.ndarray,
    rotations: np.ndarray,
    point_counts: np.ndarray,
) -> None:
    instances = []
    for idx, instance_id in enumerate(instance_ids.tolist()):
        instances.append(
            {
                "instance_id": int(instance_id),
                "n_points": int(point_counts[idx]),
                "centroid": centroids[idx].astype(np.float64, copy=False).tolist(),
                "axes": rotation_to_axis_vectors(rotations[idx], ORIENTATION_AXIS_LENGTH).astype(np.float64, copy=False).tolist(),
            }
        )
    payload = {
        "format": ORIENTATION_FORMAT,
        "input_ply": input_ply.name,
        "min_component_size": int(min_component_size),
        "axis_length": float(ORIENTATION_AXIS_LENGTH),
        "n_instances": len(instances),
        "instances": instances,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def load_instance_orientations(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("format") != ORIENTATION_FORMAT:
        raise ValueError(f"Unsupported orientation format in {path}: {payload.get('format')}")
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError(f"Orientation file {path} is missing an instances list")
    parsed_instances = []
    for entry in instances:
        centroid = np.asarray(entry["centroid"], dtype=np.float32)
        axes = np.asarray(entry["axes"], dtype=np.float32)
        if centroid.shape != (3,):
            raise ValueError(f"Bad centroid shape for instance {entry.get('instance_id')}: {centroid.shape}")
        if axes.shape != (3, 3):
            raise ValueError(f"Bad axes shape for instance {entry.get('instance_id')}: {axes.shape}")
        parsed_instances.append(
            {
                "instance_id": int(entry["instance_id"]),
                "n_points": int(entry.get("n_points", 0)),
                "centroid": centroid,
                "axes": axes,
            }
        )
    payload["instances"] = parsed_instances
    return payload
