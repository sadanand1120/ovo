import argparse
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData
from tqdm.auto import tqdm

from map_runtime.config import load_config


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "configs"
REPLICA_CONFIG_PATH = CONFIG_DIR / "replica.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "input" / "Replica"
DEFAULT_OVO_SEMANTIC_GT_ROOT = DEFAULT_OUTPUT_ROOT / "semantic_gt"
REQUIRED_SCENE_FILES = ("results", "traj.txt")
LABEL_FILT_DIR = "label-filt"
INSTANCE_FILT_DIR = "instance-filt"
FULL_REPLICA_LINK_NAMES = (
    "glass.sur",
    "mesh.ply",
    "preseg.bin",
    "preseg.json",
    "semantic.bin",
    "semantic.json",
    "textures",
    "habitat",
)
FULL_REPLICA_REQUIRED_NAMES = (
    "mesh.ply",
    "semantic.json",
    "semantic.bin",
    "habitat/info_semantic.json",
    "habitat/mesh_semantic.ply",
)


def frame_stem_to_id(stem: str) -> int:
    if stem.startswith("frame"):
        return int(stem[len("frame") :])
    if stem.startswith("depth"):
        return int(stem[len("depth") :])
    return int(stem)


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if (dst.exists() or dst.is_symlink()) and src.resolve() == dst.resolve() and not copy:
        return
    remove_existing(dst)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return
    os.symlink(src.resolve(), dst)


def discover_scenes(source_root: Path) -> list[str]:
    scenes = []
    for scene_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if scene_dir.name == "semantic_gt":
            continue
        if not (scene_dir / "results").is_dir():
            continue
        if not (scene_dir / "traj.txt").is_file():
            continue
        if not (source_root / f"{scene_dir.name}_mesh.ply").is_file():
            continue
        scenes.append(scene_dir.name)
    return scenes


def full_replica_scene_candidates(scene_name: str) -> list[str]:
    candidates = [scene_name]
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", scene_name)
    if match:
        candidates.append(f"{match.group(1)}_{match.group(2)}")
    return candidates


def validate_full_scene(full_replica_root: Path, scene_name: str) -> Path:
    checked = []
    for candidate in full_replica_scene_candidates(scene_name):
        scene_dir = full_replica_root / candidate
        checked.append(str(scene_dir))
        if not scene_dir.is_dir():
            continue
        missing = [rel_name for rel_name in FULL_REPLICA_REQUIRED_NAMES if not (scene_dir / rel_name).exists()]
        if not missing:
            return scene_dir
    raise FileNotFoundError(f"No matching full Replica scene for {scene_name}. Checked: {checked}")


def load_replica_camera_intrinsics(height: int, width: int) -> np.ndarray:
    config = load_config(REPLICA_CONFIG_PATH)
    cam = config["cam"]
    scale_x = float(width) / float(cam["W"])
    scale_y = float(height) / float(cam["H"])
    return np.array(
        [
            [float(cam["fx"]) * scale_x, 0.0, float(cam["cx"]) * scale_x],
            [0.0, float(cam["fy"]) * scale_y, float(cam["cy"]) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_replica_pose_lookup(scene_dir: Path) -> dict[int, np.ndarray]:
    frame_ids = sorted(frame_stem_to_id(path.stem) for path in scene_dir.joinpath("results").glob("frame*.jpg"))
    if not frame_ids:
        frame_ids = sorted(frame_stem_to_id(path.stem) for path in scene_dir.joinpath("results").glob("frame*.png"))
    poses = []
    with open(scene_dir / "traj.txt", "r") as handle:
        for line in handle:
            poses.append(np.array(list(map(float, line.split())), dtype=np.float32).reshape(4, 4))
    if len(frame_ids) != len(poses):
        raise RuntimeError(f"Replica frame/pose mismatch in {scene_dir}: {len(frame_ids)} frames vs {len(poses)} poses.")
    return {frame_id: pose for frame_id, pose in zip(frame_ids, poses)}


def build_depth_frame_lookup(scene_dir: Path) -> dict[int, Path]:
    depth_paths = list(scene_dir.joinpath("results").glob("depth*.png"))
    depth_lookup = {frame_stem_to_id(path.stem): path for path in depth_paths}
    return dict(sorted(depth_lookup.items()))


def prepare_replica_semantic_scene(full_scene_dir: Path):
    semantic_mesh_path = full_scene_dir / "habitat" / "mesh_semantic.ply"
    info_path = full_scene_dir / "habitat" / "info_semantic.json"
    ply = PlyData.read(str(semantic_mesh_path))
    vertex_data = ply["vertex"].data
    face_data = ply["face"].data
    vertices = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float32)
    face_vertices = face_data["vertex_indices"]
    face_object_ids_raw = np.asarray(face_data["object_id"], dtype=np.int32)
    triangles_list = []
    triangle_object_ids = []
    for verts, object_id in zip(face_vertices, face_object_ids_raw):
        verts = np.asarray(verts, dtype=np.int32)
        if verts.shape[0] < 3:
            continue
        for tri_idx in range(1, verts.shape[0] - 1):
            triangles_list.append([int(verts[0]), int(verts[tri_idx]), int(verts[tri_idx + 1])])
            triangle_object_ids.append(int(object_id))
    triangles = np.asarray(triangles_list, dtype=np.int32)
    face_object_ids = np.asarray(triangle_object_ids, dtype=np.int32)
    info = json.loads(info_path.read_text())
    object_to_class_id = {int(obj["id"]): int(obj["class_id"]) for obj in info.get("objects", [])}
    face_class_ids = np.array([object_to_class_id.get(int(obj_id), -1) for obj_id in face_object_ids], dtype=np.int32)

    legacy_mesh = o3d.geometry.TriangleMesh()
    legacy_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    legacy_mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    raycast_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(raycast_mesh)
    return scene, face_object_ids, face_class_ids


def render_replica_2d_gt(source_root: Path, output_scene_dir: Path, full_scene_dir: Path, scene_name: str) -> dict:
    source_scene_dir = source_root / scene_name
    pose_lookup = load_replica_pose_lookup(source_scene_dir)
    depth_lookup = build_depth_frame_lookup(source_scene_dir)
    frame_ids = sorted(set(pose_lookup) & set(depth_lookup))
    if not frame_ids:
        raise RuntimeError(f"No overlapping Replica depth frames and poses for {scene_name}")

    first_depth = cv2.imread(str(depth_lookup[frame_ids[0]]), cv2.IMREAD_UNCHANGED)
    if first_depth is None:
        raise FileNotFoundError(depth_lookup[frame_ids[0]])
    height, width = first_depth.shape[:2]
    intrinsic = load_replica_camera_intrinsics(height, width)
    ray_scene, face_object_ids, face_class_ids = prepare_replica_semantic_scene(full_scene_dir)

    label_dir = output_scene_dir / LABEL_FILT_DIR
    instance_dir = output_scene_dir / INSTANCE_FILT_DIR
    label_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    intrinsic_t = o3d.core.Tensor(intrinsic, dtype=o3d.core.Dtype.Float32)
    for frame_id in tqdm(frame_ids, desc=f"{scene_name} 2d-gt", unit="frame"):
        depth = cv2.imread(str(depth_lookup[frame_id]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(depth_lookup[frame_id])
        if depth.shape[:2] != (height, width):
            raise RuntimeError(f"Inconsistent Replica depth shape in {depth_lookup[frame_id]}: expected {(height, width)}, got {depth.shape[:2]}")
        valid_depth = depth > 0

        w2c = np.linalg.inv(pose_lookup[frame_id]).astype(np.float32)
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsic_t,
            extrinsic_matrix=o3d.core.Tensor(w2c, dtype=o3d.core.Dtype.Float32),
            width_px=width,
            height_px=height,
        )
        ans = ray_scene.cast_rays(rays)
        primitive_ids = ans["primitive_ids"].numpy().astype(np.int64)
        hit_mask = np.isfinite(ans["t_hit"].numpy()) & valid_depth

        semantic_labels = np.zeros((height, width), dtype=np.uint16)
        instance_labels = np.zeros((height, width), dtype=np.uint16)
        if hit_mask.any():
            primitive_hit = primitive_ids[hit_mask]
            instance_hit = face_object_ids[primitive_hit]
            semantic_hit = face_class_ids[primitive_hit]
            instance_labels[hit_mask] = instance_hit.astype(np.uint16, copy=False)
            semantic_valid = semantic_hit >= 0
            if semantic_valid.any():
                semantic_vals = semantic_hit[semantic_valid].astype(np.uint16, copy=False)
                hit_rows, hit_cols = np.where(hit_mask)
                semantic_labels[hit_rows[semantic_valid], hit_cols[semantic_valid]] = semantic_vals

        if not cv2.imwrite(str(label_dir / f"{frame_id}.png"), semantic_labels):
            raise RuntimeError(f"Failed to write semantic labels for frame {frame_id} in {scene_name}")
        if not cv2.imwrite(str(instance_dir / f"{frame_id}.png"), instance_labels):
            raise RuntimeError(f"Failed to write instance labels for frame {frame_id} in {scene_name}")

    return {
        "label_filt_dir": str(label_dir.resolve()),
        "instance_filt_dir": str(instance_dir.resolve()),
        "n_rendered_frames": len(frame_ids),
        "image_size": [int(height), int(width)],
    }


def validate_scene(source_root: Path, scene_name: str) -> dict:
    scene_dir = source_root / scene_name
    mesh_path = source_root / f"{scene_name}_mesh.ply"
    semantic_path = DEFAULT_OVO_SEMANTIC_GT_ROOT / f"{scene_name}.txt"
    if not scene_dir.is_dir():
        raise FileNotFoundError(scene_dir)
    for name in REQUIRED_SCENE_FILES:
        path = scene_dir / name
        if name == "results":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    if not semantic_path.is_file():
        raise FileNotFoundError(semantic_path)

    color_frames = sorted((scene_dir / "results").glob("frame*.jpg"))
    if not color_frames:
        color_frames = sorted((scene_dir / "results").glob("frame*.png"))
    depth_frames = sorted((scene_dir / "results").glob("depth*.png"))
    if not color_frames:
        raise RuntimeError(f"No Replica color frames found under {scene_dir / 'results'}")
    if not depth_frames:
        raise RuntimeError(f"No Replica depth frames found under {scene_dir / 'results'}")
    if len(color_frames) != len(depth_frames):
        raise RuntimeError(
            f"Replica frame mismatch for {scene_name}: {len(color_frames)} color vs {len(depth_frames)} depth."
        )
    return {
        "scene_name": scene_name,
        "n_frames": len(color_frames),
        "mesh_path": str(mesh_path.resolve()),
        "ovo_semantics_path": str(semantic_path.resolve()),
    }


def stage_scene(
    source_root: Path,
    output_root: Path,
    full_replica_root: Path | None,
    scene_name: str,
    copy: bool,
) -> dict:
    summary = validate_scene(source_root, scene_name)
    output_root.mkdir(parents=True, exist_ok=True)
    output_semantic_gt_root = output_root / "semantic_gt"
    output_semantic_gt_root.mkdir(parents=True, exist_ok=True)
    output_scene_dir = output_root / scene_name
    remove_existing(output_scene_dir)
    output_scene_dir.mkdir(parents=True, exist_ok=True)

    link_or_copy(source_root / scene_name / "results", output_scene_dir / "results", copy)
    link_or_copy(source_root / scene_name / "traj.txt", output_scene_dir / "traj.txt", copy)
    link_or_copy(source_root / f"{scene_name}_mesh.ply", output_root / f"{scene_name}_mesh.ply", copy)
    semantic_src = DEFAULT_OVO_SEMANTIC_GT_ROOT / f"{scene_name}.txt"
    semantic_dst = output_semantic_gt_root / f"{scene_name}.txt"
    if semantic_src.resolve() != semantic_dst.resolve():
        link_or_copy(semantic_src, semantic_dst, copy)
    if full_replica_root is not None:
        full_scene_dir = validate_full_scene(full_replica_root, scene_name)
        for name in FULL_REPLICA_LINK_NAMES:
            src = full_scene_dir / name
            if src.exists():
                link_or_copy(src, output_scene_dir / name, copy)
        summary["full_replica_scene_path"] = str(full_scene_dir.resolve())
        summary["filtered_2d_gt"] = render_replica_2d_gt(source_root, output_scene_dir, full_scene_dir, scene_name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage NICE-SLAM Replica data into the runtime layout and, when full Replica assets are provided, render per-frame label-filt/ and instance-filt/ GT.")
    parser.add_argument("--source_root", required=True, type=Path, help="NICE-SLAM Replica root containing scene dirs with results/ + traj.txt and root-level <scene>_mesh.ply files.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, type=Path, help="Runtime Replica root to populate.")
    parser.add_argument("--full_replica_root", default=None, type=Path, help="Optional full Replica root containing per-scene habitat/, semantic.*, mesh.ply, textures, etc. Used to stage Habitat assets and generate label-filt/ + instance-filt/.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene names to stage. Defaults to all valid scenes under source_root.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking them.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    full_replica_root = args.full_replica_root.resolve() if args.full_replica_root is not None else None
    if not DEFAULT_OVO_SEMANTIC_GT_ROOT.is_dir():
        raise FileNotFoundError(DEFAULT_OVO_SEMANTIC_GT_ROOT)
    scenes = args.scenes or discover_scenes(source_root)
    if not scenes:
        raise RuntimeError(f"No valid Replica scenes found under {source_root}")

    staged = []
    for scene_name in scenes:
        staged.append(stage_scene(source_root, output_root, full_replica_root, scene_name, args.copy))

    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "output_root": str(output_root),
                "full_replica_root": str(full_replica_root) if full_replica_root is not None else None,
                "copy": bool(args.copy),
                "n_scenes": len(staged),
                "scenes": staged,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
