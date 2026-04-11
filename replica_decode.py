import argparse
import json
import os
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "input" / "Replica"
DEFAULT_SEMANTIC_GT_ROOT = DEFAULT_OUTPUT_ROOT / "semantic_gt"
REQUIRED_SCENE_FILES = ("results", "traj.txt")
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


def validate_scene(source_root: Path, scene_name: str) -> dict:
    scene_dir = source_root / scene_name
    mesh_path = source_root / f"{scene_name}_mesh.ply"
    semantic_path = DEFAULT_SEMANTIC_GT_ROOT / f"{scene_name}.txt"
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
        "semantic_gt_path": str(semantic_path.resolve()),
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
    semantic_src = DEFAULT_SEMANTIC_GT_ROOT / f"{scene_name}.txt"
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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage NICE-SLAM Replica data and optional full Replica scene assets into the runtime layout.")
    parser.add_argument("--source_root", required=True, type=Path, help="NICE-SLAM Replica root containing scene dirs with results/ + traj.txt and root-level <scene>_mesh.ply files.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, type=Path, help="Runtime Replica root to populate.")
    parser.add_argument("--full_replica_root", default=None, type=Path, help="Optional full Replica root containing per-scene habitat/, semantic.*, mesh.ply, textures, etc.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene names to stage. Defaults to all valid scenes under source_root.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking them.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    full_replica_root = args.full_replica_root.resolve() if args.full_replica_root is not None else None
    if not DEFAULT_SEMANTIC_GT_ROOT.is_dir():
        raise FileNotFoundError(DEFAULT_SEMANTIC_GT_ROOT)
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
