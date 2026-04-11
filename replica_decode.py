import argparse
import json
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "input" / "Replica"
REPO_SEMANTIC_GT_ROOT = DEFAULT_OUTPUT_ROOT / "semantic_gt"
REQUIRED_SCENE_FILES = ("results", "traj.txt")


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
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


def resolve_semantic_gt_root(source_root: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        root = explicit_root
    elif (source_root / "semantic_gt").is_dir():
        root = source_root / "semantic_gt"
    elif REPO_SEMANTIC_GT_ROOT.is_dir():
        root = REPO_SEMANTIC_GT_ROOT
    else:
        raise FileNotFoundError(
            "Could not find Replica semantic_gt. Pass --semantic_gt_root or create source_root/semantic_gt."
        )
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def validate_scene(source_root: Path, semantic_gt_root: Path, scene_name: str) -> dict:
    scene_dir = source_root / scene_name
    mesh_path = source_root / f"{scene_name}_mesh.ply"
    semantic_path = semantic_gt_root / f"{scene_name}.txt"
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
    semantic_gt_root: Path,
    scene_name: str,
    copy: bool,
) -> dict:
    summary = validate_scene(source_root, semantic_gt_root, scene_name)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "semantic_gt").mkdir(parents=True, exist_ok=True)

    link_or_copy(source_root / scene_name, output_root / scene_name, copy)
    link_or_copy(source_root / f"{scene_name}_mesh.ply", output_root / f"{scene_name}_mesh.ply", copy)
    link_or_copy(semantic_gt_root / f"{scene_name}.txt", output_root / "semantic_gt" / f"{scene_name}.txt", copy)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage Replica rendered RGB-D scenes into the runtime layout.")
    parser.add_argument("--source_root", required=True, type=Path, help="Raw Replica root containing scene dirs and <scene>_mesh.ply files.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, type=Path, help="Runtime Replica root to populate.")
    parser.add_argument("--semantic_gt_root", default=None, type=Path, help="Optional semantic_gt directory containing <scene>.txt files.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene names to stage. Defaults to all valid scenes under source_root.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking them.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    semantic_gt_root = resolve_semantic_gt_root(source_root, args.semantic_gt_root.resolve() if args.semantic_gt_root is not None else None)
    scenes = args.scenes or discover_scenes(source_root)
    if not scenes:
        raise RuntimeError(f"No valid Replica scenes found under {source_root}")

    staged = []
    for scene_name in scenes:
        staged.append(stage_scene(source_root, output_root, semantic_gt_root, scene_name, args.copy))

    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "output_root": str(output_root),
                "semantic_gt_root": str(semantic_gt_root.resolve()),
                "copy": bool(args.copy),
                "n_scenes": len(staged),
                "scenes": staged,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
