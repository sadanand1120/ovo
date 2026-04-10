from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from feature_field_dsl import FeatureFieldScene, ObjSelection, PointSelection, SemanticSpec, close_to, load_scene, object_class


DEFAULT_SCENE_DIR = Path("/home/dynamo/AMRL_Research/repos/ovo/robotemp/gttemp/ScanNet/scene0000_00")
DEFAULT_POINT_SIZE = 2.0
DEFAULT_MAX_POINTS = 5_000_000

DOOR_SPEC = SemanticSpec(
    label="door",
    positive_texts=("door",),
    negative_texts=("object", "floor"),
    scoring="softmax",
    softmax_temp=0.02,
    point_score_threshold=0.40,
    min_point_fraction=0.30,
    min_point_count=2000,
)

TRASH_CAN_SPEC = SemanticSpec(
    label="trash can",
    positive_texts=("trash can",),
    negative_texts=("object", "floor"),
    scoring="softmax",
    softmax_temp=0.02,
    point_score_threshold=0.49,
    min_point_fraction=0.95,
    min_point_count=7000,
)

FLOOR_SPEC = SemanticSpec(
    label="floor",
    positive_texts=("floor",),
    negative_texts=("object",),
    scoring="softmax",
    softmax_temp=0.02,
    point_score_threshold=0.95,
)


@dataclass
class TaskBranch:
    path: list[dict[str, float | int | str]]
    bindings: dict[str, ObjSelection]


@dataclass
class TaskLeaf:
    path: list[dict[str, float | int | str]]
    floor_points: PointSelection

    def to_summary(self) -> dict[str, object]:
        return {
            "path": self.path,
            "n_floor_points": int(self.floor_points.point_indices.shape[0]),
            "point_indices": self.floor_points.point_indices.tolist(),
        }


def log(message: str) -> None:
    print(f"[task] {message}", flush=True)


def _palette(n: int) -> np.ndarray:
    base = np.array(
        [
            [1.00, 0.25, 0.25],
            [0.20, 0.85, 0.25],
            [0.20, 0.55, 1.00],
            [1.00, 0.75, 0.20],
            [0.80, 0.30, 1.00],
            [0.15, 0.90, 0.85],
        ],
        dtype=np.float32,
    )
    if n <= base.shape[0]:
        return base[:n].copy()
    rng = np.random.default_rng(0)
    extra = rng.uniform(0.15, 1.0, size=(n - base.shape[0], 3)).astype(np.float32)
    return np.concatenate([base, extra], axis=0)


def _selection_point_indices(selection: ObjSelection | PointSelection) -> np.ndarray:
    if isinstance(selection, ObjSelection):
        return selection.to_points().point_indices
    return selection.point_indices


def _selection_centroid(scene: FeatureFieldScene, selection: ObjSelection | PointSelection) -> np.ndarray:
    point_indices = _selection_point_indices(selection)
    return scene.points[point_indices].mean(axis=0).astype(np.float32, copy=False)


def _selection_key(selection: ObjSelection | PointSelection) -> bytes:
    return np.asarray(_selection_point_indices(selection), dtype=np.int64).tobytes()


def _selection_summary(selection: ObjSelection | PointSelection) -> str:
    if isinstance(selection, ObjSelection):
        ids = ",".join(str(int(x)) for x in selection.instance_ids.tolist())
        return f"instance_ids=[{ids}]"
    return f"n_points={int(selection.point_indices.shape[0])}"


def _similarity_colormap(scores: np.ndarray) -> np.ndarray:
    lo = float(np.quantile(scores, 0.01))
    hi = float(np.quantile(scores, 0.99))
    denom = max(hi - lo, 1e-6)
    scores01 = np.clip((scores - lo) / denom, 0.0, 1.0).astype(np.float32, copy=False)
    return np.stack(
        [
            scores01,
            1.0 - np.abs(scores01 - 0.5) * 2.0,
            1.0 - scores01,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _show_similarity(scene: FeatureFieldScene, *, title: str, query: SemanticSpec) -> None:
    scores = scene._semantic_point_scores(query)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene.points.astype(np.float64, copy=False))
    pcd.colors = o3d.utility.Vector3dVector(_similarity_colormap(scores).astype(np.float64, copy=False))

    print(f"\n{title}", flush=True)
    print(f"Query: {query.label}", flush=True)
    print("Close the viewer window to continue.", flush=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    render_option = vis.get_render_option()
    render_option.point_size = DEFAULT_POINT_SIZE
    render_option.background_color = np.ones(3, dtype=np.float64)
    vis.run()
    vis.destroy_window()


def _show_selections(
    scene: FeatureFieldScene,
    *,
    title: str,
    selections: list[ObjSelection | PointSelection],
    descriptions: list[str],
) -> None:
    unique_entries: list[dict[str, object]] = []
    grouped: dict[bytes, int] = {}
    for branch_idx, (selection, description) in enumerate(zip(selections, descriptions), start=1):
        key = _selection_key(selection)
        entry_idx = grouped.get(key)
        if entry_idx is None:
            grouped[key] = len(unique_entries)
            unique_entries.append(
                {
                    "selection": selection,
                    "branch_indices": [branch_idx],
                    "descriptions": [description],
                }
            )
        else:
            unique_entries[entry_idx]["branch_indices"].append(branch_idx)
            unique_entries[entry_idx]["descriptions"].append(description)

    base_points = scene.points.astype(np.float64, copy=False)
    base_colors = (scene.colors / 255.0).astype(np.float32, copy=False)
    render_colors = np.clip(base_colors * 0.45 + 0.25, 0.0, 1.0)
    palette = _palette(len(unique_entries))
    marker_radius = max(float(np.linalg.norm(scene.points.max(axis=0) - scene.points.min(axis=0))) * 0.01, 0.04)

    extra_geometries: list[o3d.geometry.Geometry] = []
    for color, entry in zip(palette, unique_entries):
        selection = entry["selection"]
        point_indices = _selection_point_indices(selection)
        render_colors[point_indices] = color
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(color.astype(np.float64, copy=False))
        sphere.translate(_selection_centroid(scene, selection).astype(np.float64, copy=False))
        extra_geometries.append(sphere)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(base_points)
    pcd.colors = o3d.utility.Vector3dVector(render_colors.astype(np.float64, copy=False))

    print(f"\n{title}", flush=True)
    print("Close the viewer window to continue.", flush=True)
    print(
        f"  total_branches={len(selections)}  unique_rendered={len(unique_entries)}",
        flush=True,
    )
    for idx, entry in enumerate(unique_entries, start=1):
        selection = entry["selection"]
        branch_indices = ",".join(str(i) for i in entry["branch_indices"])
        descs = entry["descriptions"]
        if len(descs) == 1:
            desc_text = descs[0]
        else:
            desc_text = f"{descs[0]}  (+{len(descs) - 1} overlapping branches)"
        print(
            f"  [{idx}] branches={branch_indices}  {_selection_summary(selection)}  {desc_text}",
            flush=True,
        )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    for geometry in extra_geometries:
        vis.add_geometry(geometry)
    render_option = vis.get_render_option()
    render_option.point_size = DEFAULT_POINT_SIZE
    render_option.background_color = np.ones(3, dtype=np.float64)
    vis.run()
    vis.destroy_window()


def _path_description(path: list[dict[str, float | int | str]]) -> str:
    if not path:
        return "root"
    return " -> ".join(f"{step['step']}={int(step['instance_id'])}" for step in path)


def _singleton_objects(selection: ObjSelection) -> list[ObjSelection]:
    return [
        ObjSelection(
            scene=selection.scene,
            instance_ids=selection.instance_ids[i : i + 1].copy(),
            scores=selection.scores[i : i + 1].copy(),
            matched_point_fraction=selection.matched_point_fraction[i : i + 1].copy(),
            matched_point_count=selection.matched_point_count[i : i + 1].copy(),
        )
        for i in range(len(selection))
    ]


def _fork_object_step(
    branches: list[TaskBranch],
    step_name: str,
    selector,
) -> list[TaskBranch]:
    forked: list[TaskBranch] = []
    for branch_idx, branch in enumerate(branches, start=1):
        log(f"{step_name}: evaluating branch {branch_idx}/{len(branches)} ({_path_description(branch.path)})")
        selection = selector(branch)
        log(f"{step_name}: branch {branch_idx}/{len(branches)} returned {len(selection)} candidate(s)")
        for obj in _singleton_objects(selection):
            forked.append(
                TaskBranch(
                    path=branch.path
                    + [
                        {
                            "step": step_name,
                            "instance_id": int(obj.instance_ids[0]),
                            "score": float(obj.scores[0]),
                        }
                    ],
                    bindings={**branch.bindings, step_name: obj},
                )
            )
    return forked


def _log_closest_floor_distance(scene: FeatureFieldScene, branches: list[TaskBranch]) -> None:
    floor_points = scene._semantic_points(FLOOR_SPEC)
    if not floor_points:
        log("floor debug: semantic floor selection is empty")
        return
    floor_xyz = scene.points[floor_points.point_indices]
    floor_tree = cKDTree(floor_xyz)
    log(f"floor debug: semantic floor support has {floor_points.point_indices.shape[0]} point(s)")
    for branch_idx, branch in enumerate(branches, start=1):
        trash_can = branch.bindings["trash_can"]
        trash_can_points = scene.points[trash_can.to_points().point_indices]
        dists, _ = floor_tree.query(trash_can_points, k=1, workers=-1)
        min_dist = float(np.min(dists)) if dists.size > 0 else float("inf")
        log(
            f"floor debug: branch {branch_idx}/{len(branches)} ({_path_description(branch.path)}) "
            f"closest semantic-floor distance={min_dist:.3f} m"
        )


def solve_task(scene_dir: str | Path = DEFAULT_SCENE_DIR) -> list[TaskLeaf]:
    log(f"loading scene: {scene_dir}")
    log(f"downsampling to at most {DEFAULT_MAX_POINTS} points during load")
    scene = load_scene(scene_dir, max_points=DEFAULT_MAX_POINTS)
    log(f"scene ready: {scene.points.shape[0]} points, {scene.n_instances} instances")
    log("prefetching semantic queries for door, trash can, and floor")
    scene.prefetch_semantic_scores(DOOR_SPEC, TRASH_CAN_SPEC, FLOOR_SPEC)
    log("semantic prefetch ready")

    branches = [TaskBranch(path=[], bindings={})]
    log("step 1/3: finding door candidates")
    branches = _fork_object_step(branches, "door", lambda _branch: object_class(DOOR_SPEC))
    _show_similarity(scene, title="Semantic similarity: door", query=DOOR_SPEC)
    log(f"step 1/3 complete: {len(branches)} branch(es)")
    _show_selections(
        scene,
        title="Step 1: door candidates",
        selections=[branch.bindings["door"] for branch in branches],
        descriptions=[_path_description(branch.path) for branch in branches],
    )
    log("step 2/3: finding trash cans near each door branch")
    branches = _fork_object_step(
        branches,
        "trash_can",
        lambda branch: close_to(
            branch.bindings["door"],
            TRASH_CAN_SPEC,
            distance_threshold=3.0,
            wholeobj_or_points="wholeobj",
            wholeobj_min_point_fraction=0.05,
            wholeobj_min_point_count=100,
        ),
    )
    _show_similarity(scene, title="Semantic similarity: trash can", query=TRASH_CAN_SPEC)
    log(f"step 2/3 complete: {len(branches)} branch(es)")
    _show_selections(
        scene,
        title="Step 2: trash cans near doors",
        selections=[branch.bindings["trash_can"] for branch in branches],
        descriptions=[_path_description(branch.path) for branch in branches],
    )
    _log_closest_floor_distance(scene, branches)

    leaves: list[TaskLeaf] = []
    log("step 3/3: finding floor-point leaves near each selected trash can")
    for branch_idx, branch in enumerate(branches, start=1):
        log(f"floor: evaluating branch {branch_idx}/{len(branches)} ({_path_description(branch.path)})")
        floor_points = close_to(
            branch.bindings["trash_can"],
            FLOOR_SPEC,
            distance_threshold=0.45,
            wholeobj_or_points="points",
        )
        if floor_points:
            leaves.append(TaskLeaf(path=branch.path, floor_points=floor_points))
            log(f"floor: branch {branch_idx}/{len(branches)} returned {len(floor_points)} point(s)")
        else:
            log(f"floor: branch {branch_idx}/{len(branches)} returned no points")
    _show_similarity(scene, title="Semantic similarity: floor", query=FLOOR_SPEC)
    log(f"step 3/3 complete: {len(leaves)} leaf selection(s)")
    _show_selections(
        scene,
        title="Final: floor targets near selected trash cans",
        selections=[leaf.floor_points for leaf in leaves],
        descriptions=[_path_description(leaf.path) for leaf in leaves],
    )
    return leaves


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve floor-point leaves for: Throw the trash in trash can near door.")
    parser.add_argument("scene_dir", nargs="?", default=str(DEFAULT_SCENE_DIR))
    args = parser.parse_args()

    leaves = solve_task(args.scene_dir)
    # print(json.dumps([leaf.to_summary() for leaf in leaves], indent=2))  # no need to print the summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
