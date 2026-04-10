from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from feature_field_dsl import ObjSelection, PointSelection, SemanticSpec, behind, close_to, load_scene, object_class
from task import (
    DEFAULT_MAX_POINTS,
    DEFAULT_SCENE_DIR,
    FLOOR_SPEC,
    _path_description,
    _show_selections,
    _show_similarity,
    log,
)


TABLE_SPEC = SemanticSpec(
    label="table",
    positive_texts=("table",),
    negative_texts=("object", "chair", "couch", "bed"),
    scoring="softmax",
    softmax_temp=0.02,
    point_score_threshold=0.20,
    min_point_fraction=0.1,
    min_point_count=2000,
)

CHAIR_SPEC = SemanticSpec(
    label="chair",
    positive_texts=("chair",),
    negative_texts=("object", "floor", "table"),
    scoring="softmax",
    softmax_temp=0.02,
    point_score_threshold=0.40,
    min_point_fraction=0.05,
    min_point_count=100,
)


@dataclass
class Task2Branch:
    path: list[dict[str, float | int | str]]
    table: ObjSelection
    chairs: ObjSelection | None = None


@dataclass
class Task2Leaf:
    path: list[dict[str, float | int | str]]
    chairs: ObjSelection
    floor_points: PointSelection


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


def _fork_table_step(selection: ObjSelection) -> list[Task2Branch]:
    branches: list[Task2Branch] = []
    for table in _singleton_objects(selection):
        branches.append(
            Task2Branch(
                path=[
                    {
                        "step": "table",
                        "instance_id": int(table.instance_ids[0]),
                        "score": float(table.scores[0]),
                    }
                ],
                table=table,
            )
        )
    return branches


def _branch_description(branch: Task2Branch) -> str:
    desc = _path_description(branch.path)
    if branch.chairs is not None:
        chair_ids = ",".join(str(int(x)) for x in branch.chairs.instance_ids.tolist())
        desc += f" chairs=[{chair_ids}]"
    return desc


def intersect_point_selections(scene, selections: list[PointSelection]) -> PointSelection:
    if not selections:
        return scene._make_point_selection(np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32))
    current_ids = selections[0].point_indices.copy()
    current_scores = selections[0].scores.copy()
    for selection in selections[1:]:
        common, idx_a, idx_b = np.intersect1d(
            current_ids,
            selection.point_indices,
            assume_unique=True,
            return_indices=True,
        )
        if common.shape[0] == 0:
            return scene._make_point_selection(np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32))
        current_scores = np.minimum(current_scores[idx_a], selection.scores[idx_b]).astype(np.float32, copy=False)
        current_ids = common.astype(np.int64, copy=False)
    return scene._make_point_selection(current_ids, current_scores)


def solve_task(scene_dir: str | Path = DEFAULT_SCENE_DIR) -> list[Task2Leaf]:
    log(f"loading scene: {scene_dir}")
    log(f"downsampling to at most {DEFAULT_MAX_POINTS} points during load")
    scene = load_scene(scene_dir, max_points=DEFAULT_MAX_POINTS)
    log(f"scene ready: {scene.points.shape[0]} points, {scene.n_instances} instances")
    log("prefetching semantic queries for table, chair, and floor")
    scene.prefetch_semantic_scores(TABLE_SPEC, CHAIR_SPEC, FLOOR_SPEC)
    log("semantic prefetch ready")

    log("step 1/3: finding table candidates")
    table_selection = object_class(TABLE_SPEC)
    branches = _fork_table_step(table_selection)
    _show_similarity(scene, title="Semantic similarity: table", query=TABLE_SPEC)
    log(f"step 1/3 complete: {len(branches)} branch(es)")
    _show_selections(
        scene,
        title="Step 1: table candidates",
        selections=[branch.table for branch in branches],
        descriptions=[_branch_description(branch) for branch in branches],
    )

    log("step 2/3: finding chairs behind each table")
    kept_branches: list[Task2Branch] = []
    for branch_idx, branch in enumerate(branches, start=1):
        log(f"chairs: evaluating branch {branch_idx}/{len(branches)} ({_branch_description(branch)})")
        chairs = behind(
            branch.table,
            CHAIR_SPEC,
            min_gap=0.0,
            max_gap=2.0,
            lateral_margin=1.5,
            vertical_margin=0.8,
            wholeobj_or_points="wholeobj",
            wholeobj_min_point_fraction=0.05,
            wholeobj_min_point_count=100,
        )
        log(f"chairs: branch {branch_idx}/{len(branches)} returned {len(chairs)} chair candidate(s)")
        if chairs:
            branch.chairs = chairs
            kept_branches.append(branch)
    branches = kept_branches
    _show_similarity(scene, title="Semantic similarity: chair", query=CHAIR_SPEC)
    log(f"step 2/3 complete: {len(branches)} branch(es)")
    _show_selections(
        scene,
        title="Step 2: chairs behind tables",
        selections=[branch.chairs for branch in branches],
        descriptions=[_branch_description(branch) for branch in branches],
    )

    leaves: list[Task2Leaf] = []
    log("step 3/3: intersecting per-chair floor regions")
    for branch_idx, branch in enumerate(branches, start=1):
        assert branch.chairs is not None
        chair_regions: list[PointSelection] = []
        log(
            f"floor intersection: evaluating branch {branch_idx}/{len(branches)} "
            f"({_branch_description(branch)}) with {len(branch.chairs)} chair(s)"
        )
        for chair in _singleton_objects(branch.chairs):
            region = close_to(
                chair,
                FLOOR_SPEC,
                distance_threshold=0.45,
                wholeobj_or_points="points",
            )
            log(
                f"floor intersection: chair {int(chair.instance_ids[0])} returned {len(region)} floor point(s)"
            )
            if not region:
                chair_regions = []
                break
            chair_regions.append(region)
        if not chair_regions:
            log(f"floor intersection: branch {branch_idx}/{len(branches)} returned no common region")
            continue
        intersection = intersect_point_selections(scene, chair_regions)
        if intersection:
            leaves.append(Task2Leaf(path=branch.path, chairs=branch.chairs, floor_points=intersection))
            log(
                f"floor intersection: branch {branch_idx}/{len(branches)} returned "
                f"{len(intersection)} common floor point(s) across {len(branch.chairs)} chair(s)"
            )
        else:
            log(f"floor intersection: branch {branch_idx}/{len(branches)} returned no common region")
    _show_similarity(scene, title="Semantic similarity: floor", query=FLOOR_SPEC)
    log(f"step 3/3 complete: {len(leaves)} leaf selection(s)")
    _show_selections(
        scene,
        title="Final: common floor regions for seeing all chairs",
        selections=[leaf.floor_points for leaf in leaves],
        descriptions=[
            f"{_path_description(leaf.path)} chairs=[{','.join(str(int(x)) for x in leaf.chairs.instance_ids.tolist())}]"
            for leaf in leaves
        ],
    )
    return leaves


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve common floor-view regions for: how many chairs behind the table?")
    parser.add_argument("scene_dir", nargs="?", default=str(DEFAULT_SCENE_DIR))
    args = parser.parse_args()
    solve_task(args.scene_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
