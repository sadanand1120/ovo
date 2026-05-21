from __future__ import annotations

import threading
import time
from typing import Any, List

import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


def _resolve_oracle_selected_count_bounds(
    num_gt: int,
    oracle_prune_num_inst_perc_delta: tuple[float, float] | None,
) -> tuple[int, int] | None:
    if oracle_prune_num_inst_perc_delta is None:
        return None
    if len(oracle_prune_num_inst_perc_delta) != 2:
        raise ValueError(
            "oracle_prune_num_inst_perc_delta must be a 2-tuple like (0.8, 1.1)."
        )
    lower_mult = float(oracle_prune_num_inst_perc_delta[0])
    upper_mult = float(oracle_prune_num_inst_perc_delta[1])
    if lower_mult < 0.0 or upper_mult < 0.0:
        raise ValueError("oracle_prune_num_inst_perc_delta values must be non-negative.")
    if lower_mult > upper_mult:
        raise ValueError(
            f"oracle_prune_num_inst_perc_delta lower bound {lower_mult} exceeds upper bound {upper_mult}."
        )
    lower = int(np.ceil(lower_mult * float(num_gt)))
    upper = int(np.floor(upper_mult * float(num_gt)))
    if upper < lower:
        raise ValueError(
            "oracle_prune_num_inst_perc_delta produces an empty selected-count interval "
            f"for num_gt={int(num_gt)}: [{lower}, {upper}]"
        )
    return lower, upper


def get_iou(label_id: int, confusion: np.ndarray) -> tuple[float, float]:
    tp = np.longlong(confusion[label_id, label_id])
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    fp = np.longlong(confusion[:, label_id].sum()) - tp
    denom = float(tp + fp + fn)
    if denom == 0:
        return float("nan"), float("nan")
    iou = tp / denom
    acc = tp / max(float(tp + fn), 1e-6)
    return iou, acc


def iou_acc_from_confmat(
    confmat: np.ndarray,
    num_classes: int,
    ignore: List[int],
    mask_nan: bool = True,
    verbose: bool = False,
    labels: List[str] | None = None,
):
    if verbose:
        print("\n classes \t IoU \t Acc")
        print("----------------------------")
    list_iou, list_acc, list_weight = [], [], []
    for class_idx in range(num_classes):
        if class_idx in ignore:
            continue
        iou, acc = get_iou(class_idx, confmat)
        list_iou.append(iou)
        list_acc.append(acc)
        list_weight.append(confmat[class_idx].sum())
        if verbose and labels is not None:
            print("{0:<14s}: {1:>5.2%}   {2:>6.2%}".format(labels[class_idx], iou, acc))

    iou_values = np.array(list_iou)
    acc_values = np.array(list_acc)
    weights_values = np.array(list_weight)

    if mask_nan:
        iou_valid_mask = ~np.isnan(iou_values)
        acc_valid_mask = ~np.isnan(acc_values)
    else:
        iou_valid_mask = np.ones_like(iou_values, dtype=bool)
        acc_valid_mask = np.ones_like(acc_values, dtype=bool)
    return iou_values, iou_valid_mask, weights_values, acc_values, acc_valid_mask


def average_precision_from_ranked_matches(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    if num_gt <= 0:
        return float("nan")
    if tp.size == 0:
        return 0.0
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(num_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = np.r_[0.0, recall, 1.0]
    precision = np.r_[0.0, precision, 0.0]
    for idx in range(precision.shape[0] - 2, -1, -1):
        precision[idx] = max(precision[idx], precision[idx + 1])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def finalize_instance_labels_and_scores(
    raw_labels: np.ndarray,
    raw_instance_scores: np.ndarray,
    min_component_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(raw_labels, dtype=np.int32).copy()
    raw_instance_scores = np.asarray(raw_instance_scores, dtype=np.float32)
    valid = labels >= 0
    if valid.any() and int(min_component_size) > 1:
        uniq, counts = np.unique(labels[valid], return_counts=True)
        keep = uniq[counts >= int(min_component_size)]
        labels[~np.isin(labels, keep)] = -1
        valid = labels >= 0
    if not valid.any():
        return labels, np.empty((0,), dtype=np.float32)
    unique_labels = np.unique(labels[valid]).astype(np.int64, copy=False)
    if unique_labels[-1] >= raw_instance_scores.shape[0]:
        raise ValueError(
            f"raw_instance_scores has length {raw_instance_scores.shape[0]}, but labels contain id {int(unique_labels[-1])}."
        )
    relabeled_scores = raw_instance_scores[unique_labels].astype(np.float32, copy=False)
    _, relabeled = np.unique(labels[valid], return_inverse=True)
    labels[valid] = relabeled.astype(np.int32, copy=False)
    return labels, relabeled_scores


def compute_instance_ap_dataset(
    entries: list[dict],
    class_ids: np.ndarray,
    iou_thresholds: tuple[float, ...],
    mean_ap_thresholds: tuple[float, ...] | None = None,
) -> tuple[dict, dict]:
    class_ids = np.asarray(class_ids, dtype=np.int32)
    iou_thresholds = tuple(float(th) for th in iou_thresholds)
    if mean_ap_thresholds is None:
        mean_ap_thresholds = iou_thresholds
    else:
        mean_ap_thresholds = tuple(float(th) for th in mean_ap_thresholds)
    missing_mean_thresholds = sorted(set(mean_ap_thresholds) - set(iou_thresholds))
    if missing_mean_thresholds:
        raise ValueError(f"mean_ap_thresholds must be a subset of iou_thresholds; missing={missing_mean_thresholds}")
    if class_ids.size == 0:
        metrics = {"ap": float("nan")}
        diagnostics = {"num_eval_classes": 0, "num_eval_entries": int(len(entries))}
        for th in iou_thresholds:
            metrics[f"ap_{int(round(th * 100)):02d}"] = float("nan")
        return metrics, diagnostics

    ap_by_threshold = {float(th): [] for th in iou_thresholds}
    num_gt_by_class = {}
    num_pred_by_class = {}

    for class_id in class_ids.tolist():
        gt_local_ids = []
        preds = []
        num_gt = 0
        num_pred = 0
        for entry_idx, entry in enumerate(entries):
            gt_ids = np.flatnonzero(np.asarray(entry["gt_class_ids"], dtype=np.int32) == int(class_id))
            pred_ids = np.flatnonzero(np.asarray(entry["pred_class_ids"], dtype=np.int32) == int(class_id))
            gt_local_ids.append(gt_ids.astype(np.int32, copy=False))
            num_gt += int(gt_ids.size)
            num_pred += int(pred_ids.size)
            if pred_ids.size == 0:
                continue
            scores = np.asarray(entry["pred_scores"], dtype=np.float32)[pred_ids]
            for pred_id, score in zip(pred_ids.tolist(), scores.tolist()):
                preds.append((float(score), int(entry_idx), int(pred_id)))
        if num_gt == 0:
            continue

        num_gt_by_class[int(class_id)] = int(num_gt)
        num_pred_by_class[int(class_id)] = int(num_pred)
        preds.sort(key=lambda item: (-item[0], item[1], item[2]))

        for threshold in iou_thresholds:
            matched = {entry_idx: np.zeros(gt_local_ids[entry_idx].shape[0], dtype=bool) for entry_idx in range(len(entries))}
            tp = np.zeros((len(preds),), dtype=np.float32)
            fp = np.zeros((len(preds),), dtype=np.float32)
            for pred_rank, (_, entry_idx, pred_id) in enumerate(preds):
                gt_ids = gt_local_ids[entry_idx]
                if gt_ids.size == 0:
                    fp[pred_rank] = 1.0
                    continue
                iou_col = np.asarray(entries[entry_idx]["iou"][gt_ids, pred_id], dtype=np.float32)
                available = ~matched[entry_idx]
                candidate = np.flatnonzero((iou_col >= float(threshold)) & available)
                if candidate.size == 0:
                    fp[pred_rank] = 1.0
                    continue
                best_local = candidate[np.argmax(iou_col[candidate])]
                matched[entry_idx][best_local] = True
                tp[pred_rank] = 1.0
            ap_by_threshold[float(threshold)].append(average_precision_from_ranked_matches(tp, fp, num_gt))

    metrics = {}
    for threshold in iou_thresholds:
        values = np.asarray(ap_by_threshold[float(threshold)], dtype=np.float64)
        metrics[f"ap_{int(round(threshold * 100)):02d}"] = float(values.mean()) if values.size > 0 else float("nan")
    threshold_values = np.asarray(
        [metrics[f"ap_{int(round(threshold * 100)):02d}"] for threshold in mean_ap_thresholds],
        dtype=np.float64,
    )
    metrics["ap"] = float(np.nanmean(threshold_values)) if threshold_values.size > 0 else float("nan")
    diagnostics = {
        "num_eval_classes": int(len(num_gt_by_class)),
        "num_eval_entries": int(len(entries)),
        "gt_instances_per_class": num_gt_by_class,
        "pred_instances_per_class": num_pred_by_class,
    }
    return metrics, diagnostics


def collect_instance_candidates(
    point_gids: np.ndarray,
    raw_instance_scores: np.ndarray,
    min_component_size: int,
    *,
    allowed_gids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_gids = np.asarray(point_gids, dtype=np.int32)
    raw_instance_scores = np.asarray(raw_instance_scores, dtype=np.float32)
    valid = point_gids >= 0
    if not valid.any():
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    gids, counts = np.unique(point_gids[valid], return_counts=True)
    gids = gids.astype(np.int32, copy=False)
    counts = counts.astype(np.int32, copy=False)
    keep = counts >= int(min_component_size)
    if allowed_gids is not None:
        allowed_lookup = np.zeros((max(raw_instance_scores.shape[0], int(gids.max()) + 1),), dtype=bool)
        for gid in allowed_gids:
            gid = int(gid)
            if 0 <= gid < allowed_lookup.shape[0]:
                allowed_lookup[gid] = True
        keep &= allowed_lookup[gids]
    gids = gids[keep]
    counts = counts[keep]
    if gids.size == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    if int(gids.max()) >= raw_instance_scores.shape[0]:
        raise ValueError(
            f"raw_instance_scores has length {raw_instance_scores.shape[0]}, but candidate gids contain id {int(gids.max())}."
        )
    order = np.argsort(gids, kind="stable")
    gids = gids[order]
    counts = counts[order]
    scores = raw_instance_scores[gids].astype(np.float32, copy=False)
    return gids, scores, counts


def build_instance_iou_from_candidates(
    pred_points: np.ndarray,
    point_gids: np.ndarray,
    pred_gids: np.ndarray,
    gt_points: np.ndarray,
    gt_instance_labels: np.ndarray,
    *,
    transfer_k: int = 5,
    chunk_size: int = 100_000,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pred_points = np.asarray(pred_points, dtype=np.float32)
    point_gids = np.asarray(point_gids, dtype=np.int32)
    pred_gids = np.asarray(pred_gids, dtype=np.int32)
    gt_points = np.asarray(gt_points, dtype=np.float32)
    gt_instance_labels = np.asarray(gt_instance_labels, dtype=np.int32)
    valid_gt = gt_instance_labels >= 0
    gt_ids = np.unique(gt_instance_labels[valid_gt]).astype(np.int32, copy=False)
    if gt_ids.size == 0 or pred_gids.size == 0:
        return (
            np.zeros((gt_ids.shape[0], pred_gids.shape[0]), dtype=np.float32),
            gt_ids,
            {
                "matched_instance_count": 0,
                "assigned_gt_vertices": int(valid_gt.sum()),
                "transfer_k": int(min(int(transfer_k), pred_points.shape[0])) if pred_points.shape[0] > 0 else 0,
                "source_pred_points": int(pred_points.shape[0]),
            },
        )

    gt_points_valid = gt_points[valid_gt]
    gt_labels_valid = gt_instance_labels[valid_gt]
    gt_ids, gt_local = np.unique(gt_labels_valid, return_inverse=True)
    gt_count = np.bincount(gt_local, minlength=gt_ids.shape[0]).astype(np.int64, copy=False)
    k = min(int(transfer_k), int(pred_points.shape[0]))
    if k <= 0:
        return (
            np.zeros((gt_ids.shape[0], pred_gids.shape[0]), dtype=np.float32),
            gt_ids.astype(np.int32, copy=False),
            {
                "matched_instance_count": 0,
                "assigned_gt_vertices": int(valid_gt.sum()),
                "transfer_k": 0,
                "source_pred_points": int(pred_points.shape[0]),
            },
        )

    stage_progress = None
    chunk_progress = None
    try:
        if progress_desc is not None:
            stage_progress = tqdm(total=3, desc=progress_desc, unit="stage", leave=False, dynamic_ncols=True)
            stage_progress.set_postfix_str(
                f"build kd-tree ({int(pred_points.shape[0]):,} pred pts)",
                refresh=True,
            )
        pred_tree = cKDTree(pred_points)
        if stage_progress is not None:
            stage_progress.update()
            stage_progress.set_postfix_str(
                f"knn query ({int(gt_points_valid.shape[0]):,} gt pts, k={int(k)})",
                refresh=True,
            )
        _, knn_idx = pred_tree.query(gt_points_valid, k=k, workers=16)
        knn_idx = np.asarray(knn_idx, dtype=np.int64)
        if k == 1:
            knn_idx = knn_idx[:, None]

        gid_lookup = np.full((max(int(pred_gids.max()) + 1, 1),), -1, dtype=np.int32)
        gid_lookup[pred_gids] = np.arange(pred_gids.shape[0], dtype=np.int32)
        vote_threshold = (k // 2) + 1
        contingency = np.zeros((gt_ids.shape[0], pred_gids.shape[0]), dtype=np.int64)
        pred_count = np.zeros((pred_gids.shape[0],), dtype=np.int64)
        point_gid_slots = int(point_gids.shape[1])
        chunk_size = max(int(chunk_size), 1)
        if stage_progress is not None:
            stage_progress.update()
            stage_progress.set_postfix_str(
                f"vote chunks ({int(gt_points_valid.shape[0]):,} gt pts, chunk={int(chunk_size):,})",
                refresh=True,
            )
            chunk_progress = tqdm(
                range(0, int(gt_points_valid.shape[0]), chunk_size),
                desc=f"{progress_desc}: vote chunks",
                unit="chunk",
                leave=False,
                dynamic_ncols=True,
            )
            iterator = chunk_progress
        else:
            iterator = range(0, int(gt_points_valid.shape[0]), chunk_size)
        for start in iterator:
            end = min(int(start) + chunk_size, int(gt_points_valid.shape[0]))
            chunk_knn_idx = knn_idx[int(start):end]
            chunk_rows = point_gids[chunk_knn_idx]
            vote_counts = np.zeros((int(end - start), pred_gids.shape[0]), dtype=np.uint8)
            for nn_idx in range(k):
                for slot_idx in range(point_gid_slots):
                    gids = chunk_rows[:, nn_idx, slot_idx]
                    valid = (gids >= 0) & (gids < gid_lookup.shape[0])
                    if not valid.any():
                        continue
                    local = gid_lookup[gids[valid]]
                    keep = local >= 0
                    if not keep.any():
                        continue
                    row_idx = np.flatnonzero(valid)[keep]
                    np.add.at(vote_counts, (row_idx, local[keep]), 1)
            assigned = vote_counts >= vote_threshold
            pred_count += assigned.sum(axis=0, dtype=np.int64)
            gt_local_chunk = gt_local[int(start):end]
            for local_gt in np.unique(gt_local_chunk):
                mask = gt_local_chunk == int(local_gt)
                contingency[int(local_gt)] += assigned[mask].sum(axis=0, dtype=np.int64)
        if stage_progress is not None:
            stage_progress.update()
    finally:
        if chunk_progress is not None:
            chunk_progress.close()
        if stage_progress is not None:
            stage_progress.close()

    union = gt_count[:, None] + pred_count[None, :] - contingency
    iou = np.where(union > 0, contingency / union, 0.0).astype(np.float32)
    diagnostics = {
        "matched_instance_count": int(np.count_nonzero(pred_count > 0)),
        "assigned_gt_vertices": int(gt_points_valid.shape[0]),
        "transfer_k": int(k),
        "source_pred_points": int(pred_points.shape[0]),
    }
    return iou, gt_ids.astype(np.int32, copy=False), diagnostics


def _build_oracle_feasible_graph(
    iou_pred_gt: np.ndarray,
    threshold: float,
    *,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    iou_pred_gt = np.asarray(iou_pred_gt, dtype=np.float32)
    num_pred, _num_gt = iou_pred_gt.shape
    stage_progress = None
    eligible_progress = None
    feasible_progress = None
    dominance_progress = None
    try:
        if progress_desc is not None:
            stage_progress = tqdm(total=3, desc=progress_desc, unit="stage", leave=False, dynamic_ncols=True)
            stage_progress.set_postfix_str(
                f"eligible predictions ({int(num_pred):,} preds)",
                refresh=True,
            )
        eligible = []
        pred_iterator = range(num_pred)
        if progress_desc is not None:
            eligible_progress = tqdm(
                pred_iterator,
                total=int(num_pred),
                desc=f"{progress_desc}: eligible predictions",
                unit="pred",
                leave=False,
                dynamic_ncols=True,
            )
            pred_iterator = eligible_progress
        for pred_idx in pred_iterator:
            eligible.append(np.flatnonzero(iou_pred_gt[pred_idx] >= float(threshold)).astype(np.int32))
        num_eligible_predictions = int(sum(1 for gt_idx in eligible if gt_idx.size > 0))
        if stage_progress is not None:
            stage_progress.update()
            stage_progress.set_postfix_str(
                f"build feasible edges ({int(num_eligible_predictions):,} eligible preds)",
                refresh=True,
            )

        feasible: set[tuple[int, int]] = set()
        blockers: dict[tuple[int, int], tuple[int, ...]] = {}
        feasible_iterator = enumerate(eligible)
        if progress_desc is not None:
            feasible_progress = tqdm(
                feasible_iterator,
                total=int(num_pred),
                desc=f"{progress_desc}: feasible edges",
                unit="pred",
                leave=False,
                dynamic_ncols=True,
            )
            feasible_iterator = feasible_progress
        for pred_idx, gt_candidates in feasible_iterator:
            row = iou_pred_gt[pred_idx]
            for gt_idx in gt_candidates.tolist():
                feasible.add((pred_idx, int(gt_idx)))
                pred_blockers = []
                iou_ref = float(row[int(gt_idx)])
                for other_gt in gt_candidates.tolist():
                    if int(other_gt) == int(gt_idx):
                        continue
                    other_iou = float(row[int(other_gt)])
                    if other_iou > iou_ref or (other_iou == iou_ref and int(other_gt) < int(gt_idx)):
                        pred_blockers.append(int(other_gt))
                blockers[(pred_idx, int(gt_idx))] = tuple(pred_blockers)
        if stage_progress is not None:
            stage_progress.update()
            stage_progress.set_postfix_str(
                f"dominance prune ({int(len(feasible)):,} feasible edges)",
                refresh=True,
            )

        changed = True
        dominance_iterations = 0
        while changed:
            dominance_iterations += 1
            changed = False
            to_remove: list[tuple[int, int]] = []
            feasible_edges = list(feasible)
            if dominance_progress is not None:
                dominance_progress.close()
                dominance_progress = None
            edge_iterator = feasible_edges
            if progress_desc is not None:
                dominance_progress = tqdm(
                    feasible_edges,
                    total=int(len(feasible_edges)),
                    desc=f"{progress_desc}: dominance iter {int(dominance_iterations)}",
                    unit="edge",
                    leave=False,
                    dynamic_ncols=True,
                )
                edge_iterator = dominance_progress
            for pred_idx, gt_idx in edge_iterator:
                valid = True
                for blocker_gt in blockers[(pred_idx, gt_idx)]:
                    if not any((earlier_pred, blocker_gt) in feasible for earlier_pred in range(pred_idx)):
                        valid = False
                        break
                if not valid:
                    to_remove.append((pred_idx, gt_idx))
            if to_remove:
                changed = True
                for edge in to_remove:
                    feasible.discard(edge)
            if stage_progress is not None:
                stage_progress.set_postfix_str(
                    f"dominance prune iter={int(dominance_iterations)} edges={int(len(feasible)):,}",
                    refresh=True,
                )

        feasible_by_pred: dict[int, list[int]] = {}
        for pred_idx, gt_idx in sorted(feasible):
            feasible_by_pred.setdefault(int(pred_idx), []).append(int(gt_idx))
        if stage_progress is not None:
            stage_progress.update()
        return {
            "eligible": eligible,
            "num_eligible_predictions": int(num_eligible_predictions),
            "feasible_by_pred": feasible_by_pred,
            "blockers": blockers,
            "num_feasible_edges": int(len(feasible)),
            "dominance_iterations": int(dominance_iterations),
        }
    finally:
        if eligible_progress is not None:
            eligible_progress.close()
        if feasible_progress is not None:
            feasible_progress.close()
        if dominance_progress is not None:
            dominance_progress.close()
        if stage_progress is not None:
            stage_progress.close()


def _solve_single_threshold_oracle_pruning(
    iou_gt_pred: np.ndarray,
    pred_scores: np.ndarray,
    threshold: float,
    *,
    pred_ids: np.ndarray,
    oracle_prune_num_inst_perc_delta: tuple[float, float] | None,
    max_time_s: float | None,
    num_workers: int,
    progress_desc: str | None,
) -> dict[str, Any]:
    from ortools.sat.python import cp_model

    num_gt, num_pred = iou_gt_pred.shape
    order = np.lexsort((pred_ids, -pred_scores))
    ordered_pred_ids = pred_ids[order]
    iou_pred_gt = iou_gt_pred[:, order].T
    selected_count_bounds = _resolve_oracle_selected_count_bounds(
        int(num_gt),
        oracle_prune_num_inst_perc_delta,
    )
    graph = _build_oracle_feasible_graph(
        iou_pred_gt,
        float(threshold),
        progress_desc=progress_desc,
    )
    feasible_by_pred = graph["feasible_by_pred"]
    blockers = graph["blockers"]
    if graph["num_eligible_predictions"] == 0:
        return {
            "selected_pred_ids": np.empty((0,), dtype=np.int32),
            "selected_pred_mask": np.zeros((num_pred,), dtype=bool),
            "matched_gt_indices": np.empty((0,), dtype=np.int32),
            "oracle_status": "no_eligible_predictions",
            "oracle_optimal_tp_count": 0,
            "oracle_threshold": float(threshold),
            "oracle_mode": "single_threshold",
            "oracle_selected_count_bounds": None if selected_count_bounds is None else list(selected_count_bounds),
        }
    if not feasible_by_pred:
        return {
            "selected_pred_ids": np.empty((0,), dtype=np.int32),
            "selected_pred_mask": np.zeros((num_pred,), dtype=bool),
            "matched_gt_indices": np.empty((0,), dtype=np.int32),
            "oracle_status": "no_feasible_edges",
            "oracle_optimal_tp_count": 0,
            "oracle_threshold": float(threshold),
            "oracle_mode": "single_threshold",
            "oracle_selected_count_bounds": None if selected_count_bounds is None else list(selected_count_bounds),
            "oracle_num_predictions": int(num_pred),
            "oracle_num_predictions_eligible": int(graph["num_eligible_predictions"]),
            "oracle_num_gt": int(num_gt),
            "oracle_num_feasible_edges": int(graph["num_feasible_edges"]),
            "oracle_num_dominance_iterations": int(graph["dominance_iterations"]),
        }

    model = cp_model.CpModel()
    y: dict[tuple[int, int], cp_model.IntVar] = {}
    model_progress = None
    solve_progress = None
    solve_status_holder: dict[str, Any] = {}
    try:
        item_iterator = list(feasible_by_pred.items())
        if progress_desc is not None:
            model_progress = tqdm(
                item_iterator,
                total=int(len(item_iterator)),
                desc=f"{progress_desc}: cp-sat model",
                unit="pred",
                leave=False,
                dynamic_ncols=True,
            )
            item_iterator = model_progress
        for pred_idx, gt_list in item_iterator:
            for gt_idx in gt_list:
                y[(pred_idx, gt_idx)] = model.NewBoolVar(f"y_{pred_idx}_{gt_idx}")
        for pred_idx, gt_list in feasible_by_pred.items():
            model.Add(sum(y[(pred_idx, gt_idx)] for gt_idx in gt_list) <= 1)
        for gt_idx in range(num_gt):
            gt_edges = [y[(pred_idx, gt_idx)] for pred_idx, gt_list in feasible_by_pred.items() if int(gt_idx) in gt_list]
            if gt_edges:
                model.Add(sum(gt_edges) <= 1)
        for pred_idx, gt_list in feasible_by_pred.items():
            for gt_idx in gt_list:
                var = y[(pred_idx, gt_idx)]
                for blocker_gt in blockers[(pred_idx, gt_idx)]:
                    blocker_edges = [y[(earlier_pred, blocker_gt)] for earlier_pred in range(pred_idx) if (earlier_pred, blocker_gt) in y]
                    model.Add(var <= sum(blocker_edges))
        if selected_count_bounds is not None:
            lower, upper = selected_count_bounds
            selected_count = sum(y.values())
            model.Add(selected_count >= int(lower))
            model.Add(selected_count <= int(upper))
        model.Maximize(sum(y.values()))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = int(num_workers)
        if max_time_s is not None:
            solver.parameters.max_time_in_seconds = float(max_time_s)

        class _OracleProgressCallback(cp_model.CpSolverSolutionCallback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar

            def on_solution_callback(self):
                solve_status_holder["best_tp"] = int(self.ObjectiveValue())
                if self.progress_bar is not None:
                    self.progress_bar.set_postfix_str(
                        f"best_tp={int(self.ObjectiveValue())}",
                        refresh=True,
                    )

        if progress_desc is not None:
            solve_progress = tqdm(
                total=1,
                desc=f"{progress_desc}: cp-sat solve",
                unit="solve",
                leave=False,
                dynamic_ncols=True,
            )
        callback = _OracleProgressCallback(solve_progress)
        def _run_cp_sat_solve() -> None:
            try:
                solve_status_holder["status"] = solver.Solve(model, callback if solve_progress is not None else None)
            except Exception as exc:  # noqa: BLE001
                solve_status_holder["exception"] = exc

        solve_thread = threading.Thread(target=_run_cp_sat_solve, daemon=True)
        solve_thread.start()
        solve_start_time = time.monotonic()
        if solve_progress is not None:
            while solve_thread.is_alive():
                elapsed_s = time.monotonic() - solve_start_time
                best_tp = solve_status_holder.get("best_tp")
                postfix = f"elapsed={elapsed_s:.1f}s"
                if best_tp is not None:
                    postfix += f" best_tp={int(best_tp)}"
                solve_progress.set_postfix_str(postfix, refresh=True)
                time.sleep(0.5)
        solve_thread.join()
        if "exception" in solve_status_holder:
            raise solve_status_holder["exception"]
        status = int(solve_status_holder["status"])
        if solve_progress is not None:
            elapsed_s = time.monotonic() - solve_start_time
            branches = int(solver.NumBranches())
            conflicts = int(solver.NumConflicts())
            best_bound = float(solver.BestObjectiveBound())
            best_tp = solve_status_holder.get("best_tp")
            postfix = f"elapsed={elapsed_s:.1f}s branches={branches:,} conflicts={conflicts:,}"
            if best_tp is not None:
                postfix += f" best_tp={int(best_tp)}"
            if np.isfinite(best_bound):
                postfix += f" best_bound={best_bound:.1f}"
            solve_progress.set_postfix_str(postfix, refresh=True)
            solve_progress.update()
    finally:
        if model_progress is not None:
            model_progress.close()
        if solve_progress is not None:
            solve_progress.close()

    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "selected_pred_ids": np.empty((0,), dtype=np.int32),
            "selected_pred_mask": np.zeros((num_pred,), dtype=bool),
            "matched_gt_indices": np.empty((0,), dtype=np.int32),
            "oracle_status": status_name,
            "oracle_optimal_tp_count": 0,
            "oracle_threshold": float(threshold),
            "oracle_mode": "single_threshold",
            "oracle_selected_count_bounds": None if selected_count_bounds is None else list(selected_count_bounds),
            "oracle_num_predictions": int(num_pred),
            "oracle_num_predictions_eligible": int(graph["num_eligible_predictions"]),
            "oracle_num_gt": int(num_gt),
            "oracle_num_feasible_edges": int(len(y)),
            "oracle_num_dominance_iterations": int(graph["dominance_iterations"]),
        }

    selected_pred_ids = []
    matched_gt_indices = []
    for (pred_idx, gt_idx), var in sorted(y.items()):
        if solver.Value(var) != 1:
            continue
        selected_pred_ids.append(int(ordered_pred_ids[pred_idx]))
        matched_gt_indices.append(int(gt_idx))
    selected_pred_ids = np.asarray(selected_pred_ids, dtype=np.int32)
    matched_gt_indices = np.asarray(matched_gt_indices, dtype=np.int32)
    if selected_count_bounds is not None:
        lower, upper = selected_count_bounds
        if selected_pred_ids.shape[0] > int(upper):
            return {
                "selected_pred_ids": np.empty((0,), dtype=np.int32),
                "selected_pred_mask": np.zeros((num_pred,), dtype=bool),
                "matched_gt_indices": np.empty((0,), dtype=np.int32),
                "oracle_status": "no_feasible_selected_count",
                "oracle_optimal": True,
                "oracle_optimal_tp_count": 0,
                "oracle_threshold": float(threshold),
                "oracle_mode": "single_threshold",
                "oracle_selected_count_bounds": list(selected_count_bounds),
                "oracle_num_predictions": int(num_pred),
                "oracle_num_predictions_eligible": int(graph["num_eligible_predictions"]),
                "oracle_num_gt": int(num_gt),
                "oracle_num_feasible_edges": int(len(y)),
                "oracle_num_dominance_iterations": int(graph["dominance_iterations"]),
            }
        filler_needed = max(0, int(lower) - int(selected_pred_ids.shape[0]))
        if filler_needed > 0:
            selected_id_set = {int(pred_id) for pred_id in selected_pred_ids.tolist()}
            filler_pred_ids = np.asarray(
                [
                    int(pred_id)
                    for pred_id in pred_ids[np.lexsort((pred_ids, -pred_scores))].tolist()
                    if int(pred_id) not in selected_id_set
                ][-filler_needed:],
                dtype=np.int32,
            )
            if filler_pred_ids.shape[0] < filler_needed:
                return {
                    "selected_pred_ids": np.empty((0,), dtype=np.int32),
                    "selected_pred_mask": np.zeros((num_pred,), dtype=bool),
                    "matched_gt_indices": np.empty((0,), dtype=np.int32),
                    "oracle_status": "no_feasible_selected_count",
                    "oracle_optimal": True,
                    "oracle_optimal_tp_count": 0,
                    "oracle_threshold": float(threshold),
                    "oracle_mode": "single_threshold",
                    "oracle_selected_count_bounds": list(selected_count_bounds),
                    "oracle_num_predictions": int(num_pred),
                    "oracle_num_predictions_eligible": int(graph["num_eligible_predictions"]),
                    "oracle_num_gt": int(num_gt),
                    "oracle_num_feasible_edges": int(len(y)),
                    "oracle_num_dominance_iterations": int(graph["dominance_iterations"]),
                }
            selected_pred_ids = np.concatenate([selected_pred_ids, filler_pred_ids], axis=0)
    selected_mask = np.zeros((num_pred,), dtype=bool)
    if selected_pred_ids.size > 0:
        pred_id_to_local = {int(pred_id): idx for idx, pred_id in enumerate(pred_ids.tolist())}
        selected_mask[[pred_id_to_local[int(pred_id)] for pred_id in selected_pred_ids.tolist()]] = True
    return {
        "selected_pred_ids": selected_pred_ids,
        "selected_pred_mask": selected_mask,
        "matched_gt_indices": matched_gt_indices,
        "oracle_status": status_name,
        "oracle_optimal": bool(status == cp_model.OPTIMAL),
        "oracle_optimal_tp_count": int(matched_gt_indices.shape[0]),
        "oracle_threshold": float(threshold),
        "oracle_mode": "single_threshold",
        "oracle_selected_count_bounds": None if selected_count_bounds is None else list(selected_count_bounds),
        "oracle_num_predictions": int(num_pred),
        "oracle_num_predictions_eligible": int(graph["num_eligible_predictions"]),
        "oracle_num_gt": int(num_gt),
        "oracle_num_feasible_edges": int(len(y)),
        "oracle_num_dominance_iterations": int(graph["dominance_iterations"]),
    }


def _solve_exact_mean_ap_oracle_pruning(
    iou_gt_pred: np.ndarray,
    pred_scores: np.ndarray,
    *,
    pred_ids: np.ndarray,
    oracle_prune_num_inst_perc_delta: tuple[float, float] | None,
    max_time_s: float | None,
    num_workers: int,
    progress_desc: str | None,
) -> dict[str, Any]:
    thresholds = tuple(float(x) for x in np.arange(0.50, 1.00, 0.05))
    num_gt, num_pred = iou_gt_pred.shape
    selected_count_bounds = _resolve_oracle_selected_count_bounds(
        int(num_gt),
        oracle_prune_num_inst_perc_delta,
    )
    order = np.lexsort((pred_ids, -pred_scores))
    ordered_pred_ids_all = pred_ids[order]
    iou_pred_gt_all = iou_gt_pred[:, order].T
    union_keep = np.any(iou_pred_gt_all >= float(thresholds[0]), axis=1)
    ordered_pred_ids = ordered_pred_ids_all[union_keep]
    iou_pred_gt = iou_pred_gt_all[union_keep]
    num_reduced_pred = int(iou_pred_gt.shape[0])
    ineligible_positions = np.flatnonzero(~union_keep).astype(np.int32, copy=False)
    eligible_positions = np.flatnonzero(union_keep).astype(np.int32, copy=False)
    if num_reduced_pred == 0:
        lower_count = 0 if selected_count_bounds is None else int(selected_count_bounds[0])
        if lower_count > 0:
            if int(ineligible_positions.shape[0]) < lower_count:
                return {
                    "selected_pred_ids": np.empty((0,), dtype=np.int32),
                    "selected_pred_mask": np.zeros((num_pred,), dtype=bool),
                    "matched_gt_indices": np.empty((0,), dtype=np.int32),
                    "oracle_status": "no_feasible_selected_count",
                    "oracle_optimal_tp_count": 0,
                    "oracle_threshold": None,
                    "oracle_mode": "mean_ap_exact",
                    "oracle_selected_count_bounds": None if selected_count_bounds is None else list(selected_count_bounds),
                    "oracle_num_predictions": int(num_pred),
                    "oracle_num_predictions_eligible": 0,
                    "oracle_num_gt": int(num_gt),
                }
            filler_positions = ineligible_positions[-lower_count:]
            selected_pred_ids = ordered_pred_ids_all[filler_positions].astype(np.int32, copy=False)
            selected_mask = np.zeros((num_pred,), dtype=bool)
            pred_id_to_local = {int(pred_id): idx for idx, pred_id in enumerate(pred_ids.tolist())}
            selected_mask[[pred_id_to_local[int(pred_id)] for pred_id in selected_pred_ids.tolist()]] = True
            selected_metrics, _ = compute_instance_metrics_from_iou(
                iou_gt_pred[:, selected_mask],
                pred_scores[selected_mask],
            )
            return {
                "selected_pred_ids": selected_pred_ids,
                "selected_pred_mask": selected_mask,
                "matched_gt_indices": np.empty((0,), dtype=np.int32),
                "oracle_status": "OPTIMAL",
                "oracle_optimal": True,
                "oracle_optimal_tp_count": 0,
                "oracle_threshold": None,
                "oracle_mode": "mean_ap_exact",
                "oracle_selected_count_bounds": None if selected_count_bounds is None else list(selected_count_bounds),
                "oracle_num_predictions": int(num_pred),
                "oracle_num_predictions_eligible": 0,
                "oracle_num_gt": int(num_gt),
                "oracle_exact_mean_ap": float(selected_metrics["ap"]),
                "oracle_exact_mean_ap_25": float(selected_metrics["ap_25"]),
                "oracle_exact_mean_ap_50": float(selected_metrics["ap_50"]),
                "oracle_threshold_tp_counts": {f"{threshold_value:.2f}": 0 for threshold_value in thresholds},
                "oracle_num_feasible_edges_by_threshold": {f"{threshold_value:.2f}": 0 for threshold_value in thresholds},
            }
        return {
            "selected_pred_ids": np.empty((0,), dtype=np.int32),
            "selected_pred_mask": np.zeros((num_pred,), dtype=bool),
            "matched_gt_indices": np.empty((0,), dtype=np.int32),
            "oracle_status": "no_eligible_predictions",
            "oracle_optimal_tp_count": 0,
            "oracle_threshold": None,
            "oracle_mode": "mean_ap_exact",
            "oracle_selected_count_bounds": None if selected_count_bounds is None else list(selected_count_bounds),
            "oracle_num_predictions": int(num_pred),
            "oracle_num_predictions_eligible": 0,
            "oracle_num_gt": int(num_gt),
        }
    max_exact_pred = 20
    if num_reduced_pred > max_exact_pred:
        raise RuntimeError(
            f"Exact mean-AP oracle pruning currently supports at most {max_exact_pred} eligible predictions; "
            f"got {num_reduced_pred}."
        )

    candidate_lists_by_threshold: list[list[tuple[int, ...]]] = []
    feasible_edges_by_threshold: dict[str, int] = {}
    for threshold_value in thresholds:
        threshold_candidates: list[tuple[int, ...]] = []
        feasible_edges = 0
        for pred_idx in range(num_reduced_pred):
            row = iou_pred_gt[pred_idx]
            candidate = np.flatnonzero(row >= float(threshold_value))
            if candidate.size > 0:
                order_idx = np.lexsort((candidate, -row[candidate]))
                ordered_gt = tuple(int(candidate[idx]) for idx in order_idx.tolist())
            else:
                ordered_gt = ()
            threshold_candidates.append(ordered_gt)
            feasible_edges += len(ordered_gt)
        candidate_lists_by_threshold.append(threshold_candidates)
        feasible_edges_by_threshold[f"{threshold_value:.2f}"] = int(feasible_edges)

    total_masks = (1 << num_reduced_pred) - 1
    solve_progress = None
    best_mask = 0
    best_filler_count = 0
    best_mean_ap = -1.0
    best_tp_counts = {f"{threshold_value:.2f}": 0 for threshold_value in thresholds}
    lower_count = 1 if selected_count_bounds is None else int(selected_count_bounds[0])
    upper_count = int(num_pred) if selected_count_bounds is None else int(selected_count_bounds[1])
    if upper_count < 1 or lower_count > int(num_pred):
        return {
            "selected_pred_ids": np.empty((0,), dtype=np.int32),
            "selected_pred_mask": np.zeros((num_pred,), dtype=bool),
            "matched_gt_indices": np.empty((0,), dtype=np.int32),
            "oracle_status": "no_feasible_selected_count",
            "oracle_optimal": True,
            "oracle_optimal_tp_count": 0,
            "oracle_threshold": None,
            "oracle_mode": "mean_ap_exact",
            "oracle_selected_count_bounds": None if selected_count_bounds is None else list(selected_count_bounds),
            "oracle_num_predictions": int(num_pred),
            "oracle_num_predictions_eligible": int(num_reduced_pred),
            "oracle_num_gt": int(num_gt),
            "oracle_num_feasible_edges_by_threshold": feasible_edges_by_threshold,
        }
    lower_count = max(1, lower_count)
    upper_count = min(int(num_pred), upper_count)
    num_ineligible = int(ineligible_positions.shape[0])
    start_time = time.monotonic()
    try:
        if progress_desc is not None:
            solve_progress = tqdm(
                total=total_masks,
                desc=f"{progress_desc}: exact exhaustive solve",
                unit="subset",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            )

        progress_batch = 0
        for mask in range(1, total_masks + 1):
            selected_count = int(mask.bit_count())
            if selected_count > upper_count:
                if solve_progress is not None:
                    progress_batch += 1
                    if progress_batch >= 256 or mask == total_masks:
                        solve_progress.update(progress_batch)
                        progress_batch = 0
                        elapsed_s = time.monotonic() - start_time
                        solve_progress.set_postfix_str(
                            f"elapsed={elapsed_s:.1f}s best_ap={best_mean_ap:.4f}",
                            refresh=True,
                        )
                continue
            filler_count = max(0, lower_count - selected_count)
            if filler_count > num_ineligible:
                if solve_progress is not None:
                    progress_batch += 1
                    if progress_batch >= 256 or mask == total_masks:
                        solve_progress.update(progress_batch)
                        progress_batch = 0
                        elapsed_s = time.monotonic() - start_time
                        solve_progress.set_postfix_str(
                            f"elapsed={elapsed_s:.1f}s best_ap={best_mean_ap:.4f}",
                            refresh=True,
                        )
                continue
            selected_eligible_mask = np.array([(mask >> pred_idx) & 1 for pred_idx in range(num_reduced_pred)], dtype=bool)
            selected_eligible_positions = eligible_positions[selected_eligible_mask]
            selected_eligible_pred_indices = np.flatnonzero(selected_eligible_mask).astype(np.int32, copy=False)
            selected_filler_positions = (
                ineligible_positions[-filler_count:] if filler_count > 0 else np.empty((0,), dtype=np.int32)
            )
            threshold_aps: list[float] = []
            threshold_tp_counts: dict[str, int] = {}
            for threshold_idx, threshold_value in enumerate(thresholds):
                assigned_gt_bits = 0
                tp_sequence: list[int] = []
                tp_count = 0
                eligible_ptr = 0
                filler_ptr = 0
                while eligible_ptr < selected_eligible_positions.shape[0] or filler_ptr < selected_filler_positions.shape[0]:
                    take_eligible = filler_ptr >= selected_filler_positions.shape[0]
                    if not take_eligible and eligible_ptr < selected_eligible_positions.shape[0]:
                        take_eligible = int(selected_eligible_positions[eligible_ptr]) < int(selected_filler_positions[filler_ptr])
                    if take_eligible:
                        matched = 0
                        reduced_pred_idx = int(selected_eligible_pred_indices[eligible_ptr])
                        for gt_idx in candidate_lists_by_threshold[threshold_idx][reduced_pred_idx]:
                            gt_bit = 1 << int(gt_idx)
                            if assigned_gt_bits & gt_bit:
                                continue
                            assigned_gt_bits |= gt_bit
                            matched = 1
                            tp_count += 1
                            break
                        tp_sequence.append(matched)
                        eligible_ptr += 1
                    else:
                        tp_sequence.append(0)
                        filler_ptr += 1
                fp_sequence = [1 - tp for tp in tp_sequence]
                threshold_aps.append(
                    average_precision_from_ranked_matches(
                        np.asarray(tp_sequence, dtype=np.float32),
                        np.asarray(fp_sequence, dtype=np.float32),
                        int(num_gt),
                    )
                )
                threshold_tp_counts[f"{threshold_value:.2f}"] = int(tp_count)
            mean_ap = float(np.mean(threshold_aps))
            if mean_ap > best_mean_ap:
                best_mean_ap = mean_ap
                best_mask = int(mask)
                best_filler_count = int(filler_count)
                best_tp_counts = threshold_tp_counts
            if solve_progress is not None:
                progress_batch += 1
                if progress_batch >= 256 or mask == total_masks:
                    solve_progress.update(progress_batch)
                    progress_batch = 0
                    elapsed_s = time.monotonic() - start_time
                    solve_progress.set_postfix_str(
                        f"elapsed={elapsed_s:.1f}s best_ap={best_mean_ap:.4f}",
                        refresh=True,
                    )
    finally:
        if solve_progress is not None:
            solve_progress.close()

    selected_sorted_mask = np.array([(best_mask >> pred_idx) & 1 for pred_idx in range(num_reduced_pred)], dtype=bool)
    selected_pred_ids = ordered_pred_ids[selected_sorted_mask].astype(np.int32, copy=False)
    if best_filler_count > 0:
        filler_pred_ids = ordered_pred_ids_all[ineligible_positions[-best_filler_count:]].astype(np.int32, copy=False)
        selected_pred_ids = np.concatenate([selected_pred_ids, filler_pred_ids], axis=0)
    selected_mask = np.zeros((num_pred,), dtype=bool)
    if selected_pred_ids.size > 0:
        pred_id_to_local = {int(pred_id): idx for idx, pred_id in enumerate(pred_ids.tolist())}
        selected_mask[[pred_id_to_local[int(pred_id)] for pred_id in selected_pred_ids.tolist()]] = True
    selected_metrics, _ = compute_instance_metrics_from_iou(
        iou_gt_pred[:, selected_mask],
        pred_scores[selected_mask],
    )
    return {
        "selected_pred_ids": selected_pred_ids,
        "selected_pred_mask": selected_mask,
        "matched_gt_indices": np.empty((0,), dtype=np.int32),
        "oracle_status": "OPTIMAL",
        "oracle_optimal": True,
        "oracle_optimal_tp_count": int(sum(best_tp_counts.values())),
        "oracle_threshold": None,
        "oracle_mode": "mean_ap_exact",
        "oracle_num_predictions": int(num_pred),
        "oracle_num_predictions_eligible": int(num_reduced_pred),
        "oracle_num_gt": int(num_gt),
        "oracle_exact_mean_ap": float(selected_metrics["ap"]),
        "oracle_exact_mean_ap_25": float(selected_metrics["ap_25"]),
        "oracle_exact_mean_ap_50": float(selected_metrics["ap_50"]),
        "oracle_threshold_tp_counts": best_tp_counts,
        "oracle_num_feasible_edges_by_threshold": feasible_edges_by_threshold,
        "oracle_selected_count_bounds": None if selected_count_bounds is None else list(selected_count_bounds),
    }


def solve_oracle_pruning_for_ap(
    iou_gt_pred: np.ndarray,
    pred_scores: np.ndarray,
    threshold: float | None,
    *,
    pred_ids: np.ndarray | None = None,
    oracle_prune_num_inst_perc_delta: tuple[float, float] | None = None,
    max_time_s: float | None = None,
    num_workers: int = 16,
    progress_desc: str | None = None,
) -> dict[str, Any]:
    iou_gt_pred = np.asarray(iou_gt_pred, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    if iou_gt_pred.ndim != 2:
        raise ValueError(f"Expected iou_gt_pred to have shape [num_gt, num_pred], got {iou_gt_pred.shape}.")
    num_gt, num_pred = iou_gt_pred.shape
    if pred_scores.shape != (num_pred,):
        raise ValueError(f"pred_scores shape mismatch: expected {(num_pred,)}, got {pred_scores.shape}.")
    if pred_ids is None:
        pred_ids = np.arange(num_pred, dtype=np.int32)
    else:
        pred_ids = np.asarray(pred_ids, dtype=np.int32)
        if pred_ids.shape != (num_pred,):
            raise ValueError(f"pred_ids shape mismatch: expected {(num_pred,)}, got {pred_ids.shape}.")
    if num_gt == 0 or num_pred == 0:
        return {
            "selected_pred_ids": np.empty((0,), dtype=np.int32),
            "selected_pred_mask": np.zeros((num_pred,), dtype=bool),
            "matched_gt_indices": np.empty((0,), dtype=np.int32),
            "oracle_status": "trivial",
            "oracle_optimal_tp_count": 0,
            "oracle_threshold": None if threshold is None else float(threshold),
            "oracle_mode": "mean_ap_exact" if threshold is None else "single_threshold",
            "oracle_selected_count_bounds": None
            if oracle_prune_num_inst_perc_delta is None
            else [int(v) for v in _resolve_oracle_selected_count_bounds(int(num_gt), oracle_prune_num_inst_perc_delta)],
        }
    if threshold is None:
        return _solve_exact_mean_ap_oracle_pruning(
            iou_gt_pred,
            pred_scores,
            pred_ids=pred_ids,
            oracle_prune_num_inst_perc_delta=oracle_prune_num_inst_perc_delta,
            max_time_s=max_time_s,
            num_workers=num_workers,
            progress_desc=progress_desc,
        )
    return _solve_single_threshold_oracle_pruning(
        iou_gt_pred,
        pred_scores,
        float(threshold),
        pred_ids=pred_ids,
        oracle_prune_num_inst_perc_delta=oracle_prune_num_inst_perc_delta,
        max_time_s=max_time_s,
        num_workers=num_workers,
        progress_desc=progress_desc,
    )


def compute_instance_metrics_from_iou(
    iou_gt_pred: np.ndarray,
    pred_scores: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    iou_gt_pred = np.asarray(iou_gt_pred, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    num_gt, num_pred = iou_gt_pred.shape
    if pred_scores.shape != (num_pred,):
        raise ValueError(f"pred_scores shape mismatch: expected {(num_pred,)}, got {pred_scores.shape}.")
    if num_gt == 0:
        metrics = {"ap": float("nan")}
        for threshold in (0.25,) + tuple(float(x) for x in np.arange(0.50, 1.00, 0.05)):
            metrics[f"ap_{int(round(threshold * 100)):02d}"] = float("nan")
        diagnostics = {"gt_instance_count": 0, "pred_instance_count": int(num_pred)}
        return metrics, diagnostics

    class_ids = np.array([0], dtype=np.int32)
    gt_class_ids = np.zeros((num_gt,), dtype=np.int32)
    pred_class_ids = np.zeros((num_pred,), dtype=np.int32)
    report_thresholds = (0.25,) + tuple(float(x) for x in np.arange(0.50, 1.00, 0.05))
    mean_thresholds = tuple(float(x) for x in np.arange(0.50, 1.00, 0.05))
    metrics, ap_diag = compute_instance_ap_dataset(
        entries=[
            {
                "iou": iou_gt_pred,
                "gt_class_ids": gt_class_ids,
                "pred_class_ids": pred_class_ids,
                "pred_scores": pred_scores,
            }
        ],
        class_ids=class_ids,
        iou_thresholds=report_thresholds,
        mean_ap_thresholds=mean_thresholds,
    )
    diagnostics = {
        "gt_instance_count": int(num_gt),
        "pred_instance_count": int(num_pred),
        "ignored_gt_instance_count": 0,
        "instance_metric_mode": "class_agnostic_ap",
        "instance_score_source": "seed_support_ratio",
        **ap_diag,
    }
    return metrics, diagnostics
