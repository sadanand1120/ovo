from __future__ import annotations

from typing import List

import numpy as np


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


def compute_instance_ap_dataset(
    entries: list[dict],
    class_ids: np.ndarray,
    iou_thresholds: tuple[float, ...],
) -> tuple[dict, dict]:
    class_ids = np.asarray(class_ids, dtype=np.int32)
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
    threshold_values = np.asarray(list(metrics.values()), dtype=np.float64)
    metrics["ap"] = float(np.nanmean(threshold_values)) if threshold_values.size > 0 else float("nan")
    diagnostics = {
        "num_eval_classes": int(len(num_gt_by_class)),
        "num_eval_entries": int(len(entries)),
        "gt_instances_per_class": num_gt_by_class,
        "pred_instances_per_class": num_pred_by_class,
    }
    return metrics, diagnostics
