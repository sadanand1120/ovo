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
