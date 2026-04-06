from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
EVAL_CONFIGS = {
    "scannet": ROOT / "configs" / "scannet_eval.yaml",
    "replica": ROOT / "configs" / "replica_eval.yaml",
}


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def process_txt(path: Path) -> np.ndarray:
    return np.array(path.read_text().splitlines(), dtype=np.int64)


def evaluate_scan(pr_file: Path, gt_file: Path, confusion: np.ndarray, map_gt_ids: Optional[Dict[int, int]], ignore: List[int]) -> None:
    pr_ids = process_txt(pr_file)
    gt_ids = process_txt(gt_file)
    if map_gt_ids is not None:
        remap = dict(map_gt_ids)
        for label in np.unique(gt_ids):
            remap.setdefault(int(label), -1)
        gt_ids = np.vectorize(remap.get)(gt_ids)
    for gt_val, pr_val in zip(gt_ids, pr_ids):
        if gt_val in ignore:
            continue
        confusion[gt_val][pr_val] += 1


def get_iou_acc(label_id: int, confusion: np.ndarray) -> Tuple[float, float]:
    tp = np.longlong(confusion[label_id, label_id])
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    fp = np.longlong(confusion[:, label_id].sum()) - tp
    denom = float(tp + fp + fn)
    if denom == 0:
        return float("nan"), float("nan")
    return tp / denom, tp / max(float(tp + fn), 1e-6)


def iou_acc_from_confmat(confmat: np.ndarray, num_classes: int, ignore: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iou_values, acc_values, weights = [], [], []
    for idx in range(num_classes):
        if idx in ignore:
            continue
        iou, acc = get_iou_acc(idx, confmat)
        iou_values.append(iou)
        acc_values.append(acc)
        weights.append(confmat[idx].sum())
    iou_values = np.array(iou_values)
    acc_values = np.array(acc_values)
    weights = np.array(weights)
    return iou_values, ~np.isnan(iou_values), weights, acc_values, ~np.isnan(acc_values)


def infer_dataset(run_path: Path) -> str:
    dataset = run_path.parent.name.lower()
    if dataset not in EVAL_CONFIGS:
        raise ValueError(f"Could not infer dataset from {run_path}. Expected parent dir to be ScanNet or Replica.")
    return dataset


def compute_metrics(confusion: np.ndarray, dataset_info: dict) -> dict:
    ignore = dataset_info.get("ignore", []).copy()
    iou_vals, iou_mask, weights, acc_vals, acc_mask = iou_acc_from_confmat(confusion, dataset_info["num_classes"], ignore)
    return {
        "mIoU": float(np.mean(iou_vals[iou_mask])),
        "mAcc": float(np.mean(acc_vals[acc_mask])),
        "f-mIoU": float(np.sum(iou_vals[iou_mask] * weights[iou_mask]) / weights[iou_mask].sum()),
        "f-mAcc": float(np.sum(acc_vals[acc_mask] * weights[acc_mask]) / weights[acc_mask].sum()),
    }


def read_avg_fps(scene_dir: Path) -> float:
    fps_path = scene_dir / "logger" / "avg_fps.log"
    return float(fps_path.read_text().strip()) if fps_path.exists() else float("nan")


def main() -> None:
    parser = ArgumentParser(description="Summarize a completed OVO run.")
    parser.add_argument("run_path", type=Path, help="Run directory, e.g. data/output/ScanNet/scannet_hvs_siglip_vanilla.preserved")
    args = parser.parse_args()

    run_path = args.run_path.resolve()
    dataset = infer_dataset(run_path)
    dataset_info = load_config(EVAL_CONFIGS[dataset])
    gt_root = ROOT / "data" / "input" / ("ScanNet" if dataset == "scannet" else "Replica") / "semantic_gt"

    pred_root = run_path / dataset_info["dataset"]
    scene_files = sorted(p for p in pred_root.glob("*.txt") if p.name != "statistics.txt")
    scenes = [p.stem for p in scene_files]
    if not scenes:
        raise FileNotFoundError(f"No prediction txt files found under {pred_root}")

    per_scene = []
    confusions = []
    for scene in scenes:
        confusion = np.zeros((dataset_info["num_classes"], dataset_info["num_classes"]), dtype=np.ulonglong)
        evaluate_scan(
            pred_root / f"{scene}.txt",
            gt_root / f"{scene}.txt",
            confusion,
            dataset_info.get("map_to_reduced"),
            dataset_info.get("ignore", []).copy(),
        )
        confusions.append(confusion)
        metrics = compute_metrics(confusion, dataset_info)
        metrics["FPS"] = read_avg_fps(run_path / scene)
        metrics["scene"] = scene
        per_scene.append(metrics)

    aggregate = compute_metrics(np.sum(confusions, axis=0), dataset_info)
    fps_values = [item["FPS"] for item in per_scene if not np.isnan(item["FPS"])]
    aggregate["FPS"] = float(np.mean(fps_values)) if fps_values else float("nan")

    print("Aggregate")
    print(
        f"mIoU {aggregate['mIoU']*100:.2f} | "
        f"mAcc {aggregate['mAcc']*100:.2f} | "
        f"f-mIoU {aggregate['f-mIoU']*100:.2f} | "
        f"f-mAcc {aggregate['f-mAcc']*100:.2f} | "
        f"FPS {aggregate['FPS']:.3f}"
    )
    print()
    print("Per scene")
    for item in per_scene:
        print(
            f"{item['scene']}: "
            f"mIoU {item['mIoU']*100:.2f} | "
            f"mAcc {item['mAcc']*100:.2f} | "
            f"f-mIoU {item['f-mIoU']*100:.2f} | "
            f"f-mAcc {item['f-mAcc']*100:.2f} | "
            f"FPS {item['FPS']:.3f}"
        )


if __name__ == "__main__":
    main()
