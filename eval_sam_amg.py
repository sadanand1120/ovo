import argparse
import json
from pathlib import Path

import cv2  # Load before torch in this env.
import numpy as np
import torch
from tqdm.auto import tqdm

from map_runtime.metrics_utils import compute_instance_ap_dataset
from map_runtime.sam_masks import GTInstanceMaskExtractor, SAM_AMG_LEVELS, SAMAutomaticMaskConfig, SAMMaskExtractor, mask_score
from map_runtime.sam2_tracking import SAM2_MODE
from map_runtime.scene import INPUT_DIR, canonical_dataset_name


DEFAULT_AMG_CONFIG = SAMAutomaticMaskConfig()
MASK_AP_THRESHOLDS = tuple(np.arange(0.5, 1.0, 0.05).tolist())


def frame_stem_to_id(stem: str) -> int:
    if stem.startswith("frame"):
        return int(stem[len("frame") :])
    return int(stem)


def build_frame_lookup(
    scene_dir: Path,
    dataset_name: str,
    frame_samples: int | None,
    frame_sample_seed: int,
) -> dict[int, Path]:
    if dataset_name == "ScanNet":
        candidates = [
            path
            for path in (scene_dir / "color").iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    else:
        candidates = list((scene_dir / "results").glob("frame*.jpg")) + list((scene_dir / "results").glob("frame*.png"))
    frame_lookup = {frame_stem_to_id(path.stem): path for path in candidates}
    if frame_samples is not None:
        frame_ids = np.asarray(sorted(frame_lookup.keys()), dtype=np.int32)
        if frame_samples < frame_ids.size:
            rng = np.random.default_rng(frame_sample_seed)
            keep_ids = np.sort(rng.choice(frame_ids, size=frame_samples, replace=False))
            frame_lookup = {int(frame_id): frame_lookup[int(frame_id)] for frame_id in keep_ids.tolist()}
    return dict(sorted(frame_lookup.items()))


def load_color_frame(frame_path: Path) -> np.ndarray:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame_path)
    return image


def build_amg_config(args: argparse.Namespace) -> SAMAutomaticMaskConfig:
    return SAMAutomaticMaskConfig(
        sort_mode=args.sam_sort_mode,
        min_mask_area_perc=args.sam_min_mask_area_perc,
        points_per_side=args.sam_points_per_side,
        points_per_batch=args.sam_points_per_batch,
        pred_iou_thresh=args.sam_pred_iou_thresh,
        stability_score_thresh=args.sam_stability_score_thresh,
        stability_score_offset=args.sam_stability_score_offset,
        mask_threshold=args.sam_mask_threshold,
        box_nms_thresh=args.sam_box_nms_thresh,
        crop_n_layers=args.sam_crop_n_layers,
        crop_nms_thresh=args.sam_crop_nms_thresh,
        crop_overlap_ratio=args.sam_crop_overlap_ratio,
        crop_n_points_downscale_factor=args.sam_crop_n_points_downscale_factor,
        min_mask_region_area=args.sam_min_mask_region_area,
        use_m2m=bool(args.sam_use_m2m),
        multimask_output=not bool(args.sam_disable_multimask_output),
    )


def labels_to_instance_masks(labels: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    masks = []
    scores = []
    for label in np.unique(labels).tolist():
        if label < 0:
            continue
        mask = labels == int(label)
        if not mask.any():
            continue
        masks.append(mask)
        scores.append(1.0)
    return masks, np.asarray(scores, dtype=np.float32)


def sam_masks_with_scores(
    pred_extractor: SAMMaskExtractor,
    image_rgb: np.ndarray,
    min_mask_area_perc: float,
    sort_mode: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    height, width = image_rgb.shape[:2]
    min_mask_area = float(min_mask_area_perc) * height * width
    raw_masks = pred_extractor.mask_generator.generate(image_rgb)
    masks = []
    scores = []
    for mask in raw_masks:
        if float(mask.get("area", 0.0)) < min_mask_area:
            continue
        masks.append(np.asarray(mask["segmentation"], dtype=bool))
        scores.append(float(mask_score(mask, sort_mode)))
    return masks, np.asarray(scores, dtype=np.float32)


def build_mask_iou_matrix(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> np.ndarray:
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return np.zeros((len(gt_masks), len(pred_masks)), dtype=np.float32)
    gt_flat = np.stack([mask.reshape(-1) for mask in gt_masks], axis=0).astype(bool, copy=False)
    pred_flat = np.stack([mask.reshape(-1) for mask in pred_masks], axis=0).astype(bool, copy=False)
    intersection = (gt_flat[:, None, :] & pred_flat[None, :, :]).sum(axis=2, dtype=np.int64)
    gt_area = gt_flat.sum(axis=1, dtype=np.int64)
    pred_area = pred_flat.sum(axis=1, dtype=np.int64)
    union = gt_area[:, None] + pred_area[None, :] - intersection
    iou = np.zeros((gt_flat.shape[0], pred_flat.shape[0]), dtype=np.float32)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    return iou


def evaluate_scene(args: argparse.Namespace) -> dict:
    scene_dir = INPUT_DIR / canonical_dataset_name(args.dataset_name) / args.scene_name
    if not scene_dir.exists():
        raise FileNotFoundError(scene_dir)

    frame_lookup = build_frame_lookup(scene_dir, args.dataset_name, args.frame_samples, args.frame_sample_seed)
    if not frame_lookup:
        raise RuntimeError(f"No frames found under {scene_dir}")

    gt_extractor = GTInstanceMaskExtractor(args.dataset_name, args.scene_name)
    pred_extractor = SAMMaskExtractor(
        args.device,
        args.sam_model_level_inst,
        amg_config=build_amg_config(args),
        sam2_mode=args.sam2_mode,
        sam2_hydra_overrides=tuple(args.sam2_hydra_override or ()),
        sam2_apply_postprocessing=not bool(args.sam2_disable_postprocessing),
    )

    entries = []
    progress = tqdm(frame_lookup.items(), total=len(frame_lookup), desc=args.scene_name, unit="frame")
    for frame_id, frame_path in progress:
        image_bgr = load_color_frame(frame_path)
        gt_labels = gt_extractor.extract_labels(frame_id, image_bgr.shape[:2])
        gt_masks, _ = labels_to_instance_masks(gt_labels)
        pred_masks, pred_scores = sam_masks_with_scores(
            pred_extractor,
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            args.sam_min_mask_area_perc,
            args.sam_sort_mode,
        )
        entries.append(
            {
                "iou": build_mask_iou_matrix(gt_masks, pred_masks),
                "gt_class_ids": np.ones((len(gt_masks),), dtype=np.int32),
                "pred_class_ids": np.ones((len(pred_masks),), dtype=np.int32),
                "pred_scores": pred_scores.astype(np.float32, copy=False),
            }
        )
        progress.set_postfix(gt=len(gt_masks), pred=len(pred_masks), refresh=False)

    metrics, ap_diag = compute_instance_ap_dataset(entries, class_ids=np.array([1], dtype=np.int32), iou_thresholds=MASK_AP_THRESHOLDS)
    summary = {
        "dataset_name": args.dataset_name,
        "scene_name": args.scene_name,
        "metric_name": "mask_ap",
        "metric_value": float(metrics["ap"]),
        "metrics": metrics,
        "n_frames": int(len(entries)),
        "frame_samples": None if args.frame_samples is None else int(args.frame_samples),
        "frame_sample_seed": int(args.frame_sample_seed),
        "sam_model_level_inst": int(args.sam_model_level_inst),
        "diagnostics": ap_diag,
    }
    amg = build_amg_config(args)
    summary.update(
        {
            "sam_sort_mode": amg.sort_mode,
            "sam_min_mask_area_perc": amg.min_mask_area_perc,
            "sam_points_per_side": amg.points_per_side,
            "sam_points_per_batch": amg.points_per_batch,
            "sam_pred_iou_thresh": amg.pred_iou_thresh,
            "sam_stability_score_thresh": amg.stability_score_thresh,
            "sam_stability_score_offset": amg.stability_score_offset,
            "sam_mask_threshold": amg.mask_threshold,
            "sam_box_nms_thresh": amg.box_nms_thresh,
            "sam_crop_n_layers": amg.crop_n_layers,
            "sam_crop_nms_thresh": amg.crop_nms_thresh,
            "sam_crop_overlap_ratio": amg.crop_overlap_ratio,
            "sam_crop_n_points_downscale_factor": amg.crop_n_points_downscale_factor,
            "sam_min_mask_region_area": amg.min_mask_region_area,
            "sam_use_m2m": amg.use_m2m,
            "sam_multimask_output": amg.multimask_output,
            "sam2_mode": args.sam2_mode,
            "sam2_hydra_overrides": list(args.sam2_hydra_override or ()),
            "sam2_apply_postprocessing": not bool(args.sam2_disable_postprocessing),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAM AMG on every frame of a decoded scene and score it against GT instance masks using standard mask AP."
    )
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet"])
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--frame_samples", type=int, default=None)
    parser.add_argument("--frame_sample_seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--sam-model-level-inst", type=int, choices=sorted(SAM_AMG_LEVELS), default=13)
    parser.add_argument("--sam-sort-mode", choices=["area", "predicted_iou", "stability", "score"], default=DEFAULT_AMG_CONFIG.sort_mode)
    parser.add_argument("--sam-min-mask-area-perc", type=float, default=DEFAULT_AMG_CONFIG.min_mask_area_perc)
    parser.add_argument("--sam-points-per-side", type=int, default=DEFAULT_AMG_CONFIG.points_per_side)
    parser.add_argument("--sam-points-per-batch", type=int, default=DEFAULT_AMG_CONFIG.points_per_batch)
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=DEFAULT_AMG_CONFIG.pred_iou_thresh)
    parser.add_argument("--sam-stability-score-thresh", type=float, default=DEFAULT_AMG_CONFIG.stability_score_thresh)
    parser.add_argument("--sam-stability-score-offset", type=float, default=DEFAULT_AMG_CONFIG.stability_score_offset)
    parser.add_argument("--sam-mask-threshold", type=float, default=DEFAULT_AMG_CONFIG.mask_threshold)
    parser.add_argument("--sam-box-nms-thresh", type=float, default=DEFAULT_AMG_CONFIG.box_nms_thresh)
    parser.add_argument("--sam-crop-n-layers", type=int, default=DEFAULT_AMG_CONFIG.crop_n_layers)
    parser.add_argument("--sam-crop-nms-thresh", type=float, default=DEFAULT_AMG_CONFIG.crop_nms_thresh)
    parser.add_argument("--sam-crop-overlap-ratio", type=float, default=DEFAULT_AMG_CONFIG.crop_overlap_ratio)
    parser.add_argument("--sam-crop-n-points-downscale-factor", type=int, default=DEFAULT_AMG_CONFIG.crop_n_points_downscale_factor)
    parser.add_argument("--sam-min-mask-region-area", type=int, default=DEFAULT_AMG_CONFIG.min_mask_region_area)
    parser.add_argument("--sam-use-m2m", action="store_true")
    parser.add_argument("--sam-disable-multimask-output", action="store_true")
    parser.add_argument("--sam2-mode", type=str, default=SAM2_MODE)
    parser.add_argument("--sam2-hydra-override", action="append", default=None)
    parser.add_argument("--sam2-disable-postprocessing", action="store_true")
    args = parser.parse_args()

    summary = evaluate_scene(args)
    if args.save_json:
        out_path = INPUT_DIR / canonical_dataset_name(args.dataset_name) / args.scene_name / "sam_amg_eval.json"
        summary["json_path"] = str(out_path)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
