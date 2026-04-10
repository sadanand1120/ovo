from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


INPUT_DIR = Path("data/input")
DATASET_DIRS = {"Replica": "Replica", "ScanNet": "ScanNet"}
SAM_SORT_MODE = "area"
SAM_MIN_MASK_AREA_PERC = 0.01
SAM_POINTS_PER_SIDE = 8
SAM_POINTS_PER_BATCH = 64
SAM_PRED_IOU_THRESH = 0.88
SAM_STABILITY_SCORE_THRESH = 0.95
SAM_STABILITY_SCORE_OFFSET = 1.0
SAM_BOX_NMS_THRESH = 0.7
SAM_CROP_N_LAYERS = 0
SAM_CROP_NMS_THRESH = 0.7
SAM_CROP_OVERLAP_RATIO = 0.3413333333333333
SAM_CROP_N_POINTS_DOWNSCALE_FACTOR = 1
SAM_MIN_MASK_REGION_AREA = 0
SAM_OUTPUT_MODE = "binary_mask"
SAM1_LEVELS = {
    11: ("vit_b", INPUT_DIR / "sam_ckpts" / "sam_vit_b_01ec64.pth"),
    12: ("vit_l", INPUT_DIR / "sam_ckpts" / "sam_vit_l_0b3195.pth"),
    13: ("vit_h", INPUT_DIR / "sam_ckpts" / "sam_vit_h_4b8939.pth"),
}


def canonical_dataset_name(dataset_name: str) -> str:
    return DATASET_DIRS[dataset_name]


def mask_score(mask: dict, sort_mode: str) -> float:
    predicted_iou = float(mask.get("predicted_iou", 0.0))
    stability = float(mask.get("stability_score", 0.0))
    area = float(mask.get("area", 0.0))
    if sort_mode == "predicted_iou":
        return predicted_iou
    if sort_mode == "stability":
        return stability
    if sort_mode == "score":
        return predicted_iou * stability
    return area


def flatten_masks(masks: list[dict], image_shape: tuple[int, int, int], sort_mode: str, min_mask_area_perc: float) -> np.ndarray:
    height, width = image_shape[:2]
    min_mask_area = float(min_mask_area_perc) * height * width
    masks = [mask for mask in masks if float(mask.get("area", 0.0)) >= min_mask_area]
    if not masks:
        return np.full((height, width), -1, dtype=np.int32)
    scores = np.asarray([mask_score(mask, sort_mode) for mask in masks], dtype=np.float32)
    order = np.argsort(scores)[::-1]
    winning_idx = np.full((height, width), -1, dtype=np.int32)
    winning_score = np.full((height, width), -np.inf, dtype=np.float32)
    for raw_idx in order:
        segmentation = np.asarray(masks[raw_idx]["segmentation"], dtype=bool)
        better = segmentation & (scores[raw_idx] > winning_score)
        winning_idx[better] = int(raw_idx)
        winning_score[better] = float(scores[raw_idx])
    labels = np.full((height, width), -1, dtype=np.int32)
    compact_id = 0
    for raw_idx in order:
        claimed = winning_idx == int(raw_idx)
        if claimed.sum() < min_mask_area:
            continue
        if claimed.any():
            labels[claimed] = compact_id
            compact_id += 1
    return labels


class SAMMaskExtractor:
    def __init__(self, device: str, model_level: int) -> None:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        if int(model_level) not in SAM1_LEVELS:
            raise ValueError(f"Unsupported SAM1 level {model_level}. Expected one of {sorted(SAM1_LEVELS)}.")
        model_type, checkpoint_path = SAM1_LEVELS[int(model_level)]
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.model_level = int(model_level)
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path)).to(self.device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=SAM_POINTS_PER_SIDE,
            points_per_batch=SAM_POINTS_PER_BATCH,
            pred_iou_thresh=SAM_PRED_IOU_THRESH,
            stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
            stability_score_offset=SAM_STABILITY_SCORE_OFFSET,
            box_nms_thresh=SAM_BOX_NMS_THRESH,
            crop_n_layers=SAM_CROP_N_LAYERS,
            crop_nms_thresh=SAM_CROP_NMS_THRESH,
            crop_overlap_ratio=SAM_CROP_OVERLAP_RATIO,
            crop_n_points_downscale_factor=SAM_CROP_N_POINTS_DOWNSCALE_FACTOR,
            point_grids=None,
            min_mask_region_area=SAM_MIN_MASK_REGION_AREA,
            output_mode=SAM_OUTPUT_MODE,
        )

    @torch.inference_mode()
    def extract_labels(self, image: np.ndarray) -> np.ndarray:
        masks = self.mask_generator.generate(image)
        return flatten_masks(masks, image.shape, SAM_SORT_MODE, SAM_MIN_MASK_AREA_PERC)


class GTInstanceMaskExtractor:
    def __init__(self, dataset_name: str, scene_name: str) -> None:
        if dataset_name != "ScanNet":
            raise ValueError("GT instance masks are only available for ScanNet.")
        self.mask_dir = INPUT_DIR / canonical_dataset_name(dataset_name) / scene_name / "instance-filt"
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Missing decoded GT instance masks at {self.mask_dir}. Run scannet_decode_sens.py --extract_2d_gt_filt first."
            )

    def extract_labels(self, frame_id: int, image_shape: tuple[int, int]) -> np.ndarray:
        path = self.mask_dir / f"{frame_id}.png"
        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if labels is None:
            raise FileNotFoundError(f"Missing GT instance mask {path}")
        if labels.shape[:2] != image_shape:
            labels = cv2.resize(labels, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        labels = labels.astype(np.int32, copy=False)
        labels[labels == 0] = -1
        valid = labels >= 0
        if valid.any():
            _, local = np.unique(labels[valid], return_inverse=True)
            labels[valid] = local.astype(np.int32, copy=False)
        return labels
