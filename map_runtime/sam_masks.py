from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import io
import sys

import numpy as np
import torch

from map_runtime.sam2_tracking import _sam2_apply_postprocessing, _sam2_hydra_overrides, _sam2_levels, _sam2_mode


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdParty" / "segment-anything-2"))

_input_dir = Path("data/input")
_sam1_levels = {
    11: ("vit_b", _input_dir / "sam_ckpts" / "sam_vit_b_01ec64.pth"),
    12: ("vit_l", _input_dir / "sam_ckpts" / "sam_vit_l_0b3195.pth"),
    13: ("vit_h", _input_dir / "sam_ckpts" / "sam_vit_h_4b8939.pth"),
}
_sam_amg_levels = {**_sam1_levels, **_sam2_levels}


@dataclass(frozen=True)
class SAMMaskExtractorConfig:
    model_level: int = 13
    sort_mode: str = "area"
    min_mask_area_perc: float = 0.01
    points_per_side: int = 24
    points_per_batch: int = 128
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.92
    stability_score_offset: float = 1.0
    mask_threshold: float = 0.0
    box_nms_thresh: float = 0.7
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7
    crop_overlap_ratio: float = 0.6
    crop_n_points_downscale_factor: int = 1
    point_grids: list[np.ndarray] | None = None
    min_mask_region_area: int = 0
    output_mode: str = "binary_mask"
    use_m2m: bool = False
    multimask_output: bool = True
    score_pred_iou_power: float = 2.0
    score_stability_power: float = 1.0
    score_area_power: float = 0.0
    mask_overlap_rescore_thresh: float = 0.0
    mask_overlap_rescore_power: float = 1.0
    mask_dedupe_iou_thresh: float = 0.85
    mask_containment_thresh: float = 0.0


# TODO: check if all this is even needed
def mask_score(
    mask: dict,
    sort_mode: str,
    pred_iou_power: float,
    stability_power: float,
    area_power: float,
) -> float:
    predicted_iou = max(float(mask.get("predicted_iou", 0.0)), 1e-12)
    stability = max(float(mask.get("stability_score", 0.0)), 1e-12)
    area = max(float(mask.get("area", 0.0)), 1.0)
    if sort_mode == "predicted_iou":
        return (predicted_iou**pred_iou_power) * (area**area_power)
    if sort_mode == "stability":
        return (stability**stability_power) * (area**area_power)
    if sort_mode == "score":
        return (predicted_iou**pred_iou_power) * (stability**stability_power) * (area**area_power)
    return area


def rescore_masks_by_redundancy(
    masks: list[np.ndarray],
    scores: np.ndarray,
    overlap_thresh: float,
    overlap_power: float,
) -> np.ndarray:
    if len(masks) <= 1 or overlap_power <= 0 or overlap_thresh <= 0:
        return scores
    order = np.argsort(scores)[::-1]
    adjusted = scores.astype(np.float32, copy=True)
    kept_flat: list[np.ndarray] = []
    kept_areas: list[int] = []
    for idx in order.tolist():
        mask_flat = masks[idx].reshape(-1)
        area = int(mask_flat.sum())
        if area <= 0:
            adjusted[idx] = 0.0
            continue
        max_overlap = 0.0
        for prev_flat, prev_area in zip(kept_flat, kept_areas):
            intersection = int(np.logical_and(mask_flat, prev_flat).sum(dtype=np.int64))
            smaller_area = min(area, prev_area)
            if smaller_area <= 0:
                continue
            max_overlap = max(max_overlap, intersection / smaller_area)
        if max_overlap > overlap_thresh:
            penalty = max(0.0, 1.0 - max_overlap) ** overlap_power
            adjusted[idx] = float(adjusted[idx] * penalty)
        kept_flat.append(mask_flat)
        kept_areas.append(area)
    return adjusted


def suppress_redundant_masks(
    masks: list[np.ndarray],
    scores: np.ndarray,
    iou_thresh: float,
    containment_thresh: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    if len(masks) <= 1 or (iou_thresh <= 0 and containment_thresh <= 0):
        return masks, scores
    order = np.argsort(scores)[::-1]
    kept_masks: list[np.ndarray] = []
    kept_scores: list[float] = []
    kept_flat: list[np.ndarray] = []
    kept_areas: list[int] = []
    for idx in order.tolist():
        mask = masks[idx]
        mask_flat = mask.reshape(-1)
        area = int(mask_flat.sum())
        if area <= 0:
            continue
        is_redundant = False
        for prev_flat, prev_area in zip(kept_flat, kept_areas):
            intersection = int(np.logical_and(mask_flat, prev_flat).sum(dtype=np.int64))
            if iou_thresh > 0:
                union = area + prev_area - intersection
                if union > 0 and (intersection / union) > iou_thresh:
                    is_redundant = True
                    break
            if containment_thresh > 0:
                smaller_area = min(area, prev_area)
                if smaller_area > 0 and (intersection / smaller_area) > containment_thresh:
                    is_redundant = True
                    break
        if is_redundant:
            continue
        kept_masks.append(mask)
        kept_scores.append(float(scores[idx]))
        kept_flat.append(mask_flat)
        kept_areas.append(area)
    return kept_masks, np.asarray(kept_scores, dtype=np.float32)


def processed_masks_and_scores(
    masks: list[dict],
    image_shape: tuple[int, int, int],
    mask_config: SAMMaskExtractorConfig,
    *,
    max_mask_area_perc: float = 0.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    height, width = image_shape[:2]
    min_mask_area = float(mask_config.min_mask_area_perc) * height * width
    max_mask_area = None if max_mask_area_perc <= 0 else float(max_mask_area_perc) * height * width
    filtered_masks = [
        mask
        for mask in masks
        if float(mask.get("area", 0.0)) >= min_mask_area
        and (max_mask_area is None or float(mask.get("area", 0.0)) <= max_mask_area)
    ]
    if not filtered_masks:
        return [], np.zeros((0,), dtype=np.float32)
    segmentations = [np.asarray(mask["segmentation"], dtype=bool) for mask in filtered_masks]
    scores = np.asarray(
        [
            mask_score(
                mask,
                mask_config.sort_mode,
                mask_config.score_pred_iou_power,
                mask_config.score_stability_power,
                mask_config.score_area_power,
            )
            for mask in filtered_masks
        ],
        dtype=np.float32,
    )
    scores = rescore_masks_by_redundancy(
        segmentations,
        scores,
        mask_config.mask_overlap_rescore_thresh,
        mask_config.mask_overlap_rescore_power,
    )
    segmentations, scores = suppress_redundant_masks(
        segmentations,
        scores,
        mask_config.mask_dedupe_iou_thresh,
        mask_config.mask_containment_thresh,
    )
    return segmentations, scores


def flatten_masks(masks: list[dict], image_shape: tuple[int, int, int], mask_config: SAMMaskExtractorConfig) -> np.ndarray:
    height, width = image_shape[:2]
    min_mask_area = float(mask_config.min_mask_area_perc) * height * width
    segmentations, scores = processed_masks_and_scores(masks, image_shape, mask_config)
    if len(segmentations) == 0:
        return np.full((height, width), -1, dtype=np.int32)
    order = np.argsort(scores)[::-1]
    winning_idx = np.full((height, width), -1, dtype=np.int32)
    winning_score = np.full((height, width), -np.inf, dtype=np.float32)
    for raw_idx in order:
        better = segmentations[raw_idx] & (scores[raw_idx] > winning_score)
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
    def __init__(
        self,
        device: str,
        *,
        config: "SAMMaskExtractorConfig",
    ) -> None:
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.config = config
        self.model_level = int(self.config.model_level)
        self.config_path = None
        if self.model_level in _sam1_levels:
            self._build_sam1_generator()
        elif self.model_level in _sam2_levels:
            self._build_sam2_generator()
        else:
            raise ValueError(f"Unsupported AMG level {self.model_level}. Expected one of {sorted(_sam_amg_levels)}.")

    def _build_sam1_generator(self) -> None:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        model_type, checkpoint_path = _sam1_levels[self.model_level]
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path)).to(self.device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=self.config.points_per_side,
            points_per_batch=self.config.points_per_batch,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            stability_score_offset=self.config.stability_score_offset,
            box_nms_thresh=self.config.box_nms_thresh,
            crop_n_layers=self.config.crop_n_layers,
            crop_nms_thresh=self.config.crop_nms_thresh,
            crop_overlap_ratio=self.config.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
            point_grids=self.config.point_grids,
            min_mask_region_area=self.config.min_mask_region_area,
            output_mode=self.config.output_mode,
        )

    def _build_sam2_generator(self) -> None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        checkpoint_name, config_path = _sam2_levels[self.model_level]
        checkpoint_path = _input_dir / "sam_ckpts" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        self.model_type = Path(config_path).stem
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sam2 = build_sam2(
                config_path,
                str(checkpoint_path),
                device=self.device,
                mode=_sam2_mode,
                hydra_overrides_extra=list(_sam2_hydra_overrides),
                apply_postprocessing=_sam2_apply_postprocessing,
            )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=self.config.points_per_side,
            points_per_batch=self.config.points_per_batch,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_score_thresh,
            stability_score_offset=self.config.stability_score_offset,
            mask_threshold=self.config.mask_threshold,
            box_nms_thresh=self.config.box_nms_thresh,
            crop_n_layers=self.config.crop_n_layers,
            crop_nms_thresh=self.config.crop_nms_thresh,
            crop_overlap_ratio=self.config.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
            point_grids=self.config.point_grids,
            min_mask_region_area=self.config.min_mask_region_area,
            output_mode=self.config.output_mode,
            use_m2m=self.config.use_m2m,
            multimask_output=self.config.multimask_output,
        )

    @torch.inference_mode()
    def extract_masks(self, image: np.ndarray, *, max_mask_area_perc: float = 0.0) -> tuple[list[np.ndarray], np.ndarray]:
        masks = self.mask_generator.generate(image)
        return processed_masks_and_scores(
            masks,
            image.shape,
            self.config,
            max_mask_area_perc=max_mask_area_perc,
        )

    @torch.inference_mode()
    def extract_labels(self, image: np.ndarray) -> np.ndarray:
        masks = self.mask_generator.generate(image)
        return flatten_masks(masks, image.shape, self.config)
