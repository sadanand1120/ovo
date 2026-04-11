from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import io
import sys

import cv2
import numpy as np
import torch

from map_runtime.sam2_tracking import SAM2_APPLY_POSTPROCESSING, SAM2_HYDRA_OVERRIDES, SAM2_LEVELS, SAM2_MODE


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdParty" / "segment-anything-2"))

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
SAM_AMG_LEVELS = {**SAM1_LEVELS, **SAM2_LEVELS}


@dataclass(frozen=True)
class SAMAutomaticMaskConfig:
    sort_mode: str = SAM_SORT_MODE
    min_mask_area_perc: float = SAM_MIN_MASK_AREA_PERC
    points_per_side: int = SAM_POINTS_PER_SIDE
    points_per_batch: int = SAM_POINTS_PER_BATCH
    pred_iou_thresh: float = SAM_PRED_IOU_THRESH
    stability_score_thresh: float = SAM_STABILITY_SCORE_THRESH
    stability_score_offset: float = SAM_STABILITY_SCORE_OFFSET
    mask_threshold: float = 0.0
    box_nms_thresh: float = SAM_BOX_NMS_THRESH
    crop_n_layers: int = SAM_CROP_N_LAYERS
    crop_nms_thresh: float = SAM_CROP_NMS_THRESH
    crop_overlap_ratio: float = SAM_CROP_OVERLAP_RATIO
    crop_n_points_downscale_factor: int = SAM_CROP_N_POINTS_DOWNSCALE_FACTOR
    point_grids: list[np.ndarray] | None = None
    min_mask_region_area: int = SAM_MIN_MASK_REGION_AREA
    output_mode: str = SAM_OUTPUT_MODE
    use_m2m: bool = False
    multimask_output: bool = True


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
    def __init__(
        self,
        device: str,
        model_level: int,
        amg_config: SAMAutomaticMaskConfig | None = None,
        *,
        sam2_mode: str = SAM2_MODE,
        sam2_hydra_overrides: tuple[str, ...] = SAM2_HYDRA_OVERRIDES,
        sam2_apply_postprocessing: bool = SAM2_APPLY_POSTPROCESSING,
    ) -> None:
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.model_level = int(model_level)
        self.config_path = None
        self.amg_config = amg_config or SAMAutomaticMaskConfig()
        self.sam2_mode = sam2_mode
        self.sam2_hydra_overrides = tuple(sam2_hydra_overrides)
        self.sam2_apply_postprocessing = bool(sam2_apply_postprocessing)
        if self.model_level in SAM1_LEVELS:
            self._build_sam1_generator()
        elif self.model_level in SAM2_LEVELS:
            self._build_sam2_generator()
        else:
            raise ValueError(f"Unsupported AMG level {model_level}. Expected one of {sorted(SAM_AMG_LEVELS)}.")

    def _build_sam1_generator(self) -> None:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        model_type, checkpoint_path = SAM1_LEVELS[self.model_level]
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path)).to(self.device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=self.amg_config.points_per_side,
            points_per_batch=self.amg_config.points_per_batch,
            pred_iou_thresh=self.amg_config.pred_iou_thresh,
            stability_score_thresh=self.amg_config.stability_score_thresh,
            stability_score_offset=self.amg_config.stability_score_offset,
            box_nms_thresh=self.amg_config.box_nms_thresh,
            crop_n_layers=self.amg_config.crop_n_layers,
            crop_nms_thresh=self.amg_config.crop_nms_thresh,
            crop_overlap_ratio=self.amg_config.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.amg_config.crop_n_points_downscale_factor,
            point_grids=self.amg_config.point_grids,
            min_mask_region_area=self.amg_config.min_mask_region_area,
            output_mode=self.amg_config.output_mode,
        )

    def _build_sam2_generator(self) -> None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        checkpoint_name, config_path = SAM2_LEVELS[self.model_level]
        checkpoint_path = INPUT_DIR / "sam_ckpts" / checkpoint_name
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
                mode=self.sam2_mode,
                hydra_overrides_extra=list(self.sam2_hydra_overrides),
                apply_postprocessing=self.sam2_apply_postprocessing,
            )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam2,
            points_per_side=self.amg_config.points_per_side,
            points_per_batch=self.amg_config.points_per_batch,
            pred_iou_thresh=self.amg_config.pred_iou_thresh,
            stability_score_thresh=self.amg_config.stability_score_thresh,
            stability_score_offset=self.amg_config.stability_score_offset,
            mask_threshold=self.amg_config.mask_threshold,
            box_nms_thresh=self.amg_config.box_nms_thresh,
            crop_n_layers=self.amg_config.crop_n_layers,
            crop_nms_thresh=self.amg_config.crop_nms_thresh,
            crop_overlap_ratio=self.amg_config.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.amg_config.crop_n_points_downscale_factor,
            point_grids=self.amg_config.point_grids,
            min_mask_region_area=self.amg_config.min_mask_region_area,
            output_mode=self.amg_config.output_mode,
            use_m2m=self.amg_config.use_m2m,
            multimask_output=self.amg_config.multimask_output,
        )

    @torch.inference_mode()
    def extract_labels(self, image: np.ndarray) -> np.ndarray:
        masks = self.mask_generator.generate(image)
        return flatten_masks(
            masks,
            image.shape,
            self.amg_config.sort_mode,
            self.amg_config.min_mask_area_perc,
        )


class GTInstanceMaskExtractor:
    def __init__(self, dataset_name: str, scene_name: str) -> None:
        self.mask_dir = INPUT_DIR / canonical_dataset_name(dataset_name) / scene_name / "instance-filt"
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Missing decoded GT instance masks at {self.mask_dir}."
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
