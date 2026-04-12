from __future__ import annotations

import contextlib
import io
import heapq
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize
import torchvision.transforms.functional as TVF
import yaml

from map_runtime.sam2_tracking import INPUT_DIR, SAM2_LEVELS, build_label_masks


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "thirdParty" / "segment-anything-2"))


OVO_TRACK_TH = 100
OVO_PAPER_MATCH_DISTANCE_TH = 0.05
OVO_SIGLIP_MODEL_CARD = "SigLIP-384"
OVO_MASK_RES = 384
OVO_TOP_K_VIEWS = 10
OVO_FEATURE_FUSION = "l1_medoid"
OVO_WEIGHTS_PREDICTOR_DIR = INPUT_DIR / "weights_predictor"
OVO_WEIGHTS_PREDICTOR_FUSION = "none"
OVO_WEIGHTS_PREDICTOR_FUSION_CHOICES = ("none", "learned", "fixed_weights", "hovsg", "adaptive_weights", "concept_fusion")
OVO_PAPER_SAM_MODEL_LEVEL = 24
OVO_PAPER_SAM_POINTS_PER_SIDE = 16
OVO_PAPER_SAM_PRED_IOU_THRESH = 0.8
OVO_PAPER_SAM_STABILITY_SCORE_THRESH = 0.95
OVO_PAPER_SAM_MIN_MASK_REGION_AREA = 0
OVO_PAPER_USE_M2M = False
OVO_PAPER_NMS_IOU_TH = 0.8
OVO_PAPER_NMS_SCORE_TH = 0.7
OVO_PAPER_NMS_INNER_TH = 0.5
OVO_BBOX_MARGIN = 50
OVO_FIXED_W_MASKED = 0.4418
OVO_FIXED_W_GLOBAL = 0.1

OVO_MODEL_CARDS = {
    "SigLIP": "hf-hub:timm/ViT-SO400M-14-SigLIP",
    "SigLIP-384": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
    "SigLIP2-384": "hf-hub:timm/ViT-SO400M-16-SigLIP2-384",
    "ViT-H-14": "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "ViT-B-16-qg": "hf-hub:apple/DFN2B-CLIP-ViT-B-16",
    "ViT-L-14-qg": "hf-hub:apple/DFN2B-CLIP-ViT-L-14-39B",
    "ViT-H-14-qg": "hf-hub:apple/DFN5B-CLIP-ViT-H-14",
    "ViT-H-14-378qg": "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378",
}
OVO_MODEL_DIMS = {
    "SigLIP": 1152,
    "SigLIP-384": 1152,
    "SigLIP2-384": 1152,
    "ViT-H-14": 1024,
    "ViT-B-16-qg": 512,
    "ViT-L-14-qg": 768,
    "ViT-H-14-qg": 1024,
    "ViT-H-14-378qg": 1024,
}

ACTIVATION_DICT = {
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "sigmoid": nn.Sigmoid,
}


def load_openclip_backbone(model_name: str, device: str, *, use_half: bool) -> tuple[torch.nn.Module, object, Compose, int, str]:
    if model_name not in OVO_MODEL_CARDS:
        raise ValueError(f"Unsupported OVO/OpenCLIP backbone '{model_name}'. Expected one of {sorted(OVO_MODEL_CARDS)}.")
    pretrained = OVO_MODEL_CARDS[model_name]
    model, preprocess = open_clip.create_model_from_pretrained(
        pretrained,
        precision="fp16" if use_half else "fp32",
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(pretrained)
    tf_to_keep = [tf for tf in preprocess.transforms if isinstance(tf, Resize) or isinstance(tf, CenterCrop) or isinstance(tf, Normalize)]
    return model, tokenizer, Compose(tf_to_keep), int(OVO_MODEL_DIMS[model_name]), pretrained


def block_mlp(i_dim: int, h_dim: int, o_dim: int, n_layers: int, act_key: str = "leaky_relu") -> nn.Sequential:
    activation = ACTIVATION_DICT[act_key]
    layers: list[nn.Module] = [nn.Linear(i_dim, h_dim), activation()]
    for _ in range(int(n_layers)):
        layers.extend((nn.Linear(h_dim, h_dim), activation()))
    layers.append(nn.Linear(h_dim, o_dim))
    return nn.Sequential(*layers)


class WeightsPredictorMerger(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=config["transformer"]["d_model"],
            nhead=config["transformer"].get("nhead", 8),
            dim_feedforward=config["transformer"]["dim_feedforward"],
            dropout=0.1,
            activation="relu",
        )
        self.att_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["transformer"]["n_layers"])
        self.mlp = block_mlp(**config["mlp"])

    def forward(self, input_clips: torch.Tensor) -> torch.Tensor:
        batch, n_clips, clips_dim = input_clips.shape
        x = self.att_encoder(input_clips)
        weights = self.mlp(x.flatten(-2, -1))
        if weights.shape[-1] != 3:
            weights = weights.reshape(batch, n_clips, clips_dim)
            weights = F.softmax(weights, dim=-2)
        else:
            weights = F.softmax(weights, dim=-1).unsqueeze(-1)
        clips = (input_clips * weights).sum(-2)
        return F.normalize(clips, dim=-1)


def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device, dtype=torch.int64)

    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    in_height = masks.any(dim=-1)
    in_width = masks.any(dim=-2)
    row_coords = torch.arange(h, device=masks.device)[None, :]
    col_coords = torch.arange(w, device=masks.device)[None, :]
    bottom_edges = (in_height * row_coords).max(dim=-1).values
    top_edges = (in_height * row_coords + h * (~in_height)).min(dim=-1).values
    right_edges = (in_width * col_coords).max(dim=-1).values
    left_edges = (in_width * col_coords + w * (~in_width)).min(dim=-1).values
    empty = (right_edges < left_edges) | (bottom_edges < top_edges)
    boxes = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1).to(torch.int64)
    boxes[empty] = 0
    if len(shape) > 2:
        boxes = boxes.reshape(*shape[:-2], 4)
    else:
        boxes = boxes[0]
    return boxes


def batched_box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    box_xywh = box_xyxy.clone()
    box_xywh[:, 2] -= box_xywh[:, 0]
    box_xywh[:, 3] -= box_xywh[:, 1]
    return box_xywh


def increase_bbox_by_margin(bbox: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x -= margin
    y -= margin
    w += margin * 2
    h += margin * 2
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    return int(x), int(y), int(max(0, w)), int(max(0, h))


def pad_to_square(img: torch.Tensor) -> torch.Tensor:
    c, h, w = img.shape
    side = max(h, w)
    out = torch.zeros((c, side, side), dtype=img.dtype, device=img.device)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    out[:, y0 : y0 + h, x0 : x0 + w] = img
    return out


def get_seg_img(mask: torch.Tensor, bbox: tuple[int, int, int, int], image: torch.Tensor) -> torch.Tensor:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return torch.zeros((3, 1, 1), dtype=image.dtype, device=image.device)
    crop = image[:, y : y + h, x : x + w].clone()
    crop_mask = mask[y : y + h, x : x + w]
    crop[:, ~crop_mask] = 0
    return crop


def get_bbox_img(bbox: tuple[int, int, int, int], image: torch.Tensor, bbox_margin: int) -> torch.Tensor:
    x, y, w, h = increase_bbox_by_margin(bbox, bbox_margin)
    crop = image[:, y : y + h, x : x + w].clone()
    if crop.numel() == 0:
        return torch.zeros((3, 1, 1), dtype=image.dtype, device=image.device)
    return crop


def segmap_to_clip_crops(
    binary_maps: torch.Tensor,
    image: torch.Tensor,
    *,
    out_l: int,
    also_bbox: bool,
    bbox_margin: int = OVO_BBOX_MARGIN,
) -> torch.Tensor:
    if binary_maps.numel() == 0:
        channels = 6 if also_bbox else 3
        return torch.empty((0, channels, out_l, out_l), dtype=image.dtype, device=image.device)
    boxes_xyxy = batched_mask_to_box(binary_maps)
    boxes_xywh = batched_box_xyxy_to_xywh(boxes_xyxy)
    crops = []
    for mask, box in zip(binary_maps, boxes_xywh.tolist()):
        bbox = tuple(int(v) for v in box)
        seg_img = get_seg_img(mask, bbox, image)
        if also_bbox:
            bbox_img = TVF.resize(get_bbox_img(bbox, image, bbox_margin), (out_l, out_l))
            crops.append(torch.cat((TVF.resize(seg_img, (out_l, out_l)), bbox_img), dim=0))
        else:
            crops.append(TVF.resize(pad_to_square(seg_img), (out_l, out_l)))
    return torch.stack(crops, dim=0)


def fuse_clips(
    clip_g: torch.Tensor,
    clip_seg: torch.Tensor,
    clip_bbox: torch.Tensor,
    fusion_mode: str,
    *,
    w_masked: float = OVO_FIXED_W_MASKED,
    w_global: float = OVO_FIXED_W_GLOBAL,
) -> torch.Tensor:
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    if fusion_mode in {"hovsg", "fixed_weights"}:
        w_local = float(w_masked)
        clip_local = F.normalize(clip_seg * w_local + clip_bbox * (1.0 - w_local), dim=-1)
        if fusion_mode == "fixed_weights":
            global_weight = clip_local.new_full((clip_local.shape[0], 1), float(w_global))
        else:
            global_weight = torch.softmax(cos(clip_g, clip_local), dim=0).unsqueeze(-1)
        return F.normalize(clip_g * global_weight + clip_local * (1.0 - global_weight), dim=-1)
    if fusion_mode == "adaptive_weights":
        w_local = (cos(clip_seg, clip_bbox) * float(w_masked)).unsqueeze(-1)
        clip_local = F.normalize(clip_seg * w_local + clip_bbox * (1.0 - w_local), dim=-1)
        global_weight = (cos(clip_g, clip_local) * float(w_global)).unsqueeze(-1)
        return F.normalize(clip_g * global_weight + clip_local * (1.0 - global_weight), dim=-1)
    if fusion_mode == "concept_fusion":
        global_weight = torch.softmax(cos(clip_g, clip_bbox), dim=0).unsqueeze(-1)
        return F.normalize(global_weight * clip_g + (1.0 - global_weight) * clip_bbox, dim=-1)
    if fusion_mode == "vanilla":
        return clip_seg
    raise ValueError(f"Unsupported OVO view fusion mode '{fusion_mode}'.")


def mask2segmap(masks: list[dict], image: np.ndarray, *, sort: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if sort:
        masks = heapq.nlargest(len(masks), masks, key=lambda x: float(x["stability_score"]))
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    binary_maps: list[np.ndarray] = []
    for i, mask in enumerate(masks):
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        binary_maps.append(segmentation)
        seg_map_mask = segmentation.copy()
        if sort:
            mask_overlap = np.logical_and(seg_map > -1, seg_map_mask)
            seg_map_mask[mask_overlap] = False
        seg_map[seg_map_mask] = int(i)
    if not binary_maps:
        return np.array([]), np.array([])
    return seg_map, np.stack(binary_maps, axis=0)


def filter_masks(keep: torch.Tensor, masks_result: list[dict]) -> list[dict]:
    keep_ids = set(keep.int().cpu().numpy().tolist())
    return [mask for i, mask in enumerate(masks_result) if i in keep_ids]


def mask_nms(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    iou_thr: float = OVO_PAPER_NMS_IOU_TH,
    score_thr: float = OVO_PAPER_NMS_SCORE_TH,
    inner_thr: float = OVO_PAPER_NMS_INNER_TH,
) -> torch.Tensor:
    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float32)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float32, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float32, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float32)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float32)
            iou = intersection / union.clamp_min(1.0)
            iou_matrix[i, j] = iou
            if intersection / masks_area[i].clamp_min(1.0) < 0.5 and intersection / masks_area[j].clamp_min(1.0) >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j].clamp_min(1.0)) * (intersection / masks_area[i].clamp_min(1.0))
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i].clamp_min(1.0) >= 0.85 and intersection / masks_area[j].clamp_min(1.0) < 0.5:
                inner_iou = 1 - (intersection / masks_area[j].clamp_min(1.0)) * (intersection / masks_area[i].clamp_min(1.0))
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max = iou_matrix.max(dim=0).values
    inner_iou_max_u = torch.triu(inner_iou_matrix, diagonal=1).max(dim=0).values
    inner_iou_max_l = torch.tril(inner_iou_matrix, diagonal=1).max(dim=0).values
    keep = (iou_max <= iou_thr) & (scores > score_thr) & (inner_iou_max_u <= 1 - inner_thr) & (inner_iou_max_l <= 1 - inner_thr)

    if keep.sum() == 0:
        topk = scores.topk(min(3, scores.shape[0])).indices
        keep[topk] = True
    return idx[keep]


def masks_update(
    masks_result: list[dict],
    *,
    iou_thr: float = OVO_PAPER_NMS_IOU_TH,
    score_thr: float = OVO_PAPER_NMS_SCORE_TH,
    inner_thr: float = OVO_PAPER_NMS_INNER_TH,
) -> list[dict]:
    if not masks_result:
        return []
    seg_pred = torch.from_numpy(np.stack([mask["segmentation"] for mask in masks_result], axis=0))
    iou_pred = torch.from_numpy(np.stack([mask["predicted_iou"] for mask in masks_result], axis=0))
    stability = torch.from_numpy(np.stack([mask["stability_score"] for mask in masks_result], axis=0))
    scores = stability * iou_pred
    keep_mask_nms = mask_nms(seg_pred, scores, iou_thr=iou_thr, score_thr=score_thr, inner_thr=inner_thr)
    return filter_masks(keep_mask_nms, masks_result)


class OVOPaperSAMMaskExtractor:
    def __init__(self, device: str, model_level: int = OVO_PAPER_SAM_MODEL_LEVEL) -> None:
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.model_level = int(model_level)
        if self.model_level not in SAM2_LEVELS:
            raise ValueError(f"OVO paper extractor expects a SAM2 level, got {model_level}.")
        checkpoint_name, config_path = SAM2_LEVELS[self.model_level]
        checkpoint_path = INPUT_DIR / "sam_ckpts" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        self.model_type = Path(config_path).stem
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.nms_iou_th = OVO_PAPER_NMS_IOU_TH
        self.nms_score_th = OVO_PAPER_NMS_SCORE_TH
        self.nms_inner_th = OVO_PAPER_NMS_INNER_TH
        self._build_generator()

    def _build_generator(self) -> None:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from sam2.modeling.sam import transformer as sam_transformer

        sam_transformer.USE_FLASH_ATTN = False
        sam_transformer.MATH_KERNEL_ON = True
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sam = build_sam2(
                self.config_path,
                str(self.checkpoint_path),
                device=self.device,
                mode="eval",
                apply_postprocessing=False,
            )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam,
            points_per_side=OVO_PAPER_SAM_POINTS_PER_SIDE,
            pred_iou_thresh=OVO_PAPER_SAM_PRED_IOU_THRESH,
            stability_score_thresh=OVO_PAPER_SAM_STABILITY_SCORE_THRESH,
            min_mask_region_area=OVO_PAPER_SAM_MIN_MASK_REGION_AREA,
            use_m2m=OVO_PAPER_USE_M2M,
        )

    @torch.inference_mode()
    def extract_labels(self, image: np.ndarray) -> np.ndarray:
        masks = self.mask_generator.generate(image)
        if len(masks) == 0:
            return np.full(image.shape[:2], -1, dtype=np.int32)
        masks_default = masks_update(
            masks,
            iou_thr=self.nms_iou_th,
            score_thr=self.nms_score_th,
            inner_thr=self.nms_inner_th,
        )
        seg_map, _ = mask2segmap(masks_default, image)
        if seg_map.size == 0:
            return np.full(image.shape[:2], -1, dtype=np.int32)
        return seg_map.astype(np.int32, copy=False)


def l1_medoid(clips: torch.Tensor) -> torch.Tensor:
    if clips.shape[0] == 1:
        return clips[0]
    distances = torch.abs(clips[:, None, :] - clips[None, :, :]).sum(dim=(1, 2))
    return clips[int(distances.argmin().item())]


def avg_pooling(clips: torch.Tensor) -> torch.Tensor:
    return clips.mean(dim=0)


@dataclass
class OVOFeatureObservation:
    area: int
    descriptor: torch.Tensor


class OVORegionFeatureExtractor:
    def __init__(
        self,
        device: str,
        *,
        model_name: str = OVO_SIGLIP_MODEL_CARD,
        mask_res: int = OVO_MASK_RES,
        use_half: bool = True,
        weights_predictor_fusion: str = OVO_WEIGHTS_PREDICTOR_FUSION,
        weights_predictor_path: Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name
        self.mask_res = int(mask_res)
        self.use_half = bool(use_half) and self.device.type == "cuda"
        self.weights_predictor_fusion = str(weights_predictor_fusion)
        if self.weights_predictor_fusion not in OVO_WEIGHTS_PREDICTOR_FUSION_CHOICES:
            raise ValueError(
                f"Unsupported OVO weights-predictor fusion '{self.weights_predictor_fusion}'. "
                f"Expected one of {list(OVO_WEIGHTS_PREDICTOR_FUSION_CHOICES)}."
            )
        self.embed_type = "vanilla" if self.weights_predictor_fusion == "none" else self.weights_predictor_fusion
        self.weights_predictor_path = Path(weights_predictor_path) if weights_predictor_path is not None else OVO_WEIGHTS_PREDICTOR_DIR
        self.model, self.tokenizer, self.preprocess, self.feature_dim, self.pretrained = load_openclip_backbone(
            model_name,
            str(self.device),
            use_half=self.use_half,
        )
        self.clips_fusion_model: WeightsPredictorMerger | None = None
        if self.embed_type == "learned":
            config_path = self.weights_predictor_path / "hparams.yaml"
            model_path = self.weights_predictor_path / "model.pt"
            if not config_path.exists():
                raise FileNotFoundError(config_path)
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            with config_path.open("r") as f:
                model_config = yaml.safe_load(f)
            model_weights = torch.load(model_path, map_location=str(self.device), weights_only=False)
            self.clips_fusion_model = WeightsPredictorMerger(model_config["model"]).eval().to(self.device)
            self.clips_fusion_model.load_state_dict(model_weights)
            if self.use_half:
                self.clips_fusion_model.half()

    @torch.inference_mode()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        processed = self.preprocess(images)
        if self.use_half:
            processed = processed.half()
        embeds = self.model.encode_image(processed)
        return F.normalize(embeds, dim=-1)

    @torch.inference_mode()
    def encode_text(self, phrases: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(phrases).to(self.device)
        embeds = self.model.encode_text(tokens).float()
        return F.normalize(embeds, dim=-1)

    @torch.inference_mode()
    def extract_visible_instance_descriptors(self, image_np: np.ndarray, labels_np: np.ndarray) -> tuple[list[int], torch.Tensor, np.ndarray]:
        label_masks = build_label_masks(labels_np)
        if not label_masks:
            return [], torch.empty((0, self.feature_dim), device=self.device), np.zeros((0,), dtype=np.int32)
        gids = [int(gid) for gid, _ in label_masks]
        areas = np.asarray([int(mask.sum()) for _, mask in label_masks], dtype=np.int32)
        image = torch.from_numpy(image_np.transpose((2, 0, 1))).to(self.device)
        binary_maps = torch.stack([torch.from_numpy(mask).to(self.device) for _, mask in label_masks], dim=0)
        also_bbox = self.embed_type != "vanilla"
        crops = segmap_to_clip_crops(binary_maps, image, out_l=self.mask_res, also_bbox=also_bbox).float().div_(255.0)
        if crops.shape[0] == 0:
            return [], torch.empty((0, self.feature_dim), device=self.device), np.zeros((0,), dtype=np.int32)
        if self.embed_type == "vanilla":
            return gids, self.encode_image(crops[:, :3]).float(), areas

        global_descriptor = self.encode_image(image[None].float().div_(255.0)).repeat(crops.shape[0], 1)
        seg_bbox_inputs = torch.cat((crops[:, :3], crops[:, 3:]), dim=0)
        seg_bbox_descriptors = self.encode_image(seg_bbox_inputs)
        seg_descriptors = seg_bbox_descriptors[: crops.shape[0]]
        bbox_descriptors = seg_bbox_descriptors[crops.shape[0] :]
        if self.embed_type == "learned":
            if self.clips_fusion_model is None:
                raise RuntimeError("OVO learned fusion model was not initialized.")
            fusion_inputs = torch.cat(
                (global_descriptor[:, None], seg_descriptors[:, None], bbox_descriptors[:, None]),
                dim=1,
            )
            fusion_inputs = fusion_inputs.to(dtype=torch.float16 if self.use_half else torch.float32)
            descriptors = self.clips_fusion_model(fusion_inputs).float()
        else:
            descriptors = fuse_clips(global_descriptor, seg_descriptors, bbox_descriptors, self.embed_type).float()
        return gids, F.normalize(descriptors, dim=-1), areas


class OVOObjectFeatureBank:
    def __init__(
        self,
        extractor: OVORegionFeatureExtractor,
        *,
        top_k_views: int = OVO_TOP_K_VIEWS,
        fusion: str = OVO_FEATURE_FUSION,
    ) -> None:
        self.extractor = extractor
        self.feature_dim = extractor.feature_dim
        self.top_k_views = int(top_k_views)
        self.fusion = fusion
        self.observations: dict[int, list[OVOFeatureObservation]] = {}

    def update(self, image_np: np.ndarray, labels_np: np.ndarray) -> None:
        gids, descriptors, areas = self.extractor.extract_visible_instance_descriptors(image_np, labels_np)
        if not gids:
            return
        descriptors = descriptors.detach().cpu()
        for gid, descriptor, area in zip(gids, descriptors, areas.tolist()):
            bucket = self.observations.setdefault(int(gid), [])
            bucket.append(OVOFeatureObservation(area=int(area), descriptor=descriptor))
            if self.top_k_views > 0 and len(bucket) > self.top_k_views:
                bucket.sort(key=lambda obs: obs.area, reverse=True)
                del bucket[self.top_k_views :]

    def aggregate_descriptor(self, gid: int) -> torch.Tensor | None:
        observations = self.observations.get(int(gid))
        if not observations:
            return None
        clips = torch.stack([obs.descriptor for obs in observations], dim=0)
        if self.fusion == "avg_pooling":
            return avg_pooling(clips)
        if self.fusion != "l1_medoid":
            raise ValueError(f"Unsupported OVO feature fusion '{self.fusion}'.")
        return l1_medoid(clips)

    def build_descriptor_table(self, num_instances: int) -> np.ndarray:
        table = np.zeros((int(num_instances), self.feature_dim), dtype=np.float32)
        for gid in range(int(num_instances)):
            descriptor = self.aggregate_descriptor(gid)
            if descriptor is not None:
                table[gid] = descriptor.numpy().astype(np.float32, copy=False)
        return table


class OVOOnlineInstanceManager:
    def __init__(
        self,
        *,
        seed_mask_extractor,
        textregion_mask_extractor,
        shared_amg_extractor: bool,
        use_inst_gt: bool,
        track_th: int = OVO_TRACK_TH,
    ) -> None:
        self.seed_mask_extractor = seed_mask_extractor
        self.textregion_mask_extractor = textregion_mask_extractor
        self.shared_amg_extractor = bool(shared_amg_extractor)
        self.use_inst_gt = bool(use_inst_gt)
        self.track_th = int(track_th)
        self.instances: dict[int, dict[str, int]] = {}
        self.point_labels = np.empty((0,), dtype=np.int32)
        self.point_labels_dirty = True
        self.next_gid = 0
        self.stats = {
            "ovo_seed_frames": 0,
            "ovo_births": 0,
            "ovo_reused_instances": 0,
            "ovo_track_th": self.track_th,
        }

    def close(self) -> None:
        return

    def extend_point_labels(self, n_new: int) -> None:
        if n_new > 0:
            self.point_labels = np.concatenate((self.point_labels, np.full((int(n_new),), -1, dtype=np.int32)))
            self.point_labels_dirty = True

    def assign_new_points(self, gids: np.ndarray, frame_id: int) -> None:
        if gids.size == 0:
            return
        start = self.point_labels.shape[0]
        self.extend_point_labels(int(gids.size))
        self.point_labels[start:] = gids.astype(np.int32, copy=False)
        self.point_labels_dirty = True
        valid = gids >= 0
        if valid.any():
            label_vals, label_counts = np.unique(gids[valid], return_counts=True)
            for gid, count in zip(label_vals.tolist(), label_counts.tolist()):
                state = self.instances.setdefault(int(gid), {"n_points": 0, "last_seen": int(frame_id)})
                state["n_points"] += int(count)
                state["last_seen"] = int(frame_id)

    def cleanup_tentatives(self, frame_id: int, is_seed_frame: bool) -> None:
        return

    def num_active_instances(self) -> int:
        return len(self.instances)

    def num_existing_instances(self) -> int:
        return len(self.instances)

    def extract_textregion_labels(self, image_np: np.ndarray, seed_labels_np: np.ndarray | None) -> np.ndarray:
        if self.use_inst_gt and seed_labels_np is not None:
            return seed_labels_np
        if seed_labels_np is not None and self.textregion_mask_extractor is self.seed_mask_extractor:
            return seed_labels_np
        if self.textregion_mask_extractor is None:
            raise RuntimeError("TextRegion SAM extractor was not initialized.")
        return self.textregion_mask_extractor.extract_labels(image_np)

    def _extract_seed_labels(self, frame_id: int, image_np: np.ndarray) -> np.ndarray:
        if self.use_inst_gt:
            return self.seed_mask_extractor.extract_labels(frame_id, image_np.shape[:2])
        return self.seed_mask_extractor.extract_labels(image_np)

    def prepare_frame_labels(
        self,
        frame_id: int,
        is_seed_frame: bool,
        image_np: np.ndarray,
        depth_np: np.ndarray,
        c2w_np: np.ndarray,
        point_ids_full: torch.Tensor,
        gid_img_full: torch.Tensor,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        del depth_np, c2w_np
        if not is_seed_frame:
            current = gid_img_full.cpu().numpy().astype(np.int32, copy=True)
            return None, None, current

        self.stats["ovo_seed_frames"] += 1
        seed_labels = self._extract_seed_labels(frame_id, image_np)
        seed_masks = build_label_masks(seed_labels)
        inst_img_full = np.full(image_np.shape[:2], -1, dtype=np.int32)
        reused_instances = 0

        for _, seed_mask in seed_masks:
            mask_tensor = torch.from_numpy(seed_mask).to(point_ids_full.device)
            map_points = point_ids_full[mask_tensor]
            map_points = map_points[map_points >= 0].long()
            if int(map_points.numel()) <= self.track_th:
                continue

            point_labels = self.point_labels[map_points.cpu().numpy()]
            assigned = point_labels >= 0
            unassigned_point_ids = map_points[~torch.from_numpy(assigned).to(map_points.device)]
            gid = -1
            if int(assigned.sum()) > self.track_th:
                assigned_labels = point_labels[assigned]
                label_vals, label_counts = np.unique(assigned_labels, return_counts=True)
                gid = int(label_vals[int(label_counts.argmax())])
                reused_instances += 1
            elif int(unassigned_point_ids.numel()) > self.track_th:
                gid = self.next_gid
                self.next_gid += 1
                self.instances[gid] = {"n_points": 0, "last_seen": int(frame_id)}
                self.stats["ovo_births"] += 1

            if gid < 0:
                continue
            if int(unassigned_point_ids.numel()) > 0:
                self.point_labels[unassigned_point_ids.cpu().numpy()] = int(gid)
                self.point_labels_dirty = True
            inst_img_full[seed_mask] = int(gid)
            state = self.instances.setdefault(int(gid), {"n_points": 0, "last_seen": int(frame_id)})
            state["last_seen"] = int(frame_id)

        self.stats["ovo_reused_instances"] += reused_instances
        return inst_img_full, seed_labels, inst_img_full
