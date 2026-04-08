import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import cv2  # Keep OpenCV loaded before torch in the container env.
import numpy as np
import open3d as o3d
import open_clip
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ovo import geometry_utils, io_utils
from ovo.datasets import get_dataset


DATASET_DIRS = {"replica": "Replica", "scannet": "ScanNet"}
CONFIG_DIR = Path("configs")
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output/rgb_maps")
TIMING_PATH = "timing.json"
CLIP_FEATURE_FILE = "clip_feats.npy"
DEFAULT_MAP_EVERY = 8
DEFAULT_DOWNSCALE_RES = 2
DEFAULT_K_POOLING = 1
DEFAULT_MAX_FRAME_POINTS = 5_000_000
DEFAULT_MATCH_DISTANCE_TH = 0.03
DEFAULT_INSTANCE_MATCH_TH = 3
DEFAULT_INSTANCE_NEW_TH = 80
DEFAULT_INSTANCE_DOMINANT_FRAC_TH = 0.4
CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_LOAD_SIZE = 1024
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_GLOBAL_PATCH_THRESHOLD = 0.07
SAM_CHECKPOINT_PATH = INPUT_DIR / "sam_ckpts" / "sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"
SAM_SORT_MODE = "area"
SAM_MIN_MASK_AREA_PERC = 0.01
SAM_POINTS_PER_SIDE = 8


def canonical_dataset_name(dataset_name: str) -> str:
    return DATASET_DIRS[dataset_name.lower()]


def load_dataset(dataset_name: str, scene_name: str, frame_limit: int | None):
    config = io_utils.load_config(CONFIG_DIR / "ovo.yaml")
    io_utils.update_recursive(config, io_utils.load_config(CONFIG_DIR / f"{dataset_name.lower()}.yaml"))
    dataset_cfg = {
        **config["cam"],
        "input_path": str(INPUT_DIR / canonical_dataset_name(dataset_name) / scene_name),
        "frame_limit": config["data"].get("frame_limit", -1) if frame_limit is None else frame_limit,
    }
    return get_dataset(dataset_name.lower())(dataset_cfg)


def as_int(value) -> int:
    return int(float(value))


def resolve_resized_hw(width: int, height: int, size: int) -> tuple[int, int]:
    if width <= height:
        return size, max(1, int(round(height * size / width)))
    return max(1, int(round(width * size / height))), size


def pad_to_multiple(batch: torch.Tensor, patch_size: int) -> torch.Tensor:
    pad_h = (-batch.shape[-2]) % patch_size
    pad_w = (-batch.shape[-1]) % patch_size
    if pad_h == 0 and pad_w == 0:
        return batch
    return F.pad(batch, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def interpolate_positional_embedding(positional_embedding: torch.Tensor, x: torch.Tensor, patch_size: int, height: int, width: int) -> torch.Tensor:
    num_patches = x.shape[1] - 1
    num_original_patches = positional_embedding.shape[0] - 1
    if num_patches == num_original_patches and height == width:
        return positional_embedding.to(x.dtype)

    dim = x.shape[-1]
    class_pos_embed = positional_embedding[:1]
    patch_pos_embed = positional_embedding[1:]
    grid_h = height // patch_size
    grid_w = width // patch_size
    patch_per_axis = int(np.sqrt(num_original_patches))
    patch_pos_embed = patch_pos_embed.reshape(1, patch_per_axis, patch_per_axis, dim).permute(0, 3, 1, 2)
    patch_pos_embed = F.interpolate(patch_pos_embed, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(-1, dim)
    return torch.cat([class_pos_embed, patch_pos_embed], dim=0).to(x.dtype)


def labels_to_patch_weights(labels: torch.Tensor, grid_h: int, grid_w: int, padded_h: int, padded_w: int) -> torch.Tensor:
    labels = labels.long()
    valid_labels = torch.unique(labels[labels >= 0])
    if valid_labels.numel() == 0:
        return torch.zeros((0, grid_h * grid_w), device=labels.device)
    masks = torch.stack([(labels == label).float() for label in valid_labels], dim=0)
    pad_h = padded_h - masks.shape[-2]
    pad_w = padded_w - masks.shape[-1]
    if pad_h > 0 or pad_w > 0:
        masks = F.pad(masks, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    weights = F.interpolate(masks[:, None], size=(grid_h, grid_w), mode="bilinear", align_corners=False)[:, 0]
    weights = weights.reshape(weights.shape[0], -1).clamp_(0.0, 1.0)
    keep = weights.sum(dim=1) > 0
    return weights[keep]


def remove_global_patches(mask_weights: torch.Tensor, patch_features: torch.Tensor, threshold: float) -> torch.Tensor:
    if mask_weights.numel() == 0:
        return mask_weights
    patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    patch_similarity = patch_features @ patch_features.T
    patch_to_region = patch_similarity @ mask_weights.T
    patch_to_region_avg = patch_to_region / mask_weights.sum(dim=-1).clamp_min(1e-6)
    belong = patch_to_region_avg * mask_weights.T
    belong_avg = belong.sum(dim=-1) / mask_weights.sum(dim=0).clamp_min(1e-6)
    outside = patch_to_region_avg * (1.0 - mask_weights).T
    outside_avg = outside.sum(dim=-1) / (1.0 - mask_weights).sum(dim=0).clamp_min(1e-6)
    difference = belong_avg - outside_avg
    filtered = mask_weights.clone()
    filtered[:, difference < threshold] = 0
    keep = filtered.sum(dim=1) > 0
    return filtered[keep]


def compute_normals_from_depth(x: torch.Tensor, y: torch.Tensor, depth: torch.Tensor, intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    vertex_map = torch.stack(
        (
            (x - cx) * depth / fx,
            (y - cy) * depth / fy,
            depth,
        ),
        dim=-1,
    )
    valid = depth > 0
    normals = torch.zeros_like(vertex_map)
    normal_valid = torch.zeros_like(valid)
    dx = vertex_map[1:-1, 2:] - vertex_map[1:-1, :-2]
    dy = vertex_map[2:, 1:-1] - vertex_map[:-2, 1:-1]
    inner_normals = torch.linalg.cross(dy, dx, dim=-1)
    inner_norm = torch.linalg.norm(inner_normals, dim=-1, keepdim=True)
    inner_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & (inner_norm[..., 0] > 1e-8)
    )
    inner_normals = inner_normals / inner_norm.clamp_min(1e-8)
    center = vertex_map[1:-1, 1:-1]
    flip = (inner_normals * center).sum(dim=-1) > 0
    inner_normals[flip] = -inner_normals[flip]
    normals[1:-1, 1:-1] = inner_normals
    normal_valid[1:-1, 1:-1] = inner_valid
    normals[~normal_valid] = 0
    return normals, normal_valid


def invert_rigid_transform(c2w: torch.Tensor) -> torch.Tensor:
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    world_to_camera = torch.empty_like(c2w)
    rotation_t = rotation.transpose(0, 1)
    world_to_camera[:3, :3] = rotation_t
    world_to_camera[:3, 3] = -(rotation_t @ translation)
    world_to_camera[3] = c2w.new_tensor((0.0, 0.0, 0.0, 1.0))
    return world_to_camera


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
    def __init__(self, device: str) -> None:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT_PATH)).to(self.device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=SAM_POINTS_PER_SIDE)

    @torch.inference_mode()
    def extract_labels(self, image: np.ndarray) -> np.ndarray:
        masks = self.mask_generator.generate(image)
        return flatten_masks(masks, image.shape, SAM_SORT_MODE, SAM_MIN_MASK_AREA_PERC)


class GTInstanceMaskExtractor:
    def __init__(self, dataset_name: str, scene_name: str) -> None:
        if dataset_name.lower() != "scannet":
            raise ValueError("GT instance masks are only available for ScanNet.")
        self.mask_dir = INPUT_DIR / canonical_dataset_name(dataset_name) / scene_name / "instance-filt"
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing decoded GT instance masks at {self.mask_dir}. Run scannet_decode_sens.py --extract_2d_gt_filt first.")

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


class OnlineInstanceTracker:
    def __init__(self, match_th: int, new_th: int, dominant_frac_th: float) -> None:
        self.match_th = int(match_th)
        self.new_th = int(new_th)
        self.dominant_frac_th = float(dominant_frac_th)
        self.point_labels = np.empty((0,), dtype=np.int32)
        self.next_instance_id = 0

    def append_new_points(self, n_new: int) -> None:
        if n_new > 0:
            self.point_labels = np.concatenate((self.point_labels, np.full((int(n_new),), -1, dtype=np.int32)))

    def update(self, point_ids_ds: torch.Tensor, mask_labels_ds: torch.Tensor) -> None:
        valid = (point_ids_ds >= 0) & (mask_labels_ds >= 0)
        if not valid.any():
            return
        point_ids = point_ids_ds[valid].cpu().numpy().astype(np.int64, copy=False)
        mask_labels = mask_labels_ds[valid].cpu().numpy().astype(np.int32, copy=False)
        order = np.lexsort((point_ids, mask_labels))
        point_ids = point_ids[order]
        mask_labels = mask_labels[order]
        starts = np.flatnonzero(np.r_[True, mask_labels[1:] != mask_labels[:-1]])
        ends = np.r_[starts[1:], mask_labels.shape[0]]
        for start, end in zip(starts, ends):
            ids = np.unique(point_ids[start:end])
            if ids.shape[0] < self.match_th:
                continue
            labels = self.point_labels[ids]
            assigned = labels >= 0
            if assigned.sum() > 0:
                label_vals, label_counts = np.unique(labels[assigned], return_counts=True)
                dominant_idx = int(label_counts.argmax())
                dominant = int(label_vals[dominant_idx])
                dominant_frac = float(label_counts[dominant_idx]) / float(assigned.sum())
                if assigned.sum() >= self.match_th and dominant_frac >= self.dominant_frac_th:
                    self.point_labels[ids[~assigned]] = dominant
                    continue
            if (~assigned).sum() >= self.new_th:
                self.point_labels[ids[~assigned]] = self.next_instance_id
                self.next_instance_id += 1


class DenseCLIPExtractor:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.model = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            precision="fp32",
        )[0].eval().to(self.device)
        self.visual = self.model.visual
        patch_size = self.visual.patch_size
        self.patch_size = int(patch_size if isinstance(patch_size, int) else patch_size[0])
        self.feature_dim = int(self.visual.proj.shape[1] if self.visual.proj is not None else self.visual.ln_post.normalized_shape[0])
        self.pre_resblocks = tuple(self.visual.transformer.resblocks[:-1])
        self.last_resblock = self.visual.transformer.resblocks[-1]

    @torch.inference_mode()
    def extract_dense(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        image = image.permute(2, 0, 1).float().div_(255.0)
        orig_h, orig_w = image.shape[-2:]
        resized_h, resized_w = resolve_resized_hw(orig_w, orig_h, CLIP_LOAD_SIZE)
        image = F.interpolate(image[None], size=(resized_h, resized_w), mode="bicubic", align_corners=False, antialias=True)[0]
        mean = image.new_tensor(CLIP_MEAN).view(3, 1, 1)
        std = image.new_tensor(CLIP_STD).view(3, 1, 1)
        image = ((image - mean) / std)[None]
        image = pad_to_multiple(image, self.patch_size)

        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x = self.visual.conv1(image)
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                cls = self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                x = torch.cat([cls, x], dim=1)
                x = x + interpolate_positional_embedding(self.visual.positional_embedding, x, self.patch_size, image.shape[-2], image.shape[-1])
                x = self.visual.ln_pre(x)
                for block in self.pre_resblocks:
                    x = block(x)
                x_ln = self.last_resblock.ln_1(x)
                qkv = F.linear(x_ln, self.last_resblock.attn.in_proj_weight, self.last_resblock.attn.in_proj_bias)
        else:
            x = self.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
            x = x + interpolate_positional_embedding(self.visual.positional_embedding, x, self.patch_size, image.shape[-2], image.shape[-1])
            x = self.visual.ln_pre(x)
            for block in self.pre_resblocks:
                x = block(x)
            x_ln = self.last_resblock.ln_1(x)
            qkv = F.linear(x_ln, self.last_resblock.attn.in_proj_weight, self.last_resblock.attn.in_proj_bias)

        grid_h = image.shape[-2] // self.patch_size
        grid_w = image.shape[-1] // self.patch_size
        _, _, v = qkv.chunk(3, dim=-1)
        v = v.float()
        baseline = F.linear(v[:, 1:, :], self.last_resblock.attn.out_proj.weight, self.last_resblock.attn.out_proj.bias)
        baseline = self.visual.ln_post(baseline)
        if self.visual.proj is not None:
            baseline = baseline @ self.visual.proj
        baseline = baseline.reshape(grid_h, grid_w, -1)

        labels = labels.to(self.device)
        if labels.shape != (orig_h, orig_w):
            labels = F.interpolate(labels[None, None].float(), size=(orig_h, orig_w), mode="nearest")[0, 0].long()
        mask_weights = labels_to_patch_weights(labels, grid_h, grid_w, image.shape[-2], image.shape[-1])
        mask_weights = remove_global_patches(mask_weights, baseline.reshape(-1, baseline.shape[-1]), CLIP_GLOBAL_PATCH_THRESHOLD)
        if mask_weights.shape[0] > 0:
            num_heads = self.last_resblock.attn.num_heads
            embed_dim = v.shape[-1]
            head_dim = embed_dim // num_heads
            v_multi_head = v[:, 1:, :].reshape(1, grid_h * grid_w, num_heads, head_dim).permute(0, 2, 1, 3).reshape(num_heads, grid_h * grid_w, head_dim)
            attn_weights = mask_weights.unsqueeze(0).repeat(num_heads, 1, 1).to(dtype=v_multi_head.dtype)
            attn_output = torch.bmm(attn_weights, v_multi_head)
            attn_output = attn_output.permute(1, 0, 2).reshape(mask_weights.shape[0], embed_dim)
            attn_output = self.last_resblock.attn.out_proj(attn_output)
            attn_output = attn_output + self.last_resblock.mlp(self.last_resblock.ln_2(attn_output))
            region_features = self.visual.ln_post(attn_output)
            if self.visual.proj is not None:
                region_features = region_features @ self.visual.proj
            patch_weights_t = mask_weights.T
            patch_sum = patch_weights_t @ region_features
            patch_norm = patch_weights_t.sum(dim=1, keepdim=True).clamp_min(1e-6)
            region_dense = patch_sum / patch_norm
            use_baseline = patch_weights_t.sum(dim=1) <= 0
            region_dense[use_baseline] = baseline.reshape(-1, baseline.shape[-1])[use_baseline]
            x = region_dense.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2)
        else:
            x = baseline.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2)

        x = x.float()
        x = F.interpolate(x, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
        if (resized_h, resized_w) != (orig_h, orig_w):
            x = F.interpolate(x, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        return x[0].permute(1, 2, 0).contiguous()


class RGBMapper:
    def __init__(
        self,
        intrinsics: np.ndarray,
        device: str,
        map_every: int,
        downscale_res: int,
        k_pooling: int,
        max_frame_points: int,
        match_distance_th: float,
        dataset_name: str,
        scene_name: str,
        use_inst_gt: bool,
    ) -> None:
        self.device = device
        self.cam_intrinsics = torch.tensor(intrinsics.astype(np.float32), device=device)
        self.map_every = max(1, int(map_every))
        self.downscale_res = max(1, int(downscale_res))
        self.max_frame_points = as_int(max_frame_points)
        self.match_distance_th = float(match_distance_th)
        self.k_pooling = int(k_pooling)
        self.clip_extractor = DenseCLIPExtractor(device)
        self.use_inst_gt = bool(use_inst_gt)
        self.mask_extractor = GTInstanceMaskExtractor(dataset_name, scene_name) if self.use_inst_gt else SAMMaskExtractor(device)
        self.instance_tracker = OnlineInstanceTracker(
            DEFAULT_INSTANCE_MATCH_TH,
            DEFAULT_INSTANCE_NEW_TH,
            DEFAULT_INSTANCE_DOMINANT_FRAC_TH,
        )
        if k_pooling > 1 and k_pooling % 2 == 0:
            raise ValueError("k_pooling must be odd.")

        self.n_points = 0
        self.points = torch.empty((0, 3), device=device)
        self.colors = torch.empty((0, 3), device=device, dtype=torch.uint8)
        self.normals = torch.empty((0, 3), device=device)
        self.color_sum = torch.empty((0, 3), device=device)
        self.normal_sum = torch.empty((0, 3), device=device)
        self.obs_count = torch.empty((0,), device=device)
        self.feature_tmpdir = Path(tempfile.mkdtemp(prefix="rgb_map_feats_"))
        self.feature_tmp_path = self.feature_tmpdir / "clip_feats.bin"
        self.feature_tmp_file = open(self.feature_tmp_path, "wb")
        self.n_features = 0
        self.cached_depth_shape = None
        self.cached_x = None
        self.cached_y = None
        self.cached_row_ids = None
        self.cached_col_ids = None

        if k_pooling > 1:
            pooling = torch.nn.MaxPool2d(kernel_size=k_pooling, stride=1, padding=k_pooling // 2)
            self.pooling = lambda mask: ~(pooling((~mask[None]).float())[0].bool())
        else:
            self.pooling = lambda mask: mask
        self.downscale = (lambda x: x) if self.downscale_res == 1 else (lambda x: x[::self.downscale_res, ::self.downscale_res])

    def should_map_frame(self, frame_id: int) -> bool:
        return frame_id % self.map_every == 0

    def _ensure_capacity(self, min_capacity: int) -> None:
        if min_capacity <= self.points.shape[0]:
            return
        new_capacity = max(min_capacity, max(1 << 18, self.points.shape[0] * 2))

        def grow(buffer: torch.Tensor, *shape_tail: int) -> torch.Tensor:
            expanded = torch.empty((new_capacity, *shape_tail), device=buffer.device, dtype=buffer.dtype)
            if self.n_points > 0:
                expanded[:self.n_points] = buffer[:self.n_points]
            return expanded

        self.points = grow(self.points, 3)
        self.colors = grow(self.colors, 3)
        self.normals = grow(self.normals, 3)
        self.color_sum = grow(self.color_sum, 3)
        self.normal_sum = grow(self.normal_sum, 3)
        self.obs_count = grow(self.obs_count)

    def _get_cached_grids(self, depth_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cached_depth_shape != depth_shape:
            h, w = depth_shape
            full_y, full_x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
            self.cached_x = self.downscale(full_x)
            self.cached_y = self.downscale(full_y)
            ds_h, ds_w = self.cached_x.shape
            self.cached_row_ids, self.cached_col_ids = torch.meshgrid(
                torch.arange(ds_h, device=self.device),
                torch.arange(ds_w, device=self.device),
                indexing="ij",
            )
            self.cached_depth_shape = depth_shape
        return self.cached_x, self.cached_y, self.cached_row_ids, self.cached_col_ids

    def _append_points(self, points: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, features: torch.Tensor) -> None:
        start = self.n_points
        end = start + int(points.shape[0])
        self._ensure_capacity(end)
        self.points[start:end] = points
        self.colors[start:end] = colors
        self.normals[start:end] = normals
        self.color_sum[start:end] = colors.float()
        self.normal_sum[start:end] = normals.float()
        self.obs_count[start:end] = 1
        self.n_points = end
        feature_block = np.ascontiguousarray(
            torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy(),
            dtype=np.float16,
        )
        feature_block.tofile(self.feature_tmp_file)
        self.n_features += int(features.shape[0])

    def _update_observed_points(self, point_ids: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor) -> None:
        if point_ids.numel() == 0:
            return
        point_ids = point_ids.long()
        unique_ids = torch.unique(point_ids)
        self.color_sum.index_add_(0, point_ids, colors.float())
        self.normal_sum.index_add_(0, point_ids, normals.float())
        self.obs_count.index_add_(0, point_ids, torch.ones((point_ids.shape[0],), device=self.device))
        counts = self.obs_count[unique_ids].unsqueeze(1)
        self.colors[unique_ids] = torch.round(self.color_sum[unique_ids] / counts).clamp_(0.0, 255.0).to(torch.uint8)
        mean_normals = self.normal_sum[unique_ids] / counts
        self.normals[unique_ids] = mean_normals / torch.linalg.norm(mean_normals, dim=1, keepdim=True).clamp_min(1e-8)

    def add_frame(self, frame_data) -> None:
        frame_id, image_np, depth_np, c2w_np = frame_data[:4]
        if not self.should_map_frame(frame_id):
            return
        if np.isinf(c2w_np).any() or np.isnan(c2w_np).any():
            return

        labels_np = self.mask_extractor.extract_labels(frame_id, image_np.shape[:2]) if self.use_inst_gt else self.mask_extractor.extract_labels(image_np)
        instance_labels_full = torch.from_numpy(labels_np).to(self.device)
        c2w = torch.from_numpy(c2w_np).to(self.device)
        depth = torch.from_numpy(depth_np).to(self.device)
        image = torch.from_numpy(image_np).to(self.device)
        full_image = image
        h, w = depth.shape
        x, y, row_ids, col_ids = self._get_cached_grids((h, w))
        mask = depth > 0
        point_ids_full = torch.full((h, w), -1, dtype=torch.int32, device=self.device)

        if self.n_points > 0:
            frustum_corners = geometry_utils.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
            w2c = invert_rigid_transform(c2w)
            frustum_mask = geometry_utils.compute_frustum_point_ids(self.points[: self.n_points], frustum_corners, device=self.device)
            if frustum_mask.numel() > 0:
                matched_ids, matches = geometry_utils.match_3d_points_to_2d_pixels(
                    depth,
                    w2c,
                    self.points[frustum_mask],
                    self.cam_intrinsics,
                    self.match_distance_th,
                )
                if matches.numel() > 0:
                    global_ids = frustum_mask[matched_ids].to(point_ids_full.dtype)
                    point_ids_full[matches[:, 1], matches[:, 0]] = global_ids
                    mask[matches[:, 1], matches[:, 0]] = False
            mask = self.pooling(mask)

        depth = self.downscale(depth)
        mask = self.downscale(mask)
        image = self.downscale(image)
        point_ids_ds = self.downscale(point_ids_full)
        instance_labels_ds = self.downscale(instance_labels_full)
        normals_cam, normal_valid = compute_normals_from_depth(x, y, depth, self.cam_intrinsics)
        visible_existing = (point_ids_ds >= 0) & normal_valid
        if visible_existing.any():
            visible_ids = point_ids_ds[visible_existing]
            visible_colors = image[visible_existing].reshape(-1, 3)
            visible_normals = normals_cam[visible_existing].reshape(-1, 3)
            visible_normals = torch.einsum("ij,mj->mi", c2w[:3, :3], visible_normals)
            visible_normals = visible_normals / torch.linalg.norm(visible_normals, dim=1, keepdim=True).clamp_min(1e-8)
            self._update_observed_points(visible_ids, visible_colors, visible_normals)
        mask = mask & normal_valid

        if mask.any():
            x_keep = x[mask]
            y_keep = y[mask]
            row_keep = row_ids[mask]
            col_keep = col_ids[mask]
            depth_keep = depth[mask]
            colors = image[mask].reshape(-1, 3)
            normals_cam = normals_cam[mask].reshape(-1, 3)
            if depth_keep.shape[0] > self.max_frame_points:
                keep = torch.linspace(0, depth_keep.shape[0] - 1, self.max_frame_points, device=self.device).round().long()
                x_keep, y_keep = x_keep[keep], y_keep[keep]
                row_keep, col_keep = row_keep[keep], col_keep[keep]
                depth_keep, colors = depth_keep[keep], colors[keep]
                normals_cam = normals_cam[keep]

            dense_clip = self.clip_extractor.extract_dense(full_image, instance_labels_full)
            features = dense_clip[y_keep, x_keep].half()
            x_3d = (x_keep - self.cam_intrinsics[0, 2]) * depth_keep / self.cam_intrinsics[0, 0]
            y_3d = (y_keep - self.cam_intrinsics[1, 2]) * depth_keep / self.cam_intrinsics[1, 1]
            points = torch.stack((x_3d, y_3d, depth_keep, torch.ones_like(depth_keep)), dim=1)
            points = torch.einsum("ij,mj->mi", c2w, points)[:, :3]
            normals = torch.einsum("ij,mj->mi", c2w[:3, :3], normals_cam)
            normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-8)
            old_n = self.n_points
            new_ids = torch.arange(old_n, old_n + points.shape[0], device=self.device, dtype=torch.int32)
            point_ids_ds[row_keep, col_keep] = new_ids
            self.instance_tracker.append_new_points(points.shape[0])
            self.instance_tracker.update(point_ids_ds, instance_labels_ds)
            self._append_points(points, colors, normals, features)
            return

        self.instance_tracker.update(point_ids_ds, instance_labels_ds)

    def save(self, output_dir: Path, stats: dict) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_start = time.perf_counter()
        progress = tqdm(total=4, desc=f"{output_dir.name} save", unit="stage", dynamic_ncols=True)
        timings = {}
        try:
            progress.set_postfix_str("write ply", refresh=True)
            stage_start = time.perf_counter()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points[: self.n_points].cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(self.colors[: self.n_points].cpu().numpy().astype(np.float32) / 255.0)
            pcd.normals = o3d.utility.Vector3dVector(self.normals[: self.n_points].cpu().numpy())
            o3d.io.write_point_cloud(str(output_dir / "rgb_map.ply"), pcd)
            timings["write_ply_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("store clip", refresh=True)
            stage_start = time.perf_counter()
            self.feature_tmp_file.flush()
            self.feature_tmp_file.close()
            with open(output_dir / CLIP_FEATURE_FILE, "wb") as f:
                np.lib.format.write_array_header_2_0(
                    f,
                    {
                        "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                        "fortran_order": False,
                        "shape": (self.n_features, self.clip_extractor.feature_dim),
                    },
                )
                with open(self.feature_tmp_path, "rb") as src:
                    shutil.copyfileobj(src, f, length=16 * 1024 * 1024)
            timings["store_clip_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("instance labels", refresh=True)
            stage_start = time.perf_counter()
            np.save(output_dir / "instance_labels.npy", self.instance_tracker.point_labels)
            timings["instance_labels_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("stats", refresh=True)
            stage_start = time.perf_counter()
            stats = {
                **stats,
                "instance_supervision": "gt" if self.use_inst_gt else "sam",
                "instance_label_path": "instance_labels.npy",
                "clip_feature_path": CLIP_FEATURE_FILE,
                "clip_feature_storage": "npy",
                "rgb_normal_point_fusion": True,
                "clip_feature_mode": "clip_textregion",
                "clip_feature_fusion": False,
                "instance_match_th": DEFAULT_INSTANCE_MATCH_TH,
                "instance_new_th": DEFAULT_INSTANCE_NEW_TH,
                "instance_dominant_frac_th": DEFAULT_INSTANCE_DOMINANT_FRAC_TH,
            }
            if not self.use_inst_gt:
                stats.update(
                    {
                        "sam_model_type": SAM_MODEL_TYPE,
                        "sam_checkpoint_path": str(SAM_CHECKPOINT_PATH),
                        "sam_points_per_side": SAM_POINTS_PER_SIDE,
                        "sam_sort_mode": SAM_SORT_MODE,
                        "sam_min_mask_area_perc": SAM_MIN_MASK_AREA_PERC,
                    }
                )
            with open(output_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            timings["stats_sec"] = time.perf_counter() - stage_start
            progress.update()
        finally:
            progress.set_postfix_str("stats", refresh=True)
            progress.close()
            if not self.feature_tmp_file.closed:
                self.feature_tmp_file.close()
            shutil.rmtree(self.feature_tmpdir, ignore_errors=True)
        timings["save_total_sec"] = time.perf_counter() - save_start
        return timings


def main(args):
    run_start = time.perf_counter()
    dataset_name = args.dataset_name.lower()
    output_dir = Path(args.output_root) / canonical_dataset_name(dataset_name) / args.scene_name
    dataset_load_start = time.perf_counter()
    dataset = load_dataset(dataset_name, args.scene_name, args.frame_limit)
    dataset_load_sec = time.perf_counter() - dataset_load_start
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapper = RGBMapper(
        intrinsics=dataset.intrinsics,
        device=device,
        map_every=args.map_every,
        downscale_res=args.downscale_res,
        k_pooling=args.k_pooling,
        max_frame_points=args.max_frame_points,
        match_distance_th=args.match_distance_th,
        dataset_name=dataset_name,
        scene_name=args.scene_name,
        use_inst_gt=args.use_inst_gt,
    )

    progress = tqdm(range(len(dataset)), desc=args.scene_name, unit="frame")
    frame_loop_start = time.perf_counter()
    for frame_id in progress:
        if not mapper.should_map_frame(frame_id):
            continue
        mapper.add_frame(dataset[frame_id])
        progress.set_postfix(points=mapper.n_points, refresh=False)
    frame_loop_sec = time.perf_counter() - frame_loop_start

    save_timings = mapper.save(
        output_dir,
        {
            "dataset_name": canonical_dataset_name(dataset_name),
            "scene_name": args.scene_name,
            "n_frames": len(dataset),
            "n_points": mapper.n_points,
            "has_normals": True,
            "device": device,
            "map_every": mapper.map_every,
            "downscale_res": args.downscale_res,
            "k_pooling": mapper.k_pooling,
            "max_frame_points": mapper.max_frame_points,
            "match_distance_th": mapper.match_distance_th,
            "clip_model_name": CLIP_MODEL_NAME,
            "clip_pretrained": CLIP_PRETRAINED,
            "clip_load_size": CLIP_LOAD_SIZE,
            "clip_skip_center_crop": True,
            "clip_feature_dim": mapper.clip_extractor.feature_dim,
            "clip_feature_dtype": "float16",
            "clip_feature_path": CLIP_FEATURE_FILE,
            "clip_feature_bytes": mapper.n_points * mapper.clip_extractor.feature_dim * 2,
            "clip_feature_gib": mapper.n_points * mapper.clip_extractor.feature_dim * 2 / 1024**3,
            "rgb_normal_point_fusion": True,
            "clip_feature_mode": "clip_textregion",
            "clip_feature_fusion": False,
        },
    )
    timing_summary = {
        "dataset_load_sec": dataset_load_sec,
        "frame_loop_sec": frame_loop_sec,
        "save": save_timings,
        "total_sec": time.perf_counter() - run_start,
    }
    with open(output_dir / TIMING_PATH, "w") as f:
        json.dump(timing_summary, f, indent=2)
    print(json.dumps({"timing": timing_summary}, indent=2))
    print(output_dir / "rgb_map.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a standalone RGB pointcloud map from RGB-D + GT poses.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet", "replica", "scannet"])
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--output_root", default=str(OUTPUT_DIR))
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--map_every", type=int, default=DEFAULT_MAP_EVERY)
    parser.add_argument("--downscale_res", type=int, default=DEFAULT_DOWNSCALE_RES)
    parser.add_argument("--k_pooling", type=int, default=DEFAULT_K_POOLING)
    parser.add_argument("--max_frame_points", type=int, default=DEFAULT_MAX_FRAME_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)
    parser.add_argument("--use-inst-gt", action="store_true", help="Use decoded ScanNet instance-filt masks instead of SAM for instance evidence.")
    main(parser.parse_args())
