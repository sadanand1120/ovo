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

from map_runtime import geometry
from map_runtime.instance_pipeline import (
    SAMInstancePipeline,
)
from map_runtime.scene import (
    canonical_dataset_name,
    get_tracked_pose,
    load_dataset_and_slam,
)


OUTPUT_DIR = Path("data/output/rgb_maps")
TIMING_PATH = "timing.json"
CLIP_FEATURE_FILE = "clip_feats.npy"
DEFAULT_MAP_EVERY = 8
DEFAULT_POINT_SAMPLE_STRIDE = 2
DEFAULT_MAX_FRAME_POINTS = 5_000_000
DEFAULT_MATCH_DISTANCE_TH = 0.03
CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_LOAD_SIZE = 1024
FEATURE_WRITE_CHUNK_SIZE = 200_000
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_GLOBAL_PATCH_THRESHOLD = 0.07


def as_int(value) -> int:
    return int(float(value))


def resolve_resized_hw(width: int, height: int, size: int) -> tuple[int, int]:
    if width <= height:
        return size, max(1, int(round(height * size / width)))
    return max(1, int(round(width * size / height))), size


def stride_sample_2d(x: torch.Tensor, stride: int) -> torch.Tensor:
    if stride == 1:
        return x
    return x[::stride, ::stride]


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
        total_num_frames: int,
        map_every: int,
        point_sample_stride: int,
        max_frame_points: int,
        match_distance_th: float,
    ) -> None:
        self.device = device
        self.cam_intrinsics = torch.tensor(intrinsics.astype(np.float32), device=device)
        self.map_every = max(1, int(map_every))
        self.point_sample_stride = max(1, int(point_sample_stride))
        self.max_frame_points = as_int(max_frame_points)
        self.match_distance_th = float(match_distance_th)
        self.clip_extractor = DenseCLIPExtractor(device)
        self.instance_manager = SAMInstancePipeline(device=device, total_num_frames=total_num_frames)

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

        self.sample_seed_grid = lambda x: stride_sample_2d(x, self.point_sample_stride)

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

    def _get_cached_sampled_grids(self, depth_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cached_depth_shape != depth_shape:
            h, w = depth_shape
            full_y, full_x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
            self.cached_x = self.sample_seed_grid(full_x)
            self.cached_y = self.sample_seed_grid(full_y)
            ds_h, ds_w = self.cached_x.shape
            self.cached_row_ids, self.cached_col_ids = torch.meshgrid(
                torch.arange(ds_h, device=self.device),
                torch.arange(ds_w, device=self.device),
                indexing="ij",
            )
            self.cached_depth_shape = depth_shape
        return self.cached_x, self.cached_y, self.cached_row_ids, self.cached_col_ids

    def _append_points(self, points: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, features: torch.Tensor | None = None) -> None:
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
        if features is None:
            return
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

    def add_frame(self, frame_data, c2w_override=None, is_seed_frame: bool | None = None) -> None:
        frame_id, image_np, depth_np = frame_data[:3]
        if is_seed_frame is None:
            is_seed_frame = self.should_map_frame(frame_id)
        c2w_np = frame_data[3] if c2w_override is None else c2w_override
        if c2w_np is None:
            self.instance_manager.maybe_prune(int(frame_id))
            return
        if isinstance(c2w_np, torch.Tensor):
            c2w = c2w_np.to(self.device, dtype=torch.float32)
            c2w_np = c2w.detach().cpu().numpy()
        else:
            c2w_np = np.asarray(c2w_np, dtype=np.float32)
            c2w = torch.from_numpy(c2w_np).to(self.device)
        if np.isinf(c2w_np).any() or np.isnan(c2w_np).any():
            self.instance_manager.maybe_prune(int(frame_id))
            return

        depth = torch.from_numpy(depth_np).to(self.device)
        image = torch.from_numpy(image_np).to(self.device)
        full_image = image
        h, w = depth.shape
        x, y, row_ids, col_ids = self._get_cached_sampled_grids((h, w))
        mask = depth > 0
        point_ids_full = torch.full((h, w), -1, dtype=torch.int32, device=self.device)
        seed_labels_np = self.instance_manager.extract_seed_labels(image_np) if is_seed_frame else None
        if is_seed_frame:
            tr_labels_np = self.instance_manager.extract_textregion_labels(image_np, seed_labels_np)
            tr_labels_full = torch.from_numpy(tr_labels_np).to(self.device)
        else:
            tr_labels_full = None

        if self.n_points > 0:
            frustum_corners = geometry.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
            w2c = invert_rigid_transform(c2w)
            frustum_mask = geometry.compute_frustum_point_ids(self.points[: self.n_points], frustum_corners, device=self.device)
            if frustum_mask.numel() > 0:
                matched_ids, matches = geometry.match_3d_points_to_2d_pixels(
                    depth,
                    w2c,
                    self.points[frustum_mask],
                    self.cam_intrinsics,
                    self.match_distance_th,
                )
                if matches.numel() > 0:
                    global_ids = frustum_mask[matched_ids].long()
                    point_ids_full[matches[:, 1], matches[:, 0]] = global_ids.to(point_ids_full.dtype)
                    mask[matches[:, 1], matches[:, 0]] = False

        if not is_seed_frame:
            self.instance_manager.process_nonseed_frame(
                int(frame_id),
                image_np,
                point_ids_full.detach().cpu().numpy(),
            )
            return

        depth = self.sample_seed_grid(depth)
        mask = self.sample_seed_grid(mask)
        image = self.sample_seed_grid(image)
        point_ids_sampled = self.sample_seed_grid(point_ids_full)
        normals_cam, normal_valid = compute_normals_from_depth(x, y, depth, self.cam_intrinsics)
        visible_existing = (point_ids_sampled >= 0) & normal_valid
        if visible_existing.any():
            visible_ids = point_ids_sampled[visible_existing]
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

            dense_clip = self.clip_extractor.extract_dense(full_image, tr_labels_full)
            features = dense_clip[y_keep, x_keep].half()
            x_3d = (x_keep - self.cam_intrinsics[0, 2]) * depth_keep / self.cam_intrinsics[0, 0]
            y_3d = (y_keep - self.cam_intrinsics[1, 2]) * depth_keep / self.cam_intrinsics[1, 1]
            points = torch.stack((x_3d, y_3d, depth_keep, torch.ones_like(depth_keep)), dim=1)
            points = torch.einsum("ij,mj->mi", c2w, points)[:, :3]
            normals = torch.einsum("ij,mj->mi", c2w[:3, :3], normals_cam)
            normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-8)
            old_n = self.n_points
            new_ids = torch.arange(old_n, old_n + points.shape[0], device=self.device, dtype=torch.int32)
            self._append_points(points, colors, normals, features)
            self.instance_manager.extend_for_new_points(int(points.shape[0]))
            point_ids_sampled[row_keep, col_keep] = new_ids
        self.instance_manager.process_seed_frame(
            int(frame_id),
            image_np,
            point_ids_full.detach().cpu().numpy(),
            seed_labels_np,
        )

    def save(self, output_dir: Path, stats: dict) -> dict:
        self.instance_manager.close()
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
            with open(output_dir / CLIP_FEATURE_FILE, "wb") as f:
                np.lib.format.write_array_header_2_0(
                    f,
                    {
                        "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                        "fortran_order": False,
                        "shape": (self.n_points, self.clip_extractor.feature_dim),
                    },
                )
                self.feature_tmp_file.flush()
                self.feature_tmp_file.close()
                with open(self.feature_tmp_path, "rb") as src:
                    shutil.copyfileobj(src, f, length=16 * 1024 * 1024)
            timings["store_clip_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("instance labels", refresh=True)
            stage_start = time.perf_counter()
            instance_stats = self.instance_manager.save_outputs(output_dir, final_frame_id=self.instance_manager.total_num_frames - 1)
            timings["instance_labels_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("stats", refresh=True)
            stage_start = time.perf_counter()
            sam_amg_config = self.instance_manager.seed_mask_extractor.amg_config
            stats = {
                **stats,
                "instance_supervision": "sam",
                "textregion_supervision": "sam",
                **instance_stats,
                "clip_feature_path": CLIP_FEATURE_FILE,
                "clip_feature_storage": "npy",
                "rgb_normal_point_fusion": True,
                "clip_feature_mode": "clip_textregion",
                "sam_sort_mode": sam_amg_config.sort_mode,
                "sam_min_mask_area_perc": sam_amg_config.min_mask_area_perc,
                "sam_points_per_side": sam_amg_config.points_per_side,
                "sam_points_per_batch": sam_amg_config.points_per_batch,
                "sam_pred_iou_thresh": sam_amg_config.pred_iou_thresh,
                "sam_stability_score_thresh": sam_amg_config.stability_score_thresh,
                "sam_stability_score_offset": sam_amg_config.stability_score_offset,
                "sam_mask_threshold": sam_amg_config.mask_threshold,
                "sam_box_nms_thresh": sam_amg_config.box_nms_thresh,
                "sam_crop_n_layers": sam_amg_config.crop_n_layers,
                "sam_crop_nms_thresh": sam_amg_config.crop_nms_thresh,
                "sam_crop_overlap_ratio": sam_amg_config.crop_overlap_ratio,
                "sam_crop_n_points_downscale_factor": sam_amg_config.crop_n_points_downscale_factor,
                "sam_min_mask_region_area": sam_amg_config.min_mask_region_area,
                "sam_output_mode": sam_amg_config.output_mode,
                "sam_use_m2m": sam_amg_config.use_m2m,
                "sam_multimask_output": sam_amg_config.multimask_output,
                "sam_amg_extractors_shared": self.instance_manager.shared_amg_extractor,
            }
            if self.instance_manager.textregion_mask_extractor is not None:
                stats.update(
                    {
                        "sam_model_level_textregion": self.instance_manager.textregion_mask_extractor.model_level,
                        "sam_model_type_textregion": self.instance_manager.textregion_mask_extractor.model_type,
                        "sam_checkpoint_path_textregion": str(self.instance_manager.textregion_mask_extractor.checkpoint_path),
                        "sam_config_textregion": self.instance_manager.textregion_mask_extractor.config_path,
                    }
                )
            stats.update(
                {
                    "sam_model_level_inst": self.instance_manager.seed_mask_extractor.model_level,
                    "sam_model_type_inst": self.instance_manager.seed_mask_extractor.model_type,
                    "sam_checkpoint_path_inst": str(self.instance_manager.seed_mask_extractor.checkpoint_path),
                    "sam_config_inst": self.instance_manager.seed_mask_extractor.config_path,
                }
            )
            with open(output_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            timings["stats_sec"] = time.perf_counter() - stage_start
            progress.update()
        finally:
            progress.set_postfix_str("stats", refresh=True)
            progress.close()
            if self.feature_tmp_file is not None and not self.feature_tmp_file.closed:
                self.feature_tmp_file.close()
            if self.feature_tmpdir is not None:
                shutil.rmtree(self.feature_tmpdir, ignore_errors=True)
        timings["save_total_sec"] = time.perf_counter() - save_start
        return timings


def build_run_stats(
    *,
    mapper: RGBMapper,
    config: dict,
    dataset_name: str,
    scene_name: str,
    device: str,
    n_frames: int,
    point_sample_stride: int,
) -> dict:
    return {
        "dataset_name": canonical_dataset_name(dataset_name),
        "scene_name": scene_name,
        "n_frames": n_frames,
        "n_points": mapper.n_points,
        "has_normals": True,
        "device": device,
        "slam_module": config["slam"].get("slam_module", "vanilla"),
        "slam_close_loops": bool(config["slam"].get("close_loops", True)),
        "map_every": mapper.map_every,
        "point_sample_stride": point_sample_stride,
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
    }


def run_scene_build(
    *,
    dataset_name: str,
    scene_name: str,
    output_root: str | Path,
    frame_limit: int | None,
    slam_module: str | None,
    disable_loop_closure: bool,
    config_path: str,
    map_every: int,
    point_sample_stride: int,
    max_frame_points: int,
    match_distance_th: float,
    extra_stats: dict | None = None,
    snapshot_hook=None,
) -> tuple[Path, dict, dict]:
    run_start = time.perf_counter()
    output_dir = Path(output_root) / canonical_dataset_name(dataset_name) / scene_name
    dataset_load_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, dataset, slam_backbone = load_dataset_and_slam(
        dataset_name=dataset_name,
        scene_name=scene_name,
        device=device,
        frame_limit=frame_limit,
        config_path=config_path,
        slam_module=slam_module,
        disable_loop_closure=disable_loop_closure,
    )
    dataset_load_sec = time.perf_counter() - dataset_load_start
    sample_image = dataset[0][1]
    source_height, source_width = sample_image.shape[:2]
    mapper = RGBMapper(
        intrinsics=dataset.intrinsics,
        device=device,
        total_num_frames=len(dataset),
        map_every=map_every,
        point_sample_stride=point_sample_stride,
        max_frame_points=max_frame_points,
        match_distance_th=match_distance_th,
    )

    progress = tqdm(range(len(dataset)), desc=scene_name, unit="frame")
    frame_loop_start = time.perf_counter()
    try:
        for frame_id in progress:
            frame_data = dataset[frame_id]
            prev_n = mapper.n_points
            estimated_c2w = get_tracked_pose(slam_backbone, frame_data)
            mapper.add_frame(frame_data, c2w_override=estimated_c2w)
            if snapshot_hook is not None:
                snapshot_hook(frame_id, prev_n, mapper.n_points, estimated_c2w)
            progress.set_postfix(
                points=mapper.n_points,
                active=mapper.instance_manager.num_active_instances(),
                objs=mapper.instance_manager.num_existing_instances(),
                refresh=False,
            )
    finally:
        progress.close()
    frame_loop_sec = time.perf_counter() - frame_loop_start

    stats = build_run_stats(
        mapper=mapper,
        config=config,
        dataset_name=dataset_name,
        scene_name=scene_name,
        device=device,
        n_frames=len(dataset),
        point_sample_stride=point_sample_stride,
    )
    if extra_stats:
        stats.update(extra_stats)
    save_timings = mapper.save(
        output_dir,
        stats,
    )
    timing_summary = {
        "dataset_load_sec": dataset_load_sec,
        "frame_loop_sec": frame_loop_sec,
        "save": save_timings,
        "total_sec": time.perf_counter() - run_start,
    }
    with open(output_dir / TIMING_PATH, "w") as f:
        json.dump(timing_summary, f, indent=2)
    del slam_backbone
    return output_dir, timing_summary, {
        "dataset_intrinsics": dataset.intrinsics.astype(np.float32, copy=True),
        "source_width": int(source_width),
        "source_height": int(source_height),
    }


def add_build_args(parser: argparse.ArgumentParser, *, default_output_root: str | Path, default_map_every: int) -> None:
    parser.add_argument("--output_root", default=str(default_output_root))
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--slam_module", type=str, default=None, help="Override slam backend, e.g. vanilla, orbslam, or cuvslam.")
    parser.add_argument("--disable_loop_closure", action="store_true", help="Disable ORB-SLAM loop closure/global BA updates by forcing slam.close_loops=false.")
    parser.add_argument("--config_path", type=str, default="configs/ovo.yaml", help="Base runtime config file to load.")
    parser.add_argument("--map_every", type=int, default=default_map_every)
    parser.add_argument("--point_sample_stride", type=int, default=DEFAULT_POINT_SAMPLE_STRIDE, help="Seed-frame point-sampling stride used for geometry/normal/label sampling before point fusion.")
    parser.add_argument("--max_frame_points", type=int, default=DEFAULT_MAX_FRAME_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)


def main(args):
    output_dir, timing_summary, _ = run_scene_build(
        dataset_name=args.dataset_name,
        scene_name=args.scene_name,
        output_root=args.output_root,
        frame_limit=args.frame_limit,
        slam_module=args.slam_module,
        disable_loop_closure=args.disable_loop_closure,
        config_path=args.config_path,
        map_every=args.map_every,
        point_sample_stride=args.point_sample_stride,
        max_frame_points=args.max_frame_points,
        match_distance_th=args.match_distance_th,
    )
    print(json.dumps({"timing": timing_summary}, indent=2))
    print(output_dir / "rgb_map.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a standalone RGB pointcloud map from RGB-D using the selected SLAM pose backend.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet"])
    parser.add_argument("--scene_name", required=True)
    add_build_args(parser, default_output_root=OUTPUT_DIR, default_map_every=DEFAULT_MAP_EVERY)
    parsed = parser.parse_args()
    main(parsed)
