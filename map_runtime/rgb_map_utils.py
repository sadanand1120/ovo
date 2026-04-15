from __future__ import annotations

import numpy as np
import open_clip
import torch
import torch.nn.functional as F


CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_LOAD_SIZE = 1024
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_GLOBAL_PATCH_THRESHOLD = 0.07


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


def interpolate_positional_embedding(
    positional_embedding: torch.Tensor,
    x: torch.Tensor,
    patch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
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


def compute_normals_from_depth(
    x: torch.Tensor,
    y: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
                cls = self.visual.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0],
                    1,
                    x.shape[-1],
                    dtype=x.dtype,
                    device=x.device,
                )
                x = torch.cat([cls, x], dim=1)
                x = x + interpolate_positional_embedding(
                    self.visual.positional_embedding,
                    x,
                    self.patch_size,
                    image.shape[-2],
                    image.shape[-1],
                )
                x = self.visual.ln_pre(x)
                for block in self.pre_resblocks:
                    x = block(x)
                x_ln = self.last_resblock.ln_1(x)
                qkv = F.linear(x_ln, self.last_resblock.attn.in_proj_weight, self.last_resblock.attn.in_proj_bias)
        else:
            x = self.visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = self.visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0],
                1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat([cls, x], dim=1)
            x = x + interpolate_positional_embedding(
                self.visual.positional_embedding,
                x,
                self.patch_size,
                image.shape[-2],
                image.shape[-1],
            )
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
        mask_weights = remove_global_patches(
            mask_weights,
            baseline.reshape(-1, baseline.shape[-1]),
            CLIP_GLOBAL_PATCH_THRESHOLD,
        )
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
