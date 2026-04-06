import argparse
import json
from pathlib import Path
import tempfile

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
DEFAULT_MAP_EVERY = 8
DEFAULT_DOWNSCALE_RES = 2
DEFAULT_K_POOLING = 1
DEFAULT_MAX_FRAME_POINTS = 5_000_000
DEFAULT_MATCH_DISTANCE_TH = 0.03
CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
CLIP_PRETRAINED = "openai"
CLIP_LOAD_SIZE = 1024
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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


class DenseCLIPExtractor:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.model = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
            precision="fp32",
        )[0].eval().to(self.device)
        visual = self.model.visual
        patch_size = visual.patch_size
        self.patch_size = int(patch_size if isinstance(patch_size, int) else patch_size[0])
        self.feature_dim = int(visual.proj.shape[1] if visual.proj is not None else visual.ln_post.normalized_shape[0])

    @torch.inference_mode()
    def extract_dense(self, image: torch.Tensor) -> torch.Tensor:
        image = image.permute(2, 0, 1).float().div_(255.0)
        orig_h, orig_w = image.shape[-2:]
        resized_h, resized_w = resolve_resized_hw(orig_w, orig_h, CLIP_LOAD_SIZE)
        image = F.interpolate(image[None], size=(resized_h, resized_w), mode="bicubic", align_corners=False, antialias=True)[0]
        mean = image.new_tensor(CLIP_MEAN).view(3, 1, 1)
        std = image.new_tensor(CLIP_STD).view(3, 1, 1)
        image = ((image - mean) / std)[None]
        image = pad_to_multiple(image, self.patch_size)

        visual = self.model.visual
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x = visual.conv1(image)
                x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
                cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                x = torch.cat([cls, x], dim=1)
                x = x + interpolate_positional_embedding(visual.positional_embedding, x, self.patch_size, image.shape[-2], image.shape[-1])
                x = visual.ln_pre(x)
                *layers, last_resblock = visual.transformer.resblocks
                if layers:
                    x = torch.nn.Sequential(*layers)(x)
                v_weight = last_resblock.attn.in_proj_weight[-last_resblock.attn.embed_dim :]
                v_bias = last_resblock.attn.in_proj_bias[-last_resblock.attn.embed_dim :]
                v = F.linear(last_resblock.ln_1(x), v_weight, v_bias)
                x = F.linear(v, last_resblock.attn.out_proj.weight, last_resblock.attn.out_proj.bias)
                x = x[:, 1:, :]
                x = visual.ln_post(x)
                if visual.proj is not None:
                    x = x @ visual.proj
        else:
            x = visual.conv1(image)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)
            x = x + interpolate_positional_embedding(visual.positional_embedding, x, self.patch_size, image.shape[-2], image.shape[-1])
            x = visual.ln_pre(x)
            *layers, last_resblock = visual.transformer.resblocks
            if layers:
                x = torch.nn.Sequential(*layers)(x)
            v_weight = last_resblock.attn.in_proj_weight[-last_resblock.attn.embed_dim :]
            v_bias = last_resblock.attn.in_proj_bias[-last_resblock.attn.embed_dim :]
            v = F.linear(last_resblock.ln_1(x), v_weight, v_bias)
            x = F.linear(v, last_resblock.attn.out_proj.weight, last_resblock.attn.out_proj.bias)
            x = x[:, 1:, :]
            x = visual.ln_post(x)
            if visual.proj is not None:
                x = x @ visual.proj

        grid_h = image.shape[-2] // self.patch_size
        grid_w = image.shape[-1] // self.patch_size
        x = x.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2).float()
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
    ) -> None:
        self.device = device
        self.cam_intrinsics = torch.tensor(intrinsics.astype(np.float32), device=device)
        self.map_every = max(1, int(map_every))
        self.max_frame_points = as_int(max_frame_points)
        self.match_distance_th = float(match_distance_th)
        self.clip_extractor = DenseCLIPExtractor(device)
        if k_pooling > 1 and k_pooling % 2 == 0:
            raise ValueError("k_pooling must be odd.")

        self.points = torch.empty((0, 3), device=device)
        self.colors = torch.empty((0, 3), device=device, dtype=torch.uint8)
        self.normals = torch.empty((0, 3), device=device)
        self.feature_tmpdir = tempfile.TemporaryDirectory(prefix="rgb_map_feats_")
        self.feature_shards = []
        self.n_features = 0

        if k_pooling > 1:
            pooling = torch.nn.MaxPool2d(kernel_size=k_pooling, stride=1, padding=k_pooling // 2)
            self.pooling = lambda mask: ~(pooling((~mask[None]).float())[0].bool())
        else:
            self.pooling = lambda mask: mask
        self.downscale = (lambda x: x) if downscale_res == 1 else (lambda x: x[::downscale_res, ::downscale_res])

    def should_map_frame(self, frame_id: int) -> bool:
        return frame_id % self.map_every == 0

    def _append_points(self, points: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, features: torch.Tensor) -> None:
        self.points = torch.vstack((self.points, points))
        self.colors = torch.vstack((self.colors, colors))
        self.normals = torch.vstack((self.normals, normals))
        shard_path = Path(self.feature_tmpdir.name) / f"{len(self.feature_shards):06d}.npy"
        np.save(shard_path, features.cpu().numpy().astype(np.float16, copy=False))
        self.feature_shards.append(shard_path)
        self.n_features += int(features.shape[0])

    def add_frame(self, frame_data) -> None:
        frame_id, image, depth, c2w = frame_data[:4]
        if not self.should_map_frame(frame_id):
            return
        if np.isinf(c2w).any() or np.isnan(c2w).any():
            return

        c2w = torch.from_numpy(c2w).to(self.device)
        depth = torch.from_numpy(depth.astype(np.float32)).to(self.device)
        image = torch.from_numpy(image.astype(np.uint8)).to(self.device)
        full_image = image
        h, w = depth.shape
        y, x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
        mask = depth > 0

        if self.points.shape[0] > 0:
            frustum_corners = geometry_utils.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
            frustum_mask = geometry_utils.compute_frustum_point_ids(self.points, frustum_corners, device=self.device)
            if frustum_mask.numel() > 0:
                _, matches = geometry_utils.match_3d_points_to_2d_pixels(
                    depth,
                    torch.linalg.inv(c2w),
                    self.points[frustum_mask],
                    self.cam_intrinsics,
                    self.match_distance_th,
                )
                if matches.numel() > 0:
                    mask[matches[:, 1], matches[:, 0]] = False
            mask = self.pooling(mask)

        if not mask.any().item():
            return

        x, y = self.downscale(x), self.downscale(y)
        depth, mask, image = self.downscale(depth), self.downscale(mask), self.downscale(image)
        normals_cam, normal_valid = compute_normals_from_depth(x, y, depth, self.cam_intrinsics)
        mask = mask & normal_valid
        if not mask.any().item():
            return
        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        colors = image[mask].reshape(-1, 3)
        normals_cam = normals_cam[mask].reshape(-1, 3)
        if depth.shape[0] > self.max_frame_points:
            keep = torch.linspace(0, depth.shape[0] - 1, self.max_frame_points, device=self.device).round().long()
            x, y, depth, colors = x[keep], y[keep], depth[keep], colors[keep]
            normals_cam = normals_cam[keep]

        dense_clip = self.clip_extractor.extract_dense(full_image)
        features = dense_clip[y, x].half()
        x_3d = (x - self.cam_intrinsics[0, 2]) * depth / self.cam_intrinsics[0, 0]
        y_3d = (y - self.cam_intrinsics[1, 2]) * depth / self.cam_intrinsics[1, 1]
        points = torch.stack((x_3d, y_3d, depth, torch.ones_like(depth)), dim=1)
        points = torch.einsum("ij,mj->mi", c2w, points)[:, :3]
        normals = torch.einsum("ij,mj->mi", c2w[:3, :3], normals_cam)
        normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-8)
        self._append_points(points, colors, normals, features)

    def save(self, output_dir: Path, stats: dict) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.colors.cpu().numpy().astype(np.float32) / 255.0)
        pcd.normals = o3d.utility.Vector3dVector(self.normals.cpu().numpy())
        o3d.io.write_point_cloud(str(output_dir / "rgb_map.ply"), pcd)
        clip_feats = np.lib.format.open_memmap(
            output_dir / "clip_feats.npy",
            mode="w+",
            dtype=np.float16,
            shape=(self.n_features, self.clip_extractor.feature_dim),
        )
        offset = 0
        for shard_path in self.feature_shards:
            shard = np.load(shard_path, mmap_mode="r")
            next_offset = offset + shard.shape[0]
            clip_feats[offset:next_offset] = shard
            offset = next_offset
        del clip_feats
        self.feature_tmpdir.cleanup()
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)


def main(args):
    dataset_name = args.dataset_name.lower()
    dataset = load_dataset(dataset_name, args.scene_name, args.frame_limit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapper = RGBMapper(
        intrinsics=dataset.intrinsics,
        device=device,
        map_every=args.map_every,
        downscale_res=args.downscale_res,
        k_pooling=args.k_pooling,
        max_frame_points=args.max_frame_points,
        match_distance_th=args.match_distance_th,
    )

    progress = tqdm(range(len(dataset)), desc=args.scene_name, unit="frame")
    for frame_id in progress:
        mapper.add_frame(dataset[frame_id])
        progress.set_postfix(points=int(mapper.points.shape[0]), refresh=False)

    output_dir = Path(args.output_root) / canonical_dataset_name(dataset_name) / args.scene_name
    mapper.save(
        output_dir,
        {
            "dataset_name": canonical_dataset_name(dataset_name),
            "scene_name": args.scene_name,
            "n_frames": len(dataset),
            "n_points": int(mapper.points.shape[0]),
            "has_normals": True,
            "device": device,
            "map_every": mapper.map_every,
            "downscale_res": args.downscale_res,
            "k_pooling": args.k_pooling,
            "max_frame_points": mapper.max_frame_points,
            "match_distance_th": mapper.match_distance_th,
            "clip_model_name": CLIP_MODEL_NAME,
            "clip_pretrained": CLIP_PRETRAINED,
            "clip_load_size": CLIP_LOAD_SIZE,
            "clip_skip_center_crop": True,
            "clip_feature_dim": mapper.clip_extractor.feature_dim,
            "clip_feature_dtype": "float16",
            "clip_feature_path": "clip_feats.npy",
            "clip_feature_bytes": int(mapper.points.shape[0]) * mapper.clip_extractor.feature_dim * 2,
            "clip_feature_gib": int(mapper.points.shape[0]) * mapper.clip_extractor.feature_dim * 2 / 1024**3,
        },
    )
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
    main(parser.parse_args())
