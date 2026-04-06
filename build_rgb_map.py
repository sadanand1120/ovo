import argparse
import json
from pathlib import Path

import cv2  # Keep OpenCV loaded before torch in the container env.
import numpy as np
import open3d as o3d
import torch
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
        if k_pooling > 1 and k_pooling % 2 == 0:
            raise ValueError("k_pooling must be odd.")

        self.points = torch.empty((0, 3), device=device)
        self.colors = torch.empty((0, 3), device=device, dtype=torch.uint8)

        if k_pooling > 1:
            pooling = torch.nn.MaxPool2d(kernel_size=k_pooling, stride=1, padding=k_pooling // 2)
            self.pooling = lambda mask: ~(pooling((~mask[None]).float())[0].bool())
        else:
            self.pooling = lambda mask: mask
        self.downscale = (lambda x: x) if downscale_res == 1 else (lambda x: x[::downscale_res, ::downscale_res])

    def should_map_frame(self, frame_id: int) -> bool:
        return frame_id % self.map_every == 0

    def _append_points(self, points: torch.Tensor, colors: torch.Tensor) -> None:
        self.points = torch.vstack((self.points, points))
        self.colors = torch.vstack((self.colors, colors))

    def add_frame(self, frame_data) -> None:
        frame_id, image, depth, c2w = frame_data[:4]
        if not self.should_map_frame(frame_id):
            return
        if np.isinf(c2w).any() or np.isnan(c2w).any():
            return

        c2w = torch.from_numpy(c2w).to(self.device)
        depth = torch.from_numpy(depth.astype(np.float32)).to(self.device)
        image = torch.from_numpy(image.astype(np.uint8)).to(self.device)
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
        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        colors = image[mask].reshape(-1, 3)
        if depth.shape[0] > self.max_frame_points:
            keep = torch.linspace(0, depth.shape[0] - 1, self.max_frame_points, device=self.device).round().long()
            x, y, depth, colors = x[keep], y[keep], depth[keep], colors[keep]

        x_3d = (x - self.cam_intrinsics[0, 2]) * depth / self.cam_intrinsics[0, 0]
        y_3d = (y - self.cam_intrinsics[1, 2]) * depth / self.cam_intrinsics[1, 1]
        points = torch.stack((x_3d, y_3d, depth, torch.ones_like(depth)), dim=1)
        points = torch.einsum("ij,mj->mi", c2w, points)[:, :3]
        self._append_points(points, colors)

    def save(self, output_dir: Path, stats: dict) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.colors.cpu().numpy().astype(np.float32) / 255.0)
        o3d.io.write_point_cloud(str(output_dir / "rgb_map.ply"), pcd)
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
            "device": device,
            "map_every": mapper.map_every,
            "downscale_res": args.downscale_res,
            "k_pooling": args.k_pooling,
            "max_frame_points": mapper.max_frame_points,
            "match_distance_th": mapper.match_distance_th,
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
