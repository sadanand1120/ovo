"""Dataset loaders used by the custom RGB-map workflow."""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        resize_ratio = dataset_config.get("resize_ratio", 1.0)
        self.height = int(dataset_config["H"] * resize_ratio)
        self.width = int(dataset_config["W"] * resize_ratio)
        self.fx = dataset_config["fx"] * resize_ratio
        self.fy = dataset_config["fy"] * resize_ratio
        self.cx = dataset_config["cx"] * resize_ratio
        self.cy = dataset_config["cy"] * resize_ratio

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(dataset_config["distortion"]) if "distortion" in dataset_config else None
        self.crop_edge = dataset_config["crop_edge"] if "crop_edge" in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class Replica(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as handle:
            lines = handle.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]


class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "color").glob("*.jpg")),
            key=lambda path: int(os.path.basename(path)[:-4]),
        )
        self.depth_paths = sorted(
            list((self.dataset_path / "depth").glob("*.png")),
            key=lambda path: int(os.path.basename(path)[:-4]),
        )
        self.load_poses(self.dataset_path / "pose")
        depth_th = dataset_config.get("depth_th", 0)
        self.depth_th = depth_th if depth_th > 0 else None

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(path.glob("*.txt"), key=lambda pose_path: int(os.path.basename(pose_path)[:-4]))
        for pose_path in pose_paths:
            with open(pose_path, "r") as handle:
                lines = handle.readlines()
            rows = [list(map(float, line.split(" "))) for line in lines]
            c2w = np.array(rows).reshape(4, 4).astype(np.float32)
            self.poses.append(c2w)

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        lr_color_data = cv2.resize(color_data, (self.dataset_config["W"], self.dataset_config["H"]))

        depth_data = cv2.imread(str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        if self.depth_th is not None:
            depth_data[depth_data > self.depth_th] = 0
        edge = self.crop_edge
        if edge > 0:
            lr_color_data = lr_color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        return index, lr_color_data, depth_data, self.poses[index], color_data


def get_dataset(dataset_name: str):
    if dataset_name == "Replica":
        return Replica
    if dataset_name == "ScanNet":
        return ScanNet
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
