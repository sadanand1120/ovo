from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import load_config, update_recursive
from .datasets import get_dataset
from .slam_backends import get_slam_backbone


DATASET_CONFIG_NAMES = {
    "Replica": "replica",
    "ScanNet": "scannet",
}
CONFIG_DIR = Path("configs")
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")


def canonical_dataset_name(dataset_name: str) -> str:
    if dataset_name not in DATASET_CONFIG_NAMES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset_name


def build_scene_config(
    scene: str,
    dataset: str,
    config_path: str | Path = CONFIG_DIR / "ovo.yaml",
    slam_module: str | None = None,
    frame_limit: int | None = None,
    disable_loop_closure: bool = False,
) -> dict:
    config = load_config(config_path)
    config.setdefault("data", {})
    config.setdefault("slam", {})
    if slam_module is not None:
        config["slam"]["slam_module"] = slam_module
    if disable_loop_closure:
        config["slam"]["close_loops"] = False

    dataset_cfg = load_config(CONFIG_DIR / f"{DATASET_CONFIG_NAMES[dataset]}.yaml")
    update_recursive(config, dataset_cfg)

    config["data"]["scene_name"] = scene
    config["data"]["input_path"] = str(INPUT_DIR / canonical_dataset_name(dataset) / scene)
    if frame_limit is not None:
        config["data"]["frame_limit"] = frame_limit
    return config


def load_dataset(
    dataset_name: str,
    scene_name: str,
    frame_limit: int | None = None,
    config_path: str | Path = CONFIG_DIR / "ovo.yaml",
    slam_module: str | None = None,
    disable_loop_closure: bool = False,
):
    config = build_scene_config(
        scene=scene_name,
        dataset=dataset_name,
        config_path=config_path,
        slam_module=slam_module,
        frame_limit=frame_limit,
        disable_loop_closure=disable_loop_closure,
    )
    dataset = get_dataset(dataset_name)({**config["data"], **config["cam"]})
    return config, dataset


def load_dataset_and_slam(
    dataset_name: str,
    scene_name: str,
    device: str,
    frame_limit: int | None = None,
    config_path: str | Path = CONFIG_DIR / "ovo.yaml",
    slam_module: str | None = None,
    disable_loop_closure: bool = False,
):
    config, dataset = load_dataset(
        dataset_name=dataset_name,
        scene_name=scene_name,
        frame_limit=frame_limit,
        config_path=config_path,
        slam_module=slam_module,
        disable_loop_closure=disable_loop_closure,
    )
    config["device"] = device
    cam_intrinsics = torch.tensor(dataset.intrinsics.astype(np.float32), device=device)
    slam_backbone = get_slam_backbone(config, dataset, cam_intrinsics)
    return config, dataset, slam_backbone


def get_tracked_pose(slam_backbone, frame_data):
    slam_backbone.track_camera(frame_data)
    if not np.any(frame_data[2] > 0):
        return None
    return slam_backbone.get_c2w(int(frame_data[0]))
