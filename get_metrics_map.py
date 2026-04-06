import argparse
import json
import math
from pathlib import Path

import cv2  # Keep OpenCV loaded before torch in the container env.
import imageio.v3 as iio
import numpy as np
import open3d as o3d
import torch
from tqdm.auto import tqdm

from ovo import io_utils
from ovo.datasets import get_dataset


DATASET_DIRS = {"replica": "Replica", "scannet": "ScanNet"}
CONFIG_DIR = Path("configs")
INPUT_DIR = Path("data/input")


def canonical_dataset_name(dataset_name: str) -> str:
    return DATASET_DIRS[dataset_name.lower()]


def resolve_ply_path(input_path: str) -> Path:
    path = Path(input_path)
    return path if path.suffix == ".ply" else path / "rgb_map.ply"


def infer_scene_info_from_path(ply_path: Path):
    scene_name = ply_path.parent.name
    dataset_name = ply_path.parent.parent.name
    if dataset_name.lower() not in DATASET_DIRS:
        raise ValueError(f"Could not infer dataset from path: {ply_path}")
    return dataset_name, scene_name


def load_dataset(dataset_name: str, scene_name: str, frame_limit: int | None):
    config = io_utils.load_config(CONFIG_DIR / "ovo.yaml")
    io_utils.update_recursive(config, io_utils.load_config(CONFIG_DIR / f"{dataset_name.lower()}.yaml"))
    dataset_cfg = {
        **config["cam"],
        "input_path": str(INPUT_DIR / canonical_dataset_name(dataset_name) / scene_name),
        "frame_limit": config["data"].get("frame_limit", -1) if frame_limit is None else frame_limit,
    }
    return get_dataset(dataset_name.lower())(dataset_cfg)


def choose_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_valid_c2w(c2w: np.ndarray) -> bool:
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        return False
    return abs(np.linalg.det(c2w[:3, :3])) > 1e-8


def render_pointcloud_view(points: np.ndarray, colors: np.ndarray, intrinsics: np.ndarray, c2w: np.ndarray, height: int, width: int):
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    cam = np.hstack((points.astype(np.float32), ones)) @ w2c.T
    z = cam[:, 2]
    valid = z > 0
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width), dtype=bool)

    cam = cam[valid]
    z = z[valid]
    colors = colors[valid]

    u = np.rint(intrinsics[0, 0] * cam[:, 0] / z + intrinsics[0, 2]).astype(np.int32)
    v = np.rint(intrinsics[1, 1] * cam[:, 1] / z + intrinsics[1, 2]).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8), np.zeros((height, width), dtype=bool)

    u = u[valid]
    v = v[valid]
    z = z[valid]
    colors = colors[valid]

    flat = v * width + u
    order = np.argsort(z)
    first = np.unique(flat[order], return_index=True)[1]
    chosen = order[first]

    render = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=bool)
    render[v[chosen], u[chosen]] = colors[chosen]
    mask[v[chosen], u[chosen]] = True
    return render, mask


def render_pointcloud_view_torch(points_h: torch.Tensor, colors: torch.Tensor, intrinsics: torch.Tensor, c2w: torch.Tensor, height: int, width: int):
    w2c = torch.linalg.inv(c2w)
    cam = points_h @ w2c.T
    z = cam[:, 2]
    valid = z > 0
    if not valid.any():
        return torch.zeros((height, width, 3), dtype=torch.uint8, device=points_h.device), torch.zeros((height, width), dtype=torch.bool, device=points_h.device)

    cam = cam[valid]
    z = z[valid]
    colors = colors[valid]

    u = torch.round(intrinsics[0, 0] * cam[:, 0] / z + intrinsics[0, 2]).long()
    v = torch.round(intrinsics[1, 1] * cam[:, 1] / z + intrinsics[1, 2]).long()
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not valid.any():
        return torch.zeros((height, width, 3), dtype=torch.uint8, device=points_h.device), torch.zeros((height, width), dtype=torch.bool, device=points_h.device)

    z = z[valid]
    colors = colors[valid]
    flat = (v[valid] * width + u[valid]).long()

    order_z = torch.argsort(z, stable=True)
    flat = flat[order_z]
    colors = colors[order_z]
    order_flat = torch.argsort(flat, stable=True)
    flat = flat[order_flat]
    colors = colors[order_flat]

    keep = torch.ones(flat.shape[0], dtype=torch.bool, device=points_h.device)
    keep[1:] = flat[1:] != flat[:-1]
    flat = flat[keep]
    colors = colors[keep]

    render_flat = torch.zeros((height * width, 3), dtype=torch.uint8, device=points_h.device)
    mask_flat = torch.zeros((height * width,), dtype=torch.bool, device=points_h.device)
    render_flat[flat] = colors
    mask_flat[flat] = True
    return render_flat.view(height, width, 3), mask_flat.view(height, width)


def compute_psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    pred = pred[mask].astype(np.float32) / 255.0
    gt = gt[mask].astype(np.float32) / 255.0
    mse = np.mean((pred - gt) ** 2)
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def compute_psnr_torch(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    pred = pred[mask].float() / 255.0
    gt = gt[mask].float() / 255.0
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def main(args):
    ply_path = resolve_ply_path(args.input_path)
    dataset_name, scene_name = infer_scene_info_from_path(ply_path)
    dataset_name = dataset_name.lower()
    dataset = load_dataset(dataset_name, scene_name, args.frame_limit)
    device = choose_device()

    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.rint(np.asarray(pcd.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    points_h = None
    colors_t = None
    intrinsics_t = None
    if device == "cuda":
        points_h = torch.from_numpy(np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))).to(device)
        colors_t = torch.from_numpy(colors).to(device)
        intrinsics_t = torch.from_numpy(dataset.intrinsics.astype(np.float32)).to(device)

    psnr_values = []
    coverages = []
    skipped_frames = 0
    saved_frames = set()
    save_dir = None
    if args.save_pngs:
        sample_count = min(5, len(dataset))
        sample_ids = np.random.default_rng(0).choice(len(dataset), size=sample_count, replace=False)
        saved_frames = set(sample_ids.tolist())
        save_dir = ply_path.with_name("psnr_views")
        save_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(range(len(dataset)), desc=f"PSNR {scene_name}", unit="frame")
    for frame_id in progress:
        frame_data = dataset[frame_id]
        gt = frame_data[1]
        c2w = frame_data[3]
        if not is_valid_c2w(c2w):
            skipped_frames += 1
            progress.set_postfix(psnr="skip", cov="skip", refresh=False)
            continue
        if device == "cuda":
            gt_t = torch.from_numpy(gt.astype(np.uint8)).to(device)
            c2w_t = torch.from_numpy(c2w.astype(np.float32)).to(device)
            render_t, mask_t = render_pointcloud_view_torch(points_h, colors_t, intrinsics_t, c2w_t, gt.shape[0], gt.shape[1])
            psnr = compute_psnr_torch(render_t, gt_t, mask_t)
            coverage = float(mask_t.float().mean().item())
            render = render_t.detach().cpu().numpy() if frame_id in saved_frames else None
        else:
            render, mask = render_pointcloud_view(points, colors, dataset.intrinsics, c2w, gt.shape[0], gt.shape[1])
            psnr = compute_psnr(render, gt, mask)
            coverage = float(mask.mean())
        if not math.isnan(psnr):
            psnr_values.append(psnr)
        coverages.append(coverage)
        if frame_id in saved_frames:
            side_by_side = np.concatenate((gt, render), axis=1)
            iio.imwrite(save_dir / f"{frame_id:04d}.png", side_by_side)
        progress.set_postfix(psnr=f"{psnr:.2f}" if not math.isnan(psnr) else "nan", cov=f"{coverage:.3f}", refresh=False)

    summary = {
        "dataset_name": canonical_dataset_name(dataset_name),
        "scene_name": scene_name,
        "device": device,
        "n_eval_frames": len(dataset),
        "n_skipped_frames": skipped_frames,
        "n_points": int(points.shape[0]),
        "mean_psnr": float(np.mean(psnr_values)) if psnr_values else float("nan"),
        "mean_coverage": float(np.mean(coverages)) if coverages else 0.0,
    }
    print(json.dumps(summary, indent=2))

    if args.save_json:
        out_path = ply_path.with_name("metrics_psnr.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(out_path)
    if save_dir is not None:
        print(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RGB map PSNR by rendering saved pointcloud views back into dataset cameras.")
    parser.add_argument("input_path", help="Path to rgb_map.ply or its containing directory.")
    parser.add_argument("--frame_limit", type=int, default=None, help="Override number of frames to evaluate.")
    parser.add_argument("--save_json", action="store_true", help="Save summary to metrics_psnr.json next to the map.")
    parser.add_argument("--save-pngs", action="store_true", help="Save 5 sampled rendered-vs-GT side-by-side PNGs.")
    main(parser.parse_args())
