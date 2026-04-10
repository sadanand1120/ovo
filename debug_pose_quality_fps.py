from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from debug_visualizer import render_scene_video
from map_runtime.scene import CONFIG_DIR, OUTPUT_DIR, canonical_dataset_name, load_dataset_and_slam


def rotation_error_deg(gt_pose: np.ndarray, pred_pose: np.ndarray) -> float:
    relative = gt_pose[:3, :3].T @ pred_pose[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def summarize_pose_metrics(
    gt_poses: dict[int, np.ndarray],
    pred_poses: dict[int, np.ndarray],
    elapsed_s: float,
    total_frames: int,
) -> dict:
    frame_ids = sorted(
        frame_id
        for frame_id in (set(gt_poses) & set(pred_poses))
        if np.isfinite(gt_poses[frame_id]).all() and np.isfinite(pred_poses[frame_id]).all()
    )
    if not frame_ids:
        raise ValueError("No overlapping valid GT and predicted poses")

    trans_err = []
    xz_err = []
    rot_err = []
    for frame_id in frame_ids:
        gt_pose = gt_poses[frame_id]
        pred_pose = pred_poses[frame_id]
        delta = gt_pose[:3, 3] - pred_pose[:3, 3]
        trans_err.append(float(np.linalg.norm(delta)))
        xz_err.append(float(np.linalg.norm(delta[[0, 2]])))
        rot_err.append(rotation_error_deg(gt_pose, pred_pose))

    trans_err = np.asarray(trans_err, dtype=np.float64)
    xz_err = np.asarray(xz_err, dtype=np.float64)
    rot_err = np.asarray(rot_err, dtype=np.float64)

    return {
        "tracked_frames": len(frame_ids),
        "coverage": len(frame_ids) / len(gt_poses),
        "fps": total_frames / max(elapsed_s, 1e-6),
        "mean_trans_m": float(trans_err.mean()),
        "rmse_trans_m": float(np.sqrt(np.mean(trans_err ** 2))),
        "final_trans_m": float(trans_err[-1]),
        "mean_xz_m": float(xz_err.mean()),
        "rmse_xz_m": float(np.sqrt(np.mean(xz_err ** 2))),
        "final_xz_m": float(xz_err[-1]),
        "mean_rot_deg": float(rot_err.mean()),
        "median_rot_deg": float(np.median(rot_err)),
        "final_rot_deg": float(rot_err[-1]),
    }


def run_scene(
    scene: str,
    dataset_name: str,
    slam_module: str,
    output_root: Path,
    config_path: Path,
    frame_limit: int | None,
    disable_loop_closure: bool,
    video_fps: int,
    pred_label: str,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, dataset, slam_backbone = load_dataset_and_slam(
        dataset_name=dataset_name,
        scene_name=scene,
        device=device,
        frame_limit=frame_limit,
        config_path=config_path,
        slam_module=slam_module,
        disable_loop_closure=disable_loop_closure,
    )

    pred_poses = {}
    gt_poses = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for frame_id in range(len(dataset)):
        frame_data = dataset[frame_id]
        gt_pose = frame_data[3]
        if np.isfinite(gt_pose).all():
            gt_poses[frame_id] = gt_pose
        slam_backbone.track_camera(frame_data)
        pred_pose = slam_backbone.get_c2w(frame_id)
        if pred_pose is not None:
            pred_poses[frame_id] = pred_pose.detach().cpu().numpy() if isinstance(pred_pose, torch.Tensor) else np.asarray(pred_pose)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.time() - t0

    scene_output = output_root / scene
    scene_output.mkdir(parents=True, exist_ok=True)
    pose_path = scene_output / "estimated_c2w.npy"
    with open(pose_path, "wb") as handle:
        torch.save(pred_poses, handle)

    metrics = summarize_pose_metrics(gt_poses, pred_poses, elapsed_s, len(dataset))
    with open(scene_output / "summary.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    render_scene_video(
        scene_dir=Path(config["data"]["input_path"]),
        predicted_path=pose_path,
        output_path=output_root / "debug_videos" / f"{scene}.mp4",
        fps=video_fps,
        panel_size=900,
        margin=70,
        pred_label=pred_label,
    )

    del slam_backbone
    torch.cuda.empty_cache()
    return metrics


def aggregate_metrics(scene_metrics: dict[str, dict]) -> dict:
    keys = [key for key in next(iter(scene_metrics.values())).keys() if key not in {"tracked_frames"}]
    summary = {}
    for key in keys:
        summary[key] = float(np.mean([metrics[key] for metrics in scene_metrics.values()]))
    summary["tracked_frames"] = int(sum(metrics["tracked_frames"] for metrics in scene_metrics.values()))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracker-only backend benchmark with pose metrics and debug video export.")
    parser.add_argument("--dataset_name", default="ScanNet", choices=["Replica", "ScanNet"])
    parser.add_argument("--slam_module", choices=["vanilla", "orbslam", "cuvslam"], required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--config_path", type=Path, default=CONFIG_DIR / "ovo.yaml")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--disable_loop_closure", action="store_true")
    parser.add_argument("--video_fps", type=int, default=4)
    parser.add_argument("--pred_label", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or OUTPUT_DIR / "pose_debug" / canonical_dataset_name(args.dataset_name) / args.slam_module
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pred_label = args.pred_label or args.slam_module
    scene_metrics = {}
    for scene in args.scenes:
        scene_metrics[scene] = run_scene(
            scene=scene,
            dataset_name=args.dataset_name,
            slam_module=args.slam_module,
            output_root=output_root,
            config_path=args.config_path,
            frame_limit=args.frame_limit,
            disable_loop_closure=args.disable_loop_closure,
            video_fps=args.video_fps,
            pred_label=pred_label,
        )
        print(scene, scene_metrics[scene])

    aggregate = aggregate_metrics(scene_metrics)
    with open(output_root / "summary.json", "w") as handle:
        json.dump({"per_scene": scene_metrics, "average": aggregate}, handle, indent=2)
    print("Average", aggregate)


if __name__ == "__main__":
    main()
