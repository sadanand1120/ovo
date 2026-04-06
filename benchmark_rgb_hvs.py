import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ovo import io_utils


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
DEFAULT_OUTPUT_ROOT = ROOT / "data/output/rgb_hvs_benchmark"
DEFAULT_MAP_EVERY = 8
DEFAULT_DOWNSCALE_RES = 2
DEFAULT_K_POOLING = 1
DEFAULT_MAX_FRAME_POINTS = 5_000_000
DEFAULT_MATCH_DISTANCE_TH = 0.03


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_scene_specs(scene_specs: list[str] | None):
    if not scene_specs:
        return [(scene, None) for scene in io_utils.load_config(CONFIG_DIR / "scannet_eval.yaml")["scenes"]]
    parsed = []
    for spec in scene_specs:
        scene_name, _, frame_limit = spec.partition(":")
        parsed.append((scene_name, int(frame_limit) if frame_limit else None))
    return parsed


def print_summary(rows: list[dict]) -> None:
    print("\nPer-scene metrics")
    for row in rows:
        print(
            f"{row['scene_name']}: "
            f"PSNR={row['mean_psnr']:.3f}, "
            f"coverage={row['mean_coverage']:.3f}, "
            f"points={row['n_points']}, "
            f"build_s={row['build_seconds']:.1f}, "
            f"metric_s={row['metrics_seconds']:.1f}"
        )

    print(
        "\nMean over 5 HVS scenes: "
        f"PSNR={np.mean([row['mean_psnr'] for row in rows]):.3f}, "
        f"coverage={np.mean([row['mean_coverage'] for row in rows]):.3f}, "
        f"build_s={np.mean([row['build_seconds'] for row in rows]):.1f}, "
        f"metric_s={np.mean([row['metrics_seconds'] for row in rows]):.1f}"
    )


def main(args):
    scene_specs = parse_scene_specs(args.scenes)
    output_root = Path(args.output_root)
    rows = []

    for scene_name, scene_frame_limit in scene_specs:
        frame_limit = args.frame_limit if args.frame_limit is not None else scene_frame_limit
        build_cmd = [
            sys.executable,
            "build_rgb_map.py",
            "--dataset_name",
            "ScanNet",
            "--scene_name",
            scene_name,
            "--output_root",
            str(output_root),
            "--max_frame_points",
            str(args.max_frame_points),
            "--match_distance_th",
            str(args.match_distance_th),
        ]
        for key in ("map_every", "downscale_res", "k_pooling"):
            value = getattr(args, key)
            if value is not None:
                build_cmd.extend([f"--{key}", str(value)])
        if frame_limit is not None:
            build_cmd.extend(["--frame_limit", str(frame_limit)])
        metrics_cmd = [
            sys.executable,
            "get_metrics_map.py",
            str(output_root / "ScanNet" / scene_name),
            "--save_json",
        ]
        if args.save_pngs:
            metrics_cmd.append("--save-pngs")
        if frame_limit is not None:
            metrics_cmd.extend(["--frame_limit", str(frame_limit)])

        t0 = time.perf_counter()
        run_command(build_cmd)
        build_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        run_command(metrics_cmd)
        metrics_seconds = time.perf_counter() - t0

        with open(output_root / "ScanNet" / scene_name / "metrics_psnr.json", "r") as f:
            row = json.load(f)
        row["build_seconds"] = build_seconds
        row["metrics_seconds"] = metrics_seconds
        rows.append(row)

    print_summary(rows)
    if args.save_json:
        out_path = output_root / "ScanNet" / "benchmark_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "dataset_name": "ScanNet",
                    "scenes": [scene for scene, _ in scene_specs],
                    "max_frame_points": args.max_frame_points,
                    "map_every": args.map_every,
                    "downscale_res": args.downscale_res,
                    "k_pooling": args.k_pooling,
                    "match_distance_th": args.match_distance_th,
                    "per_scene": rows,
                    "mean_psnr": float(np.mean([row["mean_psnr"] for row in rows])),
                    "mean_coverage": float(np.mean([row["mean_coverage"] for row in rows])),
                    "mean_build_seconds": float(np.mean([row["build_seconds"] for row in rows])),
                    "mean_metrics_seconds": float(np.mean([row["metrics_seconds"] for row in rows])),
                },
                f,
                indent=2,
            )
        print(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RGB-map build + PSNR benchmark on the 5 ScanNet HVS scenes.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene specs like scene0011_00 or scene0050_00:200.")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--map_every", type=int, default=DEFAULT_MAP_EVERY)
    parser.add_argument("--downscale_res", type=int, default=DEFAULT_DOWNSCALE_RES)
    parser.add_argument("--k_pooling", type=int, default=DEFAULT_K_POOLING)
    parser.add_argument("--max_frame_points", type=int, default=DEFAULT_MAX_FRAME_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)
    parser.add_argument("--save-pngs", action="store_true")
    parser.add_argument("--save_json", action="store_true")
    main(parser.parse_args())
