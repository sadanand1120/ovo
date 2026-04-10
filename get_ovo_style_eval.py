import argparse
import gc
import json
import time
from copy import deepcopy
from pathlib import Path

import cv2  # Keep OpenCV loaded before torch in this env.
import numpy as np
import torch
from tqdm.auto import tqdm

from build_rgb_map import (
    CLIP_FEATURE_FILE,
    CLIP_LOAD_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DEFAULT_DOWNSCALE_RES,
    DEFAULT_K_POOLING,
    DEFAULT_MAP_EVERY,
    DEFAULT_MATCH_DISTANCE_TH,
    DEFAULT_MAX_FRAME_POINTS,
    RGBMapper,
    TIMING_PATH,
    canonical_dataset_name,
    get_tracked_pose,
    load_dataset_and_slam,
)
from get_metrics_map import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FEATURE_PROB_TH,
    FEATURE_TEXT_TEMPLATE,
    OVO_FEATURE_AGG,
    build_confusion,
    classify_instance_features_ovo_style,
    confusion_to_metrics,
    encode_class_texts,
    load_dataset_info,
    load_pred_map,
    load_scannet_gt,
    map_gt_labels_to_eval_ids,
    round_for_print,
    transfer_semantic_labels_ovo_style,
)
from visualize_rgb_map import resolve_instance_labels


OUTPUT_DIR = Path("data/output/ovo_style_eval")


def build_scene(scene_name: str, args: argparse.Namespace) -> tuple[Path, dict]:
    dataset_name = args.dataset_name
    output_dir = Path(args.output_root) / canonical_dataset_name(dataset_name) / scene_name
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil
                shutil.rmtree(child)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    dataset_load_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, dataset, slam_backbone = load_dataset_and_slam(
        dataset_name=dataset_name,
        scene_name=scene_name,
        device=device,
        frame_limit=args.frame_limit,
        config_path=args.config_path,
        slam_module=args.slam_module,
        disable_loop_closure=args.disable_loop_closure,
    )
    dataset_load_sec = time.perf_counter() - dataset_load_start
    mapper = RGBMapper(
        intrinsics=dataset.intrinsics,
        device=device,
        map_every=args.map_every,
        downscale_res=args.downscale_res,
        k_pooling=args.k_pooling,
        max_frame_points=args.max_frame_points,
        match_distance_th=args.match_distance_th,
        dataset_name=dataset_name,
        scene_name=scene_name,
        use_inst_gt=args.use_inst_gt,
        sam_model_level_inst=args.sam_model_level_inst,
        sam_model_level_textregion=args.sam_model_level_textregion,
        sam2_model_level_track=args.sam2_model_level_track,
    )

    progress = tqdm(range(len(dataset)), desc=scene_name, unit="frame", dynamic_ncols=True)
    frame_loop_start = time.perf_counter()
    for frame_id in progress:
        frame_data = dataset[frame_id]
        estimated_c2w = get_tracked_pose(slam_backbone, frame_data)
        mapper.add_frame(frame_data, c2w_override=estimated_c2w)
        progress.set_postfix(
            points=mapper.n_points,
            active=mapper.instance_manager.num_active_instances(),
            objs=mapper.instance_manager.num_existing_instances(),
            refresh=False,
        )
    frame_loop_sec = time.perf_counter() - frame_loop_start

    save_timings = mapper.save(
        output_dir,
        {
            "dataset_name": canonical_dataset_name(dataset_name),
            "scene_name": scene_name,
            "n_frames": len(dataset),
            "n_points": mapper.n_points,
            "has_normals": True,
            "device": device,
            "slam_module": config["slam"].get("slam_module", "vanilla"),
            "slam_close_loops": bool(config["slam"].get("close_loops", True)),
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
            "clip_feature_path": CLIP_FEATURE_FILE,
            "clip_feature_bytes": mapper.n_points * mapper.clip_extractor.feature_dim * 2,
            "clip_feature_gib": mapper.n_points * mapper.clip_extractor.feature_dim * 2 / 1024**3,
            "clip_feature_mode": "clip_textregion",
            "rgb_normal_point_fusion": True,
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
    del slam_backbone
    return output_dir, timing_summary


def evaluate_scene_ovo_style(
    scene_name: str,
    run_dir: Path,
    raw_root: Path,
    dataset_info: dict,
    text_embeds: torch.Tensor,
    background_idx: int | None,
    feature_prob_th: float,
    chunk_size: int,
    min_component_size: int,
) -> tuple[dict, np.ndarray, dict]:
    gt = load_scannet_gt(scene_name, raw_root)
    pred = load_pred_map(run_dir / "rgb_map.ply")
    pred_instance_labels = resolve_instance_labels(run_dir, pred["points"].shape[0], min_component_size)
    gt_semantic = map_gt_labels_to_eval_ids(gt["semantic_raw"], dataset_info)

    instance_classes, diag_1 = classify_instance_features_ovo_style(
        pred["clip_features"],
        pred_instance_labels,
        text_embeds,
        background_idx,
        feature_prob_th,
        chunk_size,
    )
    pred_semantic_labels, diag_2 = transfer_semantic_labels_ovo_style(
        pred["points"],
        pred_instance_labels,
        instance_classes,
        gt["points"],
    )
    confusion = build_confusion(gt_semantic, pred_semantic_labels, dataset_info["num_classes"], dataset_info.get("ignore", []))
    metrics = confusion_to_metrics(confusion, dataset_info)
    return metrics, confusion, {**diag_1, **diag_2}


def render_table(rows: list[dict], total_metrics: dict) -> str:
    headers = ["scene", "mIoU", "mAcc", "f-mIoU", "f-mAcc", "build_s"]
    table_rows = [
        [
            row["scene"],
            f"{row['mIoU']:.3f}",
            f"{row['mAcc']:.3f}",
            f"{row['f_mIoU']:.3f}",
            f"{row['f_mAcc']:.3f}",
            f"{row['build_sec']:.1f}",
        ]
        for row in rows
    ]
    table_rows.append(
        [
            "ALL_5_HVS",
            f"{total_metrics['mIoU']:.3f}",
            f"{total_metrics['mAcc']:.3f}",
            f"{total_metrics['f_mIoU']:.3f}",
            f"{total_metrics['f_mAcc']:.3f}",
            "-",
        ]
    )
    widths = [max(len(header), *(len(row[i]) for row in table_rows)) for i, header in enumerate(headers)]
    lines = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in table_rows:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    dataset_name = args.dataset_name
    if dataset_name != "ScanNet":
        raise NotImplementedError("get_ovo_style_eval.py currently supports ScanNet only.")

    dataset_info = deepcopy(load_dataset_info(dataset_name))
    if args.ignore_background:
        dataset_info["ignore"] = dataset_info.get("ignore", []).copy() + dataset_info.get("background_reduced_ids", [])
    class_names = dataset_info.get("class_names_reduced", dataset_info.get("class_names"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_embeds = encode_class_texts(class_names, device)
    background_idx = class_names.index("background") if "background" in class_names else None
    scenes = args.scenes or dataset_info["scenes"]
    raw_root = Path(args.scannet_raw_root)

    per_scene_rows = []
    confusion_sum = np.zeros((dataset_info["num_classes"], dataset_info["num_classes"]), dtype=np.ulonglong)
    diagnostics = {}

    for scene_name in scenes:
        run_dir, timing = build_scene(scene_name, args)
        metrics, confusion, diag = evaluate_scene_ovo_style(
            scene_name,
            run_dir,
            raw_root,
            dataset_info,
            text_embeds,
            background_idx,
            args.feature_prob_th,
            args.chunk_size,
            args.min_component_size,
        )
        confusion_sum += confusion
        per_scene_rows.append(
            {
                "scene": scene_name,
                "mIoU": metrics["mIoU"],
                "mAcc": metrics["mAcc"],
                "f_mIoU": metrics["f_mIoU"],
                "f_mAcc": metrics["f_mAcc"],
                "build_sec": timing["total_sec"],
            }
        )
        diagnostics[scene_name] = {
            "timing": timing,
            "semantic_ovo_style": diag,
        }
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_metrics = confusion_to_metrics(confusion_sum, dataset_info)
    summary = {
        "dataset_name": canonical_dataset_name(dataset_name),
        "scenes": scenes,
        "use_inst_gt": bool(args.use_inst_gt),
        "feature_prob_th": float(args.feature_prob_th),
        "feature_text_template": FEATURE_TEXT_TEMPLATE,
        "ovo_feature_agg": OVO_FEATURE_AGG,
        "min_component_size": int(args.min_component_size),
        "metrics_per_scene": per_scene_rows,
        "metrics_all_5_hvs": total_metrics,
        "diagnostics": diagnostics,
    }

    print(render_table(per_scene_rows, total_metrics))
    print()
    print(json.dumps(round_for_print(summary), indent=2))

    if args.save_json:
        out_root = Path(args.output_root) / canonical_dataset_name(dataset_name)
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "ovo_style_eval_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the 5 ScanNet HVS scenes and report OVO-style semantic metrics directly comparable to the paper table.")
    parser.add_argument("--dataset_name", required=True, choices=["ScanNet"])
    parser.add_argument("--output_root", default=str(OUTPUT_DIR))
    parser.add_argument("--scannet_raw_root", required=True, help="ScanNet raw scans root containing aggregation and segs files.")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--slam_module", type=str, default=None, help="Override slam backend, e.g. vanilla, orbslam, or cuvslam.")
    parser.add_argument("--disable_loop_closure", action="store_true", help="Disable ORB-SLAM loop closure/global BA updates by forcing slam.close_loops=false.")
    parser.add_argument("--config_path", type=str, default="configs/ovo.yaml", help="Base runtime config file to load.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional override scene list. Defaults to the 5 HVS scenes from configs/scannet_eval.yaml.")
    parser.add_argument("--map_every", type=int, default=DEFAULT_MAP_EVERY)
    parser.add_argument("--downscale_res", type=int, default=DEFAULT_DOWNSCALE_RES)
    parser.add_argument("--k_pooling", type=int, default=DEFAULT_K_POOLING)
    parser.add_argument("--max_frame_points", type=int, default=DEFAULT_MAX_FRAME_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)
    parser.add_argument("--sam-model-level-inst", type=int, choices=[11, 12, 13], default=13)
    parser.add_argument("--sam-model-level-textregion", type=int, choices=[11, 12, 13], default=13)
    parser.add_argument("--sam2-model-level-track", type=int, choices=[21, 22, 23, 24], default=24)
    parser.add_argument("--use-inst-gt", action="store_true")
    parser.add_argument("--feature_prob_th", type=float, default=DEFAULT_FEATURE_PROB_TH)
    parser.add_argument("--min_component_size", type=int, default=2000)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--ignore_background", action="store_true")
    parser.add_argument("--save_json", action="store_true")
    main(parser.parse_args())
