import argparse
import gc
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from build_rgb_map import (
    DEFAULT_MAX_FRAME_POINTS,
    DEFAULT_POINT_SAMPLE_STRIDE,
    run_scene_build,
)
from get_metrics_map import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MATCH_DISTANCE_TH,
    DEFAULT_OVO_SCORE_TH,
    FEATURE_TEXT_TEMPLATE,
    INPUT_DIR,
    OVO_FEATURE_AGG,
    OVO_TEXT_TEMPLATE,
    build_confusion,
    canonical_dataset_name,
    classify_instance_features_ovo_style,
    confusion_to_metrics,
    encode_class_texts,
    load_dataset_info,
    load_gt,
    load_pred_map,
    map_gt_labels_to_eval_ids,
    round_for_print,
    transfer_semantic_labels_ovo_style,
)

OVO_EVAL_OUTPUT_DIR = Path("data/output/ovo_style_eval")
PAPER_MAP_EVERY = 10


def format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}" if np.isfinite(value) else "nan"


def infer_paper_method_label(slam_module: str | None) -> str:
    if (slam_module or "vanilla").lower() == "vanilla":
        return "OVO-mapping (ours) \u2020"
    return "OVO-SLAM (ours)"


def render_replica_paper_table(method_label: str, metrics: dict) -> str:
    rows = [
        ["", "All", "", "Head", "", "Common", "", "Tail", ""],
        ["Method", "mIoU", "mAcc", "mIoU", "mAcc", "mIoU", "mAcc", "mIoU", "mAcc"],
        [
            method_label,
            format_percent(metrics["mIoU"]),
            format_percent(metrics["mAcc"]),
            format_percent(metrics["iou_head"]),
            format_percent(metrics["acc_head"]),
            format_percent(metrics["iou_comm"]),
            format_percent(metrics["acc_comm"]),
            format_percent(metrics["iou_tail"]),
            format_percent(metrics["acc_tail"]),
        ],
    ]
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows)


def render_scannet_paper_table(method_label: str, metrics: dict) -> str:
    rows = [
        ["", "ScanNet20", "", "", ""],
        ["Method", "mIoU", "mAcc", "f-mIoU", "f-mAcc"],
        [
            method_label,
            format_percent(metrics["mIoU"]),
            format_percent(metrics["mAcc"]),
            format_percent(metrics["f_mIoU"]),
            format_percent(metrics["f_mAcc"]),
        ],
    ]
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(rows[0]))]
    return "\n".join("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows)


def render_paper_table(dataset_name: str, slam_module: str | None, metrics: dict) -> str:
    method_label = infer_paper_method_label(slam_module)
    if dataset_name == "Replica":
        title = "Table 2 style: Evaluation on Replica with the 51 most common labels"
        body = render_replica_paper_table(method_label, metrics)
    else:
        title = "Table 4 style: Quantitative results on ScanNetv2"
        body = render_scannet_paper_table(method_label, metrics)
    notes = ["\u2020 Uses GT camera poses.", "\u2021 Uses GT camera poses and 3D geometry."]
    return "\n".join([title, body, *notes])


def evaluate_scene_ovo_style(
    dataset_name: str,
    scene_name: str,
    run_dir: Path,
    dataset_info: dict,
    text_embeds: torch.Tensor,
    ovo_score_th: float,
    chunk_size: int,
    min_component_size: int,
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray, dict]:
    from visualize_rgb_map import resolve_instance_labels

    gt = load_gt(dataset_name, scene_name, args)
    pred = load_pred_map(run_dir / "rgb_map.ply")
    pred_instance_labels = resolve_instance_labels(run_dir, pred["points"].shape[0], min_component_size)
    gt_semantic = map_gt_labels_to_eval_ids(gt["semantic_raw"], dataset_info)
    instance_classes, _, diag_1 = classify_instance_features_ovo_style(
        pred["clip_features"],
        pred_instance_labels,
        text_embeds,
        ovo_score_th,
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


def run_dataset_ovo_style_eval(args: argparse.Namespace) -> None:
    dataset_name = args.dataset_name
    dataset_info = deepcopy(load_dataset_info(dataset_name))
    if args.ignore_background:
        dataset_info["ignore"] = dataset_info.get("ignore", []).copy() + dataset_info.get("background_reduced_ids", [])
    class_names = dataset_info.get("class_names_reduced", dataset_info.get("class_names"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_embeds = encode_class_texts(class_names, device, template=OVO_TEXT_TEMPLATE)
    scenes = args.scenes or dataset_info["scenes"]
    per_scene_rows = []
    confusion_sum = np.zeros((dataset_info["num_classes"], dataset_info["num_classes"]), dtype=np.ulonglong)
    diagnostics = {}

    for scene_name in scenes:
        output_dir, timing_summary, _ = run_scene_build(
            dataset_name=dataset_name,
            scene_name=scene_name,
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
        metrics, confusion, diag = evaluate_scene_ovo_style(
            dataset_name,
            scene_name,
            output_dir,
            dataset_info,
            text_embeds,
            args.ovo_score_th,
            args.chunk_size,
            args.min_component_size,
            args,
        )
        confusion_sum += confusion
        per_scene_rows.append(
            {
                "scene": scene_name,
                "mIoU": metrics["mIoU"],
                "mAcc": metrics["mAcc"],
                "f_mIoU": metrics["f_mIoU"],
                "f_mAcc": metrics["f_mAcc"],
                "build_sec": timing_summary["total_sec"],
            }
        )
        diagnostics[scene_name] = {
            "timing": timing_summary,
            "semantic_ovo_style": diag,
        }
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_metrics = confusion_to_metrics(confusion_sum, dataset_info)
    summary = {
        "dataset_name": canonical_dataset_name(dataset_name),
        "scenes": scenes,
        "ovo_score_th": float(args.ovo_score_th),
        "feature_text_template": FEATURE_TEXT_TEMPLATE,
        "ovo_text_template": OVO_TEXT_TEMPLATE,
        "ovo_feature_agg": OVO_FEATURE_AGG,
        "min_component_size": int(args.min_component_size),
        "metrics_per_scene": per_scene_rows,
        "metrics_all": total_metrics,
        "paper_method_label": infer_paper_method_label(args.slam_module),
        "diagnostics": diagnostics,
    }
    if dataset_name == "ScanNet":
        summary["scannet_raw_root"] = str(Path(args.scannet_raw_root).resolve()) if args.scannet_raw_root else None
    elif dataset_name == "Replica":
        replica_root = Path(args.replica_root) if args.replica_root else INPUT_DIR / "Replica"
        summary["replica_root"] = str(replica_root.resolve())

    print(render_paper_table(dataset_name, args.slam_module, total_metrics))
    print()
    print(json.dumps(round_for_print(summary), indent=2))

    if args.save_json:
        out_root = Path(args.output_root) / canonical_dataset_name(dataset_name)
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "ovo_style_eval_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scenes and report dataset-level OVO-style semantic metrics.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet"])
    parser.add_argument("--output_root", default=str(OVO_EVAL_OUTPUT_DIR))
    parser.add_argument("--scannet_raw_root", default=None, help="ScanNet raw scans root containing aggregation and segs files.")
    parser.add_argument("--replica_root", default=None, help="Replica root containing semantic_gt/ and *_mesh.ply files. Defaults to data/input/Replica.")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--slam_module", type=str, default=None, help="Override slam backend, e.g. vanilla, orbslam, or cuvslam.")
    parser.add_argument("--disable_loop_closure", action="store_true", help="Disable ORB-SLAM loop closure/global BA updates by forcing slam.close_loops=false.")
    parser.add_argument("--config_path", type=str, default="configs/ovo.yaml", help="Base runtime config file to load.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional dataset-eval scene override. Defaults to the dataset scenes from the eval config.")
    parser.add_argument("--map_every", type=int, default=PAPER_MAP_EVERY)
    parser.add_argument("--point_sample_stride", type=int, default=DEFAULT_POINT_SAMPLE_STRIDE, help="Seed-frame point-sampling stride used during map construction.")
    parser.add_argument("--max_frame_points", type=int, default=DEFAULT_MAX_FRAME_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)
    parser.add_argument("--ovo_score_th", type=float, default=DEFAULT_OVO_SCORE_TH)
    parser.add_argument("--min_component_size", type=int, default=2000)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--ignore_background", action="store_true")
    parser.add_argument("--save_json", action="store_true")
    run_dataset_ovo_style_eval(parser.parse_args())


if __name__ == "__main__":
    main()
