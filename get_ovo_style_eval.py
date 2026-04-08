import argparse
import gc
import json
import time
from copy import deepcopy
from pathlib import Path

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
    load_dataset,
)
from get_metrics_map import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FEATURE_PROB_TH,
    build_confusion,
    confusion_to_metrics,
    encode_class_texts,
    load_dataset_info,
    load_pred_map,
    load_scannet_gt,
    map_gt_labels_to_eval_ids,
    round_for_print,
)
from ovo import eval_utils
from visualize_rgb_map import resolve_instance_labels


OUTPUT_DIR = Path("data/output/ovo_style_eval")
FEATURE_TEXT_TEMPLATE = "This is a photo of a {}"
FEATURE_SOFTMAX_TEMP = 0.01


def classify_instance_features_ovo_style(
    clip_features: np.ndarray,
    pred_instance_labels: np.ndarray,
    class_names: list[str],
    feature_prob_th: float,
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    valid = pred_instance_labels >= 0
    if not valid.any():
        return np.empty((0,), dtype=np.int32), {
            "num_instances": 0,
            "mean_points_per_instance": 0.0,
            "mean_max_prob": float("nan"),
        }

    instance_ids = pred_instance_labels[valid]
    num_instances = int(instance_ids.max()) + 1
    feat_dim = clip_features.shape[1]
    feature_sum = np.zeros((num_instances, feat_dim), dtype=np.float32)
    feature_count = np.zeros((num_instances,), dtype=np.int64)

    for start in tqdm(
        range(0, pred_instance_labels.shape[0], chunk_size),
        desc="instance feature agg",
        unit="chunk",
        leave=False,
        dynamic_ncols=True,
    ):
        end = min(start + chunk_size, pred_instance_labels.shape[0])
        labels_chunk = pred_instance_labels[start:end]
        valid_chunk = labels_chunk >= 0
        if not valid_chunk.any():
            continue
        feats_chunk = np.array(clip_features[start:end], copy=True, dtype=np.float32)
        np.nan_to_num(feats_chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.add.at(feature_sum, labels_chunk[valid_chunk], feats_chunk[valid_chunk])
        np.add.at(feature_count, labels_chunk[valid_chunk], 1)

    keep = feature_count > 0
    instance_ids_kept = np.flatnonzero(keep).astype(np.int32)
    instance_desc = feature_sum[keep] / feature_count[keep, None].clip(min=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_embeds = encode_class_texts(class_names, device)
    desc = torch.from_numpy(instance_desc).to(device).float()
    desc = desc / desc.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    logits = (desc @ text_embeds.T) / FEATURE_SOFTMAX_TEMP
    probs = logits.softmax(dim=-1)
    max_prob, pred_class = probs.max(dim=-1)
    pred_class = pred_class.cpu().numpy().astype(np.int32)
    max_prob = max_prob.cpu().numpy().astype(np.float32)

    if feature_prob_th > 0 and "background" in class_names:
        background_idx = int(class_names.index("background"))
        pred_class[max_prob < feature_prob_th] = background_idx

    full_pred_class = np.full((num_instances,), -1, dtype=np.int32)
    full_pred_class[instance_ids_kept] = pred_class
    diagnostics = {
        "num_instances": int(instance_ids_kept.shape[0]),
        "mean_points_per_instance": float(feature_count[keep].mean()),
        "mean_max_prob": float(max_prob.mean()) if max_prob.size > 0 else float("nan"),
    }
    return full_pred_class, diagnostics


def transfer_semantic_labels_ovo_style(
    pred_points: np.ndarray,
    pred_instance_labels: np.ndarray,
    instance_classes: np.ndarray,
    gt_points: np.ndarray,
) -> tuple[np.ndarray, dict]:
    valid = pred_instance_labels >= 0
    if not valid.any():
        return np.full((gt_points.shape[0],), -1, dtype=np.int32), {
            "matched_instance_count": 0,
            "assigned_gt_vertices": 0,
        }

    mesh_instance_labels, _, matched_instance_ids = eval_utils.match_labels_to_vtx(
        torch.from_numpy(pred_instance_labels[valid].astype(np.int64)),
        torch.from_numpy(pred_points[valid].astype(np.float32)),
        torch.from_numpy(gt_points.astype(np.float32)),
        filter_unasigned=True,
        tree="kd",
        verbose=False,
    )
    mesh_instance_labels = mesh_instance_labels.cpu().numpy().astype(np.int32, copy=False)
    mesh_semantic_labels = instance_classes[mesh_instance_labels]
    diagnostics = {
        "matched_instance_count": int(matched_instance_ids.shape[0]),
        "assigned_gt_vertices": int(mesh_semantic_labels.shape[0]),
    }
    return mesh_semantic_labels, diagnostics


def build_scene(scene_name: str, args: argparse.Namespace) -> tuple[Path, dict]:
    dataset_name = args.dataset_name.lower()
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
    dataset = load_dataset(dataset_name, scene_name, args.frame_limit)
    dataset_load_sec = time.perf_counter() - dataset_load_start
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    )

    progress = tqdm(range(len(dataset)), desc=scene_name, unit="frame", dynamic_ncols=True)
    frame_loop_start = time.perf_counter()
    for frame_id in progress:
        if not mapper.should_map_frame(frame_id):
            continue
        mapper.add_frame(dataset[frame_id])
        progress.set_postfix(points=mapper.n_points, refresh=False)
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
    return output_dir, timing_summary


def evaluate_scene_ovo_style(scene_name: str, run_dir: Path, raw_root: Path, dataset_info: dict, feature_prob_th: float, chunk_size: int, min_component_size: int) -> tuple[dict, np.ndarray, dict]:
    gt = load_scannet_gt(scene_name, raw_root)
    pred = load_pred_map(run_dir / "rgb_map.ply")
    pred_instance_labels = resolve_instance_labels(run_dir, pred["points"].shape[0], min_component_size)
    gt_semantic = map_gt_labels_to_eval_ids(gt["semantic_raw"], dataset_info)
    class_names = dataset_info.get("class_names_reduced", dataset_info.get("class_names"))

    instance_classes, diag_1 = classify_instance_features_ovo_style(
        pred["clip_features"],
        pred_instance_labels,
        class_names,
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
    dataset_name = args.dataset_name.lower()
    if dataset_name != "scannet":
        raise NotImplementedError("get_ovo_style_eval.py currently supports ScanNet only.")

    dataset_info = deepcopy(load_dataset_info(dataset_name))
    if args.ignore_background:
        dataset_info["ignore"] = dataset_info.get("ignore", []).copy() + dataset_info.get("background_reduced_ids", [])
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
    parser.add_argument("--dataset_name", required=True, choices=["ScanNet", "scannet"])
    parser.add_argument("--output_root", default=str(OUTPUT_DIR))
    parser.add_argument("--scannet_raw_root", required=True, help="ScanNet raw scans root containing aggregation and segs files.")
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional override scene list. Defaults to the 5 HVS scenes from configs/scannet_eval.yaml.")
    parser.add_argument("--map_every", type=int, default=DEFAULT_MAP_EVERY)
    parser.add_argument("--downscale_res", type=int, default=DEFAULT_DOWNSCALE_RES)
    parser.add_argument("--k_pooling", type=int, default=DEFAULT_K_POOLING)
    parser.add_argument("--max_frame_points", type=int, default=DEFAULT_MAX_FRAME_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)
    parser.add_argument("--use-inst-gt", action="store_true")
    parser.add_argument("--feature_prob_th", type=float, default=DEFAULT_FEATURE_PROB_TH)
    parser.add_argument("--min_component_size", type=int, default=2000)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--ignore_background", action="store_true")
    parser.add_argument("--save_json", action="store_true")
    main(parser.parse_args())
