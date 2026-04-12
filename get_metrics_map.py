import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import open_clip
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
from tqdm.auto import tqdm

from map_runtime.config import load_config
from map_runtime.metrics_utils import compute_instance_ap_dataset, iou_acc_from_confmat


DATASET_CONFIG_NAMES = {"Replica": "replica", "ScanNet": "scannet"}
CONFIG_DIR = Path("configs")
INPUT_DIR = Path("data/input")
TIMING_PATH = "timing.json"
STATS_PATH = "stats.json"
CLIP_FEATURE_FILE = "clip_feats.npy"
CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
CLIP_PRETRAINED = "openai"
FEATURE_TEXT_TEMPLATE = "{}"
FEATURE_SOFTMAX_TEMP = 0.01
OVO_TEXT_TEMPLATE = "This is a photo of a {}"
DEFAULT_MATCH_DISTANCE_TH = 0.03
DEFAULT_FEATURE_PROB_TH = 0.0
DEFAULT_OVO_SCORE_TH = 0.0
DEFAULT_CHUNK_SIZE = 100_000
FEATURE_INSTANCE_AGG = "mean_probs"
OVO_FEATURE_AGG = "mean_l2norm_point_features_then_cosine"
GEOM_THRESHOLDS = (0.01, 0.03, 0.05)
INSTANCE_AP_THRESHOLDS = (0.25, 0.5)
METRIC_ROW_ORDER = (
    ("geometry", "chamfer_l1_m"),
    ("geometry", "fscore_3cm"),
    ("geometry", "coverage"),
    ("rgb", "psnr"),
    ("normals", "mean_angle_deg"),
    ("feature", "mIoU"),
    ("feature", "mAcc"),
    ("semantic_ovo_style", "mIoU"),
    ("semantic_ovo_style", "mAcc"),
    ("semantic_ovo_style", "f_mIoU"),
    ("semantic_ovo_style", "f_mAcc"),
    ("instance", "ap"),
    ("instance", "ap_25"),
    ("instance", "ap_50"),
)
TIMING_ROW_ORDER = (
    ("timing.total_sec", ("total_sec",)),
    ("timing.dataset_load_sec", ("dataset_load_sec",)),
    ("timing.frame_loop_sec", ("frame_loop_sec",)),
    ("timing.save.save_total_sec", ("save", "save_total_sec")),
    ("timing.save.write_ply_sec", ("save", "write_ply_sec")),
    ("timing.save.store_clip_sec", ("save", "store_clip_sec")),
    ("timing.save.instance_labels_sec", ("save", "instance_labels_sec")),
    ("timing.save.stats_sec", ("save", "stats_sec")),
)


def canonical_dataset_name(dataset_name: str) -> str:
    if dataset_name not in DATASET_CONFIG_NAMES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset_name


def resolve_ply_path(input_path: str) -> Path:
    path = Path(input_path)
    return path if path.suffix == ".ply" else path / "rgb_map.ply"


def load_feature_model_spec(map_dir: Path) -> tuple[str, str]:
    stats_path = map_dir / STATS_PATH
    if not stats_path.exists():
        return CLIP_MODEL_NAME, CLIP_PRETRAINED
    stats = json.loads(stats_path.read_text())
    return str(stats.get("clip_model_name", CLIP_MODEL_NAME)), str(stats.get("clip_pretrained", CLIP_PRETRAINED))


def load_openclip_text_model(device: str, model_name: str, pretrained: str):
    if pretrained.startswith("hf-hub:"):
        model = open_clip.create_model_from_pretrained(pretrained)[0].eval().to(device)
        tokenizer = open_clip.get_tokenizer(pretrained)
    else:
        model = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)[0].eval()
        tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


def infer_scene_info_from_path(ply_path: Path) -> tuple[str, str]:
    scene_name = ply_path.parent.name
    dataset_name = ply_path.parent.parent.name
    if dataset_name not in DATASET_CONFIG_NAMES:
        raise ValueError(f"Could not infer dataset from path: {ply_path}")
    return dataset_name, scene_name


def load_dataset_info(dataset_name: str) -> dict:
    return load_config(CONFIG_DIR / f"{DATASET_CONFIG_NAMES[dataset_name]}_eval.yaml")


def read_label_txt(path: Path) -> np.ndarray:
    return np.array(path.read_text().splitlines(), dtype=np.int32)


def load_ply_vertices(ply_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    vertex = PlyData.read(str(ply_path))["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    labels = np.asarray(vertex["label"], dtype=np.int32) if "label" in vertex.dtype.names else None
    normals = None
    if {"nx", "ny", "nz"}.issubset(vertex.dtype.names):
        normals = np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
    return points, colors, labels, normals


def resolve_scannet_gt_paths(scene_name: str, raw_root: str | Path) -> tuple[Path, Path, Path]:
    scene_input = INPUT_DIR / "ScanNet" / scene_name
    scan_root = Path(raw_root) / scene_name
    labels_ply = scene_input / f"{scene_name}_vh_clean_2.labels.ply"
    if not labels_ply.exists():
        labels_ply = scan_root / f"{scene_name}_vh_clean_2.labels.ply"
    seg_path = scan_root / f"{scene_name}_vh_clean_2.0.010000.segs.json"
    agg_path = scan_root / f"{scene_name}.aggregation.json"
    if not labels_ply.exists():
        raise ValueError(f"Missing ScanNet labels mesh: {labels_ply}")
    if not seg_path.exists() or not agg_path.exists():
        raise ValueError(f"Missing ScanNet instance GT files under {scan_root}")
    return labels_ply, seg_path, agg_path


def load_scannet_gt(scene_name: str, raw_root: str | Path) -> dict:
    labels_ply, seg_path, agg_path = resolve_scannet_gt_paths(scene_name, raw_root)
    points, colors, semantic_raw, _ = load_ply_vertices(labels_ply)
    normals_path = INPUT_DIR / "ScanNet" / scene_name / f"{scene_name}_vh_clean_2.vertex_normals.npy"
    if not normals_path.exists():
        raise FileNotFoundError(
            f"Missing decoded GT normals: {normals_path}. Re-run scannet_decode_sens.py for {scene_name}."
        )
    normals = np.load(normals_path)

    seg_indices = np.asarray(json.loads(seg_path.read_text())["segIndices"], dtype=np.int32)
    agg = json.loads(agg_path.read_text())
    instance_labels = np.full(seg_indices.shape[0], -1, dtype=np.int32)
    for inst_id, group in enumerate(agg["segGroups"]):
        instance_labels[np.isin(seg_indices, group["segments"])] = inst_id

    return {
        "points": points,
        "colors": colors,
        "normals": normals,
        "semantic_raw": semantic_raw,
        "instance_labels": instance_labels,
    }


def resolve_replica_gt_paths(scene_name: str, replica_root: str | Path | None) -> tuple[Path, Path, Path]:
    root = Path(replica_root) if replica_root is not None else INPUT_DIR / "Replica"
    semantic_path = root / "semantic_gt" / f"{scene_name}.txt"
    mesh_path = root / f"{scene_name}_mesh.ply"
    habitat_mesh_path = root / scene_name / "habitat" / "mesh_semantic.ply"
    if not semantic_path.exists():
        raise ValueError(f"Missing Replica semantic GT labels: {semantic_path}")
    if not mesh_path.exists():
        raise ValueError(f"Missing Replica mesh: {mesh_path}")
    if not habitat_mesh_path.exists():
        raise ValueError(f"Missing Replica habitat semantic mesh: {habitat_mesh_path}")
    return semantic_path, mesh_path, habitat_mesh_path


def project_face_labels_to_vertices(face_vertices, face_labels: np.ndarray, n_vertices: int) -> np.ndarray:
    lengths = np.fromiter((len(v) for v in face_vertices), dtype=np.int32, count=len(face_vertices))
    vertex_indices = np.concatenate(face_vertices).astype(np.int64)
    repeated_labels = np.repeat(face_labels.astype(np.int64), lengths)
    pairs = np.stack([vertex_indices, repeated_labels], axis=1)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    order = np.lexsort((-counts, unique_pairs[:, 0]))
    ordered_pairs = unique_pairs[order]
    first_per_vertex = np.ones(ordered_pairs.shape[0], dtype=bool)
    first_per_vertex[1:] = ordered_pairs[1:, 0] != ordered_pairs[:-1, 0]
    selected = ordered_pairs[first_per_vertex]
    labels = np.full(n_vertices, -1, dtype=np.int32)
    labels[selected[:, 0].astype(np.int64)] = selected[:, 1].astype(np.int32)
    return labels


def load_replica_vertex_instance_labels(habitat_mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    points, _, _, _ = load_ply_vertices(habitat_mesh_path)
    ply = PlyData.read(str(habitat_mesh_path))
    face = ply["face"].data
    if "object_id" not in face.dtype.names:
        raise ValueError(f"Replica habitat semantic mesh missing face field 'object_id': {habitat_mesh_path}")
    face_vertices = face["vertex_indices"]
    face_object_ids = np.asarray(face["object_id"], dtype=np.int32)
    vertex_instance_labels = project_face_labels_to_vertices(face_vertices, face_object_ids, points.shape[0])
    return points, vertex_instance_labels


def load_replica_gt(scene_name: str, replica_root: str | Path | None) -> dict:
    semantic_path, mesh_path, habitat_mesh_path = resolve_replica_gt_paths(scene_name, replica_root)
    points, colors, _, normals = load_ply_vertices(mesh_path)
    semantic_raw = read_label_txt(semantic_path)
    if semantic_raw.shape[0] != points.shape[0]:
        raise ValueError(
            f"Replica semantic GT size mismatch for {scene_name}: "
            f"{semantic_raw.shape[0]} labels vs {points.shape[0]} mesh vertices."
        )
    habitat_points, habitat_instance_labels = load_replica_vertex_instance_labels(habitat_mesh_path)
    if habitat_points.shape == points.shape and np.allclose(habitat_points, points):
        instance_labels = habitat_instance_labels
    else:
        _, nn_idx = cKDTree(habitat_points).query(points, k=1, workers=-1)
        instance_labels = habitat_instance_labels[np.asarray(nn_idx, dtype=np.int64)]
    return {
        "points": points,
        "colors": colors,
        "normals": normals,
        "semantic_raw": semantic_raw,
        "instance_labels": instance_labels,
    }


def load_gt(dataset_name: str, scene_name: str, args: argparse.Namespace) -> dict:
    if dataset_name == "ScanNet":
        if args.scannet_raw_root is None:
            raise ValueError("--scannet_raw_root is required for ScanNet metrics.")
        return load_scannet_gt(scene_name, Path(args.scannet_raw_root))
    if dataset_name == "Replica":
        return load_replica_gt(scene_name, args.replica_root)
    raise NotImplementedError(f"Unsupported dataset for metrics: {dataset_name}")


def load_pred_map(ply_path: Path) -> dict:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.rint(np.asarray(pcd.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    if points.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")
    if normals.shape[0] != points.shape[0]:
        raise ValueError(f"{ply_path} has no stored normals. Rebuild the map with the current build_rgb_map.py.")
    clip_path = ply_path.with_name(CLIP_FEATURE_FILE)
    if not clip_path.exists():
        raise ValueError(f"Missing CLIP features: {clip_path}")
    instance_label_path = ply_path.with_name("instance_labels.npy")
    if not instance_label_path.exists():
        raise ValueError(f"Missing instance labels for {ply_path.parent}")
    return {
        "points": points,
        "colors": colors,
        "normals": normals,
        "clip_features": np.load(clip_path, mmap_mode="r"),
    }


def compute_nn_associations(gt_points: np.ndarray, pred_points: np.ndarray) -> dict:
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    gt_to_pred_d, gt_to_pred_idx = pred_tree.query(gt_points, k=1, workers=-1)
    pred_to_gt_d, pred_to_gt_idx = gt_tree.query(pred_points, k=1, workers=-1)
    return {
        "gt_to_pred_d": gt_to_pred_d.astype(np.float32),
        "gt_to_pred_idx": gt_to_pred_idx.astype(np.int64),
        "pred_to_gt_d": pred_to_gt_d.astype(np.float32),
        "pred_to_gt_idx": pred_to_gt_idx.astype(np.int64),
    }


def safe_psnr_from_rgb(pred_rgb: np.ndarray, gt_rgb: np.ndarray) -> float:
    pred = pred_rgb.astype(np.float32) / 255.0
    gt = gt_rgb.astype(np.float32) / 255.0
    mse = np.mean((pred - gt) ** 2)
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    out = np.zeros_like(x, dtype=np.float32)
    valid = norm[:, 0] > 1e-8
    out[valid] = x[valid] / norm[valid]
    return out


def map_gt_labels_to_eval_ids(gt_labels_raw: np.ndarray, dataset_info: dict) -> np.ndarray:
    mapping = dataset_info.get("map_to_reduced")
    if mapping is None:
        return gt_labels_raw.astype(np.int32)
    mapped = np.full(gt_labels_raw.shape, -1, dtype=np.int32)
    for src, dst in mapping.items():
        mapped[gt_labels_raw == int(src)] = int(dst)
    return mapped


def build_confusion(gt_labels: np.ndarray, pred_labels: np.ndarray, num_classes: int, ignore: list[int]) -> np.ndarray:
    ignore_set = set(int(x) for x in ignore)
    valid = ~np.isin(gt_labels, list(ignore_set))
    gt = gt_labels[valid].astype(np.int64)
    pred = pred_labels[valid].astype(np.int64)
    pred = np.clip(pred, 0, num_classes - 1)
    flat = gt * num_classes + pred
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes).astype(np.ulonglong)


def confusion_to_metrics(confusion: np.ndarray, dataset_info: dict) -> dict:
    iou_values, iou_valid_mask, weights_values, acc_values, acc_valid_mask = iou_acc_from_confmat(
        confusion,
        dataset_info["num_classes"],
        dataset_info.get("ignore", []).copy(),
        mask_nan=True,
        verbose=False,
        labels=dataset_info.get("class_names_reduced", dataset_info.get("class_names")),
    )
    metrics = {
        "mIoU": float(np.mean(iou_values[iou_valid_mask])) if iou_valid_mask.any() else float("nan"),
        "mAcc": float(np.mean(acc_values[acc_valid_mask])) if acc_valid_mask.any() else float("nan"),
        "f_mIoU": float(np.sum(iou_values[iou_valid_mask] * weights_values[iou_valid_mask]) / weights_values[iou_valid_mask].sum()) if iou_valid_mask.any() else float("nan"),
        "f_mAcc": float(np.sum(acc_values[acc_valid_mask] * weights_values[acc_valid_mask]) / weights_values[acc_valid_mask].sum()) if acc_valid_mask.any() else float("nan"),
    }
    if iou_values.shape[0] == 51:
        thirds = iou_values.shape[0] // 3
        for idx, split in enumerate(("head", "comm", "tail")):
            start = thirds * idx
            end = thirds * (idx + 1)
            split_iou = iou_values[start:end]
            split_acc = acc_values[start:end]
            split_iou_mask = iou_valid_mask[start:end]
            split_acc_mask = acc_valid_mask[start:end]
            metrics[f"iou_{split}"] = float(np.mean(split_iou[split_iou_mask])) if split_iou_mask.any() else float("nan")
            metrics[f"acc_{split}"] = float(np.mean(split_acc[split_acc_mask])) if split_acc_mask.any() else float("nan")
    return metrics


def compute_geometry_metrics(assoc: dict) -> tuple[dict, dict]:
    gt_to_pred = assoc["gt_to_pred_d"]
    pred_to_gt = assoc["pred_to_gt_d"]
    metrics = {
        "chamfer_l1_m": float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean())),
    }
    diagnostics = {}
    for th in GEOM_THRESHOLDS:
        precision = float((pred_to_gt <= th).mean())
        recall = float((gt_to_pred <= th).mean())
        fscore = 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))
        key = f"{int(round(th * 100))}cm"
        if key == "3cm":
            metrics[f"fscore_{key}"] = fscore
            diagnostics[f"precision_{key}"] = precision
            diagnostics[f"recall_{key}"] = recall
    return metrics, diagnostics


def compute_rgb_metrics(gt_colors: np.ndarray, pred_colors: np.ndarray) -> dict:
    return {
        "psnr": safe_psnr_from_rgb(pred_colors, gt_colors),
    }


def compute_normal_metrics(gt_normals: np.ndarray, pred_normals: np.ndarray) -> dict:
    gt_normals = normalize_rows(gt_normals)
    pred_normals = normalize_rows(pred_normals)
    dots = np.abs(np.sum(gt_normals * pred_normals, axis=1)).clip(0.0, 1.0)
    angles = np.degrees(np.arccos(dots))
    return {
        "mean_angle_deg": float(angles.mean()),
    }


@torch.inference_mode()
def encode_class_texts(
    class_names: list[str],
    device: str,
    template: str = FEATURE_TEXT_TEMPLATE,
    *,
    model_name: str = CLIP_MODEL_NAME,
    pretrained: str = CLIP_PRETRAINED,
) -> torch.Tensor:
    model, tokenizer = load_openclip_text_model(device, model_name, pretrained)
    phrases = [template.format(name) for name in class_names]
    text = model.encode_text(tokenizer(phrases).to(device)).float()
    text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return text / text.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def grouped_reduce(labels: np.ndarray, values: np.ndarray, sum_buffer: np.ndarray, count_buffer: np.ndarray) -> None:
    order = np.argsort(labels, kind="stable")
    labels_sorted = labels[order]
    values_sorted = values[order]
    starts = np.flatnonzero(np.r_[True, labels_sorted[1:] != labels_sorted[:-1]])
    unique_labels = labels_sorted[starts]
    sum_buffer[unique_labels] += np.add.reduceat(values_sorted, starts, axis=0)
    count_buffer[unique_labels] += np.diff(np.r_[starts, labels_sorted.shape[0]])


def pool_instance_clip_features(
    clip_features: np.ndarray,
    pred_instance_labels: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid_instances = pred_instance_labels >= 0
    feature_dim = int(clip_features.shape[1])
    if not valid_instances.any():
        return np.zeros((0, feature_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    num_instances = int(pred_instance_labels[valid_instances].max()) + 1
    feature_sum = np.zeros((num_instances, feature_dim), dtype=np.float32)
    feature_count = np.zeros((num_instances,), dtype=np.int64)

    for start in range(0, int(clip_features.shape[0]), chunk_size):
        end = min(start + chunk_size, int(clip_features.shape[0]))
        chunk = np.array(clip_features[start:end], copy=True).astype(np.float32, copy=False)
        chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        chunk = normalize_rows(chunk)
        labels_chunk = pred_instance_labels[start:end]
        valid_chunk = labels_chunk >= 0
        if valid_chunk.any():
            grouped_reduce(labels_chunk[valid_chunk], chunk[valid_chunk], feature_sum, feature_count)

    keep = feature_count > 0
    feature_sum[keep] /= feature_count[keep, None].clip(min=1)
    feature_sum[keep] = normalize_rows(feature_sum[keep])
    return feature_sum, feature_count


def score_pred_points_and_instances(
    clip_features: np.ndarray,
    pred_instance_labels: np.ndarray,
    text_embeds: torch.Tensor,
    background_idx: int | None,
    feature_prob_th: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    device = str(text_embeds.device)
    n_points = int(clip_features.shape[0])
    pred_point_labels = np.empty((n_points,), dtype=np.int32)
    max_point_probs = np.empty((n_points,), dtype=np.float32)
    num_classes = int(text_embeds.shape[0])
    valid_instances = pred_instance_labels >= 0
    if valid_instances.any():
        num_instances = int(pred_instance_labels[valid_instances].max()) + 1
        score_sum = np.zeros((num_instances, num_classes), dtype=np.float32)
        score_count = np.zeros((num_instances,), dtype=np.int64)
    else:
        num_instances = 0
        score_sum = np.zeros((0, num_classes), dtype=np.float32)
        score_count = np.zeros((0,), dtype=np.int64)

    text_embeds_compute = text_embeds.to(device)
    if device == "cuda":
        text_embeds_compute = text_embeds_compute.to(dtype=torch.float16)

    for start in tqdm(
        range(0, n_points, chunk_size),
        desc="semantic score",
        unit="chunk",
        leave=False,
        dynamic_ncols=True,
    ):
        end = min(start + chunk_size, n_points)
        chunk = torch.from_numpy(np.array(clip_features[start:end], copy=True)).float()
        chunk = torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        chunk = chunk.to(device)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                chunk = chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                logits = (chunk @ text_embeds_compute.T) / FEATURE_SOFTMAX_TEMP
                probs = logits.softmax(dim=-1)
        else:
            chunk = chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            logits = (chunk @ text_embeds_compute.T) / FEATURE_SOFTMAX_TEMP
            probs = logits.softmax(dim=-1)
        probs_np = probs.float().cpu().numpy().astype(np.float32, copy=False)
        pred = probs_np.argmax(axis=1).astype(np.int32, copy=False)
        prob = probs_np[np.arange(probs_np.shape[0]), pred].astype(np.float32, copy=False)
        if background_idx is not None and feature_prob_th > 0:
            pred[prob < feature_prob_th] = int(background_idx)
        pred_point_labels[start:end] = pred
        max_point_probs[start:end] = prob
        if valid_instances.any():
            labels_chunk = pred_instance_labels[start:end]
            valid_chunk = labels_chunk >= 0
            if valid_chunk.any():
                grouped_reduce(labels_chunk[valid_chunk], probs_np[valid_chunk], score_sum, score_count)

    if not valid_instances.any():
        instance_classes = np.empty((0,), dtype=np.int32)
        instance_diag = {
            "num_instances": 0,
            "mean_points_per_instance": 0.0,
            "mean_max_prob": float("nan"),
            "agg_mode": FEATURE_INSTANCE_AGG,
        }
    else:
        keep = score_count > 0
        score_mean = score_sum[keep] / score_count[keep, None].clip(min=1)
        instance_classes_kept = score_mean.argmax(axis=1).astype(np.int32, copy=False)
        instance_max_prob = score_mean.max(axis=1).astype(np.float32, copy=False)
        if feature_prob_th > 0 and background_idx is not None:
            instance_classes_kept[instance_max_prob < feature_prob_th] = background_idx
        instance_classes = np.full((num_instances,), -1, dtype=np.int32)
        instance_classes[np.flatnonzero(keep)] = instance_classes_kept
        instance_diag = {
            "num_instances": int(keep.sum()),
            "mean_points_per_instance": float(score_count[keep].mean()) if keep.any() else 0.0,
            "mean_max_prob": float(instance_max_prob.mean()) if instance_max_prob.size > 0 else float("nan"),
            "agg_mode": FEATURE_INSTANCE_AGG,
        }

    return pred_point_labels, max_point_probs, instance_classes, instance_diag


def summarize_feature_transfer(
    pred_point_labels: np.ndarray,
    max_point_probs: np.ndarray,
    gt_to_pred_idx: np.ndarray,
    feature_prob_th: float,
) -> tuple[np.ndarray, dict]:
    pred_labels = pred_point_labels[gt_to_pred_idx]
    diagnostics = {
        "feature_softmax_temp": FEATURE_SOFTMAX_TEMP,
        "feature_prob_th": float(feature_prob_th),
        "feature_mean_max_prob": float(max_point_probs[gt_to_pred_idx].mean()) if gt_to_pred_idx.size > 0 else float("nan"),
        "feature_unique_pred_points": int(np.unique(gt_to_pred_idx).shape[0]),
        "feature_query_points": int(gt_to_pred_idx.shape[0]),
    }
    return pred_labels, diagnostics


def classify_instance_features_ovo_style(
    clip_features: np.ndarray,
    pred_instance_labels: np.ndarray,
    text_embeds: torch.Tensor,
    score_th: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    instance_descriptors, feature_count = pool_instance_clip_features(
        clip_features,
        pred_instance_labels,
        chunk_size,
    )
    if instance_descriptors.shape[0] == 0:
        instance_classes = np.empty((0,), dtype=np.int32)
        instance_scores = np.empty((0,), dtype=np.float32)
        instance_diag = {
            "num_instances": 0,
            "mean_points_per_instance": 0.0,
            "mean_max_score": float("nan"),
            "score_th": float(score_th),
            "text_template": OVO_TEXT_TEMPLATE,
            "agg_mode": OVO_FEATURE_AGG,
        }
        return instance_classes, instance_scores, instance_diag

    valid = feature_count > 0
    instance_classes = np.full((instance_descriptors.shape[0],), -1, dtype=np.int32)
    instance_scores = np.full((instance_descriptors.shape[0],), -np.inf, dtype=np.float32)
    descriptors_t = torch.from_numpy(instance_descriptors[valid]).to(text_embeds.device, dtype=text_embeds.dtype)
    similarity = (descriptors_t @ text_embeds.T).float().cpu().numpy()
    instance_classes_kept = similarity.argmax(axis=1).astype(np.int32, copy=False)
    instance_max_score = similarity[np.arange(similarity.shape[0]), instance_classes_kept].astype(np.float32, copy=False)
    instance_scores[np.flatnonzero(valid)] = instance_max_score
    if score_th > 0:
        reject = instance_max_score <= score_th
        instance_classes_kept[reject] = -1
    instance_classes[np.flatnonzero(valid)] = instance_classes_kept
    instance_diag = {
        "num_instances": int(valid.sum()),
        "mean_points_per_instance": float(feature_count[valid].mean()) if valid.any() else 0.0,
        "mean_max_score": float(instance_max_score.mean()) if instance_max_score.size > 0 else float("nan"),
        "score_th": float(score_th),
        "text_template": OVO_TEXT_TEMPLATE,
        "agg_mode": OVO_FEATURE_AGG,
    }
    return instance_classes, instance_scores, instance_diag


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

    valid_labels = pred_instance_labels[valid]
    valid_points = pred_points[valid]
    k = min(5, valid_points.shape[0])
    _, knn_idx = cKDTree(valid_points).query(gt_points, k=k, workers=-1)
    if k == 1:
        mesh_instance_labels = valid_labels[np.asarray(knn_idx, dtype=np.int64)]
    else:
        knn_labels = valid_labels[np.asarray(knn_idx, dtype=np.int64)]
        mesh_instance_labels = torch.mode(torch.from_numpy(knn_labels.astype(np.int64, copy=False)), dim=1).values.numpy().astype(np.int32, copy=False)
    matched_instance_ids = np.unique(mesh_instance_labels)
    mesh_semantic_labels = instance_classes[mesh_instance_labels]
    diagnostics = {
        "matched_instance_count": int(matched_instance_ids.size),
        "assigned_gt_vertices": int(mesh_semantic_labels.shape[0]),
    }
    return mesh_semantic_labels, diagnostics


def build_iou_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_gt = gt_labels >= 0
    gt = gt_labels[valid_gt]
    pred = pred_labels[valid_gt]
    uniq_gt, gt_inv = np.unique(gt, return_inverse=True)
    valid_pred = pred >= 0
    uniq_pred = np.unique(pred[valid_pred])
    if uniq_gt.size == 0 or uniq_pred.size == 0:
        return np.zeros((uniq_gt.size, uniq_pred.size), dtype=np.float32), uniq_gt, uniq_pred, np.bincount(gt_inv, minlength=uniq_gt.size)

    pred_map = {pred_id: i for i, pred_id in enumerate(uniq_pred)}
    pred_inv = np.asarray([pred_map[p] for p in pred[valid_pred]], dtype=np.int32)
    contingency = np.zeros((uniq_gt.shape[0], uniq_pred.shape[0]), dtype=np.int64)
    np.add.at(contingency, (gt_inv[valid_pred], pred_inv), 1)
    gt_count = np.bincount(gt_inv, minlength=uniq_gt.shape[0])
    pred_count = np.bincount(pred_inv, minlength=uniq_pred.shape[0])
    union = gt_count[:, None] + pred_count[None, :] - contingency
    iou = np.where(union > 0, contingency / union, 0.0).astype(np.float32)
    return iou, uniq_gt, uniq_pred, gt_count


def majority_class_per_instance(
    instance_labels: np.ndarray,
    semantic_labels: np.ndarray,
    instance_ids: np.ndarray,
    ignore_labels: list[int],
) -> np.ndarray:
    ignore_set = set(int(x) for x in ignore_labels)
    class_ids = np.full((instance_ids.shape[0],), -1, dtype=np.int32)
    for idx, instance_id in enumerate(instance_ids.tolist()):
        mask = instance_labels == int(instance_id)
        semantic = semantic_labels[mask]
        semantic = semantic[(semantic >= 0) & (~np.isin(semantic, list(ignore_set)))]
        if semantic.size == 0:
            continue
        values, counts = np.unique(semantic.astype(np.int32, copy=False), return_counts=True)
        class_ids[idx] = int(values[np.argmax(counts)])
    return class_ids


def compute_instance_metrics(
    gt_instance_labels: np.ndarray,
    gt_semantic_labels: np.ndarray,
    pred_instance_labels: np.ndarray,
    pred_instance_classes: np.ndarray,
    pred_instance_scores: np.ndarray,
    dataset_info: dict,
) -> tuple[dict, dict]:
    iou, uniq_gt, uniq_pred, _ = build_iou_matrix(gt_instance_labels, pred_instance_labels)
    ignore_labels = dataset_info.get("ignore", []).copy() + dataset_info.get("background_reduced_ids", [])
    if uniq_gt.size == 0:
        metrics = {"ap": float("nan")}
        for threshold in INSTANCE_AP_THRESHOLDS:
            metrics[f"ap_{int(round(threshold * 100)):02d}"] = float("nan")
        diagnostics = {"gt_instance_count": 0, "pred_instance_count": int(uniq_pred.shape[0])}
        return metrics, diagnostics

    gt_class_ids = majority_class_per_instance(gt_instance_labels, gt_semantic_labels, uniq_gt, ignore_labels)
    pred_class_ids = pred_instance_classes[uniq_pred] if uniq_pred.size > 0 else np.empty((0,), dtype=np.int32)
    pred_scores = pred_instance_scores[uniq_pred] if uniq_pred.size > 0 else np.empty((0,), dtype=np.float32)
    class_ids = np.unique(gt_class_ids[gt_class_ids >= 0])
    metrics, ap_diag = compute_instance_ap_dataset(
        entries=[
            {
                "iou": iou,
                "gt_class_ids": gt_class_ids,
                "pred_class_ids": pred_class_ids,
                "pred_scores": pred_scores,
            }
        ],
        class_ids=class_ids,
        iou_thresholds=INSTANCE_AP_THRESHOLDS,
    )
    diagnostics = {
        "gt_instance_count": int(np.sum(gt_class_ids >= 0)),
        "pred_instance_count": int(np.sum(pred_class_ids >= 0)),
        "ignored_gt_instance_count": int(np.sum(gt_class_ids < 0)),
        **ap_diag,
    }
    return metrics, diagnostics


def round_for_print(value, digits: int = 3):
    if isinstance(value, dict):
        return {key: round_for_print(val, digits) for key, val in value.items()}
    if isinstance(value, list):
        return [round_for_print(val, digits) for val in value]
    if isinstance(value, tuple):
        return tuple(round_for_print(val, digits) for val in value)
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return round(float(value), digits)
        return float(value)
    return value


def load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def flatten_metric_summary(summary: dict) -> dict[str, float]:
    flat = {}
    for section, key in METRIC_ROW_ORDER:
        section_data = summary["metrics"].get(section)
        if section_data is not None and key in section_data:
            flat[f"{section}.{key}"] = float(section_data[key])
    return flat


def flatten_timing_summary(timing: dict | None) -> dict[str, float]:
    if timing is None:
        return {}
    flat = {}
    for label, path in TIMING_ROW_ORDER:
        value = timing
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            flat[label] = float(value)
    return flat


def format_num(value: float | None) -> str:
    if value is None:
        return "-"
    if not math.isfinite(float(value)):
        return str(float(value))
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def format_delta(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "-"
    delta = float(current) - float(baseline)
    if not math.isfinite(delta):
        return str(delta)
    text = f"{delta:+.6f}".rstrip("0").rstrip(".")
    return text if text not in {"+", "-"} else f"{delta:+.1f}"


def is_lower_better(row_name: str) -> bool:
    return row_name in {"geometry.chamfer_l1_m", "normals.mean_angle_deg"} or row_name.startswith("timing.")


def format_verdict(row_name: str, current: float | None, baseline: float | None) -> str:
    if row_name.startswith("timing.") and row_name != "timing.frame_loop_sec":
        return "DON'T CARE"
    if current is None or baseline is None:
        return "-"
    baseline = float(baseline)
    current = float(current)
    if not math.isfinite(current) or not math.isfinite(baseline):
        return "-"
    scale = abs(baseline)
    if scale < 1e-12:
        return "-"
    rel_change = (baseline - current) / scale if is_lower_better(row_name) else (current - baseline) / scale
    if abs(rel_change) < 0.01:
        return "-"
    strength = 3 if abs(rel_change) >= 0.2 else 2 if abs(rel_change) >= 0.1 else 1
    return ("Y" if rel_change > 0 else "N") * strength


def render_compare_table(title: str, row_names: list[str], current_values: dict[str, float], baseline_values: dict[str, float]) -> str:
    if not row_names:
        return ""
    rows = [(name, current_values.get(name), baseline_values.get(name)) for name in row_names]
    name_w = max(len(title), len("name"), *(len(name) for name, _, _ in rows))
    cur_w = max(len("current"), *(len(format_num(cur)) for _, cur, _ in rows))
    base_w = max(len("baseline"), *(len(format_num(base)) for _, _, base in rows))
    delta_w = max(len("delta"), *(len(format_delta(cur, base)) for _, cur, base in rows))
    verdict_w = max(len("verdict"), *(len(format_verdict(name, cur, base)) for name, cur, base in rows))
    lines = [
        title,
        f"{'name'.ljust(name_w)}  {'current'.rjust(cur_w)}  {'baseline'.rjust(base_w)}  {'delta'.rjust(delta_w)}  {'verdict'.rjust(verdict_w)}",
        f"{'-' * name_w}  {'-' * cur_w}  {'-' * base_w}  {'-' * delta_w}  {'-' * verdict_w}",
    ]
    for name, current, baseline in rows:
        lines.append(
            f"{name.ljust(name_w)}  {format_num(current).rjust(cur_w)}  {format_num(baseline).rjust(base_w)}  {format_delta(current, baseline).rjust(delta_w)}  {format_verdict(name, current, baseline).rjust(verdict_w)}"
        )
    return "\n".join(lines)


def print_compare_report(current_summary: dict, current_run_dir: Path, baseline_input: str) -> None:
    baseline_ply = resolve_ply_path(baseline_input)
    baseline_run_dir = baseline_ply.parent
    baseline_summary = load_json_if_exists(baseline_ply.with_name("metrics.json"))
    if baseline_summary is None:
        raise ValueError(f"Missing baseline metrics.json next to {baseline_ply}")

    current_metrics = flatten_metric_summary(current_summary)
    baseline_metrics = flatten_metric_summary(baseline_summary)
    current_timing = flatten_timing_summary(load_json_if_exists(current_run_dir / TIMING_PATH))
    baseline_timing = flatten_timing_summary(load_json_if_exists(baseline_run_dir / TIMING_PATH))
    current_stats = load_json_if_exists(current_run_dir / STATS_PATH) or {}
    baseline_stats = load_json_if_exists(baseline_run_dir / STATS_PATH) or {}

    metric_rows = list(current_metrics)
    timing_rows = [name for name, _ in TIMING_ROW_ORDER if name in current_timing or name in baseline_timing]

    print()
    print(f"compare: {current_run_dir}")
    print(f"baseline: {baseline_run_dir}")
    if current_stats or baseline_stats:
        print(
            "rgb_normal_point_fusion: "
            f"current={current_stats.get('rgb_normal_point_fusion', 'unknown')} "
            f"baseline={baseline_stats.get('rgb_normal_point_fusion', 'unknown')}"
        )
        print(
            "clip_feature_fusion: "
            f"current={current_stats.get('clip_feature_fusion', 'unknown')} "
            f"baseline={baseline_stats.get('clip_feature_fusion', 'unknown')}"
        )
    print(render_compare_table("metrics", metric_rows, current_metrics, baseline_metrics))
    if timing_rows:
        print()
        print(render_compare_table("timing", timing_rows, current_timing, baseline_timing))


def main(args: argparse.Namespace) -> None:
    ply_path = resolve_ply_path(args.input_path)
    dataset_name, scene_name = infer_scene_info_from_path(ply_path)
    progress = tqdm(total=6, desc=scene_name, unit="stage", dynamic_ncols=True)
    try:
        progress.set_postfix_str("load gt", refresh=True)
        dataset_info = load_dataset_info(dataset_name)
        gt = load_gt(dataset_name, scene_name, args)
        progress.update()

        progress.set_postfix_str("load map", refresh=True)
        pred = load_pred_map(ply_path)
        progress.update()

        progress.set_postfix_str("nn assoc", refresh=True)
        assoc = compute_nn_associations(gt["points"], pred["points"])
        gt_to_pred_idx = assoc["gt_to_pred_idx"]
        gt_to_pred_d = assoc["gt_to_pred_d"]
        coverage = float((gt_to_pred_d <= args.match_distance_th).mean())
        transferred_rgb = pred["colors"][gt_to_pred_idx]
        transferred_normals = pred["normals"][gt_to_pred_idx] if gt["normals"] is not None else None
        progress.update()

        progress.set_postfix_str("semantic", refresh=True)
        gt_semantic = map_gt_labels_to_eval_ids(gt["semantic_raw"], dataset_info)
        class_names = dataset_info.get("class_names_reduced", dataset_info.get("class_names"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        feature_model_name, feature_pretrained = load_feature_model_spec(ply_path.parent)
        text_embeds = encode_class_texts(class_names, device, model_name=feature_model_name, pretrained=feature_pretrained)
        ovo_text_embeds = encode_class_texts(class_names, device, template=OVO_TEXT_TEMPLATE, model_name=feature_model_name, pretrained=feature_pretrained)
        background_idx = class_names.index("background") if "background" in class_names else None
        from visualize_rgb_map import resolve_instance_labels

        pred_instance_labels = resolve_instance_labels(
            ply_path.parent,
            pred["points"].shape[0],
            args.min_component_size,
        )
        pred_point_labels, max_point_probs, _, _ = score_pred_points_and_instances(
            pred["clip_features"],
            pred_instance_labels,
            text_embeds,
            background_idx,
            args.feature_prob_th,
            args.chunk_size,
        )
        pred_feature_labels, feature_diag = summarize_feature_transfer(
            pred_point_labels,
            max_point_probs,
            gt_to_pred_idx,
            args.feature_prob_th,
        )
        feature_conf = build_confusion(gt_semantic, pred_feature_labels, dataset_info["num_classes"], dataset_info.get("ignore", []))
        feature_metrics = confusion_to_metrics(feature_conf, dataset_info)
        feature_metrics = {key: feature_metrics[key] for key in ("mIoU", "mAcc")}
        progress.update()

        progress.set_postfix_str("instances", refresh=True)
        instance_classes, instance_scores, semantic_ovo_diag_1 = classify_instance_features_ovo_style(
            pred["clip_features"],
            pred_instance_labels,
            ovo_text_embeds,
            args.ovo_score_th,
            args.chunk_size,
        )
        pred_semantic_ovo_labels, semantic_ovo_diag_2 = transfer_semantic_labels_ovo_style(
            pred["points"],
            pred_instance_labels,
            instance_classes,
            gt["points"],
        )
        semantic_ovo_conf = build_confusion(
            gt_semantic,
            pred_semantic_ovo_labels,
            dataset_info["num_classes"],
            dataset_info.get("ignore", []),
        )
        semantic_ovo_metrics = confusion_to_metrics(semantic_ovo_conf, dataset_info)
        if gt["instance_labels"] is not None:
            transferred_instance_labels = pred_instance_labels[gt_to_pred_idx]
            instance_metrics, instance_diag = compute_instance_metrics(
                gt["instance_labels"],
                gt_semantic,
                transferred_instance_labels,
                instance_classes,
                instance_scores,
                dataset_info,
            )
        else:
            instance_metrics, instance_diag = None, None
        progress.update()

        progress.set_postfix_str("summary", refresh=True)
        geometry_metrics, geometry_diag = compute_geometry_metrics(assoc)
        rgb_metrics = compute_rgb_metrics(gt["colors"], transferred_rgb) if gt["colors"] is not None else None
        normal_metrics = compute_normal_metrics(gt["normals"], transferred_normals) if gt["normals"] is not None else None
        geometry_metrics["coverage"] = coverage
        progress.update()
    finally:
        progress.close()

    summary = {
        "metrics": {
            "geometry": geometry_metrics,
            "rgb": rgb_metrics,
            "normals": normal_metrics,
            "feature": feature_metrics,
            "semantic_ovo_style": semantic_ovo_metrics,
            "instance": instance_metrics,
        },
        "diagnostics": {
            "dataset_name": canonical_dataset_name(dataset_name),
            "scene_name": scene_name,
            "n_pred_points": int(pred["points"].shape[0]),
            "n_gt_vertices": int(gt["points"].shape[0]),
            "match_distance_th_m": float(args.match_distance_th),
            "geometry": geometry_diag,
            "feature": feature_diag,
            "semantic_ovo_style": {
                **semantic_ovo_diag_1,
                **semantic_ovo_diag_2,
            },
            "feature_text_template": FEATURE_TEXT_TEMPLATE,
            "ovo_text_template": OVO_TEXT_TEMPLATE,
            "instance": None if instance_diag is None else {
                **instance_diag,
                "min_component_size": int(args.min_component_size),
            },
            "gt_has_normals": bool(gt["normals"] is not None),
            "gt_has_instances": bool(gt["instance_labels"] is not None),
        },
    }
    if dataset_name == "ScanNet":
        summary["diagnostics"]["scannet_raw_root"] = str(Path(args.scannet_raw_root).resolve())
    elif dataset_name == "Replica":
        replica_root = Path(args.replica_root) if args.replica_root is not None else INPUT_DIR / "Replica"
        summary["diagnostics"]["replica_root"] = str(replica_root.resolve())
    print(json.dumps(round_for_print(summary), indent=2))

    if args.save_json:
        out_path = ply_path.with_name("metrics.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(out_path)
    if args.compare:
        print_compare_report(summary, ply_path.parent, args.compare)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute custom-map metrics against dataset GT assets.")
    parser.add_argument("input_path", help="Path to rgb_map.ply or its containing directory.")
    parser.add_argument("--save_json", action="store_true", help="Save summary to metrics.json next to the map.")
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH, help="Distance threshold used in geometry coverage diagnostics.")
    parser.add_argument("--feature_prob_th", type=float, default=DEFAULT_FEATURE_PROB_TH, help="Minimum class softmax probability before assigning background.")
    parser.add_argument("--ovo_score_th", type=float, default=DEFAULT_OVO_SCORE_TH, help="Minimum cosine similarity before assigning an OVO-style instance class.")
    parser.add_argument("--min_component_size", type=int, default=2000)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--compare", default="", help="Optional baseline run directory or rgb_map.ply path to compare against after computing current metrics.")
    parser.add_argument("--scannet_raw_root", default=None, help="ScanNet raw scans root containing aggregation and segs files, e.g. /path/to/scannet_v2/scans.")
    parser.add_argument("--replica_root", default=None, help="Replica root containing semantic_gt/ and *_mesh.ply files. Defaults to data/input/Replica.")
    main(parser.parse_args())
