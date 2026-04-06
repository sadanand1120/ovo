import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import open_clip
from plyfile import PlyData
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import torch
from tqdm.auto import tqdm

from ovo import eval_utils, io_utils


DATASET_DIRS = {"replica": "Replica", "scannet": "ScanNet"}
CONFIG_DIR = Path("configs")
INPUT_DIR = Path("data/input")
CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
CLIP_PRETRAINED = "openai"
FEATURE_TEXT_TEMPLATE = "This is a photo of a {}"
FEATURE_SOFTMAX_TEMP = 0.01
DEFAULT_MATCH_DISTANCE_TH = 0.03
DEFAULT_FEATURE_PROB_TH = 0.0
DEFAULT_CHUNK_SIZE = 100_000
GEOM_THRESHOLDS = (0.01, 0.03, 0.05)
INSTANCE_IOU_THRESHOLDS = (0.5,)


def canonical_dataset_name(dataset_name: str) -> str:
    return DATASET_DIRS[dataset_name.lower()]


def resolve_ply_path(input_path: str) -> Path:
    path = Path(input_path)
    return path if path.suffix == ".ply" else path / "rgb_map.ply"


def infer_scene_info_from_path(ply_path: Path) -> tuple[str, str]:
    scene_name = ply_path.parent.name
    dataset_name = ply_path.parent.parent.name
    if dataset_name.lower() not in DATASET_DIRS:
        raise ValueError(f"Could not infer dataset from path: {ply_path}")
    return dataset_name, scene_name


def load_dataset_info(dataset_name: str) -> dict:
    return io_utils.load_config(CONFIG_DIR / f"{dataset_name.lower()}_eval.yaml")


def load_ply_vertices(ply_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    vertex = PlyData.read(str(ply_path))["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    labels = np.asarray(vertex["label"], dtype=np.int32) if "label" in vertex.dtype.names else None
    return points, colors, labels


def resolve_default_scannet_raw_root() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset" / "scannet_v2" / "scans"


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
    points, colors, semantic_raw = load_ply_vertices(labels_ply)

    mesh = o3d.io.read_triangle_mesh(str(labels_ply))
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

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


def load_pred_map(ply_path: Path) -> dict:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.rint(np.asarray(pcd.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    normals = np.asarray(pcd.normals, dtype=np.float32)
    if points.shape[0] == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")
    if normals.shape[0] != points.shape[0]:
        raise ValueError(f"{ply_path} has no stored normals. Rebuild the map with the current build_rgb_map.py.")
    clip_path = ply_path.with_name("clip_feats.npy")
    if not clip_path.exists():
        raise ValueError(f"Missing CLIP features: {clip_path}")
    instance_edge_path = ply_path.with_name("instance_edges.npz")
    if not instance_edge_path.exists():
        raise ValueError(f"Missing instance evidence: {instance_edge_path}")
    return {
        "points": points,
        "colors": colors,
        "normals": normals,
        "clip_path": clip_path,
        "instance_edge_path": instance_edge_path,
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
    iou_values, iou_valid_mask, weights_values, acc_values, acc_valid_mask = eval_utils.iou_acc_from_confmat(
        confusion,
        dataset_info["num_classes"],
        dataset_info.get("ignore", []).copy(),
        mask_nan=True,
        verbose=False,
        labels=dataset_info.get("class_names_reduced", dataset_info.get("class_names")),
    )
    return {
        "mIoU": float(np.mean(iou_values[iou_valid_mask])) if iou_valid_mask.any() else float("nan"),
        "mAcc": float(np.mean(acc_values[acc_valid_mask])) if acc_valid_mask.any() else float("nan"),
        "f_mIoU": float(np.sum(iou_values[iou_valid_mask] * weights_values[iou_valid_mask]) / weights_values[iou_valid_mask].sum()) if iou_valid_mask.any() else float("nan"),
        "f_mAcc": float(np.sum(acc_values[acc_valid_mask] * weights_values[acc_valid_mask]) / weights_values[acc_valid_mask].sum()) if acc_valid_mask.any() else float("nan"),
    }


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
def encode_class_texts(class_names: list[str], device: str) -> torch.Tensor:
    model = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device)[0].eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    phrases = [FEATURE_TEXT_TEMPLATE.format(name) for name in class_names]
    text = model.encode_text(tokenizer(phrases).to(device)).float()
    return text / text.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def classify_transferred_features(
    clip_path: Path,
    gt_to_pred_idx: np.ndarray,
    class_names: list[str],
    feature_prob_th: float,
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    features_mm = np.load(clip_path, mmap_mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_embeds = encode_class_texts(class_names, device)
    background_idx = class_names.index("background") if "background" in class_names else None

    pred_labels = np.empty(gt_to_pred_idx.shape[0], dtype=np.int32)
    max_probs = np.empty(gt_to_pred_idx.shape[0], dtype=np.float32)
    for start in tqdm(range(0, gt_to_pred_idx.shape[0], chunk_size), desc="Feature semantics", unit="chunk"):
        end = min(start + chunk_size, gt_to_pred_idx.shape[0])
        chunk_idx = gt_to_pred_idx[start:end]
        chunk = torch.from_numpy(np.array(features_mm[chunk_idx], copy=True)).float()
        chunk = torch.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
        chunk = chunk.to(device)
        chunk = chunk / chunk.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        logits = (chunk @ text_embeds.T) / FEATURE_SOFTMAX_TEMP
        probs = logits.softmax(dim=-1)
        prob, pred = probs.max(dim=-1)
        pred = pred.cpu().numpy().astype(np.int32)
        prob = prob.cpu().numpy().astype(np.float32)
        if background_idx is not None and feature_prob_th > 0:
            pred[prob < feature_prob_th] = int(background_idx)
        pred_labels[start:end] = pred
        max_probs[start:end] = prob
    diagnostics = {
        "feature_softmax_temp": FEATURE_SOFTMAX_TEMP,
        "feature_prob_th": float(feature_prob_th),
        "feature_mean_max_prob": float(max_probs.mean()),
    }
    return pred_labels, diagnostics


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


def compute_instance_metrics(gt_labels: np.ndarray, pred_labels: np.ndarray) -> tuple[dict, dict]:
    iou, uniq_gt, uniq_pred, gt_count = build_iou_matrix(gt_labels, pred_labels)
    if uniq_gt.size == 0:
        return {
            "mwcov": float("nan"),
            "f1_50": float("nan"),
        }, {
            "gt_instance_count": 0,
            "pred_instance_count": 0,
        }
    if uniq_pred.size == 0:
        metrics = {"mwcov": 0.0, "f1_50": 0.0}
    else:
        best_iou = iou.max(axis=1)
        metrics = {
            "mwcov": float(np.sum(best_iou * gt_count) / np.sum(gt_count)),
        }
        row_ind, col_ind = linear_sum_assignment(1.0 - iou)
        matched_ious = iou[row_ind, col_ind]
        for th in INSTANCE_IOU_THRESHOLDS:
            tp = int((matched_ious >= th).sum())
            precision = 0.0 if uniq_pred.size == 0 else tp / float(uniq_pred.size)
            recall = 0.0 if uniq_gt.size == 0 else tp / float(uniq_gt.size)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            metrics[f"f1_{int(th * 100):02d}"] = float(f1)

    diagnostics = {
        "gt_instance_count": int(uniq_gt.shape[0]),
        "pred_instance_count": int(uniq_pred.shape[0]),
    }
    return metrics, diagnostics


def main(args: argparse.Namespace) -> None:
    ply_path = resolve_ply_path(args.input_path)
    dataset_name, scene_name = infer_scene_info_from_path(ply_path)
    dataset_name = dataset_name.lower()
    if dataset_name != "scannet":
        raise NotImplementedError("get_metrics_map.py currently supports ScanNet only.")

    raw_root = Path(args.scannet_raw_root) if args.scannet_raw_root else resolve_default_scannet_raw_root()
    dataset_info = load_dataset_info(dataset_name)
    gt = load_scannet_gt(scene_name, raw_root)
    pred = load_pred_map(ply_path)

    assoc = compute_nn_associations(gt["points"], pred["points"])
    gt_to_pred_idx = assoc["gt_to_pred_idx"]
    gt_to_pred_d = assoc["gt_to_pred_d"]
    coverage = float((gt_to_pred_d <= args.match_distance_th).mean())

    transferred_rgb = pred["colors"][gt_to_pred_idx]
    transferred_normals = pred["normals"][gt_to_pred_idx]

    gt_semantic = map_gt_labels_to_eval_ids(gt["semantic_raw"], dataset_info)
    class_names = dataset_info.get("class_names_reduced", dataset_info.get("class_names"))
    pred_feature_labels, feature_diag = classify_transferred_features(
        pred["clip_path"],
        gt_to_pred_idx,
        class_names,
        args.feature_prob_th,
        args.chunk_size,
    )
    feature_conf = build_confusion(gt_semantic, pred_feature_labels, dataset_info["num_classes"], dataset_info.get("ignore", []))
    feature_metrics = confusion_to_metrics(feature_conf, dataset_info)
    feature_metrics = {key: feature_metrics[key] for key in ("mIoU", "mAcc")}

    from visualize_rgb_map import compute_instance_labels

    pred_instance_labels = compute_instance_labels(
        pred["instance_edge_path"],
        pred["points"].shape[0],
        args.tau_same,
        args.n_min,
        args.min_component_size,
    )[gt_to_pred_idx]
    instance_metrics, instance_diag = compute_instance_metrics(gt["instance_labels"], pred_instance_labels)

    geometry_metrics, geometry_diag = compute_geometry_metrics(assoc)
    rgb_metrics = compute_rgb_metrics(gt["colors"], transferred_rgb)
    normal_metrics = compute_normal_metrics(gt["normals"], transferred_normals)
    geometry_metrics["coverage"] = coverage

    summary = {
        "metrics": {
            "geometry": geometry_metrics,
            "rgb": rgb_metrics,
            "normals": normal_metrics,
            "feature": feature_metrics,
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
            "instance": {
                **instance_diag,
                "tau_same": float(args.tau_same),
                "n_min": int(args.n_min),
                "min_component_size": int(args.min_component_size),
            },
        },
    }
    print(json.dumps(summary, indent=2))

    if args.save_json:
        out_path = ply_path.with_name("metrics.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute 3D map metrics on ScanNet GT mesh vertices.")
    parser.add_argument("input_path", help="Path to rgb_map.ply or its containing directory.")
    parser.add_argument("--save_json", action="store_true", help="Save summary to metrics.json next to the map.")
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH, help="Distance threshold used in geometry coverage diagnostics.")
    parser.add_argument("--feature_prob_th", type=float, default=DEFAULT_FEATURE_PROB_TH, help="Minimum class softmax probability before assigning background.")
    parser.add_argument("--tau_same", type=float, default=0.65)
    parser.add_argument("--n_min", type=int, default=1)
    parser.add_argument("--min_component_size", type=int, default=75)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--scannet_raw_root", default="", help="Optional ScanNet raw scans root; defaults to the repo-adjacent dataset path.")
    main(parser.parse_args())
