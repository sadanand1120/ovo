import argparse
import json
import sys
import time
from pathlib import Path
import shutil
import tempfile

import cv2  # Keep OpenCV loaded before torch in the container env.
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_rgb_map import (  # noqa: E402
    BIRTH_MIN_POINTS,
    CLIP_FEATURE_FILE,
    CLIP_LOAD_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    CONFIG_DIR,
    DATASET_DIRS,
    DEFAULT_DOWNSCALE_RES,
    DEFAULT_K_POOLING,
    DEFAULT_MAP_EVERY,
    DEFAULT_MATCH_DISTANCE_TH,
    DEFAULT_MAX_FRAME_POINTS,
    DenseCLIPExtractor,
    INPUT_DIR,
    OUTPUT_DIR,
    TIMING_PATH,
    as_int,
    canonical_dataset_name,
    compute_normals_from_depth,
    invert_rigid_transform,
    load_dataset,
)
from get_metrics_map import compute_nn_associations, load_scannet_gt, normalize_rows  # noqa: E402
from ovo import geometry_utils  # noqa: E402
from ovo.sam_mask_utils import GTInstanceMaskExtractor, SAM1_LEVELS, SAMMaskExtractor  # noqa: E402
from ovo.sam2_utils import SAM2_LEVELS  # noqa: E402


DEFAULT_SCANNET_RAW_ROOT = Path("../../dataset/scannet_v2/scans")


def rotate_vectors(vectors: np.ndarray, axes: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(angles_deg.astype(np.float32, copy=False))
    cos = np.cos(angles)[:, None]
    sin = np.sin(angles)[:, None]
    cross = np.cross(axes, vectors)
    return normalize_rows(vectors * cos + cross * sin)


def sample_perpendicular_axes(vectors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rand = rng.standard_normal(vectors.shape, dtype=np.float32)
    axes = np.cross(vectors, rand)
    bad = np.linalg.norm(axes, axis=1) < 1e-6
    if bad.any():
        fallback = np.zeros_like(vectors, dtype=np.float32)
        fallback[:, 0] = 1.0
        fallback[np.abs(vectors[:, 0]) > 0.9] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axes[bad] = np.cross(vectors[bad], fallback[bad])
    return normalize_rows(axes)


def project_normals_with_noise(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    gt_normals: np.ndarray,
    assoc: dict,
    target_mean_angle_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    pred_to_gt_idx = assoc["pred_to_gt_idx"]
    gt_to_pred_idx = assoc["gt_to_pred_idx"]
    pred_normals = gt_normals[pred_to_gt_idx].copy()
    sums = np.zeros_like(pred_normals, dtype=np.float32)
    np.add.at(sums, gt_to_pred_idx, gt_normals)
    counts = np.bincount(gt_to_pred_idx, minlength=pred_points.shape[0]).astype(np.int64)
    covered = counts > 0
    if covered.any():
        pred_normals[covered] = normalize_rows(sums[covered])
    pred_normals = normalize_rows(pred_normals)
    if target_mean_angle_deg <= 0:
        assigned = pred_normals[gt_to_pred_idx]
        dots = np.abs((gt_normals * assigned).sum(axis=1)).clip(0.0, 1.0)
        return pred_normals, float(np.degrees(np.arccos(dots)).mean())

    axes = sample_perpendicular_axes(pred_normals, rng)
    samples = np.abs(rng.standard_normal(pred_normals.shape[0], dtype=np.float32))

    def eval_mean(scale: float) -> float:
        rotated = rotate_vectors(pred_normals, axes, np.clip(samples * scale, 0.0, 179.0))
        assigned = rotated[gt_to_pred_idx]
        dots = np.abs((gt_normals * assigned).sum(axis=1)).clip(0.0, 1.0)
        return float(np.degrees(np.arccos(dots)).mean())

    baseline = eval_mean(0.0)
    if target_mean_angle_deg <= baseline + 1e-4:
        return pred_normals, baseline
    lo, hi = 0.0, max(target_mean_angle_deg, 1.0)
    while eval_mean(hi) < target_mean_angle_deg and hi < 360.0:
        hi *= 2.0
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if eval_mean(mid) < target_mean_angle_deg:
            lo = mid
        else:
            hi = mid
    final = rotate_vectors(pred_normals, axes, np.clip(samples * hi, 0.0, 179.0))
    return final, eval_mean(hi)


def dominant_instance_per_pred(gt_instance_labels: np.ndarray, gt_to_pred_idx: np.ndarray, pred_to_gt_idx: np.ndarray, n_pred: int) -> np.ndarray:
    pred_labels = gt_instance_labels[pred_to_gt_idx].astype(np.int32, copy=True)
    order = np.argsort(gt_to_pred_idx, kind="stable")
    pred_sorted = gt_to_pred_idx[order]
    label_sorted = gt_instance_labels[order]
    starts = np.flatnonzero(np.r_[True, pred_sorted[1:] != pred_sorted[:-1]])
    ends = np.r_[starts[1:], pred_sorted.shape[0]]
    for start, end in zip(starts.tolist(), ends.tolist()):
        labels = label_sorted[start:end]
        labels = labels[labels >= 0]
        if labels.size == 0:
            pred_labels[pred_sorted[start]] = -1
            continue
        uniq, counts = np.unique(labels, return_counts=True)
        pred_labels[pred_sorted[start]] = int(uniq[counts.argmax()])
    if pred_labels.shape[0] != n_pred:
        raise ValueError("Projected instance label size mismatch.")
    return pred_labels


def remap_labels(labels: np.ndarray) -> np.ndarray:
    out = labels.copy()
    valid = out >= 0
    if valid.any():
        _, inv = np.unique(out[valid], return_inverse=True)
        out[valid] = inv.astype(np.int32, copy=False)
    return out


def merge_instances(labels: np.ndarray, points: np.ndarray, target_count: int) -> np.ndarray:
    labels = remap_labels(labels)
    valid = labels >= 0
    current = int(labels.max()) + 1 if valid.any() else 0
    if target_count <= 0 or target_count >= current:
        return labels
    while current > target_count:
        centroids = np.stack([points[labels == gid].mean(axis=0) for gid in range(current)], axis=0)
        counts = np.asarray([(labels == gid).sum() for gid in range(current)], dtype=np.int64)
        src = int(np.argmin(counts))
        dists = np.linalg.norm(centroids - centroids[src], axis=1)
        dists[src] = np.inf
        dst = int(np.argmin(dists))
        labels[labels == src] = dst
        labels = remap_labels(labels)
        current = int(labels.max()) + 1 if (labels >= 0).any() else 0
    return labels


def split_largest_instance(labels: np.ndarray, points: np.ndarray) -> np.ndarray:
    labels = remap_labels(labels)
    valid = labels >= 0
    if not valid.any():
        return labels
    uniq, counts = np.unique(labels[valid], return_counts=True)
    gid = int(uniq[counts.argmax()])
    mask = labels == gid
    pts = points[mask]
    if pts.shape[0] < 2:
        return labels
    centered = pts - pts.mean(axis=0, keepdims=True)
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    proj = centered @ axis
    cut = np.median(proj)
    split_mask = np.zeros_like(mask)
    split_mask[np.flatnonzero(mask)[proj > cut]] = True
    if split_mask.sum() == 0 or split_mask.sum() == mask.sum():
        split_mask[np.flatnonzero(mask)[::2]] = True
    labels[split_mask] = int(labels.max()) + 1
    return remap_labels(labels)


def find_adjacent_boundary_targets(labels: np.ndarray, points: np.ndarray, k_neighbors: int = 12) -> tuple[np.ndarray, np.ndarray]:
    valid = labels >= 0
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 2:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int32)
    pts = points[valid_idx]
    lbls = labels[valid_idx]
    k = min(int(k_neighbors) + 1, pts.shape[0])
    knn = cKDTree(pts).query(pts, k=k, workers=-1)[1]
    if k == 1:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int32)
    if knn.ndim == 1:
        knn = knn[:, None]
    neighbor_labels = lbls[knn[:, 1:]]
    diff = neighbor_labels != lbls[:, None]
    keep = diff.any(axis=1)
    if not keep.any():
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int32)
    targets = np.full((keep.sum(),), -1, dtype=np.int32)
    for out_i, src_i in enumerate(np.flatnonzero(keep).tolist()):
        uniq, counts = np.unique(neighbor_labels[src_i][diff[src_i]], return_counts=True)
        targets[out_i] = int(uniq[counts.argmax()])
    return valid_idx[keep], targets


def steal_adjacent_instance_chunks(labels: np.ndarray, points: np.ndarray, noise_frac: float, rng: np.random.Generator) -> np.ndarray:
    labels = remap_labels(labels)
    valid = labels >= 0
    if not valid.any() or noise_frac <= 0:
        return labels
    target_noise = min(int(valid.sum()), int(round(valid.sum() * noise_frac)))
    if target_noise <= 0:
        return labels

    seed_idx, seed_targets = find_adjacent_boundary_targets(labels, points)
    if seed_idx.size == 0:
        return labels

    original_labels = labels.copy()
    changed = np.zeros(labels.shape[0], dtype=bool)
    label_indices = {gid: np.flatnonzero(original_labels == gid) for gid in np.unique(original_labels[valid]).tolist()}
    label_trees = {gid: cKDTree(points[idx]) for gid, idx in label_indices.items()}
    changed_total = 0

    for seed_pos in rng.permutation(seed_idx.shape[0]).tolist():
        if changed_total >= target_noise:
            break
        seed = int(seed_idx[seed_pos])
        src = int(original_labels[seed])
        dst = int(seed_targets[seed_pos])
        if src < 0 or dst < 0 or src == dst or labels[seed] != src:
            continue
        donor_pool = label_indices[src]
        donor_keep = (~changed[donor_pool]) & (labels[donor_pool] == src)
        donor_idx = donor_pool[donor_keep]
        if donor_idx.size == 0:
            continue
        target_pool = label_indices.get(dst)
        if target_pool is None or target_pool.size == 0:
            continue
        remaining = target_noise - changed_total
        max_chunk = min(remaining, max(8, donor_idx.size // 5))
        min_chunk = min(max_chunk, max(1, donor_idx.size // 50))
        chunk_size = int(rng.integers(min_chunk, max_chunk + 1))
        seed_point = points[seed : seed + 1]
        dist_seed = np.linalg.norm(points[donor_idx] - seed_point, axis=1)
        dist_target = label_trees[dst].query(points[donor_idx], k=1, workers=-1)[0]
        score = dist_seed + 0.35 * dist_target
        chosen = donor_idx[np.argsort(score, kind="stable")[:chunk_size]]
        labels[chosen] = dst
        changed[chosen] = True
        changed_total += int(chosen.size)
    return remap_labels(labels)


def adjust_instance_labels(labels: np.ndarray, points: np.ndarray, target_count: int, noise_frac: float, rng: np.random.Generator) -> np.ndarray:
    labels = remap_labels(labels)
    valid = labels >= 0
    current = int(labels.max()) + 1 if valid.any() else 0
    if target_count > 0:
        if target_count < current:
            labels = merge_instances(labels, points, target_count)
        while target_count > 0 and (int(labels.max()) + 1 if (labels >= 0).any() else 0) < target_count:
            updated = split_largest_instance(labels, points)
            if np.array_equal(updated, labels):
                break
            labels = updated
    return steal_adjacent_instance_chunks(labels, points, noise_frac, rng)


def project_gt_equivalent_attributes(
    pred_points: np.ndarray,
    scene_name: str,
    scannet_raw_root: str | Path,
    normals_noise_deg: float,
    instance_num_instances: int,
    instance_noise_frac: float,
    instance_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    gt = load_scannet_gt(scene_name, scannet_raw_root)
    assoc = compute_nn_associations(gt["points"], pred_points)
    rng = np.random.default_rng(int(instance_seed))
    pred_normals, achieved_normal_noise = project_normals_with_noise(
        pred_points,
        gt["points"],
        gt["normals"],
        assoc,
        float(normals_noise_deg),
        rng,
    )
    pred_instance_labels = dominant_instance_per_pred(
        gt["instance_labels"],
        assoc["gt_to_pred_idx"],
        assoc["pred_to_gt_idx"],
        pred_points.shape[0],
    )
    pred_instance_labels = adjust_instance_labels(
        pred_instance_labels,
        pred_points,
        int(instance_num_instances),
        float(instance_noise_frac),
        rng,
    )
    stats = {
        "gt_equivalent_normals_noise_deg_target": float(normals_noise_deg),
        "gt_equivalent_normals_noise_deg_achieved": float(achieved_normal_noise),
        "gt_equivalent_instance_num_instances": int(instance_num_instances),
        "gt_equivalent_instance_noise_frac": float(instance_noise_frac),
        "gt_equivalent_instance_rng_seed": int(instance_seed),
        "gt_equivalent_scannet_raw_root": str(Path(scannet_raw_root).resolve()),
    }
    return pred_normals, pred_instance_labels.astype(np.int32, copy=False), stats


class GTEquivalentMaskManager:
    def __init__(
        self,
        device: str,
        dataset_name: str,
        scene_name: str,
        use_inst_gt: bool,
        sam_model_level_inst: int,
        sam_model_level_textregion: int,
    ) -> None:
        self.use_inst_gt = bool(use_inst_gt)
        self.sam_model_level_inst = int(sam_model_level_inst)
        self.sam_model_level_textregion = int(sam_model_level_textregion)
        self.seed_mask_extractor = (
            GTInstanceMaskExtractor(dataset_name, scene_name) if self.use_inst_gt else SAMMaskExtractor(device, self.sam_model_level_inst)
        )
        self.textregion_mask_extractor = None if self.use_inst_gt else SAMMaskExtractor(device, sam_model_level_textregion)
        self.point_labels = np.empty((0,), dtype=np.int32)
        self.point_labels_dirty = False
        self.stats: dict[str, int] = {}
        self.tracker = None

    def close(self) -> None:
        return

    def num_active_instances(self) -> int:
        return 0

    def num_existing_instances(self) -> int:
        return 0

    def prepare_frame_labels(self, frame_id: int, is_seed_frame: bool, image_np: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not is_seed_frame:
            return None, None
        if self.use_inst_gt:
            seed_labels = self.seed_mask_extractor.extract_labels(frame_id, image_np.shape[:2])
        else:
            seed_labels = self.seed_mask_extractor.extract_labels(image_np)
        return np.full(seed_labels.shape, -1, dtype=np.int32), seed_labels

    def extract_textregion_labels(self, image_np: np.ndarray, seed_labels_np: np.ndarray | None) -> np.ndarray:
        if self.use_inst_gt and seed_labels_np is not None:
            return seed_labels_np
        if (not self.use_inst_gt) and seed_labels_np is not None and self.sam_model_level_inst == self.sam_model_level_textregion:
            return seed_labels_np
        if self.textregion_mask_extractor is None:
            raise RuntimeError("TextRegion SAM extractor was not initialized.")
        return self.textregion_mask_extractor.extract_labels(image_np)

    def assign_new_points(self, n_new: int) -> None:
        if n_new > 0:
            self.point_labels = np.concatenate((self.point_labels, np.full((int(n_new),), -1, dtype=np.int32)))


class GTEquivalentRGBMapper:
    def __init__(
        self,
        intrinsics: np.ndarray,
        device: str,
        map_every: int,
        downscale_res: int,
        k_pooling: int,
        max_frame_points: int,
        match_distance_th: float,
        dataset_name: str,
        scene_name: str,
        use_inst_gt: bool,
        sam_model_level_inst: int,
        sam_model_level_textregion: int,
        sam2_model_level_track: int,
        scannet_raw_root: str | Path,
        normals_noise_deg: float,
        instance_num_instances: int,
        instance_noise_frac: float,
        instance_seed: int,
    ) -> None:
        self.dataset_name = dataset_name.lower()
        self.scene_name = scene_name
        self.device = device
        self.cam_intrinsics = torch.tensor(intrinsics.astype(np.float32), device=device)
        self.map_every = max(1, int(map_every))
        self.downscale_res = max(1, int(downscale_res))
        self.max_frame_points = as_int(max_frame_points)
        self.match_distance_th = float(match_distance_th)
        self.k_pooling = int(k_pooling)
        self.use_inst_gt = bool(use_inst_gt)
        self.sam_model_level_inst = int(sam_model_level_inst)
        self.sam_model_level_textregion = int(sam_model_level_textregion)
        self.sam2_model_level_track = int(sam2_model_level_track)
        self.scannet_raw_root = Path(scannet_raw_root)
        self.normals_noise_deg = float(normals_noise_deg)
        self.instance_num_instances = int(instance_num_instances)
        self.instance_noise_frac = float(instance_noise_frac)
        self.instance_seed = int(instance_seed)
        self.clip_extractor = DenseCLIPExtractor(device)
        self.mask_manager = GTEquivalentMaskManager(
            device,
            dataset_name,
            scene_name,
            use_inst_gt,
            sam_model_level_inst,
            sam_model_level_textregion,
        )

        self.n_points = 0
        self.points = torch.empty((0, 3), device=device)
        self.colors = torch.empty((0, 3), device=device, dtype=torch.uint8)
        self.normals = torch.empty((0, 3), device=device)
        self.color_sum = torch.empty((0, 3), device=device)
        self.normal_sum = torch.empty((0, 3), device=device)
        self.obs_count = torch.empty((0,), device=device)
        self.feature_tmpdir = Path(tempfile.mkdtemp(prefix="rgb_map_feats_"))
        self.feature_tmp_path = self.feature_tmpdir / "clip_feats.bin"
        self.feature_tmp_file = open(self.feature_tmp_path, "wb")
        self.n_features = 0
        self.cached_depth_shape = None
        self.cached_x = None
        self.cached_y = None
        self.cached_row_ids = None
        self.cached_col_ids = None

        if k_pooling > 1 and k_pooling % 2 == 0:
            raise ValueError("k_pooling must be odd.")
        if k_pooling > 1:
            pooling = torch.nn.MaxPool2d(kernel_size=k_pooling, stride=1, padding=k_pooling // 2)
            self.pooling = lambda mask: ~(pooling((~mask[None]).float())[0].bool())
        else:
            self.pooling = lambda mask: mask
        self.downscale = (lambda x: x) if self.downscale_res == 1 else (lambda x: x[::self.downscale_res, ::self.downscale_res])

    def should_map_frame(self, frame_id: int) -> bool:
        return frame_id % self.map_every == 0

    def _ensure_capacity(self, min_capacity: int) -> None:
        if min_capacity <= self.points.shape[0]:
            return
        new_capacity = max(min_capacity, max(1 << 18, self.points.shape[0] * 2))

        def grow(buffer: torch.Tensor, *shape_tail: int) -> torch.Tensor:
            expanded = torch.empty((new_capacity, *shape_tail), device=buffer.device, dtype=buffer.dtype)
            if self.n_points > 0:
                expanded[:self.n_points] = buffer[:self.n_points]
            return expanded

        self.points = grow(self.points, 3)
        self.colors = grow(self.colors, 3)
        self.normals = grow(self.normals, 3)
        self.color_sum = grow(self.color_sum, 3)
        self.normal_sum = grow(self.normal_sum, 3)
        self.obs_count = grow(self.obs_count)

    def _get_cached_grids(self, depth_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cached_depth_shape != depth_shape:
            h, w = depth_shape
            full_y, full_x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
            self.cached_x = self.downscale(full_x)
            self.cached_y = self.downscale(full_y)
            ds_h, ds_w = self.cached_x.shape
            self.cached_row_ids, self.cached_col_ids = torch.meshgrid(
                torch.arange(ds_h, device=self.device),
                torch.arange(ds_w, device=self.device),
                indexing="ij",
            )
            self.cached_depth_shape = depth_shape
        return self.cached_x, self.cached_y, self.cached_row_ids, self.cached_col_ids

    def _append_points(self, points: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, features: torch.Tensor) -> None:
        start = self.n_points
        end = start + int(points.shape[0])
        self._ensure_capacity(end)
        self.points[start:end] = points
        self.colors[start:end] = colors
        self.normals[start:end] = normals
        self.color_sum[start:end] = colors.float()
        self.normal_sum[start:end] = normals.float()
        self.obs_count[start:end] = 1
        self.n_points = end
        feature_block = np.ascontiguousarray(
            torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy(),
            dtype=np.float16,
        )
        feature_block.tofile(self.feature_tmp_file)
        self.n_features += int(features.shape[0])

    def _update_observed_points(self, point_ids: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor) -> None:
        if point_ids.numel() == 0:
            return
        point_ids = point_ids.long()
        unique_ids = torch.unique(point_ids)
        self.color_sum.index_add_(0, point_ids, colors.float())
        self.normal_sum.index_add_(0, point_ids, normals.float())
        self.obs_count.index_add_(0, point_ids, torch.ones((point_ids.shape[0],), device=self.device))
        counts = self.obs_count[unique_ids].unsqueeze(1)
        self.colors[unique_ids] = torch.round(self.color_sum[unique_ids] / counts).clamp_(0.0, 255.0).to(torch.uint8)
        mean_normals = self.normal_sum[unique_ids] / counts
        self.normals[unique_ids] = mean_normals / torch.linalg.norm(mean_normals, dim=1, keepdim=True).clamp_min(1e-8)

    def add_frame(self, frame_data) -> None:
        frame_id, image_np, depth_np, c2w_np = frame_data[:4]
        is_seed_frame = self.should_map_frame(frame_id)
        if np.isinf(c2w_np).any() or np.isnan(c2w_np).any():
            return

        c2w = torch.from_numpy(c2w_np).to(self.device)
        depth = torch.from_numpy(depth_np).to(self.device)
        image = torch.from_numpy(image_np).to(self.device)
        full_image = image
        h, w = depth.shape
        x, y, row_ids, col_ids = self._get_cached_grids((h, w))
        mask = depth > 0
        point_ids_full = torch.full((h, w), -1, dtype=torch.int32, device=self.device)

        if self.n_points > 0:
            frustum_corners = geometry_utils.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
            w2c = invert_rigid_transform(c2w)
            frustum_mask = geometry_utils.compute_frustum_point_ids(self.points[: self.n_points], frustum_corners, device=self.device)
            if frustum_mask.numel() > 0:
                matched_ids, matches = geometry_utils.match_3d_points_to_2d_pixels(
                    depth,
                    w2c,
                    self.points[frustum_mask],
                    self.cam_intrinsics,
                    self.match_distance_th,
                )
                if matches.numel() > 0:
                    global_ids = frustum_mask[matched_ids].long()
                    point_ids_full[matches[:, 1], matches[:, 0]] = global_ids.to(point_ids_full.dtype)
                    mask[matches[:, 1], matches[:, 0]] = False
            mask = self.pooling(mask)

        _, seed_labels_np = self.mask_manager.prepare_frame_labels(frame_id, is_seed_frame, image_np)
        if not is_seed_frame or seed_labels_np is None:
            return

        tr_labels_np = self.mask_manager.extract_textregion_labels(image_np, seed_labels_np)
        tr_labels_full = torch.from_numpy(tr_labels_np).to(self.device)
        depth = self.downscale(depth)
        mask = self.downscale(mask)
        image = self.downscale(image)
        point_ids_ds = self.downscale(point_ids_full)
        normals_cam, normal_valid = compute_normals_from_depth(x, y, depth, self.cam_intrinsics)
        visible_existing = (point_ids_ds >= 0) & normal_valid
        if visible_existing.any():
            visible_ids = point_ids_ds[visible_existing]
            visible_colors = image[visible_existing].reshape(-1, 3)
            visible_normals = normals_cam[visible_existing].reshape(-1, 3)
            visible_normals = torch.einsum("ij,mj->mi", c2w[:3, :3], visible_normals)
            visible_normals = visible_normals / torch.linalg.norm(visible_normals, dim=1, keepdim=True).clamp_min(1e-8)
            self._update_observed_points(visible_ids, visible_colors, visible_normals)
        mask = mask & normal_valid

        if mask.any():
            x_keep = x[mask]
            y_keep = y[mask]
            row_keep = row_ids[mask]
            col_keep = col_ids[mask]
            depth_keep = depth[mask]
            colors = image[mask].reshape(-1, 3)
            normals_cam = normals_cam[mask].reshape(-1, 3)
            if depth_keep.shape[0] > self.max_frame_points:
                keep = torch.linspace(0, depth_keep.shape[0] - 1, self.max_frame_points, device=self.device).round().long()
                x_keep, y_keep = x_keep[keep], y_keep[keep]
                row_keep, col_keep = row_keep[keep], col_keep[keep]
                depth_keep, colors = depth_keep[keep], colors[keep]
                normals_cam = normals_cam[keep]

            dense_clip = self.clip_extractor.extract_dense(full_image, tr_labels_full)
            features = dense_clip[y_keep, x_keep].half()
            x_3d = (x_keep - self.cam_intrinsics[0, 2]) * depth_keep / self.cam_intrinsics[0, 0]
            y_3d = (y_keep - self.cam_intrinsics[1, 2]) * depth_keep / self.cam_intrinsics[1, 1]
            points = torch.stack((x_3d, y_3d, depth_keep, torch.ones_like(depth_keep)), dim=1)
            points = torch.einsum("ij,mj->mi", c2w, points)[:, :3]
            normals = torch.einsum("ij,mj->mi", c2w[:3, :3], normals_cam)
            normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-8)
            self._append_points(points, colors, normals, features)
            self.mask_manager.assign_new_points(points.shape[0])

    def _project_gt_equivalents(self) -> dict:
        if self.dataset_name != "scannet":
            raise NotImplementedError("GT-equivalent build currently supports ScanNet only.")
        pred_points = self.points[: self.n_points].cpu().numpy()
        pred_normals, pred_instance_labels, stats = project_gt_equivalent_attributes(
            pred_points,
            self.scene_name,
            self.scannet_raw_root,
            self.normals_noise_deg,
            self.instance_num_instances,
            self.instance_noise_frac,
            self.instance_seed,
        )
        self.normals[: self.n_points] = torch.from_numpy(pred_normals).to(self.device)
        self.mask_manager.point_labels = pred_instance_labels.astype(np.int32, copy=False)
        return stats

    def save(self, output_dir: Path, stats: dict) -> dict:
        equiv_stats = self._project_gt_equivalents()
        output_dir.mkdir(parents=True, exist_ok=True)
        save_start = time.perf_counter()
        progress = tqdm(total=4, desc=f"{output_dir.name} save", unit="stage", dynamic_ncols=True)
        timings = {}
        try:
            progress.set_postfix_str("write ply", refresh=True)
            stage_start = time.perf_counter()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points[: self.n_points].cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(self.colors[: self.n_points].cpu().numpy().astype(np.float32) / 255.0)
            pcd.normals = o3d.utility.Vector3dVector(self.normals[: self.n_points].cpu().numpy())
            o3d.io.write_point_cloud(str(output_dir / "rgb_map.ply"), pcd)
            timings["write_ply_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("store clip", refresh=True)
            stage_start = time.perf_counter()
            self.feature_tmp_file.flush()
            self.feature_tmp_file.close()
            with open(output_dir / CLIP_FEATURE_FILE, "wb") as f:
                np.lib.format.write_array_header_2_0(
                    f,
                    {
                        "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                        "fortran_order": False,
                        "shape": (self.n_features, self.clip_extractor.feature_dim),
                    },
                )
                with open(self.feature_tmp_path, "rb") as src:
                    shutil.copyfileobj(src, f, length=16 * 1024 * 1024)
            timings["store_clip_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("instance labels", refresh=True)
            stage_start = time.perf_counter()
            np.save(output_dir / "instance_labels.npy", self.mask_manager.point_labels)
            timings["instance_labels_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("stats", refresh=True)
            stage_start = time.perf_counter()
            stats = {
                **stats,
                "instance_supervision": "gt_equivalent_projection",
                "instance_label_path": "instance_labels.npy",
                "clip_feature_path": CLIP_FEATURE_FILE,
                "clip_feature_storage": "npy",
                "rgb_normal_point_fusion": True,
                "clip_feature_mode": "clip_textregion",
                "clip_feature_fusion": False,
                "textregion_supervision": "gt" if self.use_inst_gt else "sam",
                "instance_tracking_backend": "gt_equivalent_projection",
                "sam2_model_level_track": self.sam2_model_level_track,
                "sam2_model_level_track_used": False,
                **equiv_stats,
            }
            if self.use_inst_gt:
                stats["sam_model_level_textregion"] = "gt_masks"
            else:
                stats.update(
                    {
                        "sam_model_level_inst": self.sam_model_level_inst,
                        "sam_model_type_inst": self.mask_manager.seed_mask_extractor.model_type,
                        "sam_checkpoint_path_inst": str(self.mask_manager.seed_mask_extractor.checkpoint_path),
                        "sam_model_level_textregion": self.sam_model_level_textregion,
                        "sam_model_type_textregion": self.mask_manager.textregion_mask_extractor.model_type,
                        "sam_checkpoint_path_textregion": str(self.mask_manager.textregion_mask_extractor.checkpoint_path),
                    }
                )
            with open(output_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            timings["stats_sec"] = time.perf_counter() - stage_start
            progress.update()
        finally:
            progress.close()
            if not self.feature_tmp_file.closed:
                self.feature_tmp_file.close()
            shutil.rmtree(self.feature_tmpdir, ignore_errors=True)
        timings["save_total_sec"] = time.perf_counter() - save_start
        return timings


def main(args: argparse.Namespace) -> None:
    run_start = time.perf_counter()
    dataset_name = args.dataset_name.lower()
    output_dir = Path(args.output_root) / canonical_dataset_name(dataset_name) / args.scene_name
    dataset_load_start = time.perf_counter()
    dataset = load_dataset(dataset_name, args.scene_name, args.frame_limit)
    dataset_load_sec = time.perf_counter() - dataset_load_start
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapper = GTEquivalentRGBMapper(
        intrinsics=dataset.intrinsics,
        device=device,
        map_every=args.map_every,
        downscale_res=args.downscale_res,
        k_pooling=args.k_pooling,
        max_frame_points=args.max_frame_points,
        match_distance_th=args.match_distance_th,
        dataset_name=dataset_name,
        scene_name=args.scene_name,
        use_inst_gt=args.use_inst_gt,
        sam_model_level_inst=args.sam_model_level_inst,
        sam_model_level_textregion=args.sam_model_level_textregion,
        sam2_model_level_track=args.sam2_model_level_track,
        scannet_raw_root=args.scannet_raw_root,
        normals_noise_deg=args.normals_noise_deg,
        instance_num_instances=args.instance_num_instances,
        instance_noise_frac=args.instance_noise_frac,
        instance_seed=args.instance_seed,
    )

    progress = tqdm(range(len(dataset)), desc=args.scene_name, unit="frame")
    frame_loop_start = time.perf_counter()
    for frame_id in progress:
        mapper.add_frame(dataset[frame_id])
        progress.set_postfix(points=mapper.n_points, refresh=False)
    frame_loop_sec = time.perf_counter() - frame_loop_start

    save_timings = mapper.save(
        output_dir,
        {
            "dataset_name": canonical_dataset_name(dataset_name),
            "scene_name": args.scene_name,
            "n_frames": len(dataset),
            "n_points": mapper.n_points,
            "has_normals": True,
            "device": device,
            "map_every": mapper.map_every,
            "downscale_res": args.downscale_res,
            "k_pooling": mapper.k_pooling,
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
            "rgb_normal_point_fusion": True,
            "clip_feature_mode": "clip_textregion",
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
    print(json.dumps({"timing": timing_summary}, indent=2))
    print(output_dir / "rgb_map.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a GT-equivalent RGB pointcloud map from RGB-D + GT poses.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet", "replica", "scannet"])
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--output_root", default=str(OUTPUT_DIR))
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--map_every", type=int, default=DEFAULT_MAP_EVERY)
    parser.add_argument("--downscale_res", type=int, default=DEFAULT_DOWNSCALE_RES)
    parser.add_argument("--k_pooling", type=int, default=DEFAULT_K_POOLING)
    parser.add_argument("--max_frame_points", type=int, default=DEFAULT_MAX_FRAME_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)
    parser.add_argument("--sam-model-level-inst", type=int, choices=sorted(SAM1_LEVELS), default=13)
    parser.add_argument("--sam-model-level-textregion", type=int, choices=sorted(SAM1_LEVELS), default=13)
    parser.add_argument("--sam2-model-level-track", type=int, choices=sorted(SAM2_LEVELS), default=24)
    parser.add_argument("--use-inst-gt", action="store_true", help="Use decoded ScanNet instance-filt masks instead of SAM for TextRegion masks.")
    parser.add_argument("--scannet-raw-root", default=str(DEFAULT_SCANNET_RAW_ROOT))
    parser.add_argument("--normals-noise-deg", type=float, default=0.0)
    parser.add_argument("--instance-num-instances", type=int, default=0, help="Target instance count after GT projection. <=0 keeps the projected GT count.")
    parser.add_argument("--instance-noise-frac", type=float, default=0.0, help="Fraction of projected instance labels to randomly corrupt.")
    parser.add_argument("--instance-seed", type=int, default=0)
    main(parser.parse_args())
