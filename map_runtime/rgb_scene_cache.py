from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import open3d as o3d
import torch
from tqdm.auto import tqdm

from map_runtime import geometry
from map_runtime.defaults import (
    CACHE_MANIFEST_FILE,
    CLIP_FEATURE_FILE,
    CLIP_LOAD_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    FRAME_CACHE_DIR,
    FRAME_CACHE_FILE_TEMPLATE,
    POINT_BIRTH_FRAME_FILE,
    STATS_PATH,
    TIMING_PATH,
)
from map_runtime.rgb_map_utils import (
    DenseCLIPExtractor,
    compute_normals_from_depth,
    invert_rigid_transform,
)
from map_runtime.sam_masks import SAMMaskExtractor, SAMMaskExtractorConfig
from map_runtime.scene import canonical_dataset_name, get_tracked_pose, load_dataset_and_slam

def as_int(value) -> int:
    return int(float(value))


class RGBSceneCacheBuilder:
    def __init__(
        self,
        intrinsics: np.ndarray,
        device: str,
        map_every: int,
        max_total_points: int,
        match_distance_th: float,
    ) -> None:
        self.device = device
        self.cam_intrinsics = torch.tensor(intrinsics.astype(np.float32), device=device)
        self.map_every = max(1, int(map_every))
        self.max_total_points = as_int(max_total_points)
        self.match_distance_th = float(match_distance_th)
        self.clip_extractor = DenseCLIPExtractor(device)
        self.textregion_mask_extractor = SAMMaskExtractor(
            device,
            config=SAMMaskExtractorConfig(),
        )

        self.n_points = 0
        self.total_points_original = 0
        self.points = torch.empty((0, 3), device=device)
        self.colors = torch.empty((0, 3), device=device, dtype=torch.uint8)
        self.normals = torch.empty((0, 3), device=device)
        self.color_sum = torch.empty((0, 3), device=device)
        self.normal_sum = torch.empty((0, 3), device=device)
        self.obs_count = torch.empty((0,), device=device)
        self.point_birth_frame = torch.empty((0,), dtype=torch.int32, device=device)
        self.feature_tmpdir = Path(tempfile.mkdtemp(prefix="rgb_cache_feats_"))
        self.feature_tmp_path = self.feature_tmpdir / "clip_feats.bin"
        self.feature_tmp_file = open(self.feature_tmp_path, "wb")
        self.n_features = 0
        self.cached_depth_shape = None
        self.cached_x = None
        self.cached_y = None

    def _select_keep_indices(self, total_count: int, keep_count: int) -> torch.Tensor:
        if keep_count < 0 or keep_count > total_count:
            raise ValueError(f"Invalid keep_count={keep_count} for total_count={total_count}.")
        if keep_count == total_count:
            return torch.arange(total_count, device=self.device)
        if keep_count == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        step = float(total_count) / float(keep_count)
        keep = torch.floor(torch.arange(keep_count, device=self.device, dtype=torch.float64) * step).long()
        return keep

    def _rewrite_feature_store(self, keep_indices: torch.Tensor, original_count: int) -> None:
        if self.feature_tmp_file is not None and not self.feature_tmp_file.closed:
            self.feature_tmp_file.flush()
            self.feature_tmp_file.close()
        keep_np = keep_indices.detach().cpu().numpy()
        src = np.memmap(
            self.feature_tmp_path,
            dtype=np.float16,
            mode="r",
            shape=(int(original_count), int(self.clip_extractor.feature_dim)),
        )
        rewritten_path = self.feature_tmpdir / "clip_feats_resampled.bin"
        chunk_size = 131072
        with open(rewritten_path, "wb") as dst:
            chunk_iter = range(0, keep_np.shape[0], chunk_size)
            for start in tqdm(chunk_iter, desc="downsample clip", unit="chunk", dynamic_ncols=True):
                end = min(start + chunk_size, keep_np.shape[0])
                np.ascontiguousarray(src[keep_np[start:end]], dtype=np.float16).tofile(dst)
        del src
        rewritten_path.replace(self.feature_tmp_path)
        self.feature_tmp_file = open(self.feature_tmp_path, "ab")
        self.n_features = int(keep_np.shape[0])

    @staticmethod
    def _remap_point_ids(point_ids: np.ndarray, point_id_map: np.ndarray) -> np.ndarray:
        point_ids = np.asarray(point_ids, dtype=np.int32)
        remapped = np.full(point_ids.shape, -1, dtype=np.int32)
        valid = point_ids >= 0
        if valid.any():
            remapped[valid] = point_id_map[point_ids[valid]]
        return remapped

    def finalize_for_output(self, output_dir: Path) -> None:
        original_count = int(self.n_points)
        if self.total_points_original == 0:
            self.total_points_original = original_count
        if original_count <= int(self.max_total_points):
            print(
                f"[downsample] not needed: total_points_original={original_count:,} "
                f"<= max_total_points={int(self.max_total_points):,}"
            )
            return
        print(
            f"[downsample] needed: total_points_original={original_count:,} "
            f"> max_total_points={int(self.max_total_points):,}"
        )
        keep = self._select_keep_indices(original_count, int(self.max_total_points))
        keep_np = keep.detach().cpu().numpy()
        point_id_map = np.full((original_count,), -1, dtype=np.int32)
        point_id_map[keep_np] = np.arange(keep_np.shape[0], dtype=np.int32)
        self.points[: keep.shape[0]] = self.points[keep]
        self.colors[: keep.shape[0]] = self.colors[keep]
        self.normals[: keep.shape[0]] = self.normals[keep]
        self.color_sum[: keep.shape[0]] = self.color_sum[keep]
        self.normal_sum[: keep.shape[0]] = self.normal_sum[keep]
        self.obs_count[: keep.shape[0]] = self.obs_count[keep]
        self.point_birth_frame[: keep.shape[0]] = self.point_birth_frame[keep]
        self.n_points = int(keep.shape[0])
        self._rewrite_feature_store(keep, original_count)

        cache_paths = sorted((output_dir / FRAME_CACHE_DIR).glob("*.npz"))
        for cache_path in tqdm(cache_paths, desc="remap frame cache", unit="frame", dynamic_ncols=True):
            with np.load(cache_path) as data:
                frame_cache = {key: np.array(data[key], copy=True) for key in data.files}
            frame_cache["point_ids_before"] = self._remap_point_ids(frame_cache["point_ids_before"], point_id_map)
            frame_cache["point_ids_after"] = self._remap_point_ids(frame_cache["point_ids_after"], point_id_map)
            frame_cache["new_point_mask"] = np.asarray(
                frame_cache["new_point_mask"].astype(bool, copy=False) & (frame_cache["point_ids_after"] >= 0),
                dtype=bool,
            )
            np.savez_compressed(cache_path, **frame_cache)
        print(
            f"[downsample] done: kept {self.n_points:,} / {original_count:,} points "
            f"and remapped {len(cache_paths):,} cached frames"
        )

    def should_map_frame(self, frame_id: int) -> bool:
        return frame_id % self.map_every == 0

    def _ensure_capacity(self, min_capacity: int) -> None:
        if min_capacity <= self.points.shape[0]:
            return
        new_capacity = max(min_capacity, max(1 << 18, self.points.shape[0] * 2))

        def grow(buffer: torch.Tensor, *shape_tail: int) -> torch.Tensor:
            expanded = torch.empty((new_capacity, *shape_tail), device=buffer.device, dtype=buffer.dtype)
            if self.n_points > 0:
                expanded[: self.n_points] = buffer[: self.n_points]
            return expanded

        self.points = grow(self.points, 3)
        self.colors = grow(self.colors, 3)
        self.normals = grow(self.normals, 3)
        self.color_sum = grow(self.color_sum, 3)
        self.normal_sum = grow(self.normal_sum, 3)
        self.obs_count = grow(self.obs_count)
        self.point_birth_frame = grow(self.point_birth_frame)

    def _get_cached_grids(self, depth_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cached_depth_shape != depth_shape:
            h, w = depth_shape
            full_y, full_x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
            self.cached_x = full_x
            self.cached_y = full_y
            self.cached_depth_shape = depth_shape
        return self.cached_x, self.cached_y

    def _append_points(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        normals: torch.Tensor,
        features: torch.Tensor | None = None,
        *,
        birth_frame_id: int,
    ) -> None:
        start = self.n_points
        end = start + int(points.shape[0])
        self._ensure_capacity(end)
        self.points[start:end] = points
        self.colors[start:end] = colors
        self.normals[start:end] = normals
        self.color_sum[start:end] = colors.float()
        self.normal_sum[start:end] = normals.float()
        self.obs_count[start:end] = 1
        self.point_birth_frame[start:end] = int(birth_frame_id)
        self.n_points = end
        if features is None:
            return
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

    def add_frame(self, frame_data, c2w_override=None) -> dict:
        frame_id, image_np, depth_np = frame_data[:3]
        is_seed_frame = self.should_map_frame(frame_id)
        c2w_np = frame_data[3] if c2w_override is None else c2w_override

        h, w = depth_np.shape
        point_ids_before = torch.full((h, w), -1, dtype=torch.int32, device=self.device)
        point_ids_after = torch.full((h, w), -1, dtype=torch.int32, device=self.device)
        new_point_mask = np.zeros((h, w), dtype=bool)
        filter_depth_mask = np.zeros((h, w), dtype=bool)
        filter_unmatched_mask = np.zeros((h, w), dtype=bool)
        filter_normal_mask = np.zeros((h, w), dtype=bool)
        valid_pose = c2w_np is not None

        if valid_pose:
            if isinstance(c2w_np, torch.Tensor):
                c2w = c2w_np.to(self.device, dtype=torch.float32)
                c2w_np = c2w.detach().cpu().numpy()
            else:
                c2w_np = np.asarray(c2w_np, dtype=np.float32)
                c2w = torch.from_numpy(c2w_np).to(self.device)
            if np.isinf(c2w_np).any() or np.isnan(c2w_np).any():
                valid_pose = False

        if not valid_pose:
            return {
                "frame_id": int(frame_id),
                "is_seed_frame": bool(is_seed_frame),
                "valid_pose": False,
                "rgb": np.asarray(image_np, dtype=np.uint8),
                "depth": np.asarray(depth_np, dtype=np.float32),
                "c2w": None if c2w_np is None else np.asarray(c2w_np, dtype=np.float32),
                "point_ids_before": point_ids_before.cpu().numpy(),
                "point_ids_after": point_ids_after.cpu().numpy(),
                "new_point_mask": new_point_mask,
                "filter_depth_mask": filter_depth_mask,
                "filter_unmatched_mask": filter_unmatched_mask,
                "filter_normal_mask": filter_normal_mask,
            }

        depth = torch.from_numpy(depth_np).to(self.device)
        image = torch.from_numpy(image_np).to(self.device)
        full_image = image
        x, y = self._get_cached_grids((h, w))
        mask = depth > 0

        if self.n_points > 0:
            frustum_corners = geometry.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
            w2c = invert_rigid_transform(c2w)
            frustum_mask = geometry.compute_frustum_point_ids(self.points[: self.n_points], frustum_corners, device=self.device)
            if frustum_mask.numel() > 0:
                matched_ids, matches = geometry.match_3d_points_to_2d_pixels(
                    depth,
                    w2c,
                    self.points[frustum_mask],
                    self.cam_intrinsics,
                    self.match_distance_th,
                )
                if matches.numel() > 0:
                    global_ids = frustum_mask[matched_ids].long()
                    point_ids_before[matches[:, 1], matches[:, 0]] = global_ids.to(point_ids_before.dtype)
                    point_ids_after[matches[:, 1], matches[:, 0]] = global_ids.to(point_ids_after.dtype)
                    mask[matches[:, 1], matches[:, 0]] = False

        if is_seed_frame:
            tr_labels_np = self.textregion_mask_extractor.extract_labels(image_np)
            tr_labels_full = torch.from_numpy(tr_labels_np).to(self.device)
            filter_depth_mask = np.asarray((depth > 0).detach().cpu().numpy(), dtype=bool)
            filter_unmatched_mask = np.asarray(mask.detach().cpu().numpy(), dtype=bool)
            normals_cam, normal_valid = compute_normals_from_depth(x, y, depth, self.cam_intrinsics)
            visible_existing = (point_ids_after >= 0) & normal_valid
            if visible_existing.any():
                visible_ids = point_ids_after[visible_existing]
                visible_colors = image[visible_existing].reshape(-1, 3)
                visible_normals = normals_cam[visible_existing].reshape(-1, 3)
                visible_normals = torch.einsum("ij,mj->mi", c2w[:3, :3], visible_normals)
                visible_normals = visible_normals / torch.linalg.norm(visible_normals, dim=1, keepdim=True).clamp_min(1e-8)
                self._update_observed_points(visible_ids, visible_colors, visible_normals)

            mask = mask & normal_valid
            filter_normal_mask = np.asarray(mask.detach().cpu().numpy(), dtype=bool)
            if mask.any():
                x_keep = x[mask]
                y_keep = y[mask]
                depth_keep = depth[mask]
                colors = image[mask].reshape(-1, 3)
                normals_cam = normals_cam[mask].reshape(-1, 3)
                candidate_count = int(depth_keep.shape[0])
                self.total_points_original += candidate_count
                if depth_keep.numel() > 0:
                    dense_clip = self.clip_extractor.extract_dense(full_image, tr_labels_full)
                    features = dense_clip[y_keep, x_keep].half()
                    x_3d = (x_keep - self.cam_intrinsics[0, 2]) * depth_keep / self.cam_intrinsics[0, 0]
                    y_3d = (y_keep - self.cam_intrinsics[1, 2]) * depth_keep / self.cam_intrinsics[1, 1]
                    points = torch.stack((x_3d, y_3d, depth_keep, torch.ones_like(depth_keep)), dim=1)
                    points = torch.einsum("ij,mj->mi", c2w, points)[:, :3]
                    normals = torch.einsum("ij,mj->mi", c2w[:3, :3], normals_cam)
                    normals = normals / torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-8)
                    old_n = self.n_points
                    new_ids = torch.arange(old_n, old_n + points.shape[0], device=self.device, dtype=torch.int32)
                    point_ids_after[y_keep, x_keep] = new_ids
                    new_point_mask[y_keep.cpu().numpy(), x_keep.cpu().numpy()] = True
                    self._append_points(
                        points,
                        colors,
                        normals,
                        features,
                        birth_frame_id=int(frame_id),
                    )

        return {
            "frame_id": int(frame_id),
            "is_seed_frame": bool(is_seed_frame),
            "valid_pose": True,
            "rgb": np.asarray(image_np, dtype=np.uint8),
            "depth": np.asarray(depth_np, dtype=np.float32),
            "c2w": np.asarray(c2w_np, dtype=np.float32),
            "point_ids_before": point_ids_before.cpu().numpy(),
            "point_ids_after": point_ids_after.cpu().numpy(),
            "new_point_mask": new_point_mask,
            "filter_depth_mask": filter_depth_mask,
            "filter_unmatched_mask": filter_unmatched_mask,
            "filter_normal_mask": filter_normal_mask,
        }

    def save(self, output_dir: Path, stats: dict) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_start = time.perf_counter()
        timings = {}

        stage_start = time.perf_counter()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points[: self.n_points].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.colors[: self.n_points].cpu().numpy().astype(np.float32) / 255.0)
        pcd.normals = o3d.utility.Vector3dVector(self.normals[: self.n_points].cpu().numpy())
        o3d.io.write_point_cloud(str(output_dir / "rgb_map.ply"), pcd)
        timings["write_ply_sec"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        with open(output_dir / CLIP_FEATURE_FILE, "wb") as handle:
            np.lib.format.write_array_header_2_0(
                handle,
                {
                    "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                    "fortran_order": False,
                    "shape": (self.n_points, self.clip_extractor.feature_dim),
                },
            )
            self.feature_tmp_file.flush()
            self.feature_tmp_file.close()
            with open(self.feature_tmp_path, "rb") as src:
                shutil.copyfileobj(src, handle, length=16 * 1024 * 1024)
        timings["store_clip_sec"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        np.save(output_dir / POINT_BIRTH_FRAME_FILE, self.point_birth_frame[: self.n_points].cpu().numpy().astype(np.int32, copy=False))
        timings["point_birth_frame_sec"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        with open(output_dir / STATS_PATH, "w") as handle:
            json.dump(stats, handle, indent=2)
        timings["stats_sec"] = time.perf_counter() - stage_start

        if self.feature_tmp_file is not None and not self.feature_tmp_file.closed:
            self.feature_tmp_file.close()
        if self.feature_tmpdir is not None:
            shutil.rmtree(self.feature_tmpdir, ignore_errors=True)

        timings["save_total_sec"] = time.perf_counter() - save_start
        return timings


def write_frame_cache(output_dir: Path, frame_cache: dict) -> None:
    frame_cache_dir = output_dir / FRAME_CACHE_DIR
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = frame_cache_dir / FRAME_CACHE_FILE_TEMPLATE.format(frame_id=int(frame_cache["frame_id"]))
    c2w = frame_cache["c2w"]
    np.savez_compressed(
        cache_path,
        frame_id=np.int32(frame_cache["frame_id"]),
        is_seed_frame=np.bool_(frame_cache["is_seed_frame"]),
        valid_pose=np.bool_(frame_cache["valid_pose"]),
        rgb=frame_cache["rgb"],
        depth=frame_cache["depth"],
        c2w=np.full((4, 4), np.nan, dtype=np.float32) if c2w is None else c2w.astype(np.float32, copy=False),
        point_ids_before=frame_cache["point_ids_before"].astype(np.int32, copy=False),
        point_ids_after=frame_cache["point_ids_after"].astype(np.int32, copy=False),
        new_point_mask=frame_cache["new_point_mask"].astype(bool, copy=False),
        filter_depth_mask=frame_cache["filter_depth_mask"].astype(bool, copy=False),
        filter_unmatched_mask=frame_cache["filter_unmatched_mask"].astype(bool, copy=False),
        filter_normal_mask=frame_cache["filter_normal_mask"].astype(bool, copy=False),
    )


def build_run_stats(
    *,
    builder: RGBSceneCacheBuilder,
    config: dict,
    dataset_name: str,
    scene_name: str,
    device: str,
    n_frames: int,
) -> dict:
    return {
        "dataset_name": canonical_dataset_name(dataset_name),
        "scene_name": scene_name,
        "n_frames": n_frames,
        "n_points": builder.n_points,
        "has_normals": True,
        "device": device,
        "slam_module": config["slam"].get("slam_module", "vanilla"),
        "slam_close_loops": bool(config["slam"].get("close_loops", True)),
        "map_every": builder.map_every,
        "max_total_points": builder.max_total_points,
        "total_points_original": builder.total_points_original,
        "match_distance_th": builder.match_distance_th,
        "clip_model_name": CLIP_MODEL_NAME,
        "clip_pretrained": CLIP_PRETRAINED,
        "clip_load_size": CLIP_LOAD_SIZE,
        "clip_feature_dim": builder.clip_extractor.feature_dim,
        "clip_feature_dtype": "float16",
        "clip_feature_path": CLIP_FEATURE_FILE,
        "point_birth_frame_path": POINT_BIRTH_FRAME_FILE,
        "frame_cache_dir": FRAME_CACHE_DIR,
        "rgb_normal_point_fusion": True,
        "textregion_supervision": "sam",
        "instance_supervision": "none",
        "sam_model_level_textregion": builder.textregion_mask_extractor.model_level,
        "sam_model_type_textregion": builder.textregion_mask_extractor.model_type,
        "sam_checkpoint_path_textregion": str(builder.textregion_mask_extractor.checkpoint_path),
        "sam_config_textregion": builder.textregion_mask_extractor.config_path,
    }


def write_cache_manifest(
    output_dir: Path,
    *,
    dataset_name: str,
    scene_name: str,
    intrinsics: np.ndarray,
    source_width: int,
    source_height: int,
    n_frames: int,
    n_points: int,
    map_every: int,
) -> None:
    manifest = {
        "version": 1,
        "dataset_name": canonical_dataset_name(dataset_name),
        "scene_name": scene_name,
        "n_frames": int(n_frames),
        "n_points": int(n_points),
        "map_every": int(map_every),
        "intrinsics": intrinsics.astype(np.float32).tolist(),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "frame_cache_dir": FRAME_CACHE_DIR,
        "rgb_map_path": "rgb_map.ply",
        "clip_feature_path": CLIP_FEATURE_FILE,
        "point_birth_frame_path": POINT_BIRTH_FRAME_FILE,
        "stats_path": STATS_PATH,
        "timing_path": TIMING_PATH,
    }
    with open(output_dir / CACHE_MANIFEST_FILE, "w") as handle:
        json.dump(manifest, handle, indent=2)


def run_scene_cache_build(
    *,
    dataset_name: str,
    scene_name: str,
    output_root: str | Path,
    frame_limit: int | None,
    slam_module: str | None,
    disable_loop_closure: bool,
    config_path: str,
    map_every: int,
    max_total_points: int,
    match_distance_th: float,
) -> tuple[Path, dict]:
    run_start = time.perf_counter()
    output_dir = Path(output_root) / canonical_dataset_name(dataset_name) / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_load_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config, dataset, slam_backbone = load_dataset_and_slam(
        dataset_name=dataset_name,
        scene_name=scene_name,
        device=device,
        frame_limit=frame_limit,
        config_path=config_path,
        slam_module=slam_module,
        disable_loop_closure=disable_loop_closure,
    )
    dataset_load_sec = time.perf_counter() - dataset_load_start
    sample_image = dataset[0][1]
    source_height, source_width = sample_image.shape[:2]

    builder = RGBSceneCacheBuilder(
        intrinsics=dataset.intrinsics,
        device=device,
        map_every=map_every,
        max_total_points=max_total_points,
        match_distance_th=match_distance_th,
    )

    progress = tqdm(range(len(dataset)), desc=scene_name, unit="frame")
    frame_loop_start = time.perf_counter()
    try:
        for frame_id in progress:
            frame_data = dataset[frame_id]
            estimated_c2w = get_tracked_pose(slam_backbone, frame_data)
            frame_cache = builder.add_frame(frame_data, c2w_override=estimated_c2w)
            write_frame_cache(output_dir, frame_cache)
            progress.set_postfix(
                points=builder.n_points,
                refresh=False,
            )
    finally:
        progress.close()
    frame_loop_sec = time.perf_counter() - frame_loop_start

    builder.finalize_for_output(output_dir)

    stats = build_run_stats(
        builder=builder,
        config=config,
        dataset_name=dataset_name,
        scene_name=scene_name,
        device=device,
        n_frames=len(dataset),
    )
    save_timings = builder.save(output_dir, stats)
    write_cache_manifest(
        output_dir,
        dataset_name=dataset_name,
        scene_name=scene_name,
        intrinsics=dataset.intrinsics,
        source_width=source_width,
        source_height=source_height,
        n_frames=len(dataset),
        n_points=builder.n_points,
        map_every=map_every,
    )
    timing_summary = {
        "dataset_load_sec": dataset_load_sec,
        "frame_loop_sec": frame_loop_sec,
        "save": save_timings,
        "total_sec": time.perf_counter() - run_start,
    }
    with open(output_dir / TIMING_PATH, "w") as handle:
        json.dump(timing_summary, handle, indent=2)
    del slam_backbone
    return output_dir, timing_summary
