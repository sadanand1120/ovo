import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import cv2  # Keep OpenCV loaded before torch in the container env.
import numpy as np
import open3d as o3d
import torch
from tqdm.auto import tqdm

from map_runtime import geometry
from map_runtime.defaults import (
    CLIP_FEATURE_FILE,
    CLIP_LOAD_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MAP_EVERY,
    DEFAULT_MATCH_DISTANCE_TH,
    DEFAULT_MAX_TOTAL_POINTS,
    DEFAULT_RGB_MAP_OUTPUT_ROOT,
    INSTANCE_GID_SLOTS_FILE,
    INSTANCE_LABEL_FILE,
    INSTANCE_SUPPORT_FILE,
    TIMING_PATH,
)
from map_runtime.rgb_map_utils import (
    DenseCLIPExtractor,
    compute_normals_from_depth,
    invert_rigid_transform,
)
from map_runtime.sam_instance_runtime import SAMInstanceRuntime, SAMInstanceRuntimeConfig
from map_runtime.sam2_tracking import _input_dir, _sam2_apply_postprocessing, _sam2_hydra_overrides, _sam2_levels, _sam2_mode
from map_runtime.scene import (
    canonical_dataset_name,
    get_tracked_pose,
    load_dataset_and_slam,
)

def as_int(value) -> int:
    return int(float(value))


class RGBMapper:
    def __init__(
        self,
        intrinsics: np.ndarray,
        device: str,
        total_frames: int,
        map_every: int,
        max_total_points: int,
        match_distance_th: float,
        instance_config: SAMInstanceRuntimeConfig | None = None,
    ) -> None:
        self.device = device
        self.cam_intrinsics = torch.tensor(intrinsics.astype(np.float32), device=device)
        self.total_frames = int(total_frames)
        self.map_every = max(1, int(map_every))
        self.max_total_points = as_int(max_total_points)
        self.match_distance_th = float(match_distance_th)
        self.clip_extractor = DenseCLIPExtractor(device)
        self.instance_runtime = SAMInstanceRuntime(
            config=SAMInstanceRuntimeConfig() if instance_config is None else instance_config,
            device=device,
            total_frames=self.total_frames,
            map_every=self.map_every,
            n_points=0,
        )

        self.n_points = 0
        self.total_points_original = 0
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

    def finalize_for_output(self) -> None:
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
        self.points[: keep.shape[0]] = self.points[keep]
        self.colors[: keep.shape[0]] = self.colors[keep]
        self.normals[: keep.shape[0]] = self.normals[keep]
        self.color_sum[: keep.shape[0]] = self.color_sum[keep]
        self.normal_sum[: keep.shape[0]] = self.normal_sum[keep]
        self.obs_count[: keep.shape[0]] = self.obs_count[keep]
        self.n_points = int(keep.shape[0])
        self.instance_runtime.keep_point_ids(keep.detach().cpu().numpy())
        self._rewrite_feature_store(keep, original_count)
        print(f"[downsample] done: kept {self.n_points:,} / {original_count:,} points")

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

    def _get_cached_grids(self, depth_shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cached_depth_shape != depth_shape:
            h, w = depth_shape
            full_y, full_x = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
            self.cached_x = full_x
            self.cached_y = full_y
            self.cached_depth_shape = depth_shape
        return self.cached_x, self.cached_y

    def _append_points(self, points: torch.Tensor, colors: torch.Tensor, normals: torch.Tensor, features: torch.Tensor | None = None) -> None:
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

    def add_frame(self, frame_data, c2w_override=None, is_seed_frame: bool | None = None) -> None:
        frame_id, image_np, depth_np = frame_data[:3]
        if is_seed_frame is None:
            is_seed_frame = self.should_map_frame(frame_id)
        c2w_np = frame_data[3] if c2w_override is None else c2w_override
        h, w = depth_np.shape
        point_ids_after = torch.full((h, w), -1, dtype=torch.int32, device=self.device)
        valid_pose = c2w_np is not None
        seed_labels_np = None

        if valid_pose:
            if isinstance(c2w_np, torch.Tensor):
                c2w = c2w_np.to(self.device, dtype=torch.float32)
                c2w_np = c2w.detach().cpu().numpy()
            else:
                c2w_np = np.asarray(c2w_np, dtype=np.float32)
                c2w = torch.from_numpy(c2w_np).to(self.device)
            valid_pose = not (np.isinf(c2w_np).any() or np.isnan(c2w_np).any())

        if valid_pose:
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
                        point_ids_after[matches[:, 1], matches[:, 0]] = global_ids.to(point_ids_after.dtype)
                        mask[matches[:, 1], matches[:, 0]] = False

            if is_seed_frame:
                seed_labels_np = self.instance_runtime.extract_seed_labels(image_np)
                tr_labels_full = torch.from_numpy(seed_labels_np).to(self.device)
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
                        self._append_points(points, colors, normals, features)

        self.instance_runtime.ensure_point_capacity(self.n_points)
        self.instance_runtime.process_frame(
            frame_id=int(frame_id),
            is_seed_frame=bool(is_seed_frame),
            valid_pose=bool(valid_pose),
            rgb=image_np,
            point_ids_after=point_ids_after.detach().cpu().numpy(),
            seed_labels=seed_labels_np,
        )

    def save(self, output_dir: Path, stats: dict) -> dict:
        self.instance_runtime.close()
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
            with open(output_dir / CLIP_FEATURE_FILE, "wb") as f:
                np.lib.format.write_array_header_2_0(
                    f,
                    {
                        "descr": np.lib.format.dtype_to_descr(np.dtype(np.float16)),
                        "fortran_order": False,
                        "shape": (self.n_points, self.clip_extractor.feature_dim),
                    },
                )
                self.feature_tmp_file.flush()
                self.feature_tmp_file.close()
                with open(self.feature_tmp_path, "rb") as src:
                    shutil.copyfileobj(src, f, length=16 * 1024 * 1024)
            timings["store_clip_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("instance labels", refresh=True)
            stage_start = time.perf_counter()
            np.save(output_dir / INSTANCE_GID_SLOTS_FILE, self.instance_runtime.export_point_gids())
            np.save(output_dir / INSTANCE_LABEL_FILE, self.instance_runtime.export_collapsed_labels())
            np.save(output_dir / INSTANCE_SUPPORT_FILE, self.instance_runtime.export_support_counts())
            timings["instance_labels_sec"] = time.perf_counter() - stage_start
            progress.update()

            progress.set_postfix_str("stats", refresh=True)
            stage_start = time.perf_counter()
            sam_mask_config = self.instance_runtime.seed_mask_extractor.config
            track_level = int(self.instance_runtime.config.tracker.model_level)
            stats = {
                **stats,
                "instance_supervision": "sam",
                "textregion_supervision": "sam",
                "instance_gid_slots_path": INSTANCE_GID_SLOTS_FILE,
                "instance_label_path": INSTANCE_LABEL_FILE,
                "instance_support_path": INSTANCE_SUPPORT_FILE,
                "clip_feature_path": CLIP_FEATURE_FILE,
                "clip_feature_storage": "npy",
                "rgb_normal_point_fusion": True,
                "clip_feature_mode": "clip_textregion",
                "sam_sort_mode": sam_mask_config.sort_mode,
                "sam_min_mask_area_perc": sam_mask_config.min_mask_area_perc,
                "sam_points_per_side": sam_mask_config.points_per_side,
                "sam_points_per_batch": sam_mask_config.points_per_batch,
                "sam_pred_iou_thresh": sam_mask_config.pred_iou_thresh,
                "sam_stability_score_thresh": sam_mask_config.stability_score_thresh,
                "sam_stability_score_offset": sam_mask_config.stability_score_offset,
                "sam_mask_threshold": sam_mask_config.mask_threshold,
                "sam_box_nms_thresh": sam_mask_config.box_nms_thresh,
                "sam_crop_n_layers": sam_mask_config.crop_n_layers,
                "sam_crop_nms_thresh": sam_mask_config.crop_nms_thresh,
                "sam_crop_overlap_ratio": sam_mask_config.crop_overlap_ratio,
                "sam_crop_n_points_downscale_factor": sam_mask_config.crop_n_points_downscale_factor,
                "sam_min_mask_region_area": sam_mask_config.min_mask_region_area,
                "sam_output_mode": sam_mask_config.output_mode,
                "sam_use_m2m": sam_mask_config.use_m2m,
                "sam_multimask_output": sam_mask_config.multimask_output,
                "sam_score_pred_iou_power": sam_mask_config.score_pred_iou_power,
                "sam_score_stability_power": sam_mask_config.score_stability_power,
                "sam_score_area_power": sam_mask_config.score_area_power,
                "sam_mask_overlap_rescore_thresh": sam_mask_config.mask_overlap_rescore_thresh,
                "sam_mask_overlap_rescore_power": sam_mask_config.mask_overlap_rescore_power,
                "sam_mask_dedupe_iou_thresh": sam_mask_config.mask_dedupe_iou_thresh,
                "sam_mask_containment_thresh": sam_mask_config.mask_containment_thresh,
                "sam_amg_extractors_shared": True,
                "instance_point_gid_slots": int(self.instance_runtime.config.pipeline.point_gid_slots),
                "instance_reuse_inside_frac_th": float(self.instance_runtime.config.pipeline.reuse_inside_frac_th),
                "instance_reuse_outside_frac_th": float(self.instance_runtime.config.pipeline.reuse_outside_frac_th),
                "instance_min_mask_points": int(self.instance_runtime.config.pipeline.min_mask_points),
                "instance_min_track_visible_points": int(self.instance_runtime.config.pipeline.min_track_visible_points),
                "instance_prune_start_at": int(self.instance_runtime.config.pipeline.prune_start_at),
                "instance_prune_every_frames": int(self.instance_runtime.config.pipeline.prune_every_frames),
                "instance_prune_min_support_perc": float(self.instance_runtime.config.pipeline.prune_min_support_perc),
                "instance_prune_min_points_perc": float(self.instance_runtime.config.pipeline.prune_min_points_perc),
                **self.instance_runtime.stats,
            }
            stats.update(
                {
                    "instance_tracking_backend": "sam2_seed_mask_runtime",
                    "sam2_model_level_track": track_level,
                    "sam2_checkpoint_path_track": str(_input_dir / "sam_ckpts" / _sam2_levels[track_level][0]),
                    "sam2_config_track": _sam2_levels[track_level][1],
                    "sam2_max_num_objects": int(self.instance_runtime.config.tracker.max_num_objects),
                    "sam2_mode": _sam2_mode,
                    "sam2_hydra_overrides": list(_sam2_hydra_overrides),
                    "sam2_apply_postprocessing": _sam2_apply_postprocessing,
                }
            )
            stats.update(
                {
                    "sam_model_level_textregion": self.instance_runtime.seed_mask_extractor.model_level,
                    "sam_model_type_textregion": self.instance_runtime.seed_mask_extractor.model_type,
                    "sam_checkpoint_path_textregion": str(self.instance_runtime.seed_mask_extractor.checkpoint_path),
                    "sam_config_textregion": self.instance_runtime.seed_mask_extractor.config_path,
                    "sam_model_level_inst": self.instance_runtime.seed_mask_extractor.model_level,
                    "sam_model_type_inst": self.instance_runtime.seed_mask_extractor.model_type,
                    "sam_checkpoint_path_inst": str(self.instance_runtime.seed_mask_extractor.checkpoint_path),
                    "sam_config_inst": self.instance_runtime.seed_mask_extractor.config_path,
                }
            )
            with open(output_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            timings["stats_sec"] = time.perf_counter() - stage_start
            progress.update()
        finally:
            progress.set_postfix_str("stats", refresh=True)
            progress.close()
            if self.feature_tmp_file is not None and not self.feature_tmp_file.closed:
                self.feature_tmp_file.close()
            if self.feature_tmpdir is not None:
                shutil.rmtree(self.feature_tmpdir, ignore_errors=True)
        timings["save_total_sec"] = time.perf_counter() - save_start
        return timings


def build_run_stats(
    *,
    mapper: RGBMapper,
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
        "n_points": mapper.n_points,
        "has_normals": True,
        "device": device,
        "slam_module": config["slam"].get("slam_module", "vanilla"),
        "slam_close_loops": bool(config["slam"].get("close_loops", True)),
        "map_every": mapper.map_every,
        "max_total_points": mapper.max_total_points,
        "total_points_original": mapper.total_points_original,
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
        "sam2_seed_frames": 0 if n_frames <= 0 else 1 + ((int(n_frames) - 1) // int(mapper.map_every)),
    }


def run_scene_build(
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
    extra_stats: dict | None = None,
    snapshot_hook=None,
) -> tuple[Path, dict, dict]:
    run_start = time.perf_counter()
    output_dir = Path(output_root) / canonical_dataset_name(dataset_name) / scene_name
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
    mapper = RGBMapper(
        intrinsics=dataset.intrinsics,
        device=device,
        total_frames=len(dataset),
        map_every=map_every,
        max_total_points=max_total_points,
        match_distance_th=match_distance_th,
    )

    progress = tqdm(range(len(dataset)), desc=scene_name, unit="frame")
    frame_loop_start = time.perf_counter()
    try:
        for frame_id in progress:
            frame_data = dataset[frame_id]
            prev_n = mapper.n_points
            estimated_c2w = get_tracked_pose(slam_backbone, frame_data)
            mapper.add_frame(frame_data, c2w_override=estimated_c2w)
            if snapshot_hook is not None:
                snapshot_hook(frame_id, prev_n, mapper.n_points, estimated_c2w)
            progress.set_postfix(
                points=mapper.n_points,
                active=mapper.instance_runtime.num_active_instances(),
                objs=mapper.instance_runtime.num_existing_instances(),
                refresh=False,
            )
    finally:
        progress.close()
    frame_loop_sec = time.perf_counter() - frame_loop_start

    mapper.finalize_for_output()

    stats = build_run_stats(
        mapper=mapper,
        config=config,
        dataset_name=dataset_name,
        scene_name=scene_name,
        device=device,
        n_frames=len(dataset),
    )
    if extra_stats:
        stats.update(extra_stats)
    save_timings = mapper.save(
        output_dir,
        stats,
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
    return output_dir, timing_summary, {
        "dataset_intrinsics": dataset.intrinsics.astype(np.float32, copy=True),
        "source_width": int(source_width),
        "source_height": int(source_height),
    }


def add_build_args(parser: argparse.ArgumentParser, *, default_output_root: str | Path) -> None:
    parser.add_argument("--output_root", default=str(default_output_root))
    parser.add_argument("--frame_limit", type=int, default=None)
    parser.add_argument("--slam_module", type=str, default=None, help="Override slam backend, e.g. vanilla, orbslam, or cuvslam.")
    parser.add_argument("--disable_loop_closure", action="store_true", help="Disable ORB-SLAM loop closure/global BA updates by forcing slam.close_loops=false.")
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH), help="Base runtime config file to load.")
    parser.add_argument("--map_every", type=int, default=DEFAULT_MAP_EVERY)
    parser.add_argument("--max_total_points", type=int, default=DEFAULT_MAX_TOTAL_POINTS)
    parser.add_argument("--match_distance_th", type=float, default=DEFAULT_MATCH_DISTANCE_TH)


def main(args):
    output_dir, timing_summary, _ = run_scene_build(
        dataset_name=args.dataset_name,
        scene_name=args.scene_name,
        output_root=args.output_root,
        frame_limit=args.frame_limit,
        slam_module=args.slam_module,
        disable_loop_closure=args.disable_loop_closure,
        config_path=args.config_path,
        map_every=args.map_every,
        max_total_points=args.max_total_points,
        match_distance_th=args.match_distance_th,
    )
    print(json.dumps({"timing": timing_summary}, indent=2))
    print(output_dir / "rgb_map.ply")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a standalone RGB pointcloud map from RGB-D using the selected SLAM pose backend.")
    parser.add_argument("--dataset_name", required=True, choices=["Replica", "ScanNet"])
    parser.add_argument("--scene_name", required=True)
    add_build_args(parser, default_output_root=DEFAULT_RGB_MAP_OUTPUT_ROOT)
    parsed = parser.parse_args()
    main(parsed)
