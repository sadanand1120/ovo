from __future__ import annotations

from dataclasses import asdict
import html
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from plyfile import PlyData
import torch
from tqdm.auto import tqdm

from map_runtime.debug_panels import color_for_id, compose_instance_debug_grid, overlay_header, render_sorted_label_map
from map_runtime.defaults import CACHE_MANIFEST_FILE, CLIP_FEATURE_FILE, FRAME_CACHE_DIR, POINT_BIRTH_FRAME_FILE
from map_runtime.instance_label_video import write_instance_label_video_from_cache
from map_runtime.sam2_tracking import SAM2VideoTracker, build_label_masks
from map_runtime.sam_instance_runtime import InstanceBucket, SAMInstanceRuntime, SAMInstanceRuntimeConfig


class SceneCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        with open(self.cache_dir / CACHE_MANIFEST_FILE, "r") as handle:
            self.manifest = json.load(handle)
        self.n_frames = int(self.manifest["n_frames"])
        self.n_points = int(self.manifest["n_points"])
        self.map_every = int(self.manifest["map_every"])
        self.frame_cache_dir = self.cache_dir / self.manifest.get("frame_cache_dir", FRAME_CACHE_DIR)
        point_birth_frame_path = self.cache_dir / self.manifest.get("point_birth_frame_path", POINT_BIRTH_FRAME_FILE)
        self.point_birth_frame = np.load(point_birth_frame_path, mmap_mode="r")

    def load_frame(self, frame_id: int) -> dict[str, Any]:
        cache_path = self.frame_cache_dir / f"{int(frame_id):06d}.npz"
        with np.load(cache_path) as data:
            return {
                "frame_id": int(data["frame_id"]),
                "is_seed_frame": bool(data["is_seed_frame"]),
                "valid_pose": bool(data["valid_pose"]),
                "rgb": np.array(data["rgb"], copy=True),
                "depth": np.array(data["depth"], copy=True),
                "c2w": np.array(data["c2w"], copy=True),
                "point_ids_before": np.array(data["point_ids_before"], copy=True),
                "point_ids_after": np.array(data["point_ids_after"], copy=True),
                "new_point_mask": np.array(data["new_point_mask"], copy=True),
                "filter_depth_mask": np.array(data["filter_depth_mask"], copy=True),
                "filter_unmatched_mask": np.array(data["filter_unmatched_mask"], copy=True),
                "filter_normal_mask": np.array(data["filter_normal_mask"], copy=True),
            }


DEBUG_INSTANCE_CACHE_DIR = "cache_instances"
DEBUG_INSTANCE_CACHE_FILE = "full_state.pkl"


def _debug_render_matplotlib_figure(fig) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(buffer[:, :, :3])


def _debug_write_video_frames(frames: list[np.ndarray], output_path: Path, fps: float) -> Path:
    if not frames:
        raise ValueError("No frames were provided for debug video export.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(frames[0].shape[1]), int(frames[0].shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return output_path


class CachedSAMInstanceDebugger(SAMInstanceRuntime):
    def __init__(
        self,
        cache_dir: str | Path,
        config: SAMInstanceRuntimeConfig,
        *,
        device: str,
    ) -> None:
        self.cache = SceneCache(cache_dir)
        super().__init__(
            config=config,
            device=device,
            total_frames=self.cache.n_frames,
            map_every=self.cache.map_every,
            n_points=self.cache.n_points,
        )
        self.reset()

    def reset(self, *, n_points: int | None = None) -> None:
        super().reset(n_points=self.cache.n_points if n_points is None else int(n_points))
        if not hasattr(self, "_debug_map_xyz"):
            self._debug_map_xyz: np.ndarray | None = None
        if not hasattr(self, "_debug_map_rgb"):
            self._debug_map_rgb: np.ndarray | None = None
        self.last_seed_labels: np.ndarray | None = None
        self.last_seed_frame_id = -1
        self.current_view: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.debug_timeline_snapshots: list[dict[str, Any]] = []
        self.debug_gid_stats: dict[int, dict[str, Any]] = {}
        self.debug_last_gt_selection_summary: dict[str, Any] | None = None
        self.debug_last_plots_videos_summary: dict[str, Any] | None = None
        self.debug_last_learned_collapse_summary: dict[str, Any] | None = None
        self.debug_last_learned_fit_frame_id: int | None = None

    def _summarize_frame(self, view: dict[str, Any]) -> None:
        self.history.append(
            {
                "frame_id": int(view["frame_id"]),
                "is_seed_frame": bool(view["is_seed_frame"]),
                "valid_pose": bool(view["valid_pose"]),
                "num_buckets": int(len(self.buckets)),
                "seeded_gids": sorted(int(gid) for gid in self.seeded_gids),
                "pruned_gids": [int(gid) for gid in view["pruned_gids"]],
                "decision_count": int(len(view["decisions"])),
            }
        )

    def _debug_instance_cache_dir(self) -> Path:
        return self.cache.cache_dir / DEBUG_INSTANCE_CACHE_DIR

    def _debug_instance_cache_path(self) -> Path:
        return self._debug_instance_cache_dir() / DEBUG_INSTANCE_CACHE_FILE

    def _debug_optimal_metric_gid_labels_path(self, *, frame_id: int | None = None, min_component_size: int) -> Path:
        target_frame = int(self.current_frame_id if frame_id is None else frame_id)
        return (
            self._debug_instance_cache_dir()
            / f"debug_optimal_metric_gid_labels_frame_{target_frame:06d}_mincomp_{int(min_component_size)}.npy"
        )

    def _debug_collect_timeline_snapshot(self) -> dict[str, Any]:
        frame_id = int(self.current_frame_id)
        num_seed_frames = int(self._num_seed_frames_processed())
        current_map_points = int(self._debug_current_map_points(frame_id))
        bucket_ids = sorted(int(gid) for gid in self.buckets)
        if current_map_points > 0:
            point_perc = np.asarray(
                [100.0 * float(self.buckets[gid].point_count) / float(current_map_points) for gid in bucket_ids],
                dtype=np.float32,
            )
        else:
            point_perc = np.empty((0,), dtype=np.float32)
        if num_seed_frames > 0:
            support_perc = np.asarray(
                [100.0 * float(self.buckets[gid].support_frames) / float(num_seed_frames) for gid in bucket_ids],
                dtype=np.float32,
            )
        else:
            support_perc = np.empty((0,), dtype=np.float32)
        return {
            "frame_id": frame_id,
            "num_seed_frames": num_seed_frames,
            "current_map_points": current_map_points,
            "num_gids": int(len(bucket_ids)),
            "bucket_ids": bucket_ids,
            "point_perc": point_perc,
            "support_perc": support_perc,
        }

    def _debug_export_instance_cache_payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "frame_id": int(self.current_frame_id),
            "next_gid": int(self.next_gid),
            "tracker_frame_idx": int(self.tracker_frame_idx),
            "point_gids": np.array(self.point_gids, copy=True),
            "buckets": [asdict(self.buckets[gid]) for gid in sorted(self.buckets)],
            "seeded_gids": sorted(int(gid) for gid in self.seeded_gids),
            "debug_support_seed_ordinals_by_gid": {
                int(gid): [int(v) for v in values]
                for gid, values in self.debug_support_seed_ordinals_by_gid.items()
            },
            "debug_birth_seed_ordinal_by_gid": {
                int(gid): int(value)
                for gid, value in self.debug_birth_seed_ordinal_by_gid.items()
            },
            "last_seed_labels": None if self.last_seed_labels is None else np.array(self.last_seed_labels, copy=True),
            "last_seed_frame_id": int(self.last_seed_frame_id),
            "current_view": self.current_view,
            "history": list(self.history),
            "debug_timeline_snapshots": list(self.debug_timeline_snapshots),
            "debug_gid_stats": dict(self.debug_gid_stats),
            "debug_last_gt_selection_summary": self.debug_last_gt_selection_summary,
            "debug_last_plots_videos_summary": self.debug_last_plots_videos_summary,
        }

    def _debug_save_instance_cache(self) -> Path:
        cache_dir = self._debug_instance_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._debug_instance_cache_path()
        with open(cache_path, "wb") as handle:
            pickle.dump(self._debug_export_instance_cache_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return cache_path

    def _debug_apply_instance_cache_payload(self, payload: dict[str, Any]) -> None:
        self.point_gids = np.asarray(payload["point_gids"], dtype=np.int32)
        self.buckets = {
            int(entry["gid"]): InstanceBucket(**entry)
            for entry in payload["buckets"]
        }
        self.next_gid = int(payload["next_gid"])
        self.tracker_frame_idx = int(payload["tracker_frame_idx"])
        self.current_frame_id = int(payload["frame_id"])
        self.seeded_gids = {int(gid) for gid in payload["seeded_gids"]}
        self.debug_support_seed_ordinals_by_gid = {
            int(gid): [int(v) for v in values]
            for gid, values in payload["debug_support_seed_ordinals_by_gid"].items()
        }
        self.debug_birth_seed_ordinal_by_gid = {
            int(gid): int(value)
            for gid, value in payload["debug_birth_seed_ordinal_by_gid"].items()
        }
        self.last_seed_labels = None if payload["last_seed_labels"] is None else np.asarray(payload["last_seed_labels"], dtype=np.int32)
        self.last_seed_frame_id = int(payload["last_seed_frame_id"])
        self.current_view = payload["current_view"]
        self.history = list(payload["history"])
        self.debug_timeline_snapshots = list(payload.get("debug_timeline_snapshots", []))
        self.debug_gid_stats = dict(payload.get("debug_gid_stats", {}))
        self.debug_last_gt_selection_summary = payload.get("debug_last_gt_selection_summary")
        self.debug_last_plots_videos_summary = payload.get("debug_last_plots_videos_summary")
        self.debug_last_learned_collapse_summary = None
        self.debug_last_learned_fit_frame_id = None
        self._refresh_gid_scores()

    def _debug_load_instance_cache(self) -> bool:
        cache_path = self._debug_instance_cache_path()
        if not cache_path.exists():
            return False
        with open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
        self.reset()
        self._debug_apply_instance_cache_payload(payload)
        return True

    def step(self) -> dict[str, Any]:
        next_frame_id = int(self.current_frame_id + 1)
        if next_frame_id >= self.total_frames:
            raise IndexError("No more frames left in the cache.")
        frame = self.cache.load_frame(next_frame_id)
        bucket_snapshot_before = self._bucket_snapshot_by_gid()
        point_gids_before = np.array(self.point_gids, copy=True)
        projected_before = self._project_primary_labels(frame["point_ids_after"])
        all_gids_before = self._project_all_gid_rows(frame["point_ids_after"])
        result = self.process_frame(
            frame_id=int(frame["frame_id"]),
            is_seed_frame=bool(frame["is_seed_frame"]),
            valid_pose=bool(frame["valid_pose"]),
            rgb=frame["rgb"],
            point_ids_after=frame["point_ids_after"],
        )
        seed_labels = result.seed_labels
        tracker_labels = result.tracker_labels
        decisions = result.decisions
        pruned_gids = result.pruned_gids
        if frame["is_seed_frame"] and seed_labels is not None:
            self.last_seed_labels = np.array(seed_labels, copy=True)
            self.last_seed_frame_id = int(frame["frame_id"])

        projected_after = self._project_primary_labels(frame["point_ids_after"])
        all_gids_after = self._project_all_gid_rows(frame["point_ids_after"])
        changed_point_ids = np.flatnonzero(np.any(point_gids_before != self.point_gids, axis=1))
        point_gid_changes = [
            {
                "point_id": int(point_id),
                "before": point_gids_before[point_id].astype(np.int32, copy=True).tolist(),
                "after": self.point_gids[point_id].astype(np.int32, copy=True).tolist(),
            }
            for point_id in changed_point_ids.tolist()
        ]
        view = {
            "frame_id": int(frame["frame_id"]),
            "is_seed_frame": bool(frame["is_seed_frame"]),
            "valid_pose": bool(frame["valid_pose"]),
            "rgb": frame["rgb"],
            "depth": frame["depth"],
            "seed_labels": seed_labels,
            "tracker_labels": tracker_labels,
            "projected_before": projected_before,
            "projected_after": projected_after,
            "all_gids_before": all_gids_before,
            "all_gids_after": all_gids_after,
            "last_seed_labels": None if self.last_seed_labels is None else np.array(self.last_seed_labels, copy=True),
            "last_seed_frame_id": int(self.last_seed_frame_id),
            "new_point_mask": frame["new_point_mask"],
            "filter_depth_mask": frame["filter_depth_mask"],
            "filter_unmatched_mask": frame["filter_unmatched_mask"],
            "filter_normal_mask": frame["filter_normal_mask"],
            "decisions": [asdict(decision) for decision in decisions],
            "pruned_gids": [int(gid) for gid in pruned_gids],
            "bucket_snapshot_before": bucket_snapshot_before,
            "bucket_snapshot": self._bucket_snapshot(),
            "bucket_snapshot_after": self._bucket_snapshot_by_gid(),
            "point_gid_changes": point_gid_changes,
            "seeded_gids": sorted(int(gid) for gid in self.seeded_gids),
            "point_ids_before": frame["point_ids_before"],
            "point_ids_after": frame["point_ids_after"],
        }
        self.current_frame_id = int(frame["frame_id"])
        self.current_view = view
        self._summarize_frame(view)
        return view

    def seek(self, frame_id: int) -> dict[str, Any]:
        frame_id = int(frame_id)
        if frame_id < 0 or frame_id >= self.total_frames:
            raise ValueError(f"frame_id must be in [0, {self.total_frames - 1}]")
        if frame_id < self.current_frame_id:
            self.reset()
        while self.current_frame_id < frame_id:
            self.step()
        if self.current_view is None:
            raise RuntimeError("Debugger did not produce a current view.")
        return self.current_view

    def run_all(self, upto: int = -1, *, progress_desc: str = "run_all") -> dict[str, Any]:
        upto = int(upto)
        target_frame = self.total_frames - 1 if upto == -1 else upto
        if target_frame < 0 or target_frame >= self.total_frames:
            raise ValueError(f"upto must be -1 or in [0, {self.total_frames - 1}]")
        if target_frame == self.total_frames - 1 and self._debug_load_instance_cache():
            if self.current_view is None:
                raise RuntimeError("Loaded instance cache but current_view is missing.")
            return self.current_view
        if target_frame < self.current_frame_id:
            self.reset()
        if target_frame == self.total_frames - 1:
            self.debug_timeline_snapshots = []
        progress = tqdm(
            range(self.current_frame_id + 1, target_frame + 1),
            desc=progress_desc,
            unit="frame",
            leave=False,
            dynamic_ncols=True,
        )
        try:
            for _ in progress:
                self.step()
                if target_frame == self.total_frames - 1:
                    self.debug_timeline_snapshots.append(self._debug_collect_timeline_snapshot())
        finally:
            progress.close()
        if self.current_view is None:
            raise RuntimeError("Debugger did not produce a current view.")
        if target_frame == self.total_frames - 1:
            self._debug_save_instance_cache()
        return self.current_view

    def next_seed_frame(self) -> dict[str, Any]:
        target = self.current_frame_id + 1
        while target < self.total_frames and (target % self.cache.map_every) != 0:
            target += 1
        if target >= self.total_frames:
            raise IndexError("No later seed frame exists in the cache.")
        return self.seek(target)

    def render_current_panel(self, enabled_panels: dict[str, bool] | None = None) -> np.ndarray:
        if self.current_view is None:
            raise RuntimeError("Run step() or seek() before rendering.")
        image_shape = self.current_view["rgb"].shape[:2]
        if self.current_view["is_seed_frame"]:
            current_labels = self.current_view["seed_labels"]
            current_title = "Current SAM Masks"
            current_subtitle = f"seed frame={self.current_view['frame_id']}"
        else:
            current_labels = self.current_view["tracker_labels"]
            current_title = "Current SAM Masks"
            current_subtitle = f"tracked frame={self.current_view['frame_id']}"
        if current_labels is None:
            current_labels = np.full(image_shape, -1, dtype=np.int32)
        last_seed_labels = self.current_view["last_seed_labels"]
        if last_seed_labels is None:
            last_seed_labels = np.full(image_shape, -1, dtype=np.int32)
        current_point_mask = np.asarray(self.current_view["point_ids_after"] >= 0, dtype=bool)
        last_seed_frame_id = int(self.current_view["last_seed_frame_id"])
        last_seed_subtitle = "none" if last_seed_frame_id < 0 else f"seed frame={last_seed_frame_id}"
        return compose_instance_debug_grid(
            rgb=self.current_view["rgb"],
            frame_id=int(self.current_view["frame_id"]),
            last_seed_labels=last_seed_labels,
            current_labels=current_labels,
            collapsed_before=self.current_view["projected_before"],
            collapsed_after=self.current_view["projected_after"],
            all_gids_before=self.current_view["all_gids_before"],
            all_gids_after=self.current_view["all_gids_after"],
            current_title=current_title,
            current_subtitle=current_subtitle,
            last_seed_subtitle=last_seed_subtitle,
            new_point_mask=self.current_view["new_point_mask"],
            current_point_mask=current_point_mask,
            new_point_subtitle=f"is_seed={self.current_view['is_seed_frame']}",
            current_point_subtitle=f"is_seed={self.current_view['is_seed_frame']}",
            depth=self.current_view["depth"],
            filter_depth_mask=self.current_view["filter_depth_mask"],
            filter_unmatched_mask=self.current_view["filter_unmatched_mask"],
            filter_normal_mask=self.current_view["filter_normal_mask"],
            enabled_panels=enabled_panels,
        )

    def simulate_video(self, upto: int = -1) -> Path:
        upto = int(upto)
        if upto == -1:
            target_frame = self.total_frames - 1
        else:
            target_frame = upto
        if target_frame < 0 or target_frame >= self.total_frames:
            raise ValueError(f"upto must be -1 or in [0, {self.total_frames - 1}]")

        output_dir = self.cache.cache_dir / "debug_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        upto_tag = "full" if upto == -1 else f"{target_frame:06d}"
        output_path = output_dir / f"panel_{upto_tag}.mp4"
        if output_path.exists():
            output_path.unlink()

        try:
            from tqdm.auto import tqdm
        except Exception:  # noqa: BLE001
            tqdm = None

        self.reset()
        writer: cv2.VideoWriter | None = None
        iterator = range(target_frame + 1)
        progress = tqdm(iterator, desc="simulate_video", unit="frame") if tqdm is not None else iterator
        try:
            for _ in progress:
                self.step()
                panel = np.asarray(self.render_current_panel(), dtype=np.uint8)
                if writer is None:
                    height, width = panel.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        8.0,
                        (int(width), int(height)),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {output_path}")
                writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        finally:
            if writer is not None:
                writer.release()
            if tqdm is not None and hasattr(progress, "close"):
                progress.close()

        try:
            from IPython.display import Video, display

            display(Video(filename=str(output_path), embed=False, html_attributes="controls"))
        except Exception:  # noqa: BLE001
            pass
        return output_path

    def simulate_sam_onlyseed_video(self, upto: int = -1, map_every: int = 8) -> Path:
        upto = int(upto)
        map_every = int(map_every)
        if map_every <= 0:
            raise ValueError("map_every must be >= 1")
        if upto == -1:
            target_frame = self.total_frames - 1
        else:
            target_frame = upto
        if target_frame < 0 or target_frame >= self.total_frames:
            raise ValueError(f"upto must be -1 or in [0, {self.total_frames - 1}]")

        output_dir = self.cache.cache_dir / "debug_videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        upto_tag = "full" if upto == -1 else f"{target_frame:06d}"
        output_path = output_dir / f"sam_only_mapevery_{map_every}_{upto_tag}.mp4"
        if output_path.exists():
            output_path.unlink()

        try:
            from tqdm.auto import tqdm
        except Exception:  # noqa: BLE001
            tqdm = None

        mask_extractor = self._ensure_mask_extractor()
        local_tracker: SAM2VideoTracker | None = None
        tracker_frame_idx = 0
        writer: cv2.VideoWriter | None = None
        iterator = range(target_frame + 1)
        progress = tqdm(iterator, desc="simulate_sam_onlyseed_video", unit="frame") if tqdm is not None else iterator
        try:
            for frame_id in progress:
                frame = self.cache.load_frame(int(frame_id))
                detector_active = (int(frame_id) % map_every) == 0
                labels = np.full(frame["rgb"].shape[:2], -1, dtype=np.int32)
                mode_text = "detector active" if detector_active else "tracker active"
                if detector_active:
                    if not frame["valid_pose"]:
                        if local_tracker is not None:
                            local_tracker.close()
                            local_tracker = None
                        tracker_frame_idx = 0
                        mode_text = "detector active (invalid pose)"
                    else:
                        labels = mask_extractor.extract_labels(frame["rgb"]).astype(np.int32, copy=False)
                        seed_pairs = build_label_masks(labels, max_objects=int(self.config.tracker.max_num_objects))
                        if seed_pairs:
                            if local_tracker is None:
                                local_tracker = SAM2VideoTracker(
                                    frame["rgb"],
                                    config=self.config.tracker,
                                )
                                local_tracker.reset_and_seed_masks(seed_pairs)
                            else:
                                local_tracker.restart_and_seed_masks(frame["rgb"], seed_pairs)
                            tracker_frame_idx = 1
                        else:
                            if local_tracker is not None:
                                local_tracker.close()
                                local_tracker = None
                            tracker_frame_idx = 0
                else:
                    if local_tracker is None:
                        mode_text = "tracker active (no seed)"
                    else:
                        local_tracker.append_frame(tracker_frame_idx, frame["rgb"])
                        tracked_masks = local_tracker.track_frame(tracker_frame_idx)
                        tracker_frame_idx += 1
                        labels = self._build_tracker_labels(tracked_masks, frame["rgb"].shape[:2])

                panel = overlay_header(
                    render_sorted_label_map(labels),
                    "SAM Instances",
                    f"frame={int(frame_id)} | {mode_text}",
                )
                if writer is None:
                    height, width = panel.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        8.0,
                        (int(width), int(height)),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer for {output_path}")
                writer.write(cv2.cvtColor(np.asarray(panel, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        finally:
            if writer is not None:
                writer.release()
            if local_tracker is not None:
                local_tracker.close()
            if tqdm is not None and hasattr(progress, "close"):
                progress.close()

        try:
            from IPython.display import Video, display

            display(Video(filename=str(output_path), embed=False, html_attributes="controls"))
        except Exception:  # noqa: BLE001
            pass
        return output_path

    def _debug_current_map_points(self, frame_id: int) -> int:
        return int(np.searchsorted(self.cache.point_birth_frame, int(frame_id), side="right"))

    @staticmethod
    def _debug_filter_metric_gid_labels(
        metric_gid_labels: np.ndarray,
        min_component_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        filtered_labels = np.asarray(metric_gid_labels, dtype=np.int32).copy()
        valid = filtered_labels >= 0
        if valid.any() and int(min_component_size) > 1:
            uniq, counts = np.unique(filtered_labels[valid], return_counts=True)
            keep = uniq[counts >= int(min_component_size)]
            filtered_labels[~np.isin(filtered_labels, keep)] = -1
            valid = filtered_labels >= 0
        surviving_gids = np.unique(filtered_labels[valid]).astype(np.int32, copy=False) if valid.any() else np.empty((0,), dtype=np.int32)
        return filtered_labels, surviving_gids

    def _debug_resolve_optimal_selected_gids(
        self,
        *,
        scannet_raw_root: str | Path | None,
        replica_root: str | Path | None,
        min_component_size: int,
        progress_desc: str | None = None,
    ) -> dict[str, Any]:
        from get_metrics_map import compute_nn_associations, load_gt

        dataset_name = str(self.cache.manifest["dataset_name"])
        scene_name = str(self.cache.manifest["scene_name"])
        expected_scannet_root = None if scannet_raw_root is None else str(scannet_raw_root)
        expected_replica_root = None if replica_root is None else str(replica_root)
        metric_gid_labels_path = self._debug_optimal_metric_gid_labels_path(min_component_size=int(min_component_size))
        cached = self.debug_last_gt_selection_summary
        if (
            isinstance(cached, dict)
            and int(cached.get("frame_id", -1)) == int(self.current_frame_id)
            and int(cached.get("min_component_size", -1)) == int(min_component_size)
            and cached.get("scannet_raw_root") == expected_scannet_root
            and cached.get("replica_root") == expected_replica_root
            and "selected_gids" in cached
            and cached.get("filtered_metric_gid_labels_path") == str(metric_gid_labels_path)
            and metric_gid_labels_path.exists()
        ):
            return {
                **cached,
                "selected_gids": {int(gid) for gid in cached["selected_gids"]},
                "surviving_gids": [int(gid) for gid in cached.get("surviving_gids", [])],
            }
        eval_args = SimpleNamespace(
            scannet_raw_root=expected_scannet_root,
            replica_root=expected_replica_root,
        )
        stage_desc = "resolve_gt_selection" if progress_desc is None else str(progress_desc)
        progress = tqdm(total=5, desc=stage_desc, unit="stage", leave=False, dynamic_ncols=True)
        try:
            progress.set_postfix_str("load gt", refresh=True)
            gt = load_gt(dataset_name, scene_name, eval_args)
            progress.update()
            if gt["instance_labels"] is None:
                raise RuntimeError(f"{dataset_name}/{scene_name} does not expose GT instance labels needed for GT selection.")

            progress.set_postfix_str("load map", refresh=True)
            ply_path = self.cache.cache_dir / "rgb_map.ply"
            vertex = PlyData.read(str(ply_path))["vertex"].data
            pred_points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
            progress.update()
            if pred_points.shape[0] != self.point_gids.shape[0]:
                raise ValueError(
                    f"Point-count mismatch between debugger state and rgb_map.ply: {self.point_gids.shape[0]} vs {pred_points.shape[0]}"
                )

            progress.set_postfix_str("build/query kd-tree", refresh=True)
            assoc = compute_nn_associations(gt["points"], pred_points)
            pred_to_gt_idx = assoc["pred_to_gt_idx"]
            progress.update()

            progress.set_postfix_str("optimal collapse", refresh=True)
            metric_gid_labels, collapse_diag = self._resolve_metric_instance_gid_labels(
                collapse_mode="optimal",
                pred_to_gt_idx=pred_to_gt_idx,
                gt_instance_labels=gt["instance_labels"],
                progress_desc=f"{stage_desc}: optimal collapse",
            )
            filtered_metric_gid_labels, surviving_gids = self._debug_filter_metric_gid_labels(metric_gid_labels, int(min_component_size))
            progress.update()

            progress.set_postfix_str("transfer 5-nn", refresh=True)
            from get_metrics_map import transfer_instance_labels_ovo_style

            transferred_gids, transfer_diag = transfer_instance_labels_ovo_style(
                pred_points,
                filtered_metric_gid_labels,
                gt["points"],
            )
            gt_valid = np.asarray(gt["instance_labels"], dtype=np.int32) >= 0
            selected_gids = np.unique(transferred_gids[gt_valid]).astype(np.int32, copy=False)
            selected_gids = selected_gids[selected_gids >= 0]
            progress.update()
        finally:
            progress.close()
        metric_gid_labels_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(metric_gid_labels_path, filtered_metric_gid_labels.astype(np.int32, copy=False))
        summary = {
            "frame_id": int(self.current_frame_id),
            "min_component_size": int(min_component_size),
            "scannet_raw_root": expected_scannet_root,
            "replica_root": expected_replica_root,
            "selected_gids": sorted(int(gid) for gid in selected_gids.tolist()),
            "surviving_gids": [int(gid) for gid in surviving_gids.tolist()],
            "filtered_metric_gid_labels_path": str(metric_gid_labels_path),
            "collapse_diag": collapse_diag,
            "transfer_diag": transfer_diag,
        }
        self.debug_last_gt_selection_summary = summary
        self._debug_save_instance_cache()
        return {
            **summary,
            "selected_gids": {int(gid) for gid in summary["selected_gids"]},
        }

    def _debug_render_binary_signal_frame(
        self,
        *,
        gid_stats: dict[str, Any],
    ) -> np.ndarray:
        from matplotlib.figure import Figure

        gid = int(gid_stats["gid"])
        selected = bool(gid_stats["selected"])
        total_seed_frames = int(gid_stats["total_seed_frames"])
        birth_seed_ordinal = int(gid_stats["birth_seed_ordinal"])
        support_ordinals_arr = np.asarray(gid_stats["support_seed_ordinals"], dtype=np.int32)
        signal_arr = np.asarray(gid_stats["binary_signal"], dtype=np.float32)
        fig = Figure(figsize=(10, 4.5), dpi=160)
        ax = fig.add_subplot(111)
        x = np.arange(1, int(total_seed_frames) + 1, dtype=np.int32)
        y = np.full((int(total_seed_frames),), np.nan, dtype=np.float32)
        if birth_seed_ordinal <= total_seed_frames:
            y[birth_seed_ordinal - 1 :] = signal_arr
        color = "#1b7f3b" if selected else "#b22222"
        ax.plot(x, y, color=color, linewidth=2.5, drawstyle="steps-post")
        if support_ordinals_arr.size > 0:
            ax.scatter(support_ordinals_arr, np.ones_like(support_ordinals_arr), color=color, s=28, zorder=3)
        ax.axvline(birth_seed_ordinal, color="#444444", linestyle="--", linewidth=1.5)
        ax.set_xlim(1, max(1, int(total_seed_frames)))
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Seed frame ordinal")
        ax.set_ylabel("Support signal")
        ax.set_yticks([0, 1])
        bucket = self.buckets[int(gid)]
        final_support_ratio = float(gid_stats["support_perc"])
        ax.set_title(
            f"gid={int(gid)} | selected={bool(selected)} | birth_seed={birth_seed_ordinal} | "
            f"support_frames={int(bucket.support_frames)} | support%={final_support_ratio:.3f}"
        )
        ax.grid(True, axis="both", alpha=0.25)
        info_lines = [
            f"birth_frame={int(bucket.birth_frame)}",
            f"last_support_frame={int(bucket.last_support_frame)}",
            f"point_count_final={int(bucket.point_count)}",
            f"n00={int(gid_stats['n00'])} n01={int(gid_stats['n01'])}",
            f"n10={int(gid_stats['n10'])} n11={int(gid_stats['n11'])}",
            f"support_ordinals={support_ordinals_arr.tolist()}",
        ]
        ax.text(
            0.995,
            0.02,
            "\n".join(info_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#666666"},
        )
        fig.tight_layout()
        frame = _debug_render_matplotlib_figure(fig)
        fig.clear()
        return frame

    def get_gid_stats(self, gid: int | None = None) -> dict[int, dict[str, Any]] | dict[str, Any]:
        if not self.debug_gid_stats:
            raise RuntimeError("No gid stats available yet. Run plots_videos() first.")
        if gid is None:
            return {int(k): dict(v) for k, v in self.debug_gid_stats.items()}
        gid = int(gid)
        if gid not in self.debug_gid_stats:
            raise KeyError(f"gid={gid} is not present in debug_gid_stats.")
        return dict(self.debug_gid_stats[gid])

    def _debug_get_map_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        if self._debug_map_xyz is None or self._debug_map_rgb is None:
            ply_path = self.cache.cache_dir / "rgb_map.ply"
            vertex = PlyData.read(str(ply_path))["vertex"].data
            xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
            rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
            if xyz.shape[0] != self.point_gids.shape[0]:
                raise ValueError(
                    f"Point-count mismatch between debugger state and rgb_map.ply: {self.point_gids.shape[0]} vs {xyz.shape[0]}"
                )
            self._debug_map_xyz = xyz
            self._debug_map_rgb = rgb
        return self._debug_map_xyz, self._debug_map_rgb

    def _debug_load_gt_points(
        self,
        *,
        scannet_raw_root: str | Path | None = None,
        replica_root: str | Path | None = None,
    ) -> dict[str, Any]:
        dataset_name = str(self.cache.manifest["dataset_name"])
        scene_name = str(self.cache.manifest["scene_name"])
        scannet_key = None if scannet_raw_root is None else str(scannet_raw_root)
        replica_key = None if replica_root is None else str(replica_root)
        cache_key = (dataset_name, scene_name, scannet_key, replica_key)
        if not hasattr(self, "_debug_gt_cache"):
            self._debug_gt_cache: dict[tuple[str, str, str | None, str | None], dict[str, Any]] = {}
        cached = self._debug_gt_cache.get(cache_key)
        if cached is not None:
            return cached
        from get_metrics_map import load_gt

        gt = load_gt(
            dataset_name,
            scene_name,
            SimpleNamespace(
                scannet_raw_root=scannet_key,
                replica_root=replica_key,
            ),
        )
        self._debug_gt_cache[cache_key] = gt
        return gt

    @staticmethod
    def _debug_rgb_strings(rgb_values: np.ndarray) -> list[str]:
        return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb_values.tolist()]

    def threedshow(
        self,
        *,
        gid: int | None = None,
        gt_id: int | None = None,
        scannet_raw_root: str | Path | None = None,
        replica_root: str | Path | None = None,
        max_background_points: int | None = 250_000,
    ):
        if gid is None and gt_id is None:
            raise ValueError("Pass gid=... and/or gt_id=... to threedshow().")

        xyz, rgb = self._debug_get_map_geometry()
        total_points = int(xyz.shape[0])
        background_idx = np.arange(total_points, dtype=np.int64)
        sampled_background = False
        if max_background_points is not None and total_points > int(max_background_points):
            stride = int(np.ceil(float(total_points) / float(int(max_background_points))))
            background_idx = background_idx[::stride]
            sampled_background = True

        import plotly.graph_objects as go
        from IPython.display import display

        traces: list[go.Scatter3d] = [
            go.Scatter3d(
                x=xyz[background_idx, 0],
                y=xyz[background_idx, 1],
                z=xyz[background_idx, 2],
                mode="markers",
                name=(
                    f"rgb map ({int(background_idx.size):,}/{total_points:,} pts)"
                    if sampled_background
                    else f"rgb map ({total_points:,} pts)"
                ),
                marker={
                    "size": 1.2,
                    "color": self._debug_rgb_strings(rgb[background_idx]),
                    "opacity": 0.24,
                },
                hoverinfo="skip",
            )
        ]

        title_parts: list[str] = []
        if gid is not None:
            gid = int(gid)
            gid_mask = np.any(self.point_gids == gid, axis=1)
            gid_idx = np.flatnonzero(gid_mask)
            if gid_idx.size == 0:
                raise ValueError(f"gid={gid} is not present in the current map state.")
            gid_color = tuple(int(v) for v in color_for_id(gid).tolist())
            traces.append(
                go.Scatter3d(
                    x=xyz[gid_idx, 0],
                    y=xyz[gid_idx, 1],
                    z=xyz[gid_idx, 2],
                    mode="markers",
                    name=f"pred gid={gid} ({gid_idx.size:,} pts)",
                    marker={
                        "size": 2.8,
                        "color": f"rgb({gid_color[0]},{gid_color[1]},{gid_color[2]})",
                        "opacity": 0.95,
                    },
                    customdata=np.stack([gid_idx], axis=1),
                    hovertemplate="pred gid=%{fullData.name}<br>point_id=%{customdata[0]}<extra></extra>",
                )
            )
            title_parts.append(f"gid={gid}")

        if gt_id is not None:
            gt_id = int(gt_id)
            gt = self._debug_load_gt_points(
                scannet_raw_root=scannet_raw_root,
                replica_root=replica_root,
            )
            gt_instance_labels = gt.get("instance_labels")
            if gt_instance_labels is None:
                raise RuntimeError("GT instance labels are not available for this dataset.")
            gt_instance_labels = np.asarray(gt_instance_labels, dtype=np.int32)
            gt_mask = gt_instance_labels == gt_id
            gt_idx = np.flatnonzero(gt_mask)
            if gt_idx.size == 0:
                raise ValueError(f"gt_id={gt_id} is not present in the GT instance labels.")
            gt_points = np.asarray(gt["points"], dtype=np.float32)
            traces.append(
                go.Scatter3d(
                    x=gt_points[gt_idx, 0],
                    y=gt_points[gt_idx, 1],
                    z=gt_points[gt_idx, 2],
                    mode="markers",
                    name=f"gt_id={gt_id} ({gt_idx.size:,} pts)",
                    marker={
                        "size": 2.8,
                        "color": "rgb(255,0,255)",
                        "opacity": 0.95,
                    },
                    customdata=np.stack([gt_idx], axis=1),
                    hovertemplate="gt_id=%{fullData.name}<br>gt_point_id=%{customdata[0]}<extra></extra>",
                )
            )
            title_parts.append(f"gt_id={gt_id}")

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=" | ".join(title_parts),
            showlegend=True,
            legend={"title": {"text": "Highlights"}},
            margin={"l": 0, "r": 0, "b": 0, "t": 40},
            scene={
                "aspectmode": "data",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            },
        )
        display(fig)
        return None

    def fit(
        self,
        *,
        scannet_raw_root: str | Path | None = None,
        replica_root: str | Path | None = None,
        min_component_size: int = 2000,
        embed_dim: int = 8,
        num_layers: int = 2,
        epochs: int = 2000,
        pruning_thresh: float = 0.6,
        loss_mode: str = "wbce",
        focal_gamma: float = 2.0,
        plot_every: int = 25,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        if self.current_frame_id < 0:
            raise RuntimeError("Run step() or seek() before calling fit().")
        selection_info = self._debug_resolve_optimal_selected_gids(
            scannet_raw_root=scannet_raw_root,
            replica_root=replica_root,
            min_component_size=int(min_component_size),
            progress_desc="fit resolve_gt_selection",
        )
        diagnostics = self._fit_learned_gid_selector(
            selected_gids=selection_info["selected_gids"],
            embed_dim=int(embed_dim),
            num_layers=int(num_layers),
            epochs=int(epochs),
            pruning_thresh=float(pruning_thresh),
            loss_mode=str(loss_mode),
            focal_gamma=float(focal_gamma),
            plot_every=int(plot_every),
            show_training_plot=bool(show_progress),
        )
        self.debug_gid_stats = self._debug_build_gid_stats(selected_gids=selection_info["selected_gids"])
        self.debug_last_learned_collapse_summary = {
            **diagnostics,
            "target_selected_gids": sorted(int(gid) for gid in selection_info["selected_gids"]),
            "min_component_size": int(min_component_size),
        }
        self.debug_last_learned_fit_frame_id = int(self.current_frame_id)
        compact_keys = [
            "learned_num_gids",
            "learned_num_selected",
            "learned_num_rejected",
            "learned_embed_dim",
            "learned_num_layers",
            "learned_epochs",
            "learned_pruning_thresh",
            "learned_loss_mode",
            "learned_focal_alpha",
            "learned_focal_gamma",
            "learned_plot_every",
            "learned_pos_weight",
            "learned_train_loss_final",
            "learned_train_acc_final",
            "learned_train_selected_acc_final",
            "learned_train_non_selected_acc_final",
            "learned_train_selected_f1_final",
            "learned_train_device",
            "min_component_size",
        ]
        return {
            key: self.debug_last_learned_collapse_summary[key]
            for key in compact_keys
            if key in self.debug_last_learned_collapse_summary
        }

    @staticmethod
    def _debug_render_histogram_frame(
        *,
        values: np.ndarray,
        bins: np.ndarray,
        x_max: float,
        y_max: int,
        title: str,
        threshold_value: float,
        selected_safe_threshold_value: float | None,
        annotation_lines: list[str],
        color: str,
    ) -> np.ndarray:
        from matplotlib.figure import Figure

        fig = Figure(figsize=(10, 4.5), dpi=160)
        ax = fig.add_subplot(111)
        values = np.asarray(values, dtype=np.float32)
        counts = np.zeros((max(0, bins.shape[0] - 1),), dtype=np.float32)
        patches = []
        if values.size > 0:
            counts, _, patches = ax.hist(values, bins=bins, color=color, edgecolor="black", linewidth=0.6)
            for count, patch in zip(counts.tolist(), patches):
                if count <= 0:
                    continue
                x_center = float(patch.get_x() + patch.get_width() * 0.5)
                ax.text(
                    x_center,
                    float(count) + max(0.15, 0.01 * float(y_max)),
                    str(int(round(count))),
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                    color="black",
                )
        ax.axvline(float(threshold_value), color="#b22222", linestyle="--", linewidth=2.0)
        if selected_safe_threshold_value is not None:
            ax.axvline(float(selected_safe_threshold_value), color="#222222", linestyle=":", linewidth=2.0)
        ax.set_xlim(0.0, float(x_max))
        ax.set_ylim(0, max(1, int(y_max)))
        ax.set_xlabel("Percentage")
        ax.set_ylabel("num_gids")
        ax.set_title(title)
        ax.set_xticks(np.linspace(0.0, float(x_max), 16))
        y_tick_count = min(16, max(2, int(y_max) + 1))
        y_ticks = np.unique(np.rint(np.linspace(0.0, float(y_max), y_tick_count)).astype(np.int32))
        ax.set_yticks(y_ticks.tolist())
        ax.grid(True, axis="y", alpha=0.25)
        ax.text(
            0.995,
            0.98,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#666666"},
        )
        fig.tight_layout()
        frame = _debug_render_matplotlib_figure(fig)
        fig.clear()
        return frame

    def plots_videos(
        self,
        *,
        scannet_raw_root: str | Path | None = None,
        replica_root: str | Path | None = None,
        min_component_size: int = 2000,
        output_dir: str | Path | None = None,
        fps: float = 4.0,
        hist_bins: int = 150,
    ) -> dict[str, Any]:
        output_dir = self.cache.cache_dir / "debug_videos" if output_dir is None else Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.run_all(upto=-1, progress_desc="plots_videos run_all")
        timeline_snapshots = list(self.debug_timeline_snapshots)
        cached_summary = self.debug_last_plots_videos_summary
        expected_scannet_root = None if scannet_raw_root is None else str(scannet_raw_root)
        expected_replica_root = None if replica_root is None else str(replica_root)
        if isinstance(cached_summary, dict):
            cached_binary = cached_summary.get("binary_signal")
            cached_points = cached_summary.get("timeline_points_perc")
            cached_support = cached_summary.get("timeline_support_perc")
            if (
                int(cached_summary.get("frame_id", -1)) == int(self.current_frame_id)
                and int(cached_summary.get("min_component_size", -1)) == int(min_component_size)
                and cached_summary.get("scannet_raw_root") == expected_scannet_root
                and cached_summary.get("replica_root") == expected_replica_root
                and isinstance(cached_binary, str)
                and isinstance(cached_points, str)
                and isinstance(cached_support, str)
                and Path(cached_binary).exists()
                and Path(cached_points).exists()
                and Path(cached_support).exists()
            ):
                return dict(cached_summary)

        selection_info = self._debug_resolve_optimal_selected_gids(
            scannet_raw_root=scannet_raw_root,
            replica_root=replica_root,
            min_component_size=int(min_component_size),
            progress_desc="plots_videos resolve_gt_selection",
        )

        selected_gids = selection_info["selected_gids"]
        surviving_final_gids = [int(gid) for gid in sorted(self.buckets)]
        non_selected_final_gids = [int(gid) for gid in surviving_final_gids if int(gid) not in selected_gids]
        selected_final_gids = [int(gid) for gid in surviving_final_gids if int(gid) in selected_gids]
        ordered_binary_gids = non_selected_final_gids + selected_final_gids
        total_seed_frames = int(self._num_seed_frames_processed())
        self.debug_gid_stats = self._debug_build_gid_stats(selected_gids=selected_gids)

        binary_frames: list[np.ndarray] = []
        if ordered_binary_gids:
            binary_progress = tqdm(ordered_binary_gids, desc="plots_videos binary_signal", unit="gid", leave=False, dynamic_ncols=True)
            try:
                for gid in binary_progress:
                    binary_progress.set_postfix_str(f"gid={int(gid)}", refresh=True)
                    binary_frames.append(
                        self._debug_render_binary_signal_frame(
                            gid_stats=self.debug_gid_stats[int(gid)],
                        )
                    )
            finally:
                binary_progress.close()
        else:
            binary_frames.append(
                self._debug_render_histogram_frame(
                    values=np.empty((0,), dtype=np.float32),
                    bins=np.linspace(0.0, 100.0, 11, dtype=np.float32),
                    x_max=100.0,
                    y_max=1,
                    title="Binary support lifecycle",
                    threshold_value=0.0,
                    annotation_lines=[
                        "No surviving gids at final frame.",
                        f"frame_id={int(self.current_frame_id)}",
                        f"selected_gid_count={len(selected_final_gids)}",
                    ],
                    color="#4c78a8",
                )
            )
        binary_path = _debug_write_video_frames(binary_frames, output_dir / "binary_signal.mp4", fps=float(fps))

        bins = np.linspace(0.0, 100.0, int(hist_bins) + 1, dtype=np.float32)
        max_point_hist = 0
        max_support_hist = 0
        for snapshot in timeline_snapshots:
            point_counts, _ = np.histogram(snapshot["point_perc"], bins=bins)
            support_counts, _ = np.histogram(snapshot["support_perc"], bins=bins)
            max_point_hist = max(max_point_hist, int(point_counts.max()) if point_counts.size > 0 else 0)
            max_support_hist = max(max_support_hist, int(support_counts.max()) if support_counts.size > 0 else 0)
        max_point_hist = max(1, max_point_hist)
        max_support_hist = max(1, max_support_hist)
        point_threshold = 100.0 * float(self.config.pipeline.prune_min_points_perc)
        support_threshold = 100.0 * float(self.config.pipeline.prune_min_support_perc)

        point_frames: list[np.ndarray] = []
        point_progress = tqdm(timeline_snapshots, desc="plots_videos timeline_points_perc", unit="frame", leave=False, dynamic_ncols=True)
        try:
            for snapshot in point_progress:
                frame_id = int(snapshot["frame_id"])
                values = np.asarray(snapshot["point_perc"], dtype=np.float32)
                bucket_ids = [int(gid) for gid in snapshot["bucket_ids"]]
                selected_alive_values = np.asarray(
                    [float(values[idx]) for idx, gid in enumerate(bucket_ids) if int(gid) in selected_gids],
                    dtype=np.float32,
                )
                selected_safe_threshold = (
                    float(selected_alive_values.min()) if selected_alive_values.size > 0 else None
                )
                annotation_lines = [
                    f"frame_id={frame_id}",
                    f"num_gids={int(snapshot['num_gids'])}",
                    f"map_points={int(snapshot['current_map_points'])}",
                    f"seed_frames={int(snapshot['num_seed_frames'])}",
                    f"mean={float(values.mean()) if values.size > 0 else 0.0:.4f}%",
                    f"median={float(np.median(values)) if values.size > 0 else 0.0:.4f}%",
                    f"max={float(values.max()) if values.size > 0 else 0.0:.4f}%",
                    f"prune_min_points_perc={point_threshold:.4f}%",
                    "selected_safe_points_perc="
                    + ("n/a" if selected_safe_threshold is None else f"{selected_safe_threshold:.4f}%"),
                ]
                point_frames.append(
                    self._debug_render_histogram_frame(
                        values=values,
                        bins=bins,
                        x_max=100.0,
                        y_max=max_point_hist,
                        title="Timeline points percentage histogram",
                        threshold_value=point_threshold,
                        selected_safe_threshold_value=selected_safe_threshold,
                        annotation_lines=annotation_lines,
                        color="#4c78a8",
                    )
                )
        finally:
            point_progress.close()
        points_path = _debug_write_video_frames(point_frames, output_dir / "timeline_points_perc.mp4", fps=float(fps))

        support_frames: list[np.ndarray] = []
        support_progress = tqdm(timeline_snapshots, desc="plots_videos timeline_support_perc", unit="frame", leave=False, dynamic_ncols=True)
        try:
            for snapshot in support_progress:
                frame_id = int(snapshot["frame_id"])
                values = np.asarray(snapshot["support_perc"], dtype=np.float32)
                bucket_ids = [int(gid) for gid in snapshot["bucket_ids"]]
                selected_alive_values = np.asarray(
                    [float(values[idx]) for idx, gid in enumerate(bucket_ids) if int(gid) in selected_gids],
                    dtype=np.float32,
                )
                selected_safe_threshold = (
                    float(selected_alive_values.min()) if selected_alive_values.size > 0 else None
                )
                annotation_lines = [
                    f"frame_id={frame_id}",
                    f"num_gids={int(snapshot['num_gids'])}",
                    f"map_points={int(snapshot['current_map_points'])}",
                    f"seed_frames={int(snapshot['num_seed_frames'])}",
                    f"mean={float(values.mean()) if values.size > 0 else 0.0:.4f}%",
                    f"median={float(np.median(values)) if values.size > 0 else 0.0:.4f}%",
                    f"max={float(values.max()) if values.size > 0 else 0.0:.4f}%",
                    f"prune_min_support_perc={support_threshold:.4f}%",
                    "selected_safe_support_perc="
                    + ("n/a" if selected_safe_threshold is None else f"{selected_safe_threshold:.4f}%"),
                ]
                support_frames.append(
                    self._debug_render_histogram_frame(
                        values=values,
                        bins=bins,
                        x_max=100.0,
                        y_max=max_support_hist,
                        title="Timeline support percentage histogram",
                        threshold_value=support_threshold,
                        selected_safe_threshold_value=selected_safe_threshold,
                        annotation_lines=annotation_lines,
                        color="#72b7b2",
                    )
                )
        finally:
            support_progress.close()
        support_path = _debug_write_video_frames(support_frames, output_dir / "timeline_support_perc.mp4", fps=float(fps))

        summary = {
            "binary_signal": str(binary_path),
            "timeline_points_perc": str(points_path),
            "timeline_support_perc": str(support_path),
            "frame_id": int(self.current_frame_id),
            "min_component_size": int(min_component_size),
            "scannet_raw_root": expected_scannet_root,
            "replica_root": expected_replica_root,
            "selected_gids": sorted(int(gid) for gid in selected_gids),
            "selected_gid_count": int(len(selected_final_gids)),
            "non_selected_gid_count": int(len(non_selected_final_gids)),
        }
        self.debug_last_plots_videos_summary = summary
        self._debug_save_instance_cache()
        return summary

    def get_metrics(
        self,
        *,
        scannet_raw_root: str | Path | None = None,
        replica_root: str | Path | None = None,
        min_component_size: int = 1,
        ovo_score_th: float = 0.0,
        chunk_size: int = 100_000,
        use_collapse_mode: str = "support",
        use_optimal_text_matching: bool = False,
        save_json: bool = False,
        video_output_dir: str | Path | None = None,
        write_label_video: bool = True,
    ) -> dict[str, Any]:
        if self.current_frame_id < 0:
            raise RuntimeError("Run step() or seek() before calling get_metrics().")

        from get_metrics_map import (
            classify_instance_features_ovo_style,
            compute_instance_metrics,
            compute_nn_associations,
            encode_class_texts,
            finalize_instance_labels_and_scores,
            load_dataset_info,
            load_gt,
            map_gt_labels_to_eval_ids,
            ovo_text_template,
            transfer_instance_labels_ovo_style,
        )

        dataset_name = str(self.cache.manifest["dataset_name"])
        scene_name = str(self.cache.manifest["scene_name"])
        eval_args = SimpleNamespace(
            scannet_raw_root=None if scannet_raw_root is None else str(scannet_raw_root),
            replica_root=None if replica_root is None else str(replica_root),
        )
        total_stages = 7
        progress = tqdm(total=total_stages, desc=f"{scene_name} debugger metrics", unit="stage", dynamic_ncols=True)
        try:
            progress.set_postfix_str("load gt", refresh=True)
            gt = load_gt(dataset_name, scene_name, eval_args)
            dataset_info = load_dataset_info(dataset_name)
            progress.update()

            progress.set_postfix_str("load map", refresh=True)
            ply_path = self.cache.cache_dir / "rgb_map.ply"
            if not ply_path.exists():
                raise FileNotFoundError(ply_path)
            vertex = PlyData.read(str(ply_path))["vertex"].data
            pred_points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
            if pred_points.shape[0] != self.point_gids.shape[0]:
                raise ValueError(
                    f"Point-count mismatch between debugger state and rgb_map.ply: {self.point_gids.shape[0]} vs {pred_points.shape[0]}"
                )
            clip_path = self.cache.cache_dir / CLIP_FEATURE_FILE
            if not clip_path.exists():
                raise FileNotFoundError(clip_path)
            clip_features = np.load(clip_path, mmap_mode="r")
            if int(clip_features.shape[0]) != pred_points.shape[0]:
                raise ValueError(
                    f"Point-count mismatch between clip_feats and rgb_map.ply: {clip_features.shape[0]} vs {pred_points.shape[0]}"
                )
            progress.update()

            progress.set_postfix_str("resolve instances", refresh=True)
            assoc = compute_nn_associations(gt["points"], pred_points)
            pred_to_gt_idx = assoc["pred_to_gt_idx"]
            collapse_mode = str(use_collapse_mode)
            if collapse_mode == "optimal":
                progress.set_postfix_str("resolve instances (optimal)", refresh=True)
            elif collapse_mode == "learned":
                progress.set_postfix_str("resolve instances (learned)", refresh=True)
                if self.debug_last_learned_collapse_summary is None:
                    raise RuntimeError("Learned collapse requires a prior debugger.fit(...).")
                if self.debug_last_learned_fit_frame_id != int(self.current_frame_id):
                    raise RuntimeError("Learned collapse fit is stale for the current debugger state. Re-run debugger.fit(...).")
                learned_selected_gids = {
                    int(gid)
                    for gid in self.debug_last_learned_collapse_summary.get("learned_pred_selected_gids", [])
                }
            else:
                learned_selected_gids = None
            metric_gid_labels, collapse_diag = self._resolve_metric_instance_gid_labels(
                collapse_mode=collapse_mode,
                pred_to_gt_idx=pred_to_gt_idx if collapse_mode == "optimal" else None,
                gt_instance_labels=gt["instance_labels"] if collapse_mode == "optimal" else None,
                learned_selected_gids=learned_selected_gids if collapse_mode == "learned" else None,
            )
            if collapse_mode == "learned" and self.debug_last_learned_collapse_summary is not None:
                collapse_diag = {
                    **self.debug_last_learned_collapse_summary,
                    **collapse_diag,
                }
            metric_point_instance_labels, metric_instance_scores = finalize_instance_labels_and_scores(
                metric_gid_labels,
                self._raw_instance_support_scores(),
                int(min_component_size),
            )
            gt_point_instance_labels = np.full((pred_points.shape[0],), -1, dtype=np.int32)
            if gt["instance_labels"] is not None:
                gt_point_instance_labels = np.asarray(gt["instance_labels"], dtype=np.int32)[pred_to_gt_idx]
            gt_semantic = map_gt_labels_to_eval_ids(gt["semantic_raw"], dataset_info)
            progress.update()

            progress.set_postfix_str("encode text", refresh=True)
            class_names = dataset_info.get("class_names_reduced", dataset_info.get("class_names"))
            device = "cuda" if self.device != "cpu" and torch.cuda.is_available() else "cpu"
            ovo_text_embeds = encode_class_texts(
                class_names,
                device,
                template=ovo_text_template(bool(use_optimal_text_matching)),
            )
            progress.update()

            progress.set_postfix_str("classify instances", refresh=True)
            instance_classes, _semantic_instance_scores, semantic_ovo_diag = classify_instance_features_ovo_style(
                clip_features,
                metric_point_instance_labels,
                ovo_text_embeds,
                float(ovo_score_th),
                int(chunk_size),
                progress_desc="ovo instance pooling",
                use_optimal_text_matching=bool(use_optimal_text_matching),
            )
            progress.update()

            progress.set_postfix_str("transfer + ap", refresh=True)
            transferred_instance_labels, instance_transfer_diag = transfer_instance_labels_ovo_style(
                pred_points,
                metric_point_instance_labels,
                gt["points"],
            )
            if gt["instance_labels"] is None:
                instance_metrics, instance_diag = None, None
            else:
                instance_metrics, instance_diag = compute_instance_metrics(
                    gt["instance_labels"],
                    gt_semantic,
                    transferred_instance_labels,
                    instance_classes,
                    metric_instance_scores,
                    dataset_info,
                )
                instance_diag = {
                    **instance_diag,
                    **instance_transfer_diag,
                }
            progress.update()

            if write_label_video:
                progress.set_postfix_str("write label video", refresh=True)
                output_dir = self.cache.cache_dir / "debug_videos" if video_output_dir is None else Path(video_output_dir)
                mode_tag = collapse_mode
                output_name = (
                    f"{mode_tag}_collapse_frame_{int(self.current_frame_id):06d}"
                    f"_mincomp_{int(min_component_size)}.mp4"
                )
                instance_video_path = write_instance_label_video_from_cache(
                    self.cache.cache_dir,
                    metric_point_instance_labels,
                    gt_point_labels=gt_point_instance_labels,
                    output_path=output_dir / output_name,
                    title="Instance Labels",
                    subtitle_prefix="projected",
                    upto_frame=int(self.current_frame_id),
                )
            else:
                progress.set_postfix_str("skip label video", refresh=True)
                instance_video_path = None
            progress.update()
        finally:
            progress.close()

        summary = {
            "frame_id": int(self.current_frame_id),
            "dataset_name": dataset_name,
            "scene_name": scene_name,
            "metrics": {
                "instance": instance_metrics,
            },
            "diagnostics": {
                "instance": instance_diag,
                "semantic_ovo_style": semantic_ovo_diag,
                "pred_instance_count": int(
                    np.sum(np.unique(metric_point_instance_labels[metric_point_instance_labels >= 0]) >= 0)
                ),
                "min_component_size": int(min_component_size),
                "ovo_score_th": float(ovo_score_th),
                "use_optimal_text_matching": bool(use_optimal_text_matching),
                "ovo_text_template": ovo_text_template(bool(use_optimal_text_matching)),
                "ovo_feature_agg": semantic_ovo_diag.get("agg_mode"),
                "chunk_size": int(chunk_size),
                "instance_video_label_source": "metric_point_instance_labels",
                "instance_score_source": "seed_support_ratio",
                "collapse_mode": collapse_mode,
                **collapse_diag,
            },
        }
        summary["instance_video_path"] = str(instance_video_path)
        if save_json:
            out_path = self.cache.cache_dir / f"debugger_metrics_frame_{int(self.current_frame_id):06d}.json"
            with open(out_path, "w") as handle:
                json.dump(summary, handle, indent=2)
            summary["saved_path"] = str(out_path)
        return summary

    @staticmethod
    def _annotate_mask_ids(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
        canvas = np.asarray(image, dtype=np.uint8).copy()
        labels = np.asarray(labels, dtype=np.int32)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 2
        height, width = canvas.shape[:2]
        for local_id in np.unique(labels).tolist():
            if local_id < 0:
                continue
            ys, xs = np.nonzero(labels == int(local_id))
            if ys.size == 0:
                continue
            cy = int(np.round(ys.mean()))
            cx = int(np.round(xs.mean()))
            text = str(int(local_id))
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
            pad_x = 6
            pad_y = 5
            bubble_w = text_w + 2 * pad_x
            bubble_h = text_h + baseline + 2 * pad_y
            x0 = int(np.clip(cx - (bubble_w // 2), 0, max(0, width - bubble_w)))
            y0 = int(np.clip(cy - (bubble_h // 2), 0, max(0, height - bubble_h)))
            x1 = x0 + bubble_w
            y1 = y0 + bubble_h
            text_x = x0 + pad_x
            text_y = y0 + pad_y + text_h
            segment_color = tuple(int(x) for x in color_for_id(int(local_id)).tolist())
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), thickness=-1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), segment_color, thickness=1)
            cv2.putText(canvas, text, (text_x, text_y), font, font_scale, segment_color, text_thickness, cv2.LINE_AA)
        return canvas

    def show_sam_masks(self) -> None:
        if self.current_view is None:
            raise RuntimeError("Run step() or seek() before calling show_sam_masks().")
        if self.current_view["is_seed_frame"]:
            labels = self.current_view["seed_labels"]
            frame_id = int(self.current_view["frame_id"])
            title = f"seed SAM masks frame={frame_id}"
        else:
            labels = self.current_view["last_seed_labels"]
            frame_id = int(self.current_view["last_seed_frame_id"])
            title = f"last seed SAM masks frame={frame_id}"
        if labels is None:
            raise RuntimeError("No seed SAM masks are available in the current view.")

        from map_runtime.debug_panels import render_sorted_label_map

        image = render_sorted_label_map(labels)
        image = self._annotate_mask_ids(image, labels)

        import matplotlib.pyplot as plt
        from IPython.display import display

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    def show(
        self,
        *,
        local_id: int | None = None,
        gid: int | None = None,
    ) -> None:
        if self.current_view is None:
            raise RuntimeError("Run step() or seek() before calling show().")
        if local_id is None and gid is None:
            raise ValueError("Pass local_id=..., gid=..., or both.")

        local_mask = None
        gid_mask = None
        if local_id is not None:
            if not self.current_view["is_seed_frame"]:
                raise ValueError("local_id is only available on seed frames.")
            if self.current_view["seed_labels"] is None:
                raise RuntimeError("Current view has no seed_labels.")
            local_label = int(local_id)
            local_mask = np.asarray(self.current_view["seed_labels"] == local_label, dtype=bool)
            if not local_mask.any():
                raise ValueError(f"local_id={local_label} is not present in the current seed masks.")
        if gid is not None:
            gid_label = int(gid)
            gid_mask = self._project_gid_membership(
                np.asarray(self.current_view["point_ids_after"], dtype=np.int64),
                gid_label,
            )
            if not gid_mask.any():
                raise ValueError(f"gid={gid_label} is not present in the current visible raw gid memberships.")

        mask_shape = None
        if local_mask is not None:
            mask_shape = local_mask.shape
        if gid_mask is not None:
            mask_shape = gid_mask.shape if mask_shape is None else mask_shape
        image = np.zeros((*mask_shape, 3), dtype=np.uint8)

        if local_mask is not None and gid_mask is not None:
            local_only = local_mask & ~gid_mask
            gid_only = gid_mask & ~local_mask
            overlap = local_mask & gid_mask
            image[local_only] = np.array((0, 255, 0), dtype=np.uint8)
            image[gid_only] = np.array((255, 165, 0), dtype=np.uint8)
            image[overlap] = np.array((255, 0, 255), dtype=np.uint8)
            legend_items = [
                ("local_id", (0, 255, 0)),
                ("gid", (255, 165, 0)),
                ("overlap", (255, 0, 255)),
            ]
        elif local_mask is not None:
            color = color_for_id(int(local_id))
            image[local_mask] = color
            legend_items = [(f"local_id={int(local_id)}", tuple(int(x) for x in color.tolist()))]
        else:
            color = color_for_id(int(gid))
            image[gid_mask] = color
            legend_items = [(f"gid={int(gid)}", tuple(int(x) for x in color.tolist()))]

        box_w = 210
        box_h = 32 + 26 * len(legend_items)
        x0 = max(8, image.shape[1] - box_w - 8)
        y0 = 8
        cv2.rectangle(image, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), thickness=-1)
        cv2.rectangle(image, (x0, y0), (x0 + box_w, y0 + box_h), (180, 180, 180), thickness=1)
        for idx, (name, color) in enumerate(legend_items):
            yy = y0 + 20 + idx * 26
            cv2.rectangle(image, (x0 + 12, yy - 10), (x0 + 30, yy + 8), color, thickness=-1)
            cv2.putText(image, name, (x0 + 40, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        try:
            import matplotlib.pyplot as plt
            from IPython.display import display

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(image)
            ax.axis("off")
            if local_id is not None and gid is not None:
                title = f"local_id={int(local_id)} vs gid={int(gid)}"
            elif local_id is not None:
                title = f"local_id={int(local_id)}"
            else:
                title = f"gid={int(gid)}"
            ax.set_title(title)
            fig.tight_layout()
            display(fig)
            plt.close(fig)
        except Exception:
            pass
        return None

    def current_text_summary(self) -> str:
        if self.current_view is None:
            return "No frame processed yet."
        lines = [
            f"frame_id={self.current_view['frame_id']} is_seed_frame={self.current_view['is_seed_frame']} valid_pose={self.current_view['valid_pose']}",
            f"seeded_gids={self.current_view['seeded_gids']}",
            f"pruned_gids={self.current_view['pruned_gids']}",
            f"num_buckets={len(self.current_view['bucket_snapshot'])}",
            "",
            "decisions:",
        ]
        if not self.current_view["decisions"]:
            lines.append("  <none>")
        else:
            decisions = sorted(
                self.current_view["decisions"],
                key=lambda d: (
                    d["local_mask_id"] is None,
                    -1 if d["local_mask_id"] is None else int(d["local_mask_id"]),
                ),
            )
            for decision in decisions:
                lines.append(
                    "  "
                    + ", ".join(
                        [
                            f"local_mask_id={decision['local_mask_id']}",
                            f"action={decision['action']}",
                            f"gid={decision['gid']}",
                            f"visible_points={decision['visible_points']}",
                            f"added_points={decision['added_points']}",
                            f"overflow_points={decision['overflow_points']}",
                            f"candidate_gid={decision['candidate_gid']}",
                            f"inside_frac={decision['inside_frac']}",
                            f"outside_frac={decision['outside_frac']}",
                        ]
                    )
                )
        lines.extend(["", "bucket_snapshot:"])
        bucket_snapshot = sorted(self.current_view["bucket_snapshot"], key=lambda b: int(b["gid"]))
        for bucket in bucket_snapshot[:20]:
            lines.append(
                "  "
                + ", ".join(
                    [
                        f"gid={bucket['gid']}",
                        f"support_frames={bucket['support_frames']}",
                        f"point_count={bucket['point_count']}",
                        f"last_support_frame={bucket['last_support_frame']}",
                        f"birth_frame={bucket['birth_frame']}",
                    ]
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _html_token(value: Any, *, bold: bool = False) -> str:
        text = html.escape(str(value))
        return f"<b>{text}</b>" if bold else text

    def _bucket_line_html(self, bucket: dict[str, Any], other: dict[str, Any] | None) -> str:
        fields = ["gid", "support_frames", "point_count", "last_support_frame", "birth_frame"]
        parts = []
        for field in fields:
            bold = other is None or bucket.get(field) != other.get(field)
            parts.append(f"{field}={self._html_token(bucket.get(field), bold=bold)}")
        return "  " + ", ".join(parts)

    def _point_gid_line_html(self, point_id: int, row: list[int], other: list[int]) -> str:
        slots = [
            self._html_token(value, bold=(int(value) != int(other[idx])))
            for idx, value in enumerate(row)
        ]
        return f"  point_id={html.escape(str(int(point_id)))}, gids=[{', '.join(slots)}]"

    def current_html_summary(self) -> str:
        if self.current_view is None:
            return "<pre>No frame processed yet.</pre>"
        summary_lines: list[str] = [
            html.escape(
                f"frame_id={self.current_view['frame_id']} is_seed_frame={self.current_view['is_seed_frame']} valid_pose={self.current_view['valid_pose']}"
            ),
            html.escape(f"seeded_gids={self.current_view['seeded_gids']}"),
            html.escape(f"pruned_gids={self.current_view['pruned_gids']}"),
            html.escape(f"num_buckets={len(self.current_view['bucket_snapshot'])}"),
            "",
            "decisions:",
        ]
        if not self.current_view["decisions"]:
            summary_lines.append("  &lt;none&gt;")
        else:
            decisions = sorted(
                self.current_view["decisions"],
                key=lambda d: (
                    d["local_mask_id"] is None,
                    -1 if d["local_mask_id"] is None else int(d["local_mask_id"]),
                ),
            )
            for decision in decisions:
                summary_lines.append(
                    "  "
                    + html.escape(
                        ", ".join(
                            [
                                f"local_mask_id={decision['local_mask_id']}",
                                f"action={decision['action']}",
                                f"gid={decision['gid']}",
                                f"visible_points={decision['visible_points']}",
                                f"added_points={decision['added_points']}",
                                f"overflow_points={decision['overflow_points']}",
                                f"candidate_gid={decision['candidate_gid']}",
                                f"inside_frac={decision['inside_frac']}",
                                f"outside_frac={decision['outside_frac']}",
                            ]
                        )
                    )
                )
        summary_lines.extend(["", "bucket_snapshot:"])
        bucket_snapshot = sorted(self.current_view["bucket_snapshot"], key=lambda b: int(b["gid"]))
        for bucket in bucket_snapshot[:20]:
            summary_lines.append(
                "  "
                + html.escape(
                    ", ".join(
                        [
                            f"gid={bucket['gid']}",
                            f"support_frames={bucket['support_frames']}",
                            f"point_count={bucket['point_count']}",
                            f"last_support_frame={bucket['last_support_frame']}",
                            f"birth_frame={bucket['birth_frame']}",
                        ]
                    )
                )
            )

        buckets_before = {int(bucket["gid"]): bucket for bucket in self.current_view["bucket_snapshot_before"]}
        buckets_after = {int(bucket["gid"]): bucket for bucket in self.current_view["bucket_snapshot_after"]}
        transition_before_lines = ["state_transition:", "  buckets_before:"]
        if not buckets_before:
            transition_before_lines.append("    &lt;none&gt;")
        else:
            for gid in sorted(buckets_before):
                transition_before_lines.append(self._bucket_line_html(buckets_before[gid], buckets_after.get(gid)))
        transition_after_lines = ["  buckets_after:"]
        if not buckets_after:
            transition_after_lines.append("    &lt;none&gt;")
        else:
            for gid in sorted(buckets_after):
                transition_after_lines.append(self._bucket_line_html(buckets_after[gid], buckets_before.get(gid)))

        point_gid_changes_all = sorted(
            self.current_view["point_gid_changes"],
            key=lambda item: (
                -max(
                    sum(int(gid) >= 0 for gid in item["before"]),
                    sum(int(gid) >= 0 for gid in item["after"]),
                ),
                int(item["point_id"]),
            ),
        )
        point_gid_changes = point_gid_changes_all[:100]
        point_before_lines: list[str] = ["before:"]
        point_after_lines: list[str] = ["after:"]
        if not point_gid_changes:
            point_before_lines.append("  &lt;none&gt;")
            point_after_lines.append("  &lt;none&gt;")
        else:
            for item in point_gid_changes:
                point_before_lines.append(self._point_gid_line_html(int(item["point_id"]), list(item["before"]), list(item["after"])))
            for item in point_gid_changes:
                point_after_lines.append(self._point_gid_line_html(int(item["point_id"]), list(item["after"]), list(item["before"])))
        block_style = "font-family: monospace; white-space: pre;"
        return (
            f"<div style='{block_style}'>"
            + "\n".join(summary_lines)
            + "</div>"
            + f"<div style='{block_style}; margin-top: 8px;'>"
            + "\n".join(transition_before_lines + transition_after_lines)
            + "</div>"
            + "<details style='margin-top: 8px;'>"
            + f"<summary>point_gid_changes {html.escape(f'(showing {len(point_gid_changes)} of {len(point_gid_changes_all)})')}</summary>"
            + f"<div style='{block_style}; margin-top: 8px;'>"
            + "\n".join(point_before_lines + [""] + point_after_lines)
            + "</div></details>"
        )


def create_debugger_widget(debugger: CachedSAMInstanceDebugger):
    import ipywidgets as widgets
    from IPython.display import HTML, display
    import matplotlib.pyplot as plt

    panel_keys = [
        ("new_points", "new geom"),
        ("last_seed_masks", "last seed"),
        ("current_masks", "current masks"),
        ("current_points", "current points"),
        ("collapsed_before", "collapsed before"),
        ("collapsed_after", "collapsed after"),
        ("all_gids_before", "all gids before"),
        ("all_gids_after", "all gids after"),
        ("filter_depth", "filter 1"),
        ("filter_unmatched", "filter 2"),
        ("filter_normal", "filter 3"),
        ("depth_map", "depth map"),
    ]
    image_out = widgets.Output()
    text_out = widgets.Output()
    frame_target = widgets.IntText(value=max(0, debugger.current_frame_id), description="frame")
    run_to_btn = widgets.Button(description="Run To")
    run_all_btn = widgets.Button(description="Run All")
    step_btn = widgets.Button(description="Step")
    next_seed_btn = widgets.Button(description="Next Seed")
    reset_btn = widgets.Button(description="Reset")
    progress = widgets.IntProgress(value=0, min=0, max=1, description="Idle", layout=widgets.Layout(width="420px"))
    status = widgets.HTML(value="")
    panel_toggles = {
        key: widgets.Checkbox(value=True, description=label, indent=False, layout=widgets.Layout(width="140px"))
        for key, label in panel_keys
    }

    def refresh() -> None:
        with image_out:
            image_out.clear_output(wait=True)
            if debugger.current_view is None:
                print("No frame processed yet.")
            else:
                enabled_panels = {key: toggle.value for key, toggle in panel_toggles.items()}
                panel = debugger.render_current_panel(enabled_panels=enabled_panels)
                fig, ax = plt.subplots(figsize=(24, 18))
                ax.imshow(panel)
                ax.axis("off")
                fig.tight_layout()
                display(fig)
                plt.close(fig)
        with text_out:
            text_out.clear_output(wait=True)
            display(HTML(debugger.current_html_summary()))

    def set_busy(is_busy: bool, *, description: str = "Idle", value: int = 0, maximum: int = 1) -> None:
        run_to_btn.disabled = is_busy
        run_all_btn.disabled = is_busy
        step_btn.disabled = is_busy
        next_seed_btn.disabled = is_busy
        reset_btn.disabled = is_busy
        frame_target.disabled = is_busy
        progress.max = max(1, int(maximum))
        progress.value = int(min(max(0, value), progress.max))
        progress.description = description
        status.value = "" if not is_busy else f"<pre>{description}: {progress.value}/{progress.max}</pre>"

    def run_to_target(target: int, *, description: str) -> None:
        target = int(target)
        if target < 0 or target >= debugger.total_frames:
            raise ValueError(f"frame must be in [0, {debugger.total_frames - 1}]")
        if target < debugger.current_frame_id:
            debugger.reset()
        total_steps = max(0, target - debugger.current_frame_id)
        set_busy(True, description=description, value=0, maximum=max(1, total_steps))
        try:
            while debugger.current_frame_id < target:
                debugger.step()
                progress.value = min(progress.max, progress.value + 1)
                status.value = f"<pre>{description}: {progress.value}/{progress.max} (frame {debugger.current_frame_id})</pre>"
        finally:
            set_busy(False)
        frame_target.value = max(0, debugger.current_frame_id)
        refresh()

    def on_step(_):
        try:
            debugger.step()
            frame_target.value = debugger.current_frame_id
        except Exception as exc:  # noqa: BLE001
            with text_out:
                text_out.clear_output(wait=True)
                display(HTML(f"<pre>{html.escape(str(exc))}</pre>"))
        else:
            refresh()

    def on_run_to(_):
        try:
            run_to_target(int(frame_target.value), description="Run To")
        except Exception as exc:  # noqa: BLE001
            with text_out:
                text_out.clear_output(wait=True)
                display(HTML(f"<pre>{html.escape(str(exc))}</pre>"))

    def on_run_all(_):
        try:
            set_busy(True, description="Run All", value=0, maximum=1)
            debugger.run_all(upto=-1, progress_desc="Run All")
            frame_target.value = max(0, debugger.current_frame_id)
        except Exception as exc:  # noqa: BLE001
            with text_out:
                text_out.clear_output(wait=True)
                display(HTML(f"<pre>{html.escape(str(exc))}</pre>"))
        finally:
            set_busy(False)
        refresh()

    def on_next_seed(_):
        try:
            target = debugger.current_frame_id + 1
            while target < debugger.total_frames and (target % debugger.cache.map_every) != 0:
                target += 1
            if target >= debugger.total_frames:
                raise IndexError("No later seed frame exists in the cache.")
            run_to_target(target, description="Next Seed")
        except Exception as exc:  # noqa: BLE001
            with text_out:
                text_out.clear_output(wait=True)
                display(HTML(f"<pre>{html.escape(str(exc))}</pre>"))

    def on_reset(_):
        debugger.reset()
        frame_target.value = 0
        set_busy(False)
        refresh()

    step_btn.on_click(on_step)
    run_to_btn.on_click(on_run_to)
    run_all_btn.on_click(on_run_all)
    next_seed_btn.on_click(on_next_seed)
    reset_btn.on_click(on_reset)
    for toggle in panel_toggles.values():
        toggle.observe(lambda change: refresh() if change["name"] == "value" else None, names="value")

    controls = widgets.HBox([reset_btn, step_btn, next_seed_btn, frame_target, run_to_btn, run_all_btn, progress])
    toggles_row_1 = widgets.HBox([panel_toggles[key] for key, _ in panel_keys[:4]])
    toggles_row_2 = widgets.HBox([panel_toggles[key] for key, _ in panel_keys[4:8]])
    toggles_row_3 = widgets.HBox([panel_toggles[key] for key, _ in panel_keys[8:12]])
    panel = widgets.VBox([controls, status, toggles_row_1, toggles_row_2, toggles_row_3, image_out, text_out])
    set_busy(False)
    refresh()
    return panel
