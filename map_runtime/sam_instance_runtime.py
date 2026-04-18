from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

from map_runtime.sam_masks import SAMMaskExtractor, SAMMaskExtractorConfig
from map_runtime.sam2_tracking import SAM2VideoTracker, SAMTrackerConfig, build_label_masks


@dataclass
class InstanceBucket:
    gid: int
    support_frames: int
    point_count: int
    last_support_frame: int
    birth_frame: int


@dataclass
class MaskDecision:
    frame_id: int
    is_seed_frame: bool
    source: str
    local_mask_id: int | None
    action: str
    gid: int | None
    visible_points: int
    labeled_points: int
    background_points: int
    added_points: int
    overflow_points: int
    inside_frac: float | None = None
    outside_frac: float | None = None
    candidate_gid: int | None = None


@dataclass
class FrameProcessResult:
    frame_id: int
    is_seed_frame: bool
    valid_pose: bool
    seed_labels: np.ndarray | None
    tracker_labels: np.ndarray
    decisions: list[MaskDecision]
    pruned_gids: list[int]


@dataclass(frozen=True)
class SAMInstancePipelineConfig:
    point_gid_slots: int = 10
    reuse_inside_frac_th: float = 0.40
    reuse_outside_frac_th: float = 0.10
    min_mask_points: int = 1
    min_track_visible_points: int = 1
    prune_every_frames: int = 64
    prune_stale_gap_frames: int = 100000
    prune_min_support_ratio: float = 0.0
    prune_min_points: int = 1000


@dataclass(frozen=True)
class SAMInstanceRuntimeConfig:
    seed_mask: SAMMaskExtractorConfig = field(default_factory=SAMMaskExtractorConfig)
    tracker: SAMTrackerConfig = field(default_factory=SAMTrackerConfig)
    pipeline: SAMInstancePipelineConfig = field(default_factory=SAMInstancePipelineConfig)


class SAMInstanceRuntime:
    def __init__(
        self,
        config: SAMInstanceRuntimeConfig | None = None,
        *,
        device: str,
        total_frames: int,
        map_every: int,
        n_points: int = 0,
    ) -> None:
        self.config = SAMInstanceRuntimeConfig() if config is None else config
        self.device = device
        self.total_frames = int(total_frames)
        self.map_every = int(map_every)
        self._mask_extractor: SAMMaskExtractor | None = None
        self._tracker: SAM2VideoTracker | None = None
        self._reset_runtime_state(n_points=n_points)

    def _reset_runtime_state(self, *, n_points: int = 0) -> None:
        self.stats = {
            "sam2_births": 0,
            "sam2_pruned_instances": 0,
            "sam2_seed_object_truncations": 0,
        }
        self.point_gids = np.full(
            (int(n_points), int(self.config.pipeline.point_gid_slots)),
            -1,
            dtype=np.int32,
        )
        self.buckets: dict[int, InstanceBucket] = {}
        self.next_gid = 0
        self.current_frame_id = -1
        self.tracker_frame_idx = 0
        self.seeded_gids: set[int] = set()
        self._gid_scores = np.zeros((0,), dtype=np.int64)
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None

    def reset(self, *, n_points: int = 0) -> None:
        self._reset_runtime_state(n_points=n_points)

    def close(self) -> None:
        if self._tracker is not None:
            self._tracker.close()
            self._tracker = None
        self.seeded_gids.clear()
        self.tracker_frame_idx = 0

    @property
    def seed_mask_extractor(self) -> SAMMaskExtractor:
        return self._ensure_mask_extractor()

    def ensure_point_capacity(self, n_points: int) -> None:
        n_points = int(n_points)
        if n_points <= self.point_gids.shape[0]:
            return
        extra = np.full(
            (n_points - self.point_gids.shape[0], int(self.config.pipeline.point_gid_slots)),
            -1,
            dtype=np.int32,
        )
        self.point_gids = np.concatenate((self.point_gids, extra), axis=0)

    def num_active_instances(self) -> int:
        return len(self.seeded_gids)

    def num_existing_instances(self) -> int:
        return len(self.buckets)

    def export_collapsed_labels(self) -> np.ndarray:
        return self._collapse_point_gid_labels()

    def export_support_counts(self) -> np.ndarray:
        support_counts = np.zeros((int(self.next_gid),), dtype=np.int32)
        for gid, bucket in self.buckets.items():
            support_counts[int(gid)] = int(bucket.support_frames)
        return support_counts

    def extract_seed_labels(self, rgb: np.ndarray) -> np.ndarray:
        return self._ensure_mask_extractor().extract_labels(rgb).astype(np.int32, copy=False)

    def _ensure_mask_extractor(self) -> SAMMaskExtractor:
        if self._mask_extractor is None:
            self._mask_extractor = SAMMaskExtractor(
                self.device,
                config=self.config.seed_mask,
            )
        return self._mask_extractor

    def _refresh_gid_scores(self) -> None:
        if self.next_gid <= self._gid_scores.shape[0]:
            self._gid_scores[:] = 0
        else:
            self._gid_scores = np.zeros((self.next_gid,), dtype=np.int64)
        for gid, bucket in self.buckets.items():
            support = min(int(bucket.support_frames), (1 << 21) - 1)
            points = min(int(bucket.point_count), (1 << 21) - 1)
            last = min(int(bucket.last_support_frame), (1 << 20) - 1)
            self._gid_scores[int(gid)] = (support << 42) | (points << 21) | last

    def _create_gid(self, frame_id: int) -> int:
        gid = int(self.next_gid)
        self.next_gid += 1
        self.buckets[gid] = InstanceBucket(
            gid=gid,
            support_frames=0,
            point_count=0,
            last_support_frame=-1,
            birth_frame=int(frame_id),
        )
        self._refresh_gid_scores()
        self.stats["sam2_births"] += 1
        return gid

    def _record_support(self, gid: int, frame_id: int) -> None:
        bucket = self.buckets.get(int(gid))
        if bucket is None:
            return
        if bucket.last_support_frame != int(frame_id):
            bucket.support_frames += 1
        bucket.last_support_frame = int(frame_id)
        self._refresh_gid_scores()

    def _gid_membership_mask(self, point_ids: np.ndarray, gid: int) -> np.ndarray:
        if point_ids.size == 0:
            return np.zeros((0,), dtype=bool)
        rows = self.point_gids[point_ids]
        return np.any(rows == int(gid), axis=1)

    def _add_gid_to_points(self, point_ids: np.ndarray, gid: int, *, background_only: bool) -> tuple[int, int]:
        point_ids = np.asarray(point_ids, dtype=np.int64)
        point_ids = point_ids[point_ids >= 0]
        if point_ids.size == 0:
            return 0, 0
        point_ids = np.unique(point_ids)
        rows = self.point_gids[point_ids].copy()
        has_gid = np.any(rows == int(gid), axis=1)
        eligible = ~np.any(rows >= 0, axis=1) if background_only else np.ones((rows.shape[0],), dtype=bool)
        free = rows == -1
        has_free = np.any(free, axis=1)
        to_add = eligible & ~has_gid & has_free
        if to_add.any():
            free_idx = np.argmax(free, axis=1)
            rows[to_add, free_idx[to_add]] = int(gid)
            self.point_gids[point_ids] = rows
        added = int(to_add.sum())
        overflow = int((eligible & ~has_gid & ~has_free).sum())
        if added > 0:
            self.buckets[int(gid)].point_count += added
            self._refresh_gid_scores()
        return added, overflow

    def _drop_gid(self, gid: int) -> None:
        rows_mask = self.point_gids == int(gid)
        if rows_mask.any():
            affected_rows = np.flatnonzero(np.any(rows_mask, axis=1))
            self.point_gids[rows_mask] = -1
            compacted = np.full_like(self.point_gids[affected_rows], -1)
            for row_idx, row in enumerate(self.point_gids[affected_rows]):
                valid = row[row >= 0]
                compacted[row_idx, : valid.size] = valid
            self.point_gids[affected_rows] = compacted
        self.seeded_gids.discard(int(gid))
        self.buckets.pop(int(gid), None)
        self._refresh_gid_scores()

    def _prune(self, frame_id: int) -> list[int]:
        if self.config.pipeline.prune_every_frames <= 0:
            return []
        if int(frame_id) <= 0 or int(frame_id) % int(self.config.pipeline.prune_every_frames) != 0:
            return []
        min_support_frames = max(1, int(math.ceil(self.config.pipeline.prune_min_support_ratio * self.total_frames)))
        pruned = []
        for gid, bucket in list(self.buckets.items()):
            mature = int(frame_id) - int(bucket.birth_frame) >= min_support_frames
            stale = int(frame_id) - int(bucket.last_support_frame) > int(self.config.pipeline.prune_stale_gap_frames)
            low_support = mature and int(bucket.support_frames) < min_support_frames
            low_points = mature and int(bucket.point_count) < int(self.config.pipeline.prune_min_points)
            if stale or low_support or low_points:
                pruned.append(int(gid))
                self._drop_gid(int(gid))
        self.stats["sam2_pruned_instances"] += int(len(pruned))
        return pruned

    def _project_primary_labels(self, point_ids_image: np.ndarray) -> np.ndarray:
        labels = np.full(point_ids_image.shape, -1, dtype=np.int32)
        valid_mask = point_ids_image >= 0
        if not valid_mask.any():
            return labels
        visible_points = point_ids_image[valid_mask].astype(np.int64, copy=False)
        rows = self.point_gids[visible_points]
        valid_rows = rows >= 0
        if not valid_rows.any():
            return labels
        clipped = rows.clip(min=0)
        scores = np.where(valid_rows, self._gid_scores[clipped], -1)
        best_idx = np.argmax(scores, axis=1)
        best_gid = rows[np.arange(rows.shape[0]), best_idx]
        best_gid[~np.any(valid_rows, axis=1)] = -1
        labels[valid_mask] = best_gid.astype(np.int32, copy=False)
        return labels

    def _project_all_gid_rows(self, point_ids_image: np.ndarray) -> np.ndarray:
        gid_rows = np.full((*point_ids_image.shape, int(self.config.pipeline.point_gid_slots)), -1, dtype=np.int32)
        valid_mask = point_ids_image >= 0
        if not valid_mask.any():
            return gid_rows
        point_ids = point_ids_image[valid_mask].astype(np.int64, copy=False)
        gid_rows[valid_mask] = self.point_gids[point_ids]
        return gid_rows

    def _project_gid_membership(self, point_ids_image: np.ndarray, gid: int) -> np.ndarray:
        mask = np.zeros(point_ids_image.shape, dtype=bool)
        valid_mask = point_ids_image >= 0
        if not valid_mask.any():
            return mask
        point_ids = point_ids_image[valid_mask].astype(np.int64, copy=False)
        mask[valid_mask] = np.any(self.point_gids[point_ids] == int(gid), axis=1)
        return mask

    def _collapse_point_gid_labels(self) -> np.ndarray:
        labels = np.full((self.point_gids.shape[0],), -1, dtype=np.int32)
        valid_rows = self.point_gids >= 0
        if not valid_rows.any():
            return labels
        clipped = self.point_gids.clip(min=0)
        scores = np.where(valid_rows, self._gid_scores[clipped], -1)
        best_idx = np.argmax(scores, axis=1)
        best_gid = self.point_gids[np.arange(self.point_gids.shape[0]), best_idx].astype(np.int32, copy=True)
        best_gid[~np.any(valid_rows, axis=1)] = -1
        return best_gid

    def _num_seed_frames_processed(self) -> int:
        if self.current_frame_id < 0:
            return 0
        return 1 + (int(self.current_frame_id) // int(self.map_every))

    def _raw_instance_support_scores(self) -> np.ndarray:
        num_seed_frames = int(self._num_seed_frames_processed())
        if self.next_gid <= 0 or num_seed_frames <= 0:
            return np.zeros((max(0, int(self.next_gid)),), dtype=np.float32)
        scores = np.zeros((int(self.next_gid),), dtype=np.float32)
        for gid, bucket in self.buckets.items():
            scores[int(gid)] = float(bucket.support_frames) / float(num_seed_frames)
        return scores

    def _resolve_optimal_metric_instance_labels(
        self,
        pred_to_gt_idx: np.ndarray,
        gt_instance_labels: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        usual_collapse_labels = self._collapse_point_gid_labels()
        pred_to_gt_idx = np.asarray(pred_to_gt_idx, dtype=np.int64)
        gt_instance_labels = np.asarray(gt_instance_labels, dtype=np.int32)
        pred_gt_instances = gt_instance_labels[pred_to_gt_idx]
        gid_rows = self.point_gids
        valid_rows = gid_rows >= 0
        unique_gids = np.unique(gid_rows[valid_rows]) if valid_rows.any() else np.empty((0,), dtype=np.int32)
        canonical_gid_by_gt: dict[int, int] = {}
        canonical_rank_by_gt: dict[int, tuple[int, float, int, int, int]] = {}

        for gid in unique_gids.tolist():
            member_mask = np.any(gid_rows == int(gid), axis=1)
            if not member_mask.any():
                continue
            member_gt_instances = pred_gt_instances[member_mask]
            member_gt_instances = member_gt_instances[member_gt_instances >= 0]
            if member_gt_instances.size == 0:
                continue
            gt_ids, gt_counts = np.unique(member_gt_instances, return_counts=True)
            best_idx = int(np.argmax(gt_counts))
            target_gt_instance = int(gt_ids[best_idx])
            intersection = int(gt_counts[best_idx])
            purity = float(intersection / float(member_mask.sum()))
            bucket = self.buckets.get(int(gid))
            support_frames = int(bucket.support_frames) if bucket is not None else 0
            point_count = int(bucket.point_count) if bucket is not None else int(member_mask.sum())
            rank = (intersection, purity, support_frames, point_count, -int(gid))
            current_rank = canonical_rank_by_gt.get(target_gt_instance)
            if current_rank is None or rank > current_rank:
                canonical_rank_by_gt[target_gt_instance] = rank
                canonical_gid_by_gt[target_gt_instance] = int(gid)

        labels = np.full((gid_rows.shape[0],), -1, dtype=np.int32)
        matched_canonical_points = 0
        optimized_points = 0
        for point_idx in range(gid_rows.shape[0]):
            gt_instance = int(pred_gt_instances[point_idx])
            if gt_instance < 0:
                continue
            canonical_gid = canonical_gid_by_gt.get(gt_instance)
            if canonical_gid is None:
                continue
            if np.any(gid_rows[point_idx] == canonical_gid):
                matched_canonical_points += 1
                if usual_collapse_labels[point_idx] != canonical_gid:
                    optimized_points += 1
                labels[point_idx] = int(canonical_gid)

        diagnostics = {
            "optimal_collapse_gt_instances": int(len(canonical_gid_by_gt)),
            "optimal_collapse_points_matched_canonical": int(matched_canonical_points),
            "optimal_collapse_points_overridden": int(optimized_points),
        }
        unique_final = np.unique(labels[labels >= 0])
        if unique_final.size > len(canonical_gid_by_gt):
            raise RuntimeError(
                f"Optimal collapse produced {unique_final.size} unique labels from only {len(canonical_gid_by_gt)} canonical gids."
            )
        return labels, diagnostics

    def _resolve_metric_instance_gid_labels(
        self,
        *,
        use_optimal_collapse: bool = False,
        pred_to_gt_idx: np.ndarray | None = None,
        gt_instance_labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if use_optimal_collapse:
            if pred_to_gt_idx is None or gt_instance_labels is None:
                raise ValueError("Optimal collapse requires pred_to_gt_idx and gt_instance_labels.")
            labels, diagnostics = self._resolve_optimal_metric_instance_labels(pred_to_gt_idx, gt_instance_labels)
        else:
            labels = self._collapse_point_gid_labels()
            diagnostics = {}
        diagnostics = {
            **diagnostics,
            "use_optimal_collapse": bool(use_optimal_collapse),
        }
        return labels, diagnostics

    def _bucket_snapshot(self) -> list[dict[str, Any]]:
        buckets = sorted(
            self.buckets.values(),
            key=lambda bucket: (
                bucket.support_frames,
                bucket.point_count,
                bucket.last_support_frame,
                -bucket.gid,
            ),
            reverse=True,
        )
        return [asdict(bucket) for bucket in buckets]

    def _bucket_snapshot_by_gid(self) -> list[dict[str, Any]]:
        return [asdict(self.buckets[gid]) for gid in sorted(self.buckets)]

    def _build_tracker_labels(self, tracked_masks: dict[int, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        tracker_labels = np.full(shape, -1, dtype=np.int32)
        for gid, mask in sorted(tracked_masks.items(), key=lambda item: int(item[1].sum()), reverse=True):
            tracker_labels[np.asarray(mask, dtype=bool)] = int(gid)
        return tracker_labels

    def _seed_tracker(self, rgb: np.ndarray, gid_to_mask: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        if not gid_to_mask:
            self.close()
            return {}
        seed_pairs = sorted(
            ((int(gid), np.asarray(mask, dtype=bool)) for gid, mask in gid_to_mask.items() if np.asarray(mask, dtype=bool).any()),
            key=lambda item: int(item[1].sum()),
            reverse=True,
        )[: int(self.config.tracker.max_num_objects)]
        if not seed_pairs:
            self.close()
            return {}
        if self._tracker is None:
            self._tracker = SAM2VideoTracker(
                rgb,
                config=self.config.tracker,
            )
            self._tracker.reset_and_seed_masks(seed_pairs)
        else:
            self._tracker.restart_and_seed_masks(rgb, seed_pairs)
        self.seeded_gids = {int(gid) for gid, _ in seed_pairs}
        self.tracker_frame_idx = 1
        return {int(gid): mask for gid, mask in seed_pairs}

    def _step_non_seed(self, frame: dict[str, Any]) -> tuple[np.ndarray, list[MaskDecision]]:
        tracker_labels = np.full(frame["point_ids_after"].shape, -1, dtype=np.int32)
        decisions: list[MaskDecision] = []
        if self._tracker is None or not self.seeded_gids:
            return tracker_labels, decisions
        self._tracker.append_frame(self.tracker_frame_idx, frame["rgb"])
        tracked_masks = self._tracker.track_frame(self.tracker_frame_idx)
        self.tracker_frame_idx += 1
        tracker_labels = self._build_tracker_labels(tracked_masks, frame["point_ids_after"].shape)
        for gid in sorted(self.seeded_gids):
            if gid not in self.buckets:
                continue
            mask = tracked_masks.get(int(gid))
            if mask is None:
                continue
            point_ids = frame["point_ids_after"][np.asarray(mask, dtype=bool)]
            point_ids = np.unique(point_ids[point_ids >= 0])
            added, overflow = self._add_gid_to_points(point_ids, int(gid), background_only=False)
            decisions.append(
                MaskDecision(
                    frame_id=int(frame["frame_id"]),
                    is_seed_frame=False,
                    source="tracker",
                    local_mask_id=None,
                    action="track_assign" if point_ids.size > 0 else "track_no_points",
                    gid=int(gid),
                    visible_points=int(point_ids.size),
                    labeled_points=0,
                    background_points=0,
                    added_points=int(added),
                    overflow_points=int(overflow),
                )
            )
        return tracker_labels, decisions

    def _seed_mask_records(self, point_ids_after: np.ndarray, seed_pairs: list[tuple[int, np.ndarray]]) -> list[dict[str, Any]]:
        records = []
        for local_mask_id, mask in seed_pairs:
            mask = np.asarray(mask, dtype=bool)
            point_ids = np.unique(point_ids_after[mask])
            point_ids = point_ids[point_ids >= 0]
            if point_ids.size < int(self.config.pipeline.min_mask_points):
                records.append(
                    {
                        "local_mask_id": int(local_mask_id),
                        "mask": mask,
                        "point_ids": point_ids,
                        "labeled_points": np.empty((0,), dtype=np.int64),
                        "background_points": np.empty((0,), dtype=np.int64),
                        "candidate_gid": None,
                        "inside_frac": None,
                    }
                )
                continue
            rows = self.point_gids[point_ids]
            labeled_mask = np.any(rows >= 0, axis=1)
            labeled_points = point_ids[labeled_mask]
            background_points = point_ids[~labeled_mask]
            candidate_gid = None
            inside_frac = None
            if labeled_points.size > 0:
                member_gids = self.point_gids[labeled_points]
                member_gids = member_gids[member_gids >= 0]
                if member_gids.size > 0:
                    unique_gids, counts = np.unique(member_gids, return_counts=True)
                    inside_fracs = counts.astype(np.float32) / float(point_ids.size)
                    keep = inside_fracs >= float(self.config.pipeline.reuse_inside_frac_th)
                    if keep.any():
                        kept_gids = unique_gids[keep].astype(np.int64, copy=False)
                        kept_scores = self._gid_scores[kept_gids]
                        best_idx = np.lexsort((kept_gids, -kept_scores))[0]
                        candidate_gid = int(kept_gids[best_idx])
                        inside_frac = float(inside_fracs[keep][best_idx])
            records.append(
                {
                    "local_mask_id": int(local_mask_id),
                    "mask": mask,
                    "point_ids": point_ids,
                    "labeled_points": labeled_points,
                    "background_points": background_points,
                    "candidate_gid": candidate_gid,
                    "inside_frac": inside_frac,
                }
            )
        return records

    def _step_seed(
        self,
        frame: dict[str, Any],
        seed_labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[MaskDecision]]:
        seed_labels = self.extract_seed_labels(frame["rgb"]) if seed_labels is None else np.asarray(seed_labels, dtype=np.int32)
        seed_pairs = build_label_masks(seed_labels)
        decisions: list[MaskDecision] = []
        all_visible_points = np.unique(frame["point_ids_after"][frame["point_ids_after"] >= 0])
        grouped_records: dict[int, list[dict[str, Any]]] = {}
        ungrouped_records: list[dict[str, Any]] = []
        for record in self._seed_mask_records(frame["point_ids_after"], seed_pairs):
            candidate_gid = record["candidate_gid"]
            if candidate_gid is None:
                ungrouped_records.append(record)
            else:
                grouped_records.setdefault(int(candidate_gid), []).append(record)

        gid_to_mask: dict[int, np.ndarray] = {}
        for gid, records in grouped_records.items():
            union_mask = np.zeros_like(records[0]["mask"], dtype=bool)
            union_points = []
            union_background_points = []
            for record in records:
                union_mask |= record["mask"]
                union_points.append(record["point_ids"])
                union_background_points.append(record["background_points"])
            union_points_arr = np.unique(np.concatenate(union_points)) if union_points else np.empty((0,), dtype=np.int64)
            union_background_arr = np.unique(np.concatenate(union_background_points)) if union_background_points else np.empty((0,), dtype=np.int64)
            inside_membership = self._gid_membership_mask(union_points_arr, int(gid))
            inside_frac = None if union_points_arr.size == 0 else float(inside_membership.mean())
            outside_points = np.setdiff1d(all_visible_points, union_points_arr, assume_unique=False)
            outside_membership = self._gid_membership_mask(outside_points, int(gid))
            outside_frac = 0.0 if outside_points.size == 0 else float(outside_membership.mean())
            if (
                union_points_arr.size > 0
                and inside_frac is not None
                and inside_frac >= float(self.config.pipeline.reuse_inside_frac_th)
                and outside_frac <= float(self.config.pipeline.reuse_outside_frac_th)
            ):
                _added_total, overflow = self._add_gid_to_points(union_background_arr, int(gid), background_only=True)
                if union_points_arr.size >= int(self.config.pipeline.min_track_visible_points):
                    self._record_support(int(gid), int(frame["frame_id"]))
                gid_to_mask[int(gid)] = union_mask
                for record in records:
                    record_added = int(self._gid_membership_mask(record["background_points"], int(gid)).sum())
                    decisions.append(
                        MaskDecision(
                            frame_id=int(frame["frame_id"]),
                            is_seed_frame=True,
                            source="seed",
                            local_mask_id=int(record["local_mask_id"]),
                            action="reuse_existing",
                            gid=int(gid),
                            visible_points=int(record["point_ids"].size),
                            labeled_points=int(record["labeled_points"].size),
                            background_points=int(record["background_points"].size),
                            added_points=record_added,
                            overflow_points=int(overflow),
                            inside_frac=inside_frac,
                            outside_frac=float(outside_frac),
                            candidate_gid=int(gid),
                        )
                    )
            else:
                ungrouped_records.extend(records)

        for record in ungrouped_records:
            point_ids = np.asarray(record["point_ids"], dtype=np.int64)
            if point_ids.size == 0:
                decisions.append(
                    MaskDecision(
                        frame_id=int(frame["frame_id"]),
                        is_seed_frame=True,
                        source="seed",
                        local_mask_id=int(record["local_mask_id"]),
                        action="noop_no_points",
                        gid=None,
                        visible_points=0,
                        labeled_points=0,
                        background_points=0,
                        added_points=0,
                        overflow_points=0,
                        candidate_gid=record["candidate_gid"],
                    )
                )
                continue
            gid = self._create_gid(int(frame["frame_id"]))
            added, overflow = self._add_gid_to_points(point_ids, int(gid), background_only=False)
            if point_ids.size >= int(self.config.pipeline.min_track_visible_points):
                self._record_support(int(gid), int(frame["frame_id"]))
            gid_to_mask[int(gid)] = np.asarray(record["mask"], dtype=bool)
            decisions.append(
                MaskDecision(
                    frame_id=int(frame["frame_id"]),
                    is_seed_frame=True,
                    source="seed",
                    local_mask_id=int(record["local_mask_id"]),
                    action="birth_new",
                    gid=int(gid),
                    visible_points=int(record["point_ids"].size),
                    labeled_points=int(record["labeled_points"].size),
                    background_points=int(record["background_points"].size),
                    added_points=int(added),
                    overflow_points=int(overflow),
                    candidate_gid=record["candidate_gid"],
                )
            )

        seeded_masks = self._seed_tracker(frame["rgb"], gid_to_mask)
        self.stats["sam2_seed_object_truncations"] += max(0, len(gid_to_mask) - len(seeded_masks))
        tracker_labels = self._build_tracker_labels(seeded_masks, frame["point_ids_after"].shape)
        return seed_labels.astype(np.int32, copy=False), tracker_labels, decisions

    def process_frame(
        self,
        *,
        frame_id: int,
        is_seed_frame: bool,
        valid_pose: bool,
        rgb: np.ndarray,
        point_ids_after: np.ndarray,
        seed_labels: np.ndarray | None = None,
    ) -> FrameProcessResult:
        frame = {
            "frame_id": int(frame_id),
            "is_seed_frame": bool(is_seed_frame),
            "valid_pose": bool(valid_pose),
            "rgb": np.asarray(rgb, dtype=np.uint8),
            "point_ids_after": np.asarray(point_ids_after, dtype=np.int32),
        }
        tracker_labels = np.full(frame["point_ids_after"].shape, -1, dtype=np.int32)
        decisions: list[MaskDecision] = []
        resolved_seed_labels = None

        if not bool(valid_pose) and bool(is_seed_frame):
            self.close()
        elif bool(is_seed_frame):
            resolved_seed_labels, tracker_labels, decisions = self._step_seed(frame, seed_labels=seed_labels)
        else:
            tracker_labels, decisions = self._step_non_seed(frame)

        pruned_gids = self._prune(int(frame_id))
        self.current_frame_id = int(frame_id)
        return FrameProcessResult(
            frame_id=int(frame_id),
            is_seed_frame=bool(is_seed_frame),
            valid_pose=bool(valid_pose),
            seed_labels=resolved_seed_labels,
            tracker_labels=tracker_labels,
            decisions=decisions,
            pruned_gids=[int(gid) for gid in pruned_gids],
        )
