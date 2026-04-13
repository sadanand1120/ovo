from __future__ import annotations

from pathlib import Path

import numpy as np

from .sam2_tracking import SAM2VideoTracker, build_label_masks
from .sam_masks import SAMMaskExtractor


INSTANCE_LABEL_SLOTS = 10
INSTANCE_MATCH_IN_FRAC_TH = 0.85
INSTANCE_MATCH_OUT_FRAC_TH = 0.15
INSTANCE_MATCH_MIN_SUPPORT_POINTS = 64
INSTANCE_SEED_MIN_VISIBLE_POINTS = 80
INSTANCE_TRACK_MIN_VISIBLE_POINTS = 64
INSTANCE_PRUNE_EVERY_FRAMES = 128
INSTANCE_PRUNE_MAX_AGE_FRAMES = 2000
INSTANCE_PRUNE_MIN_SUPPORT_FRAC = 0.05
INSTANCE_PRUNE_MIN_POINTS = 2000
INSTANCE_TRACKER_MAX_OBJECTS = 32
INSTANCE_LABEL_FILE = "instance_labels.npy"
INSTANCE_STATE_FILE = "instance_state.npy"
INSTANCE_PUBLIC_COLLAPSE = "highest_support_then_points_then_recency"


def build_instance_mask_extractors(*, device: str) -> tuple[SAMMaskExtractor, SAMMaskExtractor, bool]:
    seed_mask_extractor = SAMMaskExtractor(device)
    return seed_mask_extractor, seed_mask_extractor, True


def compact_label_rows(rows: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return rows
    valid = rows >= 0
    order = np.argsort(~valid, axis=1, kind="stable")
    return np.take_along_axis(rows, order, axis=1)


class SAMInstancePipeline:
    def __init__(
        self,
        *,
        device: str,
        total_num_frames: int,
    ) -> None:
        seed_mask_extractor, textregion_mask_extractor, shared_amg_extractor = build_instance_mask_extractors(device=device)
        self.seed_mask_extractor = seed_mask_extractor
        self.textregion_mask_extractor = textregion_mask_extractor
        self.shared_amg_extractor = bool(shared_amg_extractor)
        self.total_num_frames = int(total_num_frames)
        self.point_multi_labels = np.empty((0, INSTANCE_LABEL_SLOTS), dtype=np.int32)
        self.buckets: dict[int, dict[str, int]] = {}
        self.next_gid = 0
        self.active_gids: set[int] = set()
        self.tracker: SAM2VideoTracker | None = None
        self.segment_next_local_idx = 0
        self.stats = {
            "instance_seed_frames": 0,
            "instance_seed_new_gids": 0,
            "instance_seed_existing_matches": 0,
            "instance_nonseed_supported_masks": 0,
            "instance_tracker_seed_object_truncations": 0,
            "instance_label_overflow_drops": 0,
            "instance_prune_runs": 0,
            "instance_pruned_age": 0,
            "instance_pruned_support": 0,
            "instance_pruned_points": 0,
        }
        self._supported_gids_in_frame: set[int] = set()

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        self.segment_next_local_idx = 0
        self.active_gids.clear()

    def num_active_instances(self) -> int:
        return len(self.active_gids)

    def num_existing_instances(self) -> int:
        return len(self.buckets)

    def extract_seed_labels(self, image_np: np.ndarray) -> np.ndarray:
        return self.seed_mask_extractor.extract_labels(image_np)

    def extract_textregion_labels(self, image_np: np.ndarray, seed_labels_np: np.ndarray) -> np.ndarray:
        if self.textregion_mask_extractor is self.seed_mask_extractor:
            return seed_labels_np
        return self.textregion_mask_extractor.extract_labels(image_np)

    def extend_for_new_points(self, n_new: int) -> None:
        if n_new <= 0:
            return
        padding = np.full((int(n_new), INSTANCE_LABEL_SLOTS), -1, dtype=np.int32)
        self.point_multi_labels = np.concatenate((self.point_multi_labels, padding), axis=0)

    def maybe_prune(self, frame_id: int, *, final: bool = False) -> None:
        if not final and INSTANCE_PRUNE_EVERY_FRAMES > 0 and ((int(frame_id) + 1) % INSTANCE_PRUNE_EVERY_FRAMES != 0):
            return
        self.stats["instance_prune_runs"] += 1
        min_support_frames = int(np.ceil(INSTANCE_PRUNE_MIN_SUPPORT_FRAC * max(1, self.total_num_frames)))
        remaining_frames = max(0, self.total_num_frames - int(frame_id) - 1)
        to_drop = []
        age_drop = support_drop = points_drop = 0
        for gid, bucket in self.buckets.items():
            drop_age = (int(frame_id) - bucket["last_support_frame"]) > INSTANCE_PRUNE_MAX_AGE_FRAMES
            drop_points = final and (bucket["num_points"] < INSTANCE_PRUNE_MIN_POINTS)
            if final:
                drop_support = bucket["support_frames"] < min_support_frames
            else:
                drop_support = (bucket["support_frames"] + remaining_frames) < min_support_frames
            if not (drop_age or drop_points or drop_support):
                continue
            to_drop.append(int(gid))
            age_drop += int(drop_age)
            support_drop += int(drop_support)
            points_drop += int(drop_points)
        if not to_drop:
            return
        self.stats["instance_pruned_age"] += age_drop
        self.stats["instance_pruned_support"] += support_drop
        self.stats["instance_pruned_points"] += points_drop
        self._drop_gids(np.asarray(to_drop, dtype=np.int32))

    def process_nonseed_frame(self, frame_id: int, image_np: np.ndarray, point_ids_full: np.ndarray) -> np.ndarray:
        self._supported_gids_in_frame.clear()
        tracked_masks = self._track_frame(image_np)
        tracked_labels = np.full(point_ids_full.shape, -1, dtype=np.int32)
        for gid, mask in tracked_masks.items():
            tracked_labels[mask] = int(gid)
            point_ids = self._mask_point_ids(point_ids_full, mask)
            if point_ids.size < INSTANCE_TRACK_MIN_VISIBLE_POINTS:
                continue
            self._add_gid_to_points(point_ids, int(gid))
            self._record_support(int(gid), int(frame_id))
            self.stats["instance_nonseed_supported_masks"] += 1
        self.maybe_prune(int(frame_id))
        if self.buckets:
            tracked_labels[~np.isin(tracked_labels, np.asarray(sorted(self.buckets), dtype=np.int32))] = -1
        else:
            tracked_labels.fill(-1)
        return tracked_labels

    def process_seed_frame(
        self,
        frame_id: int,
        image_np: np.ndarray,
        point_ids_full: np.ndarray,
        seed_labels: np.ndarray,
    ) -> np.ndarray:
        self._supported_gids_in_frame.clear()
        self.stats["instance_seed_frames"] += 1
        visible_point_ids = np.unique(point_ids_full[point_ids_full >= 0])
        inst_img_full = np.full(point_ids_full.shape, -1, dtype=np.int32)
        for _, seed_mask in build_label_masks(seed_labels):
            point_ids = self._mask_point_ids(point_ids_full, seed_mask)
            if point_ids.size < INSTANCE_SEED_MIN_VISIBLE_POINTS:
                continue
            gid = self._match_existing_gid(point_ids, visible_point_ids)
            if gid is None:
                gid = self._create_gid()
                self.stats["instance_seed_new_gids"] += 1
                assign_point_ids = point_ids
            else:
                self.stats["instance_seed_existing_matches"] += 1
                assign_point_ids = point_ids[self.point_multi_labels[point_ids, 0] < 0]
            self._add_gid_to_points(assign_point_ids, gid)
            self._record_support(gid, int(frame_id))
            inst_img_full[seed_mask] = int(gid)
        self.maybe_prune(int(frame_id))
        if self.buckets:
            inst_img_full[~np.isin(inst_img_full, np.asarray(sorted(self.buckets), dtype=np.int32))] = -1
        else:
            inst_img_full.fill(-1)
        self._reseed_tracker(image_np, inst_img_full)
        return inst_img_full

    def build_diagnostic_state(self) -> dict:
        gids = np.asarray(sorted(self.buckets), dtype=np.int32)
        support_frames = np.asarray([self.buckets[int(gid)]["support_frames"] for gid in gids], dtype=np.int32)
        num_points = np.asarray([self.buckets[int(gid)]["num_points"] for gid in gids], dtype=np.int32)
        last_support_frame = np.asarray([self.buckets[int(gid)]["last_support_frame"] for gid in gids], dtype=np.int32)
        return {
            "point_instance_labels_multi": self.point_multi_labels.copy(),
            "bucket_gid": gids,
            "bucket_support_frames": support_frames,
            "bucket_num_points": num_points,
            "bucket_last_support_frame": last_support_frame,
        }

    def collapse_public_labels(self) -> np.ndarray:
        return self._collapse_rows(self.point_multi_labels)

    def primary_labels_for_point_ids(self, point_ids: np.ndarray) -> np.ndarray:
        point_ids = np.asarray(point_ids, dtype=np.int64)
        if point_ids.size == 0:
            return np.empty((0,), dtype=np.int32)
        return self._collapse_rows(self.point_multi_labels[point_ids])

    def project_primary_labels(self, point_ids_full: np.ndarray) -> np.ndarray:
        point_ids_full = np.asarray(point_ids_full, dtype=np.int64)
        labels = np.full(point_ids_full.shape, -1, dtype=np.int32)
        valid = point_ids_full >= 0
        if not valid.any():
            return labels
        labels[valid] = self.primary_labels_for_point_ids(point_ids_full[valid])
        return labels

    def _create_gid(self) -> int:
        gid = int(self.next_gid)
        self.next_gid += 1
        self.buckets[gid] = {
            "support_frames": 0,
            "num_points": 0,
            "last_support_frame": -1,
        }
        return gid

    def _track_frame(self, image_np: np.ndarray) -> dict[int, np.ndarray]:
        if self.tracker is None or not self.active_gids:
            return {}
        local_idx = self.segment_next_local_idx
        self.tracker.append_frame(local_idx, image_np)
        tracked_masks = self.tracker.track_frame(local_idx)
        self.segment_next_local_idx += 1
        return {int(gid): mask for gid, mask in tracked_masks.items() if int(gid) in self.active_gids}

    def _mask_point_ids(self, point_ids_full: np.ndarray, mask: np.ndarray) -> np.ndarray:
        point_ids = np.unique(point_ids_full[mask])
        return point_ids[point_ids >= 0].astype(np.int64, copy=False)

    def _match_existing_gid(self, mask_point_ids: np.ndarray, visible_point_ids: np.ndarray) -> int | None:
        rows_x = self.point_multi_labels[mask_point_ids]
        labeled_x = rows_x[rows_x[:, 0] >= 0]
        if labeled_x.shape[0] == 0:
            return None
        candidate_gids = np.unique(labeled_x[labeled_x >= 0])
        if candidate_gids.size == 0:
            return None
        outside_point_ids = np.setdiff1d(visible_point_ids, mask_point_ids, assume_unique=True)
        rows_y = self.point_multi_labels[outside_point_ids] if outside_point_ids.size > 0 else np.empty((0, INSTANCE_LABEL_SLOTS), dtype=np.int32)
        labeled_y = rows_y[rows_y[:, 0] >= 0]
        best_gid = None
        best_key = None
        for gid in candidate_gids.tolist():
            x_has_gid = np.any(labeled_x == int(gid), axis=1)
            support_points = int(x_has_gid.sum())
            if support_points < INSTANCE_MATCH_MIN_SUPPORT_POINTS:
                continue
            in_frac = float(support_points) / float(labeled_x.shape[0])
            if in_frac < INSTANCE_MATCH_IN_FRAC_TH:
                continue
            if labeled_y.shape[0] == 0:
                out_frac = 0.0
            else:
                out_frac = float(np.any(labeled_y == int(gid), axis=1).sum()) / float(labeled_y.shape[0])
            if out_frac > INSTANCE_MATCH_OUT_FRAC_TH:
                continue
            bucket = self.buckets.get(int(gid))
            if bucket is None:
                continue
            key = (
                support_points,
                in_frac,
                -out_frac,
                int(bucket["support_frames"]),
                int(bucket["num_points"]),
                int(bucket["last_support_frame"]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_gid = int(gid)
        return best_gid

    def _add_gid_to_points(self, point_ids: np.ndarray, gid: int) -> None:
        if point_ids.size == 0:
            return
        point_ids = np.unique(point_ids.astype(np.int64, copy=False))
        rows = self.point_multi_labels[point_ids]
        has_gid = np.any(rows == int(gid), axis=1)
        if np.all(has_gid):
            return
        rows_need = rows[~has_gid]
        point_ids_need = point_ids[~has_gid]
        has_free = np.any(rows_need < 0, axis=1)
        if np.any(~has_free):
            self.stats["instance_label_overflow_drops"] += int((~has_free).sum())
        rows_insert = rows_need[has_free]
        point_ids_insert = point_ids_need[has_free]
        if point_ids_insert.size == 0:
            return
        free_slots = np.argmax(rows_insert < 0, axis=1)
        rows_insert[np.arange(rows_insert.shape[0]), free_slots] = int(gid)
        self.point_multi_labels[point_ids_insert] = rows_insert
        self.buckets[int(gid)]["num_points"] += int(point_ids_insert.size)

    def _record_support(self, gid: int, frame_id: int) -> None:
        gid = int(gid)
        if gid not in self.buckets:
            return
        self.buckets[gid]["last_support_frame"] = int(frame_id)
        if gid in self._supported_gids_in_frame:
            return
        self._supported_gids_in_frame.add(gid)
        self.buckets[gid]["support_frames"] += 1

    def _drop_gids(self, gids: np.ndarray) -> None:
        gids = np.unique(gids.astype(np.int32, copy=False))
        if gids.size == 0:
            return
        gid_set = set(gids.tolist())
        for gid in gids.tolist():
            self.buckets.pop(int(gid), None)
        self.active_gids.difference_update(gid_set)
        if self.point_multi_labels.size == 0:
            return
        mask = np.isin(self.point_multi_labels, gids)
        if not mask.any():
            return
        self.point_multi_labels[mask] = -1
        self.point_multi_labels = compact_label_rows(self.point_multi_labels)

    def _reseed_tracker(self, image_np: np.ndarray, inst_img_full: np.ndarray) -> None:
        survivor_masks = [(int(gid), mask) for gid, mask in build_label_masks(inst_img_full) if int(gid) in self.buckets]
        final_masks = survivor_masks[:INSTANCE_TRACKER_MAX_OBJECTS]
        visible_gids = {int(gid) for gid, _ in final_masks}
        self.stats["instance_tracker_seed_object_truncations"] += max(
            0,
            len(survivor_masks) - len(final_masks),
        )
        self.active_gids = visible_gids
        if not final_masks:
            self.close()
            return
        if self.tracker is None:
            self.tracker = SAM2VideoTracker(image_np)
            self.tracker.reset_and_seed_masks(final_masks)
        else:
            self.tracker.restart_and_seed_masks(image_np, final_masks)
        self.segment_next_local_idx = 1

    def save_outputs(self, output_dir: Path, final_frame_id: int) -> dict:
        self.maybe_prune(int(final_frame_id), final=True)
        public_labels = self.collapse_public_labels()
        diagnostic_state = self.build_diagnostic_state()
        np.save(output_dir / INSTANCE_LABEL_FILE, public_labels)
        np.save(output_dir / INSTANCE_STATE_FILE, diagnostic_state, allow_pickle=True)
        return {
            "instance_label_path": INSTANCE_LABEL_FILE,
            "instance_state_path": INSTANCE_STATE_FILE,
            "instance_label_slots": INSTANCE_LABEL_SLOTS,
            "instance_public_collapse": INSTANCE_PUBLIC_COLLAPSE,
            "instance_match_in_frac_th": INSTANCE_MATCH_IN_FRAC_TH,
            "instance_match_out_frac_th": INSTANCE_MATCH_OUT_FRAC_TH,
            "instance_match_min_support_points": INSTANCE_MATCH_MIN_SUPPORT_POINTS,
            "instance_seed_min_visible_points": INSTANCE_SEED_MIN_VISIBLE_POINTS,
            "instance_track_min_visible_points": INSTANCE_TRACK_MIN_VISIBLE_POINTS,
            "instance_prune_every_frames": INSTANCE_PRUNE_EVERY_FRAMES,
            "instance_prune_max_age_frames": INSTANCE_PRUNE_MAX_AGE_FRAMES,
            "instance_prune_min_support_frac": INSTANCE_PRUNE_MIN_SUPPORT_FRAC,
            "instance_prune_min_points": INSTANCE_PRUNE_MIN_POINTS,
            "instance_tracker_max_objects": INSTANCE_TRACKER_MAX_OBJECTS,
            "instance_final_gid_count": len(self.buckets),
            **self.stats,
        }

    def _collapse_rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int32)
        public_labels = np.full((rows.shape[0],), -1, dtype=np.int32)
        if rows.size == 0 or not self.buckets:
            return public_labels

        max_gid = max(max(self.buckets) + 1, self.next_gid)
        support_arr = np.full((max_gid,), -1, dtype=np.int32)
        num_points_arr = np.full((max_gid,), -1, dtype=np.int32)
        last_frame_arr = np.full((max_gid,), -1, dtype=np.int32)
        for gid, bucket in self.buckets.items():
            support_arr[int(gid)] = int(bucket["support_frames"])
            num_points_arr[int(gid)] = int(bucket["num_points"])
            last_frame_arr[int(gid)] = int(bucket["last_support_frame"])

        best_support = np.full((rows.shape[0],), -1, dtype=np.int32)
        best_points = np.full((rows.shape[0],), -1, dtype=np.int32)
        best_last = np.full((rows.shape[0],), -1, dtype=np.int32)
        for slot in range(INSTANCE_LABEL_SLOTS):
            gids = rows[:, slot]
            valid = gids >= 0
            if not valid.any():
                continue
            support = np.full((rows.shape[0],), -1, dtype=np.int32)
            points = np.full((rows.shape[0],), -1, dtype=np.int32)
            last = np.full((rows.shape[0],), -1, dtype=np.int32)
            support[valid] = support_arr[gids[valid]]
            points[valid] = num_points_arr[gids[valid]]
            last[valid] = last_frame_arr[gids[valid]]
            better = valid & (
                (support > best_support)
                | ((support == best_support) & (points > best_points))
                | ((support == best_support) & (points == best_points) & (last > best_last))
            )
            public_labels[better] = gids[better]
            best_support[better] = support[better]
            best_points[better] = points[better]
            best_last[better] = last[better]
        return public_labels
