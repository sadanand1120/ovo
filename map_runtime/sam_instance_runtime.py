from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

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
    prune_start_at: int = 0
    prune_every_frames: int = 64
    prune_min_support_perc: float = 0.0
    prune_min_points_perc: float = 0.00015


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
        self.debug_support_seed_ordinals_by_gid: dict[int, list[int]] = {}
        self.debug_birth_seed_ordinal_by_gid: dict[int, int] = {}
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

    def keep_point_ids(self, point_ids: np.ndarray) -> None:
        point_ids = np.asarray(point_ids, dtype=np.int64)
        if point_ids.ndim != 1:
            raise ValueError(f"point_ids must be a 1D array, got shape {point_ids.shape}.")
        if point_ids.size == self.point_gids.shape[0]:
            return
        self.point_gids = np.ascontiguousarray(self.point_gids[point_ids])
        point_counts = {int(gid): 0 for gid in self.buckets}
        valid = self.point_gids[self.point_gids >= 0]
        if valid.size > 0:
            gids, counts = np.unique(valid, return_counts=True)
            for gid, count in zip(gids.tolist(), counts.tolist()):
                point_counts[int(gid)] = int(count)
        empty_gids = []
        for gid, bucket in self.buckets.items():
            bucket.point_count = int(point_counts.get(int(gid), 0))
            if bucket.point_count <= 0:
                empty_gids.append(int(gid))
        for gid in empty_gids:
            self.seeded_gids.discard(int(gid))
            self.buckets.pop(int(gid), None)
        self._refresh_gid_scores()

    def num_active_instances(self) -> int:
        return len(self.seeded_gids)

    def num_existing_instances(self) -> int:
        return len(self.buckets)

    def export_collapsed_labels(self) -> np.ndarray:
        return self._collapse_point_gid_labels()

    def export_point_gids(self) -> np.ndarray:
        return np.array(self.point_gids, copy=True)

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
        self.debug_support_seed_ordinals_by_gid[gid] = []
        self.debug_birth_seed_ordinal_by_gid[gid] = int(self._num_seed_frames_processed(upto_frame=int(frame_id)))
        self._refresh_gid_scores()
        self.stats["sam2_births"] += 1
        return gid

    def _record_support(self, gid: int, frame_id: int) -> None:
        bucket = self.buckets.get(int(gid))
        if bucket is None:
            return
        if bucket.last_support_frame != int(frame_id):
            bucket.support_frames += 1
            seed_ordinal = int(self._num_seed_frames_processed(upto_frame=int(frame_id)))
            self.debug_support_seed_ordinals_by_gid.setdefault(int(gid), []).append(seed_ordinal)
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
        if int(frame_id) < int(self.config.pipeline.prune_start_at):
            return []
        if int(frame_id) % int(self.config.pipeline.prune_every_frames) != 0:
            return []
        num_seed_frames = int(self._num_seed_frames_processed(upto_frame=int(frame_id)))
        num_map_points = int(self.point_gids.shape[0])
        if num_seed_frames <= 0 or num_map_points <= 0:
            return []
        pruned = []
        for gid, bucket in list(self.buckets.items()):
            support_perc = float(bucket.support_frames) / float(num_seed_frames)
            points_perc = float(bucket.point_count) / float(num_map_points)
            if (
                support_perc < float(self.config.pipeline.prune_min_support_perc)
                or points_perc < float(self.config.pipeline.prune_min_points_perc)
            ):
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

    def _num_seed_frames_processed(self, *, upto_frame: int | None = None) -> int:
        frame_id = self.current_frame_id if upto_frame is None else int(upto_frame)
        if frame_id < 0:
            return 0
        return 1 + (int(frame_id) // int(self.map_every))

    def _raw_instance_support_scores(self) -> np.ndarray:
        num_seed_frames = int(self._num_seed_frames_processed())
        if self.next_gid <= 0 or num_seed_frames <= 0:
            return np.zeros((max(0, int(self.next_gid)),), dtype=np.float32)
        scores = np.zeros((int(self.next_gid),), dtype=np.float32)
        for gid, bucket in self.buckets.items():
            scores[int(gid)] = float(bucket.support_frames) / float(num_seed_frames)
        return scores

    def _debug_build_gid_stats(self, *, selected_gids: set[int] | None = None) -> dict[int, dict[str, Any]]:
        selected_gids = set() if selected_gids is None else {int(gid) for gid in selected_gids}
        total_seed_frames = int(self._num_seed_frames_processed())
        total_map_points = int(self.point_gids.shape[0])
        gid_stats: dict[int, dict[str, Any]] = {}
        for gid in sorted(self.buckets):
            bucket = self.buckets[int(gid)]
            birth_seed_ordinal = int(self.debug_birth_seed_ordinal_by_gid.get(int(gid), 1))
            support_seed_ordinals = sorted(
                int(v)
                for v in self.debug_support_seed_ordinals_by_gid.get(int(gid), [])
                if 1 <= int(v) <= int(total_seed_frames)
            )
            signal_len = max(0, int(total_seed_frames) - int(birth_seed_ordinal) + 1)
            binary_signal = np.zeros((signal_len,), dtype=np.int32)
            for ordinal in support_seed_ordinals:
                rel_idx = int(ordinal) - int(birth_seed_ordinal)
                if 0 <= rel_idx < signal_len:
                    binary_signal[rel_idx] = 1
            n00 = n01 = n10 = n11 = 0
            if signal_len >= 2:
                prev = binary_signal[:-1]
                curr = binary_signal[1:]
                n00 = int(np.sum((prev == 0) & (curr == 0)))
                n01 = int(np.sum((prev == 0) & (curr == 1)))
                n10 = int(np.sum((prev == 1) & (curr == 0)))
                n11 = int(np.sum((prev == 1) & (curr == 1)))
            support_perc = 0.0 if total_seed_frames <= 0 else (100.0 * float(bucket.support_frames) / float(total_seed_frames))
            point_perc = 0.0 if total_map_points <= 0 else (100.0 * float(bucket.point_count) / float(total_map_points))
            gid_stats[int(gid)] = {
                "gid": int(gid),
                "selected": bool(int(gid) in selected_gids),
                "birth_frame": int(bucket.birth_frame),
                "birth_seed_ordinal": int(birth_seed_ordinal),
                "last_support_frame": int(bucket.last_support_frame),
                "support_frames": int(bucket.support_frames),
                "support_perc": float(support_perc),
                "point_count": int(bucket.point_count),
                "point_perc": float(point_perc),
                "total_seed_frames": int(total_seed_frames),
                "lifecycle_seed_frames": int(signal_len),
                "support_seed_ordinals": [int(v) for v in support_seed_ordinals],
                "binary_signal": binary_signal.astype(np.int32, copy=False).tolist(),
                "n00": int(n00),
                "n01": int(n01),
                "n10": int(n10),
                "n11": int(n11),
            }
        return gid_stats

    def _fit_learned_gid_selector(
        self,
        *,
        selected_gids: set[int],
        embed_dim: int = 8,
        num_layers: int = 2,
        epochs: int = 2000,
        pruning_thresh: float = 0.6,
        loss_mode: str = "wbce",
        focal_gamma: float = 2.0,
        plot_every: int = 25,
        show_training_plot: bool = False,
    ) -> dict[str, Any]:
        num_layers = max(int(num_layers), 2)
        plot_every = max(int(plot_every), 1)
        loss_mode = str(loss_mode)
        if loss_mode not in {"wbce", "focal"}:
            raise ValueError(f"Unknown loss_mode={loss_mode!r}. Expected one of: wbce, focal.")
        gid_stats = self._debug_build_gid_stats(selected_gids=selected_gids)
        ordered_gids = sorted(int(gid) for gid in gid_stats)
        if not ordered_gids:
            return {
                "learned_num_gids": 0,
                "learned_num_selected": 0,
                "learned_num_rejected": 0,
                "learned_embed_dim": int(embed_dim),
                "learned_num_layers": int(num_layers),
                "learned_epochs": int(epochs),
                "learned_pruning_thresh": float(pruning_thresh),
                "learned_loss_mode": loss_mode,
                "learned_focal_alpha": float("nan"),
                "learned_focal_gamma": float(focal_gamma),
                "learned_plot_every": int(plot_every),
                "learned_train_loss_final": float("nan"),
                "learned_train_acc_final": float("nan"),
                "learned_train_selected_acc_final": float("nan"),
                "learned_train_non_selected_acc_final": float("nan"),
                "learned_train_selected_f1_final": float("nan"),
                "learned_feature_names": [
                    "log_support_perc",
                    "log_point_perc",
                    "n00_norm",
                    "n01_norm",
                    "n10_norm",
                    "n11_norm",
                ],
                "learned_rows": [],
                "learned_pred_selected_gids": [],
            }

        eps = 1e-6
        rows = []
        for gid in ordered_gids:
            stats = gid_stats[int(gid)]
            lifecycle_seed_frames = max(int(stats["lifecycle_seed_frames"]), 1)
            rows.append(
                {
                    "gid": int(gid),
                    "selected": float(stats["selected"]),
                    "log_support_perc": float(np.log10(max(float(stats["support_perc"]), eps))),
                    "log_point_perc": float(np.log10(max(float(stats["point_perc"]), eps))),
                    "n00_norm": float(stats["n00"]) / float(lifecycle_seed_frames),
                    "n01_norm": float(stats["n01"]) / float(lifecycle_seed_frames),
                    "n10_norm": float(stats["n10"]) / float(lifecycle_seed_frames),
                    "n11_norm": float(stats["n11"]) / float(lifecycle_seed_frames),
                }
            )

        feature_names = ["log_support_perc", "log_point_perc", "n00_norm", "n01_norm", "n10_norm", "n11_norm"]
        train_device = torch.device(self.device)
        X = torch.tensor([[row[name] for name in feature_names] for row in rows], dtype=torch.float32, device=train_device)
        y = torch.tensor([row["selected"] for row in rows], dtype=torch.float32, device=train_device).unsqueeze(1)
        y_flat = y.squeeze(1)
        pos_mask = y_flat > 0.5
        neg_mask = ~pos_mask
        num_pos = int(pos_mask.sum().item())
        num_neg = int(neg_mask.sum().item())
        X_mean = X.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        X_norm = (X - X_mean) / X_std

        torch.manual_seed(0)
        trunk_layers: list[torch.nn.Module] = [
            torch.nn.Linear(X_norm.shape[1], int(embed_dim)),
            torch.nn.ReLU(),
        ]
        for _ in range(num_layers - 2):
            trunk_layers.extend(
                [
                    torch.nn.Linear(int(embed_dim), int(embed_dim)),
                    torch.nn.ReLU(),
                ]
            )
        trunk = torch.nn.Sequential(*trunk_layers).to(train_device)
        keep_head = torch.nn.Linear(int(embed_dim), 1).to(train_device)
        params = list(trunk.parameters()) + list(keep_head.parameters())
        opt = torch.optim.Adam(params, lr=1e-2, weight_decay=1e-4)
        pos_weight = 1.0 if num_pos <= 0 or num_neg <= 0 else float(num_neg) / float(num_pos)
        focal_alpha = 0.5 if (num_pos + num_neg) <= 0 else float(num_neg) / float(num_pos + num_neg)
        num_epochs = int(epochs)
        loss_history: list[float] = []
        acc_history: list[float] = []
        pos_acc_history: list[float] = []
        neg_acc_history: list[float] = []
        f1_history: list[float] = []

        if show_training_plot:
            import matplotlib.pyplot as plt
            from IPython.display import clear_output, display

        for epoch in range(num_epochs):
            opt.zero_grad()
            hidden = trunk(X_norm)
            keep_logits = keep_head(hidden)
            if loss_mode == "wbce":
                keep_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    keep_logits,
                    y,
                    pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=train_device),
                )
            else:
                probs_loss = torch.sigmoid(keep_logits)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(keep_logits, y, reduction="none")
                pt = torch.where(y > 0.5, probs_loss, 1.0 - probs_loss).clamp_min(1e-8)
                alpha_t = torch.where(
                    y > 0.5,
                    torch.full_like(y, float(focal_alpha)),
                    torch.full_like(y, 1.0 - float(focal_alpha)),
                )
                keep_loss = (alpha_t * ((1.0 - pt) ** float(focal_gamma)) * bce).mean()
            keep_loss.backward()
            opt.step()
            with torch.no_grad():
                hidden_epoch = trunk(X_norm)
                probs_epoch = torch.sigmoid(keep_head(hidden_epoch).squeeze(1))
                preds_epoch = (probs_epoch >= float(pruning_thresh)).float()
                acc_epoch = float((preds_epoch == y.squeeze(1)).float().mean().item())
                pos_acc_epoch = float((preds_epoch[pos_mask] == y_flat[pos_mask]).float().mean().item()) if num_pos > 0 else float("nan")
                neg_acc_epoch = float((preds_epoch[neg_mask] == y_flat[neg_mask]).float().mean().item()) if num_neg > 0 else float("nan")
                pred_pos_epoch = preds_epoch > 0.5
                tp_epoch = int((pred_pos_epoch & pos_mask).sum().item())
                fp_epoch = int((pred_pos_epoch & neg_mask).sum().item())
                fn_epoch = int(((~pred_pos_epoch) & pos_mask).sum().item())
                precision_epoch = float(tp_epoch / max(tp_epoch + fp_epoch, 1))
                recall_epoch = float(tp_epoch / max(tp_epoch + fn_epoch, 1))
                f1_epoch = float((2.0 * precision_epoch * recall_epoch) / max(precision_epoch + recall_epoch, 1e-8))
            loss_history.append(float(keep_loss.item()))
            acc_history.append(acc_epoch)
            pos_acc_history.append(pos_acc_epoch)
            neg_acc_history.append(neg_acc_epoch)
            f1_history.append(f1_epoch)
            if show_training_plot and (epoch == 0 or (epoch + 1) % plot_every == 0 or (epoch + 1) == num_epochs):
                clear_output(wait=True)
                fig, axes = plt.subplots(1, 5, figsize=(22, 3.5))
                axes[0].plot(loss_history, linewidth=2)
                axes[0].set_xlabel("epoch")
                axes[0].set_ylabel("loss")
                axes[0].set_title(f"train loss | epoch={epoch + 1}/{num_epochs}")
                axes[0].grid(True, alpha=0.3)
                axes[1].plot(acc_history, linewidth=2)
                axes[1].set_xlabel("epoch")
                axes[1].set_ylabel("train acc")
                axes[1].set_title(f"train acc | current={acc_history[-1]:.4f}")
                axes[1].set_ylim(0.0, 1.0)
                axes[1].grid(True, alpha=0.3)
                axes[2].plot(pos_acc_history, linewidth=2)
                axes[2].set_xlabel("epoch")
                axes[2].set_ylabel("selected acc")
                axes[2].set_title(f"selected acc | current={pos_acc_history[-1]:.4f}" if num_pos > 0 else "selected acc | n/a")
                axes[2].set_ylim(0.0, 1.0)
                axes[2].grid(True, alpha=0.3)
                axes[3].plot(neg_acc_history, linewidth=2)
                axes[3].set_xlabel("epoch")
                axes[3].set_ylabel("non-selected acc")
                axes[3].set_title(f"non-selected acc | current={neg_acc_history[-1]:.4f}" if num_neg > 0 else "non-selected acc | n/a")
                axes[3].set_ylim(0.0, 1.0)
                axes[3].grid(True, alpha=0.3)
                axes[4].plot(f1_history, linewidth=2)
                axes[4].set_xlabel("epoch")
                axes[4].set_ylabel("selected F1")
                axes[4].set_title(f"selected F1 | current={f1_history[-1]:.4f}")
                axes[4].set_ylim(0.0, 1.0)
                axes[4].grid(True, alpha=0.3)
                fig.tight_layout()
                display(fig)
                plt.close(fig)

        with torch.no_grad():
            hidden = trunk(X_norm)
            keep_logits = keep_head(hidden).squeeze(1)
            probs = torch.sigmoid(keep_logits)
            preds = (probs >= float(pruning_thresh)).float()
            acc = float((preds == y.squeeze(1)).float().mean().item())
            pos_acc = float((preds[pos_mask] == y_flat[pos_mask]).float().mean().item()) if num_pos > 0 else float("nan")
            neg_acc = float((preds[neg_mask] == y_flat[neg_mask]).float().mean().item()) if num_neg > 0 else float("nan")
            pred_pos = preds > 0.5
            tp = int((pred_pos & pos_mask).sum().item())
            fp = int((pred_pos & neg_mask).sum().item())
            fn = int(((~pred_pos) & pos_mask).sum().item())
            precision = float(tp / max(tp + fp, 1))
            recall = float(tp / max(tp + fn, 1))
            f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-8))
            probs_cpu = probs.detach().cpu()
            preds_cpu = preds.detach().cpu()

        ranked_rows = sorted(
            [
                {
                    **row,
                    "keep_prob": float(prob),
                    "pred_selected": bool(pred),
                }
                for row, prob, pred in zip(rows, probs_cpu.tolist(), preds_cpu.tolist())
            ],
            key=lambda row: (-row["keep_prob"], row["gid"]),
        )
        pred_selected_gids = sorted(int(row["gid"]) for row in ranked_rows if bool(row["pred_selected"]))
        diagnostics = {
            "learned_num_gids": int(len(ordered_gids)),
            "learned_num_selected": int(y.sum().item()),
            "learned_num_rejected": int(len(ordered_gids) - int(y.sum().item())),
            "learned_embed_dim": int(embed_dim),
            "learned_num_layers": int(num_layers),
            "learned_epochs": int(epochs),
            "learned_pruning_thresh": float(pruning_thresh),
            "learned_loss_mode": loss_mode,
            "learned_focal_alpha": float(focal_alpha),
            "learned_focal_gamma": float(focal_gamma),
            "learned_plot_every": int(plot_every),
            "learned_pos_weight": float(pos_weight),
            "learned_train_loss_final": float(keep_loss.item()),
            "learned_train_acc_final": float(acc),
            "learned_train_selected_acc_final": float(pos_acc),
            "learned_train_non_selected_acc_final": float(neg_acc),
            "learned_train_selected_f1_final": float(f1),
            "learned_train_device": str(train_device),
            "learned_feature_names": feature_names,
            "learned_rows": ranked_rows,
            "learned_pred_selected_gids": pred_selected_gids,
        }
        return diagnostics

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
            seeded_masks = self._tracker.reset_and_seed_masks(seed_pairs)
        else:
            seeded_masks = self._tracker.restart_and_seed_masks(rgb, seed_pairs)
        self.seeded_gids = {int(gid) for gid in seeded_masks}
        self.tracker_frame_idx = 1
        return {int(gid): np.asarray(mask, dtype=bool) for gid, mask in seeded_masks.items()}

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
