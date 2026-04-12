from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


TEXT_COLOR = (255, 255, 255)
TEXT_BG = (20, 20, 20)
INSTANCE_VIDEO_NAME = "debug_instances.mp4"
TEXTREGION_VIDEO_NAME = "debug_textregion_clip.mp4"


def color_for_id(label: int) -> np.ndarray:
    if label < 0:
        return np.zeros(3, dtype=np.uint8)
    x = (int(label) * 2654435761) & 0xFFFFFFFF
    return np.array(
        [
            np.uint8((x >> 0) & 255),
            np.uint8((x >> 8) & 255),
            np.uint8((x >> 16) & 255),
        ],
        dtype=np.uint8,
    )


def colorize_label_map(labels: np.ndarray, *, dilate_kernel: int = 0) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    if labels.size == 0:
        return colored
    kernel = None
    if dilate_kernel > 1:
        kernel = np.ones((int(dilate_kernel), int(dilate_kernel)), dtype=np.uint8)
    unique_labels = np.unique(labels)
    for label in unique_labels.tolist():
        if label < 0:
            continue
        mask = labels == int(label)
        if kernel is not None:
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        colored[mask] = color_for_id(int(label))
    return colored


def overlay_labels_on_rgb(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.60,
    dilate_kernel: int = 0,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    colors = colorize_label_map(labels, dilate_kernel=dilate_kernel)
    overlay = rgb.copy()
    mask = np.any(colors > 0, axis=2)
    if mask.any():
        overlay[mask] = np.clip(
            (1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * colors[mask].astype(np.float32),
            0.0,
            255.0,
        ).astype(np.uint8)
    return overlay


def overlay_header(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), TEXT_BG, thickness=-1)
    cv2.putText(canvas, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, TEXT_COLOR, 2, cv2.LINE_AA)
    return canvas


def _resize_feature_map_for_debug(feature_map: torch.Tensor, max_side: int) -> torch.Tensor:
    feature_map = feature_map.permute(2, 0, 1)[None].float()
    h, w = feature_map.shape[-2:]
    if max(h, w) <= max_side:
        return feature_map[0].permute(1, 2, 0).contiguous()
    scale = float(max_side) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return resized[0].permute(1, 2, 0).contiguous()


def _joint_pca_rgb(
    feature_maps: list[torch.Tensor],
    *,
    sample_size: int,
) -> list[np.ndarray]:
    prepared = [_resize_feature_map_for_debug(feature_map, max_side=160) for feature_map in feature_maps]
    flattened = [feat.reshape(-1, feat.shape[-1]).float() for feat in prepared]
    joint = torch.cat(flattened, dim=0)
    if joint.shape[0] == 0:
        return [np.zeros((*feat.shape[:2], 3), dtype=np.uint8) for feat in prepared]
    step = max(1, joint.shape[0] // max(1, int(sample_size)))
    sample = joint[::step][:sample_size]
    mean = sample.mean(dim=0, keepdim=True)
    centered = sample - mean
    _, _, v = torch.pca_lowrank(centered, q=3, niter=5)
    proj_v = v[:, :3]
    low_rank_sample = centered @ proj_v
    low_rank_min = torch.quantile(low_rank_sample, 0.01, dim=0)
    low_rank_max = torch.quantile(low_rank_sample, 0.99, dim=0)
    denom = (low_rank_max - low_rank_min).clamp_min(1e-6)
    outputs = []
    for prepared_feat, feat in zip(prepared, flattened):
        low_rank = (feat - mean) @ proj_v
        rgb = ((low_rank - low_rank_min) / denom).clamp_(0.0, 1.0)
        outputs.append((rgb.reshape(*prepared_feat.shape[:2], 3).cpu().numpy() * 255.0).astype(np.uint8))
    return outputs


def _resize_to_match(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    if image.shape[:2] == (target_h, target_w):
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


class RGBMapDebugVideoWriter:
    def __init__(self, output_dir: Path, fps: int) -> None:
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.instance_writer: cv2.VideoWriter | None = None
        self.textregion_writer: cv2.VideoWriter | None = None
        self.latest_seed_frame_id: int | None = None
        self.latest_seed_rgb: np.ndarray | None = None
        self.latest_seed_labels: np.ndarray | None = None
        self.latest_seed_source: str = "seed"

    def close(self) -> None:
        if self.instance_writer is not None:
            self.instance_writer.release()
            self.instance_writer = None
        if self.textregion_writer is not None:
            self.textregion_writer.release()
            self.textregion_writer = None

    def update_latest_seed(
        self,
        frame_id: int,
        rgb: np.ndarray,
        seed_labels: np.ndarray,
        *,
        seed_source: str,
    ) -> None:
        self.latest_seed_frame_id = int(frame_id)
        self.latest_seed_rgb = np.asarray(rgb, dtype=np.uint8).copy()
        self.latest_seed_labels = np.asarray(seed_labels, dtype=np.int32).copy()
        self.latest_seed_source = seed_source

    def _ensure_writer(self, writer: cv2.VideoWriter | None, path: Path, frame_shape: tuple[int, int, int], n_panels: int) -> cv2.VideoWriter:
        if writer is not None:
            return writer
        height, width = frame_shape[:2]
        video_writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width * n_panels, height),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {path}")
        return video_writer

    def write_instance_frame(
        self,
        *,
        frame_id: int,
        rgb: np.ndarray,
        current_labels: np.ndarray | None,
        global_labels: np.ndarray | None,
        is_seed_frame: bool,
    ) -> None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        frame_shape = rgb.shape
        self.instance_writer = self._ensure_writer(
            self.instance_writer,
            self.output_dir / INSTANCE_VIDEO_NAME,
            frame_shape,
            4,
        )

        if self.latest_seed_rgb is None or self.latest_seed_labels is None:
            latest_seed_panel = rgb.copy()
            latest_seed_subtitle = "latest_seed=none"
        else:
            latest_seed_panel = overlay_labels_on_rgb(self.latest_seed_rgb, self.latest_seed_labels, alpha=0.65)
            latest_seed_subtitle = f"latest_seed={self.latest_seed_frame_id} source={self.latest_seed_source}"
            latest_seed_panel = _resize_to_match(latest_seed_panel, rgb.shape[:2])

        current_panel = rgb.copy() if current_labels is None else overlay_labels_on_rgb(rgb, current_labels, alpha=0.65)
        global_panel = rgb.copy() if global_labels is None else overlay_labels_on_rgb(rgb, global_labels, alpha=0.60, dilate_kernel=3)

        grid = np.hstack(
            (
                overlay_header(rgb, "RGB", f"frame={frame_id}"),
                overlay_header(latest_seed_panel, "Latest Seed Frame", latest_seed_subtitle),
                overlay_header(
                    current_panel,
                    "Current Active Detections",
                    f"frame={frame_id} source={'seed+track merge' if is_seed_frame else 'tracking only'}",
                ),
                overlay_header(
                    global_panel,
                    "Global Instances Projected",
                    f"frame={frame_id} view=global accumulation",
                ),
            )
        )
        self.instance_writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    def write_textregion_frame(
        self,
        *,
        frame_id: int,
        rgb: np.ndarray,
        raw_clip_dense: torch.Tensor,
        textregion_labels: np.ndarray,
        textregion_clip_dense: torch.Tensor,
        mask_source: str,
    ) -> None:
        rgb = np.asarray(rgb, dtype=np.uint8)
        frame_shape = rgb.shape
        self.textregion_writer = self._ensure_writer(
            self.textregion_writer,
            self.output_dir / TEXTREGION_VIDEO_NAME,
            frame_shape,
            4,
        )
        raw_pca, tr_pca = _joint_pca_rgb([raw_clip_dense, textregion_clip_dense], sample_size=50_000)
        raw_pca = _resize_to_match(raw_pca, rgb.shape[:2])
        tr_pca = _resize_to_match(tr_pca, rgb.shape[:2])
        mask_panel = overlay_labels_on_rgb(rgb, textregion_labels, alpha=0.65)
        grid = np.hstack(
            (
                overlay_header(rgb, "RGB", f"frame={frame_id}"),
                overlay_header(raw_pca, "CLIP PCA", "pre text-region"),
                overlay_header(mask_panel, "Text-Region Labels", f"frame={frame_id} source={mask_source}"),
                overlay_header(tr_pca, "Text-Regioned CLIP PCA", "post text-region"),
            )
        )
        self.textregion_writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
