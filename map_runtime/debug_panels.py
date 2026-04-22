from __future__ import annotations

import cv2
import numpy as np


TEXT_COLOR = (255, 255, 255)
TEXT_BG = (20, 20, 20)
HEADER_HEIGHT = 72


def _disable_panel(panel: np.ndarray, enabled: bool) -> np.ndarray:
    panel = np.asarray(panel, dtype=np.uint8)
    if enabled:
        return panel
    return np.full_like(panel, 255, dtype=np.uint8)


def color_for_id(label: int) -> np.ndarray:
    if label < 0:
        return np.zeros(3, dtype=np.uint8)
    hue = int((int(label) * 47 + 13) % 180)
    hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb[0, 0]


def colorize_label_map(labels: np.ndarray, *, dilate_kernel: int = 0) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    if labels.size == 0:
        return colored
    kernel = None
    if dilate_kernel > 1:
        kernel = np.ones((int(dilate_kernel), int(dilate_kernel)), dtype=np.uint8)
    for label in np.unique(labels).tolist():
        if label < 0:
            continue
        mask = labels == int(label)
        if kernel is not None:
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        colored[mask] = color_for_id(int(label))
    return colored


def render_sorted_label_map(labels: np.ndarray | None) -> np.ndarray:
    if labels is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int32)
    canvas = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = labels >= 0
    if not valid.any():
        return canvas
    ids, counts = np.unique(labels[valid], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    for label in order.tolist():
        canvas[labels == int(label)] = color_for_id(int(label))
    return canvas


def render_multi_gid_projection(gid_rows: np.ndarray | None) -> np.ndarray:
    if gid_rows is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    gid_rows = np.asarray(gid_rows, dtype=np.int32)
    if gid_rows.ndim != 3:
        raise ValueError(f"Expected HxWxK gid rows, got shape {gid_rows.shape}")
    canvas = np.zeros((*gid_rows.shape[:2], 3), dtype=np.uint8)
    valid = gid_rows >= 0
    if not valid.any():
        return canvas
    gids, counts = np.unique(gid_rows[valid], return_counts=True)
    order = gids[np.argsort(-counts, kind="stable")]
    for gid in order.tolist():
        canvas[np.any(gid_rows == int(gid), axis=2)] = color_for_id(int(gid))
    return canvas


def overlay_labels_on_rgb(
    rgb: np.ndarray,
    labels: np.ndarray | None,
    *,
    alpha: float = 0.60,
    dilate_kernel: int = 0,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if labels is None:
        return rgb.copy()
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


def overlay_binary_mask(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (255, 255, 0),
    alpha: float = 0.50,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if mask is None:
        return rgb.copy()
    mask = np.asarray(mask, dtype=bool)
    overlay = rgb.copy()
    if mask.any():
        overlay[mask] = np.clip(
            (1.0 - alpha) * overlay[mask].astype(np.float32) + alpha * np.asarray(color, dtype=np.float32),
            0.0,
            255.0,
        ).astype(np.uint8)
    return overlay


def render_depth_map(depth: np.ndarray | None) -> np.ndarray:
    if depth is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float32)
    canvas = np.zeros((*depth.shape, 3), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth >= 0.0)
    if valid.any():
        valid_depth = depth[valid]
        depth_min = float(valid_depth.min())
        depth_max = float(np.quantile(valid_depth, 0.99))
        if depth_max <= depth_min + 1e-8:
            gray = np.full(depth.shape, 255, dtype=np.uint8)
        else:
            clipped = np.clip(depth, depth_min, depth_max)
            gray = np.clip((clipped - depth_min) / (depth_max - depth_min), 0.0, 1.0)
            gray = np.round(gray * 255.0).astype(np.uint8)
        canvas[valid] = np.stack([gray[valid], gray[valid], gray[valid]], axis=1)
    negative = np.isfinite(depth) & (depth < 0.0)
    if negative.any():
        canvas[negative] = np.array([255, 0, 0], dtype=np.uint8)
    return canvas


def overlay_header(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    canvas = np.zeros((image.shape[0] + HEADER_HEIGHT, image.shape[1], 3), dtype=np.uint8)
    canvas[:HEADER_HEIGHT] = TEXT_BG
    canvas[HEADER_HEIGHT:] = image
    cv2.putText(canvas, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.70, TEXT_COLOR, 2, cv2.LINE_AA)
    return canvas


def _mask_count_subtitle(mask: np.ndarray | None, fallback: str) -> str:
    if mask is None:
        return fallback
    return f"count={int(np.asarray(mask, dtype=bool).sum()):,}"


def compose_instance_debug_grid(
    *,
    rgb: np.ndarray,
    frame_id: int,
    last_seed_labels: np.ndarray | None,
    current_labels: np.ndarray | None,
    collapsed_before: np.ndarray | None,
    collapsed_after: np.ndarray | None,
    all_gids_before: np.ndarray | None,
    all_gids_after: np.ndarray | None,
    current_title: str,
    current_subtitle: str,
    last_seed_subtitle: str,
    new_point_mask: np.ndarray | None = None,
    current_point_mask: np.ndarray | None = None,
    new_point_title: str = "New Geometry Points",
    new_point_subtitle: str = "",
    current_point_title: str = "Current Map Points",
    current_point_subtitle: str = "",
    depth: np.ndarray | None = None,
    filter_depth_mask: np.ndarray | None = None,
    filter_unmatched_mask: np.ndarray | None = None,
    filter_normal_mask: np.ndarray | None = None,
    enabled_panels: dict[str, bool] | None = None,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    enabled_panels = {} if enabled_panels is None else dict(enabled_panels)
    new_points_panel = _disable_panel(
        overlay_header(
        overlay_binary_mask(rgb, new_point_mask, color=(255, 255, 0), alpha=0.65),
        new_point_title,
        new_point_subtitle or _mask_count_subtitle(new_point_mask, f"frame={frame_id}"),
        ),
        enabled_panels.get("new_points", True),
    )
    last_seed_panel = _disable_panel(
        overlay_header(render_sorted_label_map(last_seed_labels), "Last Seed SAM Masks", last_seed_subtitle),
        enabled_panels.get("last_seed_masks", True),
    )
    current_panel = _disable_panel(
        overlay_header(render_sorted_label_map(current_labels), current_title, current_subtitle),
        enabled_panels.get("current_masks", True),
    )
    current_points_panel = _disable_panel(
        overlay_header(
        overlay_binary_mask(rgb, current_point_mask, color=(0, 255, 255), alpha=0.65),
        current_point_title,
        current_point_subtitle or _mask_count_subtitle(current_point_mask, f"frame={frame_id}"),
        ),
        enabled_panels.get("current_points", True),
    )
    collapsed_before_panel = _disable_panel(
        overlay_header(
        render_sorted_label_map(collapsed_before),
        "Collapsed GID Projection",
        "before current-frame update",
        ),
        enabled_panels.get("collapsed_before", True),
    )
    collapsed_after_panel = _disable_panel(
        overlay_header(
        render_sorted_label_map(collapsed_after),
        "Collapsed GID Projection",
        "after current-frame update",
        ),
        enabled_panels.get("collapsed_after", True),
    )
    all_gids_before_panel = _disable_panel(
        overlay_header(
        render_multi_gid_projection(all_gids_before),
        "All GIDs Projection",
        "before current-frame update",
        ),
        enabled_panels.get("all_gids_before", True),
    )
    all_gids_after_panel = _disable_panel(
        overlay_header(
        render_multi_gid_projection(all_gids_after),
        "All GIDs Projection",
        "after current-frame update",
        ),
        enabled_panels.get("all_gids_after", True),
    )
    filter_depth_panel = _disable_panel(
        overlay_header(
        overlay_binary_mask(rgb, filter_depth_mask, color=(255, 128, 0), alpha=0.65),
        "Filter 1: depth > 0",
        _mask_count_subtitle(filter_depth_mask, "seed frame only"),
        ),
        enabled_panels.get("filter_depth", True),
    )
    filter_unmatched_panel = _disable_panel(
        overlay_header(
        overlay_binary_mask(rgb, filter_unmatched_mask, color=(255, 0, 255), alpha=0.65),
        "Filter 2: unmatched",
        _mask_count_subtitle(filter_unmatched_mask, "seed frame only"),
        ),
        enabled_panels.get("filter_unmatched", True),
    )
    filter_normal_panel = _disable_panel(
        overlay_header(
        overlay_binary_mask(rgb, filter_normal_mask, color=(0, 255, 0), alpha=0.65),
        "Filter 3: valid normals",
        _mask_count_subtitle(filter_normal_mask, "seed frame only"),
        ),
        enabled_panels.get("filter_normal", True),
    )
    depth_neg_mask = None if depth is None else np.asarray(depth, dtype=np.float32) < 0.0
    depth_subtitle = "seed frame only" if depth is None else f"neg={int(np.asarray(depth_neg_mask, dtype=bool).sum()):,}"
    depth_panel = _disable_panel(
        overlay_header(
        render_depth_map(depth),
        "Depth Map",
        depth_subtitle,
        ),
        enabled_panels.get("depth_map", True),
    )
    top_row = np.hstack((new_points_panel, last_seed_panel, current_panel, current_points_panel))
    bottom_row = np.hstack((collapsed_before_panel, collapsed_after_panel, all_gids_before_panel, all_gids_after_panel))
    filter_row = np.hstack((depth_panel, filter_depth_panel, filter_unmatched_panel, filter_normal_panel))
    return np.vstack((top_row, bottom_row, filter_row))
