import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from orientation_utils import (
    MIN_RENDER_AXIS_LENGTH,
    ORIENTATION_AXIS_LENGTH,
    dump_instance_orientations,
    orientation_path_for_map_dir,
    orthonormalize_rotation,
)


DEFAULT_MIN_COMPONENT_SIZE = 2000
DEFAULT_ROTATION_STEP_DEG = 5.0
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 960
DEFAULT_POINT_SIZE = 1.5
DEFAULT_HIGHLIGHT_POINT_SIZE = 4.0
HIGHLIGHT_COLOR = np.array([1.0, 0.82, 0.10], dtype=np.float32)
BASE_CLOUD_NAME = "rgb_map"
HIGHLIGHT_CLOUD_NAME = "current_instance"
AXES_NAME = "orientation_axes"


def resolve_ply_path(input_path: str) -> Path:
    path = Path(input_path)
    return path if path.suffix == ".ply" else path / "rgb_map.ply"


def resolve_instance_labels(map_dir: Path, n_points: int, min_component_size: int) -> np.ndarray:
    label_path = map_dir / "instance_labels.npy"
    if not label_path.exists():
        raise FileNotFoundError(f"Missing instance labels at {label_path}")
    labels = np.load(label_path)
    if labels.shape[0] != n_points:
        raise ValueError(f"Instance label count mismatch: expected {n_points}, found {labels.shape[0]}")
    valid = labels >= 0
    if valid.any() and min_component_size > 1:
        uniq, counts = np.unique(labels[valid], return_counts=True)
        keep = uniq[counts >= int(min_component_size)]
        labels = labels.copy()
        labels[~np.isin(labels, keep)] = -1
        valid = labels >= 0
    if valid.any():
        _, relabeled = np.unique(labels[valid], return_inverse=True)
        labels[valid] = relabeled.astype(np.int32, copy=False)
    return labels


def rotation_matrix_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rotation_matrix_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def tint_colors(colors: np.ndarray, tint: np.ndarray, amount: float = 0.65) -> np.ndarray:
    return np.clip(colors * (1.0 - amount) + tint.reshape(1, 3) * amount, 0.0, 1.0)


class OrientationAnnotator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ply_path = resolve_ply_path(args.input_path)
        self.map_dir = self.ply_path.parent
        self.output_path = orientation_path_for_map_dir(self.map_dir)

        self.pcd = o3d.io.read_point_cloud(str(self.ply_path))
        self.points = np.asarray(self.pcd.points, dtype=np.float32)
        self.colors = np.asarray(self.pcd.colors, dtype=np.float32)
        if self.points.shape[0] == 0:
            raise ValueError(f"No points found in {self.ply_path}")
        if self.colors.shape[0] != self.points.shape[0]:
            raise ValueError(f"RGB point count mismatch in {self.ply_path}")

        labels = resolve_instance_labels(self.map_dir, self.points.shape[0], args.min_component_size)
        valid_mask = labels >= 0
        if not valid_mask.any():
            raise ValueError("No valid instances found after min_component_size filtering")
        self.instance_ids = np.unique(labels[valid_mask]).astype(np.int32, copy=False)
        self.instance_masks = [labels == instance_id for instance_id in self.instance_ids.tolist()]
        self.instance_point_counts = np.asarray([int(mask.sum()) for mask in self.instance_masks], dtype=np.int32)
        self.centroids = np.asarray([self.points[mask].mean(axis=0) for mask in self.instance_masks], dtype=np.float32)
        self.rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(self.instance_ids), axis=0)
        self.current_index = 0
        self.finished = False
        self.current_label3d = None

        self.axis_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0.0, 0.0, 0.0])
        self.base_pcd = o3d.geometry.PointCloud()
        self.base_pcd.points = o3d.utility.Vector3dVector(self.points.astype(np.float64, copy=False))
        self.base_pcd.colors = o3d.utility.Vector3dVector(self.colors.astype(np.float64, copy=False))

        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window("Instance Orientation Annotator", args.window_width, args.window_height)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)
        self.window.set_on_close(self._on_close)

        self.info_label = gui.Label("")
        self.help_label = gui.Label(
            "Keys: Up/Down pitch  Left/Right yaw  Q/E roll  R reset  B previous  N next/save"
        )
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.scene_widget.scene.show_skybox(False)

        self.window.add_child(self.info_label)
        self.window.add_child(self.help_label)
        self.window.add_child(self.scene_widget)

        base_material = rendering.MaterialRecord()
        base_material.shader = "defaultUnlit"
        base_material.point_size = float(args.point_size)
        self.scene_widget.scene.add_geometry(BASE_CLOUD_NAME, self.base_pcd, base_material)

        self.highlight_material = rendering.MaterialRecord()
        self.highlight_material.shader = "defaultUnlit"
        self.highlight_material.point_size = float(args.highlight_point_size)

        self.axes_material = rendering.MaterialRecord()
        self.axes_material.shader = "defaultUnlit"
        self.scene_widget.scene.add_geometry(AXES_NAME, self.axis_mesh, self.axes_material)

        bbox = self.base_pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bbox, bbox.get_center())
        self._update_current_instance()

    def _on_layout(self, layout_context: gui.LayoutContext) -> None:
        rect = self.window.content_rect
        em = int(np.ceil(layout_context.theme.font_size))
        margin = max(8, em // 2)
        info_height = em * 4
        help_height = em * 2
        self.info_label.frame = gui.Rect(rect.x + margin, rect.y + margin, rect.width - 2 * margin, info_height)
        self.help_label.frame = gui.Rect(rect.x + margin, rect.y + margin + info_height, rect.width - 2 * margin, help_height)
        scene_y = rect.y + margin + info_height + help_height + margin
        self.scene_widget.frame = gui.Rect(rect.x, scene_y, rect.width, max(1, rect.height - (scene_y - rect.y)))

    def _current_axis_vectors(self) -> np.ndarray:
        return (self.rotations[self.current_index] * ORIENTATION_AXIS_LENGTH).T.astype(np.float32, copy=False)

    def _update_info_label(self) -> None:
        inst_id = int(self.instance_ids[self.current_index])
        centroid = self.centroids[self.current_index]
        axes = self._current_axis_vectors()
        self.info_label.text = (
            f"Instance {self.current_index + 1}/{len(self.instance_ids)}"
            f"  id={inst_id}  points={int(self.instance_point_counts[self.current_index])}  axis_len={ORIENTATION_AXIS_LENGTH:.3f}\n"
            f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})\n"
            f"x={axes[0, 0]: .3f} {axes[0, 1]: .3f} {axes[0, 2]: .3f}    "
            f"y={axes[1, 0]: .3f} {axes[1, 1]: .3f} {axes[1, 2]: .3f}    "
            f"z={axes[2, 0]: .3f} {axes[2, 1]: .3f} {axes[2, 2]: .3f}"
        )

    def _update_highlight_geometry(self) -> None:
        mask = self.instance_masks[self.current_index]
        instance_pcd = o3d.geometry.PointCloud()
        instance_pcd.points = o3d.utility.Vector3dVector(self.points[mask].astype(np.float64, copy=False))
        instance_pcd.colors = o3d.utility.Vector3dVector(tint_colors(self.colors[mask], HIGHLIGHT_COLOR).astype(np.float64, copy=False))
        if self.scene_widget.scene.has_geometry(HIGHLIGHT_CLOUD_NAME):
            self.scene_widget.scene.remove_geometry(HIGHLIGHT_CLOUD_NAME)
        self.scene_widget.scene.add_geometry(HIGHLIGHT_CLOUD_NAME, instance_pcd, self.highlight_material)

    def _update_axes_geometry(self) -> None:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = (self.rotations[self.current_index] * max(ORIENTATION_AXIS_LENGTH, MIN_RENDER_AXIS_LENGTH)).astype(np.float64, copy=False)
        transform[:3, 3] = self.centroids[self.current_index].astype(np.float64, copy=False)
        self.scene_widget.scene.set_geometry_transform(AXES_NAME, transform)
        if self.current_label3d is not None:
            self.scene_widget.remove_3d_label(self.current_label3d)
        self.current_label3d = self.scene_widget.add_3d_label(
            self.centroids[self.current_index].astype(np.float32, copy=False),
            f"{self.current_index + 1}/{len(self.instance_ids)}",
        )

    def _update_current_instance(self) -> None:
        self._update_info_label()
        self._update_highlight_geometry()
        self._update_axes_geometry()
        self.window.post_redraw()

    def _rotate_current(self, local_rotation: np.ndarray) -> None:
        self.rotations[self.current_index] = orthonormalize_rotation(self.rotations[self.current_index] @ local_rotation)
        self._update_info_label()
        self._update_axes_geometry()
        self.window.post_redraw()

    def _advance(self, delta: int) -> None:
        new_index = int(np.clip(self.current_index + delta, 0, len(self.instance_ids) - 1))
        if new_index != self.current_index:
            self.current_index = new_index
            self._update_current_instance()

    def _save_and_close(self) -> None:
        dump_instance_orientations(
            self.output_path,
            self.ply_path,
            self.args.min_component_size,
            self.instance_ids,
            self.centroids,
            self.rotations,
            self.instance_point_counts,
        )
        self.finished = True
        print(self.output_path)
        self.window.close()

    def _on_key(self, event: gui.KeyEvent) -> bool:
        if event.type != gui.KeyEvent.DOWN:
            return False

        step_rad = np.deg2rad(float(self.args.rotation_step_deg))
        if event.key == gui.KeyName.UP:
            self._rotate_current(rotation_matrix_x(step_rad))
            return True
        if event.key == gui.KeyName.DOWN:
            self._rotate_current(rotation_matrix_x(-step_rad))
            return True
        if event.key == gui.KeyName.LEFT:
            self._rotate_current(rotation_matrix_y(step_rad))
            return True
        if event.key == gui.KeyName.RIGHT:
            self._rotate_current(rotation_matrix_y(-step_rad))
            return True
        if event.key == gui.KeyName.Q:
            self._rotate_current(rotation_matrix_z(step_rad))
            return True
        if event.key == gui.KeyName.E:
            self._rotate_current(rotation_matrix_z(-step_rad))
            return True
        if event.key == gui.KeyName.R:
            self.rotations[self.current_index] = np.eye(3, dtype=np.float32)
            self._update_current_instance()
            return True
        if event.key == gui.KeyName.B:
            self._advance(-1)
            return True
        if event.key == gui.KeyName.N:
            if self.current_index == len(self.instance_ids) - 1:
                self._save_and_close()
            else:
                self._advance(1)
            return True
        return False

    def _on_close(self) -> bool:
        if not self.finished:
            print("Closed without saving orientations.")
        gui.Application.instance.quit()
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively annotate per-instance orientation axes on a saved RGB map.")
    parser.add_argument("input_path", help="Path to rgb_map.ply or its containing directory.")
    parser.add_argument("--min_component_size", type=int, default=DEFAULT_MIN_COMPONENT_SIZE)
    parser.add_argument("--rotation_step_deg", type=float, default=DEFAULT_ROTATION_STEP_DEG)
    parser.add_argument("--point_size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--highlight_point_size", type=float, default=DEFAULT_HIGHLIGHT_POINT_SIZE)
    parser.add_argument("--window_width", type=int, default=DEFAULT_WINDOW_WIDTH)
    parser.add_argument("--window_height", type=int, default=DEFAULT_WINDOW_HEIGHT)
    return parser.parse_args()


def main() -> None:
    annotator = OrientationAnnotator(parse_args())
    gui.Application.instance.run()
    if annotator.finished:
        print(
            {
                "orientation_path": str(annotator.output_path),
                "n_instances": int(len(annotator.instance_ids)),
                "min_component_size": int(annotator.args.min_component_size),
            }
        )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
