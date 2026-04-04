from typing import Any, Dict, List

import cuvslam
import numpy as np
import torch

from .vanilla_mapper import VanillaMapper


def _quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat.astype(np.float32)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def _pose_to_matrix(pose: cuvslam.Pose, device: str) -> torch.Tensor:
    pose = pose.pose if hasattr(pose, "pose") else pose
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _quat_xyzw_to_rotmat(np.asarray(pose.rotation))
    c2w[:3, 3] = np.asarray(pose.translation, dtype=np.float32)
    return torch.from_numpy(c2w).to(device)


class WrapperCuVSLAM(VanillaMapper):
    """cuVSLAM-backed pose estimation with OVO's existing depth unprojection mapper."""

    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor, world_ref=torch.eye(4)) -> None:
        super().__init__(config, cam_intrinsics)
        self.world_ref = world_ref.to(self.device)
        self.timestamp_step_ns = int(1e9 / float(config["slam"].get("fps", 30.0)))
        self.depth_scale_factor = float(config["cam"]["depth_scale"])
        self.last_mapped_c2w = None
        self.map_translation_th = 0.05
        self.map_rotation_th = np.deg2rad(5.0)

        camera = cuvslam.Camera()
        camera.size = np.array([self.width, self.height], dtype=np.int32)
        camera.focal = np.array(
            [
                float(cam_intrinsics[0, 0].detach().cpu()),
                float(cam_intrinsics[1, 1].detach().cpu()),
            ],
            dtype=np.float32,
        )
        camera.principal = np.array(
            [
                float(cam_intrinsics[0, 2].detach().cpu()),
                float(cam_intrinsics[1, 2].detach().cpu()),
            ],
            dtype=np.float32,
        )

        rgbd = cuvslam.Tracker.OdometryRGBDSettings()
        rgbd.depth_scale_factor = self.depth_scale_factor
        rgbd.depth_camera_id = 0
        rgbd.enable_depth_stereo_tracking = False

        odom_config = cuvslam.Tracker.OdometryConfig()
        odom_config.async_sba = False
        odom_config.enable_final_landmarks_export = False
        odom_config.odometry_mode = cuvslam.Tracker.OdometryMode.RGBD
        odom_config.rgbd_settings = rgbd

        self.tracker = cuvslam.Tracker(cuvslam.Rig([camera]), odom_config)

    @property
    def height(self) -> int:
        return int(self.config["cam"]["H"] - 2 * self.config["cam"].get("crop_edge", 0))

    @property
    def width(self) -> int:
        return int(self.config["cam"]["W"] - 2 * self.config["cam"].get("crop_edge", 0))

    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id, rgb_image, depth_image = frame_data[:3]
        rgb_image = np.ascontiguousarray(rgb_image)
        depth_image = np.ascontiguousarray(
            np.clip(
                np.rint(depth_image.astype(np.float32) * self.depth_scale_factor),
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
        )
        pose_estimate, _ = self.tracker.track(
            frame_id * self.timestamp_step_ns,
            images=[rgb_image],
            depths=[depth_image],
        )
        if pose_estimate.world_from_rig is None:
            return
        self.estimated_c2ws[frame_id] = self.world_ref @ _pose_to_matrix(pose_estimate.world_from_rig, self.device)

    def should_map_frame(self, frame_id: int, c2w: torch.Tensor) -> bool:
        if self.last_mapped_c2w is None:
            return True
        relative = torch.linalg.inv(self.last_mapped_c2w) @ c2w
        translation = torch.linalg.norm(relative[:3, 3]).item()
        trace = relative[:3, :3].trace().item()
        trace = max(min((trace - 1.0) * 0.5, 1.0), -1.0)
        rotation = float(np.arccos(trace))
        return translation >= self.map_translation_th or rotation >= self.map_rotation_th

    def map(self, frame_data: List[Any], c2w: torch.Tensor) -> None:
        n_points = self.pcd.shape[0]
        super().map(frame_data, c2w)
        if self.pcd.shape[0] > n_points:
            self.last_mapped_c2w = c2w.detach().clone()
