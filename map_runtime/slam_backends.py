from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from . import geometry


class VanillaMapper:
    """GT-pose depth unprojection mapper used by the custom runtime."""

    def __init__(self, config: dict, cam_intrinsics: torch.Tensor) -> None:
        self.cam_intrinsics = cam_intrinsics
        self.config = config
        self.device = config.get("device", "cuda")
        self.max_frame_points = config["mapping"].get("max_frame_points", 1e5)

        self.match_distance_th = 0.03
        self.max_id = 0
        self.estimated_c2ws = {}
        self.kfs = {}
        self.map_updated = False

        self.pcd = torch.empty((0, 3), device=self.device)
        self.pcd_ids = torch.empty((0, 1), device=self.device, dtype=torch.int32)
        self.pcd_obj_ids = torch.empty((0, 1), device=self.device, dtype=torch.int32)
        self.pcd_colors = torch.empty((0, 3), device=self.device, dtype=torch.uint8)

        k_size = config["mapping"].get("k_pooling", 3)
        if k_size > 1:
            pooling = torch.nn.MaxPool2d(kernel_size=k_size, stride=1, padding=int(k_size / 2))
            self.pooling = lambda mask: ~(pooling((~mask[None]).float())[0].bool())
        else:
            self.pooling = torch.nn.Identity()
        downscale_ratio = config["mapping"].get("downscale_res", 2)
        if downscale_ratio == 1:
            self.downscale = lambda tensor: tensor
        else:
            self.downscale = lambda tensor: tensor[::downscale_ratio, ::downscale_ratio]

    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id = frame_data[0]
        c2w = frame_data[3]
        if np.isinf(c2w).sum() > 0 or np.isnan(c2w).sum() > 0:
            return
        self.estimated_c2ws[frame_id] = torch.from_numpy(c2w).to(self.device)

    def map(self, frame_data: List[Any], c2w: torch.Tensor) -> None:
        image, depth = frame_data[1:3]
        height, width = image.shape[:2]
        y, x = torch.meshgrid(torch.arange(height, device=self.device), torch.arange(width, device=self.device), indexing="ij")
        depth = torch.from_numpy(depth.astype(np.float32)).to(self.device)
        mask = depth > 0

        if self.max_id > 0:
            frustum_corners = geometry.compute_camera_frustum_corners(depth, c2w, self.cam_intrinsics)
            frustum_mask = geometry.compute_frustum_point_ids(self.pcd, frustum_corners, device=self.device)
            _, matches = geometry.match_3d_points_to_2d_pixels(depth, torch.linalg.inv(c2w), self.pcd[frustum_mask], self.cam_intrinsics, self.match_distance_th)
            mask[matches[:, 1], matches[:, 0]] = False
            mask = self.pooling(mask)

        if mask.sum() == 0:
            return

        y, x = self.downscale(y), self.downscale(x)
        depth, mask, image = self.downscale(depth), self.downscale(mask), self.downscale(image)

        x = x[mask]
        y = y[mask]
        depth = depth[mask]
        x_3d = (x - self.cam_intrinsics[0, 2]) * depth / self.cam_intrinsics[0, 0]
        y_3d = (y - self.cam_intrinsics[1, 2]) * depth / self.cam_intrinsics[1, 1]
        z_3d = depth

        points = torch.hstack((x_3d.reshape(-1, 1), y_3d.reshape(-1, 1), z_3d.reshape(-1, 1), torch.ones((x_3d.shape[0], 1), device=self.device)))
        points = torch.einsum("ij,mj->mi", c2w, points)

        self.pcd = torch.vstack((self.pcd, points[:, :3]))
        self.pcd_ids = torch.vstack((self.pcd_ids, torch.arange(self.max_id, self.max_id + points.shape[0], device=self.device, dtype=torch.int32).unsqueeze(1)))
        self.pcd_obj_ids = torch.vstack((self.pcd_obj_ids, torch.ones((points.shape[0], 1), device=self.device, dtype=torch.int32) * -1))
        self.pcd_colors = torch.vstack((self.pcd_colors, torch.from_numpy(image.astype(np.uint8)).to(self.device)[mask].reshape(-1, 3)))
        self.max_id += points.shape[0]

    def should_map_frame(self, frame_id: int, c2w: torch.Tensor) -> bool:
        return frame_id % int(self.config["mapping"].get("map_every", 10)) == 0

    def get_c2w(self, frame_id: int) -> torch.Tensor | None:
        c2w = self.estimated_c2ws.get(frame_id, None)
        if c2w is not None and c2w.device != self.device:
            c2w = c2w.to(self.device)
        return c2w

    def get_map(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pcd, self.pcd_ids, self.pcd_obj_ids.squeeze()

    def get_kfs(self) -> Dict[int, Dict[str, Any]]:
        return self.kfs

    def get_map_dict(self) -> Dict[str, Any]:
        return {
            "xyz": self.pcd.clone().detach().cpu(),
            "obj_ids": self.pcd_obj_ids.clone().detach().cpu(),
            "ids": self.pcd_ids.clone().detach().cpu(),
            "max_id": self.max_id,
            "color": self.pcd_colors.clone().detach().cpu(),
        }

    def set_map_dict(self, map_dict: Dict[str, Any]) -> None:
        self.pcd = map_dict["xyz"].to(self.device)
        self.pcd_obj_ids = map_dict["obj_ids"].to(self.device)
        self.pcd_ids = map_dict["ids"].to(self.device)
        self.max_id = map_dict["max_id"]
        self.pcd_colors = map_dict["color"].to(self.device)

    def get_cam_dict(self) -> dict[str, Any]:
        return {key: item.cpu().numpy() for key, item in self.estimated_c2ws.items()}

    def set_cam_dict(self, cam_dict: dict[str, Any]) -> None:
        self.estimated_c2ws = {int(key): torch.from_numpy(item).to(self.device) for key, item in cam_dict.items()}

    def update_pcd_obj_ids(self, pcd_objs_ids: torch.Tensor):
        self.pcd_obj_ids = pcd_objs_ids.unsqueeze(-1)

    def get_pcd_colors(self) -> np.ndarray:
        return self.pcd_colors.cpu().numpy()


def convert_pose(traj, device):
    pose = torch.cat(
        [
            torch.tensor(traj[-12:], device=device).reshape((3, 4)),
            torch.tensor([[0, 0, 0, 1]], device=device),
        ]
    )
    return pose


SCANNET_ORBSLAM_OVERRIDES = {
    "default": {"Stereo.b": 0.069253, "ORBextractor.nFeatures": 8000, "ORBextractor.iniThFAST": 15},
    "scene0000_00": {"Stereo.b": 0.069000, "ORBextractor.iniThFAST": 20},
    "scene0002_00": {"Stereo.b": 0.070000, "ORBextractor.iniThFAST": 20},
    "scene0050_00": {"Stereo.b": 0.069220},
    "scene0231_00": {"Stereo.b": 0.069220},
    "scene0378_00": {"Stereo.b": 0.069621, "ORBextractor.nFeatures": 4000, "ORBextractor.iniThFAST": 20},
    "scene0518_00": {"Stereo.b": 0.069220, "ORBextractor.nFeatures": 4000, "ORBextractor.iniThFAST": 20},
}


def _load_orb_camera(config: Dict[str, Any], cam_intrinsics: torch.Tensor) -> Dict[str, float]:
    cam_cfg = config["cam"]
    crop_edge = int(cam_cfg.get("crop_edge", 0))
    scene_path = Path(config["data"]["input_path"])
    intrinsic_path = scene_path / "intrinsic" / "intrinsic_depth.txt"

    if intrinsic_path.exists():
        intrinsic = np.loadtxt(intrinsic_path, dtype=np.float32).reshape(4, 4)
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2]) - crop_edge
        cy = float(intrinsic[1, 2]) - crop_edge
    else:
        fx = float(cam_intrinsics[0, 0].detach().cpu())
        fy = float(cam_intrinsics[1, 1].detach().cpu())
        cx = float(cam_intrinsics[0, 2].detach().cpu())
        cy = float(cam_intrinsics[1, 2].detach().cpu())

    return {
        "Camera1.fx": fx,
        "Camera1.fy": fy,
        "Camera1.cx": cx,
        "Camera1.cy": cy,
        "Camera.width": int(cam_cfg["W"] - 2 * crop_edge),
        "Camera.height": int(cam_cfg["H"] - 2 * crop_edge),
    }


def _get_orb_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    if config["dataset_name"] != "ScanNet":
        return {"Stereo.b": 0.069253, "ORBextractor.nFeatures": 8000, "ORBextractor.iniThFAST": 15}
    scene_name = config["data"]["scene_name"]
    overrides = dict(SCANNET_ORBSLAM_OVERRIDES["default"])
    overrides.update(SCANNET_ORBSLAM_OVERRIDES.get(scene_name, {}))
    return overrides


def _build_orbslam_settings(config: Dict[str, Any], cam_intrinsics: torch.Tensor) -> str:
    settings = {
        "File.version": '"1.0"',
        "Camera.type": '"PinHole"',
        **_load_orb_camera(config, cam_intrinsics),
        "Camera1.k1": 0.0,
        "Camera1.k2": 0.0,
        "Camera1.p1": 0.0,
        "Camera1.p2": 0.0,
        "Camera1.k3": 0.0,
        "Camera.fps": 30,
        "Camera.RGB": 1,
        "Stereo.ThDepth": 40.0,
        "RGBD.DepthMapFactor": 1.0,
        "ORBextractor.scaleFactor": 1.2,
        "ORBextractor.nLevels": 12,
        "ORBextractor.minThFAST": 7,
        "Viewer.KeyFrameSize": 0.05,
        "Viewer.KeyFrameLineWidth": 1.0,
        "Viewer.GraphLineWidth": 0.9,
        "Viewer.PointSize": 2.0,
        "Viewer.CameraSize": 0.08,
        "Viewer.CameraLineWidth": 3.0,
        "Viewer.ViewpointX": 0.0,
        "Viewer.ViewpointY": -0.7,
        "Viewer.ViewpointZ": -1.8,
        "Viewer.ViewpointF": 500.0,
    }
    settings.update(_get_orb_overrides(config))

    lines = ["%YAML:1.0", ""]
    lines.extend(f"{key}: {value}" for key, value in settings.items())
    return "\n".join(lines) + "\n"


class WrapperORBSLAM(VanillaMapper):
    """ORB-SLAM3-backed pose provider with vanilla depth unprojection mapping."""

    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor, world_ref=torch.eye(4)) -> None:
        super().__init__(config, cam_intrinsics)
        import orbslam3 as orbslam

        self._orbslam_module = orbslam
        self.close_loops = config["slam"].get("close_loops", True)
        self.last_big_change_id = 0
        self.map_updated = False
        self.world_ref = world_ref.to(self.device)
        self.kfs = {}

        repo_root = Path(__file__).resolve().parents[1]
        vocab_path = repo_root / "thirdParty" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"
        assert vocab_path.exists(), f"ORB vocabulary not found, review path {vocab_path}"
        self.temp_dir = tempfile.TemporaryDirectory(prefix="rgbmap_orbslam_")
        orbslam_config_path = Path(self.temp_dir.name) / f"{config['data']['scene_name']}.yaml"
        orbslam_config_path.write_text(_build_orbslam_settings(config, cam_intrinsics))

        self.orbslam = orbslam.System(
            str(vocab_path),
            str(orbslam_config_path),
            orbslam.Sensor.RGBD,
            config["slam"].get("use_viewer", False),
            not self.close_loops,
        )
        self.orbslam.initialize()

    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id, rgb_image, depth_image = frame_data[:3]
        self.orbslam.process_image_rgbd(rgb_image, depth_image, frame_id)
        tracking_state = self.orbslam.get_tracking_state()
        if tracking_state == self._orbslam_module.TrackingState.OK:
            orb_c2w = self.orbslam.get_last_trajectory_point()
            assert int(orb_c2w[0]) == frame_id, "Retrieved wrong frame pose"
            self.estimated_c2ws[frame_id] = self.world_ref @ convert_pose(orb_c2w, device=self.device)

    def should_map_frame(self, frame_id: int, c2w: torch.Tensor) -> bool:
        return True

    def map(self, frame_data, c2w) -> None:
        if self.orbslam.is_last_frame_kf():
            frame_id = frame_data[0]
            first_p_idx = self.pcd_ids.shape[0]
            super().map(frame_data, c2w)
            last_p_idx = self.pcd_ids.shape[0]
            self.kfs[frame_id] = {"id": frame_id, "pcd_idxs": (first_p_idx, last_p_idx)}

        last_big_change_id = self.orbslam.get_last_big_change_idx()
        if self.close_loops and last_big_change_id != self.last_big_change_id:
            self.last_big_change_id = last_big_change_id
            self.update_map()

    def update_map(self):
        updated_kfs = self.orbslam.get_keyframe_points()

        new_kfs = {}
        new_pcd = []
        new_pcd_ids = []
        new_pcd_obj_ids = []
        new_pcd_colors = []
        new_c2w = {}
        n_points = 0
        for updated_kf in updated_kfs:
            kf_id = int(updated_kf[0])
            kf = self.kfs.get(kf_id)
            if kf is None:
                continue

            kf_c2w = self.estimated_c2ws[kf["id"]]
            updated_kf_c2w = self.world_ref @ convert_pose(updated_kf[1:13], device=self.device)
            transform = updated_kf_c2w @ torch.linalg.inv(kf_c2w)
            updated_kf_pcd = torch.einsum(
                "mn,bn->bm",
                transform,
                torch.cat(
                    [
                        self.pcd[kf["pcd_idxs"][0] : kf["pcd_idxs"][1]],
                        torch.ones((kf["pcd_idxs"][1] - kf["pcd_idxs"][0], 1), device=self.device),
                    ],
                    dim=1,
                ),
            )[:, :3]

            old_n_points = n_points
            n_points += len(updated_kf_pcd)
            new_kfs[kf_id] = {"id": kf["id"], "pcd_idxs": (old_n_points, n_points)}
            new_pcd.append(updated_kf_pcd)
            new_pcd_ids.append(self.pcd_ids[kf["pcd_idxs"][0] : kf["pcd_idxs"][1]])
            new_pcd_obj_ids.append(self.pcd_obj_ids[kf["pcd_idxs"][0] : kf["pcd_idxs"][1]])
            new_pcd_colors.append(self.pcd_colors[kf["pcd_idxs"][0] : kf["pcd_idxs"][1]])
            new_c2w[kf["id"]] = updated_kf_c2w

        self.estimated_c2ws = new_c2w
        self.kfs = new_kfs
        self.pcd = torch.cat(new_pcd, dim=0)
        self.pcd_ids = torch.cat(new_pcd_ids, dim=0)
        self.pcd_obj_ids = torch.cat(new_pcd_obj_ids, dim=0)
        self.pcd_colors = torch.cat(new_pcd_colors, dim=0)
        self.map_updated = True

    def __del__(self) -> None:
        if hasattr(self, "orbslam"):
            self.orbslam.shutdown()
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()


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


def _pose_to_matrix(pose, device: str) -> torch.Tensor:
    pose = pose.pose if hasattr(pose, "pose") else pose
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _quat_xyzw_to_rotmat(np.asarray(pose.rotation))
    c2w[:3, 3] = np.asarray(pose.translation, dtype=np.float32)
    return torch.from_numpy(c2w).to(device)


class WrapperCuVSLAM(VanillaMapper):
    """cuVSLAM-backed pose provider with vanilla depth unprojection mapping."""

    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor, world_ref=torch.eye(4)) -> None:
        super().__init__(config, cam_intrinsics)
        import cuvslam

        self._cuvslam_module = cuvslam
        self.world_ref = world_ref.to(self.device)
        self.timestamp_step_ns = int(1e9 / float(config["slam"].get("fps", 30.0)))
        self.depth_scale_factor = float(config["cam"]["depth_scale"])
        self.last_mapped_c2w = None
        self.map_translation_th = 0.05
        self.map_rotation_th = np.deg2rad(5.0)

        camera = cuvslam.Camera()
        camera.size = np.array([self.width, self.height], dtype=np.int32)
        camera.focal = np.array(
            [float(cam_intrinsics[0, 0].detach().cpu()), float(cam_intrinsics[1, 1].detach().cpu())],
            dtype=np.float32,
        )
        camera.principal = np.array(
            [float(cam_intrinsics[0, 2].detach().cpu()), float(cam_intrinsics[1, 2].detach().cpu())],
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
            np.clip(np.rint(depth_image.astype(np.float32) * self.depth_scale_factor), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        )
        pose_estimate, _ = self.tracker.track(frame_id * self.timestamp_step_ns, images=[rgb_image], depths=[depth_image])
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


def get_slam_backbone(config: Dict[str, Any], dataset, cam_intrinsics: torch.Tensor):
    backbone = config["slam"].get("slam_module", "vanilla")
    if backbone.startswith("orbslam"):
        return WrapperORBSLAM(config, cam_intrinsics, world_ref=torch.from_numpy(dataset[0][3]))
    if backbone == "cuvslam":
        return WrapperCuVSLAM(config, cam_intrinsics, world_ref=torch.from_numpy(dataset[0][3]))
    if backbone == "vanilla":
        return VanillaMapper(config, cam_intrinsics)
    raise ValueError(f"Unsupported SLAM backend: {backbone}")
