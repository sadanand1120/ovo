from typing import Any, Dict, List
import orbslam3 as orbslam
import torch
from pathlib import Path
import tempfile
import numpy as np

from .vanilla_mapper import VanillaMapper


SCANNET_ORBSLAM_OVERRIDES = {
    "default": {"Stereo.b": 0.069253, "ORBextractor.nFeatures": 8000, "ORBextractor.iniThFAST": 15},
    "scene0000_00": {"Stereo.b": 0.069000, "ORBextractor.iniThFAST": 20},
    "scene0002_00": {"Stereo.b": 0.070000, "ORBextractor.iniThFAST": 20},
    "scene0050_00": {"Stereo.b": 0.069220},
    "scene0231_00": {"Stereo.b": 0.069220},
    "scene0378_00": {"Stereo.b": 0.069621, "ORBextractor.nFeatures": 4000, "ORBextractor.iniThFAST": 20},
    "scene0518_00": {"Stereo.b": 0.069220, "ORBextractor.nFeatures": 4000, "ORBextractor.iniThFAST": 20},
}


def convert_pose(traj, device):
    pose = torch.cat([
        torch.tensor(traj[-12:], device = device).reshape((3,4)),
        torch.tensor([[0,0,0,1]], device = device)
    ])
    return pose


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
    if config["dataset_name"].lower() != "scannet":
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
    """This class uses ORB-SLAM 3 to estimate camera posses and generates a vanilla point-cloud reconstruction by unprojecting depths"""
    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor, world_ref=torch.eye(4)) -> None:
        super().__init__(config, cam_intrinsics)

        self.close_loops = config["slam"].get("close_loops", True)
        self.last_big_change_id = 0
        self.map_updated = False
        self.world_ref = world_ref.to(self.device)
        self.kfs = {}

        repo_root = Path(__file__).resolve().parents[2]
        vocab_path = repo_root / "thirdParty" / "ORB_SLAM3" / "Vocabulary" / "ORBvoc.txt"
        assert (vocab_path).exists(), f"ORB vocabulary not found, review path {vocab_path}"
        self.temp_dir = tempfile.TemporaryDirectory(prefix="ovo_orbslam_")
        orbslam_config_path = Path(self.temp_dir.name) / f"{config['data']['scene_name']}.yaml"
        orbslam_config_path.write_text(_build_orbslam_settings(config, cam_intrinsics))

        self.orbslam = orbslam.System(str(vocab_path), str(orbslam_config_path), orbslam.Sensor.RGBD, config["slam"].get("use_viewer",False), not self.close_loops)
        self.orbslam.initialize()

    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id, rgb_image, depth_image = frame_data[:3]
        tframe = frame_id
        self.orbslam.process_image_rgbd(rgb_image, depth_image, tframe) # This actually blocks untill tracking is completed
        tracking_state = self.orbslam.get_tracking_state()
        if tracking_state == orbslam.TrackingState.OK:
            orb_c2w = self.orbslam.get_last_trajectory_point()
            assert int(orb_c2w[0]) == frame_id, "Retrieved wrong frame pose" # This should never happen
            self.estimated_c2ws[frame_id] = self.world_ref@convert_pose(orb_c2w, device = self.device)
        else:
            print(f"Tracking state: {tracking_state}!")
        return 
    
    def map(self, frame_data, c2w) -> None:
        # check if frame is a KeyFrame
        if self.orbslam.is_last_frame_kf(): # If tracking and maping are parallelized, this call would have a racing condition with ORB-SLAM2 tracking thread
            frame_id = frame_data[0]
            first_p_idx = self.pcd_ids.shape[0]
            super().map(frame_data, c2w)
            last_p_idx = self.pcd_ids.shape[0]
            self.kfs[frame_id] = {"id": frame_id , "pcd_idxs":(first_p_idx, last_p_idx)} # Assumes pcd is not pruned outside of self._update_map

        # detect loop-closure of GBA
        last_big_change_id = self.orbslam.get_last_big_change_idx()
        # LC and GBA happen one after the other, we could save some computation detecting only GBA
        if self.close_loops and last_big_change_id != self.last_big_change_id:
            self.last_big_change_id = last_big_change_id
            self.update_map()

    def update_map(self):
        print("Updating dense map ...")
        # update kfs and pcd poses:
        updated_kfs = self.orbslam.get_keyframe_points()

        new_kfs = {}
        new_pcd = []
        new_pcd_ids = []
        new_pcd_obj_ids = []
        new_pcd_colors = []
        new_c2w = {}
        n_points = 0
        for updated_kf in updated_kfs: 
            # for each keyframe, retrieve our saved keyframe,
            kf_id = int(updated_kf[0])
            kf = self.kfs.get(kf_id)
            if kf is None:
                # Why would a kf not be in self.kfs? They are only deleted when update_map is called
                # but then they shouldn't be anymore in orb_slam kfs list
                # If a Keyframe is added/tracked after orb_slam starts LC/GBA, is it going to be in the retrieved list of kfs?
                continue 

            kf_c2w = self.estimated_c2ws[kf["id"]]
            updated_kf_c2w = self.world_ref@convert_pose(updated_kf[1:13], device = self.device)

            transform = updated_kf_c2w@torch.linalg.inv(kf_c2w)
            # If transform is the identityt matrix then the KF was not modified and this could be skipped.
            # ovo.update_map() could just go over updated KFs' 3D instances to check if they should be fused with other instances
            # Measure mIoU and number of fused instances to evaluate reduced approach.
            updated_kf_pcd = torch.einsum('mn,bn->bm', transform, torch.cat([self.pcd[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]], torch.ones((kf["pcd_idxs"][1]-kf["pcd_idxs"][0],1), device=self.device)], dim=1))[:,:3]

            # kfs that are not in updated_kfs were pruned by ORB_SLAM. They will be removed together with their associated pcd
            old_n_points = n_points
            n_points += len(updated_kf_pcd)
            new_kfs[kf_id] = {"id": kf['id'] , "pcd_idxs":(old_n_points, n_points)}
            new_pcd.append(updated_kf_pcd)
            new_pcd_ids.append(self.pcd_ids[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]])
            new_pcd_obj_ids.append(self.pcd_obj_ids[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]])
            new_pcd_colors.append(self.pcd_colors[kf["pcd_idxs"][0]:kf["pcd_idxs"][1]])
            new_c2w[kf["id"]] = updated_kf_c2w

        self.estimated_c2ws = new_c2w
        self.kfs = new_kfs
        self.pcd = torch.cat(new_pcd, dim=0)
        self.pcd_ids = torch.cat(new_pcd_ids, dim=0)
        self.pcd_obj_ids = torch.cat(new_pcd_obj_ids, dim=0)
        self.pcd_colors = torch.cat(new_pcd_colors, dim=0)
        self.map_updated = True

    

    def __del__(self) -> None:
        self.orbslam.shutdown()
        self.temp_dir.cleanup()
