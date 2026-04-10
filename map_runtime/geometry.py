from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def project_3d_points(points_3d: torch.Tensor, intrinsics: torch.Tensor, w2c: torch.Tensor | None = None) -> torch.Tensor:
    if w2c is not None:
        points_3d = torch.einsum("mn,bn->bm", w2c, points_3d)
    points_3d = points_3d[..., :3] / points_3d[..., 3:]
    points_2d = torch.einsum("mn,bn->bm", intrinsics, points_3d)
    return (points_2d[:, :2] / points_2d[:, 2:]).round().int()


def match_3d_points_to_2d_pixels(
    depth: torch.Tensor,
    w2c: torch.Tensor,
    points_3d: torch.Tensor,
    intrinsics: torch.Tensor,
    th_dist: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = depth.shape
    device = points_3d.device
    if points_3d.shape[-1] == 3:
        points_3d = torch.cat((points_3d, torch.ones((points_3d.shape[0], 1), device=device, dtype=points_3d.dtype)), dim=1)

    local_points_3d = torch.einsum("mn,bn->bm", w2c, points_3d)
    points_2d = project_3d_points(local_points_3d, intrinsics)
    in_plane_mask = (
        (points_2d[:, 0] < width)
        & (points_2d[:, 1] < height)
        & (points_2d[:, 0] >= 0)
        & (points_2d[:, 1] >= 0)
    )
    in_plane_ids = torch.where(in_plane_mask)[0]
    in_points_2d = points_2d[in_plane_ids]
    forward_points_depth = local_points_3d[in_plane_ids, 2]
    dist_mask = (forward_points_depth - depth[in_points_2d[:, 1], in_points_2d[:, 0]]).abs() < th_dist
    dist_mask[depth[in_points_2d[:, 1], in_points_2d[:, 0]] == 0] = False
    matches = in_points_2d[dist_mask]
    mask = in_plane_ids[dist_mask]
    return mask, matches


def compute_camera_frustum_corners(depth_map: torch.Tensor, pose: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    height, width = depth_map.shape
    depth_map = depth_map[depth_map > 0]
    min_depth, max_depth = depth_map.min(), depth_map.max()
    corners = torch.tensor(
        [
            [0, 0, min_depth],
            [width, 0, min_depth],
            [0, height, min_depth],
            [width, height, min_depth],
            [0, 0, max_depth],
            [width, 0, max_depth],
            [0, height, max_depth],
            [width, height, max_depth],
        ],
        device=depth_map.device,
    )
    x = (corners[:, 0] - intrinsics[0, 2]) * corners[:, 2] / intrinsics[0, 0]
    y = (corners[:, 1] - intrinsics[1, 2]) * corners[:, 2] / intrinsics[1, 1]
    z = corners[:, 2]
    corners_3d = torch.vstack((x, y, z, torch.ones(x.shape[0], device=depth_map.device))).T
    corners_3d = torch.einsum("ij,mj->mi", pose, corners_3d)
    return corners_3d[:, :3]


def compute_camera_frustum_planes(frustum_corners: torch.Tensor) -> torch.Tensor:
    planes = torch.stack(
        [
            torch.linalg.cross(frustum_corners[2] - frustum_corners[0], frustum_corners[1] - frustum_corners[0]),
            torch.linalg.cross(frustum_corners[6] - frustum_corners[4], frustum_corners[5] - frustum_corners[4]),
            torch.linalg.cross(frustum_corners[4] - frustum_corners[0], frustum_corners[2] - frustum_corners[0]),
            torch.linalg.cross(frustum_corners[7] - frustum_corners[3], frustum_corners[1] - frustum_corners[3]),
            torch.linalg.cross(frustum_corners[5] - frustum_corners[1], frustum_corners[3] - frustum_corners[1]),
            torch.linalg.cross(frustum_corners[6] - frustum_corners[2], frustum_corners[0] - frustum_corners[2]),
        ]
    )
    d_vals = torch.stack([-torch.dot(plane, frustum_corners[i]) for i, plane in enumerate(planes)])
    return torch.cat([planes, d_vals[:, None]], dim=1).float()


def compute_frustum_aabb(frustum_corners: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.min(frustum_corners, axis=0).values, torch.max(frustum_corners, axis=0).values


def points_inside_aabb_mask(points: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
    return (
        (points[:, 0] >= min_corner[0])
        & (points[:, 0] <= max_corner[0])
        & (points[:, 1] >= min_corner[1])
        & (points[:, 1] <= max_corner[1])
        & (points[:, 2] >= min_corner[2])
        & (points[:, 2] <= max_corner[2])
    )


def points_inside_frustum_mask(points: torch.Tensor, frustum_planes: torch.Tensor) -> torch.Tensor:
    num_pts = points.shape[0]
    ones = torch.ones((num_pts, 1), device=points.device)
    plane_product = torch.cat([points, ones], axis=1) @ frustum_planes.T
    return torch.all(plane_product <= 0, axis=1)


def compute_frustum_point_ids(pts: torch.Tensor, frustum_corners: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    if pts.shape[0] == 0:
        return torch.tensor([], dtype=torch.int64, device=device)
    pts = pts.to(device)
    frustum_corners = frustum_corners.to(device)

    min_corner, max_corner = compute_frustum_aabb(frustum_corners)
    inside_aabb_mask = points_inside_aabb_mask(pts, min_corner, max_corner)
    candidate_ids = torch.where(inside_aabb_mask)[0]
    if candidate_ids.numel() == 0:
        return candidate_ids

    frustum_planes = compute_camera_frustum_planes(frustum_corners).to(device)
    inside_frustum_mask = points_inside_frustum_mask(pts[candidate_ids], frustum_planes)
    return candidate_ids[inside_frustum_mask]
