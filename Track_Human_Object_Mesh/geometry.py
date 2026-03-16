"""Geometry helpers for joint human-object mesh refinement."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from models import SDFGrid


def decompose_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[4,4] -> (rotvec [3], trans [3])."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_torch = torch.from_numpy(R).unsqueeze(0).float()
    rotvec = matrix_to_axis_angle(R_torch).squeeze(0).numpy()
    return rotvec.astype(np.float32), t.astype(np.float32)


def compose_T_sequence(
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
) -> torch.Tensor:
    """rotvecs [F,3], trans [F,3] -> T [F,4,4]."""
    R = axis_angle_to_matrix(rotvecs)
    T = torch.zeros(
        (rotvecs.shape[0], 4, 4),
        dtype=rotvecs.dtype,
        device=rotvecs.device,
    )
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    T[:, 3, 3] = 1.0
    return T


def apply_similarity_sequence(
    points: torch.Tensor,
    T_seq: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scaled = points * scale
    R = T_seq[:, :3, :3]
    t = T_seq[:, :3, 3]
    return torch.matmul(scaled.unsqueeze(0), R.transpose(1, 2)) + t[:, None, :]


def apply_inverse_similarity_sequence(
    points_seq: torch.Tensor,
    T_seq: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    R = T_seq[:, :3, :3]
    t = T_seq[:, :3, 3]
    return torch.matmul(points_seq - t[:, None, :], R) / scale


def project_points_with_intrinsics(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    assert points.shape[-1] == 3
    assert intrinsics.shape == (3, 3)
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    px = points[..., 0] * fx / points[..., 2] + cx
    py = points[..., 1] * fy / points[..., 2] + cy
    d = points[..., 2]
    return torch.stack([px, py, d], dim=-1)


def query_sdf(
    sdf_grid: SDFGrid,
    points: torch.Tensor,
) -> torch.Tensor:
    """Query SDF values for points. Returns [...,] values (negative = inside)."""
    shape = points.shape[:-1]
    pts = points.reshape(1, -1, 3)
    normalised = (
        (pts - sdf_grid.bbox_min)
        / (sdf_grid.bbox_max - sdf_grid.bbox_min)
        * 2.0
        - 1.0
    )
    grid = normalised[:, :, [2, 1, 0]].view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        sdf_grid.sdf_volume,
        grid,
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(*shape)


def bounded_log_scale_delta(
    raw_value: torch.Tensor,
    max_log_scale_delta: float,
) -> torch.Tensor:
    return math.fabs(max_log_scale_delta) * torch.tanh(raw_value.squeeze())
