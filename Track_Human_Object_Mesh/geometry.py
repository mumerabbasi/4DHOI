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


def compose_T(rotvec: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """rotvec [3], trans [3] -> T [4,4]."""
    R = axis_angle_to_matrix(rotvec.unsqueeze(0)).squeeze(0)
    Rt = torch.cat([R, trans.unsqueeze(1)], dim=1)
    bottom = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=rotvec.device,
    )
    return torch.cat([Rt, bottom], dim=0)


def apply_T_batch(verts: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """verts [V,3], T [4,4] -> transformed [V,3]."""
    R = T[:3, :3]
    t = T[:3, 3]
    return verts @ R.t() + t.unsqueeze(0)


def apply_similarity_batch(
    points: torch.Tensor,
    T: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return apply_T_batch(points * scale, T)


def apply_inverse_similarity_batch(
    points: torch.Tensor,
    T: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    R = T[:3, :3]
    t = T[:3, 3]
    return ((points - t.unsqueeze(0)) @ R) / scale


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


def apply_local_se3_sequence(
    points_seq: torch.Tensor,
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    if points_seq.numel() == 0:
        return points_seq
    R = axis_angle_to_matrix(rotvecs)
    centered = points_seq - centers[:, None, :]
    rotated = torch.matmul(centered, R.transpose(1, 2))
    return rotated + centers[:, None, :] + trans[:, None, :]


def project_points_normalized_torch(
    points: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = points[..., 2]
    valid = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    uv = torch.zeros(
        (*points.shape[:2], 2),
        dtype=points.dtype,
        device=points.device,
    )
    z_safe = z.clamp(min=1e-6)
    uv[..., 0] = (points[..., 0] * k[0, 0] / z_safe + k[0, 2]) / float(width)
    uv[..., 1] = (points[..., 1] * k[1, 1] / z_safe + k[1, 2]) / float(height)
    return uv, valid


def pack_projected_points(
    points_2d: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(valid.to(torch.int64), dim=1, descending=True)
    packed = torch.gather(
        points_2d,
        dim=1,
        index=order.unsqueeze(-1).expand(-1, -1, 2),
    )
    lengths = valid.sum(dim=1)
    return packed, lengths


def masked_mean_from_lengths(
    values: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    idx = torch.arange(values.shape[1], device=values.device)[None, :]
    mask = idx < lengths[:, None]
    denom = mask.sum().clamp(min=1)
    return (values * mask).sum() / denom


def masked_mean_per_lengths(
    values: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    idx = torch.arange(values.shape[1], device=values.device)[None, :]
    mask = idx < lengths[:, None]
    denom = mask.sum(dim=1).clamp(min=1).to(values.dtype)
    return (values * mask).sum(dim=1) / denom


def query_sdf(
    sdf_grid: SDFGrid,
    points: torch.Tensor,
) -> torch.Tensor:
    """Query SDF values for points. Returns [N] values (negative = inside)."""
    pts = points.unsqueeze(0)
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
    return sampled.reshape(-1)


def geodesic_distance_sq(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic distance between two rotation matrices."""
    R_rel = R1.t() @ R2
    cos_angle = (R_rel.trace() - 1.0) / 2.0
    cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return 1.0 - cos_angle


def bounded_log_scale_delta(
    raw_value: torch.Tensor,
    max_log_scale_delta: float,
) -> torch.Tensor:
    return math.fabs(max_log_scale_delta) * torch.tanh(raw_value.squeeze())
