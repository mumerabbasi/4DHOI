from __future__ import annotations

import math
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import roma
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry import compose_T_sequence
from losses import (
    _compute_intersect_diagnostic,
    compute_all_losses,
    pcd_distance,
    rotation_smoothness_loss,
    simple_smoothness_loss,
    simple_static_loss,
)
from models import (
    HumanData,
    InteractionEdge,
    InteractionNode,
    ObjectData,
    PAGObjectState,
    SDFGrid,
)


def _zero_weight_args() -> Namespace:
    return Namespace(
        tracking_weight=0.0,
        object_cd2d_weight_start=0.0,
        object_cd2d_weight_end=0.0,
        object_part_cd2d_weight_start=0.0,
        object_part_cd2d_weight_end=0.0,
        object_smooth_trans_weight_start=0.0,
        object_smooth_trans_weight_end=0.0,
        object_smooth_rot_weight_start=0.0,
        object_smooth_rot_weight_end=0.0,
        object_scale_weight_start=0.0,
        object_scale_weight_end=0.0,
        intersect_weight_start=0.0,
        intersect_weight_end=0.0,
        nocontact_weight_start=0.0,
        nocontact_weight_end=0.0,
        contact_drift_weight_start=0.0,
        contact_drift_weight_end=0.0,
        optimize_object_scale=False,
        max_log_scale_delta=0.22,
    )


def _make_test_object() -> ObjectData:
    tracked_poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    tracked_poses_torch = torch.from_numpy(tracked_poses).float()
    return ObjectData(
        name="object",
        slug="object",
        state=PAGObjectState(
            name="object",
            slug="object",
            is_translational=True,
            is_rotational=False,
        ),
        template_verts=torch.tensor([[0.1, 0.0, 1.0]], dtype=torch.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        vertex_colors=None,
        faces_torch=torch.zeros((0, 3), dtype=torch.int64),
        tracked_poses=tracked_poses,
        tracked_poses_torch=tracked_poses_torch,
        tracked_rotvecs=torch.zeros((2, 3), dtype=torch.float32),
        tracked_trans=torch.zeros((2, 3), dtype=torch.float32),
        part_vert_ids={"handle": np.array([0], dtype=np.int64)},
        part_face_ids={},
        sampled_points=torch.tensor([[0.1, 0.0, 1.0]], dtype=torch.float32),
        part_sampled_points={
            "handle": torch.tensor([[0.1, 0.0, 1.0]], dtype=torch.float32)
        },
        mask_points_2d=None,
        part_mask_points_2d={},
        sdf_grid=None,
        color_bgr=(0, 255, 255),
    )


def test_simple_static_and_smoothness_losses() -> None:
    seq = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    assert torch.isclose(simple_static_loss(seq), torch.tensor(2.5))
    assert torch.isclose(simple_smoothness_loss(seq), torch.tensor(0.25))


def test_rotation_smoothness_is_zero_at_geodesic_midpoint() -> None:
    rotvecs = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 4], [0.0, 0.0, math.pi / 2]],
        dtype=torch.float32,
    )
    rotmats = roma.rotvec_to_rotmat(rotvecs)
    assert torch.isclose(
        rotation_smoothness_loss(rotmats),
        torch.tensor(0.0, dtype=torch.float32),
        atol=1e-6,
    )


def test_pcd_distance_matches_original_mean_and_min_reductions() -> None:
    p1 = torch.tensor([[[0.0, 0.0], [5.0, 0.0]]], dtype=torch.float32)
    p2 = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    mean_val = pcd_distance(p1, p2, reduction="mean")
    min_val = pcd_distance(p1, p2, reduction="min")
    assert torch.isclose(mean_val.squeeze(0), torch.tensor(8.5))
    assert torch.isclose(min_val.squeeze(0), torch.tensor(1.0))


def test_intersect_loss_matches_inside_only_average() -> None:
    sdf_values = np.zeros((2, 2, 2), dtype=np.float32)
    for ix, x in enumerate([0.0, 1.0]):
        for iy, y in enumerate([0.0, 1.0]):
            for iz, z in enumerate([0.0, 1.0]):
                sdf_values[ix, iy, iz] = x + y + z - 1.5
    sdf_grid = SDFGrid(
        sdf_volume=torch.from_numpy(sdf_values.reshape(1, 1, 2, 2, 2)),
        bbox_min=torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32),
        bbox_max=torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32),
    )
    world_points = torch.tensor(
        [[[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]],
        dtype=torch.float32,
    )
    obj_T = compose_T_sequence(
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
    )
    scalar, per_frame = _compute_intersect_diagnostic(
        world_points,
        SimpleNamespace(sdf_grid=sdf_grid),
        obj_T,
        torch.tensor(1.0),
    )
    assert torch.isclose(scalar, torch.tensor(0.25), atol=1e-5)
    assert torch.isclose(per_frame[0], torch.tensor(0.25), atol=1e-5)


def test_compute_all_losses_matches_original_nocontact_and_contact_drift() -> None:
    human_points = torch.tensor(
        [[[0.0, 0.0, 1.0]], [[1.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    human_data = HumanData(
        base_verts=human_points,
        faces=np.zeros((0, 3), dtype=np.int32),
        faces_torch=torch.zeros((0, 3), dtype=torch.int64),
        part_points={"head": human_points},
        part_vert_ids={"head": np.array([0], dtype=np.int64)},
    )
    object_data = _make_test_object()
    edge = InteractionEdge(
        node_a=InteractionNode(
            raw_node="person 1, head",
            entity_name="person 1",
            part_name="head",
            is_human=True,
            object_slug=None,
            resolved_part_name="head",
            vert_ids=np.array([0], dtype=np.int64),
        ),
        node_b=InteractionNode(
            raw_node="object, handle",
            entity_name="object",
            part_name="handle",
            is_human=False,
            object_slug="object",
            resolved_part_name="handle",
            vert_ids=np.array([0], dtype=np.int64),
        ),
        is_continuous=True,
        is_rel_static=True,
    )
    args = _zero_weight_args()
    result = compute_all_losses(
        delta_rotvecs={"object": torch.zeros((2, 3), dtype=torch.float32)},
        delta_trans={"object": torch.zeros((2, 3), dtype=torch.float32)},
        raw_scale_deltas={"object": torch.zeros((1,), dtype=torch.float32)},
        objects={"object": object_data},
        human_data=human_data,
        interaction_edges=[edge],
        obj_keys=["object"],
        args=args,
        iteration=0,
        total_iters=10,
        k=torch.eye(3, dtype=torch.float32),
        width=16,
        height=16,
    )
    assert torch.isclose(result.nocontact, torch.tensor(0.41), atol=1e-6)
    assert torch.isclose(result.contact_drift, torch.tensor(1.0), atol=1e-6)
