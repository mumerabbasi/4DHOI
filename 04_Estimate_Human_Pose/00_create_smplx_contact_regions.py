import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import smplx
import torch
from smplx.joint_names import JOINT_NAMES

SCRIPT_DIR = Path(__file__).resolve().parent
BLENDER_BIN = Path("/my_workspace/blender-4.2.17-linux-x64/blender")
DEFAULT_SMPLX_SEG_JSON = (
    SCRIPT_DIR.parents[1]
    / "GVHMR"
    / "hmr4d"
    / "utils"
    / "body_model"
    / "smplx_vert_segmentation.json"
)
BASE_COLOR = (170, 170, 170)
CONTACT_SEGMENT_ORDER = [
    "left_hand_contact",
    "right_hand_contact",
    "left_arm_contact",
    "right_arm_contact",
    "left_leg_contact",
    "right_leg_contact",
    "left_foot_contact",
    "right_foot_contact",
    "head_contact",
    "hips_contact",
    "back_contact",
]
VISUALIZATION_SEGMENT_ORDER = CONTACT_SEGMENT_ORDER
PROJECT_SEGMENT_SOURCES = {
    "left_hand": ("leftHand", "leftHandIndex1"),
    "right_hand": ("rightHand", "rightHandIndex1"),
    "left_arm": ("leftArm", "leftForeArm"),
    "right_arm": ("rightArm", "rightForeArm"),
    "left_shoulder": ("leftShoulder",),
    "right_shoulder": ("rightShoulder",),
    "left_leg": ("leftUpLeg", "leftLeg"),
    "right_leg": ("rightUpLeg", "rightLeg"),
    "left_foot": ("leftFoot", "leftToeBase"),
    "right_foot": ("rightFoot", "rightToeBase"),
    "head": ("head",),
    "hips": ("hips",),
    "back": ("spine", "spine1", "spine2"),
}
HIP_CONTACT_SOURCE_SEGMENTS = ("hips",)
BACK_CONTACT_SOURCE_SEGMENTS = ("spine", "spine1", "spine2")
HEAD_CONTACT_SOURCE_SEGMENTS = ("head",)
SEGMENT_COLORS = {
    "left_hand_contact": (230, 25, 75),
    "right_hand_contact": (0, 130, 200),
    "left_arm_contact": (245, 130, 48),
    "right_arm_contact": (70, 240, 240),
    "left_leg_contact": (145, 30, 180),
    "right_leg_contact": (0, 128, 128),
    "left_foot_contact": (255, 225, 25),
    "right_foot_contact": (60, 180, 75),
    "head_contact": (240, 50, 230),
    "hips_contact": (210, 245, 60),
    "back_contact": (128, 0, 0),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the SMPL-X segmentation asset used by downstream contact "
            "optimization, and export the colored canonical segmented mesh."
        )
    )
    parser.add_argument(
        "--smpl_folder",
        type=str,
        default="../../GVHMR/inputs/checkpoints/body_models/",
        help="Path to the SMPL-X body model folder.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="assets/smplx_vert_segmentation.json",
        help="Where to write the generated SMPL-X segmentation asset.",
    )
    parser.add_argument(
        "--output_ply",
        type=str,
        default="assets/smplx_vert_segmentation_canonical.ply",
        help="Where to write the colored canonical segmented SMPL-X mesh.",
    )
    parser.add_argument(
        "--skip_render",
        action="store_true",
        help="Only generate the segmentation JSON and canonical PLY; do not render.",
    )
    parser.add_argument(
        "--render_output_dir",
        type=str,
        default="assets",
        help="Directory for front/back/bottom render PNGs.",
    )
    parser.add_argument(
        "--render_output_prefix",
        type=str,
        default="smplx_contact_regions",
        help="Filename prefix for the rendered front/back/bottom PNGs.",
    )
    parser.add_argument(
        "--blender_bin",
        type=str,
        default=None,
        help="Path to the Blender executable used for rendering.",
    )
    parser.add_argument("--render_width", type=int, default=1400)
    parser.add_argument("--render_height", type=int, default=1400)
    parser.add_argument("--render_resolution_percentage", type=int, default=100)
    parser.add_argument("--render_cycles_samples", type=int, default=64)
    parser.add_argument(
        "--gpu_index",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for Blender.",
    )
    parser.add_argument(
        "--render_ortho_margin",
        type=float,
        default=1.10,
        help="Multiplier around the projected mesh bounds for each orthographic view.",
    )
    parser.add_argument(
        "--render_camera_distance_scale",
        type=float,
        default=3.0,
        help="Camera distance as a multiplier of the mesh diagonal.",
    )
    parser.add_argument(
        "--wrist_forward_cutoff",
        type=float,
        default=0.0,
        help="Minimum wrist-to-fingers forward coordinate kept in the hand region.",
    )
    parser.add_argument(
        "--hand_contact_normal_threshold",
        type=float,
        default=0.2,
        help="Minimum vertex-normal alignment with the palm direction for hand contact.",
    )
    parser.add_argument(
        "--foot_contact_normal_threshold",
        type=float,
        default=0.2,
        help=(
            "Minimum vertex-normal alignment with the downward direction "
            "for foot contact."
        ),
    )
    parser.add_argument(
        "--leg_upper_leg_start_fraction",
        type=float,
        default=0.35,
        help=(
            "Fraction of the hip-to-knee span where posterior leg contact "
            "begins. This keeps the leg patch focused below the hip contact "
            "support area."
        ),
    )
    parser.add_argument(
        "--leg_lower_leg_extension_fraction",
        type=float,
        default=0.80,
        help=(
            "Fraction of the knee-to-ankle span included in posterior leg "
            "contact. Values near 1 include the backside of the calf while "
            "leaving the ankle/foot to the foot contact region."
        ),
    )
    parser.add_argument(
        "--leg_posterior_z_max_m",
        type=float,
        default=-0.07,
        help=(
            "Maximum canonical z-coordinate kept for leg contact. More "
            "negative values keep a tighter posterior leg patch."
        ),
    )
    parser.add_argument(
        "--arm_posterior_z_max_m",
        type=float,
        default=-0.08,
        help=(
            "Maximum canonical z-coordinate kept for arm contact. More "
            "negative values keep a tighter posterior arm patch."
        ),
    )
    parser.add_argument(
        "--head_posterior_z_max_m",
        type=float,
        default=-0.08,
        help=(
            "Maximum canonical z-coordinate kept for head contact. More "
            "negative values keep a tighter back-of-head patch."
        ),
    )
    parser.add_argument(
        "--hip_pelvis_lower_offset_m",
        type=float,
        default=-0.18,
        help=(
            "Lowest y-coordinate kept for hip contact, expressed as a meter "
            "offset from the pelvis joint."
        ),
    )
    parser.add_argument(
        "--hip_pelvis_upper_offset_m",
        type=float,
        default=-0.04,
        help=(
            "Highest y-coordinate kept for hip contact, expressed as a meter "
            "offset from the pelvis joint."
        ),
    )
    parser.add_argument(
        "--hip_posterior_z_max_m",
        type=float,
        default=-0.08,
        help=(
            "Maximum canonical z-coordinate kept for hip contact. More negative "
            "values keep a tighter posterior glute/ischial patch."
        ),
    )
    parser.add_argument(
        "--back_posterior_z_max_m",
        type=float,
        default=-0.07,
        help=(
            "Maximum canonical z-coordinate kept for back contact. More "
            "negative values keep a tighter posterior torso patch."
        ),
    )
    return parser


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_smplx_model(smpl_folder: Path):
    return smplx.create(
        str(smpl_folder),
        model_type="smplx",
        gender="neutral",
        num_pca_comps=12,
        flat_hand_mean=False,
        create_body_pose=False,
        create_betas=False,
        create_global_orient=False,
        create_transl=False,
    )


def get_rest_pose_output(model):
    return model(
        betas=torch.zeros(1, 10),
        body_pose=torch.zeros(1, 63),
        global_orient=torch.zeros(1, 3),
        transl=torch.zeros(1, 3),
    )


def load_segmentation_json(seg_path: Path) -> dict:
    with seg_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def normalize_segment(indices) -> list[int]:
    return sorted({int(index) for index in indices})


def combine_segments(
    source_segments: dict[str, list[int]],
    segment_names: tuple[str, ...],
) -> list[int]:
    combined: list[int] = []
    for segment_name in segment_names:
        if segment_name not in source_segments:
            raise KeyError(
                f"Missing SMPL-X segment '{segment_name}' in the coarse "
                "segmentation JSON."
            )
        combined.extend(source_segments[segment_name])
    return normalize_segment(combined)


def compute_vertex_normals(vertices: torch.Tensor, faces) -> torch.Tensor:
    faces_t = torch.as_tensor(faces, dtype=torch.long)
    triangles = vertices[faces_t]
    face_normals = torch.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        dim=1,
    )
    face_normals = (
        face_normals
        / face_normals.norm(dim=1, keepdim=True).clamp_min(1e-8)
    )

    vertex_normals = torch.zeros_like(vertices)
    for corner_id in range(3):
        vertex_normals.index_add_(0, faces_t[:, corner_id], face_normals)
    vertex_normals = (
        vertex_normals
        / vertex_normals.norm(dim=1, keepdim=True).clamp_min(1e-8)
    )
    return vertex_normals


def compute_palm_normal(rest_joints: torch.Tensor, side: str) -> torch.Tensor:
    wrist = rest_joints[JOINT_NAMES.index(f"{side}_wrist")]
    index1 = rest_joints[JOINT_NAMES.index(f"{side}_index1")]
    middle1 = rest_joints[JOINT_NAMES.index(f"{side}_middle1")]
    pinky1 = rest_joints[JOINT_NAMES.index(f"{side}_pinky1")]

    across = index1 - pinky1
    forward = middle1 - wrist
    palm_normal = torch.cross(across, forward, dim=0)
    palm_normal = palm_normal / palm_normal.norm().clamp_min(1e-8)

    up = torch.tensor([0.0, 1.0, 0.0], dtype=rest_joints.dtype)
    if torch.dot(palm_normal, up) > 0:
        palm_normal = -palm_normal
    return palm_normal


def compute_hand_forward(rest_joints: torch.Tensor, side: str) -> torch.Tensor:
    wrist = rest_joints[JOINT_NAMES.index(f"{side}_wrist")]
    middle1 = rest_joints[JOINT_NAMES.index(f"{side}_middle1")]
    hand_forward = middle1 - wrist
    return hand_forward / hand_forward.norm().clamp_min(1e-8)


def build_hand_contact_segment(
    rest_vertices: torch.Tensor,
    rest_joints: torch.Tensor,
    vertex_normals: torch.Tensor,
    full_ids: list[int],
    side: str,
    wrist_forward_cutoff: float,
    hand_contact_normal_threshold: float,
) -> list[int]:
    full_ids_t = torch.as_tensor(full_ids, dtype=torch.long)
    wrist = rest_joints[JOINT_NAMES.index(f"{side}_wrist")]
    hand_forward = compute_hand_forward(rest_joints, side)
    palm_normal = compute_palm_normal(rest_joints, side)

    hand_forward_scores = (rest_vertices[full_ids_t] - wrist) @ hand_forward
    trimmed_ids = full_ids_t[hand_forward_scores >= wrist_forward_cutoff]
    if len(trimmed_ids) == 0:
        raise RuntimeError(f"The wrist cutoff removed all hand vertices for {side}.")

    contact_scores = vertex_normals[trimmed_ids] @ palm_normal
    contact_ids = trimmed_ids[contact_scores >= hand_contact_normal_threshold]
    contact_list = normalize_segment(contact_ids.tolist())
    if not contact_list:
        raise RuntimeError(f"The hand contact region for {side} is empty.")
    return contact_list


def build_foot_contact_segment(
    rest_vertices: torch.Tensor,
    vertex_normals: torch.Tensor,
    full_ids: list[int],
    side: str,
    foot_contact_normal_threshold: float,
) -> list[int]:
    full_ids_t = torch.as_tensor(full_ids, dtype=torch.long)
    down = torch.tensor([0.0, -1.0, 0.0], dtype=rest_vertices.dtype)
    contact_scores = vertex_normals[full_ids_t] @ down
    contact_ids = full_ids_t[contact_scores >= foot_contact_normal_threshold]
    contact_list = normalize_segment(contact_ids.tolist())
    if not contact_list:
        raise RuntimeError(f"The foot contact region for {side} is empty.")
    return contact_list


def build_leg_contact_segment(
    rest_vertices: torch.Tensor,
    rest_joints: torch.Tensor,
    full_ids: list[int],
    side: str,
    leg_upper_leg_start_fraction: float,
    leg_lower_leg_extension_fraction: float,
    leg_posterior_z_max_m: float,
) -> list[int]:
    if not 0.0 <= leg_upper_leg_start_fraction <= 1.0:
        raise ValueError("--leg_upper_leg_start_fraction must be between 0 and 1.")
    if not 0.0 <= leg_lower_leg_extension_fraction <= 1.0:
        raise ValueError(
            "--leg_lower_leg_extension_fraction must be between 0 and 1."
        )

    full_ids_t = torch.as_tensor(full_ids, dtype=torch.long)
    hip_y = rest_joints[JOINT_NAMES.index(f"{side}_hip")][1]
    knee_y = rest_joints[JOINT_NAMES.index(f"{side}_knee")][1]
    ankle_y = rest_joints[JOINT_NAMES.index(f"{side}_ankle")][1]

    upper_y = float(
        hip_y + float(leg_upper_leg_start_fraction) * (knee_y - hip_y)
    )
    lower_y = float(
        knee_y + float(leg_lower_leg_extension_fraction) * (ankle_y - knee_y)
    )
    if lower_y >= upper_y:
        raise ValueError(
            "The leg contact vertical bounds are inverted; adjust "
            "--leg_upper_leg_start_fraction or "
            "--leg_lower_leg_extension_fraction."
        )

    z_max = float(leg_posterior_z_max_m)

    points = rest_vertices[full_ids_t]
    height_mask = (points[:, 1] >= lower_y) & (points[:, 1] <= upper_y)
    posterior_depth_mask = points[:, 2] <= z_max
    contact_ids = full_ids_t[height_mask & posterior_depth_mask]

    contact_list = normalize_segment(contact_ids.tolist())
    if not contact_list:
        raise RuntimeError(f"The leg contact region for {side} is empty.")
    return contact_list


def build_posterior_depth_segment(
    rest_vertices: torch.Tensor,
    full_ids: list[int],
    posterior_z_max_m: float,
    segment_name: str,
) -> list[int]:
    full_ids_t = torch.as_tensor(full_ids, dtype=torch.long)
    z_max = float(posterior_z_max_m)

    points = rest_vertices[full_ids_t]
    posterior_depth_mask = points[:, 2] <= z_max
    contact_ids = full_ids_t[posterior_depth_mask]

    contact_list = normalize_segment(contact_ids.tolist())
    if not contact_list:
        raise RuntimeError(f"The {segment_name} region is empty.")
    return contact_list


def build_hip_contact_segment(
    rest_vertices: torch.Tensor,
    rest_joints: torch.Tensor,
    full_ids: list[int],
    hip_pelvis_lower_offset_m: float,
    hip_pelvis_upper_offset_m: float,
    hip_posterior_z_max_m: float,
) -> list[int]:
    full_ids_t = torch.as_tensor(full_ids, dtype=torch.long)
    pelvis_y = rest_joints[JOINT_NAMES.index("pelvis")][1]

    lower_y = float(pelvis_y + float(hip_pelvis_lower_offset_m))
    upper_y = float(pelvis_y + float(hip_pelvis_upper_offset_m))
    if lower_y >= upper_y:
        raise ValueError(
            "The hip contact vertical bounds are inverted; adjust "
            "--hip_pelvis_lower_offset_m or --hip_pelvis_upper_offset_m."
        )

    points = rest_vertices[full_ids_t]
    height_mask = (points[:, 1] >= lower_y) & (points[:, 1] <= upper_y)
    z_max = float(hip_posterior_z_max_m)
    posterior_depth_mask = points[:, 2] <= z_max
    contact_ids = full_ids_t[height_mask & posterior_depth_mask]

    contact_list = normalize_segment(contact_ids.tolist())
    if not contact_list:
        raise RuntimeError("The hip contact region is empty.")
    return contact_list


def build_back_contact_segment(
    rest_vertices: torch.Tensor,
    full_ids: list[int],
    back_posterior_z_max_m: float,
) -> list[int]:
    return build_posterior_depth_segment(
        rest_vertices=rest_vertices,
        full_ids=full_ids,
        posterior_z_max_m=back_posterior_z_max_m,
        segment_name="back contact",
    )


def build_payload(
    model,
    source_segmentation: dict,
    wrist_forward_cutoff: float,
    hand_contact_normal_threshold: float,
    foot_contact_normal_threshold: float,
    leg_upper_leg_start_fraction: float,
    leg_lower_leg_extension_fraction: float,
    leg_posterior_z_max_m: float,
    arm_posterior_z_max_m: float,
    head_posterior_z_max_m: float,
    hip_pelvis_lower_offset_m: float,
    hip_pelvis_upper_offset_m: float,
    hip_posterior_z_max_m: float,
    back_posterior_z_max_m: float,
) -> tuple[dict, torch.Tensor]:
    rest_output = get_rest_pose_output(model)
    rest_vertices = rest_output.vertices[0].detach().cpu()
    rest_joints = rest_output.joints[0].detach().cpu()
    vertex_normals = compute_vertex_normals(rest_vertices, model.faces)

    source_segments = {
        segment_name: normalize_segment(indices)
        for segment_name, indices in source_segmentation.items()
    }
    project_segments = {
        segment_name: combine_segments(source_segments, source_names)
        for segment_name, source_names in PROJECT_SEGMENT_SOURCES.items()
    }
    contact_segments = {}
    contact_segments["left_hand_contact"] = build_hand_contact_segment(
        rest_vertices=rest_vertices,
        rest_joints=rest_joints,
        vertex_normals=vertex_normals,
        full_ids=project_segments["left_hand"],
        side="left",
        wrist_forward_cutoff=wrist_forward_cutoff,
        hand_contact_normal_threshold=hand_contact_normal_threshold,
    )
    contact_segments["right_hand_contact"] = build_hand_contact_segment(
        rest_vertices=rest_vertices,
        rest_joints=rest_joints,
        vertex_normals=vertex_normals,
        full_ids=project_segments["right_hand"],
        side="right",
        wrist_forward_cutoff=wrist_forward_cutoff,
        hand_contact_normal_threshold=hand_contact_normal_threshold,
    )
    contact_segments["left_arm_contact"] = build_posterior_depth_segment(
        rest_vertices=rest_vertices,
        full_ids=project_segments["left_arm"],
        posterior_z_max_m=arm_posterior_z_max_m,
        segment_name="left arm contact",
    )
    contact_segments["right_arm_contact"] = build_posterior_depth_segment(
        rest_vertices=rest_vertices,
        full_ids=project_segments["right_arm"],
        posterior_z_max_m=arm_posterior_z_max_m,
        segment_name="right arm contact",
    )
    contact_segments["left_leg_contact"] = build_leg_contact_segment(
        rest_vertices=rest_vertices,
        rest_joints=rest_joints,
        full_ids=project_segments["left_leg"],
        side="left",
        leg_upper_leg_start_fraction=leg_upper_leg_start_fraction,
        leg_lower_leg_extension_fraction=leg_lower_leg_extension_fraction,
        leg_posterior_z_max_m=leg_posterior_z_max_m,
    )
    contact_segments["right_leg_contact"] = build_leg_contact_segment(
        rest_vertices=rest_vertices,
        rest_joints=rest_joints,
        full_ids=project_segments["right_leg"],
        side="right",
        leg_upper_leg_start_fraction=leg_upper_leg_start_fraction,
        leg_lower_leg_extension_fraction=leg_lower_leg_extension_fraction,
        leg_posterior_z_max_m=leg_posterior_z_max_m,
    )
    contact_segments["left_foot_contact"] = build_foot_contact_segment(
        rest_vertices=rest_vertices,
        vertex_normals=vertex_normals,
        full_ids=project_segments["left_foot"],
        side="left",
        foot_contact_normal_threshold=foot_contact_normal_threshold,
    )
    contact_segments["right_foot_contact"] = build_foot_contact_segment(
        rest_vertices=rest_vertices,
        vertex_normals=vertex_normals,
        full_ids=project_segments["right_foot"],
        side="right",
        foot_contact_normal_threshold=foot_contact_normal_threshold,
    )
    contact_segments["head_contact"] = build_posterior_depth_segment(
        rest_vertices=rest_vertices,
        full_ids=combine_segments(source_segments, HEAD_CONTACT_SOURCE_SEGMENTS),
        posterior_z_max_m=head_posterior_z_max_m,
        segment_name="head contact",
    )
    contact_segments["hips_contact"] = build_hip_contact_segment(
        rest_vertices=rest_vertices,
        rest_joints=rest_joints,
        full_ids=combine_segments(source_segments, HIP_CONTACT_SOURCE_SEGMENTS),
        hip_pelvis_lower_offset_m=hip_pelvis_lower_offset_m,
        hip_pelvis_upper_offset_m=hip_pelvis_upper_offset_m,
        hip_posterior_z_max_m=hip_posterior_z_max_m,
    )
    contact_segments["back_contact"] = build_back_contact_segment(
        rest_vertices=rest_vertices,
        full_ids=combine_segments(source_segments, BACK_CONTACT_SOURCE_SEGMENTS),
        back_posterior_z_max_m=back_posterior_z_max_m,
    )

    payload = {
        "mesh_type": "smplx",
        "vertex_count": int(rest_vertices.shape[0]),
        "face_count": int(len(model.faces)),
        "contact_segment_ids": CONTACT_SEGMENT_ORDER,
        "segments": contact_segments,
    }
    validate_payload(payload)
    return payload, rest_vertices


def validate_index_list(indices: list[int], vertex_count: int, name: str) -> None:
    if not indices:
        raise RuntimeError(f"{name} is empty.")
    if indices != sorted(indices):
        raise RuntimeError(f"{name} is not sorted.")
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"{name} contains duplicate vertex ids.")
    if indices[0] < 0 or indices[-1] >= vertex_count:
        raise RuntimeError(f"{name} contains out-of-range vertex ids.")


def validate_payload(payload: dict) -> None:
    vertex_count = int(payload["vertex_count"])
    segments = payload["segments"]
    for segment_id, indices in segments.items():
        validate_index_list(indices, vertex_count, segment_id)

    for segment_id in CONTACT_SEGMENT_ORDER:
        if segment_id not in segments:
            raise RuntimeError(f"Missing contact segment '{segment_id}'.")


def write_payload(output_json: Path, payload: dict) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_visualization_colors(payload: dict) -> torch.Tensor:
    vertex_count = int(payload["vertex_count"])
    colors = torch.full((vertex_count, 3), 0, dtype=torch.uint8)
    colors[:] = torch.tensor(BASE_COLOR, dtype=torch.uint8)

    for segment_name in VISUALIZATION_SEGMENT_ORDER:
        color = torch.tensor(SEGMENT_COLORS[segment_name], dtype=torch.uint8)
        for index in payload["segments"][segment_name]:
            colors[index] = color
    return colors


def write_ascii_ply_with_vertex_colors(
    path: Path,
    vertices: torch.Tensor,
    faces,
    colors: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("ply\n")
        file_obj.write("format ascii 1.0\n")
        file_obj.write(f"element vertex {len(vertices)}\n")
        file_obj.write("property float x\n")
        file_obj.write("property float y\n")
        file_obj.write("property float z\n")
        file_obj.write("property uchar red\n")
        file_obj.write("property uchar green\n")
        file_obj.write("property uchar blue\n")
        file_obj.write(f"element face {len(faces)}\n")
        file_obj.write("property list uchar int vertex_indices\n")
        file_obj.write("end_header\n")

        for vertex, color in zip(vertices.tolist(), colors.tolist()):
            file_obj.write(
                f"{vertex[0]} {vertex[1]} {vertex[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            file_obj.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def write_canonical_visualization(
    output_path: Path,
    rest_vertices: torch.Tensor,
    faces,
    payload: dict,
) -> None:
    colors = build_visualization_colors(payload)
    write_ascii_ply_with_vertex_colors(output_path, rest_vertices, faces, colors)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_blender_env(gpu_index: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if gpu_index is not None and str(gpu_index).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index).strip()
    return env


def write_blender_driver(path: Path) -> None:
    path.write_text(
        r'''
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def import_ply(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))
    after = set(bpy.context.scene.objects)
    new_objects = list(after - before)
    if not new_objects:
        raise RuntimeError(f"Failed to import PLY: {path}")
    return new_objects[0]


def assign_vertex_color_material(obj):
    mesh = obj.data
    mat = bpy.data.materials.new(name=f"{obj.name}_vertex_color")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    attr = nodes.new(type="ShaderNodeAttribute")
    if getattr(mesh, "color_attributes", None) and len(mesh.color_attributes) > 0:
        attr.attribute_name = mesh.color_attributes[0].name
    else:
        attr.attribute_name = "Col"
    mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.65
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def configure_cycles_gpu(samples):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = int(samples)
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.diffuse_bounces = 3
    bpy.context.scene.cycles.glossy_bounces = 3
    bpy.context.scene.cycles.transparent_max_bounces = 4

    prefs = bpy.context.preferences.addons["cycles"].preferences
    selected_backend = None
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpu_devices = [device for device in prefs.devices if device.type != "CPU"]
            if gpu_devices:
                selected_backend = backend
                break
        except Exception as exc:
            print(f"Cycles GPU backend {backend} unavailable: {exc}")

    if selected_backend is None:
        bpy.context.scene.cycles.device = "CPU"
        print("Cycles GPU device unavailable; falling back to CPU")
        return

    for device in prefs.devices:
        device.use = device.type != "CPU"
    enabled = [
        f"{device.name} ({device.type})"
        for device in prefs.devices
        if device.use
    ]
    print(f"Cycles GPU backend: {selected_backend}")
    print(f"Cycles GPU devices: {enabled}")


def object_bounds_corners_world(obj):
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def object_bounds_center_world(obj):
    corners = object_bounds_corners_world(obj)
    center = Vector((0.0, 0.0, 0.0))
    for corner in corners:
        center += corner
    return center / max(len(corners), 1)


def object_bounds_diagonal_world(obj):
    corners = object_bounds_corners_world(obj)
    if not corners:
        return 1.0
    mins = Vector(
        (
            min(c.x for c in corners),
            min(c.y for c in corners),
            min(c.z for c in corners),
        )
    )
    maxs = Vector(
        (
            max(c.x for c in corners),
            max(c.y for c in corners),
            max(c.z for c in corners),
        )
    )
    return max(float((maxs - mins).length), 1.0)


def aim_object_at(obj, target, up_axis="Y"):
    direction = Vector(target) - obj.location
    if direction.length < 1e-6:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", up_axis).to_euler()


def add_shadowless_area_light(name, location, target, energy, size):
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = float(energy)
    light_data.size = float(size)
    if hasattr(light_data, "use_shadow"):
        light_data.use_shadow = False
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = Vector(location)
    aim_object_at(light_obj, target)
    return light_obj


def configure_soft_room_lighting(human_obj):
    focus = object_bounds_center_world(human_obj)
    focus.z += 0.65

    world = bpy.context.scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.78, 0.80, 0.84, 1.0)
        background.inputs["Strength"].default_value = 0.25

    add_shadowless_area_light(
        "room_overhead_softbox",
        (focus.x, focus.y, focus.z + 2.4),
        focus,
        energy=100.0,
        size=4.0,
    )
    print("Lighting: low world fill + shadowless room area light")


def camera_matrix_world(location, target, desired_up):
    location = Vector(location)
    forward = (Vector(target) - location).normalized()
    z_axis = -forward
    x_axis = Vector(desired_up).cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0))
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()
    matrix = Matrix(
        (
            (x_axis.x, y_axis.x, z_axis.x, location.x),
            (x_axis.y, y_axis.y, z_axis.y, location.y),
            (x_axis.z, y_axis.z, z_axis.z, location.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return matrix


def create_camera_for_view(name, direction, target, distance, width, height, margin, obj):
    camera_data = bpy.data.cameras.new(name)
    camera_obj = bpy.data.objects.new(name, camera_data)
    bpy.context.collection.objects.link(camera_obj)

    direction = Vector(direction).normalized()
    location = Vector(target) + direction * float(distance)
    desired_up = (0.0, 1.0, 0.0) if abs(direction.z) > 0.95 else (0.0, 0.0, 1.0)
    camera_obj.matrix_world = camera_matrix_world(location, target, desired_up)

    camera_data.type = "ORTHO"
    camera_data.clip_start = 0.01
    camera_data.clip_end = max(100.0, float(distance) * 4.0)
    aspect = max(float(width) / max(float(height), 1.0), 1e-6)

    corners = object_bounds_corners_world(obj)
    local = [camera_obj.matrix_world.inverted() @ corner for corner in corners]
    x_extent = max(c.x for c in local) - min(c.x for c in local)
    y_extent = max(c.y for c in local) - min(c.y for c in local)
    camera_data.ortho_scale = max(y_extent, x_extent / aspect) * float(margin)
    return camera_obj


argv = sys.argv
config_path = Path(argv[argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

mesh_obj = import_ply(config["input_ply"])
mesh_obj.name = "smplx_contact_regions"
assign_vertex_color_material(mesh_obj)

# SMPL-X canonical vertices are Y-up. Rotate to Blender Z-up while preserving
# canonical front/back semantics: Blender front view sees canonical +Z.
mesh_obj.rotation_euler[0] = math.radians(90.0)
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

for polygon in mesh_obj.data.polygons:
    polygon.use_smooth = True

configure_cycles_gpu(config["cycles_samples"])
bpy.context.scene.world = bpy.data.worlds.new("world") if bpy.context.scene.world is None else bpy.context.scene.world
configure_soft_room_lighting(mesh_obj)

bpy.context.scene.render.resolution_percentage = int(config["resolution_percentage"])
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.view_settings.view_transform = "Filmic"
bpy.context.scene.view_settings.look = "Medium High Contrast"
bpy.context.scene.view_settings.exposure = 0.0
bpy.context.scene.view_settings.gamma = 1.0

target = object_bounds_center_world(mesh_obj)
distance = object_bounds_diagonal_world(mesh_obj) * float(config["camera_distance_scale"])
width = int(config["width"])
height = int(config["height"])
views = [
    ("front", (0.0, -1.0, 0.0)),
    ("back", (0.0, 1.0, 0.0)),
    ("bottom", (0.0, 0.0, -1.0)),
]

bpy.ops.wm.save_as_mainfile(filepath=config["blend_path"])
for name, direction in views:
    camera_obj = create_camera_for_view(
        name=name,
        direction=direction,
        target=target,
        distance=distance,
        width=width,
        height=height,
        margin=float(config["ortho_margin"]),
        obj=mesh_obj,
    )
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.filepath = config["render_paths"][name]
    bpy.ops.render.render(write_still=True)
'''.lstrip(),
        encoding="utf-8",
    )


def render_contact_region_views(output_ply: Path, args: argparse.Namespace) -> dict:
    render_output_dir = ensure_dir(resolve_path(args.render_output_dir, SCRIPT_DIR))
    render_assets_dir = ensure_dir(
        render_output_dir / f"{args.render_output_prefix}_render_assets"
    )

    if not output_ply.exists():
        raise FileNotFoundError(f"Segmented SMPL-X PLY not found: {output_ply}")

    blender_driver_path = render_assets_dir / "render_driver.py"
    config_path = render_assets_dir / "render_config.json"
    blend_path = render_assets_dir / "smplx_contact_regions.blend"
    write_blender_driver(blender_driver_path)

    render_paths = {
        "front": str(
            (render_output_dir / f"{args.render_output_prefix}_front.png").resolve()
        ),
        "back": str(
            (render_output_dir / f"{args.render_output_prefix}_back.png").resolve()
        ),
        "bottom": str(
            (render_output_dir / f"{args.render_output_prefix}_bottom.png").resolve()
        ),
    }
    config = {
        "input_ply": str(output_ply.resolve()),
        "blend_path": str(blend_path.resolve()),
        "render_paths": render_paths,
        "width": int(args.render_width),
        "height": int(args.render_height),
        "resolution_percentage": int(args.render_resolution_percentage),
        "cycles_samples": int(args.render_cycles_samples),
        "ortho_margin": float(args.render_ortho_margin),
        "camera_distance_scale": float(args.render_camera_distance_scale),
    }
    save_json(config_path, config)

    blender_bin = (
        BLENDER_BIN.resolve()
        if args.blender_bin is None
        else resolve_path(args.blender_bin, SCRIPT_DIR)
    )
    if not blender_bin.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_bin}")

    command = [
        str(blender_bin),
        "--background",
        "--python",
        str(blender_driver_path),
        "--",
        str(config_path),
    ]
    print(f"Rendering segmented SMPL-X contact regions from: {output_ply}")
    blender_env = build_blender_env(args.gpu_index)
    if "CUDA_VISIBLE_DEVICES" in blender_env:
        print(
            "Restricting Blender CUDA devices to: "
            f"{blender_env['CUDA_VISIBLE_DEVICES']}"
        )
    subprocess.run(command, check=True, env=blender_env)
    return {
        "render_paths": render_paths,
        "blend_path": str(blend_path),
        "config_path": str(config_path),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    smpl_folder = resolve_path(args.smpl_folder, SCRIPT_DIR)
    output_json = resolve_path(args.output_json, SCRIPT_DIR)
    output_ply = resolve_path(args.output_ply, SCRIPT_DIR)

    if not smpl_folder.exists():
        raise FileNotFoundError(f"SMPL folder not found: {smpl_folder}")
    if not DEFAULT_SMPLX_SEG_JSON.exists():
        raise FileNotFoundError(
            f"SMPL-X segmentation JSON not found: {DEFAULT_SMPLX_SEG_JSON}"
        )

    model = load_smplx_model(smpl_folder)
    source_segmentation = load_segmentation_json(DEFAULT_SMPLX_SEG_JSON)
    payload, rest_vertices = build_payload(
        model=model,
        source_segmentation=source_segmentation,
        wrist_forward_cutoff=float(args.wrist_forward_cutoff),
        hand_contact_normal_threshold=float(args.hand_contact_normal_threshold),
        foot_contact_normal_threshold=float(args.foot_contact_normal_threshold),
        leg_upper_leg_start_fraction=float(args.leg_upper_leg_start_fraction),
        leg_lower_leg_extension_fraction=float(
            args.leg_lower_leg_extension_fraction
        ),
        leg_posterior_z_max_m=float(args.leg_posterior_z_max_m),
        arm_posterior_z_max_m=float(args.arm_posterior_z_max_m),
        head_posterior_z_max_m=float(args.head_posterior_z_max_m),
        hip_pelvis_lower_offset_m=float(args.hip_pelvis_lower_offset_m),
        hip_pelvis_upper_offset_m=float(args.hip_pelvis_upper_offset_m),
        hip_posterior_z_max_m=float(args.hip_posterior_z_max_m),
        back_posterior_z_max_m=float(args.back_posterior_z_max_m),
    )
    write_payload(output_json, payload)

    write_canonical_visualization(
        output_path=output_ply,
        rest_vertices=rest_vertices,
        faces=model.faces,
        payload=payload,
    )

    print(f"Wrote SMPL-X segmentation to: {output_json}")
    print(f"Wrote canonical segmented mesh to: {output_ply}")
    if not args.skip_render:
        render_record = render_contact_region_views(output_ply, args)
        print("Wrote SMPL-X contact-region renders:")
        for view_name, render_path in render_record["render_paths"].items():
            print(f"  {view_name}: {render_path}")
        print(f"Wrote Blender scene to: {render_record['blend_path']}")


if __name__ == "__main__":
    main()
