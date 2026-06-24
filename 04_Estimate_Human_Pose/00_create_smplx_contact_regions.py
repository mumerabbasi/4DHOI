import argparse
import json
from pathlib import Path

import smplx
import torch
from smplx.joint_names import JOINT_NAMES

SCRIPT_DIR = Path(__file__).resolve().parent
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
        default=0.01,
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


if __name__ == "__main__":
    main()
