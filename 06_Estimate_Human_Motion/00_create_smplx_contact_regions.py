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
PROJECT_BODY_PART_ORDER = [
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "left_shoulder",
    "right_shoulder",
    "left_leg",
    "right_leg",
    "left_foot",
    "right_foot",
    "head",
    "hips",
]
CONTACT_SEGMENT_ORDER = [
    "left_hand_inner",
    "right_hand_inner",
    "left_foot_bottom",
    "right_foot_bottom",
    "hips_contact",
]
VISUALIZATION_SEGMENT_ORDER = PROJECT_BODY_PART_ORDER + CONTACT_SEGMENT_ORDER
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
}
HIP_CONTACT_SOURCE_SEGMENTS = ("hips", "leftUpLeg", "rightUpLeg")
SEGMENT_COLORS = {
    "left_hand": (255, 182, 193),
    "right_hand": (173, 216, 230),
    "left_arm": (255, 160, 122),
    "right_arm": (95, 158, 160),
    "left_shoulder": (255, 215, 0),
    "right_shoulder": (144, 238, 144),
    "left_leg": (216, 191, 216),
    "right_leg": (176, 196, 222),
    "left_foot": (240, 230, 140),
    "right_foot": (189, 183, 107),
    "head": (255, 105, 180),
    "hips": (210, 180, 140),
    "left_hand_inner": (220, 20, 60),
    "right_hand_inner": (30, 144, 255),
    "left_foot_bottom": (255, 69, 0),
    "right_foot_bottom": (0, 191, 255),
    "hips_contact": (46, 139, 87),
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
        "--inner_normal_threshold",
        type=float,
        default=0.2,
        help="Minimum vertex-normal alignment with the palm direction for inner hands.",
    )
    parser.add_argument(
        "--foot_bottom_normal_threshold",
        type=float,
        default=0.2,
        help=(
            "Minimum vertex-normal alignment with the downward direction "
            "for foot bottoms."
        ),
    )
    parser.add_argument(
        "--hip_upper_pelvis_offset_m",
        type=float,
        default=-0.07,
        help=(
            "Highest y-coordinate kept for hip contact, expressed as a meter "
            "offset from the pelvis joint. Negative values move the cap below "
            "the pelvis."
        ),
    )
    parser.add_argument(
        "--hip_upper_leg_fraction",
        type=float,
        default=0.40,
        help=(
            "Fraction of the hip-to-knee span included for posterior upper-leg "
            "support in the hip contact patch."
        ),
    )
    parser.add_argument(
        "--hip_posterior_z_max_m",
        type=float,
        default=-0.10,
        help=(
            "Maximum canonical z-coordinate kept for hip contact. More negative "
            "values keep a tighter posterior glute/ischial patch."
        ),
    )
    parser.add_argument(
        "--hip_posterior_z_min_m",
        type=float,
        default=-0.14,
        help=(
            "Minimum canonical z-coordinate kept for hip contact. This removes "
            "the deepest back-side vertices to reduce side-view depth variance."
        ),
    )
    parser.add_argument(
        "--visualize_interaction",
        type=str,
        default=None,
        help=(
            "Optional interaction_xx folder to visualize directly from "
            "module-06 GVHMR hmr4d_results.pt."
        ),
    )
    parser.add_argument(
        "--human_motion_dir",
        type=str,
        default="output",
        help="06_Estimate_Human_Motion output root containing interaction folders.",
    )
    parser.add_argument(
        "--visualize_person",
        type=str,
        default="person_1",
        help="Human slug to visualize inside the selected interaction.",
    )
    parser.add_argument(
        "--visualize_frame",
        type=int,
        default=0,
        help="Frame index to visualize from hmr4d_results.pt.",
    )
    parser.add_argument(
        "--visualize_output_dir",
        type=str,
        default="assets/contact_region_visualizations",
        help=(
            "Where colored interaction visualization PLYs should be written. "
            "Defaults inside the module-06 assets directory."
        ),
    )
    parser.add_argument(
        "--visualize_smpl_param_space",
        choices=("incam", "global"),
        default="incam",
        help="Which GVHMR SMPL-X parameter group to visualize.",
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


def load_torch_payload(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


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
    face_normals = face_normals / face_normals.norm(dim=1, keepdim=True).clamp_min(1e-8)

    vertex_normals = torch.zeros_like(vertices)
    for corner_id in range(3):
        vertex_normals.index_add_(0, faces_t[:, corner_id], face_normals)
    vertex_normals = vertex_normals / vertex_normals.norm(
        dim=1, keepdim=True
    ).clamp_min(1e-8)
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


def build_inner_hand_segment(
    rest_vertices: torch.Tensor,
    rest_joints: torch.Tensor,
    vertex_normals: torch.Tensor,
    full_ids: list[int],
    side: str,
    wrist_forward_cutoff: float,
    inner_normal_threshold: float,
) -> list[int]:
    full_ids_t = torch.as_tensor(full_ids, dtype=torch.long)
    wrist = rest_joints[JOINT_NAMES.index(f"{side}_wrist")]
    hand_forward = compute_hand_forward(rest_joints, side)
    palm_normal = compute_palm_normal(rest_joints, side)

    hand_forward_scores = (rest_vertices[full_ids_t] - wrist) @ hand_forward
    trimmed_ids = full_ids_t[hand_forward_scores >= wrist_forward_cutoff]
    if len(trimmed_ids) == 0:
        raise RuntimeError(f"The wrist cutoff removed all hand vertices for {side}.")

    inner_scores = vertex_normals[trimmed_ids] @ palm_normal
    inner_ids = trimmed_ids[inner_scores >= inner_normal_threshold]
    inner_list = normalize_segment(inner_ids.tolist())
    if not inner_list:
        raise RuntimeError(f"The inner-hand region for {side} is empty.")
    return inner_list


def build_foot_bottom_segment(
    rest_vertices: torch.Tensor,
    vertex_normals: torch.Tensor,
    full_ids: list[int],
    side: str,
    foot_bottom_normal_threshold: float,
) -> list[int]:
    full_ids_t = torch.as_tensor(full_ids, dtype=torch.long)
    down = torch.tensor([0.0, -1.0, 0.0], dtype=rest_vertices.dtype)
    bottom_scores = vertex_normals[full_ids_t] @ down
    bottom_ids = full_ids_t[bottom_scores >= foot_bottom_normal_threshold]
    bottom_list = normalize_segment(bottom_ids.tolist())
    if not bottom_list:
        raise RuntimeError(f"The foot-bottom region for {side} is empty.")
    return bottom_list


def build_hip_contact_segment(
    rest_vertices: torch.Tensor,
    rest_joints: torch.Tensor,
    full_ids: list[int],
    hip_upper_pelvis_offset_m: float,
    hip_upper_leg_fraction: float,
    hip_posterior_z_max_m: float,
    hip_posterior_z_min_m: float,
) -> list[int]:
    if not 0.0 <= hip_upper_leg_fraction <= 1.0:
        raise ValueError("--hip_upper_leg_fraction must be between 0 and 1.")

    full_ids_t = torch.as_tensor(full_ids, dtype=torch.long)
    pelvis_y = rest_joints[JOINT_NAMES.index("pelvis")][1]
    hip_y = 0.5 * (
        rest_joints[JOINT_NAMES.index("left_hip")][1]
        + rest_joints[JOINT_NAMES.index("right_hip")][1]
    )
    knee_y = 0.5 * (
        rest_joints[JOINT_NAMES.index("left_knee")][1]
        + rest_joints[JOINT_NAMES.index("right_knee")][1]
    )

    upper_y = float(pelvis_y + float(hip_upper_pelvis_offset_m))
    lower_y = float(hip_y + float(hip_upper_leg_fraction) * (knee_y - hip_y))
    if lower_y >= upper_y:
        raise ValueError(
            "The hip contact vertical bounds are inverted; adjust "
            "--hip_upper_pelvis_offset_m or --hip_upper_leg_fraction."
        )

    points = rest_vertices[full_ids_t]
    height_mask = (points[:, 1] >= lower_y) & (points[:, 1] <= upper_y)
    z_min = float(hip_posterior_z_min_m)
    z_max = float(hip_posterior_z_max_m)
    if z_min >= z_max:
        raise ValueError(
            "--hip_posterior_z_min_m must be smaller than --hip_posterior_z_max_m."
        )
    posterior_depth_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    contact_ids = full_ids_t[height_mask & posterior_depth_mask]

    contact_list = normalize_segment(contact_ids.tolist())
    if not contact_list:
        raise RuntimeError("The hip contact region is empty.")
    return contact_list


def build_payload(
    model,
    source_segmentation: dict,
    wrist_forward_cutoff: float,
    inner_normal_threshold: float,
    foot_bottom_normal_threshold: float,
    hip_upper_pelvis_offset_m: float,
    hip_upper_leg_fraction: float,
    hip_posterior_z_max_m: float,
    hip_posterior_z_min_m: float,
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
    project_segments["left_hand_inner"] = build_inner_hand_segment(
        rest_vertices=rest_vertices,
        rest_joints=rest_joints,
        vertex_normals=vertex_normals,
        full_ids=project_segments["left_hand"],
        side="left",
        wrist_forward_cutoff=wrist_forward_cutoff,
        inner_normal_threshold=inner_normal_threshold,
    )
    project_segments["right_hand_inner"] = build_inner_hand_segment(
        rest_vertices=rest_vertices,
        rest_joints=rest_joints,
        vertex_normals=vertex_normals,
        full_ids=project_segments["right_hand"],
        side="right",
        wrist_forward_cutoff=wrist_forward_cutoff,
        inner_normal_threshold=inner_normal_threshold,
    )
    project_segments["left_foot_bottom"] = build_foot_bottom_segment(
        rest_vertices=rest_vertices,
        vertex_normals=vertex_normals,
        full_ids=project_segments["left_foot"],
        side="left",
        foot_bottom_normal_threshold=foot_bottom_normal_threshold,
    )
    project_segments["right_foot_bottom"] = build_foot_bottom_segment(
        rest_vertices=rest_vertices,
        vertex_normals=vertex_normals,
        full_ids=project_segments["right_foot"],
        side="right",
        foot_bottom_normal_threshold=foot_bottom_normal_threshold,
    )
    project_segments["hips_contact"] = build_hip_contact_segment(
        rest_vertices=rest_vertices,
        rest_joints=rest_joints,
        full_ids=combine_segments(source_segments, HIP_CONTACT_SOURCE_SEGMENTS),
        hip_upper_pelvis_offset_m=hip_upper_pelvis_offset_m,
        hip_upper_leg_fraction=hip_upper_leg_fraction,
        hip_posterior_z_max_m=hip_posterior_z_max_m,
        hip_posterior_z_min_m=hip_posterior_z_min_m,
    )

    payload = {
        "mesh_type": "smplx",
        "vertex_count": int(rest_vertices.shape[0]),
        "face_count": int(len(model.faces)),
        "body_segment_ids": PROJECT_BODY_PART_ORDER,
        "contact_segment_ids": CONTACT_SEGMENT_ORDER,
        "segments": project_segments,
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

    for segment_id in PROJECT_BODY_PART_ORDER + CONTACT_SEGMENT_ORDER:
        if segment_id not in segments:
            raise RuntimeError(f"Missing project segment '{segment_id}'.")

    for side in ("left", "right"):
        full_set = set(segments[f"{side}_hand"])
        inner_set = set(segments[f"{side}_hand_inner"])
        if not inner_set < full_set:
            raise RuntimeError(
                f"{side}_hand_inner must be a strict subset of {side}_hand."
            )
        foot_set = set(segments[f"{side}_foot"])
        bottom_set = set(segments[f"{side}_foot_bottom"])
        if not bottom_set < foot_set:
            raise RuntimeError(
                f"{side}_foot_bottom must be a strict subset of {side}_foot."
            )

    hip_contact_set = set(segments["hips_contact"])
    hip_support_set = (
        set(segments["hips"]) | set(segments["left_leg"]) | set(segments["right_leg"])
    )
    if not hip_contact_set < hip_support_set:
        raise RuntimeError(
            "hips_contact must be a strict subset of hips plus leg segments."
        )
    for segment_id in ("hips", "left_leg", "right_leg"):
        if not hip_contact_set & set(segments[segment_id]):
            raise RuntimeError(f"hips_contact has no overlap with {segment_id}.")


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


def _frame_slice(params: dict, name: str, frame_index: int) -> torch.Tensor:
    value = params[name].detach().clone().float()
    if value.ndim == 1:
        return value.view(1, -1)
    if frame_index >= value.shape[0]:
        raise IndexError(
            f"Requested frame {frame_index}, but '{name}' has only "
            f"{value.shape[0]} frame(s)."
        )
    return value[frame_index: frame_index + 1]


def get_hmr4d_frame_vertices(
    result_path: Path,
    model,
    frame_index: int,
    smpl_param_space: str,
) -> torch.Tensor:
    data = load_torch_payload(result_path)
    params_key = f"smpl_params_{smpl_param_space}"
    if params_key not in data:
        raise KeyError(f"Missing '{params_key}' in {result_path}")
    params = data[params_key]

    body_pose = _frame_slice(params, "body_pose", frame_index)
    global_orient = _frame_slice(params, "global_orient", frame_index)
    transl = _frame_slice(params, "transl", frame_index)
    betas = _frame_slice(params, "betas", frame_index)

    num_pca_comps = int(getattr(model, "num_pca_comps", None) or 12)
    num_expression_coeffs = int(getattr(model, "num_expression_coeffs", None) or 10)
    zeros_3 = torch.zeros(1, 3, dtype=torch.float32)
    zeros_hand = torch.zeros(1, num_pca_comps, dtype=torch.float32)
    zeros_expr = torch.zeros(1, num_expression_coeffs, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            left_hand_pose=zeros_hand,
            right_hand_pose=zeros_hand,
            jaw_pose=zeros_3,
            leye_pose=zeros_3,
            reye_pose=zeros_3,
            expression=zeros_expr,
        )
    return output.vertices[0].detach().cpu()


def write_interaction_visualization(
    human_motion_dir: Path,
    output_dir: Path,
    interaction_name: str,
    person_name: str,
    frame_index: int,
    smpl_param_space: str,
    model,
    payload: dict,
) -> Path:
    result_path = (
        human_motion_dir
        / interaction_name
        / "humans"
        / person_name
        / "hmr4d_results.pt"
    )
    if not result_path.exists():
        raise FileNotFoundError(f"GVHMR result file not found: {result_path}")

    vertices = get_hmr4d_frame_vertices(
        result_path=result_path,
        model=model,
        frame_index=frame_index,
        smpl_param_space=smpl_param_space,
    )
    output_path = (
        output_dir
        / interaction_name
        / person_name
        / f"frame_{frame_index:04d}_smplx_contact_regions.ply"
    )
    write_ascii_ply_with_vertex_colors(
        path=output_path,
        vertices=vertices,
        faces=model.faces,
        colors=build_visualization_colors(payload),
    )
    return output_path


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
        inner_normal_threshold=float(args.inner_normal_threshold),
        foot_bottom_normal_threshold=float(args.foot_bottom_normal_threshold),
        hip_upper_pelvis_offset_m=float(args.hip_upper_pelvis_offset_m),
        hip_upper_leg_fraction=float(args.hip_upper_leg_fraction),
        hip_posterior_z_max_m=float(args.hip_posterior_z_max_m),
        hip_posterior_z_min_m=float(args.hip_posterior_z_min_m),
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

    if args.visualize_interaction:
        human_motion_dir = resolve_path(args.human_motion_dir, SCRIPT_DIR)
        visualize_output_dir = resolve_path(args.visualize_output_dir, SCRIPT_DIR)
        visualization_path = write_interaction_visualization(
            human_motion_dir=human_motion_dir,
            output_dir=visualize_output_dir,
            interaction_name=args.visualize_interaction,
            person_name=args.visualize_person,
            frame_index=int(args.visualize_frame),
            smpl_param_space=args.visualize_smpl_param_space,
            model=model,
            payload=payload,
        )
        print(
            f"Wrote interaction contact-region visualization to: {visualization_path}"
        )


if __name__ == "__main__":
    main()
