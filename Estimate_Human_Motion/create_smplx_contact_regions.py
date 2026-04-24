import argparse
import json
from pathlib import Path

import smplx
import torch
from smplx.joint_names import JOINT_NAMES

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SMPLX_SEG_JSON = (
    SCRIPT_DIR.parents[1] / "GVHMR" / "hmr4d" / "utils" / "body_model" / "smplx_vert_segmentation.json"
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
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the SMPL-X segmentation asset used by downstream contact "
            "optimization, and export colored canonical/world-frame visualizations."
        )
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="video_01",
        help="Video name used to find Estimate_Human_Motion/output/<video_name>/humans/.",
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
        "--visualization_dir",
        type=str,
        default="assets/visualizations/smplx_vert_segmentation",
        help="Directory where canonical and first-frame world visualizations are written.",
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
        help="Minimum vertex-normal alignment with the downward direction for foot bottoms.",
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


def combine_segments(source_segments: dict[str, list[int]], segment_names: tuple[str, ...]) -> list[int]:
    combined: list[int] = []
    for segment_name in segment_names:
        if segment_name not in source_segments:
            raise KeyError(f"Missing SMPL-X segment '{segment_name}' in the coarse segmentation JSON.")
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
    vertex_normals = vertex_normals / vertex_normals.norm(dim=1, keepdim=True).clamp_min(1e-8)
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


def build_payload(
    model,
    source_segmentation: dict,
    wrist_forward_cutoff: float,
    inner_normal_threshold: float,
    foot_bottom_normal_threshold: float,
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
            raise RuntimeError(f"{side}_hand_inner must be a strict subset of {side}_hand.")
        foot_set = set(segments[f"{side}_foot"])
        bottom_set = set(segments[f"{side}_foot_bottom"])
        if not bottom_set < foot_set:
            raise RuntimeError(f"{side}_foot_bottom must be a strict subset of {side}_foot.")


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


def read_ascii_ply_mesh(path: Path) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
    vertex_count = None
    face_count = None
    header_complete = False

    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("element face "):
                face_count = int(stripped.split()[-1])
            elif stripped == "end_header":
                header_complete = True
                break

        if not header_complete or vertex_count is None or face_count is None:
            raise RuntimeError(f"Could not parse the ASCII PLY header from {path}.")

        vertices = []
        for _ in range(vertex_count):
            tokens = file_obj.readline().strip().split()
            if len(tokens) < 3:
                raise RuntimeError(f"Malformed vertex row in {path}.")
            vertices.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])

        faces: list[tuple[int, int, int]] = []
        for _ in range(face_count):
            tokens = file_obj.readline().strip().split()
            if len(tokens) < 4 or int(tokens[0]) != 3:
                raise RuntimeError(f"Malformed triangular face row in {path}.")
            faces.append((int(tokens[1]), int(tokens[2]), int(tokens[3])))

    return torch.tensor(vertices, dtype=torch.float32), faces


def write_canonical_visualization(
    output_path: Path,
    rest_vertices: torch.Tensor,
    faces,
    payload: dict,
) -> None:
    colors = build_visualization_colors(payload)
    write_ascii_ply_with_vertex_colors(output_path, rest_vertices, faces, colors)


def discover_world_frame_paths(video_name: str) -> list[tuple[str, Path]]:
    humans_root = SCRIPT_DIR / "output" / video_name / "humans"
    if not humans_root.exists():
        raise FileNotFoundError(f"Human-motion output not found: {humans_root}")

    world_paths: list[tuple[str, Path]] = []
    for person_dir in sorted(humans_root.iterdir()):
        world_frame_path = person_dir / "meshes" / "world" / "frame_0000.ply"
        if person_dir.is_dir() and world_frame_path.exists():
            world_paths.append((person_dir.name, world_frame_path))

    if not world_paths:
        raise FileNotFoundError(
            f"Could not find any world-frame first-frame meshes under: {humans_root}"
        )
    return world_paths


def write_world_visualizations(
    visualization_dir: Path,
    video_name: str,
    payload: dict,
) -> list[Path]:
    colors = build_visualization_colors(payload)
    written_paths: list[Path] = []
    for person_name, world_frame_path in discover_world_frame_paths(video_name):
        vertices, faces = read_ascii_ply_mesh(world_frame_path)
        if len(vertices) != int(payload["vertex_count"]):
            raise RuntimeError(
                f"{world_frame_path} has {len(vertices)} vertices but the segmentation expects "
                f"{payload['vertex_count']}."
            )
        if len(faces) != int(payload["face_count"]):
            raise RuntimeError(
                f"{world_frame_path} has {len(faces)} faces but the segmentation expects "
                f"{payload['face_count']}."
            )

        output_path = visualization_dir / video_name / f"{person_name}_world_frame_0000.ply"
        write_ascii_ply_with_vertex_colors(output_path, vertices, faces, colors)
        written_paths.append(output_path)
    return written_paths


def main() -> None:
    args = build_arg_parser().parse_args()
    smpl_folder = resolve_path(args.smpl_folder, SCRIPT_DIR)
    output_json = resolve_path(args.output_json, SCRIPT_DIR)
    visualization_dir = resolve_path(args.visualization_dir, SCRIPT_DIR)

    if not smpl_folder.exists():
        raise FileNotFoundError(f"SMPL folder not found: {smpl_folder}")
    if not DEFAULT_SMPLX_SEG_JSON.exists():
        raise FileNotFoundError(f"SMPL-X segmentation JSON not found: {DEFAULT_SMPLX_SEG_JSON}")

    model = load_smplx_model(smpl_folder)
    source_segmentation = load_segmentation_json(DEFAULT_SMPLX_SEG_JSON)
    payload, rest_vertices = build_payload(
        model=model,
        source_segmentation=source_segmentation,
        wrist_forward_cutoff=float(args.wrist_forward_cutoff),
        inner_normal_threshold=float(args.inner_normal_threshold),
        foot_bottom_normal_threshold=float(args.foot_bottom_normal_threshold),
    )
    write_payload(output_json, payload)

    canonical_output_path = visualization_dir / "canonical.ply"
    write_canonical_visualization(
        output_path=canonical_output_path,
        rest_vertices=rest_vertices,
        faces=model.faces,
        payload=payload,
    )
    world_output_paths = write_world_visualizations(
        visualization_dir=visualization_dir,
        video_name=args.video_name,
        payload=payload,
    )

    print(f"Wrote SMPL-X segmentation to: {output_json}")
    print(f"Wrote canonical visualization to: {canonical_output_path}")
    for world_output_path in world_output_paths:
        print(f"Wrote world-frame visualization to: {world_output_path}")


if __name__ == "__main__":
    main()
