import argparse
import json
from pathlib import Path

import numpy as np
import smplx
import torch
from tqdm import tqdm

from incam_stabilization import compute_stabilized_incam_params


def write_ascii_ply(path: Path, vertices, faces) -> None:
    """Write a triangular mesh to an ASCII PLY file."""
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def build_default_paths(video_name: str) -> tuple[Path, Path, Path]:
    """Build default input/output roots for one video."""
    script_dir = Path(__file__).parent.resolve()
    human_motion_dir = script_dir / "output" / video_name / "humans"
    camera_json_path = (
        script_dir.parent / "Generate_Video" / "output" / video_name / "resized_camera.json"
    )
    return human_motion_dir, human_motion_dir, camera_json_path


def discover_human_result_dirs(humans_root: Path) -> list[Path]:
    """Find per-human GVHMR result directories under the new layout."""
    result_dirs: list[Path] = []
    for child in sorted(humans_root.iterdir()):
        if child.is_dir() and (child / "hmr4d_results.pt").exists():
            result_dirs.append(child)
    return result_dirs


def load_scene_camera_transform(
    camera_json_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    with camera_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    world_to_camera = np.asarray(payload["world_to_camera_4x4"], dtype=np.float32)
    if world_to_camera.shape != (4, 4):
        raise ValueError(
            "Expected a 4x4 world_to_camera_4x4 matrix in "
            f"{camera_json_path}, got {world_to_camera.shape}."
        )

    rotation_world_to_camera = world_to_camera[:3, :3].astype(np.float32)
    translation_world_to_camera = world_to_camera[:3, 3].astype(np.float32)
    return rotation_world_to_camera, translation_world_to_camera


def transform_camera_to_world(
    points_camera: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return (
        points_camera - translation_world_to_camera[None]
    ) @ rotation_world_to_camera


def export_camera_and_world_ply_sequence(
    params: dict,
    camera_output_dir: Path,
    world_output_dir: Path,
    smplx_layer,
    smplx2smpl,
    faces_smpl,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    label: str,
) -> int:
    body_pose = params["body_pose"]
    betas = params["betas"]
    global_orient = params["global_orient"]
    transl = params["transl"]

    num_frames = body_pose.shape[0]
    print(f"Found {num_frames} frames for {label}.")
    print(f"Input Pose Shape ({label}): {body_pose.shape}")

    camera_output_dir.mkdir(parents=True, exist_ok=True)
    world_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {label} frames to camera/world .ply...")
    for i in tqdm(range(num_frames), desc=label, leave=False):
        curr_betas = betas[i: i + 1] if betas.shape[0] > 1 else betas[:1]

        output = smplx_layer(
            betas=curr_betas,
            body_pose=body_pose[i: i + 1],
            global_orient=global_orient[i: i + 1],
            transl=transl[i: i + 1],
        )

        smplx_verts = output.vertices[0]
        smpl_verts = torch.matmul(smplx2smpl, smplx_verts)
        camera_vertices = smpl_verts.detach().cpu().numpy()
        world_vertices = transform_camera_to_world(
            camera_vertices,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )

        filename = f"frame_{i:04d}.ply"
        write_ascii_ply(camera_output_dir / filename, camera_vertices, faces_smpl)
        write_ascii_ply(world_output_dir / filename, world_vertices, faces_smpl)

    print(f"Saved {label} camera PLYs to: {camera_output_dir}")
    print(f"Saved {label} world PLYs to: {world_output_dir}")
    return num_frames


def export_one_human(
    result_dir: Path,
    output_dir: Path,
    smplx_layer,
    smplx2smpl,
    faces_smpl,
    gvhmr_path: Path,
    camera_json_path: Path,
    stabilize_incam: bool,
) -> None:
    result_path = result_dir / "hmr4d_results.pt"
    if not result_path.exists():
        raise FileNotFoundError(f"Could not find hmr4d_results.pt in: {result_dir}")

    print(f"Loading data from {result_path}...")
    data = torch.load(result_path)

    # Export in camera-frame coordinates so the human matches other OpenCV-camera
    # assets. By default we use a stabilized camera-frame trajectory derived from
    # GVHMR's post-processed global motion to reduce visible foot sliding.
    if stabilize_incam:
        camera_params = compute_stabilized_incam_params(data, gvhmr_path)
    else:
        camera_params = data.get("smpl_params_incam_raw", data["smpl_params_incam"])
    rotation_world_to_camera, translation_world_to_camera = load_scene_camera_transform(
        camera_json_path
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    camera_output_dir = output_dir / "camera"
    world_output_dir = output_dir / "world"

    export_camera_and_world_ply_sequence(
        params=camera_params,
        camera_output_dir=camera_output_dir,
        world_output_dir=world_output_dir,
        smplx_layer=smplx_layer,
        smplx2smpl=smplx2smpl,
        faces_smpl=faces_smpl,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        label=result_dir.name,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_name",
        type=str,
        default="video_01",
        help="Video name used to build default paths for the other arguments.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing per-human GVHMR results under humans/<person_name>/.",
    )
    parser.add_argument(
        "--smpl_folder",
        type=str,
        default="../../GVHMR/inputs/checkpoints/body_models/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save exported PLY files. If not provided, defaults to "
            "Estimate_Human_Motion/output/<video_name>/humans/<person_name>/meshes/"
            " with camera/ and world/ subdirectories."
        ),
    )
    parser.add_argument(
        "--smplx2smpl_path",
        type=str,
        default="../../GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        help="Path to the smplx2smpl sparse matrix from GVHMR.",
    )
    parser.add_argument(
        "--gvhmr_path",
        type=str,
        default=None,
        help="Path to the cloned GVHMR repo, used for stabilized camera-frame export.",
    )
    parser.add_argument(
        "--camera_json",
        type=str,
        default=None,
        help=(
            "Path to Generate_Video/output/<video_name>/resized_camera.json. "
            "Used to convert camera-space meshes into scene-world meshes."
        ),
    )
    parser.add_argument(
        "--stabilize_incam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Export a stabilized camera-frame motion derived from GVHMR's "
            "post-processed global trajectory. Use --no-stabilize_incam to export "
            "the raw GVHMR incam motion instead."
        ),
    )
    args = parser.parse_args()

    default_video_dir, default_output_dir, default_camera_json = build_default_paths(
        args.video_name
    )
    default_gvhmr_path = Path(__file__).resolve().parents[2] / "GVHMR"

    video_dir = Path(args.video_dir).resolve() if args.video_dir else default_video_dir
    smpl_folder = Path(args.smpl_folder).resolve()
    smplx2smpl_path = Path(args.smplx2smpl_path).resolve()
    gvhmr_path = Path(args.gvhmr_path).resolve() if args.gvhmr_path else default_gvhmr_path
    camera_json_path = (
        Path(args.camera_json).resolve() if args.camera_json else default_camera_json
    )

    if not video_dir.exists() or not video_dir.is_dir():
        raise NotADirectoryError(f"Video directory not found: {video_dir}")
    if not camera_json_path.exists():
        raise FileNotFoundError(f"Camera JSON not found: {camera_json_path}")

    if args.output_dir is None:
        output_dir = default_output_dir
    else:
        output_dir = Path(args.output_dir).resolve()

    # 3. Setup SMPL-X model (matching GVHMR's "supermotion" config)
    print(f"Loading SMPL-X from {smpl_folder}...")
    smplx_layer = smplx.create(
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

    # Load the SMPL-X to SMPL vertex conversion matrix (6890 x 10475)
    print(f"Loading smplx2smpl matrix from {smplx2smpl_path}...")
    smplx2smpl = torch.load(smplx2smpl_path)

    smpl_layer_for_faces = smplx.create(
        str(smpl_folder),
        model_type="smpl",
        gender="neutral",
    )
    faces_smpl = smpl_layer_for_faces.faces

    human_result_dirs = discover_human_result_dirs(video_dir)
    if not human_result_dirs:
        raise FileNotFoundError(
            f"Could not find any per-human hmr4d_results.pt under: {video_dir}"
        )

    for result_dir in human_result_dirs:
        export_one_human(
            result_dir=result_dir,
            output_dir=output_dir / result_dir.name / "meshes",
            smplx_layer=smplx_layer,
            smplx2smpl=smplx2smpl,
            faces_smpl=faces_smpl,
            gvhmr_path=gvhmr_path,
            camera_json_path=camera_json_path,
            stabilize_incam=bool(args.stabilize_incam),
        )

    print(f"\nDone! Files saved in: {output_dir}")


if __name__ == "__main__":
    main()
