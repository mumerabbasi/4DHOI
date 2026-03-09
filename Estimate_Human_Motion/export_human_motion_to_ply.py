import argparse
from pathlib import Path

import smplx
import torch
from tqdm import tqdm


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        type=str,
        default="./output/video_01",
        help="Directory containing hmr4d_results.pt.",
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
            "<video_dir>/output_plys."
        ),
    )
    parser.add_argument(
        "--smplx2smpl_path",
        type=str,
        default="../../GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        help="Path to the smplx2smpl sparse matrix from GVHMR.",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir).resolve()
    result_path = video_dir / "hmr4d_results.pt"
    smpl_folder = Path(args.smpl_folder).resolve()
    smplx2smpl_path = Path(args.smplx2smpl_path).resolve()

    if not video_dir.exists() or not video_dir.is_dir():
        raise NotADirectoryError(f"Video directory not found: {video_dir}")
    if not result_path.exists():
        raise FileNotFoundError(f"Could not find hmr4d_results.pt in: {video_dir}")

    if args.output_dir is None:
        output_dir = video_dir / "output_plys"
    else:
        output_dir = Path(args.output_dir).resolve()

    # 1. Load Data
    print(f"Loading data from {result_path}...")
    data = torch.load(result_path)

    # Use smpl_params_incam so the human is in camera-frame coordinates,
    # matching other camera-frame assets (e.g. object meshes from sam3d).
    params = data["smpl_params_incam"]

    body_pose = params["body_pose"]
    betas = params["betas"]
    global_orient = params["global_orient"]
    transl = params["transl"].clone()

    num_frames = body_pose.shape[0]
    print(f"Found {num_frames} frames.")
    print(f"Input Pose Shape: {body_pose.shape}")

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

    # 4. Create Output Directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Loop through frames
    print("Exporting frames to .ply...")
    for i in tqdm(range(num_frames)):
        # Handle shape parameters (repeat or slice)
        curr_betas = betas[i: i + 1] if betas.shape[0] > 1 else betas[:1]

        output = smplx_layer(
            betas=curr_betas,
            body_pose=body_pose[i: i + 1],
            global_orient=global_orient[i: i + 1],
            transl=transl[i: i + 1],
        )

        # Convert SMPL-X vertices (10475) to SMPL vertices (6890)
        smplx_verts = output.vertices[0]  # (10475, 3)
        smpl_verts = torch.matmul(smplx2smpl, smplx_verts)  # (6890, 3)
        vertices = smpl_verts.detach().cpu().numpy()

        filename = output_dir / f"frame_{i:04d}.ply"
        write_ascii_ply(filename, vertices, faces_smpl)

    print(f"\nDone! Files saved in: {output_dir}")


if __name__ == "__main__":
    main()
