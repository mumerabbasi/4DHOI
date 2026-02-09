import os
import argparse
import torch
import smplx
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_path",
        type=str,
        default="output_human_motion/bori_1/hmr4d_results.pt"
    )
    parser.add_argument(
        "--smpl_folder",
        type=str,
        default="../../GVHMR/inputs/checkpoints/body_models/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_human_motion/bori_1/output_objs"
    )
    parser.add_argument(
        "--smplx2smpl_path",
        type=str,
        default="../../GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        help="Path to the smplx2smpl sparse matrix from GVHMR."
    )
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {args.result_path}...")
    data = torch.load(args.result_path)

    # Use smpl_params_incam so the human is in camera-frame coordinates,
    # matching other camera-frame assets (e.g. object meshes from sam3d).
    # Previously used smpl_params_global which places the human in a
    # gravity-aligned world frame (centered at origin, wrong for compositing).
    params = data['smpl_params_incam']

    body_pose = params['body_pose']
    betas = params['betas']
    global_orient = params['global_orient']
    transl = params['transl']

    num_frames = body_pose.shape[0]
    print(f"Found {num_frames} frames.")
    print(f"Input Pose Shape: {body_pose.shape}")

    # 2. Setup SMPL-X model (matching GVHMR's "supermotion" config)
    # GVHMR outputs SMPL-X parameters (body_pose dim=63 for 21 joints).
    # We must use the same SMPL-X model, then convert vertices to SMPL
    # topology via the smplx2smpl sparse matrix for correct mesh output.
    print(f"Loading SMPL-X from {args.smpl_folder}...")
    smplx_layer = smplx.create(
        args.smpl_folder,
        model_type='smplx',
        gender='neutral',
        num_pca_comps=12,
        flat_hand_mean=False,
        create_body_pose=False,
        create_betas=False,
        create_global_orient=False,
        create_transl=False,
    )

    # Load the SMPL-X to SMPL vertex conversion matrix (6890 x 10475)
    # and the SMPL face topology for OBJ export.
    print(f"Loading smplx2smpl matrix from {args.smplx2smpl_path}...")
    smplx2smpl = torch.load(args.smplx2smpl_path)

    smpl_layer_for_faces = smplx.create(
        args.smpl_folder,
        model_type='smpl',
        gender='neutral',
    )
    faces_smpl = smpl_layer_for_faces.faces

    # 3. Create Output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Loop through frames
    print("Exporting frames to .obj...")
    for i in tqdm(range(num_frames)):
        # Handle shape parameters (repeat or slice)
        if betas.shape[0] > 1:
            curr_betas = betas[i:i + 1]
        else:
            curr_betas = betas[:1]

        output = smplx_layer(
            betas=curr_betas,
            body_pose=body_pose[i:i + 1],
            global_orient=global_orient[i:i + 1],
            transl=transl[i:i + 1],
        )

        # Convert SMPL-X vertices (10475) to SMPL vertices (6890)
        smplx_verts = output.vertices[0]  # (10475, 3)
        smpl_verts = torch.matmul(smplx2smpl, smplx_verts)  # (6890, 3)
        vertices = smpl_verts.detach().cpu().numpy()

        filename = os.path.join(args.output_dir, f"frame_{i:04d}.obj")
        with open(filename, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces_smpl:
                # OBJ indices start at 1
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"\nDone! Files saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
