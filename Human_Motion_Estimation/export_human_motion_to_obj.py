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
        default="output_human_motion/frame_02_video_THUDM_CogVideoX_5b_I2V/hmr4d_results.pt"
    )
    parser.add_argument(
        "--smpl_folder",
        type=str,
        default="../../GVHMR/inputs/checkpoints/body_models/"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_objs"
    )
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {args.result_path}...")
    data = torch.load(args.result_path)

    params = data['smpl_params_global']

    body_pose = params['body_pose'].cpu()
    betas = params['betas'].cpu()
    global_orient = params['global_orient'].cpu()
    transl = params['transl'].cpu()

    num_frames = body_pose.shape[0]
    print(f"Found {num_frames} frames.")
    print(f"Input Pose Shape: {body_pose.shape}")

    # 2. Setup SMPL Layer
    print(f"Loading SMPL from {args.smpl_folder}...")
    smpl_layer = smplx.create(
        args.smpl_folder,
        model_type='smpl',
        gender='neutral',
        batch_size=1
    )

    # 3. Create Output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Loop through frames
    print("Exporting frames to .obj...")
    for i in tqdm(range(num_frames)):
        # Handle shape parameters (repeat or slice)
        if betas.shape[0] > 1:
            curr_betas = betas[i:i + 1]
        else:
            curr_betas = betas

        # Handle pose parameters (pad if missing hands)
        curr_pose = body_pose[i:i + 1]
        if curr_pose.shape[1] == 63:
            padding = torch.zeros(
                (1, 6),
                dtype=curr_pose.dtype,
                device=curr_pose.device
            )
            curr_pose = torch.cat([curr_pose, padding], dim=1)

        output = smpl_layer(
            betas=curr_betas,
            body_pose=curr_pose,
            global_orient=global_orient[i:i + 1],
            transl=transl[i:i + 1]
        )

        vertices = output.vertices.detach().cpu().numpy()[0]
        faces = smpl_layer.faces

        filename = os.path.join(args.output_dir, f"frame_{i:04d}.obj")
        with open(filename, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                # OBJ indices start at 1
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    print(f"\nDone! Files saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
