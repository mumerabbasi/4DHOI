import json
import math

import cv2
import torch
import trimesh
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, matrix_to_euler_angles

IMAGE_FILE = "frame_00/rgb.jpeg"
PLY_MODEL_FILE = "frame_00/mesh_canonical.ply"
GLB_MODEL_FILE = "frame_00/mesh_canonical.glb"
JSON_FILE = "frame_00/inference_transforms.json"

OUTPUT_FILE_PREFIX = "overlay"


def compose_transform(
    scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> Transform3d:
    """
    Args:
        scale: (..., 3) tensor of scale factors
        rotation: (..., 3, 3) tensor of rotation matrices
        translation: (..., 3) tensor of translation vectors
    """
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)


def render_aligned_overlay(image_path, model_path, json_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    h, w, _ = img.shape

    # 2. Load Model
    mesh = trimesh.load(model_path)
    if isinstance(mesh, trimesh.Scene):
        processed_mesh = mesh.dump(concatenate=True)
        verts = torch.from_numpy(processed_mesh.vertices).float().to(device)
        raw_colors = getattr(processed_mesh.visual, "vertex_colors", None)
    else:
        verts = torch.from_numpy(mesh.vertices).float().to(device)
        raw_colors = getattr(mesh.visual, "vertex_colors", None)

    # 3. Handle Coordinate Mismatch
    # GLB is Y-up, but transforms are Z-up. We must rotate Y-up -> Z-up.
    # Trimesh loads the GLB in Y-up coordinates. While blender converts to Z-up on import.
    if model_path.lower().endswith(".glb"):
        # Corrected rotation matrix for Y-up to Z-up
        r_y2z = torch.tensor(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ],
            dtype=torch.float32,
            device=device,
        )
        verts = torch.matmul(verts, r_y2z.T)

    # 4. Load Z-up Transforms
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    t_data = data["transforms"]
    quat = torch.tensor(t_data["rotation"]).reshape(-1, 4).to(device)
    trans = torch.tensor(t_data["translation"]).reshape(-1, 3).to(device)
    scale = torch.tensor(t_data["scale"]).reshape(-1, 3).to(device)

    # 5. Transform to Camera Space (Z-up context)
    rot_matrix = quaternion_to_matrix(quat)
    # Convert rotation matrix -> Euler angles (radians) in XYZ convention
    euler_rad = matrix_to_euler_angles(rot_matrix, convention="XYZ")  # shape: (N, 3)
    # Radians -> degrees
    euler_deg = euler_rad * (180.0 / math.pi)
    print("Quaternion:\n", quat)
    print("Euler Angles (degrees):\n", euler_deg)
    print("Rotation Matrix:\n", rot_matrix)
    print("Translation Vector:\n", trans)
    print("Scale Vector:\n", scale)
    tfm = compose_transform(scale=scale, rotation=rot_matrix, translation=trans)
    points_cam = tfm.transform_points(verts).squeeze(0)

    # 6. Perspective Projection with PyTorch3D Flips
    # TODO: fix focal length if intrinsics are known
    focal_length = max(h, w) * 1.15
    cx, cy = w / 2, h / 2

    z = points_cam[:, 2]
    mask_indices = torch.where(z > 0.1)[0]  # Filter points behind camera

    pts_valid = points_cam[mask_indices].cpu().numpy()
    z_v = pts_valid[:, 2]

    # --- THE FIX ---
    # In PyTorch3D: +X is left, +Y is up.
    # In Screen Space: +X is right, +Y is down.
    # Therefore, we negate both x and y during projection.
    # Source: https://github.com/facebookresearch/sam-3d-objects/issues/56#issuecomment-3614878364
    u = ((-pts_valid[:, 0] * focal_length) / z_v + cx).astype(int)
    v = ((-pts_valid[:, 1] * focal_length) / z_v + cy).astype(int)

    in_view = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_final, v_final = u[in_view], v[in_view]

    # 7. Optimized Colored Rendering
    if raw_colors is not None:
        valid_colors = raw_colors[mask_indices.cpu().numpy()][in_view]
        for i in range(len(u_final)):
            c = valid_colors[i]
            # Convert RGBA/RGB to BGR for OpenCV
            cv2.circle(
                img,
                (u_final[i], v_final[i]),
                1,
                (int(c[2]), int(c[1]), int(c[0])),
                -1,
            )
    else:
        for i in range(len(u_final)):
            cv2.circle(img, (u_final[i], v_final[i]), 1, (0, 0, 255), -1)

    # Determine output path based on model type
    if model_path.lower().endswith(".glb"):
        cv2.putText(
            img,
            "Y-up to Z-up + Py3D Flip applied",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        output_path = f"{OUTPUT_FILE_PREFIX}_glb_corrected.png"
    else:
        output_path = f"{OUTPUT_FILE_PREFIX}_ply.png"

    cv2.imwrite(output_path, img)
    print(f"Corrected overlay saved to {output_path}")


if __name__ == "__main__":
    render_aligned_overlay(IMAGE_FILE, PLY_MODEL_FILE, JSON_FILE)
    render_aligned_overlay(IMAGE_FILE, GLB_MODEL_FILE, JSON_FILE)
