from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

from common import load_json, save_json


IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}


def resolve_scannet_root(
    script_dir: Path,
    raw_scannet_root: str | None,
) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (script_dir.parent.parent / "Scannet++" / "data").resolve()


def resolve_scene_paths(
    scannet_root: Path,
    scene_context: dict[str, Any],
) -> dict[str, Path]:
    scene_id = scene_context["scene_id"]
    camera = scene_context["camera"]
    camera_source = camera["source"]
    camera_name = camera["name"]
    if camera_source not in IMAGE_SOURCE_TO_REL_PATHS:
        raise ValueError(
            f"Unsupported camera.source '{camera_source}'. "
            f"Supported values: {sorted(IMAGE_SOURCE_TO_REL_PATHS)}"
        )

    image_rel, transforms_rel = IMAGE_SOURCE_TO_REL_PATHS[camera_source]
    scene_root = scannet_root / scene_id
    return {
        "scene_root": scene_root,
        "image_path": scene_root / image_rel / camera_name,
        "transforms_path": scene_root / transforms_rel,
        "colmap_images_path": scene_root / "dslr" / "colmap" / "images.txt",
        "mesh_path": scene_root / "scans" / "mesh_aligned_0.05.ply",
    }


def build_pinhole_intrinsics(
    transforms_payload: dict[str, Any],
) -> tuple[np.ndarray, int, int]:
    width = int(transforms_payload["w"])
    height = int(transforms_payload["h"])
    intrinsics = np.array(
        [
            [
                float(transforms_payload["fl_x"]),
                0.0,
                float(transforms_payload["cx"]),
            ],
            [
                0.0,
                float(transforms_payload["fl_y"]),
                float(transforms_payload["cy"]),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics, width, height


def colmap_qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.astype(np.float64)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def load_colmap_pose(
    colmap_images_path: Path,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    for line in colmap_images_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[-1] != camera_name:
            continue
        qvec = np.asarray(list(map(float, parts[1:5])), dtype=np.float32)
        tvec = np.asarray(list(map(float, parts[5:8])), dtype=np.float32)
        return colmap_qvec_to_rotmat(qvec), tvec
    raise ValueError(
        f"Could not find camera '{camera_name}' in {colmap_images_path}"
    )


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as Trimesh: {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    colors = np.full((verts.shape[0], 3), 190, dtype=np.uint8)
    visual = getattr(mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) == verts.shape[0]:
        colors = np.asarray(vertex_colors[:, :3], dtype=np.uint8)
    return verts, faces, colors


def load_binary_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    target_h, target_w = target_hw
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(
            mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 127


def rasterize_faces(
    verts_world: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> Any:
    mesh = Meshes(
        verts=[torch.from_numpy(verts_world.astype(np.float32)).to(device)],
        faces=[torch.from_numpy(faces.astype(np.int64)).to(device)],
    )
    camera = cameras_from_opencv_projection(
        R=torch.from_numpy(
            rotation_world_to_camera.astype(np.float32)
        )[None].to(device),
        tvec=torch.from_numpy(
            translation_world_to_camera.astype(np.float32)
        )[None].to(device),
        camera_matrix=torch.from_numpy(
            intrinsics.astype(np.float32)
        )[None].to(device),
        image_size=torch.tensor(
            [[height, width]],
            dtype=torch.float32,
            device=device,
        ),
    )
    rasterizer = MeshRasterizer(
        cameras=camera,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
        ),
    )
    with torch.no_grad():
        return rasterizer(mesh)


def compute_crop_from_mask(
    mask: np.ndarray,
    target_fill_frac: float,
    padding_frac: float,
    output_aspect: float,
) -> dict[str, Any]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise RuntimeError(
            "Cannot compute target render crop from an empty target mask."
        )

    height, width = mask.shape
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    bbox_w = max(1.0, x1 - x0)
    bbox_h = max(1.0, y1 - y0)
    fill = min(max(float(target_fill_frac), 0.05), 1.0)
    pad = max(0.0, float(padding_frac))

    crop_w = max(bbox_w / fill, bbox_w * (1.0 + 2.0 * pad))
    crop_h = max(bbox_h / fill, bbox_h * (1.0 + 2.0 * pad))
    aspect = float(output_aspect)
    if aspect <= 0.0:
        raise ValueError("output_aspect must be > 0.")
    if crop_w / crop_h < aspect:
        crop_w = crop_h * aspect
    else:
        crop_h = crop_w / aspect
    if crop_w > width:
        crop_w = float(width)
        crop_h = crop_w / aspect
    if crop_h > height:
        crop_h = float(height)
        crop_w = crop_h * aspect

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    crop_x0 = cx - 0.5 * crop_w
    crop_y0 = cy - 0.5 * crop_h
    crop_x1 = cx + 0.5 * crop_w
    crop_y1 = cy + 0.5 * crop_h

    if crop_x0 < 0:
        crop_x1 -= crop_x0
        crop_x0 = 0.0
    if crop_y0 < 0:
        crop_y1 -= crop_y0
        crop_y0 = 0.0
    if crop_x1 > width:
        shift = crop_x1 - width
        crop_x0 = max(0.0, crop_x0 - shift)
        crop_x1 = float(width)
    if crop_y1 > height:
        shift = crop_y1 - height
        crop_y0 = max(0.0, crop_y0 - shift)
        crop_y1 = float(height)

    xi0 = int(np.floor(crop_x0))
    yi0 = int(np.floor(crop_y0))
    xi1 = int(np.ceil(crop_x1))
    yi1 = int(np.ceil(crop_y1))
    xi0 = max(0, min(width - 1, xi0))
    yi0 = max(0, min(height - 1, yi0))
    xi1 = max(xi0 + 1, min(width, xi1))
    yi1 = max(yi0 + 1, min(height, yi1))

    return {
        "bbox_xyxy_exclusive": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ],
        "crop_xyxy_exclusive": [xi0, yi0, xi1, yi1],
        "crop_width": int(xi1 - xi0),
        "crop_height": int(yi1 - yi0),
    }


def build_target_render_intrinsics(
    intrinsics: np.ndarray,
    crop: dict[str, Any],
    output_width: int,
    output_height: int,
) -> np.ndarray:
    crop_x0, crop_y0, crop_x1, crop_y1 = crop["crop_xyxy_exclusive"]
    crop_w = int(crop_x1 - crop_x0)
    crop_h = int(crop_y1 - crop_y0)
    scale_x = float(output_width) / float(crop_w)
    scale_y = float(output_height) / float(crop_h)

    target_intrinsics = intrinsics.astype(np.float32).copy()
    target_intrinsics[0, 0] *= scale_x
    target_intrinsics[1, 1] *= scale_y
    target_intrinsics[0, 2] = (
        target_intrinsics[0, 2] - float(crop_x0)
    ) * scale_x
    target_intrinsics[1, 2] = (
        target_intrinsics[1, 2] - float(crop_y0)
    ) * scale_y
    return target_intrinsics


def compact_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_vids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return (
        verts[unique_vids].astype(np.float32),
        inverse.reshape(-1, 3).astype(np.int64),
        colors[unique_vids].astype(np.uint8),
    )


def render_colored_mesh(
    verts_world: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    intrinsics: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    verts_compact, faces_compact, colors_compact = compact_mesh(
        verts_world,
        faces,
        vertex_colors,
    )
    fragments = rasterize_faces(
        verts_world=verts_compact,
        faces=faces_compact,
        intrinsics=intrinsics,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        width=width,
        height=height,
        device=device,
    )

    face_map_t = fragments.pix_to_face[0, ..., 0]
    bary_t = fragments.bary_coords[0, ..., 0, :]
    valid_t = face_map_t >= 0

    image = torch.zeros(
        (height, width, 3),
        dtype=torch.float32,
        device=device,
    )
    if bool(valid_t.any()):
        faces_t = torch.from_numpy(faces_compact.astype(np.int64)).to(device)
        colors_t = torch.from_numpy(
            colors_compact.astype(np.float32)
        ).to(device)
        valid_face_ids = face_map_t[valid_t]
        valid_face_verts = faces_t[valid_face_ids]
        valid_colors = colors_t[valid_face_verts]
        valid_bary = bary_t[valid_t][..., None]
        image[valid_t] = (valid_colors * valid_bary).sum(dim=1)

    image_np = image.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
    mask_np = valid_t.detach().cpu().numpy().astype(np.uint8) * 255
    return image_np, mask_np


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Render a ScanNet++ target-object surface view."
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument(
        "--outdir",
        default=None,
        help="Video output root. Target render files are written here.",
    )
    parser.add_argument("--target-fill-frac", type=float, default=0.82)
    parser.add_argument("--padding-frac", type=float, default=0.20)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else project_dir / "01_Generate_SIG" / "input_prompts" / args.video_name
    )
    selection_json_path = (
        Path(args.selection_json).resolve()
        if args.selection_json
        else project_dir
        / "02_Select_Target_Instance"
        / "output"
        / args.video_name
        / "target_selection.json"
    )
    scannet_root = resolve_scannet_root(script_dir, args.scannet_root)
    video_output_root = (
        Path(args.outdir).resolve()
        if args.outdir
        else script_dir / "output" / args.video_name
    )
    output_root = video_output_root

    input_payload = load_json(input_dir / "input_scene.json")
    scene_context = input_payload["scene_context"]
    scene_paths = resolve_scene_paths(scannet_root, scene_context)

    scene_bgr = cv2.imread(str(scene_paths["image_path"]), cv2.IMREAD_COLOR)
    if scene_bgr is None:
        raise FileNotFoundError(
            f"Failed to read scene image: {scene_paths['image_path']}"
        )

    transforms_payload = load_json(scene_paths["transforms_path"])
    intrinsics, width, height = build_pinhole_intrinsics(transforms_payload)
    if scene_bgr.shape[:2] != (height, width):
        raise ValueError(
            "Scene image shape does not match ScanNet++ camera metadata: "
            f"image={scene_bgr.shape[1]}x{scene_bgr.shape[0]}, "
            f"metadata={width}x{height}"
        )

    rotation_world_to_camera, translation_world_to_camera = load_colmap_pose(
        scene_paths["colmap_images_path"],
        scene_context["camera"]["name"],
    )

    selection_payload = load_json(selection_json_path)
    target_mask_path = selection_json_path.parent / str(
        selection_payload["target_selection"]["mask_path"]
    )
    target_mask = load_binary_mask(target_mask_path, (height, width))

    verts_world, mesh_faces, vertex_colors = load_mesh(
        scene_paths["mesh_path"]
    )

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    crop = compute_crop_from_mask(
        target_mask,
        target_fill_frac=args.target_fill_frac,
        padding_frac=args.padding_frac,
        output_aspect=float(width) / float(height),
    )
    target_intrinsics = build_target_render_intrinsics(
        intrinsics=intrinsics,
        crop=crop,
        output_width=width,
        output_height=height,
    )
    render_rgb, _render_mask = render_colored_mesh(
        verts_world=verts_world,
        faces=mesh_faces,
        vertex_colors=vertex_colors,
        intrinsics=target_intrinsics,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        width=width,
        height=height,
        device=device,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    render_path = output_root / "target_render.png"
    camera_path = output_root / "target_render_camera.json"

    save_rgb(render_path, render_rgb)
    for stale_path in (
        output_root / "target_surface_match_overlay.png",
        output_root / "target_surface_camera.json",
    ):
        if stale_path.exists():
            stale_path.unlink()

    save_json(
        camera_path,
        {
            "intrinsics_3x3": target_intrinsics.astype(float).tolist(),
        },
    )

    print(f"Wrote target render: {render_path}")
    print(f"Wrote target render camera JSON: {camera_path}")


if __name__ == "__main__":
    main()
