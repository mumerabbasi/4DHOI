from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCANNET_ROOT = PROJECT_DIR.parent / "Scannet++" / "data"
DEFAULT_SMPL_SEG_JSON = PROJECT_DIR / "05_Estimate_Human_Pose" / "assets" / "smplx_vert_segmentation.json"
VISUAL_SEGMENT_ALIASES = {
    "left_hand_inner": "left_hand",
    "right_hand_inner": "right_hand",
    "left_foot_bottom": "left_foot",
    "right_foot_bottom": "right_foot",
}


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_label(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").replace("-", " ").split())


def slugify(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def default_paths(interaction_name: str) -> dict[str, Path]:
    return {
        "optimizer_output_root": PROJECT_DIR /
        "06_Optimize_Static_Scene" /
        "output" /
        interaction_name,
        "sig_json": PROJECT_DIR /
        "01_Generate_SIG" /
        "output" /
        interaction_name /
        "scene_interaction_graph.json",
        "input_scene_json": PROJECT_DIR /
        "01_Generate_SIG" /
        "input_prompts" /
        interaction_name /
        "input_scene.json",
        "outdir": SCRIPT_DIR /
        "output" /
        interaction_name,
        "contact_spec": PROJECT_DIR /
        "04_Estimate_Contact" /
        "output" /
        interaction_name /
        "contact_spec.json",
        "contact_canvas_image": PROJECT_DIR /
        "04_Estimate_Contact" /
        "output" /
        interaction_name /
        "prompt" /
        "target_scene_crop.png",
    }


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).resolve() if raw_path else default_path.resolve()


def parse_size(text: str) -> tuple[int, int]:
    normalized = str(text).lower().replace(" ", "")
    if "x" not in normalized:
        raise ValueError(f"Expected size formatted like 480x480, got {text!r}")
    width_text, height_text = normalized.split("x", 1)
    width = int(width_text)
    height = int(height_text)
    if width <= 0 or height <= 0:
        raise ValueError(f"Size must be positive, got {text!r}")
    return width, height


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def rank_failure_tags(tags: list[str]) -> list[str]:
    priority = [
        "missing_contact",
        "severe_penetration",
        "implausible_pose",
        "wrong_interaction",
        "wrong_target",
        "no_decision",
    ]
    normalized = [slugify(tag) for tag in tags if tag]
    return sorted(
        set(normalized),
        key=lambda tag: (priority.index(tag) if tag in priority else 999, tag),
    )


def collect_contact_metrics(alignment_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    final_edges = (
        alignment_summary.get("human", {})
        .get("final_frame_0", {})
        .get("interaction_edges", [])
    )
    if not isinstance(final_edges, list):
        raise ValueError("alignment_summary human.final_frame_0.interaction_edges must be a list")

    threshold_m = float(args.contact_threshold_m)
    edges: list[dict[str, Any]] = []
    distances: list[float] = []
    failure_tags: list[str] = []

    for index, edge in enumerate(final_edges):
        if not isinstance(edge, dict):
            continue
        distance = edge.get("nocontact_distance_m")
        distance_m = float(distance) if distance is not None else None
        passed = distance_m is not None and distance_m <= threshold_m
        if not passed:
            failure_tags.append("missing_contact")
        if distance_m is not None:
            distances.append(distance_m)
        edges.append(
            {
                "index": int(index),
                "moving_part_name": edge.get("moving_part_name"),
                "moving_segment_id": edge.get("moving_segment_id"),
                "fixed_part_name": edge.get("fixed_part_name"),
                "fixed_entity_name": edge.get("fixed_entity_name"),
                "fixed_point_count": edge.get("fixed_point_count"),
                "moving_vertex_count": edge.get("moving_vertex_count"),
                "reduction": edge.get("reduction"),
                "nocontact_raw": edge.get("nocontact_raw"),
                "nocontact_distance_m": distance_m,
                "threshold_m": threshold_m,
                "pass": bool(passed),
            }
        )

    return {
        "pass": bool(edges) and all(bool(edge["pass"]) for edge in edges),
        "failure_tags": rank_failure_tags(failure_tags),
        "edge_count": int(len(edges)),
        "edges": edges,
        "mean_distance_m": float(sum(distances) / len(distances)) if distances else None,
        "max_distance_m": float(max(distances)) if distances else None,
    }


def validate_required_inputs(optimizer_root: Path, sig_json_path: Path, input_scene_json_path: Path) -> None:
    required = [
        sig_json_path,
        input_scene_json_path,
        optimizer_root / "alignment_summary.json",
        optimizer_root / "meshes" / "frame_0000_world.ply",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required evaluation input(s): " + ", ".join(missing))


def import_render_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import trimesh
        from PIL import Image
        from pytorch3d.ops import interpolate_face_attributes
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.structures import Meshes
        from pytorch3d.utils import cameras_from_opencv_projection
    except Exception as error:
        raise RuntimeError(
            "Render dependencies are unavailable. Run this inside the 4dhsi environment, e.g. "
            "`conda run -n 4dhsi python 07_Evaluate_Static_Scene/01_render_views.py ...`. "
            f"Original import error: {error}"
        ) from error
    return {
        "np": np,
        "torch": torch,
        "trimesh": trimesh,
        "Image": Image,
        "interpolate_face_attributes": interpolate_face_attributes,
        "MeshRasterizer": MeshRasterizer,
        "RasterizationSettings": RasterizationSettings,
        "cameras_from_opencv_projection": cameras_from_opencv_projection,
        "Meshes": Meshes,
    }


def colmap_qvec_to_rotmat(qvec: list[float]) -> Any:
    import numpy as np

    qw, qx, qy, qz = qvec
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def read_colmap_w2c(images_txt: Path, image_name: str) -> Any:
    import numpy as np

    if not images_txt.exists():
        raise FileNotFoundError(f"COLMAP images.txt not found: {images_txt}")
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 10 and parts[9] == image_name:
            rotation = colmap_qvec_to_rotmat([float(value) for value in parts[1:5]])
            translation = np.asarray([float(value) for value in parts[5:8]], dtype=np.float32)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = rotation
            w2c[:3, 3] = translation
            return w2c
    raise FileNotFoundError(f"Camera pose for {image_name} not found in {images_txt}")


def read_camera(input_scene: dict[str, Any], scannet_root: Path) -> dict[str, Any]:
    scene_context = input_scene.get("scene_context", {})
    scene_id = scene_context.get("scene_id")
    camera = scene_context.get("camera", {})
    image_name = camera.get("name")
    source = camera.get("source", "dslr_resized_undistorted")
    if not scene_id or not image_name:
        raise ValueError("input_scene_json must contain scene_context.scene_id and scene_context.camera.name")
    if source != "dslr_resized_undistorted":
        raise ValueError(f"Only dslr_resized_undistorted camera source is supported for Module 07 v1; got {source!r}")

    scene_dir = scannet_root / str(scene_id)
    transforms_path = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    colmap_images_path = scene_dir / "dslr" / "colmap" / "images.txt"
    image_path = scene_dir / "dslr" / "resized_undistorted_images" / str(image_name)
    scene_mesh_path = scene_dir / "scans" / "mesh_aligned_0.05.ply"
    transforms = load_json(transforms_path)
    frames = transforms.get("frames", []) + transforms.get("test_frames", [])
    frame = next((item for item in frames if Path(str(item.get("file_path", ""))).name == image_name), None)
    if frame is None:
        raise FileNotFoundError(f"Camera frame {image_name} not found in {transforms_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Original camera image not found: {image_path}")
    if not scene_mesh_path.exists():
        raise FileNotFoundError(f"ScanNet++ scene mesh not found: {scene_mesh_path}")

    return {
        "scene_id": scene_id,
        "image_name": image_name,
        "image_path": image_path,
        "scene_mesh_path": scene_mesh_path,
        "w2c": read_colmap_w2c(colmap_images_path, str(image_name)),
        "width": int(transforms["w"]),
        "height": int(transforms["h"]),
        "fx": float(transforms["fl_x"]),
        "fy": float(transforms["fl_y"]),
        "cx": float(transforms["cx"]),
        "cy": float(transforms["cy"]),
        "transforms_path": transforms_path,
        "colmap_images_path": colmap_images_path,
    }


def read_contact_crop_camera(
    base_camera: dict[str, Any],
    contact_spec_path: Path,
    contact_canvas_image: Path,
    Image: Any,
) -> dict[str, Any]:
    if not contact_spec_path.exists():
        raise FileNotFoundError(f"Contact spec JSON not found: {contact_spec_path}")
    if not contact_canvas_image.exists():
        raise FileNotFoundError(f"Contact canvas image not found: {contact_canvas_image}")

    payload = load_json(contact_spec_path)
    camera_payload = payload.get("camera")
    if not isinstance(camera_payload, dict):
        raise ValueError(f"Expected camera object in {contact_spec_path}")
    intrinsics = camera_payload.get("intrinsics_3x3")
    if not isinstance(intrinsics, list) or len(intrinsics) != 3:
        raise ValueError(f"Expected camera.intrinsics_3x3 in {contact_spec_path}")

    with Image.open(contact_canvas_image) as image:
        width, height = image.size

    crop_camera = dict(base_camera)
    crop_camera.update(
        {
            "width": int(width),
            "height": int(height),
            "fx": float(intrinsics[0][0]),
            "fy": float(intrinsics[1][1]),
            "cx": float(intrinsics[0][2]),
            "cy": float(intrinsics[1][2]),
            "contact_spec": contact_spec_path,
            "contact_canvas_image": contact_canvas_image,
        }
    )
    return crop_camera


def choose_device(torch: Any, requested: str) -> Any:
    device = torch.device(str(requested))
    if device.type != "cuda":
        raise RuntimeError("--render-device must be a CUDA device like cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError(f"--render-device {requested} was requested, but CUDA is not available")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"--render-device {requested} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available"
        )
    return device


def as_vertex_colors(mesh: Any, np: Any, color: tuple[float, float, float] | None = None) -> Any:
    vertex_count = len(mesh.vertices)
    if color is not None:
        return np.tile(np.asarray(color, dtype=np.float32), (vertex_count, 1))
    visual = getattr(mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is None or len(vertex_colors) != vertex_count:
        return np.tile(np.asarray([0.62, 0.62, 0.62], dtype=np.float32), (vertex_count, 1))
    colors = np.asarray(vertex_colors[:, :3], dtype=np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return colors


def filter_scene_faces_to_contact_camera(
    vertices_world: Any,
    faces: Any,
    camera: dict[str, Any],
    np: Any,
    max_depth_m: float = 20.0,
    border_px: float = 96.0,
) -> Any:
    w2c = np.asarray(camera["w2c"], dtype=np.float32)
    camera_points = vertices_world @ w2c[:3, :3].T + w2c[:3, 3][None, :]
    triangles = camera_points[faces]
    z = triangles[..., 2]
    positive = np.any(z > 1e-6, axis=1)
    if max_depth_m is not None:
        positive &= np.any(z < float(max_depth_m), axis=1)
    if not np.any(positive):
        return faces[:0].copy()

    z_safe = np.clip(z, 1e-6, None)
    u = float(camera["fx"]) * triangles[..., 0] / z_safe + float(camera["cx"]) - 0.5
    v = float(camera["fy"]) * triangles[..., 1] / z_safe + float(camera["cy"]) - 0.5
    u_min = np.min(u, axis=1)
    u_max = np.max(u, axis=1)
    v_min = np.min(v, axis=1)
    v_max = np.max(v, axis=1)
    overlaps = (
        positive
        & (u_max >= -float(border_px))
        & (u_min <= float(int(camera["width"]) - 1) + float(border_px))
        & (v_max >= -float(border_px))
        & (v_min <= float(int(camera["height"]) - 1) + float(border_px))
    )
    return faces[overlaps].astype(np.int64)


def compact_scene_crop(
    vertices: Any,
    faces: Any,
    colors: Any,
    np: Any,
) -> tuple[Any, Any, Any, Any]:
    if faces.shape[0] == 0:
        raise RuntimeError("No ScanNet scene faces remained after contact-camera crop filtering.")
    unique_vids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    crop_vertices = vertices[unique_vids].astype(np.float32)
    crop_faces = inverse.reshape(-1, 3).astype(np.int64)
    crop_colors = colors[unique_vids].astype(np.float32)
    return crop_vertices, crop_faces, crop_colors, unique_vids.astype(np.int64)


def shaded_human_vertex_colors(mesh: Any, np: Any, camera: dict[str, Any]) -> Any:
    base = np.asarray([0.78, 0.76, 0.72], dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    camera_forward_world = np.asarray(camera["w2c"], dtype=np.float32)[
        :3, :3].T @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    light_dir = -camera_forward_world
    light_dir = light_dir / max(float(np.linalg.norm(light_dir)), 1e-6)
    diffuse = np.maximum(normals @ light_dir, 0.0)
    fill = np.maximum(normals @ np.asarray([0.2, -0.4, 0.9], dtype=np.float32), 0.0)
    shade = 0.42 + 0.46 * diffuse + 0.18 * fill
    return np.clip(base[None, :] * shade[:, None], 0.0, 1.0)


def highlight_vertex_colors(vertex_colors: Any, vertex_ids: list[int], np: Any) -> Any:
    highlighted = np.asarray(vertex_colors, dtype=np.float32).copy()
    if vertex_ids:
        highlighted[np.asarray(vertex_ids, dtype=np.int64)] = np.asarray([1.0, 0.04, 0.02], dtype=np.float32)
    return highlighted


def filter_faces_for_camera(
        vertices: Any,
        faces: Any,
        w2c: Any,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        np: Any) -> Any:
    camera_points = vertices @ w2c[:3, :3].T + w2c[:3, 3][None, :]
    z = camera_points[:, 2]
    valid_z = z > 0.02
    u = fx * (camera_points[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (camera_points[:, 1] / np.maximum(z, 1e-6)) + cy
    margin = 128
    in_view = valid_z & (u >= -margin) & (u <= width + margin) & (v >= -margin) & (v <= height + margin)
    face_mask = in_view[faces].any(axis=1) & valid_z[faces].any(axis=1)
    if face_mask.sum() < 10:
        return faces
    return faces[face_mask]


def rasterize_colored_mesh(
    vertices: Any,
    faces: Any,
    vertex_colors: Any,
    camera: dict[str, Any],
    image_size: tuple[int, int],
    deps: dict[str, Any],
    device: Any,
) -> tuple[Any, Any]:
    np = deps["np"]
    torch = deps["torch"]
    Meshes = deps["Meshes"]
    MeshRasterizer = deps["MeshRasterizer"]
    RasterizationSettings = deps["RasterizationSettings"]
    cameras_from_opencv_projection = deps["cameras_from_opencv_projection"]
    interpolate_face_attributes = deps["interpolate_face_attributes"]

    height, width = image_size
    w2c = np.asarray(camera["w2c"], dtype=np.float32)
    scale_x = width / float(camera["width"])
    scale_y = height / float(camera["height"])
    fx = float(camera["fx"]) * scale_x
    fy = float(camera["fy"]) * scale_y
    cx = float(camera["cx"]) * scale_x
    cy = float(camera["cy"]) * scale_y

    faces = filter_faces_for_camera(vertices, faces, w2c, fx, fy, cx, cy, width, height, np)
    if len(faces) == 0:
        image = np.ones((height, width, 3), dtype=np.float32)
        depth = np.full((height, width), np.inf, dtype=np.float32)
        return image, depth
    used_vertex_ids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = vertices[used_vertex_ids].astype(np.float32)
    vertex_colors = vertex_colors[used_vertex_ids].astype(np.float32)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    verts_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces.astype(np.int64), dtype=torch.int64, device=device)
    colors_t = torch.as_tensor(vertex_colors, dtype=torch.float32, device=device).clamp(0.0, 1.0)
    mesh = Meshes(verts=[verts_t], faces=[faces_t])

    R = torch.as_tensor(w2c[:3, :3][None], dtype=torch.float32, device=device)
    T = torch.as_tensor(w2c[:3, 3][None], dtype=torch.float32, device=device)
    K = torch.as_tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    image_size_t = torch.as_tensor([[height, width]], dtype=torch.float32, device=device)
    cameras = cameras_from_opencv_projection(R=R, tvec=T, camera_matrix=K, image_size=image_size_t)
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        ),
    )
    fragments = rasterizer(mesh)
    face_attrs = colors_t[faces_t]
    pix_colors = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_attrs)[0, :, :, 0, :]
    mask = fragments.pix_to_face[0, :, :, 0] >= 0
    depth = fragments.zbuf[0, :, :, 0]
    image = torch.ones((height, width, 3), dtype=torch.float32, device=device)
    image[mask] = pix_colors[mask]
    depth_np = depth.detach().cpu().numpy()
    depth_np[~mask.detach().cpu().numpy()] = np.inf
    return image.detach().cpu().numpy(), depth_np


def rasterize_depth(
    vertices: Any,
    faces: Any,
    camera: dict[str, Any],
    image_size: tuple[int, int],
    deps: dict[str, Any],
    device: Any,
) -> Any:
    np = deps["np"]
    torch = deps["torch"]
    Meshes = deps["Meshes"]
    MeshRasterizer = deps["MeshRasterizer"]
    RasterizationSettings = deps["RasterizationSettings"]
    cameras_from_opencv_projection = deps["cameras_from_opencv_projection"]

    height, width = image_size
    if len(faces) == 0:
        return np.full((height, width), np.inf, dtype=np.float32)
    w2c = np.asarray(camera["w2c"], dtype=np.float32)
    scale_x = width / float(camera["width"])
    scale_y = height / float(camera["height"])
    fx = float(camera["fx"]) * scale_x
    fy = float(camera["fy"]) * scale_y
    cx = float(camera["cx"]) * scale_x
    cy = float(camera["cy"]) * scale_y

    faces = filter_faces_for_camera(vertices, faces, w2c, fx, fy, cx, cy, width, height, np)
    if len(faces) == 0:
        return np.full((height, width), np.inf, dtype=np.float32)
    used_vertex_ids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = vertices[used_vertex_ids].astype(np.float32)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    verts_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces.astype(np.int64), dtype=torch.int64, device=device)
    mesh = Meshes(verts=[verts_t], faces=[faces_t])

    R = torch.as_tensor(w2c[:3, :3][None], dtype=torch.float32, device=device)
    T = torch.as_tensor(w2c[:3, 3][None], dtype=torch.float32, device=device)
    K = torch.as_tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    image_size_t = torch.as_tensor([[height, width]], dtype=torch.float32, device=device)
    cameras = cameras_from_opencv_projection(R=R, tvec=T, camera_matrix=K, image_size=image_size_t)
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        ),
    )
    fragments = rasterizer(mesh)
    mask = fragments.pix_to_face[0, :, :, 0] >= 0
    depth = fragments.zbuf[0, :, :, 0]
    depth_np = depth.detach().cpu().numpy()
    depth_np[~mask.detach().cpu().numpy()] = np.inf
    return depth_np


def composite_scene_and_human(render_assets: dict[str, Any],
                              camera: dict[str, Any], image_size: tuple[int, int]) -> Any:
    np = render_assets["deps"]["np"]
    scene_rgb, scene_depth = rasterize_colored_mesh(
        vertices=render_assets["scene_vertices"],
        faces=render_assets["scene_faces"],
        vertex_colors=render_assets["scene_colors"],
        camera=camera,
        image_size=image_size,
        deps=render_assets["deps"],
        device=render_assets["device"],
    )
    human_rgb, human_depth = rasterize_colored_mesh(
        vertices=render_assets["human_vertices"],
        faces=render_assets["human_faces"],
        vertex_colors=render_assets["human_colors"],
        camera=camera,
        image_size=image_size,
        deps=render_assets["deps"],
        device=render_assets["device"],
    )
    composite = scene_rgb.copy()
    human_mask = np.isfinite(human_depth) & (human_depth < scene_depth)
    composite[human_mask] = human_rgb[human_mask]
    return (np.clip(composite, 0.0, 1.0) * 255).astype(np.uint8)


def render_scene_with_human(
    camera: dict[str, Any],
    contact_crop_camera: dict[str, Any],
    human_mesh_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    deps = import_render_deps()
    np = deps["np"]
    torch = deps["torch"]
    trimesh = deps["trimesh"]
    device = choose_device(torch, args.render_device)

    scene_mesh = trimesh.load(camera["scene_mesh_path"], process=False, force="mesh")
    human_mesh = trimesh.load(human_mesh_path, process=False, force="mesh")
    full_scene_vertices = np.asarray(scene_mesh.vertices, dtype=np.float32)
    full_scene_faces = np.asarray(scene_mesh.faces, dtype=np.int64)
    full_scene_colors = as_vertex_colors(scene_mesh, np)
    crop_faces_world_ids = filter_scene_faces_to_contact_camera(
        full_scene_vertices,
        full_scene_faces,
        contact_crop_camera,
        np,
    )
    scene_vertices, scene_faces, scene_colors, scene_vertex_source_ids = compact_scene_crop(
        full_scene_vertices,
        crop_faces_world_ids,
        full_scene_colors,
        np,
    )
    log(
        "evidence",
        "rebuilt ScanNet contact crop "
        f"faces={scene_faces.shape[0]}/{full_scene_faces.shape[0]} "
        f"verts={scene_vertices.shape[0]}/{full_scene_vertices.shape[0]}",
    )
    render_w, render_h = parse_size(args.render_image_size)
    image_size = (render_h, render_w)

    render_assets = {
        "scene_vertices": scene_vertices,
        "scene_faces": scene_faces,
        "scene_colors": scene_colors,
        "scene_source_vertex_ids": scene_vertex_source_ids,
        "scene_crop": {
            "mode": "rebuilt_from_scannet_contact_camera_frustum",
            "source_scene_mesh": str(camera["scene_mesh_path"]),
            "contact_spec": str(contact_crop_camera["contact_spec"]),
            "contact_canvas_image": str(contact_crop_camera["contact_canvas_image"]),
            "full_scene_vertex_count": int(full_scene_vertices.shape[0]),
            "full_scene_face_count": int(full_scene_faces.shape[0]),
            "crop_vertex_count": int(scene_vertices.shape[0]),
            "crop_face_count": int(scene_faces.shape[0]),
        },
        "human_vertices": np.asarray(human_mesh.vertices, dtype=np.float32),
        "human_faces": np.asarray(human_mesh.faces, dtype=np.int64),
        "human_colors": shaded_human_vertex_colors(human_mesh, np, camera),
        "deps": deps,
        "device": device,
    }

    log("evidence", f"rendering original camera view at {render_w}x{render_h} on {device}")
    return {
        **render_assets,
        "image": composite_scene_and_human(render_assets, camera, image_size),
        "human_vertices": np.asarray(human_mesh.vertices, dtype=np.float32),
        "camera": camera,
        "render_width": render_w,
        "render_height": render_h,
        "deps": deps,
    }


def project_vertices(points_world: Any, camera: dict[str, Any], render_width: int, render_height: int, np: Any) -> Any:
    w2c = np.asarray(camera["w2c"], dtype=np.float32)
    points_camera = points_world @ w2c[:3, :3].T + w2c[:3, 3][None, :]
    z = points_camera[:, 2]
    scale_x = render_width / float(camera["width"])
    scale_y = render_height / float(camera["height"])
    fx = float(camera["fx"]) * scale_x
    fy = float(camera["fy"]) * scale_y
    cx = float(camera["cx"]) * scale_x
    cy = float(camera["cy"]) * scale_y
    u = fx * (points_camera[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (points_camera[:, 1] / np.maximum(z, 1e-6)) + cy
    return np.stack([u, v, z], axis=1)


def look_at_w2c(eye: Any, target: Any, np: Any) -> Any:
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    forward = np.asarray(target - eye, dtype=np.float32)
    forward = forward / np.maximum(np.linalg.norm(forward), 1e-6)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
    right = right / np.maximum(np.linalg.norm(right), 1e-6)
    down = np.cross(forward, right)
    down = down / np.maximum(np.linalg.norm(down), 1e-6)
    rotation = np.stack([right, down, forward], axis=0).astype(np.float32)
    translation = (-rotation @ eye.astype(np.float32)).astype(np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = translation
    return w2c


def human_bbox_info(human_vertices: Any, np: Any) -> dict[str, Any]:
    human_min = human_vertices.min(axis=0)
    human_max = human_vertices.max(axis=0)
    center = (human_min + human_max) * 0.5
    extent = human_max - human_min
    human_height_m = float(extent[2])
    human_xy_extent_m = float(max(extent[0], extent[1]))
    base_radius_m = max(1.4 * human_height_m, 2.2 * human_xy_extent_m, 1.2)
    return {
        "min": human_min,
        "max": human_max,
        "center": center.astype(np.float32),
        "height_m": human_height_m,
        "xy_extent_m": human_xy_extent_m,
        "base_radius_m": float(base_radius_m),
    }


def virtual_orbit_camera(
    base_camera: dict[str, Any],
    pivot: Any,
    yaw_deg: float,
    radius_scale: float,
    elevation_deg: float,
    radius_m: float,
    fov_y_deg: float,
    output_size: tuple[int, int],
    np: Any,
) -> dict[str, Any]:
    output_width, output_height = output_size
    yaw_rad = np.deg2rad(float(yaw_deg))
    elevation_rad = np.deg2rad(float(elevation_deg))
    radius = float(radius_m) * float(radius_scale)
    horizontal = radius * float(np.cos(elevation_rad))
    eye = np.asarray(
        [
            float(pivot[0]) + horizontal * float(np.sin(yaw_rad)),
            float(pivot[1]) - horizontal * float(np.cos(yaw_rad)),
            float(pivot[2]) + radius * float(np.sin(elevation_rad)),
        ],
        dtype=np.float32,
    )
    fov_y_rad = np.deg2rad(float(fov_y_deg))
    fy = 0.5 * float(output_height) / max(float(np.tan(fov_y_rad * 0.5)), 1e-6)
    orbit = dict(base_camera)
    orbit.update(
        {
            "w2c": look_at_w2c(eye, np.asarray(pivot, dtype=np.float32), np),
            "width": int(output_width),
            "height": int(output_height),
            "fx": float(fy),
            "fy": float(fy),
            "cx": float(output_width) * 0.5,
            "cy": float(output_height) * 0.5,
            "virtual_camera_fov_y_deg": float(fov_y_deg),
            "orbit_yaw_deg": float(yaw_deg),
            "orbit_elevation_deg": float(elevation_deg),
            "orbit_radius_scale": float(radius_scale),
            "orbit_radius_m": float(radius),
            "orbit_base_radius_m": float(radius_m),
            "orbit_pivot_world": [float(value) for value in np.asarray(pivot).tolist()],
            "camera_kind": "virtual_orbit",
        }
    )
    return orbit


def human_frame_fill_metrics(
    human_vertices: Any,
    camera: dict[str, Any],
    width: int,
    height: int,
    np: Any,
) -> dict[str, Any]:
    projected = project_vertices(human_vertices, camera, width, height, np)
    valid = projected[(projected[:, 2] > 0.02) & np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])]
    if len(valid) == 0:
        return {
            "human_frame_fill_ratio": 1.0,
            "human_frame_fill_width_ratio": 1.0,
            "human_frame_fill_height_ratio": 1.0,
            "human_frame_bbox_xyxy": None,
        }
    x0, y0 = valid[:, :2].min(axis=0)
    x1, y1 = valid[:, :2].max(axis=0)
    width_ratio = float((x1 - x0) / max(float(width), 1.0))
    height_ratio = float((y1 - y0) / max(float(height), 1.0))
    return {
        "human_frame_fill_ratio": float(max(width_ratio, height_ratio)),
        "human_frame_fill_width_ratio": width_ratio,
        "human_frame_fill_height_ratio": height_ratio,
        "human_frame_bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
    }


def target_faces_for_segment(human_faces: Any, segment_vertex_ids: list[int], np: Any) -> Any:
    if not segment_vertex_ids:
        return human_faces[:0]
    segment_mask = np.zeros(int(human_faces.max()) + 1, dtype=bool)
    segment_mask[np.asarray(segment_vertex_ids, dtype=np.int64)] = True
    face_mask = segment_mask[human_faces].any(axis=1)
    return human_faces[face_mask]


def visible_surface_stats(
    candidate_depth: Any,
    occluder_depth: Any,
    np: Any,
    eps_m: float,
) -> dict[str, Any]:
    projected_mask = np.isfinite(candidate_depth)
    visible_mask = projected_mask & (
        ~np.isfinite(occluder_depth)
        | (candidate_depth <= occluder_depth + float(eps_m))
    )
    projected_pixels = int(projected_mask.sum())
    visible_pixels = int(visible_mask.sum())
    return {
        "visible_pixel_count": visible_pixels,
        "projected_pixel_count": projected_pixels,
        "visible_over_projected_surface_ratio": (
            float(visible_pixels / projected_pixels) if projected_pixels else 0.0
        ),
    }


def add_surface_visibility_ratios(
    metrics: dict[str, Any],
    rendered: dict[str, Any],
    camera: dict[str, Any],
    image_size: tuple[int, int],
    target_faces: Any,
    eps_m: float,
) -> dict[str, Any]:
    np = rendered["deps"]["np"]
    scene_depth = rasterize_depth(
        rendered["scene_vertices"],
        rendered["scene_faces"],
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    human_depth = rasterize_depth(
        rendered["human_vertices"],
        rendered["human_faces"],
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    target_depth = rasterize_depth(
        rendered["human_vertices"],
        target_faces,
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    human_stats = visible_surface_stats(
        human_depth,
        scene_depth,
        np,
        eps_m,
    )
    target_stats = visible_surface_stats(
        target_depth,
        np.minimum(scene_depth, human_depth),
        np,
        eps_m,
    )
    target_self_stats = visible_surface_stats(
        target_depth,
        human_depth,
        np,
        eps_m,
    )
    human_ratio = float(human_stats["visible_over_projected_surface_ratio"])
    target_ratio = float(target_stats["visible_over_projected_surface_ratio"])
    target_self_visible_ratio = float(target_self_stats["visible_over_projected_surface_ratio"])
    if int(target_self_stats["projected_pixel_count"]):
        target_self_occluded_ratio = 1.0 - target_self_visible_ratio
    else:
        target_self_occluded_ratio = 1.0
    enriched = dict(metrics)
    enriched["human_visible_over_total_ratio"] = human_ratio
    enriched["target_visible_over_total_ratio"] = target_ratio
    enriched["human_visible_over_projected_surface_ratio"] = human_ratio
    enriched["target_visible_over_projected_surface_ratio"] = target_ratio
    enriched["human_visible_pixel_count"] = int(human_stats["visible_pixel_count"])
    enriched["human_projected_pixel_count"] = int(human_stats["projected_pixel_count"])
    enriched["target_visible_pixel_count"] = int(target_stats["visible_pixel_count"])
    enriched["target_projected_pixel_count"] = int(target_stats["projected_pixel_count"])
    enriched["target_self_visible_over_projected_surface_ratio"] = target_self_visible_ratio
    enriched["target_self_occluded_over_projected_surface_ratio"] = target_self_occluded_ratio
    enriched["target_self_visible_pixel_count"] = int(target_self_stats["visible_pixel_count"])
    enriched["target_self_projected_pixel_count"] = int(target_self_stats["projected_pixel_count"])
    enriched["visibility_metric"] = "rendered_surface_pixels"
    enriched["visibility_occlusion_model"] = "cropped_scene_depth_plus_human_self_depth"
    enriched["occlusion_depth_eps_m"] = float(eps_m)
    enriched["selection_score"] = float(4.0 * target_ratio + 2.0 * human_ratio)
    return enriched


def add_human_visibility_metrics(
    metrics: dict[str, Any],
    rendered: dict[str, Any],
    camera: dict[str, Any],
    image_size: tuple[int, int],
    eps_m: float,
) -> dict[str, Any]:
    np = rendered["deps"]["np"]
    scene_depth = rasterize_depth(
        rendered["scene_vertices"],
        rendered["scene_faces"],
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    human_depth = rasterize_depth(
        rendered["human_vertices"],
        rendered["human_faces"],
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    human_stats = visible_surface_stats(
        human_depth,
        scene_depth,
        np,
        eps_m,
    )
    human_ratio = float(human_stats["visible_over_projected_surface_ratio"])
    height, width = image_size
    coverage = float(int(human_stats["visible_pixel_count"]) / max(width * height, 1))
    enriched = dict(metrics)
    enriched["human_visible_over_total_ratio"] = human_ratio
    enriched["human_visible_over_projected_surface_ratio"] = human_ratio
    enriched["human_visible_pixel_count"] = int(human_stats["visible_pixel_count"])
    enriched["human_projected_pixel_count"] = int(human_stats["projected_pixel_count"])
    enriched["human_visible_image_coverage_ratio"] = coverage
    enriched["visibility_metric"] = "rendered_surface_pixels"
    enriched["visibility_occlusion_model"] = "cropped_scene_depth"
    enriched["occlusion_depth_eps_m"] = float(eps_m)
    enriched["selection_score"] = float(4.0 * human_ratio + coverage)
    return enriched


def visual_segment_id(segment_id: str | None, part_name: str | None) -> str | None:
    if segment_id:
        segment_key = str(segment_id)
        if segment_key in VISUAL_SEGMENT_ALIASES:
            return VISUAL_SEGMENT_ALIASES[segment_key]
        return segment_key
    if part_name:
        return slugify(part_name)
    return None


def segment_vertices(segmentation: dict[str, Any], segment_id: str | None,
                     part_name: str | None) -> tuple[str | None, list[int]]:
    segments = segmentation.get("segments", {})
    candidates = []
    visual_id = visual_segment_id(segment_id, part_name)
    if visual_id:
        candidates.append(visual_id)
    if segment_id:
        candidates.append(str(segment_id))
    if part_name:
        part_slug = slugify(part_name)
        candidates.extend([part_slug, f"{part_slug}_inner", f"{part_slug}_bottom", f"{part_slug}_contact"])
    for candidate in candidates:
        if candidate in segments and isinstance(segments[candidate], list):
            return candidate, [int(index) for index in segments[candidate]]
    return visual_id, []


def bbox_from_projected(
    projected: Any,
    width: int,
    height: int,
    padding_frac: float,
    min_size: int,
    np: Any,
    aspect_ratio: float | None = None,
) -> list[int] | None:
    valid = projected[(projected[:, 2] > 0.02) & np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])]
    if len(valid) == 0:
        return None
    x0, y0 = valid[:, :2].min(axis=0)
    x1, y1 = valid[:, :2].max(axis=0)
    box_w = max(float(x1 - x0), 1.0)
    box_h = max(float(y1 - y0), 1.0)
    pad = max(box_w, box_h) * padding_frac
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    final_w = max(box_w + 2.0 * pad, float(min_size))
    final_h = max(box_h + 2.0 * pad, float(min_size))
    if aspect_ratio is not None and aspect_ratio > 0:
        current_aspect = final_w / max(final_h, 1.0)
        if current_aspect < aspect_ratio:
            final_w = final_h * aspect_ratio
        else:
            final_h = final_w / aspect_ratio
    bx0 = int(np.floor(cx - final_w * 0.5))
    by0 = int(np.floor(cy - final_h * 0.5))
    bx1 = int(np.ceil(cx + final_w * 0.5))
    by1 = int(np.ceil(cy + final_h * 0.5))
    return [bx0, by0, bx1, by1]


def contact_crop_min_size(part_name: str | None, segment_id: str | None, default_min_size: int) -> int:
    label = f"{part_name or ''} {segment_id or ''}".lower()
    if "hand" in label:
        return min(default_min_size, 180)
    if "foot" in label:
        return min(default_min_size, 240)
    if "hip" in label:
        return max(default_min_size, 360)
    return default_min_size


def optical_zoom_camera(
    camera: dict[str, Any],
    bbox: list[int],
    base_width: int,
    base_height: int,
    output_size: tuple[int, int],
    fill_frac: float,
) -> dict[str, Any]:
    output_width, output_height = output_size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5

    fx_base = float(camera["fx"]) * (float(base_width) / float(camera["width"]))
    fy_base = float(camera["fy"]) * (float(base_height) / float(camera["height"]))
    cx_base = float(camera["cx"]) * (float(base_width) / float(camera["width"]))
    cy_base = float(camera["cy"]) * (float(base_height) / float(camera["height"]))

    zoom = min(
        (float(output_width) * fill_frac) / bbox_width,
        (float(output_height) * fill_frac) / bbox_height,
    )
    zoom_camera = dict(camera)
    zoom_camera.update(
        {
            "width": int(output_width),
            "height": int(output_height),
            "fx": float(fx_base * zoom),
            "fy": float(fy_base * zoom),
            "cx": float(output_width * 0.5 + (cx_base - center_x) * zoom),
            "cy": float(output_height * 0.5 + (cy_base - center_y) * zoom),
            "optical_zoom_factor": float(zoom),
        }
    )
    return zoom_camera


def save_optical_zoom_render_with_visibility(
    rendered: dict[str, Any],
    bbox: list[int],
    path: Path,
    output_size: tuple[int, int],
    fill_frac: float,
    target_faces: Any,
    args: argparse.Namespace,
    base_camera: dict[str, Any],
    base_width: int,
    base_height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    zoom_camera = optical_zoom_camera(base_camera, bbox, base_width, base_height, output_size, fill_frac)
    image = composite_scene_and_human(rendered, zoom_camera, (output_size[1], output_size[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered["deps"]["Image"].fromarray(image).save(path)
    metadata = {
        "path": str(path),
        "output_size": [int(output_size[0]), int(output_size[1])],
        "optical_zoom_factor": zoom_camera["optical_zoom_factor"],
        "fx": zoom_camera["fx"],
        "fy": zoom_camera["fy"],
        "cx": zoom_camera["cx"],
        "cy": zoom_camera["cy"],
        "base_width": int(base_width),
        "base_height": int(base_height),
    }
    visibility = add_surface_visibility_ratios(
        {},
        rendered,
        zoom_camera,
        (output_size[1], output_size[0]),
        target_faces,
        float(args.occlusion_depth_eps_m),
    )
    return metadata, visibility


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def adaptive_yaws(args: argparse.Namespace) -> list[float]:
    step = max(float(args.adaptive_yaw_step_deg), 1.0)
    count = max(1, int(round(360.0 / step)))
    return [float(-180.0 + index * step) for index in range(count)]


def adaptive_orbit_camera(
    base_camera: dict[str, Any],
    pivot: Any,
    yaw: float,
    elevation: float,
    human_bbox: dict[str, Any],
    human_vertices: Any,
    args: argparse.Namespace,
    np: Any,
    output_size: tuple[int, int],
) -> tuple[dict[str, Any], float]:
    width, height = output_size
    initial = virtual_orbit_camera(
        base_camera,
        pivot,
        yaw,
        1.0,
        elevation,
        human_bbox["base_radius_m"],
        float(args.virtual_camera_fov_y_deg),
        output_size,
        np,
    )
    fill = human_frame_fill_metrics(human_vertices, initial, width, height, np)
    raw_fill = max(float(fill["human_frame_fill_ratio"]), 1e-6)
    target_fill = clamp(float(args.adaptive_radius_target_fill_ratio), 0.1, 0.95)
    radius_scale = clamp(
        raw_fill / target_fill,
        float(args.adaptive_radius_min_scale),
        float(args.adaptive_radius_max_scale),
    )
    camera = virtual_orbit_camera(
        base_camera,
        pivot,
        yaw,
        radius_scale,
        elevation,
        human_bbox["base_radius_m"],
        float(args.virtual_camera_fov_y_deg),
        output_size,
        np,
    )
    return camera, float(radius_scale)


def hard_reject_contact(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons = []
    if float(metrics.get("human_visible_over_total_ratio", 0.0)) < float(args.hard_min_human_visible_ratio):
        reasons.append("hard_low_human_visibility")
    if float(metrics.get("target_visible_over_total_ratio", 0.0)) < float(args.hard_min_target_visible_ratio):
        reasons.append("hard_low_target_visibility")
    if int(metrics.get("target_visible_pixel_count", 0)) < int(args.hard_min_target_visible_pixels):
        reasons.append("hard_low_target_visible_pixels")
    if float(metrics.get("human_frame_fill_ratio", 1.0)) > float(args.hard_max_human_frame_fill_ratio):
        reasons.append("hard_human_frame_too_full")
    return reasons


def hard_reject_global(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons = []
    if float(metrics.get("human_visible_over_total_ratio", 0.0)) < float(args.hard_min_human_visible_ratio):
        reasons.append("hard_low_human_visibility")
    if float(metrics.get("human_frame_fill_ratio", 1.0)) > float(args.hard_max_human_frame_fill_ratio):
        reasons.append("hard_human_frame_too_full")
    return reasons


def overfill_penalty(fill_ratio: float, target_ratio: float) -> float:
    if fill_ratio <= target_ratio:
        return 0.0
    return float((fill_ratio - target_ratio) / max(1.0 - target_ratio, 1e-6))


def contact_rank_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    target_visible = float(metrics.get("target_visible_over_total_ratio", 0.0))
    human_visible = float(metrics.get("human_visible_over_total_ratio", 0.0))
    self_occluded = float(metrics.get("target_self_occluded_over_projected_surface_ratio", 1.0))
    fill = float(metrics.get("human_frame_fill_ratio", 1.0))
    target_pixels = int(metrics.get("target_visible_pixel_count", 0))
    width_ratio = float(metrics.get("human_frame_fill_width_ratio", 0.0))
    height_ratio = float(metrics.get("human_frame_fill_height_ratio", 0.0))
    target_pixel_bonus = min(float(target_pixels) / 2500.0, 1.0)
    coverage_bonus = min(max(width_ratio, height_ratio), 1.0)
    return float(
        4.0 * target_visible
        + 2.0 * human_visible
        - 2.0 * self_occluded
        - overfill_penalty(fill, float(args.adaptive_radius_target_fill_ratio))
        + 0.5 * target_pixel_bonus
        + 0.25 * coverage_bonus
    )


def global_rank_score(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    human_visible = float(metrics.get("human_visible_over_total_ratio", 0.0))
    coverage = float(metrics.get("human_visible_image_coverage_ratio", 0.0))
    fill = float(metrics.get("human_frame_fill_ratio", 1.0))
    return float(
        4.0 * human_visible
        + coverage
        - overfill_penalty(fill, float(args.adaptive_radius_target_fill_ratio))
    )


def angular_separation_schedule(args: argparse.Namespace) -> list[float]:
    if args.relaxed_angular_separations_deg:
        values = parse_float_list(args.relaxed_angular_separations_deg)
    else:
        start = float(args.min_view_angular_separation_deg)
        values = [start, min(start, 10.0), min(start, 5.0), 0.0]
    schedule: list[float] = []
    for value in values:
        value = float(value)
        if value not in schedule:
            schedule.append(value)
    return schedule


def summarize_candidate_diagnostics(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    selected = 0
    hard_rejected = 0
    for item in diagnostics:
        if item.get("selection_status") == "selected":
            selected += 1
        if item.get("selection_status") == "hard_rejected":
            hard_rejected += 1
        for reason in item.get("rejection_reasons", []):
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "candidate_count": len(diagnostics),
        "selected_count": selected,
        "hard_rejected_count": hard_rejected,
        "ranked_candidate_count": len(diagnostics) - hard_rejected,
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
    }


def select_ranked_candidates(
    candidates: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates.sort(key=lambda item: item["rank_score"], reverse=True)
    best_views: list[dict[str, Any]] = []
    best_sep = angular_separation_schedule(args)[-1]

    for min_sep in angular_separation_schedule(args):
        views = []
        selected_yaws: list[float] = []
        for candidate in candidates:
            if len(views) >= max(1, int(args.contact_view_count)):
                break
            yaw = float(candidate["yaw_offset_deg"])
            if any(abs(((yaw - used + 180.0) % 360.0) - 180.0) < min_sep for used in selected_yaws):
                continue
            candidate = dict(candidate)
            candidate["view_index"] = len(views)
            candidate["selection_min_angular_separation_deg"] = float(min_sep)
            candidate["view_name"] = (
                f"view_{len(views):02d}_ranked_yaw_{int(round(candidate['yaw_offset_deg'])):+d}"
                f"_elev_{int(round(candidate['elevation_deg'])):+d}_r_{candidate['orbit_radius_scale']:.2f}"
            )
            views.append(candidate)
            selected_yaws.append(yaw)
        best_views = views
        best_sep = min_sep
        if len(views) >= max(1, int(args.contact_view_count)):
            break

    for view in best_views:
        diagnostics[int(view["candidate_id"])]["selection_status"] = "selected"
        diagnostics[int(view["candidate_id"])]["view_name"] = view["view_name"]
        diagnostics[int(view["candidate_id"])]["selection_min_angular_separation_deg"] = float(best_sep)
    return best_views


def contact_visibility_status(views: list[dict[str, Any]], segment_ids: list[int]) -> str:
    if views:
        return "selected_views"
    if not segment_ids:
        return "missing_segment_vertices"
    return "no_visible_contact_views"


def select_contact_view_cameras(
    rendered: dict[str, Any],
    base_camera: dict[str, Any],
    human_vertices: Any,
    segment_vertex_ids: list[int],
    target_faces: Any,
    args: argparse.Namespace,
    np: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_width, base_height = parse_size(args.view_planner_image_size)
    human_bbox = human_bbox_info(human_vertices, np)
    segment_vertices_np = human_vertices[np.asarray(segment_vertex_ids, dtype=np.int64)]
    segment_center = segment_vertices_np.mean(axis=0)
    contact_pivot = 0.65 * segment_center + 0.35 * human_bbox["center"]
    candidates = []
    diagnostics: list[dict[str, Any]] = []

    for yaw in adaptive_yaws(args):
        for elevation in parse_float_list(args.candidate_elevations_deg):
            camera, radius_scale = adaptive_orbit_camera(
                base_camera,
                contact_pivot,
                yaw,
                elevation,
                human_bbox,
                human_vertices,
                args,
                np,
                (base_width, base_height),
            )
            visibility = add_surface_visibility_ratios(
                {},
                rendered,
                camera,
                (base_height, base_width),
                target_faces,
                float(args.occlusion_depth_eps_m),
            )
            visibility.update(human_frame_fill_metrics(human_vertices, camera, base_width, base_height, np))
            reasons = hard_reject_contact(visibility, args)
            rank_score = contact_rank_score(visibility, args)
            visibility["rank_score"] = rank_score
            candidate_id = len(diagnostics)
            diagnostics.append(
                {
                    "candidate_id": candidate_id,
                    "camera_kind": "ranked_virtual_orbit",
                    "yaw_offset_deg": float(yaw),
                    "elevation_deg": float(elevation),
                    "orbit_radius_scale": float(radius_scale),
                    "orbit_base_radius_m": float(human_bbox["base_radius_m"]),
                    "orbit_radius_m": float(human_bbox["base_radius_m"]) * float(radius_scale),
                    "pivot_world": [float(value) for value in contact_pivot.tolist()],
                    "rank_score": rank_score,
                    "visibility": visibility,
                    "rejection_reasons": reasons,
                    "selection_status": "hard_rejected" if reasons else "ranked_candidate",
                }
            )
            if reasons:
                continue
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "camera_kind": "ranked_virtual_orbit",
                    "yaw_offset_deg": float(yaw),
                    "elevation_deg": float(elevation),
                    "camera": camera,
                    "pivot_world": [float(value) for value in contact_pivot.tolist()],
                    "orbit_radius_scale": float(radius_scale),
                    "orbit_base_radius_m": float(human_bbox["base_radius_m"]),
                    "orbit_radius_m": float(human_bbox["base_radius_m"]) * float(radius_scale),
                    "human_bbox_center_world": [float(value) for value in human_bbox["center"].tolist()],
                    "segment_center_world": [float(value) for value in segment_center.tolist()],
                    "human_height_m": float(human_bbox["height_m"]),
                    "human_xy_extent_m": float(human_bbox["xy_extent_m"]),
                    "rank_score": rank_score,
                    "visibility": visibility,
                    "selection_status": "selected",
                }
            )

    views = select_ranked_candidates(candidates, diagnostics, args)
    return views, diagnostics[: int(args.candidate_diagnostics_limit)], summarize_candidate_diagnostics(diagnostics)


def build_view_evidence(
    rendered: dict[str, Any],
    view: dict[str, Any],
    segment_vertex_ids: list[int],
    target_faces: Any,
    edge_dir: Path,
    part_name: str | None,
    segment_id: str | None,
    args: argparse.Namespace,
    np: Any,
) -> dict[str, Any]:
    base_camera = view["camera"]
    base_width, base_height = parse_size(args.render_image_size)
    human_vertices = rendered["human_vertices"]
    segment_vertices_np = human_vertices[np.asarray(segment_vertex_ids, dtype=np.int64)]
    human_projected = project_vertices(human_vertices, base_camera, base_width, base_height, np)
    segment_projected = project_vertices(segment_vertices_np, base_camera, base_width, base_height, np)

    context_size = parse_size(args.contact_context_output_size)
    local_size = parse_size(args.contact_local_output_size)
    context_bbox = bbox_from_projected(
        human_projected,
        base_width,
        base_height,
        float(args.context_padding_frac),
        1,
        np,
        aspect_ratio=float(context_size[0]) / float(context_size[1]),
    )
    local_bbox = bbox_from_projected(
        segment_projected,
        base_width,
        base_height,
        float(args.crop_padding_frac),
        contact_crop_min_size(part_name, segment_id, int(args.crop_min_size_px)),
        np,
        aspect_ratio=float(local_size[0]) / float(local_size[1]),
    )
    if context_bbox is None:
        context_bbox = [0, 0, base_width, base_height]
    if local_bbox is None:
        local_bbox = context_bbox

    view_index = int(view["view_index"])
    context_path = edge_dir / f"view_{view_index:02d}_context.png"
    local_path = edge_dir / f"view_{view_index:02d}_local_contact.png"
    context_render, final_context_visibility = save_optical_zoom_render_with_visibility(
        rendered,
        context_bbox,
        context_path,
        context_size,
        fill_frac=float(args.contact_context_fill_frac),
        target_faces=target_faces,
        args=args,
        base_camera=base_camera,
        base_width=base_width,
        base_height=base_height,
    )
    local_render, final_local_visibility = save_optical_zoom_render_with_visibility(
        rendered,
        local_bbox,
        local_path,
        local_size,
        fill_frac=float(args.contact_local_fill_frac),
        target_faces=target_faces,
        args=args,
        base_camera=base_camera,
        base_width=base_width,
        base_height=base_height,
    )
    return {
        "view_index": int(view["view_index"]),
        "view_name": view["view_name"],
        "camera_kind": view["camera_kind"],
        "yaw_offset_deg": float(view["yaw_offset_deg"]),
        "elevation_deg": float(view.get("elevation_deg", 0.0)),
        "pivot_world": view.get("pivot_world"),
        "orbit_radius_scale": view.get("orbit_radius_scale"),
        "orbit_base_radius_m": view.get("orbit_base_radius_m"),
        "orbit_radius_m": view.get("orbit_radius_m"),
        "rank_score": view.get("rank_score"),
        "selection_min_angular_separation_deg": view.get("selection_min_angular_separation_deg"),
        "human_bbox_center_world": view.get("human_bbox_center_world"),
        "segment_center_world": view.get("segment_center_world"),
        "human_height_m": view.get("human_height_m"),
        "human_xy_extent_m": view.get("human_xy_extent_m"),
        "virtual_camera_fov_y_deg": float(args.virtual_camera_fov_y_deg),
        "candidate_camera": {
            "camera_kind": view["camera_kind"],
            "yaw_offset_deg": float(view["yaw_offset_deg"]),
            "elevation_deg": float(view.get("elevation_deg", 0.0)),
            "radius_scale": view.get("orbit_radius_scale"),
            "base_radius_m": view.get("orbit_base_radius_m"),
            "radius_m": view.get("orbit_radius_m"),
            "pivot_world": view.get("pivot_world"),
            "fov_y_deg": float(args.virtual_camera_fov_y_deg),
        },
        "planner_visibility": view["visibility"],
        "selection_status": view.get("selection_status", "selected"),
        "final_context_visibility": final_context_visibility,
        "final_local_visibility": final_local_visibility,
        "context_bbox_xyxy": context_bbox,
        "local_bbox_xyxy": local_bbox,
        "images": {
            "context": str(context_path),
            "local_contact": str(local_path),
        },
        "rendering": {
            "context": context_render,
            "local_contact": local_render,
        },
    }


def select_global_view_cameras(
    rendered: dict[str, Any],
    base_camera: dict[str, Any],
    args: argparse.Namespace,
    np: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base_width, base_height = parse_size(args.view_planner_image_size)
    human_vertices = rendered["human_vertices"]
    human_bbox = human_bbox_info(human_vertices, np)
    pivot = human_bbox["center"]

    candidates = []
    diagnostics: list[dict[str, Any]] = []

    for yaw in adaptive_yaws(args):
        for elevation in parse_float_list(args.candidate_elevations_deg):
            camera, radius_scale = adaptive_orbit_camera(
                base_camera,
                pivot,
                yaw,
                elevation,
                human_bbox,
                human_vertices,
                args,
                np,
                (base_width, base_height),
            )
            visibility = add_human_visibility_metrics(
                {},
                rendered,
                camera,
                (base_height, base_width),
                float(args.occlusion_depth_eps_m),
            )
            visibility.update(human_frame_fill_metrics(human_vertices, camera, base_width, base_height, np))
            reasons = hard_reject_global(visibility, args)
            rank_score = global_rank_score(visibility, args)
            visibility["rank_score"] = rank_score
            candidate_id = len(diagnostics)
            diagnostics.append(
                {
                    "candidate_id": candidate_id,
                    "camera_kind": "ranked_virtual_orbit",
                    "yaw_offset_deg": float(yaw),
                    "elevation_deg": float(elevation),
                    "orbit_radius_scale": float(radius_scale),
                    "orbit_base_radius_m": float(human_bbox["base_radius_m"]),
                    "orbit_radius_m": float(human_bbox["base_radius_m"]) * float(radius_scale),
                    "pivot_world": [float(value) for value in pivot.tolist()],
                    "rank_score": rank_score,
                    "visibility": visibility,
                    "rejection_reasons": reasons,
                    "selection_status": "hard_rejected" if reasons else "ranked_candidate",
                }
            )
            if reasons:
                continue
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "camera_kind": "ranked_virtual_orbit",
                    "yaw_offset_deg": float(yaw),
                    "elevation_deg": float(elevation),
                    "camera": camera,
                    "pivot_world": [float(value) for value in pivot.tolist()],
                    "orbit_radius_scale": float(radius_scale),
                    "orbit_base_radius_m": float(human_bbox["base_radius_m"]),
                    "orbit_radius_m": float(human_bbox["base_radius_m"]) * float(radius_scale),
                    "human_bbox_center_world": [float(value) for value in human_bbox["center"].tolist()],
                    "human_height_m": float(human_bbox["height_m"]),
                    "human_xy_extent_m": float(human_bbox["xy_extent_m"]),
                    "rank_score": rank_score,
                    "visibility": visibility,
                    "selection_status": "selected",
                }
            )

    views = select_ranked_candidates(candidates, diagnostics, args)
    return views, diagnostics[: int(args.candidate_diagnostics_limit)], summarize_candidate_diagnostics(diagnostics)


def build_global_view_evidence(
    rendered: dict[str, Any],
    view: dict[str, Any],
    views_dir: Path,
    args: argparse.Namespace,
    np: Any,
) -> dict[str, Any]:
    base_camera = view["camera"]
    base_width, base_height = parse_size(args.render_image_size)
    human_projected = project_vertices(rendered["human_vertices"], base_camera, base_width, base_height, np)
    output_size = parse_size(args.contact_context_output_size)
    bbox = bbox_from_projected(
        human_projected,
        base_width,
        base_height,
        float(args.context_padding_frac),
        1,
        np,
        aspect_ratio=float(output_size[0]) / float(output_size[1]),
    )
    if bbox is None:
        bbox = [0, 0, base_width, base_height]

    image_path = views_dir / f"view_{int(view['view_index']):02d}.png"
    zoom_camera = optical_zoom_camera(
        base_camera,
        bbox,
        base_width,
        base_height,
        output_size,
        float(args.contact_context_fill_frac),
    )
    image = composite_scene_and_human(rendered, zoom_camera, (output_size[1], output_size[0]))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    rendered["deps"]["Image"].fromarray(image).save(image_path)
    final_visibility = add_human_visibility_metrics(
        {},
        rendered,
        zoom_camera,
        (output_size[1], output_size[0]),
        float(args.occlusion_depth_eps_m),
    )
    return {
        "view_index": int(view["view_index"]),
        "view_name": view["view_name"],
        "camera_kind": view["camera_kind"],
        "yaw_offset_deg": float(view["yaw_offset_deg"]),
        "elevation_deg": float(view.get("elevation_deg", 0.0)),
        "pivot_world": view.get("pivot_world"),
        "orbit_radius_scale": view.get("orbit_radius_scale"),
        "orbit_base_radius_m": view.get("orbit_base_radius_m"),
        "orbit_radius_m": view.get("orbit_radius_m"),
        "rank_score": view.get("rank_score"),
        "selection_min_angular_separation_deg": view.get("selection_min_angular_separation_deg"),
        "human_bbox_center_world": view.get("human_bbox_center_world"),
        "human_height_m": view.get("human_height_m"),
        "human_xy_extent_m": view.get("human_xy_extent_m"),
        "virtual_camera_fov_y_deg": float(args.virtual_camera_fov_y_deg),
        "candidate_camera": {
            "camera_kind": view["camera_kind"],
            "yaw_offset_deg": float(view["yaw_offset_deg"]),
            "elevation_deg": float(view.get("elevation_deg", 0.0)),
            "radius_scale": view.get("orbit_radius_scale"),
            "base_radius_m": view.get("orbit_base_radius_m"),
            "radius_m": view.get("orbit_radius_m"),
            "pivot_world": view.get("pivot_world"),
            "fov_y_deg": float(args.virtual_camera_fov_y_deg),
        },
        "planner_visibility": view["visibility"],
        "selection_status": view.get("selection_status", "selected"),
        "final_visibility": final_visibility,
        "bbox_xyxy": bbox,
        "image": str(image_path),
        "rendering": {
            "path": str(image_path),
            "output_size": [int(output_size[0]), int(output_size[1])],
            "optical_zoom_factor": zoom_camera["optical_zoom_factor"],
            "fx": zoom_camera["fx"],
            "fy": zoom_camera["fy"],
            "cx": zoom_camera["cx"],
            "cy": zoom_camera["cy"],
            "base_width": int(base_width),
            "base_height": int(base_height),
        },
    }


def view_summary(view: dict[str, Any]) -> dict[str, Any]:
    visibility = view.get("planner_visibility", {})
    return {
        "view_index": int(view.get("view_index", 0)),
        "yaw_deg": view.get("yaw_offset_deg"),
        "elevation_deg": view.get("elevation_deg"),
        "radius_scale": view.get("orbit_radius_scale"),
        "rank_score": view.get("rank_score"),
        "human_visible_ratio": visibility.get("human_visible_over_total_ratio"),
        "target_visible_ratio": visibility.get("target_visible_over_total_ratio"),
    }


def render_params_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "render_image_size": args.render_image_size,
        "contact_context_output_size": args.contact_context_output_size,
        "contact_local_output_size": args.contact_local_output_size,
        "view_planner_image_size": args.view_planner_image_size,
        "contact_view_count": int(args.contact_view_count),
        "candidate_elevations_deg": parse_float_list(args.candidate_elevations_deg),
        "adaptive_yaw_step_deg": float(args.adaptive_yaw_step_deg),
        "adaptive_radius_target_fill_ratio": float(args.adaptive_radius_target_fill_ratio),
        "adaptive_radius_min_scale": float(args.adaptive_radius_min_scale),
        "adaptive_radius_max_scale": float(args.adaptive_radius_max_scale),
        "virtual_camera_fov_y_deg": float(args.virtual_camera_fov_y_deg),
        "min_view_angular_separation_deg": float(args.min_view_angular_separation_deg),
        "angular_separation_schedule_deg": angular_separation_schedule(args),
        "hard_min_human_visible_ratio": float(args.hard_min_human_visible_ratio),
        "hard_min_target_visible_ratio": float(args.hard_min_target_visible_ratio),
        "hard_min_target_visible_pixels": int(args.hard_min_target_visible_pixels),
        "hard_max_human_frame_fill_ratio": float(args.hard_max_human_frame_fill_ratio),
    }


def render_views(
    metrics: dict[str, Any],
    input_scene: dict[str, Any],
    optimizer_root: Path,
    outdir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    deps = import_render_deps()
    np = deps["np"]
    Image = deps["Image"]

    evidence_dir = outdir / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    contact_dir = evidence_dir / "contact"
    global_dir = evidence_dir / "global"
    contact_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)

    camera = read_camera(input_scene, Path(args.scannet_root).resolve())
    defaults = default_paths(args.interaction_name)
    contact_spec_path = resolve_path(args.contact_spec, defaults["contact_spec"])
    contact_canvas_image = resolve_path(args.contact_canvas_image, defaults["contact_canvas_image"])
    contact_crop_camera = read_contact_crop_camera(
        camera,
        contact_spec_path,
        contact_canvas_image,
        Image,
    )
    human_mesh_path = optimizer_root / "meshes" / "frame_0000_world.ply"
    log("load", f"source ScanNet scene mesh: {camera['scene_mesh_path']}")
    log("load", f"contact spec: {contact_spec_path}")
    log("load", f"contact crop canvas: {contact_canvas_image}")
    log("load", f"SMPL-X segmentation: {Path(args.smpl_seg_json).resolve()}")
    segmentation = load_json(Path(args.smpl_seg_json).resolve())

    rendered = render_scene_with_human(camera, contact_crop_camera, human_mesh_path, args)

    selected_global_views, _global_diagnostics, global_summary = select_global_view_cameras(
        rendered,
        camera,
        args,
        np,
    )
    global_views = [
        build_global_view_evidence(rendered, view, global_dir, args, np)
        for view in selected_global_views
    ]
    log("evidence", f"global pose/penetration views={len(global_views)}")

    contact_entries = []
    for edge in metrics.get("contact", {}).get("edges", []):
        index = int(edge["index"])
        part_name = edge.get("moving_part_name")
        segment_id = edge.get("moving_segment_id")
        target = edge.get("fixed_entity_name") or edge.get("fixed_part_name")
        _crop_segment_id, segment_ids = segment_vertices(segmentation, segment_id, part_name)
        edge_slug = f"edge_{index:02d}_{slugify(part_name or 'part')}_to_{slugify(target or 'target')}"
        edge_dir = contact_dir / edge_slug
        edge_dir.mkdir(parents=True, exist_ok=True)
        views: list[dict[str, Any]] = []
        visibility_status = "missing_segment"
        if not segment_ids:
            log("warn", f"edge={index} no segmentation vertices for {segment_id or part_name}")
        else:
            target_faces = target_faces_for_segment(rendered["human_faces"], segment_ids, np)
            highlighted_rendered = {
                **rendered,
                "human_colors": highlight_vertex_colors(rendered["human_colors"], segment_ids, np),
            }
            selected_views, _diagnostics, _summary = select_contact_view_cameras(
                highlighted_rendered,
                camera,
                rendered["human_vertices"],
                segment_ids,
                target_faces,
                args,
                np,
            )
            for view in selected_views:
                views.append(
                    build_view_evidence(
                        highlighted_rendered,
                        view,
                        segment_ids,
                        target_faces,
                        edge_dir,
                        part_name,
                        segment_id,
                        args,
                        np,
                    )
                )
            visibility_status = contact_visibility_status(views, segment_ids)

        contact_entries.append(
            {
                "edge_index": index,
                "body_part": part_name,
                "target": target,
                "view_count": len(views),
                "visibility_status": visibility_status,
                "deterministic_pass": edge.get("pass"),
                "contact_distance_m": edge.get("nocontact_distance_m"),
                "views": [view_summary(view) for view in views],
            }
        )
        log(
            "evidence",
            f"edge={index} part={part_name} target={target} "
            f"visibility={visibility_status} views={len(views)}",
        )

    render_summary = {
        "interaction_name": metrics.get("interaction_name"),
        "render_mode": "ranked_virtual_orbit_scannet_crop",
        "global_view_count": len(global_views),
        "global_views": [view_summary(view) for view in global_views],
        "contact_edges": contact_entries,
        "render_params": render_params_summary(args),
        "global_selection": global_summary,
    }
    save_json(evidence_dir / "render_summary.json", render_summary)
    log("evidence", f"wrote render summary: {evidence_dir / 'render_summary.json'}")
    return render_summary


def load_render_context(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    defaults = default_paths(args.interaction_name)
    optimizer_root = resolve_path(args.optimizer_output_root, defaults["optimizer_output_root"])
    sig_json_path = resolve_path(args.sig_json, defaults["sig_json"])
    input_scene_json_path = resolve_path(args.input_scene_json, defaults["input_scene_json"])
    outdir = resolve_path(args.outdir, defaults["outdir"])
    validate_required_inputs(optimizer_root, sig_json_path, input_scene_json_path)
    log("load", f"interaction={args.interaction_name} outdir={outdir}")
    log("load", f"SIG: {sig_json_path}")
    log("load", f"input scene: {input_scene_json_path}")
    log("load", f"optimizer summary: {optimizer_root / 'alignment_summary.json'}")
    sig_payload = load_json(sig_json_path)
    input_scene = load_json(input_scene_json_path)
    alignment_summary = load_json(optimizer_root / "alignment_summary.json")
    contact = collect_contact_metrics(alignment_summary, args)
    metrics = {
        "interaction_name": alignment_summary.get("interaction_name", args.interaction_name),
        "scene_id": alignment_summary.get("scene_id"),
        "interaction": sig_payload.get("interaction", ""),
        "contact": contact,
    }
    return metrics, input_scene, optimizer_root, outdir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render static-scene verification evidence views.")
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--optimizer-output-root", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--input-scene-json", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--contact-spec", default=None)
    parser.add_argument("--contact-canvas-image", default=None)
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET_ROOT))
    parser.add_argument("--smpl-seg-json", default=str(DEFAULT_SMPL_SEG_JSON))
    parser.add_argument("--contact-threshold-m", type=float, default=0.05)
    parser.add_argument("--render-image-size", default="480x480")
    parser.add_argument("--render-device", default="cuda:0")
    parser.add_argument("--crop-padding-frac", type=float, default=1.75)
    parser.add_argument("--crop-min-size-px", type=int, default=320)
    parser.add_argument("--context-padding-frac", type=float, default=0.35)
    parser.add_argument("--contact-view-count", type=int, default=6)
    parser.add_argument("--contact-context-output-size", default="480x480")
    parser.add_argument("--contact-local-output-size", default="480x480")
    parser.add_argument("--contact-context-fill-frac", type=float, default=0.94)
    parser.add_argument("--contact-local-fill-frac", type=float, default=0.94)
    parser.add_argument("--view-planner-image-size", default="480x480")
    parser.add_argument("--candidate-elevations-deg", default="-10,0,10,20")
    parser.add_argument("--adaptive-yaw-step-deg", type=float, default=30.0)
    parser.add_argument("--adaptive-radius-target-fill-ratio", type=float, default=0.78)
    parser.add_argument("--adaptive-radius-min-scale", type=float, default=0.80)
    parser.add_argument("--adaptive-radius-max-scale", type=float, default=2.00)
    parser.add_argument("--candidate-diagnostics-limit", type=int, default=500)
    parser.add_argument("--hard-min-human-visible-ratio", type=float, default=0.20)
    parser.add_argument("--hard-min-target-visible-ratio", type=float, default=0.02)
    parser.add_argument("--hard-min-target-visible-pixels", type=int, default=25)
    parser.add_argument("--hard-max-human-frame-fill-ratio", type=float, default=0.95)
    parser.add_argument("--relaxed-angular-separations-deg", default=None)
    parser.add_argument("--virtual-camera-fov-y-deg", type=float, default=50.0)
    parser.add_argument("--min-view-angular-separation-deg", type=float, default=15.0)
    parser.add_argument("--occlusion-depth-eps-m", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, input_scene, optimizer_root, outdir = load_render_context(args)
    outdir.mkdir(parents=True, exist_ok=True)
    render_summary = render_views(metrics, input_scene, optimizer_root, outdir, args)
    log(
        "summary",
        f"rendered global_views={render_summary['global_view_count']} "
        f"contact_edges={len(render_summary['contact_edges'])}",
    )


def run_cli() -> None:
    try:
        main()
    except Exception as error:
        raise SystemExit(f"[error] {error}") from None


if __name__ == "__main__":
    run_cli()
