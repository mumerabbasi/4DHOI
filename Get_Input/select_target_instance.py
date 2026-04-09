from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

try:
    import torch
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection
except Exception as exc:  # pragma: no cover - import guard for env mismatch
    raise ImportError(
        "select_target_instance.py requires PyTorch3D. "
        "Run it inside the 'sam3d-objects' conda env."
    ) from exc

OVERLAY_PALETTE_BGR: list[tuple[int, int, int]] = [
    (0, 255, 255),
    (255, 140, 0),
    (0, 255, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (0, 0, 255),
    (255, 0, 0),
]

IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_label(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh: {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected mesh shapes for {path}: {verts.shape}, {faces.shape}")
    return verts, faces


def resolve_input_path(script_dir: Path, video_name: str, raw_input_dir: str | None) -> Path:
    if raw_input_dir:
        return Path(raw_input_dir).resolve()
    return script_dir / "input_prompts" / video_name


def resolve_output_dir(script_dir: Path, video_name: str, raw_outdir: str | None) -> Path:
    if raw_outdir:
        return Path(raw_outdir).resolve()
    return script_dir / "output" / video_name


def resolve_scannet_root(script_dir: Path, raw_scannet_root: str | None) -> Path:
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
        "segments_path": scene_root / "scans" / "segments.json",
        "segments_anno_path": scene_root / "scans" / "segments_anno.json",
    }


def build_pinhole_intrinsics(
    transforms_payload: dict[str, Any],
) -> tuple[np.ndarray, int, int]:
    width = int(transforms_payload["w"])
    height = int(transforms_payload["h"])
    intrinsics = np.array(
        [
            [float(transforms_payload["fl_x"]), 0.0, float(transforms_payload["cx"])],
            [0.0, float(transforms_payload["fl_y"]), float(transforms_payload["cy"])],
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
        f"Could not find camera '{camera_name}' in COLMAP images.txt: {colmap_images_path}"
    )


def build_candidate_instances(
    mesh_faces: np.ndarray,
    seg_indices: np.ndarray,
    seg_groups: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    face_batches: list[np.ndarray] = []
    face_instance_ids: list[np.ndarray] = []
    instance_meta: dict[int, dict[str, Any]] = {}

    for group in seg_groups:
        label = group["label"]
        if not normalize_label(label):
            continue

        object_id = int(group["objectId"])
        segments = np.asarray(group["segments"], dtype=np.int64)
        if segments.size == 0:
            continue

        vertex_mask = np.isin(seg_indices, segments)
        face_mask = np.all(vertex_mask[mesh_faces], axis=1)
        candidate_faces = mesh_faces[face_mask]
        if candidate_faces.size == 0:
            continue

        face_batches.append(candidate_faces.astype(np.int64))
        face_instance_ids.append(
            np.full((candidate_faces.shape[0],), object_id, dtype=np.int32)
        )
        instance_meta[object_id] = {
            "instance_id": object_id,
            "label": label,
        }

    if not face_batches:
        raise ValueError("No valid instance annotations were found for the scene.")

    return (
        np.concatenate(face_batches, axis=0),
        np.concatenate(face_instance_ids, axis=0),
        instance_meta,
    )


def rasterize_instance_id_map(
    verts_world: np.ndarray,
    faces: np.ndarray,
    face_instance_ids: np.ndarray,
    intrinsics: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verts_tensor = torch.from_numpy(verts_world.astype(np.float32)).to(device)
    faces_tensor = torch.from_numpy(faces.astype(np.int64)).to(device)
    mesh = Meshes(verts=[verts_tensor], faces=[faces_tensor])

    camera = cameras_from_opencv_projection(
        R=torch.from_numpy(rotation_world_to_camera.astype(np.float32))[None].to(device),
        tvec=torch.from_numpy(translation_world_to_camera.astype(np.float32))[None].to(device),
        camera_matrix=torch.from_numpy(intrinsics.astype(np.float32))[None].to(device),
        image_size=torch.tensor([[height, width]], dtype=torch.float32, device=device),
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
    fragments = rasterizer(mesh)
    primitive_ids = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    id_map = np.full((height, width), -1, dtype=np.int32)
    valid = primitive_ids >= 0
    id_map[valid] = face_instance_ids[primitive_ids[valid].astype(np.int64)]
    return id_map


def build_overlay(
    image_bgr: np.ndarray,
    selected_mask: np.ndarray,
    selected_meta: dict[str, Any],
    click_uv: tuple[int, int],
) -> np.ndarray:
    overlay = image_bgr.copy()
    color = OVERLAY_PALETTE_BGR[0]
    color_arr = np.array(color, dtype=np.float32)
    overlay_f = overlay.astype(np.float32)
    overlay_f[selected_mask] = 0.55 * overlay_f[selected_mask] + 0.45 * color_arr
    overlay = np.clip(overlay_f, 0.0, 255.0).astype(np.uint8)

    ys, xs = np.where(selected_mask)
    if xs.size > 0:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        label_text = f"{selected_meta['instance_id']}: {selected_meta['label']}"
        cv2.putText(
            overlay,
            label_text,
            (x0, max(24, y0 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.circle(overlay, click_uv, 7, (0, 0, 255), -1)

    return overlay


def parse_click_uv(raw_click_uv: str | None) -> tuple[int, int] | None:
    if raw_click_uv is None:
        return None

    cleaned = raw_click_uv.strip().replace("(", "").replace(")", "")
    if not cleaned:
        return None

    if "," in cleaned:
        parts = [part.strip() for part in cleaned.split(",")]
    else:
        parts = cleaned.split()

    if len(parts) != 2:
        raise ValueError(
            f"Expected --click_uv in the form 'u,v' or 'u v', got: {raw_click_uv!r}"
        )

    return int(parts[0]), int(parts[1])


def get_click_point(
    image_bgr: np.ndarray,
    click_uv: tuple[int, int] | None,
) -> tuple[int, int]:
    if click_uv is not None:
        return int(click_uv[0]), int(click_uv[1])

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError(
            "No --click_uv was provided and no interactive display was detected. "
            "Use --click_uv 'u,v' when running over remote SSH."
        )

    clicked: dict[str, tuple[int, int] | None] = {"uv": None}
    window_name = "Select target instance"
    prompt_image = image_bgr.copy()
    cv2.putText(
        prompt_image,
        "Left click to select target object. Press ESC to cancel.",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    def _mouse_callback(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked["uv"] = (int(x), int(y))

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, _mouse_callback)
        while True:
            canvas = prompt_image.copy()
            if clicked["uv"] is not None:
                cv2.circle(canvas, clicked["uv"], 6, (0, 0, 255), -1)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if clicked["uv"] is not None:
                break
            if key == 27:
                raise KeyboardInterrupt("Selection cancelled by user.")
    finally:
        cv2.destroyAllWindows()

    if clicked["uv"] is None:
        raise RuntimeError("No click point was selected.")
    return clicked["uv"]


def resolve_clicked_instance(
    instance_id_map: np.ndarray,
    click_u: int,
    click_v: int,
    radius: int = 4,
) -> int:
    h, w = instance_id_map.shape
    if not (0 <= click_u < w and 0 <= click_v < h):
        raise ValueError(
            f"Click ({click_u}, {click_v}) is outside image bounds {(w, h)}."
        )

    instance_id = int(instance_id_map[click_v, click_u])
    if instance_id >= 0:
        return instance_id

    u0 = max(0, click_u - radius)
    u1 = min(w, click_u + radius + 1)
    v0 = max(0, click_v - radius)
    v1 = min(h, click_v + radius + 1)
    neighborhood = instance_id_map[v0:v1, u0:u1]
    neighborhood = neighborhood[neighborhood >= 0]
    if neighborhood.size == 0:
        raise ValueError(
            "Clicked point did not intersect any candidate instance. "
            "Try clicking closer to the visible object surface."
        )

    vals, counts = np.unique(neighborhood, return_counts=True)
    return int(vals[np.argmax(counts)])


def build_mask_stats(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Selected instance is not visible in the chosen camera view.")

    return {
        "visible_bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ],
        "mask_area_px": int(mask.sum()),
    }


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Select a target ScanNet++ instance by clicking the chosen view.",
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument(
        "--click_uv",
        default="1260, 900",
        help="Click location as 'u,v'. If omitted, falls back to interactive clicking.",
    )
    args = parser.parse_args()

    input_dir = resolve_input_path(script_dir, args.video_name, args.input_dir)
    output_root = resolve_output_dir(script_dir, args.video_name, args.outdir)
    scannet_root = resolve_scannet_root(script_dir, args.scannet_root)

    input_payload = load_json(input_dir / "input_pag.json")
    scene_context = input_payload["scene_context"]
    scene_paths = resolve_scene_paths(scannet_root, scene_context)

    image_bgr = cv2.imread(str(scene_paths["image_path"]), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {scene_paths['image_path']}")

    transforms_payload = load_json(scene_paths["transforms_path"])
    intrinsics, width, height = build_pinhole_intrinsics(transforms_payload)
    rotation_world_to_camera, translation_world_to_camera = load_colmap_pose(
        scene_paths["colmap_images_path"],
        scene_context["camera"]["name"],
    )
    if image_bgr.shape[1] != width or image_bgr.shape[0] != height:
        raise ValueError(
            "Loaded image shape does not match pinhole camera dimensions: "
            f"image={image_bgr.shape[1]}x{image_bgr.shape[0]}, metadata={width}x{height}"
        )

    verts_world, faces = load_mesh(scene_paths["mesh_path"])
    segments_payload = load_json(scene_paths["segments_path"])
    seg_indices = np.asarray(segments_payload["segIndices"], dtype=np.int64)
    anno_payload = load_json(scene_paths["segments_anno_path"])

    candidate_faces, face_instance_ids, instance_meta = build_candidate_instances(
        mesh_faces=faces,
        seg_indices=seg_indices,
        seg_groups=anno_payload["segGroups"],
    )

    instance_id_map = rasterize_instance_id_map(
        verts_world=verts_world,
        faces=candidate_faces,
        face_instance_ids=face_instance_ids,
        intrinsics=intrinsics,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        width=width,
        height=height,
    )

    parsed_click_uv = parse_click_uv(args.click_uv)
    click_u, click_v = get_click_point(image_bgr, parsed_click_uv)
    selected_instance_id = resolve_clicked_instance(instance_id_map, click_u, click_v)
    selected_meta = instance_meta[selected_instance_id]
    selected_mask = instance_id_map == selected_instance_id
    visible_stats = build_mask_stats(selected_mask)
    overlay_bgr = build_overlay(
        image_bgr=image_bgr,
        selected_mask=selected_mask,
        selected_meta=selected_meta,
        click_uv=(click_u, click_v),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    target_mask_path = output_root / "target_mask.png"
    save_mask(target_mask_path, selected_mask)

    selection_payload = {
        "scene_context": scene_context,
        "target_selection": {
            "click_uv": [int(click_u), int(click_v)],
            "instance_id": int(selected_meta["instance_id"]),
            "label": selected_meta["label"],
            "selection_source": "click_uv" if parsed_click_uv is not None else "manual_click",
            "mask_path": target_mask_path.name,
            **visible_stats,
        },
    }

    selection_json_path = output_root / "target_selection.json"
    selection_json_path.write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cv2.putText(
        overlay_bgr,
        f"selected instance {selected_instance_id}: {selected_meta['label']}",
        (20, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    overlay_path = output_root / "target_overlay.png"
    cv2.imwrite(str(overlay_path), overlay_bgr)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_root}")
    print(f"Scene image: {scene_paths['image_path']}")
    print(f"Saved selection JSON: {selection_json_path}")
    print(f"Saved target mask: {target_mask_path}")
    print(f"Saved overlay image: {overlay_path}")
    print(
        "Selected target:",
        {
            "instance_id": selected_meta["instance_id"],
            "label": selected_meta["label"],
            "click_uv": [click_u, click_v],
            **visible_stats,
        },
    )


if __name__ == "__main__":
    main()
