from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
DEFAULT_THRESHOLDS = {
    "inlier_ratio_min": 0.45,
    "median_reproj_max_px": 3.0,
    "mae_max": 12.0,
    "edge_iou_min": 0.15,
}
IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}


def read_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return bgr


def load_binary_mask(
    path: Path,
    shape_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if shape_hw is not None and mask.shape != shape_hw:
        raise ValueError(
            f"Mask shape mismatch for {path}: got {mask.shape[::-1]}, "
            f"expected {shape_hw[::-1]}"
        )
    return mask > 127


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def normalize_label(text: str) -> str:
    return " ".join(
        text.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def resolve_selection_path(
    script_dir: Path,
    video_name: str,
    raw_selection_json: str | None,
) -> Path:
    if raw_selection_json:
        return Path(raw_selection_json).resolve()
    return (
        script_dir.parent
        / "Select_Target_Instance"
        / "output"
        / video_name
        / "target_selection.json"
    ).resolve()


def resolve_generated_root(
    script_dir: Path,
    video_name: str,
    raw_generated_root: str | None,
) -> Path:
    if raw_generated_root:
        return Path(raw_generated_root).resolve()
    return (script_dir / "output" / video_name).resolve()


def resolve_input_path(
    script_dir: Path,
    video_name: str,
    raw_input_dir: str | None,
) -> Path:
    if raw_input_dir:
        return Path(raw_input_dir).resolve()
    return (
        script_dir.parent
        / "Select_Target_Instance"
        / "input_prompts"
        / video_name
    ).resolve()


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
        "segments_path": scene_root / "scans" / "segments.json",
        "segments_anno_path": scene_root / "scans" / "segments_anno.json",
    }


def resolve_first_frame_path(
    generated_root: Path,
    raw_frame: str | None,
) -> Path:
    first_frames_dir = generated_root / "first_frames"
    candidate = Path(raw_frame) if raw_frame else Path("frame_00.png")
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (first_frames_dir / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Requested frame does not exist: {resolved}")
    return resolved


def resolve_video_path(generated_root: Path, raw_video: str | None) -> Path:
    if raw_video:
        candidate = Path(raw_video)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (generated_root / candidate).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Requested video does not exist: {resolved}")
        return resolved

    candidates = sorted(generated_root.glob("*.mp4"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one mp4 under {generated_root} when --video is "
            f"omitted, found {len(candidates)}."
        )
    return candidates[0].resolve()


def resolve_scene_image_path(selection_root: Path) -> Path:
    path = (selection_root / "2d" / "scene_image.png").resolve()
    if not path.exists():
        raise FileNotFoundError(f"2d scene image not found: {path}")
    return path


def resolve_object_mask_path(
    selection_root: Path,
    selection_payload: dict[str, Any],
) -> Path:
    rel_path = str(selection_payload["target_selection_2d"]["mask_path"])
    path = (selection_root / rel_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"2d target mask path does not exist: {path}. rel_path={rel_path}"
        )
    return path


def resolve_target_label(selection_payload: dict[str, Any]) -> str:
    target_label = str(selection_payload["target_selection"]["object"]).strip()
    if not target_label:
        raise ValueError("target_selection.object must be a non-empty string.")
    return target_label


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh: {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        verts.ndim != 2
        or verts.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
    ):
        raise ValueError(
            f"Unexpected mesh shapes for {path}: {verts.shape}, {faces.shape}"
        )
    return verts, faces


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
        "Could not find camera "
        f"'{camera_name}' in COLMAP images.txt: {colmap_images_path}"
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
        raise ValueError(
            "No valid instance annotations were found for the scene.")

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
    import torch
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verts_tensor = torch.from_numpy(verts_world.astype(np.float32)).to(device)
    faces_tensor = torch.from_numpy(faces.astype(np.int64)).to(device)
    mesh = Meshes(verts=[verts_tensor], faces=[faces_tensor])

    camera = cameras_from_opencv_projection(
        R=torch.from_numpy(
            rotation_world_to_camera.astype(np.float32)
        )[None].to(device),
        tvec=torch.from_numpy(
            translation_world_to_camera.astype(np.float32)
        )[None].to(device),
        camera_matrix=torch.from_numpy(intrinsics.astype(np.float32))[None].to(
            device
        ),
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
    fragments = rasterizer(mesh)
    primitive_ids = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    id_map = np.full((height, width), -1, dtype=np.int32)
    valid = primitive_ids >= 0
    id_map[valid] = face_instance_ids[primitive_ids[valid].astype(np.int64)]
    return id_map


def build_mask_stats(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError(
            "Selected instance is not visible in the chosen camera view.")
    return {
        "visible_bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ],
        "mask_area_px": int(mask.sum()),
    }


def resize_and_center_crop(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source image shape: {image.shape}")

    scale = target_width / float(src_w)
    scaled_height = int(round(src_h * scale))
    if scaled_height < target_height:
        raise ValueError(
            "Scaled height is smaller than the requested crop height: "
            f"src={(src_w, src_h)}, scaled_height={scaled_height}, "
            f"target_height={target_height}"
        )

    resized = cv2.resize(
        image,
        (target_width, scaled_height),
        interpolation=interpolation,
    )
    crop_top = (scaled_height - target_height) // 2
    cropped = resized[crop_top:crop_top + target_height, :]

    transform = {
        "source_width": int(src_w),
        "source_height": int(src_h),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "scale": float(scale),
        "scaled_height": int(scaled_height),
        "crop_top": int(crop_top),
    }
    return cropped, transform


def transform_intrinsics(
    intrinsics: np.ndarray,
    scale: float,
    crop_top: int,
) -> np.ndarray:
    updated = intrinsics.astype(np.float32).copy()
    updated[0, 0] *= scale
    updated[1, 1] *= scale
    updated[0, 2] *= scale
    updated[1, 2] = updated[1, 2] * scale - float(crop_top)
    return updated


def save_camera_payloads(
    camera_path: Path,
    intrinsics: np.ndarray,
    world_to_camera_r: np.ndarray,
    world_to_camera_t: np.ndarray,
    camera_name: str,
    width: int,
    height: int,
) -> Path:
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = world_to_camera_r.astype(np.float32)
    world_to_camera[:3, 3] = world_to_camera_t.astype(np.float32)

    camera_payload = {
        "camera_name": camera_name,
        "width": int(width),
        "height": int(height),
        "world_to_camera_4x4": world_to_camera.tolist(),
        "intrinsics": intrinsics.astype(np.float32).tolist(),
    }
    save_json(camera_path, camera_payload)
    return camera_path


def load_camera_payload(
    camera_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, int, int]:
    payload = load_json(camera_path)
    intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
    world_to_camera = np.asarray(
        payload["world_to_camera_4x4"],
        dtype=np.float32)
    width = int(payload["width"])
    height = int(payload["height"])

    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 intrinsics matrix in {camera_path}, "
            f"got {intrinsics.shape}."
        )
    if world_to_camera.shape != (4, 4):
        raise ValueError(
            "Expected a 4x4 world_to_camera_4x4 matrix in "
            f"{camera_path}, got {world_to_camera.shape}."
        )

    rotation_world_to_camera = world_to_camera[:3, :3].astype(np.float32)
    translation_world_to_camera = world_to_camera[:3, 3].astype(np.float32)
    return (
        payload,
        intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    )


def create_feature_backend() -> tuple[str, Any, int]:
    return "SIFT", cv2.SIFT_create(nfeatures=6000), cv2.NORM_L2


def estimate_homography_fullframe(
    ref_gray: np.ndarray,
    gen_gray: np.ndarray,
    ratio_test: float = 0.75,
    ransac_thresh_px: float = 4.0,
) -> dict[str, Any]:
    detector_name, detector, matcher_norm = create_feature_backend()

    kp_ref, desc_ref = detector.detectAndCompute(ref_gray, None)
    kp_gen, desc_gen = detector.detectAndCompute(gen_gray, None)
    if desc_ref is None or desc_gen is None:
        raise RuntimeError("Could not compute full-frame descriptors.")

    matcher = cv2.BFMatcher(matcher_norm)
    raw_matches = matcher.knnMatch(desc_ref, desc_gen, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_test * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        raise RuntimeError(
            f"Not enough matches for homography: {len(good_matches)}"
        )

    src_pts = np.float32(
        [kp_gen[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransac_thresh_px,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed.")

    inlier_mask = inlier_mask.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    inlier_ratio = float(n_inliers / len(good_matches))

    proj_pts = cv2.perspectiveTransform(src_pts, homography).reshape(-1, 2)
    dst_flat = dst_pts.reshape(-1, 2)
    reproj_err = np.linalg.norm(proj_pts - dst_flat, axis=1)
    inlier_err = reproj_err[inlier_mask]
    med_err = float(np.median(inlier_err)
                    ) if inlier_err.size > 0 else float("nan")
    mean_err = float(np.mean(inlier_err)
                     ) if inlier_err.size > 0 else float("nan")

    inlier_matches = [
        match for i, match in enumerate(good_matches) if inlier_mask[i]
    ]
    if len(inlier_matches) > 200:
        rng = np.random.default_rng(0)
        selected = rng.choice(len(inlier_matches), size=200, replace=False)
        draw_matches = [inlier_matches[i] for i in selected]
    else:
        draw_matches = inlier_matches

    match_vis = cv2.drawMatches(
        ref_gray,
        kp_ref,
        gen_gray,
        kp_gen,
        draw_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return {
        "detector": detector_name,
        "H": homography,
        "match_vis": match_vis,
        "n_keypoints_reference": int(len(kp_ref)),
        "n_keypoints_generated": int(len(kp_gen)),
        "n_good_matches": int(len(good_matches)),
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "median_reprojection_error_px": med_err,
        "mean_reprojection_error_px": mean_err,
    }


def compute_fullframe_metrics(
    ref_rgb: np.ndarray,
    warped_rgb: np.ndarray,
) -> dict[str, Any]:
    ref_f = ref_rgb.astype(np.float32)
    warped_f = warped_rgb.astype(np.float32)
    abs_diff = np.abs(ref_f - warped_f)
    mae = float(abs_diff.mean())

    mse = float(np.mean((ref_f - warped_f) ** 2))
    if mse <= 1e-12:
        psnr = float("inf")
    else:
        psnr = float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))

    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY)
    edges_ref = cv2.Canny(ref_gray, 100, 200) > 0
    edges_warped = cv2.Canny(warped_gray, 100, 200) > 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ref_tol = cv2.dilate((edges_ref.astype(np.uint8) * 255), kernel) > 0
    warped_tol = cv2.dilate((edges_warped.astype(np.uint8) * 255), kernel) > 0

    intersection = int(np.logical_and(ref_tol, warped_tol).sum())
    union = int(np.logical_or(ref_tol, warped_tol).sum())
    edge_iou = float(intersection / union) if union > 0 else 0.0

    diff_gray = cv2.cvtColor(
        cv2.absdiff(
            ref_rgb,
            warped_rgb),
        cv2.COLOR_RGB2GRAY)
    diff_heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_INFERNO)
    return {
        "fullframe_mae_rgb": mae,
        "fullframe_psnr_db": psnr,
        "fullframe_edge_iou": edge_iou,
        "diff_gray": diff_gray,
        "diff_heatmap_bgr": diff_heatmap,
    }


def build_verdict(metrics: dict[str, float],
                  checks_cfg: dict[str, float]) -> dict[str, Any]:
    checks = {
        "inlier_ratio_ok": (
            metrics["inlier_ratio"] >= checks_cfg["inlier_ratio_min"]
        ),
        "median_reprojection_ok": (
            metrics["median_reprojection_error_px"]
            <= checks_cfg["median_reproj_max_px"]
        ),
        "mae_ok": metrics["fullframe_mae_rgb"] <= checks_cfg["mae_max"],
        "edge_iou_ok": (
            metrics["fullframe_edge_iou"] >= checks_cfg["edge_iou_min"]
        ),
    }
    pass_fraction = float(sum(checks.values()) / len(checks))

    if pass_fraction >= 0.75:
        verdict = "Strong full-frame preservation"
    elif pass_fraction >= 0.5:
        verdict = "Moderate preservation; inspect diagnostics"
    else:
        verdict = "Weak preservation; likely scene drift"

    return {
        "checks": checks,
        "pass_fraction": pass_fraction,
        "verdict": verdict,
    }


def run_geometry_eval(
    video_name: str,
    frame_name: str,
    ref_rgb: np.ndarray,
    gen_rgb: np.ndarray,
    object_mask: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    ref_h, ref_w = ref_rgb.shape[:2]
    if gen_rgb.shape[:2] != (ref_h, ref_w):
        raise ValueError(
            f"Generated frame shape {gen_rgb.shape[:2]} does not match "
            f"reference shape {(ref_h, ref_w)}."
        )
    if object_mask.shape != (ref_h, ref_w):
        raise ValueError(
            f"Object mask shape {object_mask.shape} does not match "
            f"image shape {(ref_h, ref_w)}."
        )

    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    gen_gray = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)

    homography_info = estimate_homography_fullframe(ref_gray, gen_gray)
    warped_gen = cv2.warpPerspective(
        gen_rgb,
        homography_info["H"],
        (ref_w, ref_h),
        flags=cv2.INTER_CUBIC,
    )
    frame_metrics = compute_fullframe_metrics(ref_rgb, warped_gen)
    summary = {
        "video_name": video_name,
        "frame_name": frame_name,
        "detector": homography_info["detector"],
        "n_keypoints_reference": homography_info["n_keypoints_reference"],
        "n_keypoints_generated": homography_info["n_keypoints_generated"],
        "n_good_matches": homography_info["n_good_matches"],
        "n_inliers": homography_info["n_inliers"],
        "inlier_ratio": homography_info["inlier_ratio"],
        "median_reprojection_error_px": homography_info[
            "median_reprojection_error_px"
        ],
        "mean_reprojection_error_px": homography_info[
            "mean_reprojection_error_px"
        ],
        "fullframe_mae_rgb": frame_metrics["fullframe_mae_rgb"],
        "fullframe_psnr_db": frame_metrics["fullframe_psnr_db"],
        "fullframe_edge_iou": frame_metrics["fullframe_edge_iou"],
    }
    summary.update(build_verdict(summary, thresholds))
    artifacts = {
        "ref_rgb": ref_rgb,
        "gen_rgb": gen_rgb,
        "warped_gen": warped_gen,
        "match_vis_bgr": homography_info["match_vis"],
        "diff_gray": frame_metrics["diff_gray"],
        "diff_heatmap_bgr": frame_metrics["diff_heatmap_bgr"],
        "object_mask": object_mask,
    }
    return {"summary": summary, "artifacts": artifacts}


def to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def resize_to_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    image = to_bgr(image)
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = min(width / float(src_w), height / float(src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized_h) // 2
    left = (width - resized_w) // 2
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas


def add_tile_title(tile: np.ndarray, title: str) -> np.ndarray:
    canvas = tile.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (18, 18, 18), -1)
    cv2.putText(
        canvas,
        title,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def create_overview_panel(
    ref_rgb: np.ndarray,
    gen_rgb: np.ndarray,
    match_vis_bgr: np.ndarray,
    diff_heatmap_bgr: np.ndarray,
) -> np.ndarray:
    tile_w, tile_h = 640, 360
    tiles = [
        add_tile_title(
            resize_to_canvas(
                cv2.cvtColor(
                    ref_rgb,
                    cv2.COLOR_RGB2BGR),
                tile_w,
                tile_h),
            "Reference image",
        ),
        add_tile_title(
            resize_to_canvas(
                cv2.cvtColor(
                    gen_rgb,
                    cv2.COLOR_RGB2BGR),
                tile_w,
                tile_h),
            "Generated image (1280x720)",
        ),
        add_tile_title(
            resize_to_canvas(
                match_vis_bgr,
                tile_w,
                tile_h),
            "Inlier feature matches",
        ),
        add_tile_title(
            resize_to_canvas(
                diff_heatmap_bgr,
                tile_w,
                tile_h),
            "Absolute-difference heatmap",
        ),
    ]
    top_row = np.hstack([tiles[0], tiles[1]])
    bottom_row = np.hstack([tiles[2], tiles[3]])
    return np.vstack([top_row, bottom_row])


def create_mask_overlay(
        gen_rgb: np.ndarray,
        object_mask: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = 0.55
    mask_color = np.array([0, 0, 255], dtype=np.float32)
    overlay[object_mask] = (
        (1.0 - alpha) * overlay[object_mask] + alpha * mask_color
    )
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def create_object_crop_panel(
    ref_rgb: np.ndarray,
    warped_gen_rgb: np.ndarray,
    object_mask: np.ndarray,
) -> np.ndarray:
    ys, xs = np.where(object_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    ref_crop = ref_rgb[y0:y1 + 1, x0:x1 + 1]
    warped_crop = warped_gen_rgb[y0:y1 + 1, x0:x1 + 1]
    diff_crop_gray = cv2.cvtColor(
        cv2.absdiff(ref_crop, warped_crop),
        cv2.COLOR_RGB2GRAY,
    )
    diff_crop_heatmap = cv2.applyColorMap(diff_crop_gray, cv2.COLORMAP_MAGMA)

    tile_w, tile_h = 426, 360
    tiles = [
        add_tile_title(
            resize_to_canvas(
                cv2.cvtColor(
                    ref_crop,
                    cv2.COLOR_RGB2BGR),
                tile_w,
                tile_h),
            "Object crop: reference",
        ),
        add_tile_title(
            resize_to_canvas(
                cv2.cvtColor(
                    warped_crop,
                    cv2.COLOR_RGB2BGR),
                tile_w,
                tile_h,
            ),
            "Object crop: aligned generated",
        ),
        add_tile_title(
            resize_to_canvas(
                diff_crop_heatmap,
                tile_w,
                tile_h),
            "Object crop: abs-diff",
        ),
    ]
    return np.hstack(tiles)


def build_projected_overlay(
    generated_bgr: np.ndarray,
    projected_mask: np.ndarray,
    target_label: str,
    instance_id: int,
    instance_label: str,
) -> np.ndarray:
    overlay = generated_bgr.astype(np.float32).copy()
    mask_color = np.array([0, 255, 255], dtype=np.float32)
    overlay[projected_mask] = (
        0.60 * overlay[projected_mask] + 0.40 * mask_color
    )
    overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    ys, xs = np.where(projected_mask)
    if xs.size > 0:
        x0, y0, x1, y1 = int(
            xs.min()), int(
            ys.min()), int(
            xs.max()), int(
                ys.max())
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)

    lines = [
        f"object: {target_label}",
        f"instance: {instance_id} ({instance_label})",
        f"projected area px: {int(np.count_nonzero(projected_mask))}",
    ]
    y = 32
    for line in lines:
        cv2.putText(
            overlay,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 34
    return overlay


def build_projection_artifacts(
    scene_context: dict[str, Any],
    scannet_root: Path,
    intrinsics: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    selection_payload: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    scene_paths = resolve_scene_paths(scannet_root, scene_context)
    verts_world, faces = load_mesh(scene_paths["mesh_path"])
    segments_payload = load_json(scene_paths["segments_path"])
    anno_payload = load_json(scene_paths["segments_anno_path"])
    seg_indices = np.asarray(segments_payload["segIndices"], dtype=np.int64)

    (
        candidate_faces,
        face_instance_ids,
        instance_meta,
    ) = build_candidate_instances(
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
    instance_id = int(selection_payload["target_selection"]["instance_id"])
    if instance_id not in instance_meta:
        raise ValueError(
            f"Resolved instance id {instance_id} is missing from "
            "instance metadata."
        )

    projected_mask = instance_id_map == instance_id
    projected_stats = build_mask_stats(projected_mask)
    return {
        "instance_id": instance_id,
        "instance_label": str(instance_meta[instance_id]["label"]),
        "projected_mask": projected_mask,
        "projected_mask_area_px": int(projected_stats["mask_area_px"]),
        "projected_bbox_xyxy": projected_stats["visible_bbox_xyxy"],
    }


def load_eval_context(
    script_dir: Path,
    video_name: str,
    generated_root: str | None,
    selection_json: str | None,
    input_dir: str | None,
    scannet_root: str | None,
) -> dict[str, Any]:
    resolved_generated_root = resolve_generated_root(
        script_dir,
        video_name,
        generated_root,
    )
    selection_path = resolve_selection_path(
        script_dir, video_name, selection_json)
    selection_root = selection_path.parent
    selection_payload = load_json(selection_path)
    scene_image_path = resolve_scene_image_path(selection_root)
    object_mask_path = resolve_object_mask_path(
        selection_root, selection_payload)
    target_label = resolve_target_label(selection_payload)

    resolved_input_dir = resolve_input_path(script_dir, video_name, input_dir)
    input_payload = load_json(resolved_input_dir / "input_pag.json")
    scene_context = input_payload["scene_context"]
    resolved_scannet_root = resolve_scannet_root(script_dir, scannet_root)
    scene_paths = resolve_scene_paths(resolved_scannet_root, scene_context)
    return {
        "generated_root": resolved_generated_root,
        "selection_path": selection_path,
        "selection_payload": selection_payload,
        "scene_image_path": scene_image_path,
        "object_mask_path": object_mask_path,
        "target_label": target_label,
        "input_dir": resolved_input_dir,
        "scene_context": scene_context,
        "scannet_root": resolved_scannet_root,
        "scene_paths": scene_paths,
    }


def extract_first_frame(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(
            f"Failed to decode the first frame from: {video_path}")
    return frame_bgr


def save_eval_outputs(
    debug_root: Path,
    generated_bgr: np.ndarray,
    artifacts: dict[str, Any],
    projection: dict[str, Any],
    target_label: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    target_3d_overlay_path = debug_root / "target_3d_overlay.png"
    overview_path = debug_root / "overview_panel.png"
    mask_overlay_path = debug_root / "mask_overlay.png"
    object_crop_path = debug_root / "object_crop_panel.png"
    report_path = debug_root / "geometry_preservation_report.json"

    projected_overlay = build_projected_overlay(
        generated_bgr=generated_bgr,
        projected_mask=projection["projected_mask"],
        target_label=target_label,
        instance_id=projection["instance_id"],
        instance_label=projection["instance_label"],
    )
    overview_panel = create_overview_panel(
        ref_rgb=artifacts["ref_rgb"],
        gen_rgb=artifacts["gen_rgb"],
        match_vis_bgr=artifacts["match_vis_bgr"],
        diff_heatmap_bgr=artifacts["diff_heatmap_bgr"],
    )
    mask_overlay = create_mask_overlay(
        artifacts["gen_rgb"], artifacts["object_mask"])
    object_crop_panel = create_object_crop_panel(
        ref_rgb=artifacts["ref_rgb"],
        warped_gen_rgb=artifacts["warped_gen"],
        object_mask=artifacts["object_mask"],
    )

    save_image(target_3d_overlay_path, projected_overlay)
    save_image(overview_path, overview_panel)
    save_image(mask_overlay_path, mask_overlay)
    save_image(object_crop_path, object_crop_panel)

    summary["target_3d_overlay_path"] = str(target_3d_overlay_path)
    summary["target_instance_id"] = int(projection["instance_id"])
    summary["target_instance_label"] = projection["instance_label"]
    summary["projected_mask_area_px"] = int(
        projection["projected_mask_area_px"])
    summary["projected_bbox_xyxy"] = projection["projected_bbox_xyxy"]

    save_json(report_path, summary)
    return {
        "summary": summary,
        "outputs": {
            "target_3d_overlay_path": target_3d_overlay_path,
            "overview_path": overview_path,
            "mask_overlay_path": mask_overlay_path,
            "object_crop_path": object_crop_path,
            "report_path": report_path,
        },
    }


def run_resize_first_frame_eval(
    script_dir: Path,
    video_name: str,
    frame: str | None,
    generated_root: str | None,
    selection_json: str | None,
    input_dir: str | None,
    scannet_root: str | None,
) -> dict[str, Any]:
    context = load_eval_context(
        script_dir,
        video_name,
        generated_root,
        selection_json,
        input_dir,
        scannet_root,
    )
    frame_path = resolve_first_frame_path(context["generated_root"], frame)
    frame_stem = frame_path.stem

    scene_bgr = read_bgr(context["scene_image_path"])
    generated_bgr = read_bgr(frame_path)
    object_mask = load_binary_mask(
        context["object_mask_path"],
        shape_hw=scene_bgr.shape[:2],
    )

    reference_bgr, reference_preprocess = resize_and_center_crop(
        scene_bgr,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        interpolation=cv2.INTER_CUBIC,
    )
    resized_mask_u8, mask_transform = resize_and_center_crop(
        object_mask.astype(np.uint8) * 255,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        interpolation=cv2.INTER_NEAREST,
    )
    resized_generated_bgr, _ = resize_and_center_crop(
        generated_bgr,
        TARGET_WIDTH,
        TARGET_HEIGHT,
        interpolation=cv2.INTER_CUBIC,
    )
    resized_mask = resized_mask_u8 > 127

    if (
        reference_preprocess["crop_top"] != mask_transform["crop_top"]
        or reference_preprocess["scaled_height"]
        != mask_transform["scaled_height"]
    ):
        raise ValueError(
            "Reference image and target mask transforms diverged unexpectedly."
        )

    transforms_payload = load_json(context["scene_paths"]["transforms_path"])
    intrinsics, width, height = build_pinhole_intrinsics(transforms_payload)
    rotation_world_to_camera, translation_world_to_camera = load_colmap_pose(
        context["scene_paths"]["colmap_images_path"],
        context["scene_context"]["camera"]["name"],
    )
    if (width, height) != (
        reference_preprocess["source_width"],
        reference_preprocess["source_height"],
    ):
        raise ValueError(
            "Reference image shape does not match camera intrinsics metadata: "
            "image=("
            f"{reference_preprocess['source_width']}, "
            f"{reference_preprocess['source_height']}), "
            f"metadata={(width, height)}"
        )

    resized_intrinsics = transform_intrinsics(
        intrinsics=intrinsics,
        scale=float(reference_preprocess["scale"]),
        crop_top=int(reference_preprocess["crop_top"]),
    )

    resized_dir = ensure_dir(
        context["generated_root"] /
        "first_frames_resized")
    resized_frame_path = resized_dir / frame_path.name
    resized_mask_path = resized_dir / "target_mask.png"
    camera_path = context["generated_root"] / "resized_camera.json"
    debug_root = ensure_dir(
        context["generated_root"] / "debug" / "first_frames" / frame_stem
    )

    save_image(resized_frame_path, resized_generated_bgr)
    save_mask(resized_mask_path, resized_mask)
    camera_json_path = save_camera_payloads(
        camera_path=camera_path,
        intrinsics=resized_intrinsics,
        world_to_camera_r=rotation_world_to_camera,
        world_to_camera_t=translation_world_to_camera,
        camera_name=str(context["scene_context"]["camera"]["name"]),
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
    )

    eval_result = run_geometry_eval(
        video_name=video_name,
        frame_name=frame_stem,
        ref_rgb=cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB),
        gen_rgb=cv2.cvtColor(resized_generated_bgr, cv2.COLOR_BGR2RGB),
        object_mask=resized_mask,
        thresholds=DEFAULT_THRESHOLDS,
    )
    projection = build_projection_artifacts(
        scene_context=context["scene_context"],
        scannet_root=context["scannet_root"],
        intrinsics=resized_intrinsics,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        selection_payload=context["selection_payload"],
        width=TARGET_WIDTH,
        height=TARGET_HEIGHT,
    )
    summary = eval_result["summary"]
    summary["source_generated_path"] = str(frame_path)
    summary["resized_generated_path"] = str(resized_frame_path)
    summary["target_mask_path"] = str(resized_mask_path)
    summary["camera_json_path"] = str(camera_json_path)
    summary["target_label"] = context["target_label"]
    summary["reference_preprocess"] = reference_preprocess

    result = save_eval_outputs(
        debug_root=debug_root,
        generated_bgr=resized_generated_bgr,
        artifacts=eval_result["artifacts"],
        projection=projection,
        target_label=context["target_label"],
        summary=summary,
    )
    result["outputs"].update(
        {
            "resized_frame_path": resized_frame_path,
            "target_mask_path": resized_mask_path,
            "camera_json_path": camera_json_path,
        }
    )
    return result


def run_video_first_frame_eval(
    script_dir: Path,
    video_name: str,
    video: str | None,
    generated_root: str | None,
    selection_json: str | None,
    input_dir: str | None,
    scannet_root: str | None,
) -> dict[str, Any]:
    context = load_eval_context(
        script_dir,
        video_name,
        generated_root,
        selection_json,
        input_dir,
        scannet_root,
    )
    video_path = resolve_video_path(context["generated_root"], video)
    video_stem = video_path.stem

    camera_path = context["generated_root"] / "resized_camera.json"
    if not camera_path.exists():
        raise FileNotFoundError(
            f"Missing resized camera JSON: {camera_path}. "
            "Run resize_first_frame_and_eval.py first."
        )
    target_mask_path = (
        context["generated_root"] / "first_frames_resized" / "target_mask.png"
    )
    if not target_mask_path.exists():
        raise FileNotFoundError(
            f"Missing resized target mask: {target_mask_path}. "
            "Run resize_first_frame_and_eval.py first."
        )

    (
        _camera_payload,
        resized_intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    ) = load_camera_payload(camera_path)

    generated_bgr = extract_first_frame(video_path)
    if generated_bgr.shape[:2] != (height, width):
        raise ValueError(
            f"Extracted first frame shape "
            f"{generated_bgr.shape[1], generated_bgr.shape[0]} "
            f"does not match camera JSON resolution {(width, height)}."
        )

    scene_bgr = read_bgr(context["scene_image_path"])
    resized_mask = load_binary_mask(target_mask_path, shape_hw=(height, width))
    reference_bgr, reference_preprocess = resize_and_center_crop(
        scene_bgr,
        width,
        height,
        interpolation=cv2.INTER_CUBIC,
    )

    debug_root = ensure_dir(
        context["generated_root"] / "debug" / "videos" / video_stem
    )
    first_frame_path = debug_root / "first_frame.png"
    save_image(first_frame_path, generated_bgr)

    eval_result = run_geometry_eval(
        video_name=video_name,
        frame_name="first_frame",
        ref_rgb=cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB),
        gen_rgb=cv2.cvtColor(generated_bgr, cv2.COLOR_BGR2RGB),
        object_mask=resized_mask,
        thresholds=DEFAULT_THRESHOLDS,
    )
    projection = build_projection_artifacts(
        scene_context=context["scene_context"],
        scannet_root=context["scannet_root"],
        intrinsics=resized_intrinsics,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        selection_payload=context["selection_payload"],
        width=width,
        height=height,
    )
    summary = eval_result["summary"]
    summary["video_path"] = str(video_path)
    summary["first_frame_path"] = str(first_frame_path)
    summary["target_mask_path"] = str(target_mask_path)
    summary["camera_json_path"] = str(camera_path)
    summary["target_label"] = context["target_label"]
    summary["reference_preprocess"] = reference_preprocess

    result = save_eval_outputs(
        debug_root=debug_root,
        generated_bgr=generated_bgr,
        artifacts=eval_result["artifacts"],
        projection=projection,
        target_label=context["target_label"],
        summary=summary,
    )
    result["outputs"].update(
        {
            "first_frame_path": first_frame_path,
            "target_mask_path": target_mask_path,
            "camera_json_path": camera_path,
        }
    )
    return result
