from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))
    if not ok:
        raise IOError(f"Failed to write mask: {path}")


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


def resolve_generated_root(
    script_dir: Path,
    video_name: str,
    raw_generated_root: str | None,
) -> Path:
    if raw_generated_root:
        return Path(raw_generated_root).resolve()
    return (script_dir / "output" / video_name).resolve()


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


def resolve_optional_video_path(
    generated_root: Path,
    raw_video: str | None,
) -> Path | None:
    if raw_video:
        candidate = Path(raw_video)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (generated_root / candidate).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Requested video does not exist: {resolved}")
        return resolved

    candidates = sorted(generated_root.glob("*.mp4"))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "Multiple mp4 files found. Pass --video to choose one explicitly."
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

    _, transforms_rel = IMAGE_SOURCE_TO_REL_PATHS[camera_source]
    scene_root = scannet_root / scene_id
    return {
        "transforms_path": scene_root / transforms_rel,
        "colmap_images_path": scene_root / "dslr" / "colmap" / "images.txt",
        "camera_name": Path(camera_name).name,
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
        script_dir,
        video_name,
        selection_json,
    )
    selection_root = selection_path.parent
    selection_payload = load_json(selection_path)
    scene_image_path = resolve_scene_image_path(selection_root)
    object_mask_path = resolve_object_mask_path(selection_root, selection_payload)

    resolved_input_dir = resolve_input_path(script_dir, video_name, input_dir)
    input_payload = load_json(resolved_input_dir / "input_pag.json")
    scene_context = input_payload["scene_context"]
    resolved_scannet_root = resolve_scannet_root(script_dir, scannet_root)
    scene_paths = resolve_scene_paths(resolved_scannet_root, scene_context)

    return {
        "generated_root": resolved_generated_root,
        "scene_image_path": scene_image_path,
        "object_mask_path": object_mask_path,
        "scene_paths": scene_paths,
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


def fit_image_to_resolution(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if (src_w, src_h) == (target_width, target_height):
        return image
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_CUBIC,
    )


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


def save_camera_payload(
    camera_path: Path,
    intrinsics: np.ndarray,
    world_to_camera_r: np.ndarray,
    world_to_camera_t: np.ndarray,
    camera_name: str,
    width: int,
    height: int,
) -> None:
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = world_to_camera_r.astype(np.float32)
    world_to_camera[:3, 3] = world_to_camera_t.astype(np.float32)

    payload = {
        "camera_name": camera_name,
        "width": int(width),
        "height": int(height),
        "world_to_camera_4x4": world_to_camera.tolist(),
        "intrinsics": intrinsics.astype(np.float32).tolist(),
    }
    save_json(camera_path, payload)


def create_mask_overlay(image_bgr: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    if image_bgr.shape[:2] != object_mask.shape:
        raise ValueError(
            f"Overlay shape mismatch: image={image_bgr.shape[:2]}, "
            f"mask={object_mask.shape}"
        )

    overlay = image_bgr.astype(np.float32).copy()
    alpha = 0.55
    mask_color = np.array([0, 0, 255], dtype=np.float32)
    overlay[object_mask] = (
        (1.0 - alpha) * overlay[object_mask] + alpha * mask_color
    )
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def extract_first_frame(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to decode first frame from: {video_path}")
    return frame_bgr


def run_eval(
    script_dir: Path,
    video_name: str,
    frame: str | None,
    video: str | None,
    generated_root: str | None,
    selection_json: str | None,
    input_dir: str | None,
    scannet_root: str | None,
    target_width: int,
    target_height: int,
) -> dict[str, Path | None]:
    context = load_eval_context(
        script_dir=script_dir,
        video_name=video_name,
        generated_root=generated_root,
        selection_json=selection_json,
        input_dir=input_dir,
        scannet_root=scannet_root,
    )

    generated_root_path = context["generated_root"]
    frame_path = resolve_first_frame_path(generated_root_path, frame)
    video_path = resolve_optional_video_path(generated_root_path, video)

    scene_bgr = read_bgr(context["scene_image_path"])
    generated_bgr = read_bgr(frame_path)
    object_mask = load_binary_mask(
        context["object_mask_path"],
        shape_hw=scene_bgr.shape[:2],
    )

    reference_bgr, reference_transform = resize_and_center_crop(
        scene_bgr,
        target_width,
        target_height,
        interpolation=cv2.INTER_CUBIC,
    )
    resized_mask_u8, mask_transform = resize_and_center_crop(
        object_mask.astype(np.uint8) * 255,
        target_width,
        target_height,
        interpolation=cv2.INTER_NEAREST,
    )
    resized_generated_bgr, _ = resize_and_center_crop(
        generated_bgr,
        target_width,
        target_height,
        interpolation=cv2.INTER_CUBIC,
    )
    resized_mask = resized_mask_u8 > 127

    if (
        reference_transform["crop_top"] != mask_transform["crop_top"]
        or reference_transform["scaled_height"]
        != mask_transform["scaled_height"]
    ):
        raise ValueError("Reference image and mask transforms diverged.")

    transforms_payload = load_json(context["scene_paths"]["transforms_path"])
    intrinsics, width, height = build_pinhole_intrinsics(transforms_payload)
    if (width, height) != (
        reference_transform["source_width"],
        reference_transform["source_height"],
    ):
        raise ValueError(
            "Reference scene image shape does not match camera metadata."
        )

    rotation_world_to_camera, translation_world_to_camera = load_colmap_pose(
        context["scene_paths"]["colmap_images_path"],
        context["scene_paths"]["camera_name"],
    )
    resized_intrinsics = transform_intrinsics(
        intrinsics=intrinsics,
        scale=float(reference_transform["scale"]),
        crop_top=int(reference_transform["crop_top"]),
    )

    resized_dir = ensure_dir(generated_root_path / "first_frames_resized")
    resized_frame_path = resized_dir / frame_path.name
    resized_mask_path = resized_dir / "frame_00_target_mask.png"
    camera_json_path = generated_root_path / "resized_camera.json"

    save_image(resized_frame_path, resized_generated_bgr)
    save_mask(resized_mask_path, resized_mask)
    save_camera_payload(
        camera_path=camera_json_path,
        intrinsics=resized_intrinsics,
        world_to_camera_r=rotation_world_to_camera,
        world_to_camera_t=translation_world_to_camera,
        camera_name=str(context["scene_paths"]["camera_name"]),
        width=target_width,
        height=target_height,
    )

    overlays_dir = ensure_dir(generated_root_path / "debug")
    image_overlay_path = overlays_dir / "inpainted_first_frame_overlay.png"
    image_overlay = create_mask_overlay(resized_generated_bgr, resized_mask)
    save_image(image_overlay_path, image_overlay)

    video_overlay_path: Path | None = None
    if video_path is not None:
        video_first_frame_bgr = extract_first_frame(video_path)
        video_first_frame_bgr = fit_image_to_resolution(
            video_first_frame_bgr,
            target_width,
            target_height,
        )
        video_overlay = create_mask_overlay(video_first_frame_bgr, resized_mask)
        video_overlay_path = overlays_dir / "video_first_frame_overlay.png"
        save_image(video_overlay_path, video_overlay)

    return {
        "resized_frame_path": resized_frame_path,
        "target_mask_path": resized_mask_path,
        "camera_json_path": camera_json_path,
        "image_overlay_path": image_overlay_path,
        "video_overlay_path": video_overlay_path,
        "video_path": video_path,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate first-frame mask overlays for image and optional video."
        ),
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument(
        "--frame",
        default="frame_00.png",
        help="Frame filename under first_frames/ or an absolute path.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Optional explicit mp4 path. If omitted and no mp4 exists, "
            "video evaluation is skipped."
        ),
    )
    parser.add_argument(
        "--generated-root",
        default=None,
        help="Defaults to Generate_Video/output/<video_name>.",
    )
    parser.add_argument(
        "--selection-json",
        default=None,
        help="Path to target_selection.json.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Defaults to Select_Target_Instance/input_prompts/<video_name>.",
    )
    parser.add_argument(
        "--scannet-root",
        default=None,
        help="Defaults to the Select_Target_Instance convention.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=1280,
        help="Target output width.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=720,
        help="Target output height.",
    )
    args = parser.parse_args()

    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("--target-width and --target-height must be positive.")

    result = run_eval(
        script_dir=script_dir,
        video_name=args.video_name,
        frame=args.frame,
        video=args.video,
        generated_root=args.generated_root,
        selection_json=args.selection_json,
        input_dir=args.input_dir,
        scannet_root=args.scannet_root,
        target_width=args.target_width,
        target_height=args.target_height,
    )

    print(f"Saved resized frame: {result['resized_frame_path']}")
    print(f"Saved resized target mask: {result['target_mask_path']}")
    print(f"Saved camera JSON: {result['camera_json_path']}")
    print(f"Saved image overlay: {result['image_overlay_path']}")

    if result["video_overlay_path"] is not None:
        print(f"Saved video overlay: {result['video_overlay_path']}")
    else:
        print(
            "No mp4 found in generated root; skipped video first-frame "
            "overlay."
        )


if __name__ == "__main__":
    main()
