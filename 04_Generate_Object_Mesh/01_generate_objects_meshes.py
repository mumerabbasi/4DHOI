"""Generate object meshes from first-frame Segment_Video masks."""

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes
from mesh_generation_utils import (
    create_posed_mesh,
    discover_first_frame_stem,
    discover_objects_with_first_frame_masks,
    estimate_camera_intrinsics,
    extract_pose_components,
    find_frame_image_path,
    generate_mesh,
    load_sam3d,
    postprocess_trimesh_with_sam3d,
    sam3d_mesh_to_trimesh,
)
from rendering_utils import (
    ColorBGR,
    DEFAULT_OVERLAY_CONTOUR_THICKNESS,
    DEFAULT_OVERLAY_FILL_ALPHA,
    QualityRenderBackend,
    add_overlay_legend,
    build_object_color_map,
    camera_k_from_info,
    render_multi_object_overlay_quality,
    render_single_object_overlay_quality,
)

F_P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


@dataclass(frozen=True)
class VideoOutputPaths:
    root: Path
    meshes_dir: Path
    overlays_dir: Path
    camera_intrinsics_json: Path


@dataclass(frozen=True)
class ObjectMaskSpec:
    name: str
    mask_path: Path
    color_bgr: ColorBGR


@dataclass
class VideoGenerationContext:
    video_name: str
    first_frame_stem: str
    image_rgb: np.ndarray
    camera_k_overlay: np.ndarray
    output_paths: VideoOutputPaths
    object_specs: List[ObjectMaskSpec]


@dataclass(frozen=True)
class GeneratedObjectResult:
    name: str
    color_bgr: ColorBGR
    posed_mesh_p3d: trimesh.Trimesh
    mesh_path: Path
    overlay_path: Path


@dataclass(frozen=True)
class ObjectGenerationFailure:
    name: str
    message: str


def convert_mesh_p3d_to_cv(mesh_p3d: trimesh.Trimesh) -> trimesh.Trimesh:
    """Convert PyTorch3D camera coordinates to OpenCV camera coordinates."""
    mesh_cv = mesh_p3d.copy()
    verts_p3d = np.asarray(mesh_p3d.vertices, dtype=np.float32)
    mesh_cv.vertices = (verts_p3d @ F_P3D_TO_CV.transpose()).astype(np.float32)
    return mesh_cv


def build_output_paths(mesh_output_root: Path, video_name: str) -> VideoOutputPaths:
    """Create and return the output directory layout for one video."""
    root = (mesh_output_root / video_name).resolve()
    meshes_dir = root / "meshes"
    overlays_dir = root / "overlays"
    root.mkdir(parents=True, exist_ok=True)
    meshes_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    return VideoOutputPaths(
        root=root,
        meshes_dir=meshes_dir,
        overlays_dir=overlays_dir,
        camera_intrinsics_json=root / "camera_intrinsics.json",
    )


def load_frame_rgb(frame_path: Path) -> np.ndarray:
    """Load an image file as RGB."""
    image_bgr = cv2.imread(str(frame_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not load first frame image: {frame_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a binary object mask from disk."""
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    return (mask_gray > 127).astype("uint8")


def build_default_paths(video_name: str) -> tuple[Path, Path]:
    """Build default input/output roots for a given video name."""
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent
    input_dir = project_dir / "Segment_Video" / "output" / video_name
    output_dir = script_dir / "output"
    return input_dir, output_dir


def build_video_context(
    video_name: str,
    input_dir: Path,
    sam3d: Any,
    mesh_output_root: Path,
) -> VideoGenerationContext:
    """Collect all per-video inputs and derived state needed for generation."""
    output_paths = build_output_paths(mesh_output_root, video_name)

    frames_dir = input_dir / "_frames"
    first_frame_stem = discover_first_frame_stem(input_dir)
    first_frame_path = find_frame_image_path(frames_dir, first_frame_stem)
    if first_frame_path is None:
        raise FileNotFoundError(f"First-frame image not found for stem: {first_frame_stem}")

    image_rgb = load_frame_rgb(first_frame_path)
    objects = discover_objects_with_first_frame_masks(input_dir, first_frame_stem)
    object_color_map = build_object_color_map([name for name, _ in objects])
    object_specs = [
        ObjectMaskSpec(name=name, mask_path=mask_path, color_bgr=object_color_map[name])
        for name, mask_path in objects
    ]

    print(f"Video: {video_name}")
    print(f"First frame: {first_frame_stem}")
    print(f"Objects: {[spec.name for spec in object_specs]}")
    print(f"Output: {output_paths.root}\n")

    print("Estimating intrinsics from first frame...")
    camera_info = estimate_camera_intrinsics(sam3d, image_rgb)
    with output_paths.camera_intrinsics_json.open("w", encoding="utf-8") as f:
        json.dump(camera_info, f, indent=2)
    print(f"Saved: {output_paths.camera_intrinsics_json}")

    return VideoGenerationContext(
        video_name=video_name,
        first_frame_stem=first_frame_stem,
        image_rgb=image_rgb,
        camera_k_overlay=camera_k_from_info(camera_info),
        output_paths=output_paths,
        object_specs=object_specs,
    )


def generate_object_result(
    object_spec: ObjectMaskSpec,
    context: VideoGenerationContext,
    sam3d: Any,
    overlay_backend: QualityRenderBackend,
    overlay_device: str,
    simplify_ratio: float,
) -> GeneratedObjectResult:
    """Generate, pose, render, and save outputs for a single object."""
    mask = load_binary_mask(object_spec.mask_path)
    output = generate_mesh(sam3d, context.image_rgb, mask)
    canonical_mesh_p3d = sam3d_mesh_to_trimesh(output["mesh"][0])

    rotation_quat, translation, scale = extract_pose_components(output)
    posed_mesh_p3d = create_posed_mesh(
        canonical_mesh_p3d,
        rotation_quat,
        translation,
        scale,
    )
    posed_mesh_p3d = postprocess_trimesh_with_sam3d(
        posed_mesh_p3d,
        simplify_ratio=simplify_ratio,
        verbose=False,
    )
    posed_mesh_cv = convert_mesh_p3d_to_cv(posed_mesh_p3d)

    mesh_path = context.output_paths.meshes_dir / f"{object_spec.name}.ply"
    overlay_path = context.output_paths.overlays_dir / f"{object_spec.name}.png"
    posed_mesh_cv.export(str(mesh_path))

    overlay = render_single_object_overlay_quality(
        image_rgb=context.image_rgb,
        posed_mesh=posed_mesh_p3d,
        camera_k=context.camera_k_overlay,
        color_bgr=object_spec.color_bgr,
        backend=overlay_backend,
        device=overlay_device,
        fill_alpha=DEFAULT_OVERLAY_FILL_ALPHA,
        contour_thickness=DEFAULT_OVERLAY_CONTOUR_THICKNESS,
    )
    cv2.imwrite(str(overlay_path), overlay)

    return GeneratedObjectResult(
        name=object_spec.name,
        color_bgr=object_spec.color_bgr,
        posed_mesh_p3d=posed_mesh_p3d,
        mesh_path=mesh_path,
        overlay_path=overlay_path,
    )


def save_combined_overlay(
    context: VideoGenerationContext,
    generated_objects: List[GeneratedObjectResult],
    overlay_backend: QualityRenderBackend,
    overlay_device: str,
) -> Path:
    """Render and save a single overlay containing all generated objects."""
    overlay_all = render_multi_object_overlay_quality(
        image_rgb=context.image_rgb,
        posed_meshes=[obj.posed_mesh_p3d for obj in generated_objects],
        camera_k=context.camera_k_overlay,
        colors_bgr=[obj.color_bgr for obj in generated_objects],
        backend=overlay_backend,
        device=overlay_device,
        fill_alpha=DEFAULT_OVERLAY_FILL_ALPHA,
        contour_thickness=DEFAULT_OVERLAY_CONTOUR_THICKNESS,
    )
    overlay_all = add_overlay_legend(
        overlay_all,
        [(obj.name, obj.color_bgr) for obj in generated_objects],
    )
    combined_overlay_path = context.output_paths.overlays_dir / "combined_overlay.png"
    cv2.imwrite(str(combined_overlay_path), overlay_all)
    return combined_overlay_path


def print_generation_summary(
    generated_objects: List[GeneratedObjectResult],
    failures: List[ObjectGenerationFailure],
) -> None:
    """Print a concise success/failure summary for the video."""
    print(f"\n{'=' * 50}")
    print(
        "Summary: "
        f"{len(generated_objects)} objects generated, {len(failures)} objects failed."
    )
    if failures:
        for failure in failures:
            print(f"  Failed [{failure.name}]: {failure.message}")


def process_video_directory(
    video_name: str,
    input_dir: Path,
    sam3d: Any,
    mesh_output_root: Path,
    overlay_backend: QualityRenderBackend,
    overlay_device: str,
    simplify_ratio: float,
) -> None:
    context = build_video_context(
        video_name=video_name,
        input_dir=input_dir,
        sam3d=sam3d,
        mesh_output_root=mesh_output_root,
    )

    generated_objects: List[GeneratedObjectResult] = []
    failures: List[ObjectGenerationFailure] = []

    for object_spec in context.object_specs:
        print(f"\n{'=' * 50}")
        print(f"Object: {object_spec.name}")
        try:
            result = generate_object_result(
                object_spec=object_spec,
                context=context,
                sam3d=sam3d,
                overlay_backend=overlay_backend,
                overlay_device=overlay_device,
                simplify_ratio=simplify_ratio,
            )
            generated_objects.append(result)
            print(f"    Saved mesh (OpenCV camera coords): {result.mesh_path}")
            print(f"    Saved overlay: {result.overlay_path}")
        except Exception as exc:
            failures.append(ObjectGenerationFailure(name=object_spec.name, message=str(exc)))
            print(f"  Mesh generation failed: {exc}")

    if generated_objects:
        print(f"\n{'=' * 50}")
        print("Generating combined overlay...")
        combined_overlay_path = save_combined_overlay(
            context=context,
            generated_objects=generated_objects,
            overlay_backend=overlay_backend,
            overlay_device=overlay_device,
        )
        print(f"Saved combined overlay: {combined_overlay_path}")

    print_generation_summary(generated_objects, failures)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate first-frame object meshes from Segment_Video output"
        "(sam3d-objects env)."
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="video_01",
        help="Video name used to build default paths for the other arguments.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Segment_Video output dir with _frames/ and objects/. "
        "Defaults to ../Segment_Video/output/<video_name>/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Mesh output root. Final outputs are written to <output_dir>/<video_name>/.",
    )
    parser.add_argument(
        "--simplify_ratio",
        type=float,
        default=0.75,
        help=(
            "SAM3D mesh simplification ratio passed to postprocessing "
            "(0 disables simplification, 0.75 removes about 75 percent of faces)."
        ),
    )
    args = parser.parse_args()

    if not 0.0 <= args.simplify_ratio < 1.0:
        raise ValueError("--simplify_ratio must be in the range [0, 1).")

    default_input_dir, default_output_dir = build_default_paths(args.video_name)

    input_dir = Path(args.input_dir).resolve() if args.input_dir else default_input_dir

    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir

    overlay_backend = QualityRenderBackend(
        torch=torch,
        PerspectiveCameras=PerspectiveCameras,
        MeshRasterizer=MeshRasterizer,
        RasterizationSettings=RasterizationSettings,
        Meshes=Meshes,
    )
    overlay_device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading SAM 3D Objects...")
    sam3d = load_sam3d()
    print("SAM 3D Objects loaded successfully\n")

    process_video_directory(
        video_name=args.video_name,
        input_dir=input_dir,
        sam3d=sam3d,
        mesh_output_root=output_dir,
        overlay_backend=overlay_backend,
        overlay_device=overlay_device,
        simplify_ratio=args.simplify_ratio,
    )

    print(f"\n{'=' * 50}")
    print("Done!")


if __name__ == "__main__":
    main()
