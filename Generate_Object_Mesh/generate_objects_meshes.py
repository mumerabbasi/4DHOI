"""Generate object meshes from first-frame Segment_Video masks."""

import argparse
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import trimesh
from mesh_generation_utils import (
    compute_overlay_focal_scale,
    create_posed_mesh,
    discover_first_frame_stem,
    discover_objects_with_first_frame_masks,
    estimate_camera_intrinsics,
    extract_pose_data,
    find_frame_image_path,
    generate_mesh,
    load_sam3d,
    sam3d_mesh_to_trimesh,
    save_pose_json,
    scale_camera_intrinsics,
)
from rendering_utils import (
    build_object_color_map,
    camera_k_from_info,
    ensure_quality_backend_available,
    render_multi_object_overlay_quality,
    render_single_object_overlay_quality,
)


def process_video_directory(
    input_dir: Path,
    sam3d: Any,
    mesh_output_root: Path,
    focal_length_mm: Optional[float] = None,
    f_scale: float = 0.9,
) -> None:
    video_name = input_dir.name
    output_root = (mesh_output_root / video_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frames_dir = input_dir / "_frames"
    first_frame_stem = discover_first_frame_stem(input_dir)
    first_frame_path = find_frame_image_path(frames_dir, first_frame_stem)
    if first_frame_path is None:
        raise FileNotFoundError(f"First-frame image not found for stem: {first_frame_stem}")

    objects = discover_objects_with_first_frame_masks(input_dir, first_frame_stem)
    object_color_map = build_object_color_map([name for name, _ in objects])

    print(f"Video: {video_name}")
    print(f"First frame: {first_frame_stem}")
    print(f"Objects: {[name for name, _ in objects]}")
    print(f"Output: {output_root}\n")

    image_bgr = cv2.imread(str(first_frame_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not load first frame image: {first_frame_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("Estimating intrinsics from first frame...")
    camera_info = estimate_camera_intrinsics(sam3d, image_rgb)

    if focal_length_mm is None:
        camera_info = scale_camera_intrinsics(camera_info, f_scale)
        focal_overlay_mm = float(camera_info["blender_recommendation"]["lens_mm"])
        overlay_focal_scale = 1.0
        print(
            "Focal mode: auto+f_scale "
            f"(f_scale={f_scale}, focal_for_projection={focal_overlay_mm:.3f}mm)"
        )
    else:
        focal_overlay_mm = float(focal_length_mm)
        overlay_focal_scale = compute_overlay_focal_scale(camera_info, focal_overlay_mm)
        print(
            "Focal mode: explicit --focal_length "
            f"(focal_for_projection={focal_overlay_mm:.3f}mm, focal_scale={overlay_focal_scale:.4f})"
        )

    camera_info["focal_length_mm_used_for_overlay"] = float(focal_overlay_mm)
    if focal_length_mm is not None:
        camera_info["focal_length_mm_user_override"] = float(focal_length_mm)

    camera_intrinsics_json = output_root / "camera_intrinsics.json"
    with camera_intrinsics_json.open("w", encoding="utf-8") as f:
        json.dump(camera_info, f, indent=2)
    print(f"Saved: {camera_intrinsics_json}")

    camera_k_overlay = camera_k_from_info(camera_info, focal_scale=overlay_focal_scale)

    posed_meshes: List[trimesh.Trimesh] = []
    posed_colors: List[Tuple[int, int, int]] = []

    for obj_name, mask_path in objects:
        print(f"\n{'=' * 50}")
        print(f"Object: {obj_name}")

        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"  Warning: failed to read mask: {mask_path}")
            continue
        mask = (mask_gray > 127).astype("uint8")

        try:
            output = generate_mesh(sam3d, image_rgb, mask)
            canonical_mesh = sam3d_mesh_to_trimesh(output["mesh"][0])

            pose_data = extract_pose_data(output)
            posed_mesh = create_posed_mesh(
                canonical_mesh,
                pose_data["rotation_quat"],
                pose_data["translation"],
                pose_data["scale"],
            )

            obj_out_dir = output_root / obj_name
            obj_out_dir.mkdir(parents=True, exist_ok=True)

            canonical_mesh.export(str(obj_out_dir / "mesh.ply"))
            posed_mesh.export(str(obj_out_dir / "mesh_posed.ply"))

            save_pose_json(
                output=output,
                pose_data=pose_data,
                output_path=obj_out_dir / "pose.json",
                focal_length_mm=focal_overlay_mm,
                camera_intrinsics_json=camera_intrinsics_json,
            )

            overlay = render_single_object_overlay_quality(
                image_rgb=image_rgb,
                posed_mesh=posed_mesh,
                camera_k=camera_k_overlay,
                color_bgr=object_color_map[obj_name],
                fill_alpha=0.35,
                contour_thickness=2,
            )
            cv2.imwrite(str(obj_out_dir / "mesh_posed_overlay.png"), overlay)

            print("    Saved: mesh.ply")
            print("    Saved: pose.json")
            print("    Saved: mesh_posed.ply")
            print("    Saved: mesh_posed_overlay.png")

            posed_meshes.append(posed_mesh)
            posed_colors.append(object_color_map[obj_name])
        except Exception as exc:
            print(f"  Mesh generation failed: {exc}")

    if posed_meshes:
        print(f"\n{'=' * 50}")
        print("Generating combined quality overlay...")
        overlay_all = render_multi_object_overlay_quality(
            image_rgb=image_rgb,
            posed_meshes=posed_meshes,
            camera_k=camera_k_overlay,
            colors_bgr=posed_colors,
            fill_alpha=0.35,
            contour_thickness=2,
        )
        overlay_path = output_root / f"{first_frame_stem}_all_objects_overlay.png"
        cv2.imwrite(str(overlay_path), overlay_all)
        print(f"Saved: {overlay_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate first-frame object meshes from Segment_Video output (sam3d-objects env)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../Segment_Video/output/video_01",
        help="Segment_Video output dir with _frames/ and objects/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_first_frame",
        help="Mesh output root (<output_dir>/<video_xx>/).",
    )
    parser.add_argument(
        "--focal_length",
        type=float,
        default=None,
        help="Focal length in mm for projection (default: auto from first frame).",
    )
    parser.add_argument(
        "--f_scale",
        type=float,
        default=1.0,
        help="Scale factor for auto-estimated fx/fy/lens.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path(__file__).parent / input_dir
    input_dir = input_dir.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / args.output_dir
    output_dir = output_dir.resolve()

    ensure_quality_backend_available()

    print("Loading SAM 3D Objects...")
    sam3d = load_sam3d()
    print("SAM 3D Objects loaded successfully\n")

    process_video_directory(
        input_dir=input_dir,
        sam3d=sam3d,
        mesh_output_root=output_dir,
        focal_length_mm=args.focal_length,
        f_scale=args.f_scale,
    )

    print(f"\n{'=' * 50}")
    print("Done!")


if __name__ == "__main__":
    main()
