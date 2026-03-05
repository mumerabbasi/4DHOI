"""Generate meshes by reusing first-frame canonical meshes across sampled frames."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import trimesh
from mesh_generation_utils import (
    compute_overlay_focal_scale,
    create_posed_mesh,
    discover_frames,
    discover_objects,
    estimate_camera_intrinsics,
    extract_pose_data,
    find_frame_image_path,
    generate_mesh,
    load_sam3d,
    sam3d_mesh_to_trimesh,
    sample_frames_uniformly,
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


def save_canonical_mesh(canonical_mesh: trimesh.Trimesh, canonical_mesh_path: Path) -> None:
    canonical_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_mesh.export(str(canonical_mesh_path))
    print(f"    Saved canonical mesh: {canonical_mesh_path.name}")


def save_posed_mesh_and_overlay(
    posed_mesh: trimesh.Trimesh,
    output_dir: Path,
    image_rgb,
    camera_k,
    object_color_bgr: Tuple[int, int, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    posed_mesh_path = output_dir / "mesh_posed.ply"
    posed_mesh.export(str(posed_mesh_path))
    print(f"    Saved: {posed_mesh_path.name}")

    overlay = render_single_object_overlay_quality(
        image_rgb=image_rgb,
        posed_mesh=posed_mesh,
        camera_k=camera_k,
        color_bgr=object_color_bgr,
        fill_alpha=0.35,
        contour_thickness=2,
    )
    overlay_path = output_dir / "mesh_posed_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"    Saved: {overlay_path.name}")


def process_video_directory(
    input_dir: Path,
    sam3d: Any,
    mesh_output_root: Path,
    num_frames: int = 4,
    focal_length_mm: Optional[float] = None,
    f_scale: float = 0.8,
) -> None:
    objects = discover_objects(input_dir)
    frame_stems = discover_frames(input_dir)
    sampled = sample_frames_uniformly(frame_stems, num_frames)
    if not sampled:
        raise RuntimeError("No sampled frames available.")

    video_name = input_dir.name
    output_root = (mesh_output_root / video_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frames_dir = input_dir / "_frames"

    print(f"Video: {video_name}")
    print(f"Objects: {[n for n, _ in objects]}")
    print(f"Total frames: {len(frame_stems)}, sampled: {len(sampled)}")
    print(f"Sampled frames: {sampled}")
    print(f"Output: {output_root}\n")

    object_color_map = build_object_color_map([name for name, _ in objects])

    first_frame_stem = sampled[0]
    first_frame_path = find_frame_image_path(frames_dir, first_frame_stem)
    if first_frame_path is None:
        raise FileNotFoundError(f"Frame image not found for first sampled frame: {first_frame_stem}")

    first_frame_bgr = cv2.imread(str(first_frame_path))
    if first_frame_bgr is None:
        raise RuntimeError(f"Could not load first sampled frame: {first_frame_path}")
    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)

    print(f"Estimating intrinsics from first sampled frame only: {first_frame_stem}")
    camera_info_shared = estimate_camera_intrinsics(sam3d, first_frame_rgb)

    if focal_length_mm is None:
        camera_info_shared = scale_camera_intrinsics(camera_info_shared, f_scale)
        focal_overlay_mm = float(camera_info_shared["blender_recommendation"]["lens_mm"])
        overlay_focal_scale = 1.0
        print(
            "Focal mode: auto+f_scale "
            f"(f_scale={f_scale}, focal_for_projection={focal_overlay_mm:.3f}mm)"
        )
    else:
        focal_overlay_mm = float(focal_length_mm)
        overlay_focal_scale = compute_overlay_focal_scale(camera_info_shared, focal_overlay_mm)
        print(
            "Focal mode: explicit --focal_length "
            f"(focal_for_projection={focal_overlay_mm:.3f}mm, focal_scale={overlay_focal_scale:.4f})"
        )

    canonical_meshes: Dict[str, trimesh.Trimesh] = {}
    first_frame_outputs: Dict[str, Dict[str, Any]] = {}

    print(f"\nPrecomputing canonical meshes from first sampled frame: {first_frame_stem}")
    first_frame_output_dir = output_root / first_frame_stem
    first_frame_output_dir.mkdir(parents=True, exist_ok=True)

    for obj_name, mask_dir in objects:
        print(f"\n{'=' * 50}")
        print(f"  Canonical object: {obj_name}  |  Frame: {first_frame_stem}")

        mask_path = mask_dir / f"{first_frame_stem}.png"
        if not mask_path.exists():
            print(f"    Warning: mask not found for canonical frame - {mask_path}; skipping object.")
            continue

        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"    Warning: failed to read mask for canonical frame - {mask_path}; skipping object.")
            continue
        mask = (mask_gray > 127).astype("uint8")

        try:
            output = generate_mesh(sam3d, first_frame_rgb, mask)
            canonical_mesh = sam3d_mesh_to_trimesh(output["mesh"][0])
            canonical_meshes[obj_name] = canonical_mesh
            first_frame_outputs[obj_name] = output
            save_canonical_mesh(canonical_mesh, first_frame_output_dir / obj_name / "mesh.ply")
        except Exception as exc:
            print(f"    Canonical mesh generation failed: {exc}")

    if not canonical_meshes:
        raise RuntimeError(f"Failed to build canonical meshes from first sampled frame: {first_frame_stem}")

    for frame_idx, frame_stem in enumerate(sampled):
        print(f"\n{'#' * 60}")
        print(f"Frame {frame_idx + 1}/{len(sampled)}: {frame_stem}")
        print(f"{'#' * 60}")

        frame_path = find_frame_image_path(frames_dir, frame_stem)
        if frame_path is None:
            print(f"  Error: frame image not found for {frame_stem}")
            continue

        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            print(f"  Error: could not load {frame_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        frame_output_dir = output_root / frame_stem
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        cam_json = frame_output_dir / "camera_intrinsics.json"
        with cam_json.open("w", encoding="utf-8") as f:
            json.dump(camera_info_shared, f, indent=2)
        print(f"  Saved frame intrinsics: {cam_json}")

        camera_k_overlay = camera_k_from_info(camera_info_shared, focal_scale=overlay_focal_scale)
        posed_meshes: List[trimesh.Trimesh] = []
        posed_mesh_colors: List[Tuple[int, int, int]] = []

        for obj_name, mask_dir in objects:
            print(f"\n{'=' * 50}")
            print(f"  Object: {obj_name}  |  Frame: {frame_stem}")

            canonical_mesh = canonical_meshes.get(obj_name)
            if canonical_mesh is None:
                print("    Warning: canonical mesh unavailable from first sampled frame; skipping object.")
                continue

            try:
                if frame_stem == first_frame_stem:
                    output = first_frame_outputs.get(obj_name)
                    if output is None:
                        print("    Warning: missing cached first-frame output; skipping object.")
                        continue
                else:
                    mask_path = mask_dir / f"{frame_stem}.png"
                    if not mask_path.exists():
                        print(f"    Warning: mask not found - {mask_path}; skipping.")
                        continue
                    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_gray is None:
                        print(f"    Warning: failed to read mask - {mask_path}; skipping.")
                        continue
                    mask = (mask_gray > 127).astype("uint8")
                    output = generate_mesh(sam3d, image_rgb, mask)

                pose_data = extract_pose_data(output)
                print(f"    Rotation (quat): {pose_data['rotation_quat']}")
                print(f"    Rotation (euler xyz deg): {pose_data['euler_xyz_deg']}")
                print(f"    Translation: {pose_data['translation']}")
                print(f"    Scale: {pose_data['scale']}")

                object_frame_dir = frame_output_dir / obj_name
                object_frame_dir.mkdir(parents=True, exist_ok=True)

                if frame_stem == first_frame_stem:
                    canonical_path = object_frame_dir / "mesh.ply"
                    if not canonical_path.exists():
                        save_canonical_mesh(canonical_mesh, canonical_path)

                save_pose_json(
                    output=output,
                    pose_data=pose_data,
                    output_path=object_frame_dir / "pose.json",
                    focal_length_mm=focal_overlay_mm,
                    camera_intrinsics_json=cam_json,
                )
                print("    Saved: pose.json")

                posed_mesh = create_posed_mesh(
                    canonical_mesh,
                    pose_data["rotation_quat"],
                    pose_data["translation"],
                    pose_data["scale"],
                )

                save_posed_mesh_and_overlay(
                    posed_mesh=posed_mesh,
                    output_dir=object_frame_dir,
                    image_rgb=image_rgb,
                    camera_k=camera_k_overlay,
                    object_color_bgr=object_color_map[obj_name],
                )

                posed_meshes.append(posed_mesh)
                posed_mesh_colors.append(object_color_map[obj_name])
            except Exception as exc:
                print(f"    Mesh generation failed: {exc}")

        if posed_meshes:
            print(f"\n{'=' * 50}")
            print(f"Generating combined overlay for {frame_stem}...")
            try:
                overlay = render_multi_object_overlay_quality(
                    image_rgb=image_rgb,
                    posed_meshes=posed_meshes,
                    camera_k=camera_k_overlay,
                    colors_bgr=posed_mesh_colors,
                    fill_alpha=0.35,
                    contour_thickness=2,
                )
                overlay_path = frame_output_dir / "all_objects_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
                print(f"Saved: {overlay_path}")
            except Exception as exc:
                print(f"Failed to generate combined overlay: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3D meshes from Segment_Video output (sam3d-objects env).",
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
        default="./output_first_frame_canonical",
        help="Mesh output root (<output_dir>/<video_xx>/).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=4,
        help="Number of frames to uniformly sample (default: 4).",
    )
    parser.add_argument(
        "--focal_length",
        type=float,
        default=None,
        help="Focal length in mm for projection (default: auto from first sampled frame).",
    )
    parser.add_argument(
        "--f_scale",
        type=float,
        default=0.9,
        help="Scale factor for auto-estimated fx/fy/lens from first sampled frame (default: 0.8).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path(__file__).parent / input_dir
    input_dir = input_dir.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir
    output_dir = output_dir.resolve()

    ensure_quality_backend_available()

    print("Loading SAM 3D Objects...")
    sam3d = load_sam3d()
    print("SAM 3D Objects loaded successfully\n")

    focal_msg = f"{args.focal_length:.3f}mm" if args.focal_length is not None else "auto (MoGe first sampled frame)"
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Focal length: {focal_msg}")
    print(f"Focal scale (auto mode only): {args.f_scale}")
    print(f"Num frames: {args.num_frames}\n")

    process_video_directory(
        input_dir=input_dir,
        sam3d=sam3d,
        mesh_output_root=output_dir,
        num_frames=args.num_frames,
        focal_length_mm=args.focal_length,
        f_scale=args.f_scale,
    )

    print(f"\n{'=' * 50}")
    print("Done!")


if __name__ == "__main__":
    main()
