from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from select_target_instance_click import (
    OVERLAY_PALETTE_BGR,
    build_candidate_instances,
    build_mask_stats,
    build_pinhole_intrinsics,
    load_colmap_pose,
    load_json,
    load_mesh,
    rasterize_instance_id_map,
    resolve_input_path,
    resolve_output_dir,
    resolve_scannet_root,
    resolve_scene_paths,
    save_mask,
)


def build_visible_instances(
    instance_id_map: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    visible_instances: list[dict[str, Any]] = []
    visible_masks: dict[int, np.ndarray] = {}

    visible_ids = np.unique(instance_id_map)
    visible_ids = visible_ids[visible_ids >= 0]

    for instance_id in visible_ids.tolist():
        instance_id = int(instance_id)
        mask = instance_id_map == instance_id
        if not np.any(mask):
            continue

        stats = build_mask_stats(mask)
        visible_masks[instance_id] = mask
        visible_instances.append(
            {
                "instance_id": instance_id,
                "visible_area_px": int(stats["mask_area_px"]),
                "visible_bbox_xyxy": stats["visible_bbox_xyxy"],
            }
        )

    visible_instances.sort(
        key=lambda item: (-int(item["visible_area_px"]), int(item["instance_id"]))
    )
    return visible_instances, visible_masks


def compute_overlap_metrics(
    sam_mask: np.ndarray,
    projected_mask: np.ndarray,
) -> dict[str, Any]:
    sam_area = int(np.count_nonzero(sam_mask))
    projected_area = int(np.count_nonzero(projected_mask))
    intersection_px = int(np.count_nonzero(np.logical_and(sam_mask, projected_mask)))
    union_px = sam_area + projected_area - intersection_px

    iou = float(intersection_px / union_px) if union_px > 0 else 0.0
    dice = float((2.0 * intersection_px) / (sam_area + projected_area)) if (sam_area + projected_area) > 0 else 0.0
    precision = float(intersection_px / sam_area) if sam_area > 0 else 0.0
    recall = float(intersection_px / projected_area) if projected_area > 0 else 0.0

    return {
        "sam_area_px": sam_area,
        "projected_area_px": projected_area,
        "intersection_px": intersection_px,
        "union_px": union_px,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


def select_best_instance_match(
    target_mask: np.ndarray,
    visible_instances: list[dict[str, Any]],
    visible_masks: dict[int, np.ndarray],
) -> dict[str, Any]:
    best_match: dict[str, Any] | None = None

    for instance in visible_instances:
        instance_id = int(instance["instance_id"])
        projected_mask = visible_masks[instance_id]
        overlap = compute_overlap_metrics(target_mask, projected_mask)
        if overlap["intersection_px"] <= 0:
            continue

        overall_score = 0.70 * overlap["dice"] + 0.30 * overlap["iou"]
        candidate = {
            **overlap,
            "instance_id": instance_id,
            "overall_score": float(overall_score),
        }
        if best_match is None or candidate["overall_score"] > best_match["overall_score"]:
            best_match = candidate

    if best_match is None:
        raise ValueError(
            "The saved 2D target mask did not overlap any projected 3D instance. "
            "Re-run the 2D stage with a better interaction description or SAM3 checkpoint."
        )
    return best_match


def build_overlay(
    image_bgr: np.ndarray,
    sam_mask: np.ndarray,
    projected_mask: np.ndarray,
    target_label: str,
    matched_instance_id: int,
    overlap_iou: float,
    sam3_score: float,
) -> np.ndarray:
    overlay = image_bgr.copy().astype(np.float32)

    sam_only = np.logical_and(sam_mask, np.logical_not(projected_mask))
    projected_only = np.logical_and(projected_mask, np.logical_not(sam_mask))
    overlap = np.logical_and(sam_mask, projected_mask)

    sam_color = np.array(OVERLAY_PALETTE_BGR[1], dtype=np.float32)
    projected_color = np.array(OVERLAY_PALETTE_BGR[0], dtype=np.float32)
    overlap_color = np.array(OVERLAY_PALETTE_BGR[2], dtype=np.float32)

    overlay[sam_only] = 0.60 * overlay[sam_only] + 0.40 * sam_color
    overlay[projected_only] = 0.60 * overlay[projected_only] + 0.40 * projected_color
    overlay[overlap] = 0.45 * overlay[overlap] + 0.55 * overlap_color
    overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    union_mask = np.logical_or(sam_mask, projected_mask)
    ys, xs = np.where(union_mask)
    if xs.size > 0:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cv2.rectangle(overlay, (x0, y0), (x1, y1), tuple(int(v) for v in OVERLAY_PALETTE_BGR[2]), 2)

    lines = [
        f"target label: {target_label}",
        f"matched instance id: {matched_instance_id}",
        f"IoU {overlap_iou:.3f} | SAM3 score {sam3_score:.3f}",
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


def load_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask image: {path}")
    return mask > 0


def resolve_stage_output_dir(output_root: Path) -> Path:
    return output_root / "3d"


def resolve_target_label(metadata_2d: dict[str, Any]) -> str:
    target_label = str(metadata_2d["sam3_prompt"]).strip()
    if not target_label:
        raise ValueError("target_selection_2d.sam3_prompt must be a non-empty string.")
    return target_label


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Run the 3D target selection stage: load a saved 2D target mask, project the "
            "mesh instances, and resolve the best matching 3D object."
        ),
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument(
        "--metadata-json",
        default=None,
        help="Path to the shared target_selection.json written by the 2D stage.",
    )
    parser.add_argument(
        "--target-mask",
        default=None,
        help="Path to the 2D target mask PNG from the 2D stage.",
    )
    args = parser.parse_args()

    input_dir = resolve_input_path(script_dir, args.video_name, args.input_dir)
    output_root = resolve_output_dir(script_dir, args.video_name, args.outdir)
    stage_output_dir = resolve_stage_output_dir(output_root)
    scannet_root = resolve_scannet_root(script_dir, args.scannet_root)
    metadata_json_path = (
        Path(args.metadata_json).resolve()
        if args.metadata_json
        else output_root / "target_selection.json"
    )
    metadata_payload = load_json(metadata_json_path)
    metadata_2d = metadata_payload["target_selection_2d"]
    target_label = resolve_target_label(metadata_2d)
    input_payload = load_json(input_dir / "input_pag.json")
    scene_context = input_payload["scene_context"]

    target_mask_path = (
        Path(args.target_mask).resolve()
        if args.target_mask
        else output_root / str(metadata_2d["mask_path"])
    )
    target_mask = load_binary_mask(target_mask_path)

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
    if target_mask.shape != (height, width):
        raise ValueError(
            "Saved target mask shape does not match the camera dimensions: "
            f"mask={target_mask.shape[::-1]}, metadata={(width, height)}"
        )

    verts_world, faces = load_mesh(scene_paths["mesh_path"])
    segments_payload = load_json(scene_paths["segments_path"])
    seg_indices = np.asarray(segments_payload["segIndices"], dtype=np.int64)
    anno_payload = load_json(scene_paths["segments_anno_path"])

    candidate_faces, face_instance_ids, _instance_meta = build_candidate_instances(
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
    visible_instances, visible_masks = build_visible_instances(instance_id_map)
    if not visible_instances:
        raise ValueError("No visible 3D object instances were found in the selected camera view.")

    best_match = select_best_instance_match(
        target_mask=target_mask,
        visible_instances=visible_instances,
        visible_masks=visible_masks,
    )
    matched_instance_id = int(best_match["instance_id"])
    projected_mask = visible_masks[matched_instance_id]
    sam3_stats = build_mask_stats(target_mask)
    projected_stats = build_mask_stats(projected_mask)
    overlay_bgr = build_overlay(
        image_bgr=image_bgr,
        sam_mask=target_mask,
        projected_mask=projected_mask,
        target_label=target_label,
        matched_instance_id=matched_instance_id,
        overlap_iou=float(best_match["iou"]),
        sam3_score=float(metadata_2d["target_mask_score"]),
    )

    stage_output_dir.mkdir(parents=True, exist_ok=True)
    scene_image_path = stage_output_dir / "scene_image.png"
    projected_mask_path = stage_output_dir / "projected_target_mask.png"
    overlay_path = stage_output_dir / "target_overlay.png"
    selection_json_path = output_root / "target_selection.json"

    cv2.imwrite(str(scene_image_path), image_bgr)
    save_mask(projected_mask_path, projected_mask)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    selection_payload = {
        "target_selection_2d": {
            "sam3_prompt": target_label,
            "target_mask_score": float(metadata_2d["target_mask_score"]),
            "mask_path": str(metadata_2d["mask_path"]),
        },
        "target_selection_3d": {
            "projected_mask_path": str(Path("3d") / projected_mask_path.name),
            "overlay_path": str(Path("3d") / overlay_path.name),
            "overlap_iou": float(best_match["iou"]),
            "projected_mask_area_px": int(projected_stats["mask_area_px"]),
        },
        "target_selection": {
            "selection_source": "sam3_text_prompt",
            "instance_id": matched_instance_id,
            "object": target_label,
        },
    }
    selection_json_path.write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input directory: {input_dir}")
    print(f"Loaded 2D metadata JSON: {metadata_json_path}")
    print(f"Loaded target mask: {target_mask_path}")
    print(f"Output directory: {stage_output_dir}")
    print(f"Saved scene image copy: {scene_image_path}")
    print(f"Saved projected target mask: {projected_mask_path}")
    print(f"Saved overlay image: {overlay_path}")
    print(f"Updated shared selection JSON: {selection_json_path}")
    print(
        "Selected 3D target:",
        {
            "instance_id": matched_instance_id,
            "object": target_label,
            "selected_sam3_prompt": target_label,
            "overlap_iou": best_match["iou"],
            "sam3_model_score": metadata_2d["target_mask_score"],
            "target_mask_area_px": sam3_stats["mask_area_px"],
            "projected_mask_area_px": projected_stats["mask_area_px"],
        },
    )


if __name__ == "__main__":
    main()
