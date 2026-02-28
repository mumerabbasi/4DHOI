import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Global constants.
# ---------------------------------------------------------------------------

DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)

PART_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (255, 225, 25),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (170, 110, 40),
]

BODY_PART_TO_SEG_KEYS: dict[str, list[str]] = {
    "left hand": ["leftHand", "leftHandIndex1"],
    "right hand": ["rightHand", "rightHandIndex1"],
    "left foot": ["leftFoot", "leftToeBase"],
    "right foot": ["rightFoot", "rightToeBase"],
    "left shoulder": ["leftShoulder"],
    "right shoulder": ["rightShoulder"],
    "left arm": ["leftArm", "leftForeArm"],
    "right arm": ["rightArm", "rightForeArm"],
    "left leg": ["leftUpLeg", "leftLeg"],
    "right leg": ["rightUpLeg", "rightLeg"],
    "left hip": ["leftUpLeg"],
    "right hip": ["rightUpLeg"],
    "hips": ["hips"],
    "head": ["head"],
    "neck": ["neck"],
    "spine": ["spine", "spine1", "spine2"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_vertex_part_map(
    seg: dict[str, list[int]],
    part_specs: list[dict[str, Any]],
    part_colors: dict[str, tuple[int, int, int]],
    num_verts: int = 6890,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return per‑vertex part label (str) and per‑vertex RGB colour array.

    Vertices that belong to none of the requested parts receive
    ``DEFAULT_COLOR`` and the label ``"other"``.
    """
    vert_labels: list[str] = ["other"] * num_verts
    vert_colors = np.tile(np.array(DEFAULT_COLOR, dtype=np.uint8), (num_verts, 1))

    for part_spec in part_specs:
        part_name = str(part_spec["part_name"])
        color = np.array(part_colors[part_name], dtype=np.uint8)
        for seg_key in part_spec["seg_keys"]:
            for vi in seg.get(seg_key, []):
                if vi < num_verts:
                    vert_labels[vi] = part_name
                    vert_colors[vi] = color

    vert_labels_arr = np.array(vert_labels)
    vertex_counts = {
        str(part_spec["part_name"]): int((vert_labels_arr == str(part_spec["part_name"])).sum())
        for part_spec in part_specs
    }
    return vert_labels_arr, vert_colors, vertex_counts


def _normalize_name(name: str) -> str:
    text = re.sub(r"[_\-]+", " ", name.strip().lower())
    return re.sub(r"\s+", " ", text).strip()


def _material_name(name: str) -> str:
    return _normalize_name(name).replace(" ", "_")


def _compact_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _normalize_name(name))


def _find_single_pag_json(pag_dir: Path) -> Path:
    pag_files = sorted(pag_dir.glob("output_pag_*.json"))
    if not pag_files:
        raise FileNotFoundError(f"No PAG JSON found in: {pag_dir}")
    if len(pag_files) > 1:
        names = [p.name for p in pag_files]
        raise RuntimeError(f"Expected exactly one PAG JSON in {pag_dir}, found: {names}")
    return pag_files[0]


def _extract_pag_body_parts(pag_json_path: Path) -> list[str]:
    with pag_json_path.open("r", encoding="utf-8") as f:
        pag = json.load(f)

    body_nodes = pag.get("body part nodes", [])
    if not isinstance(body_nodes, list):
        raise ValueError(f"Invalid PAG format at {pag_json_path}: 'body part nodes' must be a list")

    ordered_parts: list[str] = []
    seen: set[str] = set()
    for node in body_nodes:
        if not isinstance(node, str):
            continue
        part_name = node.split(",", 1)[1] if "," in node else node
        part_name = _normalize_name(part_name)
        if not part_name:
            continue
        if part_name in seen:
            continue
        seen.add(part_name)
        ordered_parts.append(part_name)

    return ordered_parts


def _build_part_specs_from_pag(
    body_parts: list[str],
    seg_keys_available: set[str],
) -> list[dict[str, Any]]:
    seg_compact_to_key = {_compact_name(k): k for k in seg_keys_available}
    specs_by_name: dict[str, dict[str, Any]] = {}

    for body_part in body_parts:
        seg_keys = BODY_PART_TO_SEG_KEYS.get(body_part, [])
        if not seg_keys:
            direct = seg_compact_to_key.get(_compact_name(body_part))
            if direct is not None:
                seg_keys = [direct]

        seg_keys = [k for k in seg_keys if k in seg_keys_available]
        if not seg_keys:
            print(f"[WARN] No SMPL segmentation key found for PAG body part: '{body_part}'")
            continue

        part_name = _material_name(body_part)
        spec = specs_by_name.get(part_name)
        if spec is None:
            specs_by_name[part_name] = {
                "part_name": part_name,
                "source_names": [body_part],
                "seg_keys": list(dict.fromkeys(seg_keys)),
            }
        else:
            if body_part not in spec["source_names"]:
                spec["source_names"].append(body_part)
            spec["seg_keys"] = list(dict.fromkeys(spec["seg_keys"] + seg_keys))

    return list(specs_by_name.values())


def _assign_part_colors(part_specs: list[dict[str, Any]]) -> dict[str, tuple[int, int, int]]:
    colors: dict[str, tuple[int, int, int]] = {}
    for idx, part_spec in enumerate(part_specs):
        colors[str(part_spec["part_name"])] = PART_COLOR_PALETTE[idx % len(PART_COLOR_PALETTE)]
    return colors


def _face_part_labels(
    faces: np.ndarray,
    vert_labels: np.ndarray,
) -> np.ndarray:
    """Assign each face to a part based on majority vote of its vertices."""
    labels = []
    for f in faces:
        vl = [vert_labels[f[0]], vert_labels[f[1]], vert_labels[f[2]]]
        # Majority vote; if no majority, pick the first non‑"other" or "other"
        from collections import Counter
        c = Counter(vl)
        label = c.most_common(1)[0][0]
        labels.append(label)
    return np.array(labels)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_mtl(path: Path, part_colors: dict[str, tuple[int, int, int]]) -> None:
    """Write a shared .mtl material library."""
    with path.open("w") as f:
        # Default / other
        f.write("newmtl other\n")
        r, g, b = [c / 255.0 for c in DEFAULT_COLOR]
        f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n\n")

        for part_name, color_rgb in part_colors.items():
            f.write(f"newmtl {part_name}\n")
            r, g, b = [c / 255.0 for c in color_rgb]
            f.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n\n")


def _write_part_color_mapping(
    path: Path,
    pag_json_path: Path,
    part_specs: list[dict[str, Any]],
    part_colors: dict[str, tuple[int, int, int]],
    vertex_counts: dict[str, int],
) -> None:
    payload = {
        "pag_json_path": str(pag_json_path),
        "default_label": "other",
        "default_color_rgb": list(DEFAULT_COLOR),
        "parts": [],
    }

    for part_spec in part_specs:
        part_name = str(part_spec["part_name"])
        payload["parts"].append(
            {
                "part_name": part_name,
                "source_names": list(part_spec["source_names"]),
                "segmentation_keys": list(part_spec["seg_keys"]),
                "color_rgb": list(part_colors[part_name]),
                "num_vertices": int(vertex_counts.get(part_name, 0)),
            }
        )

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    mtl_filename: str,
) -> None:
    """Write a Wavefront .obj with per‑part material colours."""
    with path.open("w") as f:
        f.write(f"mtllib {mtl_filename}\n")

        # Vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Group faces by material
        all_labels = list(dict.fromkeys(face_labels))  # unique, order‑preserved
        for label in all_labels:
            f.write(f"\nusemtl {label}\n")
            mask = face_labels == label
            for face in faces[mask]:
                # OBJ uses 1‑based indices
                f.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def write_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vert_colors: np.ndarray,
) -> None:
    """Write an ASCII PLY with per‑vertex RGB colours."""
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v, c in zip(vertices, vert_colors):
            f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per‑frame segmented (coloured) human meshes from GVHMR results."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="output/video_03",
        help="Directory containing hmr4d_results.pt.",
    )
    parser.add_argument(
        "--smpl_folder",
        type=str,
        default="../../GVHMR/inputs/checkpoints/body_models/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save exported mesh files. If not provided, defaults "
            "to <video_dir>/output_segmented."
        ),
    )
    parser.add_argument(
        "--pag_json_path",
        type=str,
        default=None,
        help=(
            "Path to PAG JSON. If not provided, defaults to "
            "../Generate_PAG/output/<video_name>/output_pag_*.json."
        ),
    )
    parser.add_argument(
        "--smplx2smpl_path",
        type=str,
        default="../../GVHMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        help="Path to the smplx2smpl sparse matrix from GVHMR.",
    )
    parser.add_argument(
        "--seg_json_path",
        type=str,
        default="../../GVHMR/hmr4d/utils/body_model/smpl_vert_segmentation.json",
        help="Path to SMPL vertex segmentation JSON.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["obj", "ply"],
        default="obj",
        help="Output mesh format (default: obj).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    video_dir = Path(args.video_dir).resolve()
    result_path = video_dir / "hmr4d_results.pt"
    smpl_folder = Path(args.smpl_folder).resolve()
    smplx2smpl_path = Path(args.smplx2smpl_path).resolve()
    seg_json_path = Path(args.seg_json_path).resolve()
    if args.pag_json_path is None:
        pag_dir = (script_dir.parent / "Generate_PAG" / "output" / video_dir.name).resolve()
        pag_json_path = _find_single_pag_json(pag_dir)
    else:
        pag_json_path = Path(args.pag_json_path).resolve()

    if not video_dir.exists() or not video_dir.is_dir():
        raise NotADirectoryError(f"Video directory not found: {video_dir}")
    if not result_path.exists():
        raise FileNotFoundError(f"Could not find hmr4d_results.pt in: {video_dir}")
    if not seg_json_path.exists():
        raise FileNotFoundError(f"Segmentation JSON not found: {seg_json_path}")
    if not pag_json_path.exists():
        raise FileNotFoundError(f"PAG JSON not found: {pag_json_path}")

    if args.output_dir is None:
        output_dir = video_dir / "output_segmented"
    else:
        output_dir = Path(args.output_dir).resolve()

    fmt = args.format

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading data from {result_path}...")
    data = torch.load(result_path)

    params = data["smpl_params_incam"]
    body_pose = params["body_pose"]
    betas = params["betas"]
    global_orient = params["global_orient"]
    transl = params["transl"].clone()

    num_frames = body_pose.shape[0]
    print(f"Found {num_frames} frames.")
    print(f"Input Pose Shape: {body_pose.shape}")

    # ------------------------------------------------------------------
    # 2. Set up SMPL‑X model
    # ------------------------------------------------------------------
    print(f"Loading SMPL-X from {smpl_folder}...")
    smplx_layer = smplx.create(
        str(smpl_folder),
        model_type="smplx",
        gender="neutral",
        num_pca_comps=12,
        flat_hand_mean=False,
        create_body_pose=False,
        create_betas=False,
        create_global_orient=False,
        create_transl=False,
    )

    print(f"Loading smplx2smpl matrix from {smplx2smpl_path}...")
    smplx2smpl = torch.load(smplx2smpl_path)

    smpl_layer_for_faces = smplx.create(
        str(smpl_folder),
        model_type="smpl",
        gender="neutral",
    )
    faces_smpl = smpl_layer_for_faces.faces  # (F, 3) int array

    # ------------------------------------------------------------------
    # 3. Load PAG body parts + vertex segmentation and build colour map
    # ------------------------------------------------------------------
    print(f"Loading PAG from {pag_json_path}...")
    body_parts = _extract_pag_body_parts(pag_json_path)
    if not body_parts:
        raise RuntimeError(f"No 'body part nodes' found in PAG: {pag_json_path}")

    print(f"Loading segmentation from {seg_json_path}...")
    with seg_json_path.open() as f:
        seg = json.load(f)

    part_specs = _build_part_specs_from_pag(body_parts, set(seg.keys()))
    if not part_specs:
        raise RuntimeError(
            "No PAG body parts could be mapped to SMPL segmentation keys. "
            f"PAG: {pag_json_path}"
        )

    part_colors = _assign_part_colors(part_specs)
    vert_labels, vert_colors, vertex_counts = _build_vertex_part_map(
        seg=seg,
        part_specs=part_specs,
        part_colors=part_colors,
        num_verts=6890,
    )
    face_labels = _face_part_labels(faces_smpl, vert_labels)

    # ------------------------------------------------------------------
    # 4. Prepare output directory
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    part_mapping_path = output_dir / "part_color_mapping.json"
    _write_part_color_mapping(
        path=part_mapping_path,
        pag_json_path=pag_json_path,
        part_specs=part_specs,
        part_colors=part_colors,
        vertex_counts=vertex_counts,
    )
    print(f"Wrote {part_mapping_path}")

    if fmt == "obj":
        mtl_filename = "materials.mtl"
        _write_mtl(output_dir / mtl_filename, part_colors=part_colors)
        print(f"Wrote {output_dir / mtl_filename}")

    # ------------------------------------------------------------------
    # 5. Export frames
    # ------------------------------------------------------------------
    print(f"Exporting frames to .{fmt}...")
    for i in tqdm(range(num_frames)):
        curr_betas = betas[i: i + 1] if betas.shape[0] > 1 else betas[:1]

        output = smplx_layer(
            betas=curr_betas,
            body_pose=body_pose[i: i + 1],
            global_orient=global_orient[i: i + 1],
            transl=transl[i: i + 1],
        )

        smplx_verts = output.vertices[0]  # (10475, 3)
        smpl_verts = torch.matmul(smplx2smpl, smplx_verts)  # (6890, 3)
        vertices = smpl_verts.detach().cpu().numpy()

        filename = output_dir / f"frame_{i:04d}.{fmt}"

        if fmt == "obj":
            write_obj(filename, vertices, faces_smpl, face_labels, mtl_filename)
        else:
            write_ply(filename, vertices, faces_smpl, vert_colors)

    print(f"\nDone! {num_frames} files saved in: {output_dir}")


if __name__ == "__main__":
    main()
