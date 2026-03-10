"""Problem loading and preprocessing for joint human-object mesh refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

from geometry import decompose_T
from models import (
    HumanData,
    ObjectData,
    ObjectPartSegments,
    PAG,
    PAGEdge,
    PAGObjectState,
    PackedPointCloud2D,
    ProblemContext,
    ResolvedEdge,
    SDFGrid,
)
from utils import (
    ensure_dir,
    list_images,
    resolve_dirs,
    resolve_frames_dir,
    resolve_pag_path,
    resolve_smpl_seg,
)


OBJECT_COLORS_BGR: list[tuple[int, int, int]] = [
    (0, 255, 255),
    (255, 128, 0),
    (0, 255, 0),
    (255, 0, 255),
    (128, 255, 128),
    (0, 128, 255),
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

MASK_SAMPLE_SEED = 2026
SURFACE_SAMPLE_SEED = 7


def _load_intrinsics_from_alignment_summary(
    aligned_mesh_video_dir: Path,
) -> tuple[np.ndarray, Path]:
    summary_path = aligned_mesh_video_dir / "alignment_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    k = np.asarray(payload["camera"]["intrinsics_3x3"], dtype=np.float32)
    return k.reshape(3, 3), summary_path


def _sanitize(name: str) -> str:
    return name.strip().replace(" ", "_")


def _parse_pag(pag_path: Path) -> PAG:
    with pag_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    obj_states: list[PAGObjectState] = []
    for item in data.get("object states", []):
        name = item["name"].strip()
        obj_states.append(
            PAGObjectState(
                name=name,
                slug=_sanitize(name),
                is_translational=bool(item.get("is_translational", True)),
                is_rotational=bool(item.get("is_rotational", True)),
            )
        )

    body_nodes = [s.strip() for s in data.get("body part nodes", [])]
    obj_nodes = [s.strip() for s in data.get("object part nodes", [])]

    edges: list[PAGEdge] = []
    for edge in data.get("interaction edges", []):
        nodes = edge["nodes"]
        edges.append(
            PAGEdge(
                node_a=nodes[0].strip(),
                node_b=nodes[1].strip(),
                is_continuous=bool(edge.get("is_continuous", True)),
                is_rel_static=bool(edge.get("is_rel_static", False)),
            )
        )

    return PAG(
        object_states=obj_states,
        body_part_nodes=body_nodes,
        object_part_nodes=obj_nodes,
        edges=edges,
    )


def _parse_node(node_str: str) -> tuple[str, str]:
    parts = node_str.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse PAG node: '{node_str}'")
    return parts[0].strip(), parts[1].strip()


def _is_human_node(node_str: str) -> bool:
    return node_str.lower().startswith("person")


def _load_smpl_body_seg(seg_path: Path) -> dict[str, np.ndarray]:
    with seg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    result: dict[str, np.ndarray] = {}
    for pag_name, seg_keys in BODY_PART_TO_SEG_KEYS.items():
        indices: list[int] = []
        for seg_key in seg_keys:
            indices.extend(raw.get(seg_key, []))
        if indices:
            result[pag_name] = np.unique(np.array(indices, dtype=np.int64))
    return result


def _load_object_part_segments(
    seg_obj_dir: Path,
    obj_slug: str,
    mesh_faces: np.ndarray,
) -> ObjectPartSegments:
    labels_path = (
        seg_obj_dir
        / obj_slug
        / "segmented_meshes"
        / f"{obj_slug}_triangle_labels.json"
    )
    if not labels_path.exists():
        raise FileNotFoundError(f"Triangle labels not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    label_map: dict[str, int] = data["label_map"]
    tri_labels = np.array(data["triangle_labels"], dtype=np.int32)

    part_vert_ids: dict[str, np.ndarray] = {}
    part_face_ids: dict[str, np.ndarray] = {}
    for part_name, label_id in label_map.items():
        tri_mask = tri_labels == label_id
        face_subset = mesh_faces[tri_mask]
        vert_ids = np.unique(face_subset.ravel())
        if vert_ids.size > 0:
            part_vert_ids[part_name] = vert_ids
            part_face_ids[part_name] = np.flatnonzero(
                tri_mask
            ).astype(np.int64)
    return ObjectPartSegments(vert_ids=part_vert_ids, face_ids=part_face_ids)


def _subsample_indices(n: int, max_pts: int) -> np.ndarray:
    if n <= max_pts:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_pts).astype(np.int64)


def _sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    if count <= 0 or faces.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    tri = vertices[faces]
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    areas = np.linalg.norm(np.cross(edge_1, edge_2), axis=1) * 0.5
    positive = areas > 1e-12
    if not np.any(positive):
        return vertices[_subsample_indices(vertices.shape[0], count)]

    valid_tri = tri[positive]
    weights = areas[positive]
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(
        valid_tri.shape[0],
        size=count,
        replace=True,
        p=weights,
    )
    r1 = rng.random(count, dtype=np.float32)
    r2 = rng.random(count, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack(
        [
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ],
        axis=1,
    ).astype(np.float32)
    sampled_tri = valid_tri[face_ids]
    return np.sum(sampled_tri * bary[:, :, None], axis=1).astype(np.float32)


def _sample_face_barycentrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0 or faces.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
        )

    tri = vertices[faces]
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    areas = np.linalg.norm(np.cross(edge_1, edge_2), axis=1) * 0.5
    positive = areas > 1e-12
    if not np.any(positive):
        face_ids = _subsample_indices(faces.shape[0], count)
        bary = np.tile(
            np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32),
            (len(face_ids), 1),
        )
        return face_ids, bary

    valid_face_ids = np.flatnonzero(positive)
    weights = areas[positive]
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_face_ids, size=count, replace=True, p=weights)
    r1 = rng.random(count, dtype=np.float32)
    r2 = rng.random(count, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack(
        [
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ],
        axis=1,
    ).astype(np.float32)
    return chosen.astype(np.int64), bary


def _sample_sequence_points_from_barycentrics(
    verts_seq: torch.Tensor,
    faces_torch: torch.Tensor,
    face_ids: torch.Tensor,
    bary: torch.Tensor,
) -> torch.Tensor:
    if face_ids.numel() == 0:
        return torch.zeros(
            (verts_seq.shape[0], 0, 3),
            dtype=verts_seq.dtype,
            device=verts_seq.device,
        )
    face_vids = faces_torch[face_ids]
    tri = verts_seq[:, face_vids, :]
    bary_exp = bary.view(1, -1, 3, 1)
    return (tri * bary_exp).sum(dim=2)


def _mask_to_point_cloud(
    mask: np.ndarray,
    width: int,
    height: int,
    max_points: int,
) -> np.ndarray:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.stack(
        [
            (xs.astype(np.float32) + 0.5) / float(width),
            (ys.astype(np.float32) + 0.5) / float(height),
        ],
        axis=1,
    )
    idx = _subsample_indices(len(pts), max_points)
    return pts[idx].astype(np.float32)


def _pack_2d_point_clouds(
    point_arrays: list[np.ndarray],
    device: torch.device,
) -> PackedPointCloud2D:
    max_len = max((arr.shape[0] for arr in point_arrays), default=0)
    max_len = max(max_len, 1)
    packed = np.zeros((len(point_arrays), max_len, 2), dtype=np.float32)
    lengths = np.zeros((len(point_arrays),), dtype=np.int64)
    for i, arr in enumerate(point_arrays):
        lengths[i] = arr.shape[0]
        if arr.shape[0] > 0:
            packed[i, :arr.shape[0]] = arr
    return PackedPointCloud2D(
        points=torch.from_numpy(packed).to(device),
        lengths=torch.from_numpy(lengths).to(device),
    )


def _load_mask_point_clouds(
    masks_dir: Path,
    num_frames: int,
    width: int,
    height: int,
    max_points: int,
    device: torch.device,
) -> PackedPointCloud2D | None:
    if not masks_dir.exists():
        return None
    arrays: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        mask_path = masks_dir / f"frame_{frame_idx:04d}.png"
        if not mask_path.exists():
            arrays.append(np.zeros((0, 2), dtype=np.float32))
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            arrays.append(np.zeros((0, 2), dtype=np.float32))
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        arrays.append(_mask_to_point_cloud(mask, width, height, max_points))
    return _pack_2d_point_clouds(arrays, device)


def _infer_image_size(dirs: dict[str, Path]) -> tuple[int, int]:
    frames_dir = resolve_frames_dir(dirs)
    if frames_dir is not None:
        frame_paths = list_images(frames_dir)
        if frame_paths:
            frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
            if frame is not None:
                height, width = frame.shape[:2]
                return width, height

    sample_candidates = [
        dirs["seg_vid"] / "humans" / "person_1" / "masks" / "frame_0000.png",
    ]
    sample_candidates.extend(
        sorted(
            dirs["seg_vid"].glob(
                "objects/*/object_segmentation/masks/frame_0000.png"
            )
        )
    )
    for sample_path in sample_candidates:
        if sample_path.exists():
            mask = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                height, width = mask.shape[:2]
                return width, height
    raise FileNotFoundError("Could not infer image size from frames or masks.")


def _resolve_human_mask_dir(seg_vid_dir: Path) -> Path | None:
    humans_dir = seg_vid_dir / "humans"
    if not humans_dir.exists():
        return None
    candidates = sorted(
        d for d in humans_dir.iterdir()
        if d.is_dir() and (d / "masks").exists()
    )
    if not candidates:
        return None
    return candidates[0] / "masks"


def _load_human_data(
    human_verts_np: np.ndarray,
    human_faces: np.ndarray,
    body_seg: dict[str, np.ndarray],
    dirs: dict[str, Path],
    width: int,
    height: int,
    device: torch.device,
    args: argparse.Namespace,
) -> HumanData:
    base_verts = torch.from_numpy(human_verts_np).float().to(device)
    faces_torch = torch.from_numpy(human_faces.astype(np.int64)).to(device)
    part_points_base: dict[str, torch.Tensor] = {}
    for part_name, vert_ids in body_seg.items():
        sub_ids = _subsample_indices(
            len(vert_ids),
            args.num_part_surface_points,
        )
        selected = vert_ids[sub_ids]
        part_points_base[part_name] = base_verts[:, selected, :]

    face_ids_np, bary_np = _sample_face_barycentrics(
        human_verts_np[0],
        human_faces,
        args.num_human_surface_points,
        SURFACE_SAMPLE_SEED,
    )
    sampled_points_base = _sample_sequence_points_from_barycentrics(
        base_verts,
        faces_torch,
        torch.from_numpy(face_ids_np).to(device),
        torch.from_numpy(bary_np).to(device),
    )
    if "hips" in body_seg and body_seg["hips"].size > 0:
        centers = base_verts[:, body_seg["hips"], :].mean(dim=1)
    else:
        centers = base_verts.mean(dim=1)

    human_mask_dir = _resolve_human_mask_dir(dirs["seg_vid"])
    human_mask_points = None
    if human_mask_dir is not None:
        human_mask_points = _load_mask_point_clouds(
            human_mask_dir,
            human_verts_np.shape[0],
            width,
            height,
            args.num_mask_points_2d,
            device,
        )
    return HumanData(
        base_verts=base_verts,
        faces=human_faces,
        faces_torch=faces_torch,
        part_points_base=part_points_base,
        sampled_points_base=sampled_points_base,
        centers=centers,
        mask_points_2d=human_mask_points,
    )


def _load_object_mask_targets(
    seg_vid_dir: Path,
    slug: str,
    part_names: list[str],
    num_frames: int,
    width: int,
    height: int,
    device: torch.device,
    max_points: int,
) -> tuple[PackedPointCloud2D | None, dict[str, PackedPointCloud2D]]:
    object_mask_dir = (
        seg_vid_dir / "objects" / slug / "object_segmentation" / "masks"
    )
    object_mask_points = _load_mask_point_clouds(
        object_mask_dir,
        num_frames,
        width,
        height,
        max_points,
        device,
    )
    part_mask_points: dict[str, PackedPointCloud2D] = {}
    for part_name in part_names:
        part_mask_dir = (
            seg_vid_dir
            / "objects"
            / slug
            / "parts_segmentation"
            / "masks"
            / part_name
        )
        packed = _load_mask_point_clouds(
            part_mask_dir,
            num_frames,
            width,
            height,
            max_points,
            device,
        )
        if packed is not None:
            part_mask_points[part_name] = packed
    return object_mask_points, part_mask_points


def _build_sdf_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    device: torch.device,
    padding: float = 0.05,
) -> SDFGrid:
    from pysdf import SDF as PySDF

    sdf_func = PySDF(vertices.astype(np.float32), faces.astype(np.uint32))
    vmin = vertices.min(axis=0) - padding
    vmax = vertices.max(axis=0) + padding
    lin = [np.linspace(vmin[i], vmax[i], resolution) for i in range(3)]
    gx, gy, gz = np.meshgrid(lin[0], lin[1], lin[2], indexing="ij")
    query_pts = np.stack(
        [gx.ravel(), gy.ravel(), gz.ravel()],
        axis=1,
    ).astype(np.float32)
    sdf_vals = -sdf_func(query_pts)
    sdf_vol = sdf_vals.reshape(1, 1, resolution, resolution, resolution)
    return SDFGrid(
        sdf_volume=torch.from_numpy(sdf_vol.astype(np.float32)).to(device),
        bbox_min=torch.tensor(
            vmin.reshape(1, 1, 3),
            dtype=torch.float32,
            device=device,
        ),
        bbox_max=torch.tensor(
            vmax.reshape(1, 1, 3),
            dtype=torch.float32,
            device=device,
        ),
    )


def _load_tracked_poses(poses_path: Path) -> np.ndarray:
    with poses_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = sorted(data, key=lambda x: x["frame"])
    return np.array([frame["T_4x4"] for frame in frames], dtype=np.float32)


def _mean_translation_step(tracked_poses: np.ndarray) -> float:
    if tracked_poses.shape[0] < 2:
        return 0.0
    diffs = np.diff(tracked_poses[:, :3, 3], axis=0)
    return float(np.linalg.norm(diffs, axis=1).mean())


def _mean_rotation_step(tracked_poses: np.ndarray) -> float:
    if tracked_poses.shape[0] < 2:
        return 0.0
    angles: list[float] = []
    for t in range(tracked_poses.shape[0] - 1):
        R1 = tracked_poses[t, :3, :3]
        R2 = tracked_poses[t + 1, :3, :3]
        R_rel = R1.T @ R2
        cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(float(np.arccos(cos_angle)))
    return float(np.mean(angles))


def _reference_priority(od: ObjectData) -> tuple[int, int, float, float]:
    return (
        int(od.state.is_translational),
        int(od.state.is_rotational),
        _mean_translation_step(od.tracked_poses),
        _mean_rotation_step(od.tracked_poses),
    )


def _select_canonical_reference_obj(
    a_is_human: bool,
    a_obj_idx: int,
    b_is_human: bool,
    b_obj_idx: int,
    objects: dict[str, ObjectData],
    obj_keys: list[str],
) -> int:
    if a_is_human and b_is_human:
        return -1
    if a_is_human:
        return b_obj_idx
    if b_is_human:
        return a_obj_idx

    od_a = objects[obj_keys[a_obj_idx]]
    od_b = objects[obj_keys[b_obj_idx]]
    if _reference_priority(od_a) <= _reference_priority(od_b):
        return a_obj_idx
    return b_obj_idx


def _uses_mean_contact_reduction(is_human: bool, part_name: str) -> bool:
    if not is_human:
        return False
    part = part_name.lower().strip()
    return part.endswith("hand") or part.endswith("foot")


def _select_contact_reduction(
    a_is_human: bool,
    a_part_name: str,
    b_is_human: bool,
    b_part_name: str,
) -> str:
    if _uses_mean_contact_reduction(a_is_human, a_part_name):
        return "mean"
    if _uses_mean_contact_reduction(b_is_human, b_part_name):
        return "mean"
    return "min"


def _select_contact_source_is_a(
    a_is_human: bool,
    a_vert_ids: np.ndarray,
    b_is_human: bool,
    b_vert_ids: np.ndarray,
) -> bool:
    if a_is_human != b_is_human:
        return a_is_human
    return len(a_vert_ids) <= len(b_vert_ids)


def load_problem_context(
    args: argparse.Namespace,
    script_dir: Path,
    device: torch.device,
) -> ProblemContext:
    dirs = resolve_dirs(args, script_dir)
    pag_path = resolve_pag_path(args, script_dir)
    smpl_seg_path = resolve_smpl_seg(args, script_dir)
    out_dir = dirs["output"]
    ensure_dir(out_dir)

    k, intr_path = _load_intrinsics_from_alignment_summary(dirs["aligned"])
    pag = _parse_pag(pag_path)
    body_seg = _load_smpl_body_seg(smpl_seg_path)
    width, height = _infer_image_size(dirs)
    k_torch = torch.from_numpy(k).float().to(device)

    human_aligned_dir = dirs["aligned"] / "human_motion_aligned"
    if not human_aligned_dir.exists():
        raise FileNotFoundError(
            f"Human motion aligned dir missing: {human_aligned_dir}"
        )

    human_ply_paths = sorted(
        human_aligned_dir.glob("frame_*.ply"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not human_ply_paths:
        raise FileNotFoundError(f"No frame_*.ply in {human_aligned_dir}")

    print(f"Loading {len(human_ply_paths)} human mesh frames...")
    human_meshes = [
        trimesh.load(str(path), process=False) for path in human_ply_paths
    ]
    human_verts_np = np.stack(
        [np.asarray(mesh.vertices, dtype=np.float32) for mesh in human_meshes]
    )
    human_faces = np.asarray(human_meshes[0].faces, dtype=np.int32)
    num_frames = human_verts_np.shape[0]
    human_data = _load_human_data(
        human_verts_np=human_verts_np,
        human_faces=human_faces,
        body_seg=body_seg,
        dirs=dirs,
        width=width,
        height=height,
        device=device,
        args=args,
    )
    print(
        f"  Human: {num_frames} frames, {human_verts_np.shape[1]} verts, "
        f"{human_faces.shape[0]} faces"
    )

    objects: dict[str, ObjectData] = {}
    obj_keys: list[str] = []
    for idx, state in enumerate(pag.object_states):
        slug = state.slug
        mesh_path = dirs["aligned"] / "meshes" / f"{slug}.ply"
        poses_path = dirs["tracked"] / slug / "poses.json"

        if not mesh_path.exists():
            print(f"  [SKIP] {slug}: aligned mesh not found at {mesh_path}")
            continue
        if not poses_path.exists():
            print(f"  [SKIP] {slug}: tracked poses not found at {poses_path}")
            continue

        mesh = trimesh.load(str(mesh_path), process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        tracked_poses = _load_tracked_poses(poses_path)

        if tracked_poses.shape[0] > num_frames:
            tracked_poses = tracked_poses[:num_frames]
        elif tracked_poses.shape[0] < num_frames:
            pad = np.tile(
                tracked_poses[-1:],
                (num_frames - tracked_poses.shape[0], 1, 1),
            )
            tracked_poses = np.concatenate([tracked_poses, pad], axis=0)

        rotvecs = []
        trans = []
        for t in range(num_frames):
            rv, tr = decompose_T(tracked_poses[t])
            rotvecs.append(rv)
            trans.append(tr)

        try:
            part_segments = _load_object_part_segments(
                dirs["seg_obj"],
                slug,
                faces,
            )
        except FileNotFoundError:
            print(
                f"  [WARN] {slug}: part segmentation not found, "
                "using whole mesh."
            )
            part_segments = ObjectPartSegments(vert_ids={}, face_ids={})

        sampled_points = torch.from_numpy(
            _sample_surface_points(
                verts,
                faces,
                args.num_object_surface_points,
                SURFACE_SAMPLE_SEED + idx,
            )
        ).float().to(device)
        part_sampled_points: dict[str, torch.Tensor] = {}
        for part_idx, (part_name, face_ids) in enumerate(
            sorted(part_segments.face_ids.items())
        ):
            part_faces = faces[face_ids]
            part_sampled = _sample_surface_points(
                verts,
                part_faces,
                args.num_part_surface_points,
                SURFACE_SAMPLE_SEED + 97 * idx + part_idx + 1,
            )
            part_sampled_points[part_name] = torch.from_numpy(
                part_sampled
            ).float().to(device)

        object_mask_points, part_mask_points = _load_object_mask_targets(
            dirs["seg_vid"],
            slug,
            list(part_segments.face_ids.keys()),
            num_frames,
            width,
            height,
            device,
            args.num_mask_points_2d,
        )

        print(
            f"  Building SDF for {slug} ({verts.shape[0]} verts, "
            f"res={args.sdf_resolution})..."
        )
        sdf_grid = _build_sdf_grid(verts, faces, args.sdf_resolution, device)
        print("    SDF done")

        objects[slug] = ObjectData(
            name=state.name,
            slug=slug,
            state=state,
            template_verts=torch.from_numpy(verts).float().to(device),
            faces=faces,
            faces_torch=torch.from_numpy(faces.astype(np.int64)).to(device),
            tracked_poses=tracked_poses,
            tracked_poses_torch=torch.from_numpy(tracked_poses)
            .float()
            .to(device),
            tracked_rotvecs=torch.from_numpy(
                np.stack(rotvecs)
            ).float().to(device),
            tracked_trans=torch.from_numpy(
                np.stack(trans)
            ).float().to(device),
            part_vert_ids=part_segments.vert_ids,
            part_face_ids=part_segments.face_ids,
            sampled_points=sampled_points,
            part_sampled_points=part_sampled_points,
            mask_points_2d=object_mask_points,
            part_mask_points_2d=part_mask_points,
            sdf_grid=sdf_grid,
            color_bgr=OBJECT_COLORS_BGR[idx % len(OBJECT_COLORS_BGR)],
        )
        obj_keys.append(slug)
        part_names = ", ".join(part_segments.vert_ids.keys())
        print(
            f"  Loaded {slug}: {verts.shape[0]} verts, "
            f"{faces.shape[0]} faces, {len(part_segments.vert_ids)} "
            f"parts ({part_names})"
        )

    if not obj_keys:
        raise RuntimeError("No objects loaded — nothing to optimise.")

    print("\nResolving PAG edges...")
    obj_slug_to_idx = {slug: idx for idx, slug in enumerate(obj_keys)}
    resolved_edges: list[ResolvedEdge] = []

    for edge in pag.edges:
        try:
            a_entity, a_part = _parse_node(edge.node_a)
            b_entity, b_part = _parse_node(edge.node_b)
        except ValueError as exc:
            print(f"  [WARN] Skipping edge: {exc}")
            continue

        a_is_human = _is_human_node(edge.node_a)
        b_is_human = _is_human_node(edge.node_b)

        if a_is_human:
            a_obj_idx = -1
            a_part_norm = a_part.lower().strip()
            if a_part_norm not in body_seg:
                print(
                    f"  [WARN] Body part '{a_part_norm}' "
                    "not in segmentation, skipping edge."
                )
                continue
            a_vids = body_seg[a_part_norm]
        else:
            a_slug = _sanitize(a_entity)
            if a_slug not in obj_slug_to_idx:
                print(f"  [WARN] Object '{a_slug}' not loaded, skipping edge.")
                continue
            a_obj_idx = obj_slug_to_idx[a_slug]
            a_part_norm = a_part.lower().strip()
            matched = None
            for part_name, part_vids in objects[a_slug].part_vert_ids.items():
                if part_name.lower().strip() == a_part_norm:
                    matched = part_vids
                    break
            if matched is None:
                print(
                    f"  [WARN] Part '{a_part}' not found in {a_slug}, "
                    "using whole mesh."
                )
                matched = np.arange(objects[a_slug].template_verts.shape[0])
            a_vids = matched

        if b_is_human:
            b_obj_idx = -1
            b_part_norm = b_part.lower().strip()
            if b_part_norm not in body_seg:
                print(
                    f"  [WARN] Body part '{b_part_norm}' "
                    "not in segmentation, skipping edge."
                )
                continue
            b_vids = body_seg[b_part_norm]
        else:
            b_slug = _sanitize(b_entity)
            if b_slug not in obj_slug_to_idx:
                print(f"  [WARN] Object '{b_slug}' not loaded, skipping edge.")
                continue
            b_obj_idx = obj_slug_to_idx[b_slug]
            b_part_norm = b_part.lower().strip()
            matched = None
            for part_name, part_vids in objects[b_slug].part_vert_ids.items():
                if part_name.lower().strip() == b_part_norm:
                    matched = part_vids
                    break
            if matched is None:
                print(
                    f"  [WARN] Part '{b_part}' not found in {b_slug}, "
                    "using whole mesh."
                )
                matched = np.arange(objects[b_slug].template_verts.shape[0])
            b_vids = matched

        contact_reduction = _select_contact_reduction(
            a_is_human,
            a_part_norm,
            b_is_human,
            b_part_norm,
        )
        contact_source_is_a = _select_contact_source_is_a(
            a_is_human,
            a_vids,
            b_is_human,
            b_vids,
        )
        canonical_obj_idx = _select_canonical_reference_obj(
            a_is_human,
            a_obj_idx,
            b_is_human,
            b_obj_idx,
            objects,
            obj_keys,
        )

        resolved_edges.append(
            ResolvedEdge(
                a_is_human=a_is_human,
                a_object_idx=a_obj_idx,
                a_vert_ids=a_vids,
                a_part_name=a_part_norm,
                b_is_human=b_is_human,
                b_object_idx=b_obj_idx,
                b_vert_ids=b_vids,
                b_part_name=b_part_norm,
                is_continuous=edge.is_continuous,
                is_rel_static=edge.is_rel_static,
                contact_reduction=contact_reduction,
                contact_source_is_a=contact_source_is_a,
                canonical_obj_idx=canonical_obj_idx,
            )
        )
        a_label = (
            f"human:{a_part}"
            if a_is_human
            else f"{obj_keys[a_obj_idx]}:{a_part}"
        )
        b_label = (
            f"human:{b_part}"
            if b_is_human
            else f"{obj_keys[b_obj_idx]}:{b_part}"
        )
        ref_label = (
            "none" if canonical_obj_idx < 0 else obj_keys[canonical_obj_idx]
        )
        print(
            f"  Edge: {a_label} ↔ {b_label}  "
            f"(continuous={edge.is_continuous}, static={edge.is_rel_static}, "
            f"contact={contact_reduction}, ref={ref_label})"
        )

    print(f"  → {len(resolved_edges)} edges resolved.\n")

    return ProblemContext(
        dirs=dirs,
        out_dir=out_dir,
        pag_path=pag_path,
        smpl_seg_path=smpl_seg_path,
        intr_path=intr_path,
        device=device,
        k=k,
        k_torch=k_torch,
        width=width,
        height=height,
        num_frames=num_frames,
        pag=pag,
        human_verts_np=human_verts_np,
        human_faces=human_faces,
        human_data=human_data,
        objects=objects,
        obj_keys=obj_keys,
        resolved_edges=resolved_edges,
    )
