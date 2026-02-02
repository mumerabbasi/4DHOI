"""
Segment 3D point clouds using multi-view masks with depth-based occlusion handling.

This module provides functions to:
1. Load camera matrices from cameras.json
2. Project 3D points to 2D pixel coordinates
3. Check point visibility using depth maps (occlusion handling)
4. Aggregate mask votes from multiple views
5. Segment and save labeled point clouds

Usage:
    python segment_pointclouds.py --object_dir objects/iron

    # Or programmatically:
    from segment_pointclouds import PointCloudProjector, segment_point_cloud

    projector = PointCloudProjector("renders/cameras.json")
    points_3d = np.array([[x1, y1, z1], [x2, y2, z2], ...])

    # Get mask labels from all views with depth filtering
    labels = projector.get_labels_by_voting(points_3d, masks, depth_maps)
"""

import json
from pathlib import Path

import cv2
import numpy as np


class PointCloudProjector:
    """Project 3D points to 2D image coordinates using saved camera matrices."""

    def __init__(self, cameras_json_path: str):
        """
        Initialize projector with camera matrices.

        Args:
            cameras_json_path: Path to cameras.json file from rendering script.
        """
        self.cameras_json_path = Path(cameras_json_path)
        with open(cameras_json_path, "r") as f:
            self.camera_data = json.load(f)

        self.mesh_name = self.camera_data["mesh_name"]
        self.scene_center = np.array(self.camera_data["scene_center"])
        self.resolution = self.camera_data["resolution"]
        self.views = self.camera_data["views"]
        self.num_views = len(self.views)

        # Pre-compute numpy matrices for each view
        self._precompute_matrices()

    def _precompute_matrices(self) -> None:
        """Convert JSON matrices to numpy arrays for efficient computation."""
        self.intrinsics = []
        self.world_to_cameras = []
        self.camera_positions = []

        for view in self.views:
            K = np.array(view["intrinsic_matrix"])
            W2C = np.array(view["world_to_camera"])
            cam_pos = np.array(view["camera_position"])

            self.intrinsics.append(K)
            self.world_to_cameras.append(W2C)
            self.camera_positions.append(cam_pos)

    def project_points(
        self,
        points_3d: np.ndarray,
        view_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D pixel coordinates for a specific view.

        Args:
            points_3d: Nx3 array of 3D points in world coordinates.
            view_index: Index of the view (0-7).

        Returns:
            Tuple of:
            - pixels: Nx2 array of (u, v) pixel coordinates
            - visible: N boolean array indicating if point is visible
              (in front of camera and within image bounds)
            - depths: N array of point depths (distance from camera plane)
        """
        if view_index < 0 or view_index >= self.num_views:
            raise ValueError(f"view_index must be 0-{self.num_views-1}")

        points = np.asarray(points_3d)
        if points.ndim == 1:
            points = points.reshape(1, 3)

        N = points.shape[0]

        # Get matrices for this view
        K = self.intrinsics[view_index]
        W2C = self.world_to_cameras[view_index]

        # Transform to camera coordinates (homogeneous)
        # p_cam = W2C @ [p_world, 1]
        points_hom = np.hstack([points, np.ones((N, 1))])  # Nx4
        points_cam = (W2C @ points_hom.T).T  # Nx4

        # Extract x, y, z in camera space
        # Note: In Blender camera space, -Z is forward (looking direction)
        x_cam = points_cam[:, 0]
        y_cam = points_cam[:, 1]
        z_cam = points_cam[:, 2]

        # Points in front of camera have negative Z (camera looks down -Z)
        in_front = z_cam < 0

        # Depth is the distance along the camera's viewing direction
        # In Blender, -Z is forward, so depth = -z_cam
        depths = -z_cam

        # Project to 2D using intrinsic matrix
        # In Blender convention, we need to negate Z for projection
        points_cam_proj = np.stack([x_cam, y_cam, -z_cam], axis=1)  # Nx3
        points_2d_hom = (K @ points_cam_proj.T).T  # Nx3

        # Perspective division
        z_proj = points_2d_hom[:, 2]
        z_proj_safe = np.where(z_proj != 0, z_proj, 1e-10)

        u = points_2d_hom[:, 0] / z_proj_safe
        v = points_2d_hom[:, 1] / z_proj_safe

        pixels = np.stack([u, v], axis=1)

        # Check if within image bounds
        width, height = self.resolution
        in_bounds = (
            (u >= 0) & (u < width) &
            (v >= 0) & (v < height)
        )

        visible = in_front & in_bounds

        return pixels, visible, depths

    def project_points_all_views(
        self,
        points_3d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points to all views.

        Args:
            points_3d: Nx3 array of 3D points in world coordinates.

        Returns:
            Tuple of:
            - all_pixels: VxNx2 array of pixel coordinates (V=num_views)
            - all_visible: VxN boolean array of visibility
            - all_depths: VxN array of point depths
        """
        N = points_3d.shape[0]
        all_pixels = np.zeros((self.num_views, N, 2))
        all_visible = np.zeros((self.num_views, N), dtype=bool)
        all_depths = np.zeros((self.num_views, N))

        for v in range(self.num_views):
            pixels, visible, depths = self.project_points(points_3d, v)
            all_pixels[v] = pixels
            all_visible[v] = visible
            all_depths[v] = depths

        return all_pixels, all_visible, all_depths

    def get_pixel_labels(
        self,
        points_3d: np.ndarray,
        mask: np.ndarray,
        view_index: int,
        depth_map: np.ndarray | None = None,
        depth_threshold: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get mask labels for 3D points from a single view's segmentation mask.

        Args:
            points_3d: Nx3 array of 3D points.
            mask: HxW array of integer labels (segmentation mask).
            view_index: Index of the view.
            depth_map: HxW array of depth values (optional, for occlusion filtering).
            depth_threshold: Tolerance for depth comparison (default: 0.05).
                Points are considered visible if their depth is within this
                threshold of the rendered depth.

        Returns:
            Tuple of:
            - labels: N array of integer labels (-1 for non-visible points)
            - visible: N boolean array of visibility
        """
        pixels, visible, point_depths = self.project_points(points_3d, view_index)

        N = points_3d.shape[0]
        labels = np.full(N, -1, dtype=np.int32)

        # Get labels for visible points
        visible_idx = np.where(visible)[0]
        if len(visible_idx) > 0:
            u = pixels[visible_idx, 0].astype(np.int32)
            v = pixels[visible_idx, 1].astype(np.int32)

            # Clamp to valid range (should already be valid due to visibility check)
            height, width = mask.shape[:2]
            u = np.clip(u, 0, width - 1)
            v = np.clip(v, 0, height - 1)

            # Apply depth-based occlusion filtering if depth map provided
            if depth_map is not None:
                rendered_depths = depth_map[v, u]
                point_depths_visible = point_depths[visible_idx]

                # Point is visible only if its depth is close to rendered depth
                # (within threshold, meaning it's on the surface, not occluded)
                depth_valid = np.abs(point_depths_visible - rendered_depths) < depth_threshold

                # Update visibility
                visible_after_depth = visible_idx[depth_valid]
                u_valid = u[depth_valid]
                v_valid = v[depth_valid]

                labels[visible_after_depth] = mask[v_valid, u_valid]

                # Update visible array for return
                visible = np.zeros(N, dtype=bool)
                visible[visible_after_depth] = True
            else:
                labels[visible_idx] = mask[v, u]

        return labels, visible

    def get_labels_by_voting(
        self,
        points_3d: np.ndarray,
        masks: list[np.ndarray],
        depth_maps: list[np.ndarray] | None = None,
        depth_threshold: float = 0.05,
        ignore_label: int = -1,
    ) -> np.ndarray:
        """
        Get labels for 3D points by majority voting across all views.

        Args:
            points_3d: Nx3 array of 3D points.
            masks: List of V segmentation masks (one per view).
            depth_maps: List of V depth maps (optional, for occlusion filtering).
            depth_threshold: Tolerance for depth comparison.
            ignore_label: Label value to ignore in voting (default: -1).

        Returns:
            N array of integer labels (most voted label for each point).
        """
        if len(masks) != self.num_views:
            raise ValueError(f"Expected {self.num_views} masks, got {len(masks)}")
        if depth_maps is not None and len(depth_maps) != self.num_views:
            raise ValueError(f"Expected {self.num_views} depth maps, got {len(depth_maps)}")

        N = points_3d.shape[0]

        # Collect labels from all views
        all_labels = []
        for v in range(self.num_views):
            depth_map = depth_maps[v] if depth_maps is not None else None
            labels, _ = self.get_pixel_labels(
                points_3d, masks[v], v, depth_map, depth_threshold
            )
            all_labels.append(labels)

        all_labels = np.stack(all_labels, axis=0)  # VxN

        # Majority voting
        final_labels = np.full(N, ignore_label, dtype=np.int32)

        for i in range(N):
            point_labels = all_labels[:, i]
            valid_labels = point_labels[point_labels != ignore_label]

            if len(valid_labels) > 0:
                # Find most common label
                unique, counts = np.unique(valid_labels, return_counts=True)
                final_labels[i] = unique[np.argmax(counts)]

        return final_labels

    def get_view_info(self, view_index: int) -> dict:
        """Get information about a specific view."""
        return self.views[view_index]

    def get_image_path(self, view_index: int) -> Path:
        """Get the image path for a specific view."""
        image_name = self.views[view_index]["image_path"]
        return self.cameras_json_path.parent / image_name

    def get_depth_path(self, view_index: int) -> Path | None:
        """Get the depth map path for a specific view."""
        view = self.views[view_index]
        if "depth_path" in view:
            return self.cameras_json_path.parent / view["depth_path"]
        return None


def load_depth_map(depth_path: str) -> np.ndarray:
    """
    Load depth map from EXR file.

    Args:
        depth_path: Path to EXR depth file.

    Returns:
        HxW array of depth values.
    """
    # Try OpenEXR first (more reliable), fall back to cv2
    try:
        import OpenEXR
        import Imath

        exr_file = OpenEXR.InputFile(str(depth_path))
        header = exr_file.header()

        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Read depth channel (might be 'R', 'Y', or 'Z')
        channel_names = list(header["channels"].keys())
        depth_channel = "R" if "R" in channel_names else channel_names[0]

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        depth_str = exr_file.channel(depth_channel, pt)
        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = depth.reshape(height, width)

        return depth

    except ImportError:
        # Fall back to cv2 (requires opencv with EXR support)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Could not load depth map: {depth_path}")

        # If multi-channel, take first channel
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        return depth.astype(np.float32)


def load_point_cloud_from_ply(ply_path: str) -> np.ndarray:
    """
    Load point cloud from PLY file.

    Args:
        ply_path: Path to PLY file.

    Returns:
        Nx3 array of point positions.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required: pip install trimesh")

    cloud = trimesh.load(ply_path)

    if hasattr(cloud, "vertices"):
        return np.array(cloud.vertices)
    else:
        raise ValueError(f"Could not load point cloud from: {ply_path}")


def load_point_cloud_from_glb(glb_path: str) -> np.ndarray:
    """
    Load vertices from a GLB file as a point cloud.

    Note: This requires trimesh library.

    Args:
        glb_path: Path to GLB file.

    Returns:
        Nx3 array of vertex positions.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required: pip install trimesh")

    scene = trimesh.load(glb_path)

    if isinstance(scene, trimesh.Scene):
        # Combine all meshes
        vertices_list = []
        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                # Apply the transform from the scene graph
                transform = scene.graph.get(name)[0] if name in scene.graph else np.eye(4)
                verts = geometry.vertices.copy()
                # Apply transform
                verts_hom = np.hstack([verts, np.ones((len(verts), 1))])
                verts_transformed = (transform @ verts_hom.T).T[:, :3]
                vertices_list.append(verts_transformed)
        vertices = np.vstack(vertices_list) if vertices_list else np.array([])
    else:
        vertices = scene.vertices

    return vertices


def sample_point_cloud(vertices: np.ndarray, num_points: int) -> np.ndarray:
    """
    Randomly sample points from vertices.

    Args:
        vertices: Nx3 array of vertices.
        num_points: Number of points to sample.

    Returns:
        num_points x 3 array of sampled points.
    """
    if len(vertices) <= num_points:
        return vertices

    indices = np.random.choice(len(vertices), num_points, replace=False)
    return vertices[indices]


def save_labeled_point_cloud(
    points: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    label_names: dict[int, str] | None = None,
) -> None:
    """
    Save labeled point cloud to PLY file with colors.

    Args:
        points: Nx3 array of points.
        labels: N array of integer labels.
        output_path: Output PLY file path.
        label_names: Optional mapping of label IDs to names.
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required: pip install trimesh")

    # Color palette for labels
    colors_palette = [
        [0, 255, 0],      # Green
        [255, 0, 0],      # Red
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [128, 255, 0],    # Light green
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
        [0, 128, 255],    # Light blue
    ]
    unlabeled_color = [128, 128, 128]  # Gray for unlabeled

    # Assign colors based on labels
    colors = np.zeros((len(points), 4), dtype=np.uint8)
    colors[:, 3] = 255  # Alpha

    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            colors[mask, :3] = unlabeled_color
        else:
            color_idx = label % len(colors_palette)
            colors[mask, :3] = colors_palette[color_idx]

    # Create and save point cloud
    cloud = trimesh.PointCloud(vertices=points, colors=colors)
    cloud.export(output_path)
    print(f"Saved labeled point cloud: {output_path}")


def segment_point_cloud(
    object_dir: str,
    parts: list[str],
    depth_threshold: float = 0.15,
) -> dict[str, np.ndarray]:
    """
    Segment a point cloud using multi-view masks with depth-based occlusion.

    Args:
        object_dir: Path to object directory containing:
            - points.ply: Point cloud file
            - renders/: Rendered views with cameras.json and depth maps
            - masks_sam3/: Part segmentation masks
        parts: List of part names to segment.
        depth_threshold: Depth tolerance for occlusion filtering.

    Returns:
        Dictionary mapping part names to point arrays.
    """
    object_dir = Path(object_dir)

    # Load point cloud
    ply_path = object_dir / "points.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")

    print(f"Loading point cloud from: {ply_path}")
    points = load_point_cloud_from_ply(str(ply_path))
    print(f"Loaded {len(points)} points")

    # Load camera matrices
    cameras_path = object_dir / "renders" / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"cameras.json not found: {cameras_path}")

    projector = PointCloudProjector(str(cameras_path))
    print(f"Loaded {projector.num_views} camera views")

    # Load masks and depth maps for each view
    masks_dir = object_dir / "masks_sam3"
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    # Build combined masks for each view (one mask per view with all parts)
    # and create label mapping
    label_map = {name: i + 1 for i, name in enumerate(parts)}  # 0 = background
    label_map_inv = {v: k for k, v in label_map.items()}

    masks = []
    depth_maps = []

    for v in range(projector.num_views):
        view_info = projector.get_view_info(v)
        image_path = view_info["image_path"]  # e.g., "rgb/az020_el-45.png"
        # Extract view name from path (handles both old and new formats)
        if "/" in image_path:
            # New format: "rgb/az020_el+025.png"
            view_name = Path(image_path).stem  # "az020_el+025"
        else:
            # Old format: "rgb_az020_el-45.png"
            view_name = image_path.replace("rgb_", "").replace(".png", "")

        # Create combined mask for this view
        height, width = projector.resolution
        combined_mask = np.zeros((height, width), dtype=np.int32)

        for part_name in parts:
            # Find mask file for this part and view
            # Handle various naming conventions:
            # - New: "az020_el+25_surface.png" (with '+' sign)
            # - Old: "rgb_az020_el25_surface.png" (with 'rgb_' prefix, no '+' sign)
            part_clean = part_name.replace(' ', '_')
            view_name_no_plus = view_name.replace('+', '')  # Remove '+' for compatibility
            
            # Try different filename formats in order of preference
            candidates = [
                f"{view_name}_{part_clean}.png",           # New format with '+'
                f"{view_name_no_plus}_{part_clean}.png",   # New format without '+'
                f"rgb_{view_name}_{part_clean}.png",       # Old format with '+'
                f"rgb_{view_name_no_plus}_{part_clean}.png",  # Old format without '+'
            ]
            
            mask_path = None
            for candidate in candidates:
                test_path = masks_dir / candidate
                if test_path.exists():
                    mask_path = test_path
                    break

            if mask_path is not None:
                part_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if part_mask is not None:
                    # Set label where mask is positive
                    combined_mask[part_mask > 127] = label_map[part_name]

        masks.append(combined_mask)

        # Load depth map
        depth_path = projector.get_depth_path(v)
        if depth_path is not None and depth_path.exists():
            depth = load_depth_map(str(depth_path))
            depth_maps.append(depth)
        else:
            depth_maps.append(None)
            print(f"  Warning: No depth map for view {v}")

    # Check if we have any depth maps
    has_depth = any(d is not None for d in depth_maps)
    if not has_depth:
        print("Warning: No depth maps found, occlusion filtering disabled")
        depth_maps = None

    # Get labels by voting
    print("Segmenting point cloud...")
    labels = projector.get_labels_by_voting(
        points, masks, depth_maps, depth_threshold
    )

    # Print statistics
    print("\nSegmentation results:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        if label == -1:
            name = "unlabeled"
        elif label == 0:
            name = "background"
        else:
            name = label_map_inv.get(label, f"label_{label}")
        pct = count / len(labels) * 100
        print(f"  {name}: {count} points ({pct:.1f}%)")

    # Save full labeled point cloud
    output_full = object_dir / "points_labeled.ply"
    save_labeled_point_cloud(points, labels, str(output_full), label_map_inv)

    # Save individual part point clouds
    part_points = {}
    segmented_dir = object_dir / "segmented"
    segmented_dir.mkdir(exist_ok=True)

    for part_name, label_id in label_map.items():
        mask = labels == label_id
        if mask.sum() > 0:
            part_pts = points[mask]
            part_points[part_name] = part_pts

            # Save part point cloud
            output_part = segmented_dir / f"{part_name.replace(' ', '_')}.ply"
            part_labels = np.full(len(part_pts), label_id, dtype=np.int32)
            save_labeled_point_cloud(part_pts, part_labels, str(output_part))

    return part_points


def parse_pag_file(pag_path: str) -> dict[str, list[str]]:
    """
    Parse PAG file to extract objects and their parts.

    Args:
        pag_path: Path to PAG JSON file.

    Returns:
        Dictionary mapping object names to list of part names.
    """
    with open(pag_path) as f:
        pag = json.load(f)

    objects_parts = {}
    for node in pag.get("object part nodes", []):
        parts = node.split(", ", 1)
        if len(parts) == 2:
            obj_name, part_name = parts
            if obj_name not in objects_parts:
                objects_parts[obj_name] = []
            objects_parts[obj_name].append(part_name)

    return objects_parts


# Main entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Segment point clouds using multi-view masks with depth filtering."
    )
    parser.add_argument(
        "--object_dir",
        type=str,
        help="Path to object directory (e.g., objects/iron).",
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default="../Generate_PAG/output_pag_deepseek_r1_32b.json",
        help="PAG JSON file to extract parts automatically.",
    )
    parser.add_argument(
        "--objects_root",
        type=str,
        default="objects",
        help="Root directory containing object folders.",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default=None,
        help="Comma-separated list of part names (overrides PAG file).",
    )
    parser.add_argument(
        "--depth_threshold",
        type=float,
        default=0.15,
        help="Depth tolerance for occlusion filtering (default: 0.15).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Path to cameras.json (for testing projection only).",
    )
    args = parser.parse_args()

    # Test mode: just project points
    if args.cameras:
        projector = PointCloudProjector(args.cameras)
        print(f"Loaded {projector.num_views} views")
        print(f"Resolution: {projector.resolution}")
        print(f"Scene center: {projector.scene_center}")

    # Segmentation mode
    elif args.object_dir:
        object_dir = Path(args.object_dir).resolve()
        if not object_dir.exists():
            print(f"Error: Object directory not found: {object_dir}")
            exit(1)

        # Get parts
        if args.parts:
            parts = [p.strip() for p in args.parts.split(",")]
        else:
            # Try to get parts from PAG file
            pag_path = Path(args.pag_file).resolve()
            if pag_path.exists():
                objects_parts = parse_pag_file(str(pag_path))
                obj_name = object_dir.name.replace("_", " ")
                parts = objects_parts.get(obj_name, [])
                if not parts:
                    print(f"Error: No parts found for '{obj_name}' in PAG file")
                    exit(1)
            else:
                print("Error: Provide --parts or valid --pag_file")
                exit(1)

        print(f"\n{'='*60}")
        print(f"Segmenting: {object_dir.name}")
        print(f"Parts: {parts}")
        print(f"{'='*60}\n")

        segment_point_cloud(str(object_dir), parts, args.depth_threshold)

    # Process all objects from PAG file
    else:
        pag_path = Path(args.pag_file).resolve()
        if not pag_path.exists():
            print(f"Error: PAG file not found: {pag_path}")
            exit(1)

        objects_parts = parse_pag_file(str(pag_path))
        if not objects_parts:
            print("No object parts found in PAG file")
            exit(1)

        objects_root = Path(args.objects_root).resolve()
        if not objects_root.exists():
            print(f"Error: Objects root not found: {objects_root}")
            exit(1)

        print(f"Found {len(objects_parts)} objects in PAG file:")
        for obj, parts in objects_parts.items():
            print(f"  {obj}: {parts}")

        for obj_name, parts in objects_parts.items():
            dir_name = obj_name.replace(" ", "_")
            object_dir = objects_root / dir_name

            if not object_dir.exists():
                print(f"\nWarning: Object directory not found: {object_dir}, skipping...")
                continue

            if not (object_dir / "points.ply").exists():
                print(f"\nWarning: No point cloud for {obj_name}, skipping...")
                continue

            if not (object_dir / "renders" / "cameras.json").exists():
                print(f"\nWarning: No renders for {obj_name}, skipping...")
                continue

            print(f"\n{'='*60}")
            print(f"Segmenting: {obj_name}")
            print(f"{'='*60}\n")

            try:
                segment_point_cloud(str(object_dir), parts, args.depth_threshold)
            except Exception as e:
                print(f"Error processing {obj_name}: {e}")

    print("\nDone!")
