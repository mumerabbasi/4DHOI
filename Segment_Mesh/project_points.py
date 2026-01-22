"""
Utility functions for projecting 3D point clouds to 2D image coordinates.

This module provides functions to:
1. Load camera matrices from cameras.json
2. Project 3D points to 2D pixel coordinates
3. Check point visibility (in front of camera and within image bounds)
4. Aggregate mask votes from multiple views

Usage:
    from project_points import PointCloudProjector

    projector = PointCloudProjector("renders/cameras.json")
    points_3d = np.array([[x1, y1, z1], [x2, y2, z2], ...])

    # Project to a specific view
    pixels, visible = projector.project_points(points_3d, view_index=0)

    # Get mask labels from all views
    labels = projector.get_labels_by_voting(points_3d, masks)
"""

import json
from pathlib import Path

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
    ) -> tuple[np.ndarray, np.ndarray]:
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

        # Project to 2D using intrinsic matrix
        # For points behind camera, we still compute but mark as not visible
        # Avoid division by zero
        # z_safe = np.where(z_cam != 0, z_cam, 1e-10)

        # In Blender convention, we need to negate Z for projection
        # p_2d = K @ [x, y, -z] / (-z)  for standard pinhole projection
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

        return pixels, visible

    def project_points_all_views(
        self,
        points_3d: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to all views.

        Args:
            points_3d: Nx3 array of 3D points in world coordinates.

        Returns:
            Tuple of:
            - all_pixels: VxNx2 array of pixel coordinates (V=num_views)
            - all_visible: VxN boolean array of visibility
        """
        N = points_3d.shape[0]
        all_pixels = np.zeros((self.num_views, N, 2))
        all_visible = np.zeros((self.num_views, N), dtype=bool)

        for v in range(self.num_views):
            pixels, visible = self.project_points(points_3d, v)
            all_pixels[v] = pixels
            all_visible[v] = visible

        return all_pixels, all_visible

    def get_pixel_labels(
        self,
        points_3d: np.ndarray,
        mask: np.ndarray,
        view_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get mask labels for 3D points from a single view's segmentation mask.

        Args:
            points_3d: Nx3 array of 3D points.
            mask: HxW array of integer labels (segmentation mask).
            view_index: Index of the view.

        Returns:
            Tuple of:
            - labels: N array of integer labels (-1 for non-visible points)
            - visible: N boolean array of visibility
        """
        pixels, visible = self.project_points(points_3d, view_index)

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

            labels[visible_idx] = mask[v, u]

        return labels, visible

    def get_labels_by_voting(
        self,
        points_3d: np.ndarray,
        masks: list[np.ndarray],
        ignore_label: int = -1,
    ) -> np.ndarray:
        """
        Get labels for 3D points by majority voting across all views.

        Args:
            points_3d: Nx3 array of 3D points.
            masks: List of V segmentation masks (one per view).
            ignore_label: Label value to ignore in voting (default: -1).

        Returns:
            N array of integer labels (most voted label for each point).
        """
        if len(masks) != self.num_views:
            raise ValueError(f"Expected {self.num_views} masks, got {len(masks)}")

        N = points_3d.shape[0]

        # Collect labels from all views
        all_labels = []
        for v in range(self.num_views):
            labels, _ = self.get_pixel_labels(points_3d, masks[v], v)
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


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test point cloud projection")
    parser.add_argument("--cameras", required=True, help="Path to cameras.json")
    parser.add_argument("--glb", help="Path to GLB file for point cloud")
    args = parser.parse_args()

    # Load projector
    projector = PointCloudProjector(args.cameras)
    print(f"Loaded {projector.num_views} views")
    print(f"Resolution: {projector.resolution}")
    print(f"Scene center: {projector.scene_center}")

    # Test with GLB if provided
    if args.glb:
        print(f"\nLoading point cloud from: {args.glb}")
        points = load_point_cloud_from_glb(args.glb)
        print(f"Loaded {len(points)} points")

        # Sample if too many
        if len(points) > 10000:
            points = sample_point_cloud(points, 10000)
            print(f"Sampled to {len(points)} points")

        # Project to each view
        for v in range(projector.num_views):
            pixels, visible = projector.project_points(points, v)
            view_info = projector.get_view_info(v)
            print(f"View {v} (az={view_info['azimuth_deg']}°, "
                  f"el={view_info['elevation_deg']}°): "
                  f"{visible.sum()}/{len(points)} points visible")
