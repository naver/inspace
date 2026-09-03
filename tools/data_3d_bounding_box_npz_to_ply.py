# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

# Read 3D bounding box NPZ and convert to PLY
# OBB format: [x, y, z, sx, sy, sz, rotation]
# - x, y, z: center position (normalized to [-0.5, 0.5])
# - sx, sy, sz: FULL extents (full dimensions, NOT half-extents)
# - rotation: yaw angle in radians

import numpy as np
import argparse
import os
import glob
import utils3d

'''
Info:
read 3d bounding box NPZ and convert to PLY
- 3d bounding box NPZ file is in the directory: datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/3d_bounding_box/{room_name}_scene_data.npz
- PLY file is saved in the directory: datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/3d_bounding_box_ply

Usage:
python read_3d_bbox_npz.py --root_dir datasets/ERP_3D_FRONT_test
Output:
- PLY files are saved in the directory: datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/3d_bounding_box_ply
- Each PLY file contains the 3d bounding box of the asset
- Each PLY file is named as {asset_name}_bbox.ply
- Each PLY file contains the 3d bounding box of the asset
'''

parser = argparse.ArgumentParser()
args = parser.parse_args()
# args.npz_path = "datasets/ERP_3D_FRONT_test/00ad8345-45e0-45b3-867d-4a3c88c2517a/MasterBedroom-46277/3d_bounding_box/MasterBedroom-46277_scene_data.npz"
# args.npz_path = "datasets/ERP_3D_FRONT_test/00110bde-f580-40be-b8bb-88715b338a2a/LivingDiningRoom-44785/3d_bounding_box/LivingDiningRoom-44785_scene_data.npz"
# args.root_dir = "datasets/ERP_3D_FRONT_test"
args.root_dir = "datasets/ERP_3D_FRONT_test/00110bde-f580-40be-b8bb-88715b338a2a/LivingDiningRoom-44785/3d_bounding_box"

def obb_to_corners(obb):
    """
    Convert OBB (oriented bounding box) to 8 corner vertices.

    Args:
        obb: [x, y, z, sx, sy, sz, rotation] - center, FULL extents, yaw

    Returns:
        corners: [8, 3] array of corner positions
    """
    cx, cy, cz, sx, sy, sz, rot = obb

    # Create 8 corners of axis-aligned box centered at origin
    # sx, sy, sz are FULL extents, so divide by 2 to get half-extents
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    corners = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [ hx,  hy, -hz],
        [-hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [ hx,  hy,  hz],
        [-hx,  hy,  hz],
    ])

    # Apply rotation around Z axis (yaw)
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    rot_matrix = np.array([
        [cos_r, -sin_r, 0],
        [sin_r,  cos_r, 0],
        [0,      0,     1]
    ])
    corners = corners @ rot_matrix.T

    # Translate to center
    corners = corners + np.array([cx, cy, cz])

    return corners


def obb_to_wireframe_mesh(obb):
    """
    Convert OBB to wireframe mesh (12 edges as thin triangles).

    Returns:
        vertices: [8, 3]
        faces: [24, 3] - 12 edges * 2 triangles each
    """
    corners = obb_to_corners(obb)

    # Define 12 edges of a box
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # For wireframe visualization, we can just return corners
    # PLY with only vertices will show as points
    return corners, edges


def sample_edge_points(p1, p2, num_points=20):
    """Sample points along an edge."""
    t = np.linspace(0, 1, num_points)
    return p1 + np.outer(t, p2 - p1)


def sample_face_points(corners, face_indices, num_points_per_edge=10):
    """Sample points on a face (quad)."""
    # Get face corners
    c0, c1, c2, c3 = [corners[i] for i in face_indices]

    points = []
    for u in np.linspace(0, 1, num_points_per_edge):
        for v in np.linspace(0, 1, num_points_per_edge):
            # Bilinear interpolation
            p = (1-u)*(1-v)*c0 + u*(1-v)*c1 + u*v*c2 + (1-u)*v*c3
            points.append(p)
    return np.array(points)


def obb_to_dense_points(obb, edge_samples=30, fill_faces=True, face_samples=15):
    """
    Convert OBB to dense point cloud for visualization.

    Args:
        obb: [x, y, z, sx, sy, sz, rotation]
        edge_samples: number of points per edge
        fill_faces: whether to fill faces with points
        face_samples: points per edge when filling faces

    Returns:
        points: [N, 3] dense point cloud
    """
    corners = obb_to_corners(obb)

    # 12 edges of a box
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    all_points = []

    # Sample points along edges
    for i, j in edges:
        edge_points = sample_edge_points(corners[i], corners[j], edge_samples)
        all_points.append(edge_points)

    # Optionally fill faces
    if fill_faces:
        # 6 faces of a box (quad indices)
        faces = [
            (0, 1, 2, 3),  # bottom
            (4, 5, 6, 7),  # top
            (0, 1, 5, 4),  # front
            (2, 3, 7, 6),  # back
            (0, 3, 7, 4),  # left
            (1, 2, 6, 5),  # right
        ]
        for face_idx in faces:
            face_points = sample_face_points(corners, face_idx, face_samples)
            all_points.append(face_points)

    return np.vstack(all_points)


def save_bbox_as_ply(corners, output_path, color=(255, 0, 0)):
    """Save bounding box corners as PLY point cloud."""
    colors = np.tile(np.array(color, dtype=np.uint8), (len(corners), 1))
    utils3d.io.write_ply(output_path, corners.astype(np.float32), vertex_colors=colors)


def save_dense_bbox_as_ply(obb, output_path, color=(255, 0, 0), edge_samples=30, fill_faces=True):
    """Save bounding box as dense point cloud PLY."""
    points = obb_to_dense_points(obb, edge_samples=edge_samples, fill_faces=fill_faces)
    colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
    utils3d.io.write_ply(output_path, points.astype(np.float32), vertex_colors=colors)
    return len(points)


def polygon_to_dense_points(polygon_2d, z_min, z_max, edge_samples=30, face_samples=15):
    """
    Convert 2D floor/ceiling polygon to dense 3D point cloud.

    Args:
        polygon_2d: (N, 2) array of 2D vertices
        z_min: bottom Z coordinate
        z_max: top Z coordinate
        edge_samples: points per edge
        face_samples: points per edge for face grid

    Returns:
        points: (M, 3) dense point cloud
    """
    if polygon_2d is None or len(polygon_2d) < 3:
        return None

    all_points = []
    n_verts = len(polygon_2d)

    # Sample points along polygon edges (top and bottom)
    for z in [z_min, z_max]:
        for i in range(n_verts):
            p1 = np.array([polygon_2d[i, 0], polygon_2d[i, 1], z])
            p2 = np.array([polygon_2d[(i+1) % n_verts, 0], polygon_2d[(i+1) % n_verts, 1], z])
            edge_pts = sample_edge_points(p1, p2, edge_samples)
            all_points.append(edge_pts)

    # Sample vertical edges at each polygon vertex
    for i in range(n_verts):
        p1 = np.array([polygon_2d[i, 0], polygon_2d[i, 1], z_min])
        p2 = np.array([polygon_2d[i, 0], polygon_2d[i, 1], z_max])
        edge_pts = sample_edge_points(p1, p2, edge_samples)
        all_points.append(edge_pts)

    # Fill top and bottom faces with grid points
    # Use bounding box of polygon for simplicity
    min_xy = polygon_2d.min(axis=0)
    max_xy = polygon_2d.max(axis=0)

    for z in [z_min, z_max]:
        for u in np.linspace(0, 1, face_samples):
            for v in np.linspace(0, 1, face_samples):
                x = min_xy[0] + u * (max_xy[0] - min_xy[0])
                y = min_xy[1] + v * (max_xy[1] - min_xy[1])
                # Simple point-in-polygon check (for convex polygons)
                all_points.append(np.array([[x, y, z]]))

    # Fill side walls
    for i in range(n_verts):
        p0 = polygon_2d[i]
        p1 = polygon_2d[(i+1) % n_verts]

        for u in np.linspace(0, 1, face_samples):
            for v in np.linspace(0, 1, face_samples):
                x = p0[0] + u * (p1[0] - p0[0])
                y = p0[1] + u * (p1[1] - p0[1])
                z = z_min + v * (z_max - z_min)
                all_points.append(np.array([[x, y, z]]))

    return np.vstack(all_points)


def save_all_bboxes_as_ply(obbs, filenames, output_dir, edge_samples=30, fill_faces=True, face_samples=15,
                           floor_polygon=None, floor_z=None, floor_height=None):
    """Save all bounding boxes as separate PLY files with dense points."""
    os.makedirs(output_dir, exist_ok=True)

    # Color palette for different assets
    colors = [
        (255, 0, 0),    # red
        (0, 255, 0),    # green
        (0, 0, 255),    # blue
        (255, 255, 0),  # yellow
        (255, 0, 255),  # magenta
        (0, 255, 255),  # cyan
        (255, 128, 0),  # orange
        (128, 0, 255),  # purple
        (0, 128, 255),  # light blue
        (255, 0, 128),  # pink
    ]

    all_points = []
    all_colors = []

    for i, (obb, filename) in enumerate(zip(obbs, filenames)):
        color = colors[i % len(colors)]

        # Generate dense points for this bbox
        points = obb_to_dense_points(obb, edge_samples=edge_samples,
                                      fill_faces=fill_faces, face_samples=face_samples)

        # Save individual bbox
        asset_name = os.path.splitext(filename)[0]
        ply_path = os.path.join(output_dir, f'{asset_name}_bbox.ply')
        point_colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
        utils3d.io.write_ply(ply_path, points.astype(np.float32), vertex_colors=point_colors)
        print(f"Saved: {ply_path} ({asset_name}, {len(points)} points)")

        # Collect for combined visualization
        all_points.append(points)
        all_colors.extend([color] * len(points))

    # Add floor if provided
    if floor_polygon is not None and floor_z is not None and floor_height is not None:
        floor_color = (100, 150, 255)  # Blue for floor
        z_min = floor_z
        z_max = floor_z + max(floor_height, 0.02)  # Minimum thickness for visibility
        floor_points = polygon_to_dense_points(floor_polygon, z_min, z_max, edge_samples, face_samples)

        if floor_points is not None:
            # Save floor separately
            floor_ply_path = os.path.join(output_dir, 'floor_bbox.ply')
            floor_colors = np.tile(np.array(floor_color, dtype=np.uint8), (len(floor_points), 1))
            utils3d.io.write_ply(floor_ply_path, floor_points.astype(np.float32), vertex_colors=floor_colors)
            print(f"Saved: {floor_ply_path} (floor, {len(floor_points)} points)")

            # Add to combined
            all_points.append(floor_points)
            all_colors.extend([floor_color] * len(floor_points))

    # Save combined bboxes (assets + floor)
    if len(all_points) > 0:
        combined_points = np.vstack(all_points)
        combined_colors = np.array(all_colors, dtype=np.uint8)
        combined_path = os.path.join(output_dir, 'all_bboxes_combined.ply')
        utils3d.io.write_ply(combined_path, combined_points.astype(np.float32), vertex_colors=combined_colors)
        print(f"Saved combined: {combined_path} ({len(obbs)} assets + floor, {len(combined_points)} points)")


# Find all 3d_bounding_box directories
bbox_dirs = []
for root, dirs, files in os.walk(args.root_dir):
    if '3d_bounding_box' in root:
        # Look for scene_data.npz files
        npz_files = glob.glob(os.path.join(root, '*_scene_data.npz'))
        if npz_files:
            bbox_dirs.append((root, npz_files[0]))

print(f"Found {len(bbox_dirs)} 3d_bounding_box directories to process\n")

# Process each 3d_bounding_box directory
for bbox_dir, npz_path in bbox_dirs:
    print(f"Processing: {npz_path}")
    
    try:
        # Load NPZ
        data = np.load(npz_path, allow_pickle=True)
        obbs = data['obbs']
        asset_filenames = data['asset_filenames']

        # Load floor data
        floor_polygon = data['floor_polygon'] if 'floor_polygon' in data else None
        floor_z = float(data['floor_z']) if 'floor_z' in data else None
        floor_height = float(data['floor_height']) if 'floor_height' in data else None

        print(f"  Loaded {len(obbs)} asset bounding boxes")
        print(f"  Asset OBB range: x=[{obbs[:, 0].min():.3f}, {obbs[:, 0].max():.3f}], "
              f"y=[{obbs[:, 1].min():.3f}, {obbs[:, 1].max():.3f}], "
              f"z=[{obbs[:, 2].min():.3f}, {obbs[:, 2].max():.3f}]")

        if floor_polygon is not None:
            print(f"  Floor: {len(floor_polygon)} vertices, z={floor_z:.3f}, height={floor_height:.3f}")

        # Output directory (same level as 3d_bounding_box)
        output_dir = os.path.join(bbox_dir, '..', '3d_bounding_box_ply')
        output_dir = os.path.normpath(output_dir)

        # Save as PLY (assets + floor)
        save_all_bboxes_as_ply(
            obbs, asset_filenames, output_dir,
            floor_polygon=floor_polygon,
            floor_z=floor_z,
            floor_height=floor_height
        )

        print(f"  ✓ Saved PLY files to: {output_dir}\n")
        
    except Exception as e:
        print(f"  ✗ Error processing {npz_path}: {e}\n")
        continue

print(f"\nCompleted processing {len(bbox_dirs)} directories")
