# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 9 (DA2): Generate voxels from DA2 monocular depth maps (aligned with room normalization).

Converts DA2 (Depth Anything V2) ERP depth maps to 3D point clouds, then voxelizes
using the room's normalization parameters (from normalization_info.json) so that
depth voxels are spatially aligned with the GT room voxels from step3.

DA2 depth maps are monocular depth estimates (metric, in meters).

Coordinate pipeline:
1. Scale alignment: aligned_da2 = da2 * median(gt/da2) using co-valid pixels
2. Aligned DA2 depth → camera-centered point cloud (OpenGL: X=right, Y=up, -Z=forward)
3. Bbox filtering in camera space (room_info.json floor polygon XY)
4. Camera → World via R_x(pi/2): cam_X→world_X, cam_Y→world_Z, cam_-Z→world_Y
5. Ceiling removal in world Z axis
6. Room normalization: (world - center) * scale → [-0.5, 0.5]
7. Voxelize → PLY with position-based RGB colors

Input structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        erp/{view_idx}_depth_da2.npy
        erp/{view_idx}_depth.npy  (GT depth, for scale alignment)
        mesh_dumps/normalization_info.json
        camera_poses.json
        room_info.json (optional, for floor bbox filtering)

Output structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        depth_voxels_da2_{resolution}/{view_idx:04d}.ply

Usage:
    python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root datasets/ERP_3D_FRONT_test
    python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root datasets/ERP_3D_FRONT_test --resolution 64
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
import utils3d


class ProcessingLog:
    """Handles logging of processing progress to JSON file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            'step': 'step9_erp_depth_da2_to_voxels',
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_rooms': 0,
                'rooms_processed': 0,
                'rooms_failed': 0,
                'total_views': 0,
                'views_processed': 0,
                'views_failed': 0
            },
            'rooms': {}
        }

    def save(self):
        self.data['last_updated'] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_room_completed(self, room_key: str) -> bool:
        return room_key in self.data['rooms'] and self.data['rooms'][room_key].get('status') == 'completed'

    def log_room(self, room_key: str, result: dict):
        self.data['rooms'][room_key] = {
            'status': 'completed' if result['views_processed'] > 0 else 'failed',
            'views_processed': result['views_processed'],
            'views_failed': result['views_failed'],
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms, rooms_processed, rooms_failed,
                       total_views, views_processed, views_failed):
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed,
            'total_views': total_views,
            'views_processed': views_processed,
            'views_failed': views_failed
        }


def align_da2_to_gt_scale(da2_depth: np.ndarray, gt_depth: np.ndarray) -> tuple:
    """
    Compute scale factor to align DA2 monocular depth to GT depth scale.

    Uses median ratio on co-valid pixels (robust to outliers).
    aligned_da2 = da2 * scale_factor

    Args:
        da2_depth: (H, W) DA2 depth map
        gt_depth: (H, W) GT depth map

    Returns:
        (scale_factor, num_valid_pixels)
    """
    # Co-valid mask: both DA2 and GT have reasonable values
    valid = (
        (da2_depth > 0.01) & np.isfinite(da2_depth) &
        (gt_depth > 0.1) & (gt_depth < 20.0)
    )
    num_valid = valid.sum()

    if num_valid < 100:
        return 1.0, num_valid

    scale_factor = float(np.median(gt_depth[valid] / da2_depth[valid]))
    return scale_factor, num_valid


def erp_to_point_cloud(
    depth: np.ndarray,
    bbox_filter=None,
    remove_ceiling: bool = False,
    ceiling_threshold: float = 0.2,
) -> np.ndarray:
    """
    Convert DA2 ERP depth map to 3D point cloud in camera-centered coords.

    DA2 outputs metric depth in meters. Values are used directly.

    Camera space convention (OpenGL):
        X = right, Y = up, -Z = forward

    Args:
        depth: (H, W) DA2 depth map (metric, in meters)
        bbox_filter: (x0, x1, z0, z1) camera-space floor bbox to filter points
        remove_ceiling: remove ceiling points (in camera Y axis, before world transform)
        ceiling_threshold: distance from Y_max to remove (meters)

    Returns:
        points: (N, 3) valid 3D positions in camera-centered coordinates
    """
    H, W = depth.shape

    # Generate rays for equirectangular projection
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    theta = (v / H - 0.5) * np.pi
    phi = (u / W - 0.5) * 2 * np.pi

    # Convert to Cartesian (camera-centered, OpenGL convention)
    x = np.cos(theta) * np.sin(phi)
    y = -np.sin(theta)
    z = -np.cos(theta) * np.cos(phi)

    points = np.stack([x * depth, y * depth, z * depth], axis=-1)
    points = points.reshape(-1, 3)

    # Filter invalid depth (DA2 metric depth: positive and finite)
    depth_flat = depth.flatten()
    valid_mask = (depth_flat > 0.01) & (depth_flat < 50.0) & np.isfinite(depth_flat)

    points = points[valid_mask]

    # Filter by floor bbox in camera space (X and Z axes = floor plane)
    if bbox_filter is not None and len(points) > 0:
        x0, x1, z0, z1 = bbox_filter
        bbox_mask = (
            (points[:, 0] >= x0) & (points[:, 0] <= x1) &
            (points[:, 2] >= z0) & (points[:, 2] <= z1)
        )
        points = points[bbox_mask]

    # Remove ceiling in camera space (Y axis = up)
    if remove_ceiling and len(points) > 0:
        y_max = points[:, 1].max()
        ceiling_mask = points[:, 1] < (y_max - ceiling_threshold)
        points = points[ceiling_mask]

    return points


def compute_bbox_filter(bbox_min, bbox_max, camera_location):
    """
    Compute camera-space bbox filter from room_info.json floor corners.

    3D-FRONT world: X, Y = floor plane, Z = height
    Camera space: X = right (= world X), Z = backward (= -world Y)

    Args:
        bbox_min: (3,) room floor min corner in 3D-FRONT world coords
        bbox_max: (3,) room floor max corner in 3D-FRONT world coords
        camera_location: (3,) camera position in 3D-FRONT world coords

    Returns:
        (x0, x1, z0, z1) camera-space filter bounds
    """
    cam_x, cam_y, _cam_z = camera_location
    # Camera X corresponds to World X
    x0 = bbox_min[0] - cam_x
    x1 = bbox_max[0] - cam_x
    # Camera -Z corresponds to World Y, so Camera Z = -(World Y - cam_y)
    z0 = -(bbox_min[1] - cam_y)
    z1 = -(bbox_max[1] - cam_y)
    if z0 > z1:
        z0, z1 = z1, z0
    return (x0, x1, z0, z1)


def cam_to_world(points_cam, camera_location):
    """
    Convert camera-centered coordinates to 3D-FRONT world coordinates.

    The ERP camera has rotation [pi/2, 0, 0] (Euler XYZ), meaning:
    - Camera forward (-Z) → World +Y
    - Camera up (+Y) → World +Z (height)
    - Camera right (+X) → World +X

    Rotation matrix R_x(pi/2):
        [[1,  0,  0],
         [0,  0, -1],
         [0,  1,  0]]

    world = R @ cam + camera_location

    Args:
        points_cam: (N, 3) points in camera space [x_right, y_up, z_backward]
        camera_location: (3,) camera position in world coords [x, y, z_height]

    Returns:
        points_world: (N, 3) points in 3D-FRONT world coords [x, y, z_height]
    """
    points_world = np.column_stack([
        points_cam[:, 0] + camera_location[0],   # cam X → world X
        -points_cam[:, 2] + camera_location[1],   # cam -Z → world Y
        points_cam[:, 1] + camera_location[2],    # cam Y → world Z (height)
    ])
    return points_world


def voxelize_with_room_normalization(
    points_world: np.ndarray,
    center: np.ndarray,
    scale: float,
    grid_size: int = 64
) -> np.ndarray:
    """
    Voxelize point cloud using room normalization parameters.

    Args:
        points_world: (N, 3) points in world coordinates
        center: (3,) room center from normalization_info.json
        scale: float, room scale from normalization_info.json
        grid_size: voxel grid resolution

    Returns:
        voxel_centers: (M, 3) unique voxel center positions in [-0.5, 0.5] range
    """
    if len(points_world) == 0:
        return np.array([]).reshape(0, 3)

    # Apply room normalization: normalized = (world_point - center) * scale
    points_normalized = (points_world - center) * scale

    # Clip to valid range [-0.5, 0.5]
    points_normalized = np.clip(points_normalized, -0.5 + 1e-6, 0.5 - 1e-6)

    # Convert to voxel indices
    voxel_indices = ((points_normalized + 0.5) * grid_size).astype(np.int32)
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)

    # Remove duplicates
    unique_voxels = np.unique(voxel_indices, axis=0)

    # Convert back to normalized coordinates (voxel centers)
    voxel_centers = (unique_voxels + 0.5) / grid_size - 0.5

    return voxel_centers


def find_all_rooms(root: str) -> list:
    """Find all room directories that have ERP depth maps and normalization_info.json."""
    rooms = []
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path) or uuid_dir.startswith('.'):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            if not os.path.isdir(room_path) or room_name.startswith('.'):
                continue

            erp_dir = os.path.join(room_path, 'erp')
            norm_info_path = os.path.join(room_path, 'mesh_dumps', 'normalization_info.json')

            if os.path.isdir(erp_dir) and os.path.exists(norm_info_path):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path,
                })
    return rooms


def process_room(room_info: dict, resolution: int, remove_ceiling: bool = True,
                 ceiling_threshold: float = 0.2) -> dict:
    """
    Process all ERP depth views for a single room.

    Pipeline per view:
    1. ERP depth → camera-centered point cloud
    2. Bbox filter in camera space (room_info.json floor XY)
    3. Camera → World via R_x(pi/2) rotation
    4. Ceiling removal in world Z axis
    5. Room normalization → voxelize
    6. Save PLY with position-based RGB colors

    Returns:
        Dict with processing results
    """
    room_path = room_info['room_path']
    erp_dir = os.path.join(room_path, 'erp')
    output_dir = os.path.join(room_path, f'depth_voxels_da2_{resolution}')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'views_processed': 0,
        'views_failed': 0,
    }

    # Load normalization_info.json
    norm_info_path = os.path.join(room_path, 'mesh_dumps', 'normalization_info.json')
    with open(norm_info_path, 'r') as f:
        norm_info = json.load(f)
    center = np.array(norm_info['center'], dtype=np.float64)
    scale = float(norm_info['scale'])

    # Load camera_poses.json
    camera_poses_path = os.path.join(room_path, 'camera_poses.json')
    camera_views = {}
    if os.path.exists(camera_poses_path):
        with open(camera_poses_path, 'r') as f:
            camera_poses = json.load(f)
        for view in camera_poses.get('views', []):
            view_idx = view.get('view_idx')
            location = view.get('location')
            if view_idx is not None and location is not None:
                camera_views[view_idx] = np.array(location, dtype=np.float64)

    # Load room_info.json for floor bbox filtering
    room_info_path = os.path.join(room_path, 'room_info.json')
    room_bbox_min = room_bbox_max = None
    if os.path.exists(room_info_path):
        try:
            with open(room_info_path, 'r') as f:
                room_info_data = json.load(f)
            if 'min_corner' in room_info_data and 'max_corner' in room_info_data:
                room_bbox_min = np.asarray(room_info_data['min_corner'], dtype=np.float64)
                room_bbox_max = np.asarray(room_info_data['max_corner'], dtype=np.float64)
        except Exception as e:
            print(f"  [WARN] Failed to read room_info.json: {e}")

    # Find all DA2 depth files
    depth_files = sorted([f for f in os.listdir(erp_dir) if f.endswith('_depth_da2.npy')])
    depth_files.sort(key=lambda x: int(x.replace('_depth_da2.npy', '')))

    for depth_file in depth_files:
        view_idx_str = depth_file.replace('_depth_da2.npy', '')
        try:
            view_idx = int(view_idx_str)
        except ValueError:
            continue

        output_path = os.path.join(output_dir, f'{view_idx:04d}.ply')

        # Skip if already exists
        # if os.path.exists(output_path):
        #     results['views_processed'] += 1
        #     continue

        try:
            # Load DA2 depth
            depth = np.load(os.path.join(erp_dir, depth_file))

            # Load GT depth for scale alignment
            gt_depth_file = os.path.join(erp_dir, f'{view_idx_str}_depth.npy')
            if os.path.exists(gt_depth_file):
                gt_depth = np.load(gt_depth_file)
                scale_factor, n_valid = align_da2_to_gt_scale(depth, gt_depth)
                depth = depth * scale_factor
            else:
                print(f"  [WARN] No GT depth for scale alignment: {gt_depth_file}")

            # Get camera location
            camera_location = camera_views.get(view_idx)

            # Compute camera-space bbox filter from room floor polygon
            bbox_filter = None
            if room_bbox_min is not None and room_bbox_max is not None and camera_location is not None:
                bbox_filter = compute_bbox_filter(room_bbox_min, room_bbox_max, camera_location)

            # Convert scale-aligned DA2 depth to point cloud with bbox + ceiling filtering
            points_cam = erp_to_point_cloud(
                depth,
                bbox_filter=bbox_filter,
                remove_ceiling=remove_ceiling,
                ceiling_threshold=ceiling_threshold,
            )

            if len(points_cam) < 100:
                results['views_failed'] += 1
                continue

            # Convert camera → world coordinates via R_x(pi/2) rotation
            if camera_location is not None:
                points_world = cam_to_world(points_cam, camera_location)
            else:
                points_world = points_cam

            if len(points_world) < 100:
                results['views_failed'] += 1
                continue

            # Voxelize using room normalization
            voxel_centers = voxelize_with_room_normalization(
                points_world, center, scale, grid_size=resolution
            )

            if len(voxel_centers) < 10:
                results['views_failed'] += 1
                continue

            # Position-based RGB colors: map [-0.5, 0.5] → [0, 255]
            colors = ((voxel_centers + 0.5) * 255).astype(np.uint8)

            # Save as PLY with colors
            utils3d.io.write_ply(output_path, voxel_centers, vertex_colors=colors)
            results['views_processed'] += 1

        except Exception as e:
            print(f"  [ERROR] {room_info['uuid']}/{room_info['room_name']}/view{view_idx:04d}: {e}")
            results['views_failed'] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Step 9 (DA2): ERP DA2 depth to voxels (room-aligned)')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Voxel grid resolution (default: 64, matching SS encoder)')
    parser.add_argument('--remove_ceiling', action='store_true', default=True,
                        help='Remove ceiling points (default: True)')
    parser.add_argument('--ceiling_threshold', type=float, default=0.2,
                        help='Ceiling removal threshold in meters')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()

    args.remove_ceiling = True
    args.resolution = 64
    args.skip_completed = True

    # args.root = "datasets/ERP_3D_FRONT_test"

    # python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root datasets/ERP_3D_FRONT_test
    # python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root datasets/ERP_3D_FRONT_test --rank 0 --world_size 4
    # python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root datasets/ERP_3D_FRONT_test --rank 1 --world_size 4
    # python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root datasets/ERP_3D_FRONT_test --rank 2 --world_size 4
    # python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root datasets/ERP_3D_FRONT_test --rank 3 --world_size 4
    # python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root figure_sample_tmp --resolution 64
    # python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root figure_sample_please --resolution 64
    # python data_toolkit/erp/step9_erp_depth_da2_to_voxels.py --root figure_sample --resolution 64

    # Initialize logging
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step9_erp_depth_da2_to_voxels_{args.resolution}{log_suffix}.json')
    log = ProcessingLog(log_path)
    print(f"Logging to: {log_path}")

    # Find all rooms
    print("Finding rooms...")
    rooms = find_all_rooms(args.root)
    rooms.sort(key=lambda x: (x['uuid'], x['room_name']))
    total_rooms = len(rooms)
    print(f"Found {total_rooms} rooms")

    # Distribute across ranks
    start = len(rooms) * args.rank // args.world_size
    end = len(rooms) * (args.rank + 1) // args.world_size
    rooms = rooms[start:end]
    print(f"Processing {len(rooms)} rooms (rank {args.rank}/{args.world_size})")

    # Filter completed
    if args.skip_completed:
        original_count = len(rooms)
        rooms = [r for r in rooms if not log.is_room_completed(f"{r['uuid']}/{r['room_name']}")]
        skipped = original_count - len(rooms)
        if skipped > 0:
            print(f"Skipping {skipped} already completed rooms")

    # Process
    total_rooms_processed = 0
    total_rooms_failed = 0
    total_views_processed = 0
    total_views_failed = 0

    for i, room_info in enumerate(tqdm(rooms, desc="ERP DA2 depth → voxels")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"
        result = process_room(room_info, args.resolution,
                              args.remove_ceiling, args.ceiling_threshold)

        if result['views_processed'] > 0:
            total_rooms_processed += 1
        else:
            total_rooms_failed += 1
        total_views_processed += result['views_processed']
        total_views_failed += result['views_failed']

        log.log_room(room_key, result)

        if (i + 1) % args.log_interval == 0:
            log.update_summary(total_rooms, total_rooms_processed, total_rooms_failed,
                               total_views_processed + total_views_failed,
                               total_views_processed, total_views_failed)
            log.save()

    # Final log
    log.update_summary(total_rooms, total_rooms_processed, total_rooms_failed,
                       total_views_processed + total_views_failed,
                       total_views_processed, total_views_failed)
    log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {total_rooms_processed}")
    print(f"  Rooms failed: {total_rooms_failed}")
    print(f"  Views processed: {total_views_processed}")
    print(f"  Views failed: {total_views_failed}")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
