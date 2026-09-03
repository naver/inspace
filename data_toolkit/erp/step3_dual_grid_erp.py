# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 3: Convert mesh to O-Voxel (Flexible Dual Grid) for ERP_3D_FRONT dataset.

This script converts mesh dumps to O-Voxel geometry representation using
the Flexible Dual Grid algorithm.

IMPORTANT: step1_dump_mesh_erp.py already normalizes all meshes using the room's
bbox. Both room and assets are in [-0.5, 0.5] room coordinate system.

For individual assets, two modes are available:
- room_coord (default): directly convert pickle (already in room coordinate)
- normalized: re-normalize each asset individually (for maximum voxel resolution)

Input structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        mesh_dumps/full_room_wo_ceiling.pickle    # Already normalized
        mesh_dumps/individual_assets/{asset_name}.pickle  # Already in room coords
        mesh_dumps/normalization_info.json        # Room's center and scale

Output structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        dual_grid_{resolution}/full_room_wo_ceiling.vxz
        dual_grid_{resolution}/individual_assets_room_coord/{asset_name}.vxz  # room coordinate
        dual_grid_{resolution}/individual_assets_normalized/{asset_name}.vxz  # individually normalized
        dual_grid_{resolution}/individual_assets_normalized/{asset_name}_norm.json  # asset's normalization

Logging:
    datasets/ERP_3D_FRONT_test_logs/step3_dual_grid_{resolution}.json
    - Tracks processed rooms, success/failure status, timestamps
    - Enables resumable processing

Usage:
    python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512
    python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --asset_mode both
    python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --asset_mode room_coord
    python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --asset_mode normalized
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
import o_voxel
from tqdm import tqdm
from datetime import datetime
import glob


class ProcessingLog:
    """Handles logging of processing progress to JSON file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.data = self._load()

    def _load(self) -> dict:
        """Load existing log or create new one."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            'step': 'step3_dual_grid',
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_rooms': 0,
                'rooms_processed': 0,
                'rooms_failed': 0,
                'total_voxels': 0,
                'assets_processed': 0,
                'assets_failed': 0
            },
            'rooms': {}
        }

    def save(self):
        """Save log to file."""
        self.data['last_updated'] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_room_completed(self, room_key: str) -> bool:
        """Check if room has been successfully processed."""
        return room_key in self.data['rooms'] and self.data['rooms'][room_key].get('status') == 'completed'

    def log_room(self, room_key: str, result: dict):
        """Log processing result for a room."""
        self.data['rooms'][room_key] = {
            'status': 'completed' if result['room_processed'] else 'failed',
            'room_processed': result['room_processed'],
            'room_voxels': result['room_voxels'],
            'assets_processed': result['assets_processed'],
            'assets_failed': result['assets_failed'],
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms: int, rooms_processed: int, rooms_failed: int,
                       total_voxels: int, assets_processed: int, assets_failed: int):
        """Update summary statistics."""
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed,
            'total_voxels': total_voxels,
            'assets_processed': assets_processed,
            'assets_failed': assets_failed
        }


def find_all_rooms(root: str) -> list:
    """Find all room directories in the dataset."""
    rooms = []
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            if os.path.isdir(room_path) and os.path.exists(os.path.join(room_path, 'mesh_dumps')):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path
                })
    return rooms


def load_mesh_dump(pickle_path: str) -> dict:
    """Load mesh from pickle dump."""
    with open(pickle_path, 'rb') as f:
        dump = pickle.load(f)

    # Combine all objects
    start = 0
    vertices_list = []
    faces_list = []

    for obj in dump['objects']:
        if obj['vertices'].size == 0 or obj['faces'].size == 0:
            continue
        vertices_list.append(obj['vertices'])
        faces_list.append(obj['faces'] + start)
        start += len(obj['vertices'])

    if len(vertices_list) == 0:
        return None

    vertices = torch.from_numpy(np.concatenate(vertices_list, axis=0)).float()
    faces = torch.from_numpy(np.concatenate(faces_list, axis=0)).long()

    return {'vertices': vertices, 'faces': faces}


def normalize_mesh(vertices: torch.Tensor, normalize: bool = True) -> tuple:
    """
    Normalize mesh to [-0.5, 0.5] unit cube.

    Returns:
        normalized_vertices, center, scale
    """
    if not normalize:
        return vertices, torch.zeros(3), 1.0

    vertices_min = vertices.min(dim=0)[0]
    vertices_max = vertices.max(dim=0)[0]
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()

    normalized = (vertices - center) * scale
    assert torch.all(normalized >= -0.5) and torch.all(normalized <= 0.5), 'vertices out of range'

    return normalized, center.numpy().tolist(), float(scale)


def mesh_to_dual_grid(vertices: torch.Tensor, faces: torch.Tensor, resolution: int) -> tuple:
    """
    Convert mesh to O-Voxel using Flexible Dual Grid.

    Returns:
        voxel_indices, dual_vertices, intersected
    """
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False
    )

    # Pack for storage
    dual_vertices = dual_vertices * resolution - voxel_indices
    assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1 + 1e-3), 'dual_vertices out of range'
    dual_vertices = torch.clamp(dual_vertices, 0, 1)
    dual_vertices = (dual_vertices * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

    return voxel_indices, dual_vertices, intersected


def process_room(room_info: dict, resolution: int, mode: str = 'all', asset_mode: str = 'both') -> dict:
    """
    Process a single room.

    Args:
        room_info: Dict with uuid, room_name, room_path
        resolution: O-Voxel resolution (e.g., 512)
        mode: 'all', 'room_only', 'assets_only'
        asset_mode: 'both', 'room_coord', 'normalized'
            - room_coord: position relative to room (OmniPart-style)
            - normalized: individually normalized (max resolution)
            - both: save both versions

    Returns:
        Dict with processing results
    """
    room_path = room_info['room_path']
    mesh_dumps_dir = os.path.join(room_path, 'mesh_dumps')
    output_dir = os.path.join(room_path, f'dual_grid_{resolution}')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'room_processed': False,
        'room_voxels': 0,
        'assets_processed': 0,
        'assets_failed': 0,
        'normalization': None
    }

    # Process full room mesh
    # NOTE: step1 already normalized the room mesh to [-0.5, 0.5]
    if mode in ['all', 'room_only']:
        room_pickle_path = os.path.join(mesh_dumps_dir, 'full_room_wo_ceiling.pickle')
        room_output_path = os.path.join(output_dir, 'full_room_wo_ceiling.vxz')

        if os.path.exists(room_pickle_path) and not os.path.exists(room_output_path):
            try:
                mesh_data = load_mesh_dump(room_pickle_path)
                if mesh_data is not None:
                    # Vertices are already normalized in step1, use directly
                    vertices = mesh_data['vertices']

                    # Copy normalization info from mesh_dumps to output dir
                    src_norm_info_path = os.path.join(mesh_dumps_dir, 'normalization_info.json')
                    dst_norm_info_path = os.path.join(output_dir, 'normalization_info.json')
                    if os.path.exists(src_norm_info_path) and not os.path.exists(dst_norm_info_path):
                        with open(src_norm_info_path, 'r') as f:
                            norm_info = json.load(f)
                        with open(dst_norm_info_path, 'w') as f:
                            json.dump(norm_info, f, indent=2)
                        results['normalization'] = norm_info

                    # Convert to dual grid
                    voxel_indices, dual_vertices, intersected = mesh_to_dual_grid(
                        vertices, mesh_data['faces'], resolution
                    )

                    # Save
                    o_voxel.io.write_vxz(
                        room_output_path,
                        voxel_indices,
                        {'vertices': dual_vertices, 'intersected': intersected}
                    )

                    results['room_processed'] = True
                    results['room_voxels'] = len(voxel_indices)
            except Exception as e:
                print(f"Error processing room {room_info['uuid']}/{room_info['room_name']}: {e}")
        elif os.path.exists(room_output_path):
            results['room_processed'] = True
            try:
                info = o_voxel.io.read_vxz_info(room_output_path)
                results['room_voxels'] = info['num_voxel']
            except:
                pass

    # Process individual assets based on asset_mode
    # - room_coord: position relative to room (OmniPart-style part-aware generation)
    # - normalized: individually normalized (maximum voxel resolution)
    # - both: save both versions
    if mode in ['all', 'assets_only']:
        assets_pickle_dir = os.path.join(mesh_dumps_dir, 'individual_assets')
        if os.path.exists(assets_pickle_dir):
            asset_files = glob.glob(os.path.join(assets_pickle_dir, '*.pickle'))

            # Output directories based on asset_mode
            do_room_coord = asset_mode in ['both', 'room_coord']
            do_normalized = asset_mode in ['both', 'normalized']

            if do_room_coord:
                assets_room_coord_dir = os.path.join(output_dir, 'individual_assets_room_coord')
                os.makedirs(assets_room_coord_dir, exist_ok=True)
            if do_normalized:
                assets_normalized_dir = os.path.join(output_dir, 'individual_assets_normalized')
                os.makedirs(assets_normalized_dir, exist_ok=True)

            for asset_path in asset_files:
                asset_name = os.path.splitext(os.path.basename(asset_path))[0]

                # Check existing files
                room_coord_output_path = os.path.join(assets_room_coord_dir, f'{asset_name}.vxz') if do_room_coord else None
                normalized_output_path = os.path.join(assets_normalized_dir, f'{asset_name}.vxz') if do_normalized else None
                normalized_norm_path = os.path.join(assets_normalized_dir, f'{asset_name}_norm.json') if do_normalized else None

                room_coord_exists = room_coord_output_path and os.path.exists(room_coord_output_path)
                normalized_exists = normalized_output_path and os.path.exists(normalized_output_path)

                # Skip if all requested versions exist
                need_room_coord = do_room_coord and not room_coord_exists
                need_normalized = do_normalized and not normalized_exists
                if not need_room_coord and not need_normalized:
                    results['assets_processed'] += 1
                    continue

                try:
                    mesh_data = load_mesh_dump(asset_path)
                    if mesh_data is None:
                        continue

                    # NOTE: In step1, assets are already normalized using room's center/scale
                    # So vertices are already in room coordinate system [-0.5, 0.5]
                    vertices = mesh_data['vertices']
                    faces = mesh_data['faces']

                    # Version 1: Room coordinate (already in room coord from step1)
                    if need_room_coord:
                        # Check if asset is within room bounds
                        if torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5):
                            voxel_indices, dual_vertices, intersected = mesh_to_dual_grid(
                                vertices, faces, resolution
                            )
                            o_voxel.io.write_vxz(
                                room_coord_output_path,
                                voxel_indices,
                                {'vertices': dual_vertices, 'intersected': intersected}
                            )
                        else:
                            print(f"  [Warning] {asset_name}: outside room bounds, skipping room_coord")

                    # Version 2: Individually normalized (maximize voxel resolution)
                    if need_normalized:
                        # Re-normalize to asset's own [-0.5, 0.5] bbox
                        vertices_norm, asset_center, asset_scale = normalize_mesh(vertices, normalize=True)

                        voxel_indices, dual_vertices, intersected = mesh_to_dual_grid(
                            vertices_norm, faces, resolution
                        )
                        o_voxel.io.write_vxz(
                            normalized_output_path,
                            voxel_indices,
                            {'vertices': dual_vertices, 'intersected': intersected}
                        )

                        # Save asset's normalization info (relative to room coords)
                        with open(normalized_norm_path, 'w') as f:
                            json.dump({'center': asset_center, 'scale': asset_scale}, f, indent=2)

                    results['assets_processed'] += 1

                except Exception as e:
                    print(f"Error processing asset {asset_name}: {e}")
                    results['assets_failed'] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Convert mesh to O-Voxel for ERP_3D_FRONT dataset')
    parser.add_argument('--root', type=str, default='datasets/ERP_3D_FRONT',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--resolution', type=int, default=512,
                        help='O-Voxel resolution')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'room_only', 'assets_only'],
                        help='Processing mode')
    parser.add_argument('--asset_mode', type=str, default='room_coord',
                        choices=['both', 'room_coord', 'normalized'],
                        help='Asset voxelization mode: room_coord (relative position, default), normalized (max resolution), both')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()
    
    # args.root = 'datasets/ERP_3D_FRONT'
    # args.root = 'datasets/ERP_3D_FRONT_test'
    # args.root = 'datasets/_ERP_3D_FRONT_before/ERP_3D_FRONT_test'
    args.asset_mode = 'room_coord'
    args.mode = 'all'
    # args.resolution = 256 # 256, 512, 1024
    # args.resolution = 1024
    args.skip_completed = True

    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 256
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT_test --resolution 256

    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 0 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 1 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 2 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 3 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 4 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 5 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 6 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 7 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 8 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 9 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 10 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 11 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 12 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 13 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 14 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 15 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 16 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 17 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 18 --world_size 20
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 19 --world_size 20

    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 0 --world_size 4
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 1 --world_size 4
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 2 --world_size 4
    # python data_toolkit/erp/step3_dual_grid_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 3 --world_size 4

    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step3_dual_grid_{args.resolution}{log_suffix}.json')
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

    # Filter already completed rooms if requested
    if args.skip_completed:
        original_count = len(rooms)
        rooms = [r for r in rooms if not log.is_room_completed(f"{r['uuid']}/{r['room_name']}")]
        skipped = original_count - len(rooms)
        if skipped > 0:
            print(f"Skipping {skipped} already completed rooms")

    # Process rooms
    total_room_processed = 0
    total_room_failed = 0
    total_room_voxels = 0
    total_assets_processed = 0
    total_assets_failed = 0

    for i, room_info in enumerate(tqdm(rooms, desc="Processing rooms")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"
        result = process_room(room_info, args.resolution, args.mode, args.asset_mode)

        # Update counters
        if result['room_processed']:
            total_room_processed += 1
            total_room_voxels += result['room_voxels']
        else:
            total_room_failed += 1
        total_assets_processed += result['assets_processed']
        total_assets_failed += result['assets_failed']

        # Log result
        log.log_room(room_key, result)

        # Save log periodically
        if (i + 1) % args.log_interval == 0:
            log.update_summary(total_rooms, total_room_processed, total_room_failed,
                             total_room_voxels, total_assets_processed, total_assets_failed)
            log.save()

    # Final log save
    log.update_summary(total_rooms, total_room_processed, total_room_failed,
                      total_room_voxels, total_assets_processed, total_assets_failed)
    log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {total_room_processed}")
    print(f"  Rooms failed: {total_room_failed}")
    print(f"  Total room voxels: {total_room_voxels}")
    print(f"  Avg voxels per room: {total_room_voxels / max(1, total_room_processed):.0f}")
    print(f"  Assets processed: {total_assets_processed}")
    print(f"  Assets failed: {total_assets_failed}")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
