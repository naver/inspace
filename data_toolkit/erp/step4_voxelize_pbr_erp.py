# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 4: Voxelize PBR attributes for ERP_3D_FRONT dataset.

Since 3D-FRONT dataset doesn't have PBR textures, this script:
- Uses default PBR values: metallic=0, roughness=0.6
- Extracts base color from mesh textures/vertex colors if available
- Falls back to gray (0.8, 0.8, 0.8)

IMPORTANT: step2_dump_pbr_erp.py already normalizes all vertices using room's bbox.
So pbr_dumps contain pre-normalized vertices - DO NOT re-normalize for room_coord mode!

For individual assets, supports two modes:
- room_coord (default): vertices already in room coordinate, use directly
- normalized: re-normalize each asset to its own bbox (maximize voxel resolution)

Input structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        pbr_dumps/full_room_wo_ceiling.pickle    # Already normalized
        pbr_dumps/individual_assets/{asset_name}.pickle  # Already in room coords

Output structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        pbr_voxels_{resolution}/full_room_wo_ceiling.vxz
        pbr_voxels_{resolution}/individual_assets_room_coord/{asset_name}.vxz
        pbr_voxels_{resolution}/individual_assets_normalized/{asset_name}.vxz

Logging:
    datasets/ERP_3D_FRONT_test_logs/step4_voxelize_pbr_{resolution}.json

Usage:
    python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512
    python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --asset_mode both
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
            'step': 'step4_voxelize_pbr',
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
            if os.path.isdir(room_path) and os.path.exists(os.path.join(room_path, 'pbr_dumps')):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path
                })
    return rooms


def load_pbr_dump(pickle_path: str) -> dict:
    """Load PBR dump."""
    with open(pickle_path, 'rb') as f:
        dump = pickle.load(f)

    # Fix dump alpha map
    for mat in dump['materials']:
        if mat['alphaTexture'] is not None and mat['alphaMode'] == 'OPAQUE':
            mat['alphaMode'] = 'BLEND'

    # Append default material for faces without material
    dump['materials'].append({
        "baseColorFactor": [0.8, 0.8, 0.8],
        "alphaFactor": 1.0,
        "metallicFactor": 0.0,
        "roughnessFactor": 0.6,
        "alphaMode": "OPAQUE",
        "alphaCutoff": 0.5,
        "baseColorTexture": None,
        "alphaTexture": None,
        "metallicTexture": None,
        "roughnessTexture": None,
    })

    # Filter empty objects
    dump['objects'] = [
        obj for obj in dump['objects']
        if obj['vertices'].size != 0 and obj['faces'].size != 0
    ]

    if len(dump['objects']) == 0:
        return None

    return dump


def fix_mat_ids(dump: dict):
    """Fix mat_ids to use default material for -1 indices."""
    for obj in dump['objects']:
        obj['mat_ids'][obj['mat_ids'] == -1] = len(dump['materials']) - 1
        assert np.all(obj['mat_ids'] >= 0), 'invalid mat_ids'
    return dump


def normalize_pbr_dump(dump: dict):
    """
    Normalize PBR dump vertices to [-0.5, 0.5] bbox.
    Used for 'normalized' mode to maximize voxel resolution.

    Returns: normalized dump, center, scale
    """
    # Get all vertices
    vertices = torch.from_numpy(
        np.concatenate([obj['vertices'] for obj in dump['objects']], axis=0)
    ).float()

    vertices_min = vertices.min(dim=0)[0]
    vertices_max = vertices.max(dim=0)[0]
    center = (vertices_min + vertices_max) / 2
    scale = 0.99999 / (vertices_max - vertices_min).max()
    center = center.numpy().tolist()
    scale = float(scale)

    center_tensor = torch.tensor(center)
    for obj in dump['objects']:
        obj['vertices'] = (torch.from_numpy(obj['vertices']).float() - center_tensor) * scale
        obj['vertices'] = obj['vertices'].numpy()

    # Fix mat_ids
    fix_mat_ids(dump)

    return dump, center, scale


def voxelize_pbr(dump: dict, resolution: int) -> tuple:
    """
    Voxelize PBR attributes from mesh dump.

    Returns:
        coords, attributes dict
    """
    coords, attr = o_voxel.convert.blender_dump_to_volumetric_attr(
        dump,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        mip_level_offset=0,
        verbose=False,
        timing=False
    )

    # Remove unnecessary attributes
    if 'normal' in attr:
        del attr['normal']
    if 'emissive' in attr:
        del attr['emissive']

    return coords, attr


def process_room(room_info: dict, resolution: int, mode: str = 'all', asset_mode: str = 'room_coord') -> dict:
    """
    Process a single room.

    Args:
        room_info: Dict with uuid, room_name, room_path
        resolution: O-Voxel resolution
        mode: 'all', 'room_only', 'assets_only'
        asset_mode: 'both', 'room_coord', 'normalized'

    Returns:
        Dict with processing results
    """
    room_path = room_info['room_path']
    pbr_dumps_dir = os.path.join(room_path, 'pbr_dumps')
    output_dir = os.path.join(room_path, f'pbr_voxels_{resolution}')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'room_processed': False,
        'room_voxels': 0,
        'assets_processed': 0,
        'assets_failed': 0
    }

    # Process full room
    # NOTE: pbr_dumps already have normalized vertices from step2
    if mode in ['all', 'room_only']:
        room_pickle_path = os.path.join(pbr_dumps_dir, 'full_room_wo_ceiling.pickle')
        room_output_path = os.path.join(output_dir, 'full_room_wo_ceiling.vxz')

        if os.path.exists(room_pickle_path) and not os.path.exists(room_output_path):
            try:
                dump = load_pbr_dump(room_pickle_path)
                if dump is not None:
                    # Just fix mat_ids, vertices are already normalized
                    fix_mat_ids(dump)
                    coords, attr = voxelize_pbr(dump, resolution)
                    o_voxel.io.write_vxz(room_output_path, coords, attr)
                    results['room_processed'] = True
                    results['room_voxels'] = len(coords)
            except Exception as e:
                print(f"Error processing room {room_info['uuid']}/{room_info['room_name']}: {e}")
        elif os.path.exists(room_output_path):
            results['room_processed'] = True
            try:
                info = o_voxel.io.read_vxz_info(room_output_path)
                results['room_voxels'] = info['num_voxel']
            except:
                pass

    # Process individual assets
    if mode in ['all', 'assets_only']:
        assets_pickle_dir = os.path.join(pbr_dumps_dir, 'individual_assets')
        if os.path.exists(assets_pickle_dir):
            asset_files = glob.glob(os.path.join(assets_pickle_dir, '*.pickle'))

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

                room_coord_output = os.path.join(assets_room_coord_dir, f'{asset_name}.vxz') if do_room_coord else None
                normalized_output = os.path.join(assets_normalized_dir, f'{asset_name}.vxz') if do_normalized else None

                room_coord_exists = room_coord_output and os.path.exists(room_coord_output)
                normalized_exists = normalized_output and os.path.exists(normalized_output)

                need_room_coord = do_room_coord and not room_coord_exists
                need_normalized = do_normalized and not normalized_exists

                if not need_room_coord and not need_normalized:
                    results['assets_processed'] += 1
                    continue

                try:
                    # Room coordinate version
                    # NOTE: pbr_dumps already have normalized vertices from step2
                    # Just fix mat_ids and voxelize directly
                    if need_room_coord:
                        dump = load_pbr_dump(asset_path)
                        if dump is not None:
                            fix_mat_ids(dump)
                            # Check if within bounds
                            all_verts = np.concatenate([obj['vertices'] for obj in dump['objects']])
                            if np.all(all_verts >= -0.5) and np.all(all_verts <= 0.5):
                                coords, attr = voxelize_pbr(dump, resolution)
                                o_voxel.io.write_vxz(room_coord_output, coords, attr)
                            else:
                                print(f"  [Warning] {asset_name}: outside room bounds, skipping room_coord")

                    # Normalized version - re-normalize to asset's own bbox
                    if need_normalized:
                        dump = load_pbr_dump(asset_path)
                        if dump is not None:
                            dump, _, _ = normalize_pbr_dump(dump)
                            coords, attr = voxelize_pbr(dump, resolution)
                            o_voxel.io.write_vxz(normalized_output, coords, attr)

                    results['assets_processed'] += 1

                except Exception as e:
                    print(f"Error processing asset {asset_name}: {e}")
                    results['assets_failed'] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Voxelize PBR for ERP_3D_FRONT dataset')
    parser.add_argument('--root', type=str, default='datasets/ERP_3D_FRONT',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--resolution', type=int, default=512,
                        help='O-Voxel resolution')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'room_only', 'assets_only'],
                        help='Processing mode')
    parser.add_argument('--asset_mode', type=str, default='room_coord',
                        choices=['both', 'room_coord', 'normalized'],
                        help='Asset voxelization mode: room_coord (default), normalized, both')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()

    # args.root = 'datasets/ERP_3D_FRONT'
    args.asset_mode = 'room_coord'
    args.mode = 'all'
    args.resolution = 512 # 256, 512, 1024

    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT_test

    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 0 --world_size 10
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 1 --world_size 10
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 2 --world_size 10
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 3 --world_size 10
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 4 --world_size 10
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 5 --world_size 10
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 6 --world_size 10
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 7 --world_size 10
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 8 --world_size 10
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 9 --world_size 10

    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT_test --skip_completed --resolution 512 --rank 0 --world_size 4
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT_test --skip_completed --resolution 512 --rank 1 --world_size 4
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT_test --skip_completed --resolution 512 --rank 2 --world_size 4
    # python data_toolkit/erp/step4_voxelize_pbr_erp.py --root datasets/ERP_3D_FRONT_test --skip_completed --resolution 512 --rank 3 --world_size 4

    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step4_voxelize_pbr_{args.resolution}{log_suffix}.json')
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
