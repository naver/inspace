# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 4 (layout_wo_ceiling): Voxelize PBR attributes for layout_wo_ceiling only.

Simplified version of step4_voxelize_pbr_erp.py that processes ONLY layout_wo_ceiling
(no individual assets, no full_room_wo_ceiling).

Since 3D-FRONT dataset doesn't have PBR textures, this script:
- Uses default PBR values: metallic=0, roughness=0.6
- Extracts base color from mesh textures/vertex colors if available
- Falls back to gray (0.8, 0.8, 0.8)

IMPORTANT: step2_dump_pbr_erp.py already normalizes all vertices using room's bbox.
So pbr_dumps contain pre-normalized vertices - DO NOT re-normalize!

Input structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        pbr_dumps/layout_wo_ceiling.pickle    # Already normalized

Output structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        pbr_voxels_{resolution}/layout_wo_ceiling.vxz

Logging:
    datasets/ERP_3D_FRONT_logs/step4_voxelize_pbr_layout_wo_ceiling_{resolution}.json

Usage:
    python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512
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
            'step': 'step4_voxelize_pbr_layout_wo_ceiling',
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_rooms': 0,
                'rooms_processed': 0,
                'rooms_failed': 0,
                'total_voxels': 0,
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
            'status': 'completed' if result['processed'] else 'failed',
            'processed': result['processed'],
            'num_voxels': result['num_voxels'],
            'error': result.get('error', None),
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms: int, rooms_processed: int, rooms_failed: int,
                       total_voxels: int):
        """Update summary statistics."""
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed,
            'total_voxels': total_voxels,
        }


def find_all_rooms(root: str) -> list:
    """Find all room directories that have pbr_dumps directory."""
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
    """Load PBR dump from pickle file."""
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


def process_room(room_info: dict, resolution: int) -> dict:
    """
    Process a single room: voxelize layout_wo_ceiling PBR only.

    Args:
        room_info: Dict with uuid, room_name, room_path
        resolution: O-Voxel resolution

    Returns:
        Dict with processing results
    """
    room_path = room_info['room_path']
    pbr_dumps_dir = os.path.join(room_path, 'pbr_dumps')
    output_dir = os.path.join(room_path, f'pbr_voxels_{resolution}')

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'processed': False,
        'num_voxels': 0,
        'error': None
    }

    pickle_path = os.path.join(pbr_dumps_dir, 'layout_wo_ceiling.pickle')
    output_path = os.path.join(output_dir, 'layout_wo_ceiling.vxz')

    # Check if input exists
    if not os.path.exists(pickle_path):
        results['error'] = 'layout_wo_ceiling.pickle not found'
        return results

    # Skip if output already exists
    if os.path.exists(output_path):
        results['processed'] = True
        try:
            info = o_voxel.io.read_vxz_info(output_path)
            results['num_voxels'] = info['num_voxel']
        except:
            pass
        return results

    try:
        os.makedirs(output_dir, exist_ok=True)

        dump = load_pbr_dump(pickle_path)
        if dump is None:
            results['error'] = 'Empty dump (no objects with vertices/faces)'
            return results

        # Just fix mat_ids, vertices are already normalized from step2
        fix_mat_ids(dump)

        coords, attr = voxelize_pbr(dump, resolution)
        o_voxel.io.write_vxz(output_path, coords, attr)

        results['processed'] = True
        results['num_voxels'] = len(coords)

    except Exception as e:
        results['error'] = str(e)
        print(f"Error processing {room_info['uuid']}/{room_info['room_name']}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Voxelize PBR for layout_wo_ceiling only (ERP_3D_FRONT dataset)')
    parser.add_argument('--root', type=str,
                        default='datasets/ERP_3D_FRONT',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--resolution', type=int, default=512,
                        help='O-Voxel resolution')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()

    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512
    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512

    # Distributed processing:
    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 0 --world_size 6
    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 1 --world_size 6
    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 2 --world_size 6
    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 3 --world_size 6
    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 4 --world_size 6
    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --resolution 512 --rank 5 --world_size 6

    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --skip_completed --resolution 512 --rank 0 --world_size 2
    # python data_toolkit/erp/step4_voxelize_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --skip_completed --resolution 512 --rank 1 --world_size 2

    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step4_voxelize_pbr_layout_wo_ceiling_{args.resolution}{log_suffix}.json')
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
    total_processed = 0
    total_failed = 0
    total_voxels = 0
    resolution = args.resolution

    for i, room_info in enumerate(tqdm(rooms, desc="Processing rooms")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"
        result = process_room(room_info, resolution)

        # Update counters
        if result['processed']:
            total_processed += 1
            total_voxels += result['num_voxels']
        else:
            total_failed += 1

        # Log result
        log.log_room(room_key, result)

        # Save log periodically
        if (i + 1) % args.log_interval == 0:
            log.update_summary(total_rooms, total_processed, total_failed, total_voxels)
            log.save()

    # Final log save
    log.update_summary(total_rooms, total_processed, total_failed, total_voxels)
    log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {total_processed}")
    print(f"  Rooms failed: {total_failed}")
    print(f"  Total voxels: {total_voxels}")
    print(f"  Avg voxels per room: {total_voxels / max(1, total_processed):.0f}")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
