# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 1: Dump mesh for layout_wo_ceiling.obj from ERP_3D_FRONT dataset.

Simplified version of step1_dump_mesh_erp.py that processes ONLY layout_wo_ceiling.obj
(walls + floor + door + baseboard). No individual assets processing.

IMPORTANT: Uses normalization (center, scale) from full_room_wo_ceiling's
normalization_info.json so that layout voxels share the same coordinate system
as the full room. Requires step1_dump_mesh_erp.py to have run first.

Input structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        mesh/layout_wo_ceiling.obj
        mesh_dumps/normalization_info.json  # From step1_dump_mesh_erp.py (full_room)

Output structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        mesh_dumps/layout_wo_ceiling.pickle

Logging:
    datasets/ERP_3D_FRONT_logs/step1_dump_mesh_layout_wo_ceiling{_rankN}.json

Usage:
    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT
    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test
"""

import os
import sys
import json
import argparse
from pathlib import Path
from subprocess import DEVNULL, call
from tqdm import tqdm
from datetime import datetime

# Blender path
BLENDER_PATH = 'blender-4.5.1-linux-x64/blender'


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
            'step': 'step1_dump_mesh_layout_wo_ceiling',
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_rooms': 0,
                'rooms_processed': 0,
                'rooms_failed': 0
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

    def log_room(self, room_key: str, success: bool):
        """Log processing result for a room."""
        self.data['rooms'][room_key] = {
            'status': 'completed' if success else 'failed',
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms: int, rooms_processed: int, rooms_failed: int):
        """Update summary statistics."""
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed
        }


def find_all_rooms(root: str) -> list:
    """Find all room directories that have mesh/layout_wo_ceiling.obj."""
    rooms = []
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path):
            continue
        # Skip logs directory
        if uuid_dir == 'logs':
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            layout_obj = os.path.join(room_path, 'mesh', 'layout_wo_ceiling.obj')
            if os.path.isdir(room_path) and os.path.exists(layout_obj):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path
                })
    return rooms


def process_room_blender(room_path: str, output_dir: str, norm_info_path: str) -> bool:
    """
    Process room layout using Blender script.

    The Blender script handles:
    - Loading layout_wo_ceiling.obj
    - Using full_room normalization from normalization_info.json
    - Saving pickle

    Args:
        room_path: Path to room directory (containing mesh/ folder)
        output_dir: Path to output directory (mesh_dumps/)
        norm_info_path: Path to normalization_info.json from full_room_wo_ceiling

    Returns:
        True if successful, False otherwise
    """
    script_path = os.path.join(os.path.dirname(__file__), '..', 'blender_script', 'dump_mesh_layout_wo_ceiling.py')

    args = [
        BLENDER_PATH, '-b', '-P', script_path,
        '--',
        '--room_dir', room_path,
        '--output_dir', output_dir,
        '--norm_info_path', norm_info_path
    ]

    try:
        ret = call(args, stdout=DEVNULL, stderr=DEVNULL)
        layout_pickle = os.path.join(output_dir, 'layout_wo_ceiling.pickle')
        return os.path.exists(layout_pickle)
    except Exception as e:
        print(f"Error processing room {room_path}: {e}")
        return False


def process_room(room_info: dict, skip_completed: bool = False) -> bool:
    """
    Process a single room.

    Args:
        room_info: Dict with uuid, room_name, room_path
        skip_completed: If True, skip if outputs already exist

    Returns:
        True if processing succeeded (or was already done), False otherwise
    """
    room_path = room_info['room_path']
    output_dir = os.path.join(room_path, 'mesh_dumps')

    # Check if already processed
    layout_pickle = os.path.join(output_dir, 'layout_wo_ceiling.pickle')

    if skip_completed and os.path.exists(layout_pickle):
        return True

    # Check that full_room normalization_info.json exists (from step1_dump_mesh_erp.py)
    norm_info_path = os.path.join(output_dir, 'normalization_info.json')
    if not os.path.exists(norm_info_path):
        print(f"[Skip] {room_info['uuid']}/{room_info['room_name']}: normalization_info.json not found. Run step1_dump_mesh_erp.py first.")
        return False

    # Process with Blender
    return process_room_blender(room_path, output_dir, norm_info_path)


def main():
    parser = argparse.ArgumentParser(description='Dump mesh for layout_wo_ceiling.obj from ERP_3D_FRONT dataset')
    parser.add_argument('--root', type=str, default='datasets/ERP_3D_FRONT',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()

    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --rank 0 --world_size 4
    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --rank 1 --world_size 4
    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --rank 2 --world_size 4
    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --skip_completed --rank 3 --world_size 4

    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 0 --world_size 4
    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 1 --world_size 4
    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 2 --world_size 4
    # python data_toolkit/erp/step1_dump_mesh_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 3 --world_size 4

    # Check blender
    if not os.path.exists(BLENDER_PATH):
        raise RuntimeError(f"Blender not found at {BLENDER_PATH}")

    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step1_dump_mesh_layout_wo_ceiling{log_suffix}.json')
    log = ProcessingLog(log_path)
    print(f"Logging to: {log_path}")

    # Find all rooms with layout_wo_ceiling.obj
    print("Finding rooms with layout_wo_ceiling.obj...")
    rooms = find_all_rooms(args.root)
    rooms.sort(key=lambda x: (x['uuid'], x['room_name']))

    total_rooms = len(rooms)
    print(f"Found {total_rooms} rooms with layout_wo_ceiling.obj")

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

    for i, room_info in enumerate(tqdm(rooms, desc="Processing rooms")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"
        success = process_room(room_info)

        # Update counters
        if success:
            total_room_processed += 1
        else:
            total_room_failed += 1

        # Log result
        log.log_room(room_key, success)

        # Save log periodically
        if (i + 1) % args.log_interval == 0:
            log.update_summary(total_rooms, total_room_processed, total_room_failed)
            log.save()

    # Final log save
    log.update_summary(total_rooms, total_room_processed, total_room_failed)
    log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {total_room_processed}")
    print(f"  Rooms failed: {total_room_failed}")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
