# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 1: Dump mesh from ERP_3D_FRONT dataset.

Processes both full_room (scene-level) and individual_assets (asset-level) with
SHARED normalization parameters. All assets are normalized using the room's bbox,
preserving spatial alignment for OmniPart-style part-aware generation.

Input structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        mesh/full_room_wo_ceiling.obj
        mesh/individual_assets/*.glb

Output structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        mesh_dumps/full_room_wo_ceiling.pickle
        mesh_dumps/individual_assets/{asset_name}.pickle
        mesh_dumps/normalization_info.json  # Room's center and scale

Logging:
    datasets/ERP_3D_FRONT_test/logs/step1_dump_mesh.json
    - Tracks processed rooms, success/failure status, timestamps
    - Enables resumable processing

Usage:
    python data_toolkit/erp/step1_dump_mesh_erp.py --root datasets/ERP_3D_FRONT_test
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from subprocess import DEVNULL, call
from tqdm import tqdm
from datetime import datetime
import glob

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
            'step': 'step1_dump_mesh',
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_rooms': 0,
                'rooms_processed': 0,
                'rooms_failed': 0,
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
            'assets_processed': result['assets_processed'],
            'assets_failed': result['assets_failed'],
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms: int, rooms_processed: int, rooms_failed: int,
                       assets_processed: int, assets_failed: int):
        """Update summary statistics."""
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed,
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
        # Skip logs directory
        if uuid_dir == 'logs':
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            if os.path.isdir(room_path) and os.path.exists(os.path.join(room_path, 'mesh')):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path
                })
    return rooms


def process_room_blender(room_path: str, output_dir: str) -> bool:
    """
    Process room using Blender script.

    The Blender script handles:
    - Loading room mesh
    - Calculating normalization from room bbox
    - Applying same normalization to all assets
    - Saving all pickles and normalization_info.json

    Args:
        room_path: Path to room directory (containing mesh/ folder)
        output_dir: Path to output directory (mesh_dumps/)

    Returns:
        True if successful, False otherwise
    """
    script_path = os.path.join(os.path.dirname(__file__), '..', 'blender_script', 'dump_mesh_erp.py')

    args = [
        BLENDER_PATH, '-b', '-P', script_path,
        '--',
        '--room_dir', room_path,
        '--output_dir', output_dir
    ]

    try:
        ret = call(args, stdout=DEVNULL, stderr=DEVNULL)
        # Check if outputs were created
        room_pickle = os.path.join(output_dir, 'full_room_wo_ceiling.pickle')
        norm_info = os.path.join(output_dir, 'normalization_info.json')
        return os.path.exists(room_pickle) and os.path.exists(norm_info)
    except Exception as e:
        print(f"Error processing room {room_path}: {e}")
        return False


def count_assets(output_dir: str) -> tuple:
    """Count processed assets in output directory."""
    assets_dir = os.path.join(output_dir, 'individual_assets')
    if not os.path.exists(assets_dir):
        return 0, 0

    processed = len(glob.glob(os.path.join(assets_dir, '*.pickle')))
    return processed, 0  # We don't track failed assets separately in this flow


def process_room(room_info: dict) -> dict:
    """
    Process a single room.

    Args:
        room_info: Dict with uuid, room_name, room_path

    Returns:
        Dict with processing results
    """
    room_path = room_info['room_path']
    output_dir = os.path.join(room_path, 'mesh_dumps')

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'room_processed': False,
        'assets_processed': 0,
        'assets_failed': 0
    }

    # Check if already processed (room + assets)
    room_pickle = os.path.join(output_dir, 'full_room_wo_ceiling.pickle')
    norm_info = os.path.join(output_dir, 'normalization_info.json')
    assets_output_dir = os.path.join(output_dir, 'individual_assets')

    # Count source assets
    source_assets_dir = os.path.join(room_path, 'mesh', 'individual_assets')
    source_asset_count = len(glob.glob(os.path.join(source_assets_dir, '*.glb'))) if os.path.exists(source_assets_dir) else 0

    # Check if room and all assets are processed
    if os.path.exists(room_pickle) and os.path.exists(norm_info):
        output_asset_count = len(glob.glob(os.path.join(assets_output_dir, '*.pickle'))) if os.path.exists(assets_output_dir) else 0

        # Only skip if assets are also processed (or no source assets)
        if source_asset_count == 0 or output_asset_count >= source_asset_count:
            results['room_processed'] = True
            results['assets_processed'], results['assets_failed'] = count_assets(output_dir)
            return results
        # else: re-process because assets are missing

    # NOTE: Process room with Blender
    success = process_room_blender(room_path, output_dir)
    results['room_processed'] = success

    if success:
        results['assets_processed'], results['assets_failed'] = count_assets(output_dir)

    return results


def main():
    parser = argparse.ArgumentParser(description='Dump mesh from ERP_3D_FRONT dataset')
    parser.add_argument('--root', type=str, default='datasets/ERP_3D_FRONT',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers (not used currently)')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()

    # python data_toolkit/erp/step1_dump_mesh_erp.py --root datasets/ERP_3D_FRONT --skip_completed
    # python data_toolkit/erp/step1_dump_mesh_erp.py --root datasets/ERP_3D_FRONT_test
    # args.root = 'datasets/ERP_3D_FRONT_test'
    # python data_toolkit/erp/step1_dump_mesh_erp.py --root figure_sample_tmp

    # Check blender
    if not os.path.exists(BLENDER_PATH):
        raise RuntimeError(f"Blender not found at {BLENDER_PATH}")

    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step1_dump_mesh{log_suffix}.json')
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
    total_assets_processed = 0
    total_assets_failed = 0

    for i, room_info in enumerate(tqdm(rooms, desc="Processing rooms")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"
        result = process_room(room_info)

        # Update counters
        if result['room_processed']:
            total_room_processed += 1
        else:
            total_room_failed += 1
        total_assets_processed += result['assets_processed']
        total_assets_failed += result['assets_failed']

        # Log result
        log.log_room(room_key, result)

        # Save log periodically
        if (i + 1) % args.log_interval == 0:
            log.update_summary(total_rooms, total_room_processed, total_room_failed,
                             total_assets_processed, total_assets_failed)
            log.save()

    # Final log save
    log.update_summary(total_rooms, total_room_processed, total_room_failed,
                      total_assets_processed, total_assets_failed)
    log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {total_room_processed}")
    print(f"  Rooms failed: {total_room_failed}")
    print(f"  Assets processed: {total_assets_processed}")
    print(f"  Assets failed: {total_assets_failed}")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
