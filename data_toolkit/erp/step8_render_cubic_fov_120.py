# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 8: Convert ERP panorama images to cubemap views (FOV 120) for ERP_3D_FRONT dataset.

Converts equirectangular panorama images to 6 perspective views for use as
image conditions in the ERP-to-3D pipeline.

Input structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/erp/{view_idx}_colors.png

Output structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        cubic_fov_120/{view_idx}/
            front.png, right.png, back.png, left.png, top.png, bottom.png
        cubic_fov_120_concat/{view_idx}_concat.png

Logging:
    datasets/ERP_3D_FRONT_test_logs/step8_render_cubic_fov_120.json
    - Tracks processed rooms, success/failure status, timestamps
    - Enables resumable processing

Usage:
    python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT_test
    python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT_test --fov 90
    python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT_test --skip_completed
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime

try:
    import py360convert
except ImportError:
    print("Error: py360convert not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "py360convert"])
    import py360convert


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
            'step': 'step8_render_cubic_fov_120',
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_rooms': 0,
                'rooms_processed': 0,
                'rooms_failed': 0,
                'panos_processed': 0,
                'panos_skipped': 0,
                'panos_failed': 0
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
            'status': 'completed' if result['failed'] == 0 else 'partial',
            'processed': result['processed'],
            'skipped': result['skipped'],
            'failed': result['failed'],
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms: int, rooms_processed: int, rooms_failed: int,
                       panos_processed: int, panos_skipped: int, panos_failed: int):
        """Update summary statistics."""
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed,
            'panos_processed': panos_processed,
            'panos_skipped': panos_skipped,
            'panos_failed': panos_failed
        }


def convert_to_cubemap(pano_image: np.ndarray, face_size: int = 512, fov: float = 120) -> dict:
    """
    Convert panorama to 6 perspective views.

    Args:
        pano_image: Panorama image as numpy array (H, W, 3)
        face_size: Size of each face
        fov: Field of view in degrees

    Returns:
        dict with keys: 'front', 'back', 'left', 'right', 'top', 'bottom'
    """
    # Define yaw/pitch for each face direction
    face_directions = {
        'front':  (0, 0),      # yaw=0, pitch=0
        'right':  (90, 0),     # yaw=90, pitch=0
        'back':   (180, 0),    # yaw=180, pitch=0
        'left':   (270, 0),    # yaw=270, pitch=0
        'top':    (0, 90),     # yaw=0, pitch=90 (looking up)
        'bottom': (0, -90),    # yaw=0, pitch=-90 (looking down)
    }

    faces = {}
    for face_name, (yaw, pitch) in face_directions.items():
        face_img = py360convert.e2p(
            pano_image,
            fov_deg=(fov, fov),
            u_deg=yaw,
            v_deg=pitch,
            out_hw=(face_size, face_size),
            mode='bilinear'
        )
        faces[face_name] = face_img

    return faces


def create_cubic_concat(faces: dict, face_size: int = 512) -> Image.Image:
    """
    Create cubemap concatenation in cross layout.

    Layout:
             Top
       Left  Front  Right  Back
             Bottom

    Args:
        faces: dict with 6 face images
        face_size: Size of each face

    Returns:
        Concatenated image
    """
    concat_width = face_size * 4
    concat_height = face_size * 3
    cubic_concat = Image.new('RGB', (concat_width, concat_height))

    face_order = ['front', 'right', 'back', 'left', 'top', 'bottom']
    face_images = {name: Image.fromarray(faces[name].astype(np.uint8)) for name in face_order}

    # Paste faces in cross layout
    cubic_concat.paste(face_images['top'], (face_size, 0))
    cubic_concat.paste(face_images['left'], (0, face_size))
    cubic_concat.paste(face_images['front'], (face_size, face_size))
    cubic_concat.paste(face_images['right'], (face_size * 2, face_size))
    cubic_concat.paste(face_images['back'], (face_size * 3, face_size))
    cubic_concat.paste(face_images['bottom'], (face_size, face_size * 2))

    return cubic_concat


def process_single_pano(pano_path: str, output_dir: str, fov: float = 120, face_size: int = 512):
    """
    Process a single panorama image and save cubemap views.
    """
    # Load panorama image
    pano_img = Image.open(pano_path)
    pano_np = np.array(pano_img)

    # Get base filename (e.g., "0000" from "0000_colors.png")
    base_name = os.path.basename(pano_path).replace("_colors.png", "")

    # Create output directories
    cubic_dir = os.path.join(output_dir, "cubic_fov_120", base_name)
    cubic_concat_dir = os.path.join(output_dir, "cubic_fov_120_concat")
    os.makedirs(cubic_dir, exist_ok=True)
    os.makedirs(cubic_concat_dir, exist_ok=True)

    # Generate Cubemap (6 faces)
    cubemap = convert_to_cubemap(pano_np, face_size=face_size, fov=fov)

    # Save individual faces
    face_order = ['front', 'right', 'back', 'left', 'top', 'bottom']
    for face_name in face_order:
        face_img = cubemap[face_name]
        face_pil = Image.fromarray(face_img.astype(np.uint8))
        face_pil.save(os.path.join(cubic_dir, f"{face_name}.png"))

    # Create and save concatenation
    cubic_concat = create_cubic_concat(cubemap, face_size)
    cubic_concat.save(os.path.join(cubic_concat_dir, f"{base_name}_concat.png"))


def find_all_rooms(root: str) -> list:
    """Find all room directories with erp folders."""
    rooms = []
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            erp_dir = os.path.join(room_path, 'erp')
            if os.path.isdir(room_path) and os.path.exists(erp_dir):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path,
                    'erp_dir': erp_dir
                })
    return rooms


def process_room(room_info: dict, fov: float, face_size: int, skip_existing: bool) -> dict:
    """
    Process all panoramas in a single room.
    """
    room_path = room_info['room_path']
    erp_dir = room_info['erp_dir']

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'processed': 0,
        'skipped': 0,
        'failed': 0
    }

    # Find all color images
    color_files = sorted([f for f in os.listdir(erp_dir) if f.endswith('_colors.png')])

    for color_file in color_files:
        base_name = color_file.replace('_colors.png', '')
        color_path = os.path.join(erp_dir, color_file)

        # Check if output already exists
        concat_path = os.path.join(room_path, 'cubic_fov_120_concat', f'{base_name}_concat.png')
        if skip_existing and os.path.exists(concat_path):
            results['skipped'] += 1
            continue

        try:
            process_single_pano(color_path, room_path, fov=fov, face_size=face_size)
            results['processed'] += 1
        except Exception as e:
            print(f"Error processing {room_info['uuid']}/{room_info['room_name']}/{base_name}: {e}")
            results['failed'] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Render cubic FOV 120 for ERP_3D_FRONT dataset')
    # parser.add_argument('--root', type=str, required=True,
    #                     help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--fov', type=float, default=120,
                        help='Field of view in degrees')
    parser.add_argument('--face_size', type=int, default=512,
                        help='Size of each cubemap face')
    parser.add_argument('--no_skip', action='store_true',
                        help='Do not skip existing files')
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that were already completed (from log)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    # args.root = 'figure_sample_tmp'
    args.root = 'datasets/custom_samples'

    # args.root = "datasets/ERP_3D_FRONT_test"

    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT
    #python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT_test

    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 0 --world_size 10 
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 1 --world_size 10
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 2 --world_size 10
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 3 --world_size 10
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 4 --world_size 10
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 5 --world_size 10
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 6 --world_size 10
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 7 --world_size 10
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 8 --world_size 10
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT --skip_completed --rank 9 --world_size 10
    
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 0 --world_size 4
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 1 --world_size 4
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 2 --world_size 4
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 3 --world_size 4
    # python data_toolkit/erp/step8_render_cubic_fov_120.py --root figure_sample_tmp
    
    # Setup logging
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = os.path.join(os.path.dirname(args.root.rstrip('/')),
                           os.path.basename(args.root.rstrip('/')) + '_logs')
    log_path = os.path.join(log_dir, f'step8_render_cubic_fov_120{log_suffix}.json')
    log = ProcessingLog(log_path)
    print(f"Logging to: {log_path}")

    # Find all rooms
    print("Finding rooms...")
    rooms = find_all_rooms(args.root)
    total_rooms = len(rooms)
    print(f"Found {total_rooms} rooms")

    # Distribute across ranks
    start = len(rooms) * args.rank // args.world_size
    end = len(rooms) * (args.rank + 1) // args.world_size
    rooms = rooms[start:end]
    print(f"Processing {len(rooms)} rooms (rank {args.rank}/{args.world_size})")
    print(f"Settings: FOV={args.fov}, face_size={args.face_size}")

    # Process rooms
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    rooms_processed = 0
    rooms_failed = 0

    for i, room_info in enumerate(tqdm(rooms, desc="Converting to cubemap")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"

        # Skip if already completed
        if args.skip_completed and log.is_room_completed(room_key):
            total_skipped += 1
            continue

        try:
            result = process_room(room_info, args.fov, args.face_size, not args.no_skip)
            total_processed += result['processed']
            total_skipped += result['skipped']
            total_failed += result['failed']

            if result['failed'] == 0:
                rooms_processed += 1
            else:
                rooms_failed += 1

            log.log_room(room_key, result)

        except Exception as e:
            print(f"\nError processing {room_key}: {e}")
            rooms_failed += 1
            log.log_room(room_key, {'processed': 0, 'skipped': 0, 'failed': 1})

        # Save log periodically
        if (i + 1) % args.log_interval == 0:
            log.update_summary(total_rooms, rooms_processed, rooms_failed,
                               total_processed, total_skipped, total_failed)
            log.save()

    # Final log save
    log.update_summary(total_rooms, rooms_processed, rooms_failed,
                       total_processed, total_skipped, total_failed)
    log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {rooms_processed}")
    print(f"  Rooms failed: {rooms_failed}")
    print(f"  Panos processed: {total_processed}")
    print(f"  Panos skipped: {total_skipped}")
    print(f"  Panos failed: {total_failed}")
    print(f"\nOutput structure:")
    print(f"  cubic_fov_120/{{view_idx}}/: 6 cubemap faces")
    print(f"    - front.png, right.png, back.png, left.png, top.png, bottom.png")
    print(f"  cubic_fov_120_concat/{{view_idx}}_concat.png: cross layout")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
