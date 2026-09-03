# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 7: Encode Sparse Structure (SS) latents for ERP_3D_FRONT dataset.

Converts shape latents to dense sparse structure representation and encodes
using the SS encoder. This is the latent used for first-stage generation.

NOTE: SS latent encoding is only for full room, not individual assets.
Individual assets don't need SS latents for the training pipeline.

Input structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        shape_latents/{shape_encoder_name}_{resolution}/full_room_wo_ceiling.npz

Output structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        ss_latents/{ss_encoder_name}_{ss_resolution}/full_room_wo_ceiling.npz

Logging:
    datasets/ERP_3D_FRONT_test_logs/step7_encode_ss_{encoder}_{resolution}.json

Usage:
    python data_toolkit/erp/step7_encode_ss_latent_erp.py --root datasets/ERP_3D_FRONT_test \\
        --shape_latent_name shape_enc_next_dc_f16c32_fp16_512 --resolution 64
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import glob

import trellis2.models as models

torch.set_grad_enabled(False)


def clear_cuda_error():
    """Clear CUDA error state and free memory."""
    try:
        torch.cuda.synchronize()
    except:
        pass
    torch.cuda.empty_cache()


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
            'step': 'step7_encode_ss',
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

    def log_room(self, room_key: str, result: dict):
        """Log processing result for a room."""
        self.data['rooms'][room_key] = {
            'status': 'completed' if result['room_processed'] else 'failed',
            'room_processed': result['room_processed'],
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms: int, rooms_processed: int, rooms_failed: int):
        """Update summary statistics."""
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed
        }


def find_all_rooms(root: str, shape_latent_name: str) -> list:
    """Find all room directories that have shape latent files."""
    rooms = []
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            shape_latent_dir = os.path.join(room_path, 'shape_latents', shape_latent_name)
            if os.path.isdir(room_path) and os.path.exists(shape_latent_dir):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path,
                    'shape_latent_dir': shape_latent_dir
                })
    return rooms


def load_shape_latent_as_ss(npz_path: str, resolution: int) -> torch.Tensor:
    """
    Load shape latent and convert to dense sparse structure.

    Returns:
        Dense tensor [1, resolution, resolution, resolution]
    """
    data = np.load(npz_path)
    coords = data['coords']

    # Validate coords
    if not np.all(coords < resolution):
        raise ValueError(f"Coords out of range: max={coords.max()}, resolution={resolution}")

    coords = torch.from_numpy(coords).long()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    return ss


def process_room(room_info: dict, encoder, latent_name: str, shape_latent_name: str,
                 resolution: int, mode: str = 'all') -> dict:
    """
    Process a single room.

    Args:
        room_info: Dict with room info
        encoder: SS encoder model
        latent_name: Name for output folder
        shape_latent_name: Name of shape latent folder
        resolution: SS resolution (e.g., 64)
        mode: 'all', 'room_only', 'assets_only'

    Returns:
        Dict with processing results
    """
    room_path = room_info['room_path']
    shape_latent_dir = room_info['shape_latent_dir']
    output_dir = os.path.join(room_path, 'ss_latents', latent_name)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'room_processed': False,
        'assets_processed': 0,
        'assets_failed': 0
    }

    # Process full room
    if mode in ['all', 'room_only']:
        room_npz_path = os.path.join(shape_latent_dir, 'full_room_wo_ceiling.npz')
        room_output_path = os.path.join(output_dir, 'full_room_wo_ceiling.npz')

        if os.path.exists(room_npz_path) and not os.path.exists(room_output_path):
            try:
                ss = load_shape_latent_as_ss(room_npz_path, resolution)
                ss = ss.cuda()[None].float()

                z = encoder(ss, sample_posterior=False)
                torch.cuda.synchronize()

                if torch.isfinite(z).all():
                    pack = {'z': z[0].cpu().numpy()}
                    np.savez_compressed(room_output_path, **pack)
                    results['room_processed'] = True
                else:
                    print(f"[Skip] {room_info['uuid']}/{room_info['room_name']}: Non-finite latent")

            except Exception as e:
                print(f"Error processing room {room_info['uuid']}/{room_info['room_name']}: {e}")
                clear_cuda_error()

        elif os.path.exists(room_output_path):
            results['room_processed'] = True

    # Process individual assets
    if mode in ['all', 'assets_only']:
        assets_npz_dir = os.path.join(shape_latent_dir, 'individual_assets')
        if os.path.exists(assets_npz_dir):
            asset_files = glob.glob(os.path.join(assets_npz_dir, '*.npz'))
            assets_output_dir = os.path.join(output_dir, 'individual_assets')
            os.makedirs(assets_output_dir, exist_ok=True)

            for asset_path in asset_files:
                asset_name = os.path.splitext(os.path.basename(asset_path))[0]
                asset_output_path = os.path.join(assets_output_dir, f'{asset_name}.npz')

                if not os.path.exists(asset_output_path):
                    try:
                        ss = load_shape_latent_as_ss(asset_path, resolution)
                        ss = ss.cuda()[None].float()

                        z = encoder(ss, sample_posterior=False)
                        torch.cuda.synchronize()

                        if torch.isfinite(z).all():
                            pack = {'z': z[0].cpu().numpy()}
                            np.savez_compressed(asset_output_path, **pack)
                            results['assets_processed'] += 1
                        else:
                            results['assets_failed'] += 1

                    except Exception as e:
                        print(f"Error processing asset {asset_name}: {e}")
                        results['assets_failed'] += 1
                        clear_cuda_error()
                else:
                    results['assets_processed'] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Encode SS latents for ERP_3D_FRONT dataset')
    parser.add_argument('--root', type=str, default='datasets/ERP_3D_FRONT_test',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--shape_latent_name', type=str, default='shape_enc_next_dc_f16c32_fp16_512',
                        help='Name of shape latent folder (e.g., shape_enc_next_dc_f16c32_fp16_512)')
    parser.add_argument('--resolution', type=int, default=64,
                        help='SS resolution')
    parser.add_argument('--enc_pretrained', type=str,
                        default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained SS encoder model')
    parser.add_argument('--mode', type=str, default='room_only',
                        choices=['all', 'room_only', 'assets_only'],
                        help='Processing mode (room_only recommended - assets dont need SS latents)')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()

    # Override defaults for testing (comment out for production)
    # args.root = 'datasets/ERP_3D_FRONT_test'
    args.shape_latent_name = 'shape_enc_next_dc_f16c32_fp16_256'
    args.resolution = 64

    # python data_toolkit/erp/step7_encode_ss_latent_erp.py --root datasets/ERP_3D_FRONT
    # python data_toolkit/erp/step7_encode_ss_latent_erp.py --root datasets/ERP_3D_FRONT_test

    

    # Load encoder
    print("Loading SS encoder...")
    latent_name = f'{args.enc_pretrained.split("/")[-1]}_{args.resolution}'
    encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()
    print(f"SS Encoder loaded: {latent_name}")

    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step7_encode_ss_{latent_name}{log_suffix}.json')
    log = ProcessingLog(log_path)
    print(f"Logging to: {log_path}")

    # Find all rooms
    print("Finding rooms...")
    rooms = find_all_rooms(args.root, args.shape_latent_name)
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

    for i, room_info in enumerate(tqdm(rooms, desc="Encoding SS latents")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"
        result = process_room(room_info, encoder, latent_name, args.shape_latent_name,
                             args.resolution, args.mode)

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
            log.update_summary(total_rooms, total_room_processed, total_room_failed)
            log.save()

    # Final log save
    log.update_summary(total_rooms, total_room_processed, total_room_failed)
    log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {total_room_processed}")
    print(f"  Rooms failed: {total_room_failed}")
    print(f"  Assets processed: {total_assets_processed}")
    print(f"  Assets failed: {total_assets_failed}")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
