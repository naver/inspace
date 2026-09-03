# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 10: Encode DA2 depth voxels to SS (Sparse Structure) latents.

Takes DA2 depth voxel PLY files (from step9_erp_depth_da2_to_voxels) and encodes them
through the SS encoder to produce latent representations. These latents serve as
"initial voxel" conditions for the flow matching model (SDEdit-style inference from
DA2 monocular depth-based geometry).

Input structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        depth_voxels_da2_{resolution}/{view_idx:04d}.ply

Output structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        depth_voxels_da2_ss_latent/{ss_encoder_name}_{ss_resolution}/{view_idx:04d}.npz

Usage:
    python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT_test
    python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT_test --resolution 64
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
import utils3d

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
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            'step': 'step10_encode_depth_da2_voxel_ss_latent',
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


def find_all_rooms(root: str, resolution: int) -> list:
    """Find all room directories that have DA2 depth voxel PLY files."""
    rooms = []
    depth_voxels_folder = f'depth_voxels_da2_{resolution}'
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path) or uuid_dir.startswith('.'):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            if not os.path.isdir(room_path) or room_name.startswith('.'):
                continue

            depth_voxels_dir = os.path.join(room_path, depth_voxels_folder)
            if os.path.isdir(depth_voxels_dir):
                ply_files = [f for f in os.listdir(depth_voxels_dir) if f.endswith('.ply')]
                if len(ply_files) > 0:
                    rooms.append({
                        'uuid': uuid_dir,
                        'room_name': room_name,
                        'room_path': room_path,
                        'depth_voxels_dir': depth_voxels_dir,
                    })
    return rooms


def load_ply_as_ss(ply_path: str, resolution: int) -> torch.Tensor:
    """
    Load a depth voxel PLY file and convert to dense binary occupancy grid.

    The PLY contains voxel centers in [-0.5, 0.5] range.
    Convert back to voxel indices and create binary occupancy grid.

    Args:
        ply_path: Path to PLY file with voxel centers
        resolution: Grid resolution (e.g., 64)

    Returns:
        Dense tensor [1, resolution, resolution, resolution] (binary occupancy)
    """
    verts = utils3d.io.read_ply(ply_path)
    if isinstance(verts, tuple):
        verts = verts[0]  # (N, 3) positions

    # Convert from [-0.5, 0.5] normalized coords to voxel indices
    coords = ((verts + 0.5) * resolution).astype(np.int32)
    coords = np.clip(coords, 0, resolution - 1)

    # Create binary occupancy grid
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    coords_t = torch.from_numpy(coords).long()
    ss[0, coords_t[:, 0], coords_t[:, 1], coords_t[:, 2]] = 1

    return ss


def process_room(room_info: dict, encoder, latent_name: str, resolution: int) -> dict:
    """
    Process all depth voxel views for a single room.

    Returns:
        Dict with processing results
    """
    room_path = room_info['room_path']
    depth_voxels_dir = room_info['depth_voxels_dir']
    output_dir = os.path.join(room_path, 'depth_voxels_da2_ss_latent', latent_name)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'views_processed': 0,
        'views_failed': 0,
    }

    # Find all PLY files
    ply_files = sorted([f for f in os.listdir(depth_voxels_dir) if f.endswith('.ply')])

    for ply_file in ply_files:
        view_name = os.path.splitext(ply_file)[0]  # e.g., "0000"
        output_path = os.path.join(output_dir, f'{view_name}.npz')

        # Skip if already exists
        if os.path.exists(output_path):
            results['views_processed'] += 1
            continue

        ply_path = os.path.join(depth_voxels_dir, ply_file)

        try:
            # Load PLY and convert to binary occupancy grid
            ss = load_ply_as_ss(ply_path, resolution)
            ss = ss.cuda()[None].float()  # [1, 1, R, R, R]

            # Encode through SS encoder
            z = encoder(ss, sample_posterior=False)
            torch.cuda.synchronize()

            if torch.isfinite(z).all():
                pack = {'z': z[0].cpu().numpy()}  # [8, 16, 16, 16]
                np.savez_compressed(output_path, **pack)
                results['views_processed'] += 1
            else:
                print(f"  [Skip] {room_info['uuid']}/{room_info['room_name']}/{view_name}: Non-finite latent")
                results['views_failed'] += 1

        except Exception as e:
            print(f"  [ERROR] {room_info['uuid']}/{room_info['room_name']}/{view_name}: {e}")
            results['views_failed'] += 1
            clear_cuda_error()

    return results


def main():
    parser = argparse.ArgumentParser(description='Step 10: Encode DA2 depth voxels to SS latents')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Voxel grid resolution (must match step9 output)')
    parser.add_argument('--enc_pretrained', type=str,
                        default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained SS encoder model')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()
    # args.root = "datasets/ERP_3D_FRONT_test"
    # args.root = "datasets/_ERP_3D_FRONT_before/ERP_3D_FRONT_test"
    args.resolution = 64

    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT_test

    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 0 --world_size 10
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 1 --world_size 10
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 2 --world_size 10
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 3 --world_size 10
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 4 --world_size 10
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 5 --world_size 10
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 6 --world_size 10
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 7 --world_size 10
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 8 --world_size 10
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT --skip_completed --rank 9 --world_size 10

    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT_test --rank 0 --world_size 2
    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT_test --rank 1 --world_size 2
    
    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 2 --world_size 4
    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root datasets/ERP_3D_FRONT_test --skip_completed --rank 3 --world_size 4
    # python data_toolkit/erp/step10_encode_depth_da2_voxel_ss_latent.py --root figure_sample_tmp

    # Load encoder
    print("Loading SS encoder...")
    latent_name = f'{args.enc_pretrained.split("/")[-1]}_{args.resolution}'
    encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()
    print(f"SS Encoder loaded: {latent_name}")

    # Initialize logging
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step10_encode_depth_da2_voxel_ss_{latent_name}{log_suffix}.json')
    log = ProcessingLog(log_path)
    print(f"Logging to: {log_path}")

    # Find all rooms
    print("Finding rooms...")
    rooms = find_all_rooms(args.root, args.resolution)
    rooms.sort(key=lambda x: (x['uuid'], x['room_name']))
    total_rooms = len(rooms)
    print(f"Found {total_rooms} rooms with DA2 depth voxels")

    # Distribute across ranks
    start = len(rooms) * args.rank // args.world_size
    end = len(rooms) * (args.rank + 1) // args.world_size
    rooms = rooms[start:end]
    print(f"Processing {len(rooms)} rooms (rank {args.rank}/{args.world_size})")

    # Filter completed
    # if args.skip_completed:
    #     original_count = len(rooms)
    #     rooms = [r for r in rooms if not log.is_room_completed(f"{r['uuid']}/{r['room_name']}")]
    #     skipped = original_count - len(rooms)
    #     if skipped > 0:
    #         print(f"Skipping {skipped} already completed rooms")

    # Process
    total_rooms_processed = 0
    total_rooms_failed = 0
    total_views_processed = 0
    total_views_failed = 0

    for i, room_info in enumerate(tqdm(rooms, desc="DA2 depth voxels → SS latent")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"
        result = process_room(room_info, encoder, latent_name, args.resolution)

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
