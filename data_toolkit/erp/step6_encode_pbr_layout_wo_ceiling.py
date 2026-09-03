# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 6: Encode PBR latents for layout_wo_ceiling ONLY.

Simplified version of step6_encode_pbr_latent_erp.py that processes ONLY
`layout_wo_ceiling` meshes (no individual assets).

Input structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        pbr_voxels_{resolution}/layout_wo_ceiling.vxz

Output structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        pbr_latents/{encoder_name}_{resolution}/layout_wo_ceiling.npz

Logging:
    datasets/ERP_3D_FRONT_logs/step6_encode_pbr_layout_wo_ceiling_{encoder}_{resolution}.json

Usage:
    CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step6_encode_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512
    CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step6_encode_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 0 --world_size 4
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
import argparse
import torch
import numpy as np
import o_voxel
from tqdm import tqdm
from datetime import datetime

import trellis2.models as models
import trellis2.modules.sparse as sp

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
            'step': 'step6_encode_pbr_layout_wo_ceiling',
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_rooms': 0,
                'rooms_processed': 0,
                'rooms_failed': 0,
                'total_tokens': 0
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
            'tokens': result['tokens'],
            'error': result.get('error', None),
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms: int, rooms_processed: int, rooms_failed: int, total_tokens: int):
        """Update summary statistics."""
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed,
            'total_tokens': total_tokens
        }


def find_all_rooms(root: str, resolution: int) -> list:
    """Find all room directories that have pbr_voxels_{resolution} dir."""
    rooms = []
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            pbr_voxels_dir = os.path.join(room_path, f'pbr_voxels_{resolution}')
            if os.path.isdir(room_path) and os.path.exists(pbr_voxels_dir):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path,
                    'pbr_voxels_dir': pbr_voxels_dir
                })
    return rooms


def load_pbr_voxels(vxz_path: str) -> sp.SparseTensor:
    """
    Load PBR voxels and convert to SparseTensor format.

    Returns:
        SparseTensor with PBR attributes (base_color, metallic, roughness, alpha)
    """
    attrs = ['base_color', 'metallic', 'roughness', 'alpha']
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)

    # Concatenate attributes and normalize to [-1, 1]
    feats = torch.cat([attr[k] for k in attrs], dim=-1) / 255.0 * 2 - 1

    x = sp.SparseTensor(
        feats.float(),
        torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1)
    )

    return x


def is_valid_sparse_tensor(tensor) -> bool:
    """Check if sparse tensor has valid values."""
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()



def process_room_single(room_info: dict, encoder, latent_name: str) -> dict:
    """
    Process a single room through the encoder.

    Args:
        room_info: Room info dict
        encoder: PBR encoder model
        latent_name: Name for output folder

    Returns:
        Result dict
    """
    room_path = room_info['room_path']
    pbr_voxels_dir = room_info['pbr_voxels_dir']
    output_dir = os.path.join(room_path, 'pbr_latents', latent_name)
    os.makedirs(output_dir, exist_ok=True)

    result = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'processed': False,
        'tokens': 0,
        'error': None
    }

    vxz_path = os.path.join(pbr_voxels_dir, 'layout_wo_ceiling.vxz')
    output_path = os.path.join(output_dir, 'layout_wo_ceiling.npz')

    # Skip if already exists
    if os.path.exists(output_path):
        result['processed'] = True
        try:
            data = np.load(output_path)
            result['tokens'] = data['coords'].shape[0]
        except:
            pass
        return result

    if not os.path.exists(vxz_path):
        result['error'] = 'layout_wo_ceiling.vxz not found'
        return result

    try:
        # Load voxels
        x = load_pbr_voxels(vxz_path)
        if not is_valid_sparse_tensor(x):
            result['error'] = 'NaN/Inf in input'
            print(f"[Skip] {room_info['uuid']}/{room_info['room_name']}: NaN/Inf in input")
            return result

        # Encode
        z = encoder(x.cuda())
        torch.cuda.synchronize()

        if torch.isfinite(z.feats).all():
            pack = {
                'feats': z.feats.cpu().numpy().astype(np.float32),
                'coords': z.coords[:, 1:].cpu().numpy().astype(np.uint16)
            }
            np.savez_compressed(output_path, **pack)
            result['processed'] = True
            result['tokens'] = pack['coords'].shape[0]
        else:
            result['error'] = 'Non-finite latent output'
            print(f"[Skip] {room_info['uuid']}/{room_info['room_name']}: Non-finite latent")

    except Exception as e:
        result['error'] = str(e)
        print(f"Error processing {room_info['uuid']}/{room_info['room_name']}: {e}")
        try:
            torch.cuda.synchronize()
        except:
            pass
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description='Encode PBR latents for layout_wo_ceiling only')
    parser.add_argument('--root', type=str,
                        default='datasets/ERP_3D_FRONT',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--resolution', type=int, default=512,
                        help='O-Voxel resolution')
    parser.add_argument('--enc_pretrained', type=str,
                        default='microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()

    # Override defaults for testing (comment out for production)
    args.resolution = 512

    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step6_encode_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 0 --world_size 4
    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step6_encode_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 1 --world_size 4
    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step6_encode_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 2 --world_size 4
    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step6_encode_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 3 --world_size 4

    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step6_encode_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 0 --world_size 2
    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step6_encode_pbr_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 1 --world_size 2

    # Load encoder
    print("Loading encoder...")
    latent_name = f'{args.enc_pretrained.split("/")[-1]}_{args.resolution}'
    encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()
    print(f"Encoder loaded: {latent_name}")

    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step6_encode_pbr_layout_wo_ceiling_{latent_name}{log_suffix}.json')
    log = ProcessingLog(log_path)
    print(f"Logging to: {log_path}")

    # Find all rooms
    print("Finding rooms...")
    rooms = find_all_rooms(args.root, args.resolution)
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

    # Process rooms one at a time (batching causes CUDA illegal memory access with large voxels)
    total_processed = 0
    total_failed = 0
    total_tokens = 0

    for i, room_info in enumerate(tqdm(rooms, desc="Encoding PBR latents (layout_wo_ceiling)")):
        result = process_room_single(room_info, encoder, latent_name)
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"

        if result['processed']:
            total_processed += 1
            total_tokens += result['tokens']
        else:
            total_failed += 1

        log.log_room(room_key, result)

        # Save log periodically
        if (i + 1) % args.log_interval == 0:
            log.update_summary(total_rooms, total_processed, total_failed, total_tokens)
            log.save()

    # Final log save
    log.update_summary(total_rooms, total_processed, total_failed, total_tokens)
    log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {total_processed}")
    print(f"  Rooms failed: {total_failed}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg tokens per room: {total_tokens / max(1, total_processed):.0f}")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
