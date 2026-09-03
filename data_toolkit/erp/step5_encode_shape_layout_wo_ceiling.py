# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 5 (layout_wo_ceiling): Encode shape latents for layout_wo_ceiling only.

Simplified version of step5_encode_shape_latent_erp.py that processes ONLY layout_wo_ceiling
(no individual assets, no batched processing).

Input structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        dual_grid_{resolution}/layout_wo_ceiling.vxz

Output structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        shape_latents/{encoder_name}_{resolution}/layout_wo_ceiling.npz

Logging:
    datasets/ERP_3D_FRONT_logs/step5_encode_shape_layout_wo_ceiling_{encoder}_{resolution}.json

Usage:
    CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py \
        --root datasets/ERP_3D_FRONT --resolution 512
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
            'step': 'step5_encode_shape_layout_wo_ceiling',
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
            'tokens': result['tokens'],
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
    """Find all room directories that have dual_grid_{resolution} directory."""
    rooms = []
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            dual_grid_dir = os.path.join(room_path, f'dual_grid_{resolution}')
            if os.path.isdir(room_path) and os.path.exists(dual_grid_dir):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path,
                    'dual_grid_dir': dual_grid_dir
                })
    return rooms


def load_dual_grid(vxz_path: str) -> tuple:
    """
    Load dual grid data and convert to SparseTensor format.

    Returns:
        vertices (SparseTensor), intersected (SparseTensor)
    """
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)

    vertices = sp.SparseTensor(
        (attr['vertices'] / 255.0).float(),
        torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1)
    )

    intersected = vertices.replace(torch.cat([
        attr['intersected'] % 2,
        attr['intersected'] // 2 % 2,
        attr['intersected'] // 4 % 2,
    ], dim=-1).bool())

    return vertices, intersected


def is_valid_sparse_tensor(tensor) -> bool:
    """Check if sparse tensor has valid values."""
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def load_dual_grid_raw(vxz_path: str) -> tuple:
    """
    Load dual grid data and return raw tensors (for batching).

    Returns:
        feats (torch.Tensor), coords (torch.Tensor), intersected_feats (torch.Tensor)
    """
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)

    feats = (attr['vertices'] / 255.0).float()
    intersected_feats = torch.cat([
        attr['intersected'] % 2,
        attr['intersected'] // 2 % 2,
        attr['intersected'] // 4 % 2,
    ], dim=-1).bool()

    return feats, coords, intersected_feats


def process_rooms_batched(room_batch: list, encoder, latent_name: str) -> list:
    """
    Process multiple rooms in a single GPU forward pass.

    Args:
        room_batch: List of room_info dicts
        encoder: Shape encoder model
        latent_name: Name for output folder

    Returns:
        List of result dicts (same order as room_batch)
    """
    results = []
    pending = []  # (index_in_batch, room_info, vxz_path, output_path)

    # Prepare: check skip/exists for each room
    for i, room_info in enumerate(room_batch):
        room_path = room_info['room_path']
        dual_grid_dir = room_info['dual_grid_dir']
        output_dir = os.path.join(room_path, 'shape_latents', latent_name)
        os.makedirs(output_dir, exist_ok=True)

        result = {
            'uuid': room_info['uuid'],
            'room_name': room_info['room_name'],
            'processed': False,
            'tokens': 0
        }

        vxz_path = os.path.join(dual_grid_dir, 'layout_wo_ceiling.vxz')
        output_path = os.path.join(output_dir, 'layout_wo_ceiling.npz')

        if os.path.exists(output_path):
            result['processed'] = True
            try:
                data = np.load(output_path)
                result['tokens'] = data['coords'].shape[0]
            except:
                pass
            results.append(result)
        elif not os.path.exists(vxz_path):
            results.append(result)
        else:
            results.append(result)
            pending.append((len(results) - 1, room_info, vxz_path, output_path))

    if not pending:
        return results

    # Load raw data for batching
    batch_feats = []
    batch_coords = []
    batch_intersected = []
    valid_pending_indices = []  # indices into pending list

    for pi, (ri, room_info, vxz_path, output_path) in enumerate(pending):
        try:
            feats, coords, intersected_feats = load_dual_grid_raw(vxz_path)
            if torch.isfinite(feats).all() and torch.isfinite(coords.float()).all():
                batch_feats.append(feats)
                batch_coords.append(coords)
                batch_intersected.append(intersected_feats)
                valid_pending_indices.append(pi)
            else:
                print(f"[Skip] {room_info['uuid']}/{room_info['room_name']}: NaN/Inf in input")
        except Exception as e:
            print(f"Error loading {room_info['uuid']}/{room_info['room_name']}: {e}")

    if not valid_pending_indices:
        return results

    try:
        # Build batched SparseTensors
        batched_coords = []
        for batch_idx, coords in enumerate(batch_coords):
            coords_with_batch = torch.cat([
                torch.full((coords.shape[0], 1), batch_idx, dtype=coords.dtype),
                coords
            ], dim=-1)
            batched_coords.append(coords_with_batch)

        all_feats = torch.cat(batch_feats, dim=0)
        all_coords = torch.cat(batched_coords, dim=0)
        all_intersected = torch.cat(batch_intersected, dim=0)

        vertices_batched = sp.SparseTensor(all_feats, all_coords)
        intersected_batched = vertices_batched.replace(all_intersected)

        # Single forward pass
        z_batched = encoder(vertices_batched.cuda(), intersected_batched.cuda())
        torch.cuda.synchronize()

        # Split results back
        z_feats_list, z_coords_list = z_batched.to_tensor_list()

        for list_idx, pi in enumerate(valid_pending_indices):
            ri, room_info, vxz_path, output_path = pending[pi]
            z_feats = z_feats_list[list_idx]
            z_coords = z_coords_list[list_idx]

            if torch.isfinite(z_feats).all():
                pack = {
                    'feats': z_feats.cpu().numpy().astype(np.float32),
                    'coords': z_coords[:, 1:].cpu().numpy().astype(np.uint8)
                }
                np.savez_compressed(output_path, **pack)
                results[ri]['processed'] = True
                results[ri]['tokens'] = pack['coords'].shape[0]
            else:
                print(f"[Skip] {room_info['uuid']}/{room_info['room_name']}: Non-finite latent")

    except Exception as e:
        print(f"Error processing batch: {e}")
        clear_cuda_error()

    return results


def main():
    parser = argparse.ArgumentParser(description='Encode shape latents for layout_wo_ceiling only')
    parser.add_argument('--root', type=str,
                        default='datasets/ERP_3D_FRONT',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--resolution', type=int, default=512,
                        help='O-Voxel resolution')
    parser.add_argument('--enc_pretrained', type=str,
                        default='microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of rooms to encode in a single GPU forward pass')
    args = parser.parse_args()
    args.batch_size = 8

    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512

    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 0 --world_size 4
    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 1 --world_size 4
    # CUDA_VISIBLE_DEVICES=1 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 2 --world_size 4
    # CUDA_VISIBLE_DEVICES=1 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 3 --world_size 4

    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 0 --world_size 4
    # CUDA_VISIBLE_DEVICES=0 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 1 --world_size 4
    # CUDA_VISIBLE_DEVICES=1 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 2 --world_size 4
    # CUDA_VISIBLE_DEVICES=1 python data_toolkit/erp/step5_encode_shape_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 3 --world_size 4

    # Load encoder
    print("Loading encoder...")
    latent_name = f'{args.enc_pretrained.split("/")[-1]}_{args.resolution}'
    encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()
    print(f"Encoder loaded: {latent_name}")

    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step5_encode_shape_layout_wo_ceiling_{latent_name}{log_suffix}.json')
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

    # Process rooms in batches
    total_processed = 0
    total_failed = 0
    total_tokens = 0
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(rooms), batch_size), desc="Encoding shape latents (layout_wo_ceiling)"):
        batch_end = min(batch_start + batch_size, len(rooms))
        room_batch = rooms[batch_start:batch_end]

        batch_results = process_rooms_batched(room_batch, encoder, latent_name)

        for room_info, result in zip(room_batch, batch_results):
            room_key = f"{room_info['uuid']}/{room_info['room_name']}"

            if result['processed']:
                total_processed += 1
                total_tokens += result['tokens']
            else:
                total_failed += 1

            log.log_room(room_key, result)

        # Save log periodically
        rooms_done = batch_end
        if rooms_done % args.log_interval < batch_size:
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
