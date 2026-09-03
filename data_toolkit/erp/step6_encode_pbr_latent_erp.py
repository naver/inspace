# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 6: Encode PBR latents for ERP_3D_FRONT dataset.

Encodes O-Voxel PBR attributes (base_color, metallic, roughness, alpha) into
texture latents using the pretrained encoder.

For individual assets, supports two modes (matching step4_voxelize_pbr_erp.py):
- room_coord: position relative to room (sparse voxels, OmniPart-style)
- normalized: individually normalized (max resolution)

Input structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        pbr_voxels_{resolution}/full_room_wo_ceiling.vxz
        pbr_voxels_{resolution}/individual_assets_room_coord/{asset_name}.vxz
        pbr_voxels_{resolution}/individual_assets_normalized/{asset_name}.vxz

Output structure:
    datasets/ERP_3D_FRONT_test/{uuid}/{room_name}/
        pbr_latents/{encoder_name}_{resolution}/full_room_wo_ceiling.npz
        pbr_latents/{encoder_name}_{resolution}/individual_assets_room_coord/{asset_name}.npz
        pbr_latents/{encoder_name}_{resolution}/individual_assets_normalized/{asset_name}.npz

Logging:
    datasets/ERP_3D_FRONT_test_logs/step6_encode_pbr_{encoder}_{resolution}.json

Usage:
    python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512
    python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --asset_mode both
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
import glob

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
            'step': 'step6_encode_pbr',
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_rooms': 0,
                'rooms_processed': 0,
                'rooms_failed': 0,
                'total_tokens': 0,
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
            'room_tokens': result['room_tokens'],
            'assets_processed': result['assets_processed'],
            'assets_failed': result['assets_failed'],
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms: int, rooms_processed: int, rooms_failed: int,
                       total_tokens: int, assets_processed: int, assets_failed: int):
        """Update summary statistics."""
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed,
            'total_tokens': total_tokens,
            'assets_processed': assets_processed,
            'assets_failed': assets_failed
        }


def find_all_rooms(root: str, resolution: int) -> list:
    """Find all room directories that have pbr_voxels files."""
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
        SparseTensor with PBR attributes
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


def load_pbr_voxels_raw(vxz_path: str) -> tuple:
    """
    Load PBR voxels and return raw tensors (for batching).

    Returns:
        feats (torch.Tensor), coords (torch.Tensor)
    """
    attrs = ['base_color', 'metallic', 'roughness', 'alpha']
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)

    # Concatenate attributes and normalize to [-1, 1]
    feats = torch.cat([attr[k] for k in attrs], dim=-1) / 255.0 * 2 - 1

    return feats.float(), coords


def is_valid_sparse_tensor(tensor) -> bool:
    """Check if sparse tensor has valid values."""
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def process_assets_batched(asset_files: list, assets_output_dir: str, encoder, batch_size: int = 8) -> tuple:
    """
    Process multiple assets in batches for better GPU utilization.

    Args:
        asset_files: List of asset vxz file paths
        assets_output_dir: Output directory for encoded latents
        encoder: PBR encoder model
        batch_size: Number of assets to process in one batch

    Returns:
        (assets_processed, assets_failed)
    """
    assets_processed = 0
    assets_failed = 0

    # Filter out already processed assets
    pending_assets = []
    for asset_path in asset_files:
        asset_name = os.path.splitext(os.path.basename(asset_path))[0]
        asset_output_path = os.path.join(assets_output_dir, f'{asset_name}.npz')
        if not os.path.exists(asset_output_path):
            pending_assets.append((asset_path, asset_name, asset_output_path))
        else:
            assets_processed += 1

    if not pending_assets:
        return assets_processed, assets_failed

    # Process in batches
    for batch_start in range(0, len(pending_assets), batch_size):
        batch_end = min(batch_start + batch_size, len(pending_assets))
        batch_items = pending_assets[batch_start:batch_end]

        # Load batch data
        batch_feats = []
        batch_coords = []
        valid_indices = []

        for i, (asset_path, asset_name, _) in enumerate(batch_items):
            try:
                feats, coords = load_pbr_voxels_raw(asset_path)

                # Check validity
                if torch.isfinite(feats).all() and torch.isfinite(coords.float()).all():
                    batch_feats.append(feats)
                    batch_coords.append(coords)
                    valid_indices.append(i)
                else:
                    assets_failed += 1

            except Exception as e:
                print(f"Error loading asset {asset_name}: {e}")
                assets_failed += 1

        if not valid_indices:
            continue

        try:
            # Create batched SparseTensors
            # Add batch index to coords
            batched_coords = []
            for batch_idx, coords in enumerate(batch_coords):
                coords_with_batch = torch.cat([
                    torch.full((coords.shape[0], 1), batch_idx, dtype=coords.dtype),
                    coords
                ], dim=-1)
                batched_coords.append(coords_with_batch)

            all_feats = torch.cat(batch_feats, dim=0)
            all_coords = torch.cat(batched_coords, dim=0)

            voxels_batched = sp.SparseTensor(all_feats, all_coords)

            # Encode batch (single forward pass - neighbor map computed once!)
            z_batched = encoder(voxels_batched.cuda())
            torch.cuda.synchronize()

            # Split results back using layout
            z_feats_list, z_coords_list = z_batched.to_tensor_list()

            # Save individual results
            for list_idx, orig_idx in enumerate(valid_indices):
                asset_path, asset_name, asset_output_path = batch_items[orig_idx]

                z_feats = z_feats_list[list_idx]
                z_coords = z_coords_list[list_idx]

                if torch.isfinite(z_feats).all():
                    pack = {
                        'feats': z_feats.cpu().numpy().astype(np.float32),
                        'coords': z_coords[:, 1:].cpu().numpy().astype(np.uint8)  # Remove batch index
                    }
                    np.savez_compressed(asset_output_path, **pack)
                    assets_processed += 1
                else:
                    print(f"[Skip] {asset_name}: Non-finite latent")
                    assets_failed += 1

        except Exception as e:
            print(f"Error processing batch: {e}")
            assets_failed += len(valid_indices)
            clear_cuda_error()

    return assets_processed, assets_failed


def process_room(room_info: dict, encoder, latent_name: str, resolution: int, mode: str = 'all', asset_mode: str = 'both', batch_size: int = 8) -> dict:
    """
    Process a single room.

    Args:
        room_info: Dict with room info
        encoder: PBR encoder model
        latent_name: Name for output folder
        resolution: O-Voxel resolution
        mode: 'all', 'room_only', 'assets_only'
        asset_mode: 'both', 'room_coord', 'normalized'
        batch_size: Batch size for asset encoding

    Returns:
        Dict with processing results
    """
    room_path = room_info['room_path']
    pbr_voxels_dir = room_info['pbr_voxels_dir']
    output_dir = os.path.join(room_path, 'pbr_latents', latent_name)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'room_processed': False,
        'room_tokens': 0,
        'assets_processed': 0,
        'assets_failed': 0
    }

    # Process full room
    if mode in ['all', 'room_only']:
        room_vxz_path = os.path.join(pbr_voxels_dir, 'full_room_wo_ceiling.vxz')
        room_output_path = os.path.join(output_dir, 'full_room_wo_ceiling.npz')

        if os.path.exists(room_vxz_path) and not os.path.exists(room_output_path):
            try:
                voxels = load_pbr_voxels(room_vxz_path)

                if not is_valid_sparse_tensor(voxels):
                    print(f"[Skip] {room_info['uuid']}/{room_info['room_name']}: NaN/Inf in input")
                else:
                    z = encoder(voxels.cuda())
                    torch.cuda.synchronize()

                    if torch.isfinite(z.feats).all():
                        pack = {
                            'feats': z.feats.cpu().numpy().astype(np.float32),
                            'coords': z.coords[:, 1:].cpu().numpy().astype(np.uint8)
                        }
                        np.savez_compressed(room_output_path, **pack)
                        results['room_processed'] = True
                        results['room_tokens'] = pack['coords'].shape[0]
                    else:
                        print(f"[Skip] {room_info['uuid']}/{room_info['room_name']}: Non-finite latent")

            except Exception as e:
                print(f"Error processing room {room_info['uuid']}/{room_info['room_name']}: {e}")
                clear_cuda_error()

        elif os.path.exists(room_output_path):
            results['room_processed'] = True
            try:
                data = np.load(room_output_path)
                results['room_tokens'] = data['coords'].shape[0]
            except:
                pass

    # Process individual assets
    if mode in ['all', 'assets_only']:
        do_room_coord = asset_mode in ['both', 'room_coord']
        do_normalized = asset_mode in ['both', 'normalized']

        asset_dirs = []
        if do_room_coord:
            asset_dirs.append(('individual_assets_room_coord', 'individual_assets_room_coord'))
        if do_normalized:
            asset_dirs.append(('individual_assets_normalized', 'individual_assets_normalized'))

        # ============================================================
        # Batched asset processing (faster due to shared neighbor map computation)
        # ============================================================
        for input_subdir, output_subdir in asset_dirs:
            assets_vxz_dir = os.path.join(pbr_voxels_dir, input_subdir)
            if os.path.exists(assets_vxz_dir):
                asset_files = glob.glob(os.path.join(assets_vxz_dir, '*.vxz'))
                assets_output_dir = os.path.join(output_dir, output_subdir)
                os.makedirs(assets_output_dir, exist_ok=True)

                # Process assets in batches
                processed, failed = process_assets_batched(
                    asset_files, assets_output_dir, encoder, batch_size=batch_size
                )
                results['assets_processed'] += processed
                results['assets_failed'] += failed

        # ============================================================
        # OLD: Sequential asset processing (slow - neighbor map computed per asset)
        # ============================================================
        # for input_subdir, output_subdir in tqdm(asset_dirs, desc="Processing asset dirs", leave=False):
        #     assets_vxz_dir = os.path.join(pbr_voxels_dir, input_subdir)
        #     if os.path.exists(assets_vxz_dir):
        #         asset_files = glob.glob(os.path.join(assets_vxz_dir, '*.vxz'))
        #         assets_output_dir = os.path.join(output_dir, output_subdir)
        #         os.makedirs(assets_output_dir, exist_ok=True)
        #
        #         for asset_path in tqdm(asset_files, desc="Processing assets", leave=False):
        #             asset_name = os.path.splitext(os.path.basename(asset_path))[0]
        #             asset_output_path = os.path.join(assets_output_dir, f'{asset_name}.npz')
        #
        #             if not os.path.exists(asset_output_path):
        #                 try:
        #                     voxels = load_pbr_voxels(asset_path)
        #
        #                     if is_valid_sparse_tensor(voxels):
        #                         z = encoder(voxels.cuda())
        #                         torch.cuda.synchronize()
        #
        #                         if torch.isfinite(z.feats).all():
        #                             pack = {
        #                                 'feats': z.feats.cpu().numpy().astype(np.float32),
        #                                 'coords': z.coords[:, 1:].cpu().numpy().astype(np.uint8)
        #                             }
        #                             np.savez_compressed(asset_output_path, **pack)
        #                             results['assets_processed'] += 1
        #                         else:
        #                             results['assets_failed'] += 1
        #                     else:
        #                         results['assets_failed'] += 1
        #
        #                 except Exception as e:
        #                     print(f"Error processing asset {asset_name}: {e}")
        #                     results['assets_failed'] += 1
        #                     clear_cuda_error()
        #             else:
        #                 results['assets_processed'] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description='Encode PBR latents for ERP_3D_FRONT dataset')
    parser.add_argument('--root', type=str, default='datasets/ERP_3D_FRONT',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--resolution', type=int, default=512,
                        help='O-Voxel resolution')
    parser.add_argument('--enc_pretrained', type=str,
                        default='microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'room_only', 'assets_only'],
                        help='Processing mode')
    parser.add_argument('--asset_mode', type=str, default='room_coord',
                        choices=['both', 'room_coord', 'normalized'],
                        help='Asset encoding mode: room_coord (relative position), normalized (max resolution), both')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for asset encoding')
    args = parser.parse_args()

    # Override defaults for testing (comment out for production)
    # args.root = 'datasets/ERP_3D_FRONT'
    args.asset_mode = 'room_coord'
    args.mode = 'all'
    args.resolution = 512
    args.batch_size = 8

    # python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512
    # python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 0 --world_size 6

    # CUDA_VISIBLE_DEVICES=4 python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 0 --world_size 4
    # CUDA_VISIBLE_DEVICES=4 python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 1 --world_size 4
    # CUDA_VISIBLE_DEVICES=4 python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 2 --world_size 4
    # CUDA_VISIBLE_DEVICES=4 python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 3 --world_size 4

    # CUDA_VISIBLE_DEVICES=4 python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 0 --world_size 2
    # CUDA_VISIBLE_DEVICES=4 python data_toolkit/erp/step6_encode_pbr_latent_erp.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 1 --world_size 2
   
    # Load encoder
    print("Loading encoder...")
    latent_name = f'{args.enc_pretrained.split("/")[-1]}_{args.resolution}'
    encoder = models.from_pretrained(args.enc_pretrained).eval().cuda()
    print(f"Encoder loaded: {latent_name}")

    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step6_encode_pbr_{latent_name}{log_suffix}.json')
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

    # Process rooms
    total_room_processed = 0
    total_room_failed = 0
    total_room_tokens = 0
    total_assets_processed = 0
    total_assets_failed = 0

    for i, room_info in enumerate(tqdm(rooms, desc="Encoding PBR latents")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"
        result = process_room(room_info, encoder, latent_name, args.resolution, args.mode, args.asset_mode, args.batch_size)

        # Update counters
        if result['room_processed']:
            total_room_processed += 1
            total_room_tokens += result['room_tokens']
        else:
            total_room_failed += 1
        total_assets_processed += result['assets_processed']
        total_assets_failed += result['assets_failed']

        # Log result
        log.log_room(room_key, result)

        # Save log periodically
        if (i + 1) % args.log_interval == 0:
            log.update_summary(total_rooms, total_room_processed, total_room_failed,
                             total_room_tokens, total_assets_processed, total_assets_failed)
            log.save()

    # Final log save
    log.update_summary(total_rooms, total_room_processed, total_room_failed,
                      total_room_tokens, total_assets_processed, total_assets_failed)
    log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {total_room_processed}")
    print(f"  Rooms failed: {total_room_failed}")
    print(f"  Total room tokens: {total_room_tokens}")
    print(f"  Avg tokens per room: {total_room_tokens / max(1, total_room_processed):.0f}")
    print(f"  Assets processed: {total_assets_processed}")
    print(f"  Assets failed: {total_assets_failed}")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
