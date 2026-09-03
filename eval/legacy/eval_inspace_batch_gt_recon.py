# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
GT Reconstruction for InSpace evaluation samples.

Loads GT shape/texture latents from the dataset and decodes them to GLBs.
No generation is performed — this provides the upper-bound reconstruction quality.

Output structure matches eval_inspace_batch.py:
    {output_dir}/{idx:04d}_{room_name}_v{view_idx}/
        scene.glb, layout.glb, assets/, assets.glb, metadata.json

Usage:
    python eval/viewers/eval_inspace_batch_gt_recon.py
    python eval/viewers/eval_inspace_batch_gt_recon.py --gpu_id 1
    python eval/viewers/eval_inspace_batch_gt_recon.py --max_samples 3
    python eval/viewers/eval_inspace_batch_gt_recon.py --no_texture  # shape only

    # Multi-GPU parallel
    python eval/viewers/eval_inspace_batch_gt_recon.py --rank 0 --world_size 4 --gpu_id 0
    python eval/viewers/eval_inspace_batch_gt_recon.py --rank 1 --world_size 4 --gpu_id 1
"""

import os
import sys
import json
import glob
import time
import argparse
import shutil
from typing import Dict, List, Optional
from tqdm import tqdm

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.pipeline.eval_pipeline import (
    load_gt_bboxes,
    load_camera_center,
    save_mesh_glb,
)
from trellis2 import models
from trellis2.modules.sparse.basic import SparseTensor
from trellis2.utils.asset_attention_mask import (
    filter_visible_assets,
)


# ============================================================
# Sample loading
# ============================================================

def load_eval_samples(json_path):
    """Load evaluation samples from perspective_eval_dataset_selected.json."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for entry in data['samples']:
        uuid = entry['uuid']
        room_name = entry['room_name']
        view_idx = entry['view_idx']
        idx = entry['idx']
        folder_name = f"{idx:04d}_{room_name}_v{view_idx}"

        samples.append({
            'scene_id': uuid,
            'room_id': room_name,
            'view_idx': view_idx,
            'idx': idx,
            'folder_name': folder_name,
            'erp_image': entry['erp_image'],
            'perspective_image': entry['perspective_image'],
            'room_path': entry['room_path'],
        })

    return samples


# ============================================================
# GT Latent Loading
# ============================================================

def align_bbox_with_latents(bbox_filenames, latent_files):
    """Align bbox asset filenames with latent files by instance number."""
    import re

    def extract_inst(name):
        m = re.search(r'_inst(\d+)', name)
        return int(m.group(1)) if m else None

    latent_map = {}
    for lf in latent_files:
        inst = extract_inst(lf)
        if inst is not None:
            latent_map[inst] = lf

    aligned = []
    for bbox_idx, bf in enumerate(bbox_filenames):
        inst = extract_inst(bf)
        if inst is not None and inst in latent_map:
            aligned.append((bbox_idx, latent_map[inst]))

    return aligned


def load_latent(path):
    """Load coords and feats from an npz file."""
    data = np.load(path)
    coords = data['coords'].astype(np.int32)
    feats = data['feats'].astype(np.float32)
    return coords, feats


def load_gt_latents(data_dir, scene_id, room_id, shape_encoder, tex_encoder=None,
                    visibility_threshold=0.5, view_idx=0, layout_mode='floor_perimeter_clean'):
    """
    Load GT shape (and optionally texture) latents for a sample.
    Returns overall, layout, and per-asset latents with part_layouts.
    """
    sample_dir = os.path.join(data_dir, scene_id, room_id)
    room_name = room_id

    # Shape latent paths
    shape_dir = os.path.join(sample_dir, 'shape_latents', shape_encoder)
    shape_overall_path = os.path.join(shape_dir, 'full_room_wo_ceiling.npz')
    shape_assets_dir = os.path.join(shape_dir, 'individual_assets_room_coord')

    if not os.path.exists(shape_overall_path):
        return None

    shape_overall_coords, shape_overall_feats = load_latent(shape_overall_path)

    # Texture latent paths (optional)
    has_texture = tex_encoder is not None
    if has_texture:
        tex_dir = os.path.join(sample_dir, 'pbr_latents', tex_encoder)
        tex_overall_path = os.path.join(tex_dir, 'full_room_wo_ceiling.npz')
        tex_assets_dir = os.path.join(tex_dir, 'individual_assets_room_coord')
        if not os.path.exists(tex_overall_path):
            has_texture = False
        else:
            tex_overall_coords, tex_overall_feats = load_latent(tex_overall_path)

    # Load bbox and camera center for visibility filtering
    bbox_data = load_gt_bboxes(data_dir, scene_id, room_id)
    camera_center = load_camera_center(data_dir, scene_id, room_id, view_idx=view_idx)

    if bbox_data is None:
        return None

    obbs_tensor = torch.from_numpy(bbox_data['obbs']).float()
    visible_indices, _ = filter_visible_assets(
        obbs_tensor, camera_center,
        visibility_threshold=visibility_threshold,
        fov_degrees=120.0,
        image_size=512,
    )

    # Align bbox with latent files using asset_filenames (which have _inst patterns)
    if os.path.isdir(shape_assets_dir):
        asset_latent_files = sorted(os.listdir(shape_assets_dir))
    else:
        asset_latent_files = []

    aligned_assets = align_bbox_with_latents(
        bbox_data['asset_filenames'], asset_latent_files
    )
    visible_set = set(visible_indices)
    visible_aligned = [(bbox_idx, lf) for bbox_idx, lf in aligned_assets
                       if bbox_idx in visible_set]

    # Build combined latent
    all_shape_coords = []
    all_shape_feats = []
    all_tex_coords = [] if has_texture else None
    all_tex_feats = [] if has_texture else None
    part_layouts = []
    matched_obbs = []
    matched_names = []
    start_idx = 0

    # Overall (part_layouts[0])
    all_shape_coords.append(torch.from_numpy(shape_overall_coords))
    all_shape_feats.append(torch.from_numpy(shape_overall_feats))
    if has_texture:
        all_tex_coords.append(torch.from_numpy(tex_overall_coords))
        all_tex_feats.append(torch.from_numpy(tex_overall_feats))
    part_layouts.append(slice(start_idx, start_idx + shape_overall_coords.shape[0]))
    start_idx += shape_overall_coords.shape[0]

    # Layout (part_layouts[1])
    layout_path = os.path.join(shape_dir, 'layout_wo_ceiling.npz')
    has_layout = os.path.exists(layout_path)
    if has_layout:
        layout_coords, layout_feats = load_latent(layout_path)
        all_shape_coords.append(torch.from_numpy(layout_coords))
        all_shape_feats.append(torch.from_numpy(layout_feats))
        if has_texture:
            tex_layout_path = os.path.join(tex_dir, 'layout_wo_ceiling.npz')
            if os.path.exists(tex_layout_path):
                tex_layout_coords, tex_layout_feats = load_latent(tex_layout_path)
                all_tex_coords.append(torch.from_numpy(tex_layout_coords))
                all_tex_feats.append(torch.from_numpy(tex_layout_feats))
            else:
                has_texture = False
        part_layouts.append(slice(start_idx, start_idx + layout_coords.shape[0]))
        start_idx += layout_coords.shape[0]
    else:
        # Empty layout slot
        part_layouts.append(slice(start_idx, start_idx))

    # Per-asset (part_layouts[2+])
    for bbox_idx, latent_file in visible_aligned:
        shape_asset_path = os.path.join(shape_assets_dir, latent_file)
        asset_coords, asset_feats = load_latent(shape_asset_path)

        all_shape_coords.append(torch.from_numpy(asset_coords))
        all_shape_feats.append(torch.from_numpy(asset_feats))

        if has_texture:
            tex_asset_path = os.path.join(tex_assets_dir, latent_file)
            if os.path.exists(tex_asset_path):
                tex_asset_coords, tex_asset_feats = load_latent(tex_asset_path)
                all_tex_coords.append(torch.from_numpy(tex_asset_coords))
                all_tex_feats.append(torch.from_numpy(tex_asset_feats))
            else:
                # Fallback: zeros matching shape coords
                all_tex_coords.append(torch.from_numpy(asset_coords))
                all_tex_feats.append(torch.zeros(asset_coords.shape[0], tex_overall_feats.shape[1]))

        part_layouts.append(slice(start_idx, start_idx + asset_coords.shape[0]))
        start_idx += asset_coords.shape[0]

        matched_obbs.append(bbox_data['obbs'][bbox_idx])
        matched_names.append(bbox_data['asset_filenames'][bbox_idx])

    # Combine
    shape_coords = torch.cat(all_shape_coords, dim=0).int()
    shape_feats = torch.cat(all_shape_feats, dim=0).float()

    result = {
        'shape_coords': shape_coords,
        'shape_feats': shape_feats,
        'part_layouts': part_layouts,
        'has_layout': has_layout,
        'obbs': np.array(matched_obbs) if matched_obbs else np.zeros((0, 7)),
        'asset_names': matched_names,
        'camera_center': camera_center,
    }

    if has_texture:
        tex_coords = torch.cat(all_tex_coords, dim=0).int()
        tex_feats = torch.cat(all_tex_feats, dim=0).float()
        result['tex_coords'] = tex_coords
        result['tex_feats'] = tex_feats

    return result


# ============================================================
# Decode & Save
# ============================================================

@torch.no_grad()
def run_gt_recon_batch(args, samples, device='cuda'):
    """Load GT latents and decode to GLB meshes."""
    print("\n" + "=" * 60)
    print("GT Reconstruction: Load GT Latents → Decode → GLB")
    print("=" * 60)

    # GT latents are raw VAE encoder outputs — no generation-model normalization needed.
    # We only need the pretrained VAE decoders.
    shape_encoder = args.shape_encoder
    data_resolution = args.resolution

    print(f"  Shape encoder: {shape_encoder}")
    print(f"  Resolution: {data_resolution}")
    print(f"  Loading shape decoder from: {args.shape_dec}")
    shape_dec = models.from_pretrained(args.shape_dec)
    shape_dec.set_resolution(data_resolution)
    shape_dec = shape_dec.to(device).eval()

    # Texture setup
    tex_encoder = None
    pbr_dec = None
    tex_shape_dec = None
    tex_layout = None

    if args.enable_texture:
        tex_encoder = args.tex_encoder
        tex_attrs = ['base_color', 'metallic', 'roughness', 'alpha']

        channels = {'base_color': 3, 'metallic': 1, 'roughness': 1, 'emissive': 3, 'alpha': 1}
        tex_layout = {}
        start = 0
        for attr in tex_attrs:
            tex_layout[attr] = slice(start, start + channels[attr])
            start += channels[attr]

        print(f"  Texture encoder: {tex_encoder}")
        print(f"  Loading texture shape decoder from: {args.tex_shape_dec}")
        tex_shape_dec = models.from_pretrained(args.tex_shape_dec)
        tex_shape_dec.set_resolution(data_resolution)
        tex_shape_dec = tex_shape_dec.to(device).eval()

        print(f"  Loading PBR decoder from: {args.pbr_dec}")
        pbr_dec = models.from_pretrained(args.pbr_dec)
        pbr_dec = pbr_dec.to(device).eval()

    num_decoded = 0
    num_skipped = 0

    for sample in tqdm(samples, desc="GT Recon"):
        folder_name = sample['folder_name']
        out_dir = os.path.join(args.output_dir, folder_name)
        scene_glb_path = os.path.join(out_dir, 'scene.glb')

        if args.skip_existing and os.path.exists(scene_glb_path):
            num_decoded += 1
            continue

        try:
            gt_data = load_gt_latents(
                args.data_dir, sample['scene_id'], sample['room_id'],
                shape_encoder=shape_encoder,
                tex_encoder=tex_encoder,
                visibility_threshold=0.5,
                view_idx=sample['view_idx'],
                layout_mode=args.layout_mode,
            )

            if gt_data is None:
                print(f"  Skipping {folder_name}: GT latents not found")
                num_skipped += 1
                continue

            os.makedirs(out_dir, exist_ok=True)

            shape_coords = gt_data['shape_coords']
            shape_feats = gt_data['shape_feats']
            part_layouts = gt_data['part_layouts']
            has_texture = args.enable_texture and 'tex_feats' in gt_data

            if has_texture:
                tex_coords = gt_data['tex_coords']
                tex_feats = gt_data['tex_feats']

            # Helper: add batch index column (0) to coords [N,3] -> [N,4]
            def _add_batch_idx(coords):
                batch_idx = torch.zeros(coords.shape[0], 1, dtype=coords.dtype, device=coords.device)
                return torch.cat([batch_idx, coords], dim=1)

            # Helper: decode a part with texture
            def decode_part_textured(part_slice, decimation_target=500000):
                import o_voxel

                p_shape = SparseTensor(
                    coords=_add_batch_idx(shape_coords[part_slice].to(device)),
                    feats=shape_feats[part_slice].to(device),
                )
                p_tex = SparseTensor(
                    coords=_add_batch_idx(tex_coords[part_slice].to(device)),
                    feats=tex_feats[part_slice].to(device),
                )

                # GT latents are raw encoder outputs — no normalization needed
                mesh, subs = tex_shape_dec(p_shape, return_subs=True)
                vox = pbr_dec(p_tex, guide_subs=subs) * 0.5 + 0.5

                if mesh and len(mesh) > 0:
                    mesh[0].fill_holes()
                    try:
                        glb = o_voxel.postprocess.to_glb(
                            vertices=mesh[0].vertices,
                            faces=mesh[0].faces,
                            attr_volume=vox[0].feats,
                            coords=vox[0].coords[:, 1:],
                            attr_layout=tex_layout,
                            grid_size=data_resolution,
                            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                            decimation_target=decimation_target,
                            texture_size=2048,
                            remesh=True,
                            remesh_band=1,
                            remesh_project=0,
                            verbose=False,
                        )
                        return glb
                    except Exception as e:
                        print(f"    Warning: textured GLB failed: {e}, using vertex colors")
                        return mesh[0]
                return None

            # Helper: decode a part shape-only
            def decode_part_shape(part_slice, decimation_target=500000):
                p_z = SparseTensor(
                    coords=_add_batch_idx(shape_coords[part_slice].to(device)),
                    feats=shape_feats[part_slice].to(device),
                )
                # GT latents are raw encoder outputs — no normalization needed
                reps = shape_dec(p_z)
                if reps and len(reps) > 0:
                    return reps[0]
                return None

            # --- Scene (overall, part_layouts[0]) ---
            overall_slice = part_layouts[0]
            if has_texture:
                result = decode_part_textured(overall_slice, decimation_target=500000)
                if result is not None:
                    if hasattr(result, 'export'):
                        result.export(scene_glb_path)
                    else:
                        save_mesh_glb(result.vertices, result.faces, scene_glb_path)
            else:
                result = decode_part_shape(overall_slice)
                if result is not None:
                    save_mesh_glb(result.vertices, result.faces, scene_glb_path)

            # --- Layout (part_layouts[1]) ---
            layout_slice = part_layouts[1]
            n_layout = layout_slice.stop - layout_slice.start
            if n_layout > 0:
                layout_glb_path = os.path.join(out_dir, 'layout.glb')
                try:
                    if has_texture:
                        result = decode_part_textured(layout_slice, decimation_target=200000)
                        if result is not None:
                            if hasattr(result, 'export'):
                                result.export(layout_glb_path)
                            else:
                                save_mesh_glb(result.vertices, result.faces, layout_glb_path)
                    else:
                        result = decode_part_shape(layout_slice)
                        if result is not None:
                            save_mesh_glb(result.vertices, result.faces, layout_glb_path)
                except Exception as e:
                    print(f"    Warning: layout decode failed for {folder_name}: {e}")

            # --- Individual assets (part_layouts[2+]) ---
            assets_dir = os.path.join(out_dir, 'assets')
            os.makedirs(assets_dir, exist_ok=True)
            asset_start = 2
            for part_idx in range(asset_start, len(part_layouts)):
                part_slice = part_layouts[part_idx]
                n_voxels = part_slice.stop - part_slice.start
                if n_voxels < 10:
                    continue

                asset_idx = part_idx - asset_start
                if asset_idx < len(gt_data['asset_names']):
                    asset_name = gt_data['asset_names'][asset_idx]
                else:
                    asset_name = f'asset_{asset_idx:03d}'

                try:
                    safe_name = asset_name.replace('/', '_').replace(' ', '_')
                    asset_glb_path = os.path.join(assets_dir, f'{asset_idx:03d}_{safe_name}.glb')

                    if has_texture:
                        result = decode_part_textured(part_slice, decimation_target=100000)
                        if result is not None:
                            if hasattr(result, 'export'):
                                result.export(asset_glb_path)
                            else:
                                save_mesh_glb(result.vertices, result.faces, asset_glb_path)
                    else:
                        result = decode_part_shape(part_slice)
                        if result is not None:
                            save_mesh_glb(result.vertices, result.faces, asset_glb_path)
                except Exception as e:
                    print(f"    Error decoding asset {asset_idx} ({asset_name}): {e}")

            # --- Merge assets into assets.glb ---
            try:
                import trimesh
                asset_glb_files = sorted(glob.glob(os.path.join(assets_dir, '*.glb')))
                if asset_glb_files:
                    combined = trimesh.Scene()
                    for glb_file in asset_glb_files:
                        asset_mesh = trimesh.load(glb_file, force='mesh')
                        name = os.path.splitext(os.path.basename(glb_file))[0]
                        combined.add_geometry(asset_mesh, node_name=name)
                    combined.export(os.path.join(out_dir, 'assets.glb'))
            except Exception as e:
                print(f"    Warning: assets.glb merge failed for {folder_name}: {e}")

            # Copy perspective image
            if os.path.exists(sample['perspective_image']):
                shutil.copy2(sample['perspective_image'],
                             os.path.join(out_dir, 'input_perspective.png'))

            # Save metadata
            metadata = {
                'scene_id': sample['scene_id'],
                'room_id': sample['room_id'],
                'view_idx': sample['view_idx'],
                'idx': sample['idx'],
                'folder_name': folder_name,
                'mode': 'gt_recon',
                'texture_enabled': has_texture,
                'n_assets': len(gt_data['asset_names']),
                'asset_names': gt_data['asset_names'],
                'has_layout': gt_data['has_layout'],
                'erp_image': sample['erp_image'],
                'perspective_image': sample['perspective_image'],
            }
            with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            num_decoded += 1

        except Exception as e:
            print(f"  Error: {folder_name}: {e}")
            import traceback
            traceback.print_exc()

    del shape_dec
    if pbr_dec is not None:
        del pbr_dec, tex_shape_dec
    torch.cuda.empty_cache()

    print(f"\n  Decoded: {num_decoded}/{len(samples)}")
    if num_skipped > 0:
        print(f"  Skipped (no GT): {num_skipped}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='GT Reconstruction for InSpace evaluation samples')

    # Paths
    parser.add_argument('--eval_json', type=str,
                        default='evals/perspective_eval_dataset_selected.json')
    parser.add_argument('--data_dir', type=str,
                        default='datasets/ERP_3D_FRONT_test')
    parser.add_argument('--output_dir', type=str,
                        default='evals/output_InSpace_batch_gt_recon')
    parser.add_argument('--gpu_id', type=int, default=0)

    # VAE decoder paths (pretrained, no generation model needed)
    parser.add_argument('--shape_encoder', type=str,
                        default='shape_enc_next_dc_f16c32_fp16_512',
                        help='Shape latent encoder name (dataset folder)')
    parser.add_argument('--tex_encoder', type=str,
                        default='tex_enc_next_dc_f16c32_fp16_512',
                        help='Texture latent encoder name (dataset folder)')
    parser.add_argument('--resolution', type=int, default=512,
                        help='O-voxel resolution for decoder')
    parser.add_argument('--shape_dec', type=str,
                        default='microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
                        help='Pretrained shape VAE decoder')
    parser.add_argument('--tex_shape_dec', type=str,
                        default='microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
                        help='Pretrained shape decoder for textured mesh')
    parser.add_argument('--pbr_dec', type=str,
                        default='microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
                        help='Pretrained PBR texture VAE decoder')
    parser.add_argument('--enable_texture', action='store_true', default=True)
    parser.add_argument('--no_texture', dest='enable_texture', action='store_false')
    parser.add_argument('--layout_mode', type=str, default='floor_perimeter_clean',
                        choices=['floor_perimeter', 'floor_perimeter_clean', 'no_floor_assets'])

    # Common
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false')

    # Distributed
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = f'cuda:{args.gpu_id}'

    # Load samples
    all_samples = load_eval_samples(args.eval_json)
    print(f"\nGT Reconstruction Batch")
    print(f"  JSON: {args.eval_json}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Output: {args.output_dir}")
    print(f"  Texture: {args.enable_texture}")
    print(f"  Device: {device}")

    if args.max_samples > 0:
        all_samples = all_samples[:args.max_samples]
        print(f"  Limited to {len(all_samples)} samples")

    # Distributed split
    if args.world_size > 1:
        assert 0 <= args.rank < args.world_size
        samples = all_samples[args.rank::args.world_size]
        print(f"  Worker {args.rank}/{args.world_size}: {len(samples)} samples")
    else:
        samples = all_samples

    # Save config
    eval_config = vars(args).copy()
    eval_config['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    eval_config['n_samples'] = len(samples)
    with open(os.path.join(args.output_dir, f'eval_config_rank{args.rank}.json'), 'w') as f:
        json.dump(eval_config, f, indent=2)

    # Run
    run_gt_recon_batch(args, samples, device)

    print("\n" + "=" * 60)
    print("GT Reconstruction Complete!")
    print("=" * 60)
    print(f"  Output: {args.output_dir}")


if __name__ == '__main__':
    main()
