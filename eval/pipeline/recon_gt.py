# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Reconstruct GT meshes from latent space.

Decodes GT shape/texture latents (NPZ) → GLB meshes for all parts:
  - scene.glb (full_room_wo_ceiling)
  - layout.glb (layout_wo_ceiling)
  - assets/*.glb (individual_assets_room_coord/*)

These reconstructed GT meshes share the same [-0.5, 0.5] coordinate space
as predictions from eval_pipeline.py, enabling direct comparison without ICP.

Usage:
    python eval/pipeline/recon_gt.py \
        --data_dir datasets/ERP_3D_FRONT_test \
        --output_dir evals/gt_recon \
        --enable_texture
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import trimesh
from tqdm import tqdm
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from trellis2 import models
from trellis2.modules.sparse import SparseTensor


def save_mesh_glb(vertices, faces, output_path):
    """Save mesh as GLB with position-based vertex colors."""
    if vertices is None or faces is None:
        return
    v = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
    f = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces
    colors = ((v + 0.5) * 255).clip(0, 255).astype(np.uint8)
    colors = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])
    mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors)
    mesh.export(output_path)


def decode_and_save_glb(
    shape_npz_path: str,
    tex_npz_path: Optional[str],
    output_path: str,
    shape_dec,
    tex_shape_dec,
    pbr_dec,
    tex_layout: Optional[dict],
    resolution: int,
    device: str,
    decimation_target: int = 100000,
    texture_size: int = 2048,
):
    """Decode latent NPZ → GLB mesh."""
    if not os.path.exists(shape_npz_path):
        return False

    data = np.load(shape_npz_path)
    coords = torch.from_numpy(data['coords']).int()
    feats = torch.from_numpy(data['feats']).float()
    batch_idx = torch.zeros(coords.shape[0], 1, dtype=torch.int32)
    coords_4d = torch.cat([batch_idx, coords], dim=1)

    # GT NPZ features are in original space — no normalization needed
    shape_z = SparseTensor(coords=coords_4d, feats=feats).to(device)

    if tex_npz_path and os.path.exists(tex_npz_path) and tex_shape_dec is not None and pbr_dec is not None:
        # Texture mode: dual decode (shape → mesh+subs, PBR → voxel attrs)
        tex_data = np.load(tex_npz_path)
        tex_feats = torch.from_numpy(tex_data['feats']).float()
        tex_z = SparseTensor(coords=coords_4d.clone(), feats=tex_feats).to(device)

        # Shape for texture decoder (same latent, separate SparseTensor)
        shape_for_tex = SparseTensor(
            coords=coords_4d.clone(),
            feats=torch.from_numpy(data['feats']).float(),
        ).to(device)

        mesh_list, subs = tex_shape_dec(shape_for_tex, return_subs=True)
        vox = pbr_dec(tex_z, guide_subs=subs) * 0.5 + 0.5

        if mesh_list and len(mesh_list) > 0:
            mesh_list[0].fill_holes()
            try:
                import o_voxel
                glb_mesh = o_voxel.postprocess.to_glb(
                    vertices=mesh_list[0].vertices,
                    faces=mesh_list[0].faces,
                    attr_volume=vox[0].feats,
                    coords=vox[0].coords[:, 1:],
                    attr_layout=tex_layout,
                    grid_size=resolution,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=decimation_target,
                    texture_size=texture_size,
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    verbose=False,
                )
                glb_mesh.export(output_path)
                return True
            except Exception as e:
                print(f"    Warning: textured GLB failed ({e}), fallback to vertex colors")
                save_mesh_glb(mesh_list[0].vertices, mesh_list[0].faces, output_path)
                return True
    else:
        # Shape-only mode
        reps = shape_dec(shape_z)
        if reps and len(reps) > 0:
            save_mesh_glb(reps[0].vertices, reps[0].faces, output_path)
            return True

    return False


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Reconstruct GT meshes from latent NPZ files')
    parser.add_argument('--data_dir', type=str, default='datasets/ERP_3D_FRONT_test')
    parser.add_argument('--output_dir', type=str, default='evals/gt_recon')
    parser.add_argument('--enable_texture', action='store_true', help='Decode with texture (PBR)')
    parser.add_argument('--stage2_shape_config', type=str,
                        default='configs/gen/erp_slat_flow_img2shape_asset_aware_bf16.json')
    parser.add_argument('--stage2_tex_config', type=str,
                        default='configs/gen/erp_slat_flow_imgshape2tex_asset_aware_bf16.json')
    parser.add_argument('--max_samples', type=int, default=-1, help='-1 for all')
    parser.add_argument('--max_meshes', type=int, default=-1,
                        help='Max samples to decode GLB meshes for (-1 for all)')
    parser.add_argument('--skip_existing', action='store_true', help='Skip already decoded samples')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0,
                        help='Worker rank for distributed eval (0-indexed)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of workers for distributed eval')
    args = parser.parse_args()

    # GT
    # python eval/pipeline/recon_gt.py --rank 0 --world_size 2 --gpu_id 0
    # python eval/pipeline/recon_gt.py --rank 1 --world_size 2 --gpu_id 1
    
    device = f'cuda:{args.gpu_id}'
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover samples ──
    samples = []
    for scene_id in sorted(os.listdir(args.data_dir)):
        scene_dir = os.path.join(args.data_dir, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        for room_id in sorted(os.listdir(scene_dir)):
            room_dir = os.path.join(scene_dir, room_id)
            if not os.path.isdir(room_dir):
                continue
            # Check that shape latent exists
            shape_latent_path = os.path.join(
                room_dir, 'shape_latents', 'shape_enc_next_dc_f16c32_fp16_512',
                'full_room_wo_ceiling.npz')
            if os.path.exists(shape_latent_path):
                samples.append((scene_id, room_id))

    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    if args.max_meshes >= 0:
        samples = samples[:args.max_meshes]

    # Split samples across workers
    total = len(samples)
    if args.world_size > 1:
        assert 0 <= args.rank < args.world_size, \
            f"rank must be in [0, {args.world_size}), got {args.rank}"
        samples = samples[args.rank::args.world_size]

    print(f"Found {total} total samples, processing {len(samples)} in {args.data_dir}")
    if args.world_size > 1:
        print(f"Distributed: rank {args.rank}/{args.world_size}")
    print(f"Output: {args.output_dir}")
    print(f"Texture: {args.enable_texture}")

    # ── Load decoders ──
    with open(args.stage2_shape_config, 'r') as f:
        shape_config = json.load(f)
    data_resolution = shape_config['dataset']['args'].get('resolution', 512)

    pretrained_slat_dec = shape_config['dataset']['args'].get(
        'pretrained_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
    print(f"Loading shape decoder: {pretrained_slat_dec}")
    shape_dec = models.from_pretrained(pretrained_slat_dec)
    shape_dec.set_resolution(data_resolution)
    shape_dec = shape_dec.to(device).eval()

    tex_shape_dec = None
    pbr_dec = None
    tex_layout = None

    if args.enable_texture:
        with open(args.stage2_tex_config, 'r') as f:
            tex_config = json.load(f)
        tex_attrs = tex_config['dataset']['args'].get(
            'attrs', ['base_color', 'metallic', 'roughness', 'alpha'])
        channels = {'base_color': 3, 'metallic': 1, 'roughness': 1, 'emissive': 3, 'alpha': 1}
        tex_layout = {}
        start = 0
        for attr in tex_attrs:
            tex_layout[attr] = slice(start, start + channels[attr])
            start += channels[attr]

        pretrained_pbr_dec = tex_config['dataset']['args'].get(
            'pretrained_pbr_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16')
        pretrained_shape_slat_dec = tex_config['dataset']['args'].get(
            'pretrained_shape_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')

        print(f"Loading texture shape decoder: {pretrained_shape_slat_dec}")
        tex_shape_dec = models.from_pretrained(pretrained_shape_slat_dec)
        tex_shape_dec.set_resolution(data_resolution)
        tex_shape_dec = tex_shape_dec.to(device).eval()

        print(f"Loading PBR decoder: {pretrained_pbr_dec}")
        pbr_dec = models.from_pretrained(pretrained_pbr_dec)
        pbr_dec = pbr_dec.to(device).eval()

    # ── Decode all samples ──
    shape_latent_folder = 'shape_latents/shape_enc_next_dc_f16c32_fp16_512'
    tex_latent_folder = 'pbr_latents/tex_enc_next_dc_f16c32_fp16_512'

    num_decoded = 0
    num_failed = 0

    for scene_id, room_id in tqdm(samples, desc="Reconstructing GT meshes"):
        room_dir = os.path.join(args.data_dir, scene_id, room_id)
        out_dir = os.path.join(args.output_dir, scene_id, room_id, 'meshes')

        scene_glb_path = os.path.join(out_dir, 'scene.glb')
        if args.skip_existing and os.path.exists(scene_glb_path):
            num_decoded += 1
            continue

        os.makedirs(out_dir, exist_ok=True)
        assets_dir = os.path.join(out_dir, 'assets')
        os.makedirs(assets_dir, exist_ok=True)

        shape_dir = os.path.join(room_dir, shape_latent_folder)
        tex_dir = os.path.join(room_dir, tex_latent_folder) if args.enable_texture else None

        try:
            # ── 1. Scene (full_room_wo_ceiling) ──
            shape_path = os.path.join(shape_dir, 'full_room_wo_ceiling.npz')
            tex_path = os.path.join(tex_dir, 'full_room_wo_ceiling.npz') if tex_dir else None
            decode_and_save_glb(
                shape_path, tex_path, scene_glb_path,
                shape_dec, tex_shape_dec, pbr_dec, tex_layout,
                data_resolution, device,
                decimation_target=100000, texture_size=2048,
            )

            # ── 2. Layout (layout_wo_ceiling) ──
            layout_shape_path = os.path.join(shape_dir, 'layout_wo_ceiling.npz')
            layout_tex_path = os.path.join(tex_dir, 'layout_wo_ceiling.npz') if tex_dir else None
            layout_glb_path = os.path.join(out_dir, 'layout.glb')
            if os.path.exists(layout_shape_path):
                decode_and_save_glb(
                    layout_shape_path, layout_tex_path, layout_glb_path,
                    shape_dec, tex_shape_dec, pbr_dec, tex_layout,
                    data_resolution, device,
                    decimation_target=50000, texture_size=2048,
                )

            # ── 3. Individual assets ──
            asset_shape_dir = os.path.join(shape_dir, 'individual_assets_room_coord')
            asset_tex_dir = os.path.join(tex_dir, 'individual_assets_room_coord') if tex_dir else None

            if os.path.isdir(asset_shape_dir):
                asset_files = sorted([f for f in os.listdir(asset_shape_dir) if f.endswith('.npz')])
                for ai, asset_file in enumerate(asset_files):
                    asset_name = asset_file.replace('.npz', '')
                    safe_name = asset_name.replace('/', '_').replace(' ', '_')
                    asset_glb_path = os.path.join(assets_dir, f'{ai:03d}_{safe_name}.glb')

                    asset_shape_path = os.path.join(asset_shape_dir, asset_file)
                    asset_tex_path = os.path.join(asset_tex_dir, asset_file) if asset_tex_dir else None

                    try:
                        decode_and_save_glb(
                            asset_shape_path, asset_tex_path, asset_glb_path,
                            shape_dec, tex_shape_dec, pbr_dec, tex_layout,
                            data_resolution, device,
                            decimation_target=10000, texture_size=1024,
                        )
                    except Exception as e:
                        print(f"    Warning: asset {asset_name} failed: {e}")

            num_decoded += 1

        except Exception as e:
            print(f"  Error: {scene_id}/{room_id}: {e}")
            import traceback
            traceback.print_exc()
            num_failed += 1

    print(f"\nDone: {num_decoded} decoded, {num_failed} failed out of {len(samples)} samples")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
