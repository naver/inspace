# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Standalone GLB decoder for prediction results.

Reads shape_latent.npz + texture_latent.npz + bboxes.npz from eval output dirs
and decodes to GLB files. Runs independently of eval_pipeline.py.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/pipeline/decode_glb.py \
        --pred_dir evals/stage12_pipeline/random_gt \
        --enable_texture \
        --max_meshes 200 \
        --skip_existing \
        --gpu_id 0
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


def inverse_normalize(sparse_tensor, normalization):
    """Inverse normalize latent features."""
    mean = torch.tensor(normalization['mean']).reshape(1, -1).to(sparse_tensor.feats.device)
    std = torch.tensor(normalization['std']).reshape(1, -1).to(sparse_tensor.feats.device)
    return sparse_tensor.replace(feats=sparse_tensor.feats * std + mean)


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


def decode_textured_glb(shape_st, tex_st, tex_shape_dec, pbr_dec, tex_layout,
                        shape_resolution, tex_shape_norm, tex_norm,
                        output_path, decimation_target, device):
    """Decode shape+texture latents to textured GLB."""
    shape_z = shape_st.to(device)
    tex_z = tex_st.to(device)

    if tex_shape_norm:
        shape_mean = torch.tensor(tex_shape_norm['mean']).reshape(1, -1).to(device)
        shape_std = torch.tensor(tex_shape_norm['std']).reshape(1, -1).to(device)
        shape_z = shape_z.replace(feats=shape_z.feats * shape_std + shape_mean)
    if tex_norm:
        tex_z = inverse_normalize(tex_z, tex_norm)

    mesh, subs = tex_shape_dec(shape_z, return_subs=True)
    vox = pbr_dec(tex_z, guide_subs=subs) * 0.5 + 0.5

    if mesh and len(mesh) > 0:
        mesh[0].fill_holes()
        try:
            import o_voxel
            glb_mesh = o_voxel.postprocess.to_glb(
                vertices=mesh[0].vertices,
                faces=mesh[0].faces,
                attr_volume=vox[0].feats,
                coords=vox[0].coords[:, 1:],
                attr_layout=tex_layout,
                grid_size=shape_resolution,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=decimation_target,
                texture_size=2048,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
            glb_mesh.export(output_path)
            return True
        except Exception as e:
            print(f"    Warning: textured GLB failed ({e}), fallback to vertex colors")
            save_mesh_glb(mesh[0].vertices, mesh[0].faces, output_path)
            return True
    return False


def decode_shape_glb(shape_st, shape_dec, shape_norm, output_path, device):
    """Decode shape-only latent to GLB with vertex colors."""
    shape_z = shape_st.to(device)
    if shape_norm:
        shape_z = inverse_normalize(shape_z, shape_norm)
    reps = shape_dec(shape_z)
    if reps and len(reps) > 0:
        save_mesh_glb(reps[0].vertices, reps[0].faces, output_path)
        return True
    return False


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Decode prediction NPZ to GLB (standalone)')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Prediction output dir (e.g., evals/stage12_pipeline/random_gt)')
    parser.add_argument('--enable_texture', action='store_true', default=False)
    parser.add_argument('--max_meshes', type=int, default=200)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false')
    parser.add_argument('--stage2_shape_config', type=str,
                        default='configs/gen/erp_slat_flow_img2shape_asset_aware_bf16.json')
    parser.add_argument('--stage2_tex_config', type=str,
                        default='configs/gen/erp_slat_flow_imgshape2tex_asset_aware_bf16.json')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.gpu_id}'

    # ── Discover ALL samples with shape_latent.npz ──
    # texture_latent.npz is checked per-sample at runtime (may be generated in parallel)
    samples = []
    for scene_id in sorted(os.listdir(args.pred_dir)):
        scene_dir = os.path.join(args.pred_dir, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        for room_id in sorted(os.listdir(scene_dir)):
            room_dir = os.path.join(scene_dir, room_id)
            if not os.path.isdir(room_dir):
                continue
            shape_path = os.path.join(room_dir, 'shape_latent.npz')
            if os.path.exists(shape_path):
                samples.append((scene_id, room_id))

    if args.max_meshes >= 0:
        samples = samples[:args.max_meshes]

    print(f"Found {len(samples)} samples (shape_latent.npz) in {args.pred_dir}")
    print(f"Texture: {args.enable_texture} (checked per-sample at runtime)")

    # ── Load decoders ──
    with open(args.stage2_shape_config, 'r') as f:
        shape_config = json.load(f)
    shape_normalization = shape_config['dataset']['args'].get('normalization', None)
    shape_resolution = shape_config['dataset']['args'].get('resolution', 512)

    pretrained_slat_dec = shape_config['dataset']['args'].get(
        'pretrained_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
    print(f"Loading shape decoder: {pretrained_slat_dec}")
    shape_dec = models.from_pretrained(pretrained_slat_dec)
    shape_dec.set_resolution(shape_resolution)
    shape_dec = shape_dec.to(device).eval()

    tex_shape_dec = None
    pbr_dec = None
    tex_layout = None
    tex_normalization = None
    tex_shape_normalization = None

    if args.enable_texture:
        with open(args.stage2_tex_config, 'r') as f:
            tex_config = json.load(f)
        tex_normalization = tex_config['dataset']['args'].get('normalization', None)
        tex_shape_normalization = tex_config['dataset']['args'].get('shape_normalization', None)
        tex_attrs = tex_config['dataset']['args'].get('attrs', ['base_color', 'metallic', 'roughness', 'alpha'])
        channels = {'base_color': 3, 'metallic': 1, 'roughness': 1, 'emissive': 3, 'alpha': 1}
        tex_layout = {}
        start = 0
        for attr in tex_attrs:
            tex_layout[attr] = slice(start, start + channels[attr])
            start += channels[attr]

        pretrained_shape_slat_dec = tex_config['dataset']['args'].get(
            'pretrained_shape_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
        pretrained_pbr_dec = tex_config['dataset']['args'].get(
            'pretrained_pbr_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16')

        print(f"Loading texture shape decoder: {pretrained_shape_slat_dec}")
        tex_shape_dec = models.from_pretrained(pretrained_shape_slat_dec)
        tex_shape_dec.set_resolution(shape_resolution)
        tex_shape_dec = tex_shape_dec.to(device).eval()

        print(f"Loading PBR decoder: {pretrained_pbr_dec}")
        pbr_dec = models.from_pretrained(pretrained_pbr_dec)
        pbr_dec = pbr_dec.to(device).eval()

    # ── Decode ──
    num_decoded = 0
    num_skipped = 0
    num_not_ready = 0
    num_failed = 0

    for scene_id, room_id in tqdm(samples, desc="Decoding GLB"):
        room_dir = os.path.join(args.pred_dir, scene_id, room_id)
        out_dir = os.path.join(room_dir, 'meshes')
        scene_glb_path = os.path.join(out_dir, 'scene.glb')

        if args.skip_existing and os.path.exists(scene_glb_path):
            num_skipped += 1
            continue

        # Check texture_latent.npz at runtime (may be generated in parallel)
        if args.enable_texture:
            tex_path = os.path.join(room_dir, 'texture_latent.npz')
            if not os.path.exists(tex_path):
                num_not_ready += 1
                continue

        # Load shape latent
        shape_data = np.load(os.path.join(room_dir, 'shape_latent.npz'), allow_pickle=True)
        shape_coords = torch.from_numpy(shape_data['coords']).int()
        shape_feats = torch.from_numpy(shape_data['feats']).float()
        part_layouts_raw = shape_data['part_layouts']
        part_layouts = [slice(int(s[0]), int(s[1])) for s in part_layouts_raw]

        # Load texture latent
        tex_coords = None
        tex_feats = None
        use_texture = False
        if args.enable_texture:
            tex_data = np.load(tex_path, allow_pickle=True)
            tex_coords = torch.from_numpy(tex_data['coords']).int()
            tex_feats = torch.from_numpy(tex_data['feats']).float()
            use_texture = True

        # Load bbox data for asset names
        bbox_path = os.path.join(room_dir, 'bboxes.npz')
        asset_names = []
        if os.path.exists(bbox_path):
            bbox_data = np.load(bbox_path, allow_pickle=True)
            if 'asset_names' in bbox_data:
                asset_names = list(bbox_data['asset_names'])

        os.makedirs(out_dir, exist_ok=True)
        assets_dir = os.path.join(out_dir, 'assets')
        os.makedirs(assets_dir, exist_ok=True)

        try:
            # ── 1. Scene (overall, part_layouts[0]) ──
            overall_slice = part_layouts[0]
            overall_shape_st = SparseTensor(
                coords=shape_coords[overall_slice],
                feats=shape_feats[overall_slice],
            )

            if use_texture:
                overall_tex_st = SparseTensor(
                    coords=tex_coords[overall_slice],
                    feats=tex_feats[overall_slice],
                )
                decode_textured_glb(
                    overall_shape_st, overall_tex_st,
                    tex_shape_dec, pbr_dec, tex_layout,
                    shape_resolution, tex_shape_normalization, tex_normalization,
                    scene_glb_path, decimation_target=500000, device=device,
                )
            else:
                decode_shape_glb(
                    overall_shape_st, shape_dec, shape_normalization,
                    scene_glb_path, device=device,
                )

            # ── 2. Layout (part_layouts[1]) ──
            layout_slice = part_layouts[1]
            n_layout = layout_slice.stop - layout_slice.start
            if n_layout > 0:
                layout_glb_path = os.path.join(out_dir, 'layout.glb')
                layout_shape_st = SparseTensor(
                    coords=shape_coords[layout_slice],
                    feats=shape_feats[layout_slice],
                )
                try:
                    if use_texture:
                        layout_tex_st = SparseTensor(
                            coords=tex_coords[layout_slice],
                            feats=tex_feats[layout_slice],
                        )
                        decode_textured_glb(
                            layout_shape_st, layout_tex_st,
                            tex_shape_dec, pbr_dec, tex_layout,
                            shape_resolution, tex_shape_normalization, tex_normalization,
                            layout_glb_path, decimation_target=200000, device=device,
                        )
                    else:
                        decode_shape_glb(
                            layout_shape_st, shape_dec, shape_normalization,
                            layout_glb_path, device=device,
                        )
                except Exception as e:
                    print(f"    Warning: layout decode failed: {e}")

            # ── 3. Individual assets (part_layouts[2:]) ──
            asset_start = 2
            for part_idx in range(asset_start, len(part_layouts)):
                part_slice = part_layouts[part_idx]
                n_voxels = part_slice.stop - part_slice.start
                if n_voxels < 10:
                    continue

                asset_idx = part_idx - asset_start
                if asset_idx < len(asset_names):
                    aname = str(asset_names[asset_idx])
                else:
                    aname = f'asset_{asset_idx:03d}'

                safe_name = aname.replace('/', '_').replace(' ', '_')
                asset_glb_path = os.path.join(assets_dir, f'{asset_idx:03d}_{safe_name}.glb')

                try:
                    part_shape_st = SparseTensor(
                        coords=shape_coords[part_slice],
                        feats=shape_feats[part_slice],
                    )
                    if use_texture:
                        part_tex_st = SparseTensor(
                            coords=tex_coords[part_slice],
                            feats=tex_feats[part_slice],
                        )
                        decode_textured_glb(
                            part_shape_st, part_tex_st,
                            tex_shape_dec, pbr_dec, tex_layout,
                            shape_resolution, tex_shape_normalization, tex_normalization,
                            asset_glb_path, decimation_target=100000, device=device,
                        )
                    else:
                        decode_shape_glb(
                            part_shape_st, shape_dec, shape_normalization,
                            asset_glb_path, device=device,
                        )
                except Exception as e:
                    print(f"    Warning: asset {asset_idx} ({aname}) decode failed: {e}")

            num_decoded += 1

        except Exception as e:
            print(f"  Error: {scene_id}/{room_id}: {e}")
            import traceback
            traceback.print_exc()
            num_failed += 1

    print(f"\nDone: {num_decoded} decoded, {num_skipped} skipped (existing), "
          f"{num_not_ready} skipped (texture not ready), {num_failed} failed")


if __name__ == '__main__':
    main()
