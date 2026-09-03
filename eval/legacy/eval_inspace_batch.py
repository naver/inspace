# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Batch Inference for InSpace (Ours) on perspective_eval_dataset_selected.json

Runs the full InSpace pipeline (Stage 1 → BBox → Stage 2 Shape → Stage 2 Texture → Mesh)
on the curated evaluation samples, with per-sample view_idx support.

Output folder naming matches other methods: {idx:04d}_{room_name}_v{view_idx}

Usage:
    # Default: predicted bbox, floor_perimeter_clean, with texture
    python eval/viewers/eval_inspace_batch.py

    # GT bbox mode
    python eval/viewers/eval_inspace_batch.py --bbox_mode gt

    # Specific GPU
    python eval/viewers/eval_inspace_batch.py --gpu_id 1

    # Limit samples for testing
    python eval/viewers/eval_inspace_batch.py --max_samples 3

    # Multi-GPU parallel (4 GPUs)
    python eval/viewers/eval_inspace_batch.py --rank 0 --world_size 4 --gpu_id 0
    python eval/viewers/eval_inspace_batch.py --rank 1 --world_size 4 --gpu_id 1
    python eval/viewers/eval_inspace_batch.py --rank 2 --world_size 4 --gpu_id 2
    python eval/viewers/eval_inspace_batch.py --rank 3 --world_size 4 --gpu_id 3
"""

import os
import sys
import json
import glob
import time
import argparse
import shutil
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import shared utilities from eval_pipeline
from eval.pipeline.eval_pipeline import (
    find_latest_ckpt,
    load_denoiser,
    load_cubemap_images,
    load_camera_center,
    load_gt_bboxes,
    construct_stage2_input,
    inverse_normalize,
    save_mesh_glb,
)

from trellis2 import models
from trellis2.modules.sparse.basic import SparseTensor
from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from trellis2.trainers.flow_matching.mixins.erp_image_conditioned import (
    ERPImageEncoder,
    create_spatial_attention_mask,
)
from trellis2.utils.asset_attention_mask import (
    compute_overlap_groups,
    create_per_part_cross_attn_masks,
    filter_visible_assets,
)


# ============================================================
# Sample loading from JSON
# ============================================================

def load_eval_samples(json_path):
    """Load evaluation samples from perspective_eval_dataset_selected.json."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    samples = []
    for entry in data['samples']:
        # Extract scene_id (uuid) and room_id from the entry
        uuid = entry['uuid']
        room_name = entry['room_name']
        view_idx = entry['view_idx']
        idx = entry['idx']

        # Build output folder name matching other methods
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
# Phase 1: Stage 1 — Sparse Structure Generation
# ============================================================

@torch.no_grad()
def run_stage1_batch(args, samples, device='cuda'):
    """Generate sparse structure voxels for all samples."""
    print("\n" + "=" * 60)
    print("Phase 1: Stage 1 — Sparse Structure Generation")
    print("=" * 60)

    with open(args.stage1_config, 'r') as f:
        config = json.load(f)
    trainer_config = config['trainer']['args']

    ckpt_step = find_latest_ckpt(args.stage1_ckpt_dir) if args.ckpt_step == 'latest' \
        else int(args.ckpt_step)
    print(f"  Checkpoint step: {ckpt_step}")

    print("  Loading Stage 1 denoiser...")
    denoiser = load_denoiser(config, args.stage1_ckpt_dir, ckpt_step, device)

    print("  Loading ERP encoder...")
    erp_encoder = ERPImageEncoder(
        image_cond_model=trainer_config['image_cond_model'],
        feature_dim=1024,
    ).to(device)

    print("  Loading SS decoder...")
    pretrained_ss_dec = config['dataset']['args'].get(
        'pretrained_ss_dec',
        'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16'
    )
    ss_decoder = models.from_pretrained(pretrained_ss_dec).to(device).eval()

    use_spatial_attn = trainer_config.get('use_spatial_attention', False)
    spatial_attn_kwargs = {}
    if use_spatial_attn:
        spatial_attn_kwargs = {
            'voxel_resolution': trainer_config.get('voxel_resolution', 16),
            'tokens_per_face': trainer_config.get('tokens_per_face', 1029),
            'fov_degrees': trainer_config.get('spatial_attention_fov', 120.0),
            'soft_mask': trainer_config.get('spatial_attention_soft', True),
            'soft_margin': trainer_config.get('spatial_attention_soft_margin', 0.1),
        }

    sigma_min = trainer_config.get('sigma_min', 1e-5)
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=sigma_min)

    results = {}
    for sample in tqdm(samples, desc="Stage 1"):
        scene_id = sample['scene_id']
        room_id = sample['room_id']
        view_idx = sample['view_idx']
        folder_name = sample['folder_name']

        out_dir = os.path.join(args.output_dir, folder_name)
        ss_path = os.path.join(out_dir, 'ss_latent.npz')

        if args.skip_existing and os.path.exists(ss_path):
            data = np.load(ss_path)
            results[folder_name] = {
                'z': torch.from_numpy(data['z']).float(),
                'voxel': torch.from_numpy(data['voxel']),
            }
            continue

        try:
            # Use per-sample view_idx for cubemap loading
            cond = load_cubemap_images(
                args.data_dir, scene_id, room_id,
                view_idx=view_idx
            ).unsqueeze(0).to(device)
            encoded_cond = erp_encoder(cond)
            neg_cond = torch.zeros_like(encoded_cond)

            if args.noise_mode == 'sdedit':
                latent_model = config['dataset']['args'].get('latent_model', 'ss_enc_conv3d_16l8_fp16_64')
                da2_dir = os.path.join(
                    args.data_dir, scene_id, room_id,
                    'depth_voxels_da2_ss_latent', latent_model,
                )
                da2_path = os.path.join(da2_dir, f'{view_idx:04d}.npz')
                if os.path.exists(da2_path):
                    x_init = torch.from_numpy(np.load(da2_path)['z']).float().unsqueeze(0).to(device)
                    t = args.sdedit_alpha
                    gaussian_noise = torch.randn_like(x_init)
                    noise = (1 - t) * x_init + (sigma_min + (1 - sigma_min) * t) * gaussian_noise
                else:
                    print(f"  Warning: DA2 latent not found for {folder_name}, using random noise")
                    noise = torch.randn(1, 8, 16, 16, 16, device=device)
            else:
                noise = torch.randn(1, 8, 16, 16, 16, device=device)

            extra_kwargs = {}
            if use_spatial_attn:
                camera_center = load_camera_center(
                    args.data_dir, scene_id, room_id, view_idx=view_idx)
                cross_attn_mask = create_spatial_attention_mask(
                    camera_center=camera_center.unsqueeze(0).to(device),
                    **spatial_attn_kwargs,
                )
                extra_kwargs['cross_attn_mask'] = cross_attn_mask

            with torch.autocast('cuda', dtype=torch.bfloat16):
                res = sampler.sample(
                    denoiser,
                    noise=noise,
                    cond=encoded_cond,
                    neg_cond=neg_cond,
                    steps=12,
                    rescale_t=5.0,
                    guidance_strength=7.5,
                    guidance_interval=(0.6, 1.0),
                    guidance_rescale=0.7,
                    verbose=False,
                    **extra_kwargs,
                )

            z = res.samples
            voxel = ss_decoder(z.float())
            voxel_binary = (voxel > 0).cpu()

            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(
                ss_path,
                z=z[0].cpu().half().numpy(),
                voxel=voxel_binary[0].numpy(),
            )

            results[folder_name] = {
                'z': z[0].cpu().float(),
                'voxel': voxel_binary[0],
            }

        except Exception as e:
            print(f"  Error: {folder_name}: {e}")
            import traceback
            traceback.print_exc()

    del denoiser, erp_encoder, ss_decoder
    torch.cuda.empty_cache()

    print(f"  Stage 1 complete: {len(results)}/{len(samples)}")
    return results


# ============================================================
# Phase 2: 3D Bounding Box
# ============================================================

@torch.no_grad()
def run_bbox_batch(args, samples, stage1_results, device='cuda'):
    """Get 3D bounding boxes (GT or predicted) for all samples."""
    print("\n" + "=" * 60)
    print(f"Phase 2: 3D Bounding Box ({args.bbox_mode})")
    print("=" * 60)

    results = {}

    if args.bbox_mode == 'gt':
        for sample in tqdm(samples, desc="Loading GT BBox"):
            folder_name = sample['folder_name']
            if folder_name not in stage1_results:
                continue

            bbox_data = load_gt_bboxes(args.data_dir, sample['scene_id'], sample['room_id'])
            if bbox_data is None:
                print(f"  Warning: No GT bbox for {folder_name}")
                continue

            camera_center = load_camera_center(
                args.data_dir, sample['scene_id'], sample['room_id'],
                view_idx=sample['view_idx'])

            obbs_tensor = torch.from_numpy(bbox_data['obbs']).float()
            visible_indices, _ = filter_visible_assets(
                obbs_tensor, camera_center,
                visibility_threshold=0.5,
                fov_degrees=120.0,
                image_size=512,
            )

            visible_obbs = bbox_data['obbs'][visible_indices]
            visible_names = [bbox_data['asset_names'][i] for i in visible_indices]

            results[folder_name] = {
                'obbs': visible_obbs,
                'confidences': np.ones(len(visible_obbs)),
                'source': 'gt',
                'asset_names': visible_names,
                'camera_center': camera_center,
            }

    elif args.bbox_mode == 'predicted':
        from trellis2.models.bbox_centerpoint import BBoxCenterPoint

        with open(args.bbox_config, 'r') as f:
            bbox_config = json.load(f)

        model_args = bbox_config['models']['bbox_centerpoint']['args']
        bbox_model = BBoxCenterPoint(**model_args).to(device).eval()

        if args.bbox_ckpt == 'auto':
            bbox_ckpt_step = find_latest_ckpt(
                'results/bbox_centerpoint_v2', prefix='bbox_centerpoint')
            bbox_ckpt_path = os.path.join(
                'results/bbox_centerpoint_v2/ckpts',
                f'bbox_centerpoint_ema0.9999_step{bbox_ckpt_step:07d}.pt'
            )
            if not os.path.exists(bbox_ckpt_path):
                bbox_ckpt_path = os.path.join(
                    'results/bbox_centerpoint_v2/ckpts',
                    f'bbox_centerpoint_step{bbox_ckpt_step:07d}.pt'
                )
        else:
            bbox_ckpt_path = args.bbox_ckpt

        ckpt = torch.load(bbox_ckpt_path, map_location=device, weights_only=True)
        bbox_model.load_state_dict(ckpt, strict=True)
        print(f"  Loaded BBox model from: {bbox_ckpt_path}")

        for sample in tqdm(samples, desc="Predicting BBox"):
            folder_name = sample['folder_name']
            if folder_name not in stage1_results:
                continue

            voxel = stage1_results[folder_name]['voxel']
            voxel_grid = voxel.float().unsqueeze(0).to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = bbox_model(voxel_grid)
            outputs = {k: v.float() for k, v in outputs.items()}

            detections = bbox_model.decode_detections(
                outputs,
                score_threshold=args.bbox_score_threshold,
                nms_kernel=7,
                iou_nms_threshold=0.3,
            )

            det = detections[0]
            n_pred = det['pred_centers'].shape[0]
            if n_pred > 0:
                pred_obbs = torch.cat([
                    det['pred_centers'],
                    det['pred_sizes'],
                    det['pred_rotations'],
                ], dim=-1).cpu().numpy()
                pred_conf = det['pred_confidences'][:, 0].cpu().numpy()
                sort_idx = np.argsort(-pred_conf)
                pred_obbs = pred_obbs[sort_idx]
                pred_conf = pred_conf[sort_idx]
            else:
                pred_obbs = np.zeros((0, 7), dtype=np.float32)
                pred_conf = np.zeros(0, dtype=np.float32)

            camera_center = load_camera_center(
                args.data_dir, sample['scene_id'], sample['room_id'],
                view_idx=sample['view_idx'])
            pred_names = [f'asset_{i:03d}' for i in range(len(pred_obbs))]

            results[folder_name] = {
                'obbs': pred_obbs,
                'confidences': pred_conf,
                'source': 'predicted',
                'asset_names': pred_names,
                'camera_center': camera_center,
            }

        del bbox_model
        torch.cuda.empty_cache()

    # Save bbox results
    for sample in samples:
        folder_name = sample['folder_name']
        if folder_name not in results:
            continue
        bbox_res = results[folder_name]
        out_dir = os.path.join(args.output_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, 'bboxes.npz'),
            obbs=bbox_res['obbs'],
            confidences=bbox_res['confidences'],
            source=bbox_res['source'],
            asset_names=bbox_res['asset_names'],
        )

    print(f"  BBox complete: {len(results)}/{len(samples)}")
    return results


# ============================================================
# Phase 3: Stage 2-1 — Shape Generation
# ============================================================

@torch.no_grad()
def run_stage2_shape_batch(args, samples, stage1_results, bbox_results, device='cuda'):
    """Run shape generation for all samples."""
    print("\n" + "=" * 60)
    print("Phase 3: Stage 2-1 — Shape Generation")
    print("=" * 60)

    with open(args.stage2_shape_config, 'r') as f:
        config = json.load(f)
    trainer_config = config['trainer']['args']

    ckpt_step = find_latest_ckpt(args.stage2_shape_ckpt_dir) if args.ckpt_step == 'latest' \
        else int(args.ckpt_step)
    print(f"  Checkpoint step: {ckpt_step}")

    print("  Loading Stage 2 shape denoiser...")
    denoiser = load_denoiser(config, args.stage2_shape_ckpt_dir, ckpt_step, device)

    print("  Loading ERP encoder...")
    erp_encoder = ERPImageEncoder(
        image_cond_model=trainer_config['image_cond_model'],
        feature_dim=1024,
    ).to(device)

    sigma_min = trainer_config.get('sigma_min', 1e-5)
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=sigma_min)

    tokens_per_face = trainer_config.get('tokens_per_face', 1029)
    fov_degrees = trainer_config.get('fov_degrees', 120.0)
    expand_pixels = trainer_config.get('expand_pixels', 28)
    overlap_margin = trainer_config.get('overlap_margin', 0.02)
    voxel_resolution = config['models']['denoiser']['args'].get('resolution', 32)

    results = {}
    for sample in tqdm(samples, desc="Stage 2 Shape"):
        folder_name = sample['folder_name']
        if folder_name not in stage1_results or folder_name not in bbox_results:
            continue

        out_dir = os.path.join(args.output_dir, folder_name)
        shape_path = os.path.join(out_dir, 'shape_latent.npz')

        if args.skip_existing and os.path.exists(shape_path):
            data = np.load(shape_path, allow_pickle=True)
            coords = torch.from_numpy(data['coords']).int()
            feats = torch.from_numpy(data['feats']).float()
            part_layouts_raw = data['part_layouts']
            part_layouts = [slice(int(s[0]), int(s[1])) for s in part_layouts_raw]
            results[folder_name] = {
                'shape_latent': SparseTensor(coords=coords, feats=feats),
                'part_layouts': part_layouts,
            }
            continue

        try:
            voxel_64 = stage1_results[folder_name]['voxel']
            bbox_data = bbox_results[folder_name]
            obbs = bbox_data['obbs']
            camera_center = bbox_data['camera_center']

            noise_st, part_layouts, coords_32, valid_obb_indices = construct_stage2_input(
                voxel_64, obbs, device, layout_mode=args.layout_mode)

            if noise_st is None:
                print(f"  Skipping {folder_name}: no active voxels")
                continue

            obbs_filtered = obbs[valid_obb_indices] if len(valid_obb_indices) > 0 else obbs[:0]

            cond_img = load_cubemap_images(
                args.data_dir, sample['scene_id'], sample['room_id'],
                view_idx=sample['view_idx']
            ).unsqueeze(0).to(device)
            encoded_cond = erp_encoder(cond_img)
            neg_cond = torch.zeros_like(encoded_cond)

            obbs_tensor = torch.from_numpy(obbs_filtered).float()
            if obbs_tensor.shape[0] > 0:
                overlap_groups = compute_overlap_groups(obbs_tensor, margin=overlap_margin)
            else:
                overlap_groups = []

            num_parts = len(part_layouts)
            overall_voxel_coords = noise_st.coords[part_layouts[0], 1:4]
            layout_slice = part_layouts[1]
            layout_voxel_coords = noise_st.coords[layout_slice, 1:4] \
                if (layout_slice.stop - layout_slice.start) > 0 else None
            masks = create_per_part_cross_attn_masks(
                obbs=obbs_tensor if obbs_tensor.shape[0] > 0 else None,
                camera_center=camera_center,
                num_parts=num_parts,
                tokens_per_face=tokens_per_face,
                fov_degrees=fov_degrees,
                expand_pixels=expand_pixels,
                overall_voxel_coords=overall_voxel_coords,
                layout_voxel_coords=layout_voxel_coords,
                has_layout=True,
                voxel_resolution=voxel_resolution,
            )

            with torch.autocast('cuda', dtype=torch.bfloat16):
                res = sampler.sample(
                    denoiser,
                    noise=noise_st,
                    cond=encoded_cond,
                    neg_cond=neg_cond,
                    part_layouts=[part_layouts],
                    overlap_groups=[overlap_groups],
                    cross_attn_masks=[masks],
                    has_layout=[True],
                    steps=50,
                    guidance_strength=3.0,
                    verbose=False,
                )

            shape_latent = res.samples

            os.makedirs(out_dir, exist_ok=True)
            part_layouts_arr = np.array([(s.start, s.stop) for s in part_layouts])
            np.savez_compressed(
                shape_path,
                coords=shape_latent.coords.cpu().numpy(),
                feats=shape_latent.feats.cpu().numpy(),
                part_layouts=part_layouts_arr,
            )

            results[folder_name] = {
                'shape_latent': SparseTensor(
                    coords=shape_latent.coords.cpu(),
                    feats=shape_latent.feats.cpu(),
                ),
                'part_layouts': part_layouts,
                'obbs_filtered': obbs_filtered,
            }

        except Exception as e:
            print(f"  Error: {folder_name}: {e}")
            import traceback
            traceback.print_exc()

    del denoiser, erp_encoder
    torch.cuda.empty_cache()

    print(f"  Stage 2 shape complete: {len(results)}/{len(samples)}")
    return results


# ============================================================
# Phase 4: Stage 2-2 — Texture Generation
# ============================================================

@torch.no_grad()
def run_stage2_texture_batch(args, samples, bbox_results, shape_results, device='cuda'):
    """Run texture generation for all samples."""
    print("\n" + "=" * 60)
    print("Phase 4: Stage 2-2 — Texture Generation")
    print("=" * 60)

    with open(args.stage2_tex_config, 'r') as f:
        config = json.load(f)
    trainer_config = config['trainer']['args']

    ckpt_step = find_latest_ckpt(args.stage2_tex_ckpt_dir) if args.ckpt_step == 'latest' \
        else int(args.ckpt_step)
    print(f"  Checkpoint step: {ckpt_step}")

    print("  Loading Stage 2 texture denoiser...")
    denoiser = load_denoiser(config, args.stage2_tex_ckpt_dir, ckpt_step, device)

    print("  Loading ERP encoder...")
    erp_encoder = ERPImageEncoder(
        image_cond_model=trainer_config['image_cond_model'],
        feature_dim=1024,
    ).to(device)

    sigma_min = trainer_config.get('sigma_min', 1e-5)
    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=sigma_min)

    tokens_per_face = trainer_config.get('tokens_per_face', 1029)
    fov_degrees = trainer_config.get('fov_degrees', 120.0)
    expand_pixels = trainer_config.get('expand_pixels', 28)
    overlap_margin = trainer_config.get('overlap_margin', 0.02)
    voxel_resolution = config['models']['denoiser']['args'].get('resolution', 32)

    results = {}
    for sample in tqdm(samples, desc="Stage 2 Texture"):
        folder_name = sample['folder_name']
        if folder_name not in shape_results or folder_name not in bbox_results:
            continue

        out_dir = os.path.join(args.output_dir, folder_name)
        tex_path = os.path.join(out_dir, 'texture_latent.npz')

        if args.skip_existing and os.path.exists(tex_path):
            shape_data = shape_results[folder_name]
            tex_data_np = np.load(tex_path, allow_pickle=True)
            tex_coords = torch.from_numpy(tex_data_np['coords']).int()
            tex_feats = torch.from_numpy(tex_data_np['feats']).float()
            part_layouts_raw = tex_data_np['part_layouts']
            part_layouts = [slice(int(s[0]), int(s[1])) for s in part_layouts_raw]
            results[folder_name] = {
                'tex_latent': SparseTensor(coords=tex_coords, feats=tex_feats),
                'shape_latent': shape_data['shape_latent'],
                'part_layouts': part_layouts,
            }
            continue

        try:
            shape_data = shape_results[folder_name]
            shape_latent = shape_data['shape_latent']
            part_layouts = shape_data['part_layouts']
            bbox_data = bbox_results[folder_name]
            camera_center = bbox_data['camera_center']
            obbs_filtered = shape_data.get('obbs_filtered', bbox_data['obbs'])
            obbs_tensor = torch.from_numpy(obbs_filtered).float()

            concat_cond = SparseTensor(
                coords=shape_latent.coords.to(device),
                feats=shape_latent.feats.clone().to(device),
            )

            tex_noise_feats = torch.randn(shape_latent.feats.shape[0], 32, device=device)
            tex_noise = SparseTensor(
                coords=shape_latent.coords.to(device),
                feats=tex_noise_feats,
            )

            cond_img = load_cubemap_images(
                args.data_dir, sample['scene_id'], sample['room_id'],
                view_idx=sample['view_idx']
            ).unsqueeze(0).to(device)
            encoded_cond = erp_encoder(cond_img)
            neg_cond = torch.zeros_like(encoded_cond)

            if obbs_tensor.shape[0] > 0:
                overlap_groups = compute_overlap_groups(obbs_tensor, margin=overlap_margin)
            else:
                overlap_groups = []

            num_parts = len(part_layouts)
            overall_voxel_coords = tex_noise.coords[part_layouts[0], 1:4]
            layout_slice = part_layouts[1]
            layout_voxel_coords = tex_noise.coords[layout_slice, 1:4] \
                if (layout_slice.stop - layout_slice.start) > 0 else None
            masks = create_per_part_cross_attn_masks(
                obbs=obbs_tensor if obbs_tensor.shape[0] > 0 else None,
                camera_center=camera_center,
                num_parts=num_parts,
                tokens_per_face=tokens_per_face,
                fov_degrees=fov_degrees,
                expand_pixels=expand_pixels,
                overall_voxel_coords=overall_voxel_coords,
                layout_voxel_coords=layout_voxel_coords,
                has_layout=True,
                voxel_resolution=voxel_resolution,
            )

            with torch.autocast('cuda', dtype=torch.bfloat16):
                res = sampler.sample(
                    denoiser,
                    noise=tex_noise,
                    cond=encoded_cond,
                    neg_cond=neg_cond,
                    concat_cond=concat_cond,
                    part_layouts=[part_layouts],
                    overlap_groups=[overlap_groups],
                    cross_attn_masks=[masks],
                    has_layout=[True],
                    steps=50,
                    guidance_strength=3.0,
                    verbose=False,
                )

            tex_latent = res.samples

            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(
                tex_path,
                coords=tex_latent.coords.cpu().numpy(),
                feats=tex_latent.feats.cpu().numpy(),
                part_layouts=np.array([(s.start, s.stop) for s in part_layouts]),
            )

            results[folder_name] = {
                'tex_latent': SparseTensor(
                    coords=tex_latent.coords.cpu(),
                    feats=tex_latent.feats.cpu(),
                ),
                'shape_latent': shape_latent,
                'part_layouts': part_layouts,
            }

        except Exception as e:
            print(f"  Error: {folder_name}: {e}")
            import traceback
            traceback.print_exc()

    del denoiser, erp_encoder
    torch.cuda.empty_cache()

    print(f"  Stage 2 texture complete: {len(results)}/{len(samples)}")
    return results


# ============================================================
# Phase 5: Decode & Save Meshes
# ============================================================

@torch.no_grad()
def run_decode_meshes_batch(
    args, samples, bbox_results, shape_results,
    texture_results=None, device='cuda',
):
    """Decode latents to GLB meshes."""
    print("\n" + "=" * 60)
    print("Phase 5: Decode & Save Meshes")
    print("=" * 60)

    with open(args.stage2_shape_config, 'r') as f:
        shape_config = json.load(f)
    shape_normalization = shape_config['dataset']['args'].get('normalization', None)
    data_resolution = shape_config['dataset']['args'].get('resolution', 512)
    shape_resolution = data_resolution

    pretrained_slat_dec = shape_config['dataset']['args'].get(
        'pretrained_slat_dec',
        'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16'
    )
    print(f"  Loading shape decoder from: {pretrained_slat_dec}")
    shape_dec = models.from_pretrained(pretrained_slat_dec)
    shape_dec.set_resolution(shape_resolution)
    shape_dec = shape_dec.to(device).eval()

    # Load texture decoders
    pbr_dec = None
    tex_shape_dec = None
    tex_normalization = None
    tex_shape_normalization = None
    tex_layout = None

    if args.enable_texture and texture_results:
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

        pretrained_pbr_dec = tex_config['dataset']['args'].get(
            'pretrained_pbr_slat_dec',
            'microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16'
        )
        pretrained_shape_slat_dec = tex_config['dataset']['args'].get(
            'pretrained_shape_slat_dec',
            'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16'
        )
        print(f"  Loading texture shape decoder from: {pretrained_shape_slat_dec}")
        tex_shape_dec = models.from_pretrained(pretrained_shape_slat_dec)
        tex_shape_dec.set_resolution(shape_resolution)
        tex_shape_dec = tex_shape_dec.to(device).eval()

        print(f"  Loading PBR decoder from: {pretrained_pbr_dec}")
        pbr_dec = models.from_pretrained(pretrained_pbr_dec)
        pbr_dec = pbr_dec.to(device).eval()

    num_decoded = 0
    for sample in tqdm(samples, desc="Decoding meshes"):
        folder_name = sample['folder_name']
        if folder_name not in shape_results or folder_name not in bbox_results:
            continue

        out_dir = os.path.join(args.output_dir, folder_name)
        scene_glb_path = os.path.join(out_dir, 'scene.glb')

        if args.skip_existing and os.path.exists(scene_glb_path):
            num_decoded += 1
            continue

        os.makedirs(out_dir, exist_ok=True)

        shape_data = shape_results[folder_name]
        shape_latent = shape_data['shape_latent']
        part_layouts = shape_data['part_layouts']
        bbox_data = bbox_results[folder_name]

        use_texture = (args.enable_texture and texture_results and folder_name in texture_results)

        try:
            if use_texture:
                import o_voxel
                from trellis2.representations import MeshWithVoxel
                tex_data = texture_results[folder_name]
                tex_latent = tex_data['tex_latent']

                overall_slice = part_layouts[0]
                overall_shape = SparseTensor(
                    coords=shape_latent.coords[overall_slice],
                    feats=shape_latent.feats[overall_slice],
                ).to(device)
                overall_tex = SparseTensor(
                    coords=tex_latent.coords[overall_slice],
                    feats=tex_latent.feats[overall_slice],
                ).to(device)

                if tex_shape_normalization:
                    shape_mean = torch.tensor(tex_shape_normalization['mean']).reshape(1, -1).to(device)
                    shape_std = torch.tensor(tex_shape_normalization['std']).reshape(1, -1).to(device)
                    overall_shape = overall_shape.replace(feats=overall_shape.feats * shape_std + shape_mean)
                if tex_normalization:
                    overall_tex = inverse_normalize(overall_tex, tex_normalization)

                mesh, subs = tex_shape_dec(overall_shape, return_subs=True)
                vox = pbr_dec(overall_tex, guide_subs=subs) * 0.5 + 0.5
                if mesh and len(mesh) > 0:
                    mesh[0].fill_holes()
                    try:
                        attr_volume = vox[0].feats
                        attr_coords = vox[0].coords[:, 1:]
                        glb_mesh = o_voxel.postprocess.to_glb(
                            vertices=mesh[0].vertices,
                            faces=mesh[0].faces,
                            attr_volume=attr_volume,
                            coords=attr_coords,
                            attr_layout=tex_layout,
                            grid_size=shape_resolution,
                            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                            decimation_target=500000,
                            texture_size=2048,
                            remesh=True,
                            remesh_band=1,
                            remesh_project=0,
                            verbose=False,
                        )
                        glb_mesh.export(scene_glb_path)
                    except Exception as e:
                        print(f"    Warning: GLB export failed: {e}, using vertex colors")
                        save_mesh_glb(mesh[0].vertices, mesh[0].faces, scene_glb_path)
            else:
                overall_slice = part_layouts[0]
                overall_z = SparseTensor(
                    coords=shape_latent.coords[overall_slice],
                    feats=shape_latent.feats[overall_slice],
                ).to(device)
                if shape_normalization:
                    overall_z = inverse_normalize(overall_z, shape_normalization)
                reps = shape_dec(overall_z)
                if reps and len(reps) > 0:
                    save_mesh_glb(reps[0].vertices, reps[0].faces, scene_glb_path)

            # --- Decode layout (part_layouts[1]) ---
            layout_slice = part_layouts[1]
            n_layout = layout_slice.stop - layout_slice.start
            if n_layout > 0:
                layout_glb_path = os.path.join(out_dir, 'layout.glb')
                try:
                    if use_texture:
                        layout_shape = SparseTensor(
                            coords=shape_latent.coords[layout_slice],
                            feats=shape_latent.feats[layout_slice],
                        ).to(device)
                        layout_tex = SparseTensor(
                            coords=tex_latent.coords[layout_slice],
                            feats=tex_latent.feats[layout_slice],
                        ).to(device)
                        if tex_shape_normalization:
                            layout_shape = layout_shape.replace(
                                feats=layout_shape.feats * shape_std + shape_mean)
                        if tex_normalization:
                            layout_tex = inverse_normalize(layout_tex, tex_normalization)
                        lmesh, lsubs = tex_shape_dec(layout_shape, return_subs=True)
                        lvox = pbr_dec(layout_tex, guide_subs=lsubs) * 0.5 + 0.5
                        if lmesh and len(lmesh) > 0:
                            lmesh[0].fill_holes()
                            try:
                                glb_layout = o_voxel.postprocess.to_glb(
                                    vertices=lmesh[0].vertices,
                                    faces=lmesh[0].faces,
                                    attr_volume=lvox[0].feats,
                                    coords=lvox[0].coords[:, 1:],
                                    attr_layout=tex_layout,
                                    grid_size=shape_resolution,
                                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                    decimation_target=200000,
                                    texture_size=2048,
                                    remesh=True,
                                    remesh_band=1,
                                    remesh_project=0,
                                    verbose=False,
                                )
                                glb_layout.export(layout_glb_path)
                            except Exception:
                                save_mesh_glb(lmesh[0].vertices, lmesh[0].faces, layout_glb_path)
                    else:
                        layout_z = SparseTensor(
                            coords=shape_latent.coords[layout_slice],
                            feats=shape_latent.feats[layout_slice],
                        ).to(device)
                        if shape_normalization:
                            layout_z = inverse_normalize(layout_z, shape_normalization)
                        lreps = shape_dec(layout_z)
                        if lreps and len(lreps) > 0:
                            save_mesh_glb(lreps[0].vertices, lreps[0].faces, layout_glb_path)
                except Exception as e:
                    print(f"    Warning: layout decode failed for {folder_name}: {e}")

            # --- Decode individual assets (part_layouts[2+]) ---
            assets_dir = os.path.join(out_dir, 'assets')
            os.makedirs(assets_dir, exist_ok=True)
            asset_start = 2  # Assets start after overall(0) and layout(1)
            for part_idx in range(asset_start, len(part_layouts)):
                part_slice = part_layouts[part_idx]
                n_voxels = part_slice.stop - part_slice.start
                if n_voxels < 10:
                    continue

                asset_idx = part_idx - asset_start
                if asset_idx < len(bbox_data['asset_names']):
                    asset_name = bbox_data['asset_names'][asset_idx]
                else:
                    asset_name = f'asset_{asset_idx:03d}'

                try:
                    safe_name = asset_name.replace('/', '_').replace(' ', '_')
                    asset_glb_path = os.path.join(assets_dir, f'{asset_idx:03d}_{safe_name}.glb')

                    if use_texture:
                        part_shape = SparseTensor(
                            coords=shape_latent.coords[part_slice],
                            feats=shape_latent.feats[part_slice],
                        ).to(device)
                        part_tex = SparseTensor(
                            coords=tex_latent.coords[part_slice],
                            feats=tex_latent.feats[part_slice],
                        ).to(device)
                        if tex_shape_normalization:
                            part_shape = part_shape.replace(
                                feats=part_shape.feats * shape_std + shape_mean)
                        if tex_normalization:
                            part_tex = inverse_normalize(part_tex, tex_normalization)
                        pmesh, psubs = tex_shape_dec(part_shape, return_subs=True)
                        pvox = pbr_dec(part_tex, guide_subs=psubs) * 0.5 + 0.5
                        if pmesh and len(pmesh) > 0:
                            pmesh[0].fill_holes()
                            try:
                                glb_asset = o_voxel.postprocess.to_glb(
                                    vertices=pmesh[0].vertices,
                                    faces=pmesh[0].faces,
                                    attr_volume=pvox[0].feats,
                                    coords=pvox[0].coords[:, 1:],
                                    attr_layout=tex_layout,
                                    grid_size=shape_resolution,
                                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                    decimation_target=100000,
                                    texture_size=2048,
                                    remesh=True,
                                    remesh_band=1,
                                    remesh_project=0,
                                    verbose=False,
                                )
                                glb_asset.export(asset_glb_path)
                            except Exception:
                                save_mesh_glb(pmesh[0].vertices, pmesh[0].faces, asset_glb_path)
                    else:
                        part_z = SparseTensor(
                            coords=shape_latent.coords[part_slice],
                            feats=shape_latent.feats[part_slice],
                        ).to(device)
                        if shape_normalization:
                            part_z = inverse_normalize(part_z, shape_normalization)
                        reps = shape_dec(part_z)
                        if reps and len(reps) > 0:
                            save_mesh_glb(reps[0].vertices, reps[0].faces, asset_glb_path)

                except Exception as e:
                    print(f"    Error decoding asset {asset_idx} ({asset_name}): {e}")

            # --- Merge all individual asset GLBs into a single assets.glb ---
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

            # Copy perspective image to output folder for easy comparison
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
                'bbox_mode': args.bbox_mode,
                'noise_mode': args.noise_mode,
                'sdedit_alpha': args.sdedit_alpha if args.noise_mode == 'sdedit' else None,
                'layout_mode': args.layout_mode,
                'texture_enabled': use_texture,
                'erp_image': sample['erp_image'],
                'perspective_image': sample['perspective_image'],
            }
            with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            num_decoded += 1

        except Exception as e:
            print(f"  Error decoding {folder_name}: {e}")
            import traceback
            traceback.print_exc()

    del shape_dec
    if pbr_dec is not None:
        del pbr_dec, tex_shape_dec
    torch.cuda.empty_cache()

    print(f"  Decoded {num_decoded}/{len(samples)} samples")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='InSpace Batch Inference on perspective_eval_dataset_selected.json')

    # Paths
    parser.add_argument('--eval_json', type=str,
                        default='evals/perspective_eval_dataset_selected.json')
    parser.add_argument('--data_dir', type=str,
                        default='datasets/ERP_3D_FRONT_test')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir. Auto-generated with annotations if not specified.')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Stage 1
    parser.add_argument('--stage1_config', type=str,
                        default='configs/gen/erp_ss_flow_img_dit_L_16l8_bf16_spatial.json')
    parser.add_argument('--stage1_ckpt_dir', type=str,
                        default='results/erp_ss_flow_img_dit_L_16l8_bf16_spatial')
    parser.add_argument('--noise_mode', type=str, default='random',
                        choices=['random', 'sdedit'],
                        help='Stage 1 noise init: random=pure gaussian, sdedit=depth-based init')
    parser.add_argument('--sdedit_alpha', type=float, default=0.5,
                        help='SDEdit noise blend factor (0=clean init, 1=pure noise)')

    # BBox
    parser.add_argument('--bbox_mode', type=str, default='predicted',
                        choices=['gt', 'predicted'])
    parser.add_argument('--bbox_config', type=str,
                        default='configs/bbox/erp_bbox_centerpoint_v2.json')
    parser.add_argument('--bbox_ckpt', type=str, default='auto')
    parser.add_argument('--bbox_score_threshold', type=float, default=0.3)

    # Stage 2
    parser.add_argument('--stage2_shape_config', type=str,
                        default='configs/gen/erp_slat_flow_img2shape_asset_aware_bf16.json')
    parser.add_argument('--stage2_shape_ckpt_dir', type=str,
                        default='results/erp_slat_flow_img2shape_asset_aware_bf16')
    parser.add_argument('--stage2_tex_config', type=str,
                        default='configs/gen/erp_slat_flow_imgshape2tex_asset_aware_bf16.json')
    parser.add_argument('--stage2_tex_ckpt_dir', type=str,
                        default='results/erp_slat_flow_imgshape2tex_asset_aware_bf16_weight_sampling')
    parser.add_argument('--enable_texture', action='store_true', default=True)
    parser.add_argument('--no_texture', dest='enable_texture', action='store_false')
    parser.add_argument('--layout_mode', type=str, default='floor_perimeter_clean',
                        choices=['floor_perimeter', 'floor_perimeter_clean', 'no_floor_assets'])

    # Common
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false')
    parser.add_argument('--ckpt_step', type=str, default='latest')

    # Distributed
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)

    args = parser.parse_args()

    # Auto-generate output_dir with annotations if not specified
    if args.output_dir is None:
        tag = f"bbox_{args.bbox_mode}"
        if args.noise_mode == 'sdedit':
            tag += f"_sdedit{args.sdedit_alpha}"
        else:
            tag += "_random"
        args.output_dir = f"evals/output_InSpace_batch_{tag}"

    os.makedirs(args.output_dir, exist_ok=True)
    device = f'cuda:{args.gpu_id}'

    # Load samples from JSON
    all_samples = load_eval_samples(args.eval_json)
    print(f"\nInSpace Batch Inference")
    print(f"  JSON: {args.eval_json}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Output: {args.output_dir}")
    print(f"  BBox mode: {args.bbox_mode}")
    print(f"  Noise mode: {args.noise_mode}" + (f" (alpha={args.sdedit_alpha})" if args.noise_mode == 'sdedit' else ''))
    print(f"  Layout mode: {args.layout_mode}")
    print(f"  Texture: {args.enable_texture}")
    print(f"  Device: {device}")

    # Limit samples
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

    # Save eval config
    eval_config = vars(args).copy()
    eval_config['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    eval_config['n_samples'] = len(samples)
    with open(os.path.join(args.output_dir, f'eval_config_rank{args.rank}.json'), 'w') as f:
        json.dump(eval_config, f, indent=2)

    # Phase 1: Stage 1 — Sparse Structure
    stage1_results = run_stage1_batch(args, samples, device)

    # Phase 2: BBox
    bbox_results = run_bbox_batch(args, samples, stage1_results, device)

    # Phase 3: Stage 2-1 — Shape
    shape_results = run_stage2_shape_batch(args, samples, stage1_results, bbox_results, device)

    # Phase 4: Stage 2-2 — Texture
    texture_results = None
    if args.enable_texture:
        texture_results = run_stage2_texture_batch(
            args, samples, bbox_results, shape_results, device)

    # Phase 5: Decode & Save Meshes
    run_decode_meshes_batch(args, samples, bbox_results, shape_results, texture_results, device)

    # Summary
    print("\n" + "=" * 60)
    print("InSpace Batch Inference Complete!")
    print("=" * 60)
    print(f"  Output: {args.output_dir}")
    print(f"  Samples processed: {len(stage1_results)}")
    print(f"  BBox mode: {args.bbox_mode}")
    print(f"  Texture: {args.enable_texture}")



if __name__ == '__main__':
    main()

# # 4-GPU parallel
# CUDA_VISIBLE_DEVICES=3 python eval/viewers/eval_inspace_batch.py --rank 0 --world_size 4 --gpu_id 0 --noise_mode sdedit --sdedit_alpha 0.7
# CUDA_VISIBLE_DEVICES=3 python eval/viewers/eval_inspace_batch.py --rank 1 --world_size 4 --gpu_id 0 --noise_mode sdedit --sdedit_alpha 0.7
# CUDA_VISIBLE_DEVICES=5 python eval/viewers/eval_inspace_batch.py --rank 2 --world_size 4 --gpu_id 0 --noise_mode sdedit --sdedit_alpha 0.7
# CUDA_VISIBLE_DEVICES=5 python eval/viewers/eval_inspace_batch.py --rank 3 --world_size 4 --gpu_id 0 --noise_mode sdedit --sdedit_alpha 0.7

# CUDA_VISIBLE_DEVICES=2 python eval/viewers/eval_inspace_batch.py --rank 0 --world_size 4 --gpu_id 0 --noise_mode sdedit --sdedit_alpha 0.3
# CUDA_VISIBLE_DEVICES=7 python eval/viewers/eval_inspace_batch.py --rank 1 --world_size 4 --gpu_id 0 --noise_mode sdedit --sdedit_alpha 0.3
# CUDA_VISIBLE_DEVICES=7 python eval/viewers/eval_inspace_batch.py --rank 2 --world_size 4 --gpu_id 0 --noise_mode sdedit --sdedit_alpha 0.3
# CUDA_VISIBLE_DEVICES=7 python eval/viewers/eval_inspace_batch.py --rank 3 --world_size 4 --gpu_id 0 --noise_mode sdedit --sdedit_alpha 0.3

# random 
# CUDA_VISIBLE_DEVICES=5 python eval/viewers/eval_inspace_batch.py --rank 0 --world_size 4 --gpu_id 0 --noise_mode random
# CUDA_VISIBLE_DEVICES=6 python eval/viewers/eval_inspace_batch.py --rank 1 --world_size 4 --gpu_id 0 --noise_mode random
# CUDA_VISIBLE_DEVICES=6 python eval/viewers/eval_inspace_batch.py --rank 2 --world_size 4 --gpu_id 0 --noise_mode random
# CUDA_VISIBLE_DEVICES=6 python eval/viewers/eval_inspace_batch.py --rank 3 --world_size 4 --gpu_id 0 --noise_mode random

# Output structure (matches other methods)

# evals/output_InSpace_batch/
# ├── 0000_LivingDiningRoom-2610_v6/
# │   ├── scene.glb              # Textured scene mesh
# │   ├── input_perspective.png  # Input image (copied)
# │   ├── metadata.json          # Runtime config
# │   ├── ss_latent.npz          # Stage 1 intermediate
# │   ├── shape_latent.npz       # Stage 2-1 intermediate
# │   ├── texture_latent.npz     # Stage 2-2 intermediate
# │   └── bboxes.npz             # BBox results
# ├── 0001_LivingDiningRoom-515_v12/
# │   └── ...