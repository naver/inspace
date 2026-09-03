# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Full Evaluation Pipeline: Stage 1 (SS) → BBox → Stage 2 (Shape/Texture) → Mesh Output

Chains all stages of the InSpace pipeline for end-to-end evaluation:
  1. Stage 1: ERP cubemap → Sparse Structure latent → [1,64,64,64] voxel
  2. BBox: GT or predicted 3D bounding boxes
  3. Stage 2-1: Shape generation (asset-aware)
  4. Stage 2-2: Texture generation (optional, asset-aware)
  5. Decode latents → meshes + visualization

Usage:
    # Minimal test (GT bbox, shape only)
    python eval/pipeline/eval_pipeline.py \
        --data_dir datasets/ERP_3D_FRONT_test \
        --output_dir evals/stage12_test \
        --bbox_mode gt --max_samples 2

    # Full pipeline with predicted bbox
    python eval/pipeline/eval_pipeline.py \
        --bbox_mode predicted --max_samples 10

    # SDEdit mode
    python eval/pipeline/eval_pipeline.py \
        --noise_mode sdedit --sdedit_alpha 0.5 --bbox_mode gt

    # With texture generation
    python eval/pipeline/eval_pipeline.py \
        --enable_texture --bbox_mode gt --max_samples 5
"""

import os
import sys
import json
import glob
import math
import time
import argparse
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trellis2 import models
from trellis2.modules.sparse.basic import SparseTensor, sparse_cat
from trellis2.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from trellis2.trainers.flow_matching.mixins.erp_image_conditioned import (
    ERPImageEncoder,
    create_spatial_attention_mask,
)
from trellis2.utils.asset_attention_mask import (
    compute_overlap_groups,
    create_per_part_cross_attn_masks,
    filter_visible_assets,
    is_point_inside_obb,
)


# ============================================================
# Utility functions
# ============================================================

def find_latest_ckpt(ckpt_dir, prefix='denoiser', use_ema=True, ema_rate=0.9999):
    """Find the latest checkpoint step in ckpt_dir/ckpts/."""
    ckpts_dir = os.path.join(ckpt_dir, 'ckpts')
    if use_ema:
        pattern = f'{prefix}_ema{ema_rate}_step*.pt'
    else:
        pattern = f'{prefix}_step*.pt'
    files = glob.glob(os.path.join(ckpts_dir, pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No checkpoint files matching '{pattern}' in {ckpts_dir}")
    steps = [int(os.path.basename(f).split('step')[-1].split('.')[0]) for f in files]
    return max(steps)


def load_denoiser(config, ckpt_dir, ckpt_step, device='cuda', use_ema=True, ema_rate=0.9999):
    """Load denoiser model from config + checkpoint."""
    model_config = config['models']['denoiser']
    denoiser = getattr(models, model_config['name'])(**model_config['args'])
    denoiser = denoiser.to(device)

    if use_ema:
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_ema{ema_rate}_step{ckpt_step:07d}.pt')
    else:
        ckpt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_step{ckpt_step:07d}.pt')

    if not os.path.exists(ckpt_path):
        alt_path = os.path.join(ckpt_dir, 'ckpts', f'denoiser_step{ckpt_step:07d}.pt')
        if use_ema and os.path.exists(alt_path):
            print(f"  EMA ckpt not found, falling back to: {alt_path}")
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    denoiser.load_state_dict(state_dict)
    denoiser.eval()
    print(f"  Loaded denoiser from: {ckpt_path}")
    return denoiser


def discover_samples(data_dir):
    """Discover all test samples in the dataset directory."""
    samples = []
    for scene_id in sorted(os.listdir(data_dir)):
        scene_dir = os.path.join(data_dir, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        for room_id in sorted(os.listdir(scene_dir)):
            room_dir = os.path.join(scene_dir, room_id)
            if not os.path.isdir(room_dir):
                continue
            cubic_dir = os.path.join(room_dir, 'cubic_fov_120', '0000')
            if os.path.isdir(cubic_dir):
                samples.append((scene_id, room_id))
    return samples


def load_cubemap_images(data_dir, scene_id, room_id, view_idx=0, image_size=512):
    """Load 6 cubemap face images and return as [6, 3, H, W] tensor."""
    from PIL import Image
    import torchvision.transforms as T

    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    cubic_dir = os.path.join(data_dir, scene_id, room_id, 'cubic_fov_120', f'{view_idx:04d}')

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    faces = []
    for face_name in face_names:
        img_path = os.path.join(cubic_dir, f'{face_name}.png')
        img = Image.open(img_path).convert('RGB')
        faces.append(transform(img))
    return torch.stack(faces)  # [6, 3, H, W]


def load_camera_center(data_dir, scene_id, room_id, view_idx=0):
    """Load and normalize camera center from camera_poses.json + normalization_info.json."""
    sample_dir = os.path.join(data_dir, scene_id, room_id)

    camera_poses_path = os.path.join(sample_dir, 'camera_poses.json')
    with open(camera_poses_path, 'r') as f:
        camera_data = json.load(f)
    cam_location = camera_data['views'][view_idx]['location']

    norm_info_path = os.path.join(sample_dir, 'mesh_dumps', 'normalization_info.json')
    if not os.path.exists(norm_info_path):
        norm_info_path = os.path.join(sample_dir, 'dual_grid_512', 'normalization_info.json')
    with open(norm_info_path, 'r') as f:
        norm_info = json.load(f)

    center = np.array(norm_info['center'])
    scale = norm_info['scale']

    cam_world = np.array(cam_location)
    cam_normalized = (cam_world - center) * scale
    return torch.from_numpy(cam_normalized).float()


def load_gt_bboxes(data_dir, scene_id, room_id):
    """Load GT bounding boxes from NPZ file."""
    sample_dir = os.path.join(data_dir, scene_id, room_id)
    bbox_dir = os.path.join(sample_dir, '3d_bounding_box')
    npz_files = glob.glob(os.path.join(bbox_dir, '*_scene_data.npz'))
    if not npz_files:
        return None
    npz_path = npz_files[0]
    data = np.load(npz_path, allow_pickle=True)
    return {
        'obbs': data['obbs'].astype(np.float32),
        'asset_filenames': list(data['asset_filenames']),
        'asset_names': list(data['asset_names']),
        'asset_categories': list(data['asset_categories']) if 'asset_categories' in data else [],
        'n_assets': int(data['n_assets']),
        'norm_center': data['norm_center'] if 'norm_center' in data else None,
        'norm_scale': float(data['norm_scale']) if 'norm_scale' in data else None,
    }


def load_gt_voxel_grid(data_dir, scene_id, room_id, resolution=64):
    """Load GT voxel grid from PLY file. Returns [1, res, res, res] bool tensor or None."""
    gt_voxel_path = os.path.join(
        data_dir, scene_id, room_id, 'voxels_64', 'full_room_wo_ceiling.ply')
    if not os.path.exists(gt_voxel_path):
        return None
    try:
        gt_mesh = trimesh.load(gt_voxel_path)
        gt_grid = np.zeros((resolution, resolution, resolution), dtype=bool)
        coords = ((gt_mesh.vertices + 0.5) * resolution).astype(int)
        coords = np.clip(coords, 0, resolution - 1)
        gt_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        return torch.from_numpy(gt_grid).unsqueeze(0)
    except Exception:
        return None


# ============================================================
# Phase 1: Stage 1 — Sparse Structure Generation
# ============================================================

@torch.no_grad()
def run_stage1(
    args,
    samples: List[Tuple[str, str]],
    device: str = 'cuda',
) -> Dict[str, dict]:
    """Phase 1: Generate sparse structure voxels from ERP cubemap images."""
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
    for scene_id, room_id in tqdm(samples, desc="Stage 1"):
        out_dir = os.path.join(args.output_dir, scene_id, room_id)
        ss_path = os.path.join(out_dir, 'ss_latent.npz')

        if args.skip_existing and os.path.exists(ss_path):
            data = np.load(ss_path)
            results[(scene_id, room_id)] = {
                'z': torch.from_numpy(data['z']).float(),
                'voxel': torch.from_numpy(data['voxel']),
            }
            continue

        try:
            cond = load_cubemap_images(args.data_dir, scene_id, room_id).unsqueeze(0).to(device)
            encoded_cond = erp_encoder(cond)
            neg_cond = torch.zeros_like(encoded_cond)

            if args.noise_mode == 'sdedit':
                da2_dir = os.path.join(
                    args.data_dir, scene_id, room_id,
                    'depth_voxels_da2_ss_latent',
                    config['dataset']['args'].get('latent_model', 'ss_enc_conv3d_16l8_fp16_64'),
                )
                da2_path = os.path.join(da2_dir, '0000.npz')
                if os.path.exists(da2_path):
                    x_init = torch.from_numpy(np.load(da2_path)['z']).float().unsqueeze(0).to(device)
                    t = args.sdedit_alpha
                    gaussian_noise = torch.randn_like(x_init)
                    noise = (1 - t) * x_init + (sigma_min + (1 - sigma_min) * t) * gaussian_noise
                else:
                    print(f"  Warning: DA2 latent not found for {scene_id}/{room_id}, using random noise")
                    noise = torch.randn(1, 8, 16, 16, 16, device=device)
            else:
                noise = torch.randn(1, 8, 16, 16, 16, device=device)

            extra_kwargs = {}
            if use_spatial_attn:
                camera_center = load_camera_center(args.data_dir, scene_id, room_id)
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

            results[(scene_id, room_id)] = {
                'z': z[0].cpu().float(),
                'voxel': voxel_binary[0],
            }

        except Exception as e:
            print(f"  Error processing {scene_id}/{room_id}: {e}")
            import traceback
            traceback.print_exc()

    del denoiser, erp_encoder, ss_decoder
    torch.cuda.empty_cache()

    print(f"  Stage 1 complete: {len(results)}/{len(samples)} samples")
    return results


# ============================================================
# Phase 2: 3D Bounding Box
# ============================================================

@torch.no_grad()
def run_bbox(
    args,
    samples: List[Tuple[str, str]],
    stage1_results: Dict,
    device: str = 'cuda',
) -> Dict[str, dict]:
    """Phase 2: Get 3D bounding boxes (GT or predicted)."""
    print("\n" + "=" * 60)
    print(f"Phase 2: 3D Bounding Box ({args.bbox_mode})")
    print("=" * 60)

    results = {}

    if args.bbox_mode == 'gt':
        for scene_id, room_id in tqdm(samples, desc="Loading GT BBox"):
            if (scene_id, room_id) not in stage1_results:
                continue

            bbox_data = load_gt_bboxes(args.data_dir, scene_id, room_id)
            if bbox_data is None:
                print(f"  Warning: No GT bbox for {scene_id}/{room_id}")
                continue

            camera_center = load_camera_center(args.data_dir, scene_id, room_id)

            obbs_tensor = torch.from_numpy(bbox_data['obbs']).float()
            visible_indices, _ = filter_visible_assets(
                obbs_tensor, camera_center,
                visibility_threshold=0.5,
                fov_degrees=120.0,
                image_size=512,
            )

            visible_obbs = bbox_data['obbs'][visible_indices]
            visible_filenames = [bbox_data['asset_filenames'][i] for i in visible_indices]
            visible_names = [bbox_data['asset_names'][i] for i in visible_indices]
            visible_categories = [bbox_data['asset_categories'][i] for i in visible_indices] \
                if bbox_data['asset_categories'] else ['unknown'] * len(visible_indices)

            results[(scene_id, room_id)] = {
                'obbs': visible_obbs,
                'confidences': np.ones(len(visible_obbs)),
                'source': 'gt',
                'asset_filenames': visible_filenames,
                'asset_names': visible_names,
                'asset_categories': visible_categories,
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
                'ckpts/bbox_centerpoint', prefix='bbox_centerpoint')
            bbox_ckpt_path = os.path.join(
                'ckpts/bbox_centerpoint/ckpts',
                f'bbox_centerpoint_ema0.9999_step{bbox_ckpt_step:07d}.pt'
            )
            if not os.path.exists(bbox_ckpt_path):
                bbox_ckpt_path = os.path.join(
                    'ckpts/bbox_centerpoint/ckpts',
                    f'bbox_centerpoint_step{bbox_ckpt_step:07d}.pt'
                )
        else:
            bbox_ckpt_path = args.bbox_ckpt

        ckpt = torch.load(bbox_ckpt_path, map_location=device, weights_only=True)
        bbox_model.load_state_dict(ckpt, strict=True)
        print(f"  Loaded BBox model from: {bbox_ckpt_path}")

        for scene_id, room_id in tqdm(samples, desc="Predicting BBox"):
            if (scene_id, room_id) not in stage1_results:
                continue

            voxel = stage1_results[(scene_id, room_id)]['voxel']
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

            camera_center = load_camera_center(args.data_dir, scene_id, room_id)
            pred_names = [f'asset_{i:03d}' for i in range(len(pred_obbs))]

            results[(scene_id, room_id)] = {
                'obbs': pred_obbs,
                'confidences': pred_conf,
                'source': 'predicted',
                'asset_filenames': pred_names,
                'asset_names': pred_names,
                'asset_categories': ['unknown'] * len(pred_obbs),
                'camera_center': camera_center,
                'gt_bbox_path': os.path.join(
                    args.data_dir, scene_id, room_id, '3d_bounding_box'),
            }

        del bbox_model
        torch.cuda.empty_cache()

    # Save bbox results
    for (scene_id, room_id), bbox_res in results.items():
        out_dir = os.path.join(args.output_dir, scene_id, room_id)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, 'bboxes.npz'),
            obbs=bbox_res['obbs'],
            confidences=bbox_res['confidences'],
            source=bbox_res['source'],
            asset_names=bbox_res['asset_names'],
        )

    print(f"  BBox complete: {len(results)}/{len(samples)} samples")
    return results


# ============================================================
# Phase 3: Construct Stage 2 Input
# ============================================================

def detect_floor_z(coords_32):
    """Find floor Z = Z layer with the most voxels (robust to outlier Z layers)."""
    xyz = coords_32[:, 1:4].cpu().numpy().astype(int)
    unique_z, counts = np.unique(xyz[:, 2], return_counts=True)
    return int(unique_z[np.argmax(counts)])


def detect_layout_from_floor_perimeter(coords_32):
    """
    Detect layout voxels (floor + walls) from overall voxel coords.

    Method:
    1. Floor = Z layer with the most voxels
    2. Floor perimeter = floor voxels with at least 1 missing 4-neighbor
    3. Wall = any voxel whose (X,Y) is on the floor perimeter AND Z > floor
    4. Layout = floor voxels + wall voxels
    """
    xyz = coords_32[:, 1:4].cpu().numpy().astype(int)
    floor_z = detect_floor_z(coords_32)

    floor_mask_z = xyz[:, 2] == floor_z
    floor_xy_set = set(map(tuple, xyz[floor_mask_z, :2]))

    perimeter_xy = set()
    for x, y in floor_xy_set:
        if any((x + dx, y + dy) not in floor_xy_set for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            perimeter_xy.add((x, y))

    layout_mask = torch.zeros(coords_32.shape[0], dtype=torch.bool)
    for i, (x, y, z) in enumerate(xyz):
        if z == floor_z or (z > floor_z and (x, y) in perimeter_xy):
            layout_mask[i] = True

    return layout_mask


def assign_voxels_to_obbs(coords_norm: torch.Tensor, obbs: torch.Tensor) -> List[torch.Tensor]:
    """Assign voxels to OBBs by checking which voxels fall inside each OBB."""
    masks = []
    for i in range(obbs.shape[0]):
        obb = obbs[i]
        cx, cy, cz = obb[0], obb[1], obb[2]
        sx, sy, sz = obb[3], obb[4], obb[5]
        yaw = obb[6]

        local = coords_norm - torch.tensor([cx, cy, cz], device=coords_norm.device)
        cos_a = torch.cos(-yaw)
        sin_a = torch.sin(-yaw)
        rotated_x = local[:, 0] * cos_a - local[:, 1] * sin_a
        rotated_y = local[:, 0] * sin_a + local[:, 1] * cos_a
        rotated_z = local[:, 2]

        inside = (
            (rotated_x.abs() <= sx / 2) &
            (rotated_y.abs() <= sy / 2) &
            (rotated_z.abs() <= sz / 2)
        )
        masks.append(inside)
    return masks


def construct_stage2_input(
    voxel_64: torch.Tensor,
    obbs: np.ndarray,
    device: str = 'cuda',
    layout_mode: str = 'floor_perimeter',
) -> Tuple[SparseTensor, List[slice], torch.Tensor]:
    """Convert Stage 1 voxel grid + OBBs to Stage 2 SparseTensor with part_layouts.

    Args:
        layout_mode:
            - 'floor_perimeter': layout = floor + walls (perimeter detection).
              Assets include all bbox voxels including floor overlap (matches training).
            - 'floor_perimeter_clean': same layout, but assets EXCLUDE floor-layer
              voxels. Reduces floor artifacts on assets like dining tables.
            - 'no_floor_assets': alias for floor_perimeter_clean.
    """
    voxel_float = voxel_64.float().unsqueeze(0)
    voxel_32 = F.max_pool3d(voxel_float, 2, 2, 0) > 0.5

    active = torch.argwhere(voxel_32)
    coords_32 = active[:, [0, 2, 3, 4]].int()
    N_overall = coords_32.shape[0]

    if N_overall == 0:
        print("  Warning: No active voxels at 32³ resolution")
        return None, None, None, []

    coords_norm = (coords_32[:, 1:4].float() + 0.5) / 32.0 - 0.5
    obbs_tensor = torch.from_numpy(obbs).float()
    voxel_masks = assign_voxels_to_obbs(coords_norm, obbs_tensor)

    # Layout detection: floor + walls via perimeter detection (all modes use this)
    layout_mask = detect_layout_from_floor_perimeter(coords_32)

    # Determine whether to exclude floor-layer voxels from assets
    exclude_floor_from_assets = layout_mode in ('floor_perimeter_clean', 'no_floor_assets')
    floor_z = detect_floor_z(coords_32) if exclude_floor_from_assets else None

    layout_coords = coords_32[layout_mask]

    # Build part_layouts: [overall, layout, asset0, asset1, ...]
    all_coords = [coords_32]  # overall
    part_layouts = [slice(0, N_overall)]
    current_idx = N_overall

    # Insert layout as part_layouts[1]
    N_layout = layout_coords.shape[0]
    if N_layout > 0:
        all_coords.append(layout_coords)
        part_layouts.append(slice(current_idx, current_idx + N_layout))
        current_idx += N_layout
    else:
        part_layouts.append(slice(current_idx, current_idx))  # empty layout

    # Assets as part_layouts[2+]
    # Skip OBBs with 0 voxels to avoid batch ID mismatch in model forward
    valid_obb_indices = []
    for asset_idx, mask in enumerate(voxel_masks):
        # Optionally exclude floor-layer voxels from assets
        if exclude_floor_from_assets:
            floor_mask = coords_32[:, 3] == floor_z
            mask = mask & ~floor_mask

        asset_coords = coords_32[mask]
        n_asset = asset_coords.shape[0]
        if n_asset == 0:
            continue  # Skip empty OBBs entirely
        valid_obb_indices.append(asset_idx)
        all_coords.append(asset_coords)
        part_layouts.append(slice(current_idx, current_idx + n_asset))
        current_idx += n_asset

    all_coords_cat = torch.cat(all_coords, dim=0).to(device)
    N_total = all_coords_cat.shape[0]

    noise_feats = torch.randn(N_total, 32, device=device)
    noise_st = SparseTensor(
        coords=all_coords_cat,
        feats=noise_feats,
    )

    return noise_st, part_layouts, coords_32, valid_obb_indices


# ============================================================
# Phase 4: Stage 2-1 — Shape Generation
# ============================================================

@torch.no_grad()
def run_stage2_shape(
    args,
    samples: List[Tuple[str, str]],
    stage1_results: Dict,
    bbox_results: Dict,
    device: str = 'cuda',
) -> Dict[str, dict]:
    """Phase 4: Run shape generation with asset-aware attention."""
    print("\n" + "=" * 60)
    print("Phase 4: Stage 2-1 — Shape Generation")
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
    for scene_id, room_id in tqdm(samples, desc="Stage 2 Shape"):
        key = (scene_id, room_id)
        if key not in stage1_results or key not in bbox_results:
            continue

        out_dir = os.path.join(args.output_dir, scene_id, room_id)
        shape_path = os.path.join(out_dir, 'shape_latent.npz')

        if args.skip_existing and os.path.exists(shape_path):
            data = np.load(shape_path, allow_pickle=True)
            coords = torch.from_numpy(data['coords']).int()
            feats = torch.from_numpy(data['feats']).float()
            part_layouts_raw = data['part_layouts']
            part_layouts = [slice(int(s[0]), int(s[1])) for s in part_layouts_raw]
            results[key] = {
                'shape_latent': SparseTensor(coords=coords, feats=feats),
                'part_layouts': part_layouts,
            }
            continue

        try:
            voxel_64 = stage1_results[key]['voxel']
            bbox_data = bbox_results[key]
            obbs = bbox_data['obbs']
            camera_center = bbox_data['camera_center']

            noise_st, part_layouts, coords_32, valid_obb_indices = construct_stage2_input(
                voxel_64, obbs, device, layout_mode=args.layout_mode)

            if noise_st is None:
                print(f"  Skipping {scene_id}/{room_id}: no active voxels")
                continue

            # Filter OBBs to only those with voxels
            obbs_filtered = obbs[valid_obb_indices] if len(valid_obb_indices) > 0 else obbs[:0]

            cond_img = load_cubemap_images(args.data_dir, scene_id, room_id).unsqueeze(0).to(device)
            encoded_cond = erp_encoder(cond_img)
            neg_cond = torch.zeros_like(encoded_cond)

            obbs_tensor = torch.from_numpy(obbs_filtered).float()
            if obbs_tensor.shape[0] > 0:
                overlap_groups = compute_overlap_groups(obbs_tensor, margin=overlap_margin)
            else:
                overlap_groups = []

            num_parts = len(part_layouts)
            overall_voxel_coords = noise_st.coords[part_layouts[0], 1:4]
            # Extract layout voxel coords (part_layouts[1] = layout)
            layout_slice = part_layouts[1]
            layout_voxel_coords = noise_st.coords[layout_slice, 1:4] if (layout_slice.stop - layout_slice.start) > 0 else None
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

            results[key] = {
                'shape_latent': SparseTensor(
                    coords=shape_latent.coords.cpu(),
                    feats=shape_latent.feats.cpu(),
                ),
                'part_layouts': part_layouts,
                'obbs_filtered': obbs_filtered,
            }

        except Exception as e:
            print(f"  Error processing {scene_id}/{room_id}: {e}")
            import traceback
            traceback.print_exc()

    del denoiser, erp_encoder
    torch.cuda.empty_cache()

    print(f"  Stage 2 shape complete: {len(results)}/{len(samples)} samples")
    return results


# ============================================================
# Phase 5: Stage 2-2 — Texture Generation (optional)
# ============================================================

@torch.no_grad()
def run_stage2_texture(
    args,
    samples: List[Tuple[str, str]],
    bbox_results: Dict,
    shape_results: Dict,
    device: str = 'cuda',
) -> Dict[str, dict]:
    """Phase 5: Run texture generation with shape as concat_cond."""
    print("\n" + "=" * 60)
    print("Phase 5: Stage 2-2 — Texture Generation")
    print("=" * 60)

    with open(args.stage2_tex_config, 'r') as f:
        config = json.load(f)
    trainer_config = config['trainer']['args']
    dataset_config = config['dataset']['args']

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
    for scene_id, room_id in tqdm(samples, desc="Stage 2 Texture"):
        key = (scene_id, room_id)
        if key not in shape_results or key not in bbox_results:
            continue

        out_dir = os.path.join(args.output_dir, scene_id, room_id)
        tex_path = os.path.join(out_dir, 'texture_latent.npz')

        if args.skip_existing and os.path.exists(tex_path):
            shape_data = shape_results[key]
            tex_data_np = np.load(tex_path, allow_pickle=True)
            tex_coords = torch.from_numpy(tex_data_np['coords']).int()
            tex_feats = torch.from_numpy(tex_data_np['feats']).float()
            part_layouts_raw = tex_data_np['part_layouts']
            part_layouts = [slice(int(s[0]), int(s[1])) for s in part_layouts_raw]
            results[key] = {
                'tex_latent': SparseTensor(coords=tex_coords, feats=tex_feats),
                'shape_latent': shape_data['shape_latent'],
                'part_layouts': part_layouts,
            }
            continue

        try:
            shape_data = shape_results[key]
            shape_latent = shape_data['shape_latent']
            part_layouts = shape_data['part_layouts']
            bbox_data = bbox_results[key]
            camera_center = bbox_data['camera_center']
            # Use filtered OBBs from shape stage (only those with voxels)
            obbs_filtered = shape_data.get('obbs_filtered', bbox_data['obbs'])
            obbs_tensor = torch.from_numpy(obbs_filtered).float()

            # Shape sampler output is already in normalized space (same as training concat_cond).
            # Do NOT re-normalize — that would double-normalize.
            concat_cond = SparseTensor(
                coords=shape_latent.coords.to(device),
                feats=shape_latent.feats.clone().to(device),
            )

            tex_noise_feats = torch.randn(shape_latent.feats.shape[0], 32, device=device)
            tex_noise = SparseTensor(
                coords=shape_latent.coords.to(device),
                feats=tex_noise_feats,
            )

            cond_img = load_cubemap_images(args.data_dir, scene_id, room_id).unsqueeze(0).to(device)
            encoded_cond = erp_encoder(cond_img)
            neg_cond = torch.zeros_like(encoded_cond)

            if obbs_tensor.shape[0] > 0:
                overlap_groups = compute_overlap_groups(obbs_tensor, margin=overlap_margin)
            else:
                overlap_groups = []

            num_parts = len(part_layouts)
            overall_voxel_coords = tex_noise.coords[part_layouts[0], 1:4]
            # Extract layout voxel coords (part_layouts[1] = layout)
            layout_slice = part_layouts[1]
            layout_voxel_coords = tex_noise.coords[layout_slice, 1:4] if (layout_slice.stop - layout_slice.start) > 0 else None
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

            # Save texture latent NPZ
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(
                tex_path,
                coords=tex_latent.coords.cpu().numpy(),
                feats=tex_latent.feats.cpu().numpy(),
                part_layouts=np.array([(s.start, s.stop) for s in part_layouts]),
            )

            results[key] = {
                'tex_latent': SparseTensor(
                    coords=tex_latent.coords.cpu(),
                    feats=tex_latent.feats.cpu(),
                ),
                'shape_latent': shape_latent,
                'part_layouts': part_layouts,
            }

        except Exception as e:
            print(f"  Error processing {scene_id}/{room_id}: {e}")
            import traceback
            traceback.print_exc()

    del denoiser, erp_encoder
    torch.cuda.empty_cache()

    print(f"  Stage 2 texture complete: {len(results)}/{len(samples)} samples")
    return results


# ============================================================
# Phase 6: Decode & Save Meshes
# ============================================================

def inverse_normalize(z: SparseTensor, normalization: dict) -> SparseTensor:
    """Apply inverse normalization: z_original = z_normalized * std + mean."""
    mean = torch.tensor(normalization['mean']).reshape(1, -1).to(z.device)
    std = torch.tensor(normalization['std']).reshape(1, -1).to(z.device)
    return z.replace(feats=z.feats * std + mean)


def save_mesh_obj(vertices, faces, output_path):
    """Save mesh as OBJ file using trimesh."""
    if vertices is None or faces is None:
        return
    v = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
    f = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    mesh.export(output_path)


def save_mesh_glb(vertices, faces, output_path):
    """Save mesh as GLB with position-based vertex colors."""
    if vertices is None or faces is None:
        return
    v = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
    f = faces.cpu().numpy() if isinstance(faces, torch.Tensor) else faces
    colors = ((v + 0.5) * 255).clip(0, 255).astype(np.uint8)
    colors = np.column_stack([colors, np.full(len(colors), 255, dtype=np.uint8)])  # RGBA
    mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors)
    mesh.export(output_path)


@torch.no_grad()
def run_decode_meshes(
    args,
    samples: List[Tuple[str, str]],
    bbox_results: Dict,
    shape_results: Dict,
    texture_results: Optional[Dict] = None,
    device: str = 'cuda',
):
    """Phase 6: Decode latents to meshes and save as OBJ files."""
    print("\n" + "=" * 60)
    print("Phase 6: Decode & Save Meshes")
    print("=" * 60)

    # Load shape config for normalization
    with open(args.stage2_shape_config, 'r') as f:
        shape_config = json.load(f)
    shape_normalization = shape_config['dataset']['args'].get('normalization', None)
    # IMPORTANT: Use dataset resolution (512) for decoder, NOT denoiser resolution (32)
    # The decoder's flexible_dual_grid_to_mesh needs the full O-Voxel resolution
    data_resolution = shape_config['dataset']['args'].get('resolution', 512)
    shape_resolution = data_resolution

    # Load shape decoder
    pretrained_slat_dec = shape_config['dataset']['args'].get(
        'pretrained_slat_dec',
        'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16'
    )
    print(f"  Loading shape decoder from: {pretrained_slat_dec}")
    shape_dec = models.from_pretrained(pretrained_slat_dec)
    shape_dec.set_resolution(shape_resolution)
    shape_dec = shape_dec.to(device).eval()

    # Optionally load texture decoders
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

        # Build layout dict (same as erp_structured_latent.py)
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
    for scene_id, room_id in tqdm(samples, desc="Decoding meshes"):
        key = (scene_id, room_id)
        if key not in shape_results or key not in bbox_results:
            continue

        out_dir = os.path.join(args.output_dir, scene_id, room_id, 'meshes')
        scene_glb_path = os.path.join(out_dir, 'scene.glb')

        if args.skip_existing and os.path.exists(scene_glb_path):
            continue

        os.makedirs(out_dir, exist_ok=True)
        assets_dir = os.path.join(out_dir, 'assets')
        os.makedirs(assets_dir, exist_ok=True)

        shape_data = shape_results[key]
        shape_latent = shape_data['shape_latent']
        part_layouts = shape_data['part_layouts']
        bbox_data = bbox_results[key]

        use_texture = (args.enable_texture and texture_results and key in texture_results)

        try:
            if use_texture:
                from trellis2.representations import MeshWithVoxel
                tex_data = texture_results[key]
                tex_latent = tex_data['tex_latent']

                # Extract overall part
                overall_slice = part_layouts[0]
                overall_shape = SparseTensor(
                    coords=shape_latent.coords[overall_slice],
                    feats=shape_latent.feats[overall_slice],
                ).to(device)
                overall_tex = SparseTensor(
                    coords=tex_latent.coords[overall_slice],
                    feats=tex_latent.feats[overall_slice],
                ).to(device)

                # Inverse normalize
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
                    # Save scene as GLB with texture via o_voxel postprocess
                    try:
                        import o_voxel
                        attr_volume = vox[0].feats  # [N_vox, C] with layout
                        attr_coords = vox[0].coords[:, 1:]  # [N_vox, 3]
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
                        print(f"    Warning: GLB export failed: {e}")
                        # Fallback: save as GLB with vertex colors
                        save_mesh_glb(mesh[0].vertices, mesh[0].faces, scene_glb_path)

            else:
                # Shape-only mode
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

            # Decode layout (part_layouts[1])
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
                                import o_voxel
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
                    print(f"    Warning: layout decode failed: {e}")

            # Decode individual assets
            # part_layouts: [overall(0), layout(1), asset0(2), asset1(3), ...]
            asset_start = 2  # Assets start after overall and layout
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
                                import o_voxel
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

            num_decoded += 1

            # Save metadata
            metadata = {
                'scene_id': scene_id,
                'room_id': room_id,
                'bbox_source': bbox_data['source'],
                'num_assets': len(part_layouts) - 1,
                'asset_names': bbox_data['asset_names'],
                'texture_enabled': use_texture,
                'noise_mode': args.noise_mode,
                'sdedit_alpha': args.sdedit_alpha if args.noise_mode == 'sdedit' else None,
            }
            with open(os.path.join(args.output_dir, scene_id, room_id, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"  Error decoding {scene_id}/{room_id}: {e}")
            import traceback
            traceback.print_exc()

    # Free memory
    del shape_dec
    if pbr_dec is not None:
        del pbr_dec, tex_shape_dec
    torch.cuda.empty_cache()

    print(f"  Decoded {num_decoded}/{len(samples)} samples")


# ============================================================
# Visualization Helpers
# ============================================================

def _get_font():
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        small = font
    return font, small


def _ccm_colormap(coords: torch.Tensor) -> torch.Tensor:
    """Map voxel (x, y, z) to (R, G, B), normalized to actual occupied range per axis."""
    c = coords.float()
    c_min = c.min(dim=0).values
    c_max = c.max(dim=0).values
    span = (c_max - c_min).clamp(min=1.0)
    return (c - c_min) / span


def _height_colormap(coords: torch.Tensor) -> torch.Tensor:
    """Map voxel height (Z axis) to turbo colormap."""
    import matplotlib
    z = coords[:, 2].float()
    z_min, z_max = z.min(), z.max()
    if z_max - z_min < 1e-6:
        height = torch.zeros_like(z)
    else:
        height = (z - z_min) / (z_max - z_min)
    cmap = matplotlib.colormaps['turbo']
    colors_np = cmap(height.cpu().numpy())[:, :3]
    return torch.from_numpy(colors_np.astype(np.float32))


def _make_voxel_rep(coords, color, resolution):
    """Create a Voxel representation for rendering."""
    from trellis2.representations import Voxel
    return Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / resolution,
        coords=coords.cuda(),
        attrs=color.cuda(),
        layout={'color': slice(0, 3)},
    )


def _make_renderer(tile, near=0.8, far=1.6, ssaa=4):
    """Create a VoxelRenderer with standard settings."""
    from trellis2.renderers import VoxelRenderer
    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = tile
    renderer.rendering_options.near = near
    renderer.rendering_options.far = far
    renderer.rendering_options.ssaa = ssaa
    return renderer


def _render_single_view(renderer, rep, color, ext, intr, color_mode='ccm'):
    """Render a single view, returning [3, H, W] image."""
    from trellis2.renderers import VoxelRenderer
    res = renderer.render(rep, ext, intr, colors_overwrite=color.cuda())
    return res['color']


def _make_row_labels(labels, row_height, label_width=80):
    """Create a vertical strip with row labels as a PIL Image."""
    from PIL import Image, ImageDraw, ImageFont
    total_h = len(labels) * row_height
    strip = Image.new('RGB', (label_width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        tw = bbox[2] - bbox[0]
        x = (label_width - tw) // 2
        y = i * row_height + (row_height - th) // 2
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    return strip


def _make_label_strip(labels, tile_size, label_height=24):
    """Create a white strip with text labels as a PIL Image."""
    from PIL import Image, ImageDraw, ImageFont
    total_w = len(labels) * tile_size
    strip = Image.new('RGB', (total_w, label_height), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", label_height - 8)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        x = i * tile_size + (tile_size - tw) // 2
        draw.text((x, 2), label, fill=(0, 0, 0), font=font)
    return strip


@torch.no_grad()
def render_ss_exterior(voxel_grid, resolution=64, tile=512, color_mode='ccm'):
    """Render 4 exterior views of SS voxel grid arranged in 2x2 grid. Returns [3, 2*tile, 2*tile]."""
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    grid = voxel_grid.squeeze()
    coords = torch.nonzero(grid > 0, as_tuple=False)
    if coords.shape[0] == 0:
        return torch.zeros(3, 2 * tile, 2 * tile)

    if color_mode == 'height':
        color = _height_colormap(coords)
    else:
        color = _ccm_colormap(coords)
    rep = _make_voxel_rep(coords, color, resolution)
    renderer = _make_renderer(tile)

    yaw_offset = -16 / 180 * np.pi
    yaws = [i * np.pi / 2 + yaw_offset for i in range(4)]
    pitch = [20 / 180 * np.pi] * 4
    exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 30)

    image = torch.zeros(3, 2 * tile, 2 * tile).cuda()
    for j, (ext, intr) in enumerate(zip(exts, ints)):
        face = _render_single_view(renderer, rep, color, ext, intr, color_mode)
        r, c = j // 2, j % 2
        image[:, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = face
    return image.cpu()


@torch.no_grad()
def render_ss_topdown(voxel_grid, resolution=64, tile=512, color_mode='ccm'):
    """Render single top-down view of SS voxel grid. Returns [3, tile, tile]."""
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    grid = voxel_grid.squeeze()
    coords = torch.nonzero(grid > 0, as_tuple=False)
    if coords.shape[0] == 0:
        return torch.zeros(3, tile, tile)

    if color_mode == 'height':
        color = _height_colormap(coords)
    else:
        color = _ccm_colormap(coords)
    rep = _make_voxel_rep(coords, color, resolution)
    renderer = _make_renderer(tile)

    exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [90 / 180 * np.pi], 2, 30)
    face = _render_single_view(renderer, rep, color, exts[0], ints[0], color_mode)
    return face.cpu()


@torch.no_grad()
def render_ss_topdown_cam(voxel_grid, camera_center, resolution=64, tile=512, color_mode='ccm'):
    """Render top-down view with camera center overlay. Returns [3, tile, tile]."""
    from PIL import Image, ImageDraw
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    face = render_ss_topdown(voxel_grid, resolution, tile, color_mode)

    exts, ints_mat = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [90 / 180 * np.pi], 2, 30)
    cam_3d = torch.tensor(camera_center if isinstance(camera_center, list) else camera_center.tolist(),
                           dtype=torch.float32).cuda()
    point_h = torch.cat([cam_3d, torch.ones(1, device='cuda')])
    point_cam = exts[0] @ point_h
    point_proj = ints_mat[0] @ point_cam[:3]

    if point_proj[2].abs() > 1e-6:
        px = (point_proj[0] / point_proj[2]).item() * tile
        py = (point_proj[1] / point_proj[2]).item() * tile
        if -20 < px < tile + 20 and -20 < py < tile + 20:
            img_np = (face.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)
            r = 6
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 255, 255), outline=(255, 255, 255), width=2)
            face = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    return face


@torch.no_grad()
def render_ss_interior(voxel_grid, camera_center, resolution=64, tile=512, color_mode='ccm'):
    """Render 6 interior cubemap faces from camera center. Returns [6, 3, tile, tile]."""
    import utils3d.torch
    from trellis2.representations import Voxel

    grid = voxel_grid.squeeze()
    coords = torch.nonzero(grid > 0, as_tuple=False)
    if coords.shape[0] == 0:
        return torch.zeros(6, 3, tile, tile)

    world_scale = 10.0
    if color_mode == 'height':
        color = _height_colormap(coords)
    else:
        color = _ccm_colormap(coords)

    rep = Voxel(
        origin=[-0.5 * world_scale, -0.5 * world_scale, -0.5 * world_scale],
        voxel_size=world_scale / resolution,
        coords=coords.cuda(),
        attrs=color.cuda(),
        layout={'color': slice(0, 3)},
    )
    renderer = _make_renderer(tile, near=0.01 * world_scale, far=2.0 * world_scale)

    cam_np = camera_center.numpy() if isinstance(camera_center, torch.Tensor) else camera_center
    cam = torch.tensor(cam_np, dtype=torch.float32).cuda() * world_scale
    fov = torch.deg2rad(torch.tensor(120.0)).cuda()

    face_dirs = [
        [0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
    ]

    images = []
    for fi, fd in enumerate(face_dirs):
        look_at = cam + torch.tensor(fd, dtype=torch.float32).cuda()
        if fi == 4:
            up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32).cuda()
        elif fi == 5:
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()
        else:
            up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda()
        ext = utils3d.torch.extrinsics_look_at(cam, look_at, up)
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        face = _render_single_view(renderer, rep, color, ext, intr, color_mode)
        images.append(face.cpu())

    return torch.stack(images)


@torch.no_grad()
def load_gt_from_latent(data_dir, scene_id, room_id, shape_dec, shape_normalization,
                        tex_shape_dec=None, pbr_dec=None, tex_normalization=None,
                        tex_shape_normalization=None, tex_layout=None,
                        resolution=512, device='cuda'):
    """Load GT mesh by decoding GT latents — ensures coordinate alignment with predictions."""
    from trellis2.representations import MeshWithVoxel

    latent_dir = os.path.join(data_dir, scene_id, room_id,
                              'shape_latents', 'shape_enc_next_dc_f16c32_fp16_512')
    gt_path = os.path.join(latent_dir, 'full_room_wo_ceiling.npz')
    if not os.path.exists(gt_path):
        return None, None

    try:
        data = np.load(gt_path)
        coords = torch.from_numpy(data['coords']).int()
        feats = torch.from_numpy(data['feats']).float()
        batch_idx = torch.zeros(coords.shape[0], 1, dtype=torch.int32)
        coords_4d = torch.cat([batch_idx, coords], dim=1)
        # GT NPZ features are in original space (not normalized).
        # Decoders expect original space, so do NOT inverse_normalize.
        gt_shape_z = SparseTensor(coords=coords_4d, feats=feats).to(device)

        geo_reps = shape_dec(gt_shape_z)
        geo_mesh = geo_reps[0] if geo_reps else None

        # Optionally decode texture
        tex_mesh = None
        if tex_shape_dec is not None and pbr_dec is not None:
            tex_latent_dir = os.path.join(data_dir, scene_id, room_id,
                                          'pbr_latents', 'tex_enc_next_dc_f16c32_fp16_512')
            tex_path = os.path.join(tex_latent_dir, 'full_room_wo_ceiling.npz')
            if os.path.exists(tex_path):
                tex_data = np.load(tex_path)
                tex_feats = torch.from_numpy(tex_data['feats']).float()
                # GT texture feats are already in original space
                gt_tex_z = SparseTensor(coords=coords_4d.clone(), feats=tex_feats).to(device)

                # GT shape feats for texture decoder — already in original space
                gt_shape_for_tex = SparseTensor(
                    coords=coords_4d.clone(),
                    feats=torch.from_numpy(data['feats']).float(),
                ).to(device)

                mesh_list, subs = tex_shape_dec(gt_shape_for_tex, return_subs=True)
                vox = pbr_dec(gt_tex_z, guide_subs=subs) * 0.5 + 0.5
                if mesh_list and len(mesh_list) > 0:
                    tex_mesh = MeshWithVoxel(
                        mesh_list[0].vertices, mesh_list[0].faces,
                        origin=[-0.5, -0.5, -0.5], voxel_size=1 / resolution,
                        coords=vox[0].coords[:, 1:], attrs=vox[0].feats,
                        voxel_shape=torch.Size([*vox[0].shape, *vox[0].spatial_shape]),
                        layout=tex_layout,
                    )

        return geo_mesh, tex_mesh
    except Exception as e:
        print(f"  Warning: Failed to decode GT latent for {scene_id}/{room_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def draw_obb_topdown(ax, obb, color='green', alpha=0.4, label=None):
    """Draw a single OBB on a matplotlib axis (top-down XY plane)."""
    import matplotlib.pyplot as plt
    cx, cy = obb[0], obb[1]
    sx, sy = obb[3], obb[4]
    yaw = obb[6]
    cos_a = math.cos(yaw)
    sin_a = math.sin(yaw)
    hw, hh = sx / 2, sy / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    world_corners = [
        [cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a]
        for lx, ly in corners
    ]
    polygon = plt.Polygon(world_corners, closed=True,
                           facecolor=color, edgecolor=color,
                           alpha=alpha, linewidth=1)
    ax.add_patch(polygon)
    if label:
        ax.text(cx, cy, label[:15], ha='center', va='center', fontsize=5, color='white')


# ============================================================
# PBR Mesh Visualization (for textured meshes)
# ============================================================

@torch.no_grad()
def render_mesh_exterior(reps, resolution=32, tile=512, use_pbr=True):
    """Render 4 exterior views of decoded mesh reps. Returns [3, 2*tile, 2*tile]."""
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, get_renderer

    if not reps or reps[0] is None:
        return torch.zeros(3, 2 * tile, 2 * tile)

    yaw_offset = -16 / 180 * np.pi
    yaws = [i * np.pi / 2 + yaw_offset for i in range(4)]
    pitch = [20 / 180 * np.pi] * 4
    exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 30)

    rep = reps[0]
    from trellis2.representations import MeshWithVoxel
    if use_pbr and isinstance(rep, MeshWithVoxel):
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        import cv2
        from trellis2.renderers import PbrMeshRenderer, EnvMap
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/interior.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        image = torch.zeros(3, 2 * tile, 2 * tile).cuda()
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            res = renderer.render(rep, ext, intr, envmap=envmap)
            r, c = j // 2, j % 2
            image[:, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = res['base_color']
        return image.cpu()
    else:
        renderer = get_renderer(rep)
        renderer.rendering_options.resolution = tile
        image = torch.zeros(3, 2 * tile, 2 * tile).cuda()
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            res = renderer.render(rep, ext, intr)
            r, c = j // 2, j % 2
            image[:, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = res['normal']
        return image.cpu()


@torch.no_grad()
def render_mesh_topdown(reps, resolution=32, tile=512, use_pbr=True):
    """Render single top-down view of decoded mesh. Returns [3, tile, tile]."""
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, get_renderer

    if not reps or reps[0] is None:
        return torch.zeros(3, tile, tile)

    exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [90 / 180 * np.pi], 2, 30)
    rep = reps[0]

    from trellis2.representations import MeshWithVoxel
    if use_pbr and isinstance(rep, MeshWithVoxel):
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        import cv2
        from trellis2.renderers import PbrMeshRenderer, EnvMap
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/interior.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        res = renderer.render(rep, exts[0], ints[0], envmap=envmap)
        return res['base_color'].cpu()
    else:
        renderer = get_renderer(rep)
        renderer.rendering_options.resolution = tile
        res = renderer.render(rep, exts[0], ints[0])
        return res['normal'].cpu()


@torch.no_grad()
def render_mesh_topdown_cam(reps, camera_center, resolution=32, tile=512, use_pbr=True):
    """Render top-down mesh view with camera center overlay. Returns [3, tile, tile]."""
    from PIL import Image, ImageDraw
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    face = render_mesh_topdown(reps, resolution, tile, use_pbr)

    exts, ints_mat = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [90 / 180 * np.pi], 2, 30)
    cam_np = camera_center.numpy() if isinstance(camera_center, torch.Tensor) else camera_center
    cam_3d = torch.tensor(cam_np, dtype=torch.float32).cuda()
    point_h = torch.cat([cam_3d, torch.ones(1, device='cuda')])
    point_cam = exts[0] @ point_h
    point_proj = ints_mat[0] @ point_cam[:3]

    if point_proj[2].abs() > 1e-6:
        px = (point_proj[0] / point_proj[2]).item() * tile
        py = (point_proj[1] / point_proj[2]).item() * tile
        if -20 < px < tile + 20 and -20 < py < tile + 20:
            img_np = (face.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(pil_img)
            r = 8
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 255, 255), outline=(255, 255, 255), width=2)
            face = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    return face


@torch.no_grad()
def render_mesh_interior(reps, camera_center, cubemap_images=None, resolution=32, tile=512, use_pbr=True):
    """
    Render 6 interior cubemap faces of decoded mesh from camera center.

    Returns composite: [3, label_h + 2*tile, 6*tile] if cubemap_images provided,
    else [6, 3, tile, tile].
    """
    import utils3d.torch
    from trellis2.utils.render_utils import get_renderer

    if not reps or reps[0] is None:
        if cubemap_images is not None:
            return torch.zeros(3, tile * 2 + 24, 6 * tile)
        return torch.zeros(6, 3, tile, tile)

    rep = reps[0]
    from trellis2.representations import MeshWithVoxel
    if use_pbr and isinstance(rep, MeshWithVoxel):
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        import cv2
        from trellis2.renderers import PbrMeshRenderer, EnvMap
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.ssaa = 4
        renderer.rendering_options.near = 0.01
        renderer.rendering_options.far = 2.0
        renderer.rendering_options.peel_layers = 8
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/interior.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        render_key = 'base_color'
    else:
        renderer = get_renderer(rep)
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.ssaa = 4
        renderer.rendering_options.near = 0.01
        renderer.rendering_options.far = 2.0
        envmap = None
        render_key = 'normal'

    cam_np = camera_center.numpy() if isinstance(camera_center, torch.Tensor) else camera_center
    cam = torch.tensor(cam_np, dtype=torch.float32).cuda()
    fov = torch.deg2rad(torch.tensor(120.0)).cuda()

    face_dirs = [
        [0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
    ]
    face_labels = ['front (+Y)', 'right (+X)', 'back (-Y)', 'left (-X)', 'top (+Z)', 'bottom (-Z)']

    rendered_faces = []
    for fi, fd in enumerate(face_dirs):
        look_at = cam + torch.tensor(fd, dtype=torch.float32).cuda()
        if fi == 4:
            up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32).cuda()
        elif fi == 5:
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()
        else:
            up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda()
        ext = utils3d.torch.extrinsics_look_at(cam, look_at, up)
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        if envmap is not None:
            res = renderer.render(rep, ext, intr, envmap=envmap)
        else:
            res = renderer.render(rep, ext, intr)
        rendered_faces.append(res[render_key].cpu())

    rendered_stack = torch.stack(rendered_faces)  # [6, 3, tile, tile]

    if cubemap_images is not None:
        # Create composite: labels + cubemap row + rendered row (with row labels)
        label_h = max(20, tile // 10)
        label_strip = _make_label_strip(face_labels, tile, label_h)
        label_tensor = torch.tensor(np.array(label_strip)).permute(2, 0, 1).float() / 255.0

        # cubemap_images is [6, 3, H, W] — treat 6 faces as batch dim for F.interpolate
        cubemap_resized = F.interpolate(
            cubemap_images, size=(tile, tile), mode='bilinear', align_corners=False,
        )  # [6, 3, tile, tile]

        row1 = torch.cat([cubemap_resized[j] for j in range(6)], dim=2)  # [3, tile, 6*tile]
        row2 = torch.cat([rendered_faces[j] for j in range(6)], dim=2)   # [3, tile, 6*tile]

        # Add row labels
        row_label_w = 80
        row_labels = _make_row_labels(['Input', 'Predicted'], tile, row_label_w)
        row_labels_t = torch.tensor(np.array(row_labels)).permute(2, 0, 1).float() / 255.0
        label_pad = torch.ones(3, label_h, row_label_w)
        label_full = torch.cat([label_pad, label_tensor], dim=2)
        rows = torch.cat([row1, row2], dim=1)
        rows_full = torch.cat([row_labels_t, rows], dim=2)
        composite = torch.cat([label_full, rows_full], dim=1)
        return composite

    return rendered_stack


def decode_scene_rep(
    shape_latent, tex_latent, part_layouts,
    shape_dec, tex_shape_dec, pbr_dec,
    shape_normalization, tex_normalization, tex_shape_normalization,
    tex_layout, resolution, device, use_texture=False,
):
    """Decode overall scene into a representation for rendering."""
    from trellis2.representations import MeshWithVoxel

    overall_slice = part_layouts[0]
    overall_shape = SparseTensor(
        coords=shape_latent.coords[overall_slice],
        feats=shape_latent.feats[overall_slice],
    ).to(device)

    if use_texture and tex_latent is not None:
        overall_tex = SparseTensor(
            coords=tex_latent.coords[overall_slice],
            feats=tex_latent.feats[overall_slice],
        ).to(device)

        if tex_shape_normalization:
            s_mean = torch.tensor(tex_shape_normalization['mean']).reshape(1, -1).to(device)
            s_std = torch.tensor(tex_shape_normalization['std']).reshape(1, -1).to(device)
            overall_shape = overall_shape.replace(feats=overall_shape.feats * s_std + s_mean)
        if tex_normalization:
            overall_tex = inverse_normalize(overall_tex, tex_normalization)

        mesh, subs = tex_shape_dec(overall_shape, return_subs=True)
        vox = pbr_dec(overall_tex, guide_subs=subs) * 0.5 + 0.5
        if mesh and len(mesh) > 0:
            rep = MeshWithVoxel(
                mesh[0].vertices, mesh[0].faces,
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1 / resolution,
                coords=vox[0].coords[:, 1:],
                attrs=vox[0].feats,
                voxel_shape=torch.Size([*vox[0].shape, *vox[0].spatial_shape]),
                layout=tex_layout,
            )
            return [rep]
    else:
        if shape_normalization:
            overall_shape = inverse_normalize(overall_shape, shape_normalization)
        reps = shape_dec(overall_shape)
        if reps and len(reps) > 0:
            return reps

    return []


# ============================================================
# Visualization (Main)
# ============================================================

@torch.no_grad()
def run_visualization(
    args,
    samples: List[Tuple[str, str]],
    stage1_results: Dict,
    bbox_results: Dict,
    shape_results: Dict,
    texture_results: Optional[Dict] = None,
    device: str = 'cuda',
):
    """Generate comprehensive visualization images."""
    print("\n" + "=" * 60)
    print("Visualization")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    num_vis = len(samples) if args.num_vis < 0 else min(args.num_vis, len(samples))
    if num_vis == 0:
        return

    vis_samples = samples[:num_vis]

    # Load decoders for mesh rendering visualization
    shape_dec = None
    tex_shape_dec = None
    pbr_dec = None
    shape_normalization = None
    tex_normalization = None
    tex_shape_normalization = None
    tex_layout = None
    shape_resolution = 512  # default; overwritten below if shape config is loaded

    has_shape = any(k in shape_results for _, _ in [])  # lazy check below
    need_mesh_vis = any((s, r) in shape_results for s, r in vis_samples)

    if need_mesh_vis:
        with open(args.stage2_shape_config, 'r') as f:
            shape_config = json.load(f)
        shape_normalization = shape_config['dataset']['args'].get('normalization', None)
        shape_resolution = shape_config['dataset']['args'].get('resolution', 512)

        pretrained_slat_dec = shape_config['dataset']['args'].get(
            'pretrained_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
        shape_dec = models.from_pretrained(pretrained_slat_dec)
        shape_dec.set_resolution(shape_resolution)
        shape_dec = shape_dec.to(device).eval()

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
                'pretrained_pbr_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16')
            pretrained_shape_slat_dec = tex_config['dataset']['args'].get(
                'pretrained_shape_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
            tex_shape_dec = models.from_pretrained(pretrained_shape_slat_dec)
            tex_shape_dec.set_resolution(shape_resolution)
            tex_shape_dec = tex_shape_dec.to(device).eval()
            pbr_dec = models.from_pretrained(pretrained_pbr_dec)
            pbr_dec = pbr_dec.to(device).eval()

    def _save_img(tensor, path):
        """Save [3, H, W] float tensor as PNG."""
        img = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(path)

    save_concat = getattr(args, 'save_concat', False)

    for scene_id, room_id in tqdm(vis_samples, desc="Visualizing"):
        key = (scene_id, room_id)
        pred_dir = os.path.join(args.output_dir, scene_id, room_id, 'vis_pred')
        os.makedirs(pred_dir, exist_ok=True)
        if save_concat:
            concat_dir = os.path.join(args.output_dir, scene_id, room_id, 'vis_concat')
            os.makedirs(concat_dir, exist_ok=True)

        camera_center = None
        if key in bbox_results:
            camera_center = bbox_results[key]['camera_center']

        # ── 1. SS Voxel Visualization (CCM + Height colormaps) ──
        if key in stage1_results:
            voxel = stage1_results[key]['voxel']
            gt_voxel = load_gt_voxel_grid(args.data_dir, scene_id, room_id)

            for color_mode in ['ccm', 'height']:
                # Exterior
                pred_ext = render_ss_exterior(voxel, tile=512, color_mode=color_mode)
                _save_img(pred_ext, os.path.join(pred_dir, f'ss_exterior_{color_mode}.png'))
                if save_concat and gt_voxel is not None:
                    gt_ext = render_ss_exterior(gt_voxel, tile=512, color_mode=color_mode)
                    label = _make_label_strip(['GT', 'Predicted'], 1024, 24)
                    label_t = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
                    combined = torch.cat([label_t, torch.cat([gt_ext, pred_ext], dim=2)], dim=1)
                    _save_img(combined, os.path.join(concat_dir, f'ss_exterior_{color_mode}.png'))

                # Topdown
                pred_td = render_ss_topdown(voxel, tile=512, color_mode=color_mode)
                _save_img(pred_td, os.path.join(pred_dir, f'ss_topdown_{color_mode}.png'))
                if save_concat and gt_voxel is not None:
                    gt_td = render_ss_topdown(gt_voxel, tile=512, color_mode=color_mode)
                    label = _make_label_strip(['GT', 'Predicted'], 512, 24)
                    label_t = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
                    combined_td = torch.cat([label_t, torch.cat([gt_td, pred_td], dim=2)], dim=1)
                    _save_img(combined_td, os.path.join(concat_dir, f'ss_topdown_{color_mode}.png'))

                # Topdown with camera center
                if camera_center is not None:
                    pred_td_cam = render_ss_topdown_cam(voxel, camera_center, tile=512, color_mode=color_mode)
                    _save_img(pred_td_cam, os.path.join(pred_dir, f'ss_topdown_cam_{color_mode}.png'))
                    if save_concat and gt_voxel is not None:
                        gt_td_cam = render_ss_topdown_cam(gt_voxel, camera_center, tile=512, color_mode=color_mode)
                        label = _make_label_strip(['GT', 'Predicted'], 512, 24)
                        label_t = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
                        combined_td_cam = torch.cat([label_t, torch.cat([gt_td_cam, pred_td_cam], dim=2)], dim=1)
                        _save_img(combined_td_cam, os.path.join(concat_dir, f'ss_topdown_cam_{color_mode}.png'))

            # Interior
            if camera_center is not None:
                for color_mode in ['ccm', 'height']:
                    pred_interior = render_ss_interior(voxel, camera_center, tile=512, color_mode=color_mode)
                    # Pred-only: 6 faces in a row
                    pred_row = torch.cat([pred_interior[j] for j in range(6)], dim=2)
                    _save_img(pred_row, os.path.join(pred_dir, f'ss_interior_{color_mode}.png'))

                    if save_concat:
                        cubemap_imgs = load_cubemap_images(args.data_dir, scene_id, room_id)
                        face_labels = ['front (+Y)', 'right (+X)', 'back (-Y)', 'left (-X)', 'top (+Z)', 'bottom (-Z)']
                        label_h = 24
                        label_strip = _make_label_strip(face_labels, 512, label_h)
                        label_t = torch.tensor(np.array(label_strip)).permute(2, 0, 1).float() / 255.0
                        cubemap_resized = F.interpolate(cubemap_imgs, size=(512, 512), mode='bilinear', align_corners=False)
                        row_input = torch.cat([cubemap_resized[j] for j in range(6)], dim=2)
                        row_label_w = 80

                        if gt_voxel is not None:
                            gt_interior = render_ss_interior(gt_voxel, camera_center, tile=512, color_mode=color_mode)
                            row_gt = torch.cat([gt_interior[j] for j in range(6)], dim=2)
                            row_labels = _make_row_labels(['Input', 'GT', 'Predicted'], 512, row_label_w)
                            row_labels_t = torch.tensor(np.array(row_labels)).permute(2, 0, 1).float() / 255.0
                            label_pad = torch.ones(3, label_h, row_label_w)
                            label_full = torch.cat([label_pad, label_t], dim=2)
                            rows = torch.cat([row_input, row_gt, pred_row], dim=1)
                            rows_full = torch.cat([row_labels_t, rows], dim=2)
                            composite = torch.cat([label_full, rows_full], dim=1)
                        else:
                            row_labels = _make_row_labels(['Input', 'Predicted'], 512, row_label_w)
                            row_labels_t = torch.tensor(np.array(row_labels)).permute(2, 0, 1).float() / 255.0
                            label_pad = torch.ones(3, label_h, row_label_w)
                            label_full = torch.cat([label_pad, label_t], dim=2)
                            rows = torch.cat([row_input, pred_row], dim=1)
                            rows_full = torch.cat([row_labels_t, rows], dim=2)
                            composite = torch.cat([label_full, rows_full], dim=1)
                        _save_img(composite, os.path.join(concat_dir, f'ss_interior_{color_mode}.png'))

        # ── 2. BBox Visualization (GT + Predicted together) ──
        if key in bbox_results:
            bbox_data = bbox_results[key]
            obbs = bbox_data['obbs']

            fig, axes = plt.subplots(1, 2 if args.bbox_mode == 'predicted' else 1,
                                     figsize=(16 if args.bbox_mode == 'predicted' else 8, 8), dpi=100)
            if not isinstance(axes, np.ndarray):
                axes = [axes]

            # Draw current mode's bboxes
            color = 'green' if bbox_data['source'] == 'gt' else 'red'
            for i, obb in enumerate(obbs):
                name = bbox_data['asset_names'][i] if i < len(bbox_data['asset_names']) else ''
                draw_obb_topdown(axes[0], obb, color=color, label=name)
            axes[0].set_xlim(-0.55, 0.55)
            axes[0].set_ylim(-0.55, 0.55)
            axes[0].set_aspect('equal')
            axes[0].grid(True, alpha=0.2)
            axes[0].set_title(f'{bbox_data["source"].upper()} BBox: {len(obbs)} objects')

            # If predicted mode, also show GT for comparison
            if args.bbox_mode == 'predicted' and len(axes) > 1:
                gt_bbox_data = load_gt_bboxes(args.data_dir, scene_id, room_id)
                if gt_bbox_data is not None:
                    for i, obb in enumerate(gt_bbox_data['obbs']):
                        name = gt_bbox_data['asset_names'][i] if i < len(gt_bbox_data['asset_names']) else ''
                        draw_obb_topdown(axes[1], obb, color='green', label=name)
                    # Overlay predicted on GT axis too
                    for i, obb in enumerate(obbs):
                        draw_obb_topdown(axes[1], obb, color='red', alpha=0.25)
                axes[1].set_xlim(-0.55, 0.55)
                axes[1].set_ylim(-0.55, 0.55)
                axes[1].set_aspect('equal')
                axes[1].grid(True, alpha=0.2)
                axes[1].set_title(f'GT (green) + Pred (red)')

            plt.suptitle(f'{scene_id}/{room_id}', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(pred_dir, 'bbox_topdown.png'), bbox_inches='tight')
            plt.close(fig)

        # ── 3. Input Cubemap ──
        try:
            cond = load_cubemap_images(args.data_dir, scene_id, room_id)
            face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=80)
            for i, (face, name) in enumerate(zip(cond, face_names)):
                ax = axes[i // 3, i % 3]
                ax.imshow(face.permute(1, 2, 0).numpy())
                ax.set_title(name)
                ax.axis('off')
            plt.suptitle(f'Input Cubemap: {scene_id}/{room_id}', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(pred_dir, 'cubemap_input.png'), bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass

        # ── 4. Final Mesh Visualization (after Stage 2) ──
        # Geometry views (normal maps) are always generated.
        # Texture views (base_color) are generated only when texture is enabled.
        if key in shape_results and shape_dec is not None:
            try:
                shape_data = shape_results[key]
                tex_data = texture_results.get(key) if texture_results else None
                tex_latent = tex_data['tex_latent'] if tex_data else None
                use_texture = tex_latent is not None and tex_shape_dec is not None

                # Load GT by decoding latents (only needed for concat comparison)
                gt_geo_mesh, gt_tex_mesh = None, None
                if save_concat:
                    gt_geo_mesh, gt_tex_mesh = load_gt_from_latent(
                        args.data_dir, scene_id, room_id,
                        shape_dec, shape_normalization,
                        tex_shape_dec=tex_shape_dec, pbr_dec=pbr_dec,
                        tex_normalization=tex_normalization,
                        tex_shape_normalization=tex_shape_normalization,
                        tex_layout=tex_layout,
                        resolution=shape_resolution, device=device,
                    )

                # Load cubemap images once for interior views
                cubemap_imgs = None
                if camera_center is not None:
                    cubemap_imgs = load_cubemap_images(args.data_dir, scene_id, room_id)

                # --- Geometry visualization (normal maps, always) ---
                geo_reps = decode_scene_rep(
                    shape_latent=shape_data['shape_latent'],
                    tex_latent=None,
                    part_layouts=shape_data['part_layouts'],
                    shape_dec=shape_dec,
                    tex_shape_dec=None, pbr_dec=None,
                    shape_normalization=shape_normalization,
                    tex_normalization=None,
                    tex_shape_normalization=None,
                    tex_layout=None,
                    resolution=shape_resolution,
                    device=device,
                    use_texture=False,
                )

                if geo_reps:
                    gt_geo_reps = [gt_geo_mesh] if gt_geo_mesh is not None else None

                    # Geometry Exterior
                    pred_ext = render_mesh_exterior(geo_reps, shape_resolution, tile=512, use_pbr=False)
                    _save_img(pred_ext, os.path.join(pred_dir, 'geometry_exterior.png'))
                    if save_concat and gt_geo_reps:
                        gt_ext = render_mesh_exterior(gt_geo_reps, shape_resolution, tile=512, use_pbr=False)
                        label = _make_label_strip(['GT', 'Predicted'], 1024, 24)
                        label_t = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
                        combined = torch.cat([label_t, torch.cat([gt_ext, pred_ext], dim=2)], dim=1)
                        _save_img(combined, os.path.join(concat_dir, 'geometry_exterior.png'))

                    # Geometry Topdown
                    pred_td = render_mesh_topdown(geo_reps, shape_resolution, tile=512, use_pbr=False)
                    _save_img(pred_td, os.path.join(pred_dir, 'geometry_topdown.png'))
                    if save_concat and gt_geo_reps:
                        gt_td = render_mesh_topdown(gt_geo_reps, shape_resolution, tile=512, use_pbr=False)
                        label = _make_label_strip(['GT', 'Predicted'], 512, 24)
                        label_t = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
                        combined_td = torch.cat([label_t, torch.cat([gt_td, pred_td], dim=2)], dim=1)
                        _save_img(combined_td, os.path.join(concat_dir, 'geometry_topdown.png'))

                    # Geometry Topdown with camera center
                    if camera_center is not None:
                        pred_td_cam = render_mesh_topdown_cam(
                            geo_reps, camera_center, shape_resolution, tile=512, use_pbr=False)
                        _save_img(pred_td_cam, os.path.join(pred_dir, 'geometry_topdown_cam.png'))
                        if save_concat and gt_geo_reps:
                            gt_td_cam = render_mesh_topdown_cam(
                                gt_geo_reps, camera_center, shape_resolution, tile=512, use_pbr=False)
                            label = _make_label_strip(['GT', 'Predicted'], 512, 24)
                            label_t = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
                            combined_td_cam = torch.cat(
                                [label_t, torch.cat([gt_td_cam, pred_td_cam], dim=2)], dim=1)
                            _save_img(combined_td_cam, os.path.join(concat_dir, 'geometry_topdown_cam.png'))

                    # Geometry Interior
                    if camera_center is not None and cubemap_imgs is not None:
                        # Pred-only (no cubemap row)
                        pred_interior = render_mesh_interior(
                            geo_reps, camera_center, cubemap_images=None,
                            resolution=shape_resolution, tile=512, use_pbr=False)
                        pred_row = torch.cat([pred_interior[j] for j in range(6)], dim=2)
                        _save_img(pred_row, os.path.join(pred_dir, 'geometry_interior.png'))
                        if save_concat:
                            # Concat with cubemap input row
                            interior_concat = render_mesh_interior(
                                geo_reps, camera_center, cubemap_images=cubemap_imgs,
                                resolution=shape_resolution, tile=512, use_pbr=False)
                            _save_img(interior_concat, os.path.join(concat_dir, 'geometry_interior.png'))

                # --- Texture visualization (base_color, only when texture enabled) ---
                if use_texture:
                    tex_reps = decode_scene_rep(
                        shape_latent=shape_data['shape_latent'],
                        tex_latent=tex_latent,
                        part_layouts=shape_data['part_layouts'],
                        shape_dec=shape_dec,
                        tex_shape_dec=tex_shape_dec,
                        pbr_dec=pbr_dec,
                        shape_normalization=shape_normalization,
                        tex_normalization=tex_normalization,
                        tex_shape_normalization=tex_shape_normalization,
                        tex_layout=tex_layout,
                        resolution=shape_resolution,
                        device=device,
                        use_texture=True,
                    )

                    if tex_reps:
                        gt_tex_reps = [gt_tex_mesh] if gt_tex_mesh is not None else None

                        # Texture Exterior
                        pred_ext = render_mesh_exterior(tex_reps, shape_resolution, tile=512, use_pbr=True)
                        _save_img(pred_ext, os.path.join(pred_dir, 'texture_exterior.png'))
                        if save_concat and gt_tex_reps:
                            gt_ext = render_mesh_exterior(gt_tex_reps, shape_resolution, tile=512, use_pbr=True)
                            label = _make_label_strip(['GT', 'Predicted'], 1024, 24)
                            label_t = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
                            combined = torch.cat([label_t, torch.cat([gt_ext, pred_ext], dim=2)], dim=1)
                            _save_img(combined, os.path.join(concat_dir, 'texture_exterior.png'))

                        # Texture Topdown
                        pred_td = render_mesh_topdown(tex_reps, shape_resolution, tile=512, use_pbr=True)
                        _save_img(pred_td, os.path.join(pred_dir, 'texture_topdown.png'))
                        if save_concat and gt_tex_reps:
                            gt_td = render_mesh_topdown(gt_tex_reps, shape_resolution, tile=512, use_pbr=True)
                            label = _make_label_strip(['GT', 'Predicted'], 512, 24)
                            label_t = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
                            combined_td = torch.cat([label_t, torch.cat([gt_td, pred_td], dim=2)], dim=1)
                            _save_img(combined_td, os.path.join(concat_dir, 'texture_topdown.png'))

                        # Texture Topdown with camera center
                        if camera_center is not None:
                            pred_td_cam = render_mesh_topdown_cam(
                                tex_reps, camera_center, shape_resolution, tile=512, use_pbr=True)
                            _save_img(pred_td_cam, os.path.join(pred_dir, 'texture_topdown_cam.png'))
                            if save_concat and gt_tex_reps:
                                gt_td_cam = render_mesh_topdown_cam(
                                    gt_tex_reps, camera_center, shape_resolution, tile=512, use_pbr=True)
                                label = _make_label_strip(['GT', 'Predicted'], 512, 24)
                                label_t = torch.tensor(np.array(label)).permute(2, 0, 1).float() / 255.0
                                combined_td_cam = torch.cat(
                                    [label_t, torch.cat([gt_td_cam, pred_td_cam], dim=2)], dim=1)
                                _save_img(combined_td_cam, os.path.join(concat_dir, 'texture_topdown_cam.png'))

                        # Texture Interior
                        if camera_center is not None and cubemap_imgs is not None:
                            pred_interior = render_mesh_interior(
                                tex_reps, camera_center, cubemap_images=None,
                                resolution=shape_resolution, tile=512, use_pbr=True)
                            pred_row = torch.cat([pred_interior[j] for j in range(6)], dim=2)
                            _save_img(pred_row, os.path.join(pred_dir, 'texture_interior.png'))
                            if save_concat:
                                interior_concat = render_mesh_interior(
                                    tex_reps, camera_center, cubemap_images=cubemap_imgs,
                                    resolution=shape_resolution, tile=512, use_pbr=True)
                                _save_img(interior_concat, os.path.join(concat_dir, 'texture_interior.png'))

            except Exception as e:
                print(f"  Error rendering mesh vis for {scene_id}/{room_id}: {e}")
                import traceback
                traceback.print_exc()

    # Free visualization decoders
    if shape_dec is not None:
        del shape_dec
    if tex_shape_dec is not None:
        del tex_shape_dec
    if pbr_dec is not None:
        del pbr_dec
    torch.cuda.empty_cache()

    print(f"  Visualization complete for {num_vis} samples")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='InSpace Full Evaluation Pipeline (Stage 1 + BBox + Stage 2)')

    # Paths
    parser.add_argument('--data_dir', type=str,
                        default='datasets/ERP_3D_FRONT_test')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--gpu_id', type=int, default=0)

    # Stage 1
    parser.add_argument('--stage1_config', type=str,
                        default='configs/gen/erp_ss_flow_img_dit_L_16l8_bf16_spatial.json')
    parser.add_argument('--stage1_ckpt_dir', type=str,
                        default='ckpts/erp_ss_flow_img_dit_L_16l8_bf16_spatial')
    parser.add_argument('--noise_mode', type=str, default='random',
                        choices=['random', 'sdedit'])
    parser.add_argument('--sdedit_alpha', type=float, default=0.5)

    # BBox
    parser.add_argument('--bbox_mode', type=str, default='gt',
                        choices=['gt', 'predicted'])
    parser.add_argument('--bbox_config', type=str,
                        default='configs/bbox/erp_bbox_centerpoint_v2.json')
    parser.add_argument('--bbox_ckpt', type=str, default='auto')
    parser.add_argument('--bbox_score_threshold', type=float, default=0.3)

    # Stage 2
    parser.add_argument('--stage2_shape_config', type=str,
                        default='configs/gen/erp_slat_flow_img2shape_asset_aware_bf16.json')
    parser.add_argument('--stage2_shape_ckpt_dir', type=str,
                        default='ckpts/erp_slat_flow_img2shape_asset_aware_bf16')
    parser.add_argument('--stage2_tex_config', type=str,
                        default='configs/gen/erp_slat_flow_imgshape2tex_asset_aware_bf16.json')
    parser.add_argument('--stage2_tex_ckpt_dir', type=str,
                        default='ckpts/erp_slat_flow_imgshape2tex_asset_aware_bf16')
    parser.add_argument('--enable_texture', action='store_true', default=True)
    parser.add_argument('--layout_mode', type=str, default='floor_perimeter_clean',
                        choices=['floor_perimeter', 'floor_perimeter_clean', 'no_floor_assets'],
                        help='Layout detection mode: '
                             'floor_perimeter=assets include floor voxels (matches training), '
                             'floor_perimeter_clean=assets exclude floor-layer voxels, '
                             'no_floor_assets=alias for floor_perimeter_clean')

    # Common
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--num_vis', type=int, default=-1)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false')
    parser.add_argument('--save_concat', action='store_true', default=False,
                        help='Also save GT|Pred concatenated images to vis_concat/')
    parser.add_argument('--max_meshes', type=int, default=-1,
                        help='Max samples to decode GLB meshes for (-1 for all)')
    parser.add_argument('--ckpt_step', type=str, default='latest')

    # Distributed evaluation (split samples across processes)
    parser.add_argument('--rank', type=int, default=0,
                        help='Worker rank for distributed eval (0-indexed)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of workers for distributed eval')

    args = parser.parse_args()

    # Run in parallel across 4 GPUs

    # Predicted
    # python eval/pipeline/eval_pipeline.py --rank 0 --noise_mode random --bbox_mode gt --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 0
    # python eval/pipeline/eval_pipeline.py --rank 1 --noise_mode random --bbox_mode gt --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 1
    # python eval/pipeline/eval_pipeline.py --rank 2 --noise_mode random --bbox_mode gt --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 2
    # python eval/pipeline/eval_pipeline.py --rank 3 --noise_mode random --bbox_mode gt --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 3
    # Run in parallel across 4 GPUs
    # python eval/pipeline/eval_pipeline.py --rank 0 --noise_mode random --bbox_mode predicted --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 0
    # python eval/pipeline/eval_pipeline.py --rank 1 --noise_mode random --bbox_mode predicted --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 1
    # python eval/pipeline/eval_pipeline.py --rank 2 --noise_mode random --bbox_mode predicted --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 2
    # python eval/pipeline/eval_pipeline.py --rank 3 --noise_mode random --bbox_mode predicted --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 3

    # python eval/pipeline/eval_pipeline.py --rank 0 --noise_mode sdedit --sdedit_alpha 0.7 --bbox_mode gt --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 0
    # python eval/pipeline/eval_pipeline.py --rank 1 --noise_mode sdedit --sdedit_alpha 0.7 --bbox_mode gt --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 1
    # python eval/pipeline/eval_pipeline.py --rank 2 --noise_mode sdedit --sdedit_alpha 0.7 --bbox_mode gt --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 2
    # python eval/pipeline/eval_pipeline.py --rank 3 --noise_mode sdedit --sdedit_alpha 0.7 --bbox_mode gt --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 3

    # python eval/pipeline/eval_pipeline.py --rank 0 --noise_mode sdedit --sdedit_alpha 0.5 --bbox_mode predicted --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 0
    # python eval/pipeline/eval_pipeline.py --rank 1 --noise_mode sdedit --sdedit_alpha 0.5 --bbox_mode predicted --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 1
    # python eval/pipeline/eval_pipeline.py --rank 2 --noise_mode sdedit --sdedit_alpha 0.5 --bbox_mode predicted --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 2
    # python eval/pipeline/eval_pipeline.py --rank 3 --noise_mode sdedit --sdedit_alpha 0.5 --bbox_mode predicted --layout_mode floor_perimeter_clean --world_size 4 --gpu_id 3

    # Do not remove this!
    # args.enable_texture = True
    # args.bbox_mode = 'predicted'
    # args.output_dir = 'evals/stage12_pipeline'
    # args.max_samples = 2
    # args.num_vis = 20
    # args.gpu_id = 0

    # Auto-generate output dir
    if args.output_dir == '':
        tag = f'bbox_{args.bbox_mode}'
        if args.noise_mode == 'sdedit':
            tag += f'_sdedit{args.sdedit_alpha}'
        if args.enable_texture:
            tag += '_tex'
        if args.layout_mode != 'floor_perimeter':
            tag += f'_{args.layout_mode}'
        args.output_dir = os.path.join('evals', 'stage12_pipeline', tag)

    os.makedirs(args.output_dir, exist_ok=True)
    device = f'cuda:{args.gpu_id}'

    # Save eval config
    eval_config = vars(args)
    eval_config['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(args.output_dir, 'eval_config.json'), 'w') as f:
        json.dump(eval_config, f, indent=2)

    print("InSpace Full Evaluation Pipeline")
    print(f"  Output: {args.output_dir}")
    print(f"  BBox mode: {args.bbox_mode}")
    print(f"  Noise mode: {args.noise_mode}")
    print(f"  Texture: {args.enable_texture}")
    print(f"  Device: {device}")
    if args.world_size > 1:
        print(f"  Distributed: rank {args.rank}/{args.world_size}")

    # Discover samples
    all_samples = discover_samples(args.data_dir)
    print(f"\n  Found {len(all_samples)} samples in {args.data_dir}")
    if args.max_samples > 0:
        all_samples = all_samples[:args.max_samples]
        print(f"  Limited to {len(all_samples)} samples")

    # Split samples across workers
    if args.world_size > 1:
        assert 0 <= args.rank < args.world_size, \
            f"rank must be in [0, {args.world_size}), got {args.rank}"
        samples = all_samples[args.rank::args.world_size]
        # Adjust per-worker limits so total across all workers matches the original value
        if args.num_vis > 0:
            # Distribute num_vis evenly: rank gets ceil or floor share
            base, remainder = divmod(args.num_vis, args.world_size)
            args.num_vis = base + (1 if args.rank < remainder else 0)
        if args.max_meshes > 0:
            base, remainder = divmod(args.max_meshes, args.world_size)
            args.max_meshes = base + (1 if args.rank < remainder else 0)
        print(f"  Worker {args.rank}/{args.world_size}: processing {len(samples)} samples "
              f"(num_vis={args.num_vis}, max_meshes={args.max_meshes})")
    else:
        samples = all_samples

    # Phase 1: Stage 1 — Sparse Structure Generation
    stage1_results = run_stage1(args, samples, device)

    # Phase 2: BBox estimation
    bbox_results = run_bbox(args, samples, stage1_results, device)

    # Phase 4: Stage 2-1 — Shape Generation
    shape_results = run_stage2_shape(args, samples, stage1_results, bbox_results, device)

    # Phase 5: Stage 2-2 — Texture Generation (optional)
    texture_results = None
    if args.enable_texture:
        texture_results = run_stage2_texture(
            args, samples, bbox_results, shape_results, device)

    # Visualization (includes SS, BBox, and final mesh visualizations)
    run_visualization(args, samples, stage1_results, bbox_results,
                      shape_results, texture_results, device)

    # Phase 6: Decode & Save Meshes (GLB for paper figures, limited count)
    mesh_samples = samples if args.max_meshes < 0 else samples[:args.max_meshes]
    if len(mesh_samples) < len(samples):
        print(f"\n  [Mesh decode] Limited to first {len(mesh_samples)}/{len(samples)} samples (--max_meshes {args.max_meshes})")
    run_decode_meshes(args, mesh_samples, bbox_results, shape_results, texture_results, device)

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"  Output: {args.output_dir}")
    print(f"  Samples processed: {len(stage1_results)}")
    print(f"  BBox mode: {args.bbox_mode}")
    print(f"  Texture: {args.enable_texture}")


if __name__ == '__main__':
    main()
