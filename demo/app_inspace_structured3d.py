# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
InSpace Structured3D Gradio Interactive Demo

Interactive per-sample inference for the InSpace pipeline on the Structured3D
samples used to verify whether the pretrained TRELLIS prior reconstructs
structural openings (windows, doors).

Stages:
    1. Load Input (ERP -> cubemap on-the-fly, GT depth.png -> point cloud)
    2. Coarse Scene Geometry (SS flow -> 64^3 voxel, optional SDEdit from depth)
    3. 3D BBox Estimation (CenterPoint prediction)
    4. Layout and Asset-Aware Scene Generation (Shape + Texture -> Mesh)

Differences from demo/app_inspace_replicapano.py:
    - Sample identity = (scene_id, variant) where variant in {full, empty}
    - Path layout: {scene_id}/panorama/{variant}/...
    - Depth source: GT 16-bit depth.png (mm) — no DA-2 estimation needed
    - Default ERP: rgb_rawlight.png (configurable)
    - Saves to: evals/structured3d/

Usage:
    python demo/app_inspace_structured3d.py --port 7860
"""

import os
import sys
import time
import argparse
import traceback
from tqdm import tqdm

# Select the GPU before importing torch (must be set before the first torch import).
# Defaults to GPU 1; override from the shell, e.g.
#   CUDA_VISIBLE_DEVICES=0 python demo/app_inspace_structured3d.py
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print(f"[DEBUG] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

# Reuse model management, inference, and viz from the shared demo module.
from demo.app_inspace_utils import (
    model_manager,
    ensure_demo_samples,
    log,
    find_latest_ckpt,
    load_denoiser,
    detect_floor_z,
    detect_layout_from_floor_perimeter,
    assign_voxels_to_obbs,
    construct_stage2_input,
    run_bbox_predicted_single,
    run_stage2_shape_single,
    run_stage2_texture_single,
    decode_meshes_single,
    DEFAULT_STAGE1_CONFIG,
)
from demo.app_inspace_viz import (
    create_psg_plotly_figure,
    create_voxel_glb,
    create_bbox_with_voxel_glb,
    create_scene_glb,
    create_layout_glb,
    create_exploded_glb,
    render_ss_exterior,
    render_ss_topdown_cam,
    render_ss_interior,
    render_mesh_exterior,
    render_mesh_topdown_cam,
    render_mesh_interior,
    render_bbox_topdown,
)
import demo.app_inspace_utils as _stride_utils
from demo.app_inspace_ui import header_html, INSPACE_THEME, INSPACE_CSS

from trellis2 import models
from trellis2.modules.sparse.basic import SparseTensor
from trellis2.trainers.flow_matching.mixins.erp_image_conditioned import (
    ERPImageEncoder,
    create_spatial_attention_mask,
)

try:
    import py360convert
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "py360convert"])
    import py360convert


# ============================================================
# Constants
# ============================================================

DEFAULT_DATA_DIR = os.path.join(
    PROJECT_ROOT, 'datasets', 'Structured_samples')
DEFAULT_OUTPUT_ROOT = os.path.join(
    PROJECT_ROOT, 'demo_outputs', 'structured3d')
VARIANTS = ['full', 'empty']
LIGHT_CHOICES = ['rawlight', 'coldlight', 'warmlight']
FACE_ORDER = ['front', 'right', 'back', 'left', 'top', 'bottom']


# ============================================================
# Path helpers
# ============================================================

def variant_dir(data_dir, scene_id, variant):
    return os.path.join(data_dir, scene_id, 'panorama', variant)


def erp_path(data_dir, scene_id, variant, light='rawlight'):
    return os.path.join(variant_dir(data_dir, scene_id, variant), f'rgb_{light}.png')


def depth_path(data_dir, scene_id, variant):
    return os.path.join(variant_dir(data_dir, scene_id, variant), 'depth.png')


# ============================================================
# Dataset Discovery
# ============================================================

def discover_structured3d_samples(data_dir, light='rawlight'):
    """Returns list of (scene_id, variant) tuples covering both 'full' and 'empty'."""
    samples = []
    if not os.path.isdir(data_dir):
        return samples
    for scene_id in sorted(os.listdir(data_dir)):
        scene_dir = os.path.join(data_dir, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        panorama_dir = os.path.join(scene_dir, 'panorama')
        if not os.path.isdir(panorama_dir):
            continue
        for variant in VARIANTS:
            v_dir = os.path.join(panorama_dir, variant)
            if not os.path.isdir(v_dir):
                continue
            if not os.path.exists(os.path.join(v_dir, f'rgb_{light}.png')):
                continue
            samples.append((scene_id, variant))
    return samples


# ============================================================
# ERP / cubemap loading
# ============================================================

def erp_to_cubemap(erp_image, face_size=512, fov=120):
    face_directions = {
        'front': (0, 0), 'right': (90, 0), 'back': (180, 0),
        'left': (270, 0), 'top': (0, 90), 'bottom': (0, -90),
    }
    faces = {}
    for name, (yaw, pitch) in face_directions.items():
        faces[name] = py360convert.e2p(
            erp_image, fov_deg=(fov, fov),
            u_deg=yaw, v_deg=pitch,
            out_hw=(face_size, face_size), mode='bilinear',
        )
    return faces


def load_cubemap_from_saved(cubic_dir, image_size=512):
    import torchvision.transforms as T
    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    tensors = []
    for fname in FACE_ORDER:
        img = Image.open(os.path.join(cubic_dir, f'{fname}.png')).convert('RGB')
        tensors.append(transform(img))
    return torch.stack(tensors)


def load_cubemap_from_erp(data_dir, scene_id, variant, image_size=512, light='rawlight'):
    """Load cubemap as [6, 3, H, W] tensor. Uses cached files if available."""
    cubic_dir = os.path.join(variant_dir(data_dir, scene_id, variant), 'cubic_fov_120')
    all_saved = os.path.isdir(cubic_dir) and all(
        os.path.exists(os.path.join(cubic_dir, f'{fn}.png')) for fn in FACE_ORDER)
    if all_saved:
        return load_cubemap_from_saved(cubic_dir, image_size)

    import torchvision.transforms as T
    p = erp_path(data_dir, scene_id, variant, light=light)
    erp_img = np.array(Image.open(p).convert('RGB'))
    faces = erp_to_cubemap(erp_img, face_size=image_size, fov=120)
    transform = T.Compose([T.ToTensor()])
    tensors = [transform(faces[n].astype(np.uint8)) for n in FACE_ORDER]
    return torch.stack(tensors)


# ============================================================
# GT depth → point cloud
# ============================================================

def load_gt_depth_meters(data_dir, scene_id, variant):
    p = depth_path(data_dir, scene_id, variant)
    if not os.path.exists(p):
        return None
    arr = np.array(Image.open(p))
    if arr.dtype != np.uint16:
        arr = arr.astype(np.float32)
    return arr.astype(np.float32) / 1000.0


def erp_depth_to_point_cloud(rgb, depth, subsample=2, max_points=50000, max_depth=20.0,
                             remove_ceiling=False, ceiling_threshold=0.0):
    """ERP depth + RGB → 3D point cloud (camera-centered, OpenGL convention)."""
    if subsample > 1:
        rgb = rgb[::subsample, ::subsample]
        depth = depth[::subsample, ::subsample]

    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    theta = (v.astype(np.float64) / H - 0.5) * np.pi
    phi = (u.astype(np.float64) / W - 0.5) * 2 * np.pi

    x = np.cos(theta) * np.sin(phi) * depth
    y = -np.sin(theta) * depth
    z = -np.cos(theta) * np.cos(phi) * depth

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors_flat = rgb.reshape(-1, 3) if rgb is not None else np.full((H * W, 3), 128, dtype=np.uint8)
    d_flat = depth.flatten()
    valid = np.isfinite(d_flat) & (d_flat > 0.01) & (d_flat < max_depth)

    points = points[valid]
    colors_flat = colors_flat[valid]

    if remove_ceiling and len(points) > 0 and ceiling_threshold > 0:
        y_max = points[:, 1].max()
        ceiling_mask = points[:, 1] < (y_max - ceiling_threshold)
        points = points[ceiling_mask]
        colors_flat = colors_flat[ceiling_mask]

    if max_points > 0 and len(points) > max_points:
        idx = np.random.RandomState(42).choice(len(points), max_points, replace=False)
        idx = np.sort(idx)
        points = points[idx]
        colors_flat = colors_flat[idx]

    return points, colors_flat.astype(np.uint8)


def self_normalize_point_cloud(points):
    pts_min = points.min(axis=0)
    pts_max = points.max(axis=0)
    center = (pts_min + pts_max) / 2.0
    max_extent = (pts_max - pts_min).max()
    scale = 0.99999 / max_extent if max_extent > 1e-6 else 1.0
    normalized = np.clip((points - center) * scale, -0.5 + 1e-6, 0.5 - 1e-6)
    return normalized, center, scale


def cam_to_world_points(points_cam):
    return np.column_stack([
        points_cam[:, 0],
        -points_cam[:, 2],
        points_cam[:, 1],
    ])


def voxelize_point_cloud(points_norm, grid_size=64):
    voxel_indices = ((points_norm + 0.5) * grid_size).astype(np.int32)
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)
    unique_voxels = np.unique(voxel_indices, axis=0)
    occ_grid = torch.zeros(1, 1, grid_size, grid_size, grid_size, dtype=torch.float32)
    for c in unique_voxels:
        occ_grid[0, 0, c[0], c[1], c[2]] = 1.0
    voxel_centers = (unique_voxels + 0.5) / grid_size - 0.5
    return occ_grid, voxel_centers, unique_voxels


def load_gt_depth_raw_point_cloud(data_dir, scene_id, variant, light='rawlight'):
    """Load GT depth + RGB and lift to a raw camera-space point cloud."""
    depth_m = load_gt_depth_meters(data_dir, scene_id, variant)
    if depth_m is None:
        return None
    erp_p = erp_path(data_dir, scene_id, variant, light=light)
    erp_rgb = np.array(Image.open(erp_p).convert('RGB')) if os.path.exists(erp_p) else None
    if erp_rgb is None:
        erp_rgb = np.full((*depth_m.shape, 3), 128, dtype=np.uint8)

    points_vis, colors_vis = erp_depth_to_point_cloud(
        erp_rgb, depth_m, subsample=2, max_points=50000,
        remove_ceiling=False, ceiling_threshold=0.0)
    if len(points_vis) < 100:
        return None

    return {
        'points_raw': points_vis,
        'colors': colors_vis,
        'da2_depth': depth_m,
        'erp_rgb': erp_rgb,
        'is_normalized': False,
        'is_cropped': False,
    }


def crop_depth_point_cloud(da2_data, remove_ceiling=False, ceiling_threshold=0.0,
                            crop_bbox=None):
    if da2_data is None:
        return None
    erp_rgb = da2_data['erp_rgb']
    depth_m = da2_data['da2_depth']

    points_full, colors_full = erp_depth_to_point_cloud(
        erp_rgb, depth_m, subsample=1, max_points=0,
        remove_ceiling=remove_ceiling, ceiling_threshold=ceiling_threshold)
    if len(points_full) < 100:
        return None

    if crop_bbox is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bbox
        mask = (
            (points_full[:, 0] >= x_min) & (points_full[:, 0] <= x_max) &
            (points_full[:, 1] >= y_min) & (points_full[:, 1] <= y_max) &
            (points_full[:, 2] >= z_min) & (points_full[:, 2] <= z_max)
        )
        points_full = points_full[mask]
        colors_full = colors_full[mask]
    if len(points_full) < 100:
        return None

    points_vis, colors_vis = erp_depth_to_point_cloud(
        erp_rgb, depth_m, subsample=2, max_points=50000,
        remove_ceiling=remove_ceiling, ceiling_threshold=ceiling_threshold)
    if crop_bbox is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bbox
        mask = (
            (points_vis[:, 0] >= x_min) & (points_vis[:, 0] <= x_max) &
            (points_vis[:, 1] >= y_min) & (points_vis[:, 1] <= y_max) &
            (points_vis[:, 2] >= z_min) & (points_vis[:, 2] <= z_max)
        )
        points_vis = points_vis[mask]
        colors_vis = colors_vis[mask]

    da2_data = dict(da2_data)
    da2_data['points_raw'] = points_vis
    da2_data['colors'] = colors_vis
    da2_data['points_full_raw'] = points_full
    da2_data['colors_full'] = colors_full
    da2_data['is_cropped'] = True
    da2_data['is_normalized'] = False
    da2_data['crop_settings'] = {
        'remove_ceiling': remove_ceiling,
        'ceiling_threshold': ceiling_threshold,
        'crop_bbox': crop_bbox,
    }
    return da2_data


def normalize_depth_point_cloud(da2_data):
    if da2_data is None:
        return None
    points_full = da2_data.get('points_full_raw')
    if points_full is None or len(points_full) < 100:
        return None

    points_full_world = cam_to_world_points(points_full)
    points_norm_full, center, scale = self_normalize_point_cloud(points_full_world)
    camera_center = (np.array([0.0, 0.0, 0.0]) - center) * scale

    points_vis = da2_data['points_raw']
    points_vis_world = cam_to_world_points(points_vis)
    points_vis_norm = np.clip((points_vis_world - center) * scale, -0.5 + 1e-6, 0.5 - 1e-6)

    da2_data = dict(da2_data)
    da2_data['points'] = points_vis_norm
    da2_data['points_full'] = points_norm_full
    da2_data['camera_center'] = camera_center
    da2_data['center'] = center
    da2_data['scale'] = scale
    da2_data['is_normalized'] = True
    da2_data['n_points_full'] = len(points_norm_full)
    return da2_data


def compute_sdedit_initial_latent(da2_data, device='cuda'):
    """Inline step9+10: normalized point cloud → voxelize → SS latent."""
    if da2_data is None:
        return None, None
    points_norm = da2_data['points_full']
    camera_center = da2_data['camera_center']
    if len(points_norm) < 100:
        return None, None
    occ_grid, _, _ = voxelize_point_cloud(points_norm, grid_size=64)
    n_occupied = int(occ_grid.sum().item())
    log(f"[SDEdit] Voxelized: {len(points_norm)} pts -> {n_occupied} occupied voxels (64^3)")
    occ_grid = occ_grid.to(device)

    ss_encoder = model_manager.get_ss_encoder()
    with torch.no_grad():
        z = ss_encoder(occ_grid, sample_posterior=False)
    log(f"[SDEdit] SS latent: shape={list(z.shape)}, range=[{z.min():.3f}, {z.max():.3f}], std={z.std():.3f}")
    return z[0].cpu().numpy(), camera_center


# ============================================================
# Stage 1 inference (Structured3D specialization)
# ============================================================

@torch.no_grad()
def run_stage1_structured3d(
    data_dir, scene_id, variant, light,
    use_sdedit=False, alpha=0.5,
    da2_data=None,
    steps=12, cfg_strength=7.5, seed=42,
    use_spatial_mask=True,
):
    t0 = time.time()
    log(f"[Stage 1] Structured3D CSG generation (scene={scene_id}, variant={variant}, seed={seed})")
    device = model_manager.device
    config = model_manager._load_config(DEFAULT_STAGE1_CONFIG)
    trainer_config = config['trainer']['args']

    denoiser, ss_decoder = model_manager.get_stage1()
    erp_encoder = model_manager.get_erp_encoder()

    t1 = time.time()
    log("[Stage 1] Converting ERP -> cubemap -> encoding...")
    cond = load_cubemap_from_erp(data_dir, scene_id, variant, light=light).unsqueeze(0).to(device)
    encoded_cond = erp_encoder(cond)
    neg_cond = torch.zeros_like(encoded_cond)
    log(f"[Stage 1] Cubemap encoded ({time.time()-t1:.1f}s), cond range=[{encoded_cond.min():.3f}, {encoded_cond.max():.3f}]")

    torch.manual_seed(seed)
    sigma_min = trainer_config.get('sigma_min', 1e-5)

    sdedit_ok = False
    if use_sdedit and da2_data is not None:
        log(f"[Stage 1] SDEdit from GT depth (alpha={alpha})")
        psg_ss_latent, _ = compute_sdedit_initial_latent(da2_data, device)
        if psg_ss_latent is not None:
            x_init = torch.from_numpy(psg_ss_latent).float().unsqueeze(0).to(device)
            t_val = alpha
            gaussian_noise = torch.randn_like(x_init)
            noise = (1 - t_val) * x_init + (sigma_min + (1 - sigma_min) * t_val) * gaussian_noise
            sdedit_ok = True
        else:
            log("[Stage 1] SDEdit failed, using random noise")
            noise = torch.randn(1, 8, 16, 16, 16, device=device)
    else:
        log("[Stage 1] Using random Gaussian noise")
        noise = torch.randn(1, 8, 16, 16, 16, device=device)

    camera_center_np = da2_data['camera_center'] if da2_data is not None else np.zeros(3)
    camera_center_t = torch.from_numpy(camera_center_np).float()

    extra_kwargs = {}
    use_spatial = trainer_config.get('use_spatial_attention', False) and use_spatial_mask
    if use_spatial:
        cross_attn_mask = create_spatial_attention_mask(
            camera_center=camera_center_t.unsqueeze(0).to(device),
            voxel_resolution=trainer_config.get('voxel_resolution', 16),
            tokens_per_face=trainer_config.get('tokens_per_face', 1029),
            fov_degrees=trainer_config.get('spatial_attention_fov', 120.0),
            soft_mask=trainer_config.get('spatial_attention_soft', True),
            soft_margin=trainer_config.get('spatial_attention_soft_margin', 0.1),
        )
        extra_kwargs['cross_attn_mask'] = cross_attn_mask

    sampler = model_manager.get_sampler()
    t1 = time.time()

    if sdedit_ok:
        rescale_t = 5.0
        start_t = alpha
        t_seq = np.linspace(start_t, 0, steps + 1)
        # Rescale the schedule to the model's training-time convention (rescale_t=5.0);
        # feeding un-rescaled t makes the denoiser output collapse toward zero -> 0 voxels.
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        log(f"[Stage 1] SDEdit sampling t={start_t:.2f} -> 0 ({steps} steps, cfg={cfg_strength}, rescale_t={rescale_t})")
        sample = noise
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for t, t_prev in tqdm(t_pairs, desc="SDEdit Sampling"):
                out = sampler.sample_once(
                    denoiser, sample, t, t_prev, encoded_cond,
                    neg_cond=neg_cond,
                    guidance_strength=cfg_strength,
                    guidance_interval=(0.6, 1.0),
                    guidance_rescale=0.7,
                    **extra_kwargs,
                )
                sample = out.pred_x_prev
        z = sample
    else:
        log(f"[Stage 1] Sampling ({steps} steps, cfg={cfg_strength})...")
        with torch.autocast('cuda', dtype=torch.bfloat16):
            res = sampler.sample(
                denoiser, noise=noise, cond=encoded_cond, neg_cond=neg_cond,
                steps=steps, rescale_t=5.0,
                guidance_strength=cfg_strength,
                guidance_interval=(0.6, 1.0), guidance_rescale=0.7,
                verbose=True, **extra_kwargs,
            )
        z = res.samples

    torch.cuda.synchronize()
    log(f"[Stage 1] Sampling done ({time.time()-t1:.1f}s)")
    voxel = ss_decoder(z.float())
    voxel_binary = (voxel > 0).cpu()
    n_active = voxel_binary.sum().item()
    log(f"[Stage 1] Done: {n_active} active voxels ({time.time()-t0:.1f}s)")

    return {
        'ss_latent': z[0].cpu().numpy(),
        'voxel_64': voxel_binary[0].numpy(),
        'encoded_cond': encoded_cond.cpu().numpy(),
        'camera_center': camera_center_np,
    }


# ============================================================
# Monkey-patch shared cubemap loader so Stage 2 helpers work
# ============================================================

_original_load_cubemap_images = _stride_utils.load_cubemap_images


def _patched_load_cubemap_images(data_dir, scene_id, room_id, view_idx=0):
    """Reroute Stage 2's helper through Structured3D's path layout.

    For Structured3D, scene_id=scene_id, room_id=variant, view_idx ignored.
    """
    cubic_dir = os.path.join(data_dir, scene_id, room_id, 'cubic_fov_120')
    if os.path.isdir(cubic_dir):
        return _original_load_cubemap_images(data_dir, scene_id, room_id, view_idx)

    # Structured3D layout
    s3d_cubic_dir = os.path.join(data_dir, scene_id, 'panorama', room_id, 'cubic_fov_120')
    if os.path.isdir(s3d_cubic_dir) and os.path.exists(os.path.join(s3d_cubic_dir, 'front.png')):
        return load_cubemap_from_saved(s3d_cubic_dir)

    p = erp_path(data_dir, scene_id, room_id, light=_LIGHT_HOLDER['light'])
    if os.path.exists(p):
        return load_cubemap_from_erp(data_dir, scene_id, room_id, light=_LIGHT_HOLDER['light'])

    raise FileNotFoundError(f"Cannot find cubemap data for {scene_id}/{room_id} in {data_dir}")


_LIGHT_HOLDER = {'light': 'rawlight'}
_stride_utils.load_cubemap_images = _patched_load_cubemap_images


# ============================================================
# Save (delegated)
# ============================================================

def save_all_results_structured3d(output_dir, state, meshes_list=None, data_dir=DEFAULT_DATA_DIR):
    from demo.app_inspace_utils import save_all_results
    return save_all_results(output_dir, state, meshes_list, data_dir)


# ============================================================
# Dataset state
# ============================================================

SAMPLES = []
SCENE_IDS = []
SCENE_VARIANTS = {}


def init_dataset(data_dir, light):
    global SAMPLES, SCENE_IDS, SCENE_VARIANTS
    _LIGHT_HOLDER['light'] = light
    SAMPLES = discover_structured3d_samples(data_dir, light=light)
    SCENE_VARIANTS = {}
    for scene_id, variant in SAMPLES:
        SCENE_VARIANTS.setdefault(scene_id, []).append(variant)
    SCENE_IDS = sorted(SCENE_VARIANTS.keys())


# ============================================================
# Model3D viewer helpers
# ============================================================

# White background for the embedded WebGL model viewers (RGBA in 0-1).
WHITE_BG = (1.0, 1.0, 1.0, 1.0)


def remove_ceiling_from_glb(glb_path, cut_ratio=0.15):
    """Drop the top `cut_ratio` (by global Y) of a GLB and return a temp path.

    Used by the Mesh-tab enlarge view to peek inside generated rooms — the
    ceiling otherwise occludes everything below it from a top-down camera.
    Mirrors the helper in eval_structured3d/view_results.py.
    """
    if glb_path is None or not os.path.exists(glb_path):
        return None
    import tempfile
    import trimesh
    scene = trimesh.load(glb_path, force='scene')
    if not scene.geometry:
        return None
    height_axis = 1
    all_bounds = []
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        try:
            transform = scene.graph.get(name)[0]
        except Exception:
            transform = np.eye(4)
        all_bounds.append(
            trimesh.transformations.transform_points(geom.vertices, transform))
    if not all_bounds:
        return None
    all_verts = np.concatenate(all_bounds, axis=0)
    y_min = all_verts[:, height_axis].min()
    y_max = all_verts[:, height_axis].max()
    y_cut = y_max - cut_ratio * (y_max - y_min)

    new_scene = trimesh.Scene()
    for name, geom in scene.geometry.items():
        try:
            transform = scene.graph.get(name)[0]
        except Exception:
            transform = np.eye(4) if isinstance(geom, trimesh.Trimesh) else None
        if not isinstance(geom, trimesh.Trimesh):
            new_scene.add_geometry(geom, node_name=name, transform=transform)
            continue
        verts_world = trimesh.transformations.transform_points(geom.vertices, transform)
        face_heights = verts_world[geom.faces, height_axis]
        keep_mask = face_heights.max(axis=1) < y_cut
        keep_indices = np.where(keep_mask)[0]
        if len(keep_indices) == 0:
            continue
        new_geom = geom.submesh([keep_indices], append=True)
        new_scene.add_geometry(new_geom, node_name=name, transform=transform)
    if not new_scene.geometry:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    new_scene.export(tmp.name, file_type="glb")
    return tmp.name


# ============================================================
# Cubemap rendering helper for the UI
# ============================================================

def _build_cubemap_grid(faces_display, label_h=20, face_size=256):
    grid_w, grid_h = 3 * face_size, 2 * (face_size + label_h)
    img = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for fi, fname in enumerate(FACE_ORDER):
        r, c = fi // 3, fi % 3
        x_off = c * face_size
        y_off = r * (face_size + label_h)
        draw.text((x_off + face_size // 2, y_off + 2), fname,
                  fill=(0, 0, 0), font=font, anchor='mt')
        img.paste(Image.fromarray(faces_display[fname]), (x_off, y_off + label_h))
    return img


def _depth_to_image(depth_m):
    """Render a depth map (meters) as an 8-bit grayscale image with valid mask."""
    valid = depth_m > 0
    if not valid.any():
        return None
    out = np.zeros_like(depth_m, dtype=np.float32)
    lo, hi = depth_m[valid].min(), depth_m[valid].max()
    if hi - lo < 1e-6:
        out[valid] = 0.5
    else:
        out[valid] = (depth_m[valid] - lo) / (hi - lo)
    img = (out * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img, mode='L').convert('RGB')


# ============================================================
# Event handlers
# ============================================================

def on_scene_change(scene_id):
    variants = SCENE_VARIANTS.get(scene_id, VARIANTS)
    if variants:
        return gr.update(choices=variants, value=variants[0])
    return gr.update(choices=[], value=None)


def on_load_input(scene_id, variant, light, data_dir, state):
    """Load and display ERP, depth, cubemap, and GT-depth point cloud."""
    try:
        t0 = time.time()
        state = dict(state) if state else {}
        state['scene_id'] = scene_id
        state['variant'] = variant
        state['light'] = light
        _LIGHT_HOLDER['light'] = light

        # ERP image
        ep = erp_path(data_dir, scene_id, variant, light=light)
        erp_img = Image.open(ep).convert('RGB') if os.path.exists(ep) else None

        # GT depth visualization
        depth_m = load_gt_depth_meters(data_dir, scene_id, variant)
        depth_img = _depth_to_image(depth_m) if depth_m is not None else None

        # Cubemap (cache to disk under panorama/{variant}/cubic_fov_120/)
        t1 = time.time()
        v_dir = variant_dir(data_dir, scene_id, variant)
        cubic_dir = os.path.join(v_dir, 'cubic_fov_120')
        concat_dir = os.path.join(v_dir, 'cubic_fov_120_concat')

        all_faces_exist = os.path.isdir(cubic_dir) and all(
            os.path.exists(os.path.join(cubic_dir, f'{fn}.png')) for fn in FACE_ORDER)

        if all_faces_exist:
            faces_display = {}
            for fn in FACE_ORDER:
                img = Image.open(os.path.join(cubic_dir, f'{fn}.png')).convert('RGB')
                faces_display[fn] = np.array(img.resize((256, 256)))
            log(f"[Load] Cubemap loaded from disk: {time.time()-t1:.2f}s")
        elif erp_img is not None:
            erp_np = np.array(erp_img)
            faces_512 = erp_to_cubemap(erp_np, face_size=512, fov=120)
            os.makedirs(cubic_dir, exist_ok=True)
            for fname in FACE_ORDER:
                Image.fromarray(faces_512[fname].astype(np.uint8)).save(
                    os.path.join(cubic_dir, f'{fname}.png'))
            os.makedirs(concat_dir, exist_ok=True)
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig_save, axes_save = plt.subplots(2, 3, figsize=(12, 8))
            for fi, fname in enumerate(FACE_ORDER):
                r, c = fi // 3, fi % 3
                axes_save[r, c].imshow(faces_512[fname].astype(np.uint8))
                axes_save[r, c].set_title(fname)
                axes_save[r, c].axis('off')
            plt.tight_layout()
            fig_save.savefig(
                os.path.join(concat_dir, f'{scene_id}_{variant}_concat.png'),
                bbox_inches='tight', dpi=150)
            plt.close(fig_save)

            faces_display = {fn: np.array(Image.fromarray(
                faces_512[fn].astype(np.uint8)).resize((256, 256))) for fn in FACE_ORDER}
            log(f"[Load] Cubemap generated & saved: {time.time()-t1:.2f}s")
        else:
            faces_display = None

        cubemap_grid = _build_cubemap_grid(faces_display) if faces_display else None

        # Lift depth → point cloud (raw, then auto-normalize for SDEdit)
        t2 = time.time()
        da2_data = load_gt_depth_raw_point_cloud(data_dir, scene_id, variant, light=light)
        state['da2_data'] = da2_data
        log(f"[Load] GT depth lift: {time.time()-t2:.2f}s")

        psg_fig = None
        depth_status = ""
        if da2_data is not None:
            pts = da2_data['points_raw']
            psg_fig = create_psg_plotly_figure(
                pts, da2_data['colors'],
                camera_center=np.array([0.0, 0.0, 0.0]),
                show_camera_center=True, show_coordinates=True,
            )
            n_pts = len(pts)
            pt_range = (f"X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
                        f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
                        f"Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
            depth_status = (f"GT depth point cloud: {n_pts} pts (camera-space, raw)\n"
                            f"{pt_range}")

        status = f"Loaded: {scene_id}/{variant} (light={light}) ({time.time()-t0:.1f}s)"
        if depth_status:
            status += f"\n{depth_status}"

        return erp_img, depth_img, cubemap_grid, psg_fig, state, depth_status, status

    except Exception as e:
        traceback.print_exc()
        return None, None, None, None, state, "", f"Error loading input: {e}"


def on_update_depth_pointcloud(remove_ceiling, ceiling_threshold,
                                x_min, x_max, y_min, y_max, z_min, z_max,
                                use_crop, state):
    """Apply ceiling removal + region crop to the raw GT-depth point cloud.

    Mirrors demo/app_inspace_replicapano.py's `on_update_da2_pointcloud`. The
    cropped point cloud is stored back into `state['da2_data']` and used by the
    next Generate CSG step (and the Normalize step).
    """
    state = dict(state) if state else {}
    da2_data = state.get('da2_data')
    if da2_data is None:
        return None, state, "Load input first."

    # Re-lift from the original ERP+depth so the crop applies to fresh data.
    base = {
        'da2_depth': da2_data['da2_depth'],
        'erp_rgb': da2_data['erp_rgb'],
        'is_normalized': False,
        'is_cropped': False,
    }
    crop_bbox = (x_min, x_max, y_min, y_max, z_min, z_max) if use_crop else None
    cropped = crop_depth_point_cloud(
        base,
        remove_ceiling=bool(remove_ceiling),
        ceiling_threshold=float(ceiling_threshold),
        crop_bbox=crop_bbox,
    )
    if cropped is None:
        return None, state, "Crop produced an empty point cloud — relax the settings."

    state['da2_data'] = cropped
    pts = cropped['points_raw']
    psg_fig = create_psg_plotly_figure(
        pts, cropped['colors'],
        camera_center=np.array([0.0, 0.0, 0.0]),
        show_camera_center=True, show_coordinates=True,
    )
    n_full = len(cropped.get('points_full_raw', pts))
    info = (
        f"Cropped: {len(pts)} viz pts ({n_full} full-res pts)\n"
        f"Settings: remove_ceiling={remove_ceiling}, threshold={ceiling_threshold:.2f}m\n"
        f"Region crop: {crop_bbox if use_crop else 'OFF'}\n"
        f"(Click 'Normalize' next to map to [-0.5, 0.5] for the model.)"
    )
    return psg_fig, state, info


def on_normalize_depth(state):
    state = dict(state) if state else {}
    da2_data = state.get('da2_data')
    if da2_data is None:
        return None, state, "Load input first."
    # If the user has already pressed 'Update Crop', preserve those settings.
    # Otherwise, do a default full-resolution lift with no ceiling/region cropping.
    if not da2_data.get('is_cropped', False):
        da2_data = crop_depth_point_cloud(da2_data)
    da2_data = normalize_depth_point_cloud(da2_data)
    state['da2_data'] = da2_data
    if da2_data is None:
        return None, state, "Failed to normalize point cloud."
    pts = da2_data['points']
    psg_fig = create_psg_plotly_figure(
        pts, da2_data['colors'], camera_center=da2_data['camera_center'],
        show_camera_center=True, show_coordinates=True,
    )
    cc = da2_data['camera_center']
    crop = da2_data.get('crop_settings', {}) or {}
    info = (
        f"Normalized: {len(da2_data['points_full'])} voxel-source pts (model space)\n"
        f"Camera center: [{cc[0]:.3f}, {cc[1]:.3f}, {cc[2]:.3f}], scale={da2_data['scale']:.4f}\n"
        f"From crop: remove_ceiling={crop.get('remove_ceiling', False)}, "
        f"threshold={crop.get('ceiling_threshold', 0.0):.2f}m, "
        f"region={crop.get('crop_bbox') if crop.get('crop_bbox') else 'OFF'}"
    )
    return psg_fig, state, info


def on_generate_csg(scene_id, variant, light, data_dir,
                    use_sdedit, alpha, use_spatial_mask, seed, steps, cfg_strength,
                    show_camera, max_cubes, state):
    try:
        t_start = time.time()
        state = dict(state) if state else {}
        da2_data = state.get('da2_data')

        # Auto-normalize if SDEdit and we don't have a normalized point cloud yet
        if da2_data is not None and not da2_data.get('is_normalized', False):
            if use_sdedit:
                da2_data = crop_depth_point_cloud(da2_data)
                da2_data = normalize_depth_point_cloud(da2_data)
            else:
                # For non-SDEdit we still want camera_center for spatial attn
                da2_data = crop_depth_point_cloud(da2_data)
                da2_data = normalize_depth_point_cloud(da2_data)
            state['da2_data'] = da2_data

        result = run_stage1_structured3d(
            data_dir, scene_id, variant, light,
            use_sdedit=use_sdedit, alpha=alpha,
            da2_data=da2_data,
            steps=steps, cfg_strength=cfg_strength, seed=seed,
            use_spatial_mask=use_spatial_mask,
        )

        state['scene_id'] = scene_id
        state['variant'] = variant
        state['ss_latent'] = result['ss_latent']
        state['voxel_64'] = result['voxel_64']
        state['encoded_cond'] = result['encoded_cond']
        state['camera_center'] = result['camera_center']

        camera_center = result['camera_center'] if show_camera else None
        pred_glb = create_voxel_glb(
            result['voxel_64'], camera_center, show_camera=show_camera,
            max_cubes=int(max_cubes))

        n_active = int(result['voxel_64'].sum())
        status = f"CSG: {n_active} active voxels (64^3)"
        if use_sdedit:
            status += f", SDEdit alpha={alpha}"
        status += f" ({time.time()-t_start:.1f}s)"
        return pred_glb, state, status

    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error: {e}"


def on_predict_bbox(bbox_threshold, state):
    try:
        state = dict(state) if state else {}
        voxel_64 = state.get('voxel_64')
        camera_center = state.get('camera_center')
        if voxel_64 is None:
            return None, state, "Generate CSG first.", "No BBox"
        log(f"[Stage 3] Predicting BBox (threshold={bbox_threshold})")
        bbox_result = run_bbox_predicted_single(voxel_64, score_threshold=bbox_threshold)
        state['obbs'] = bbox_result['obbs']
        state['asset_names'] = bbox_result['asset_names']
        state['asset_filenames'] = bbox_result.get('asset_filenames', [])
        state['bbox_source'] = 'predicted'
        pred_glb = create_bbox_with_voxel_glb(bbox_result['obbs'], voxel_64, camera_center)
        n_obbs = len(bbox_result['obbs'])
        confs = bbox_result['confidences']
        status = f"Predicted: {n_obbs} objects"
        if n_obbs > 0:
            status += f" (conf: {confs.min():.2f}-{confs.max():.2f})"
        bbox_label = f"Predicted ({n_obbs} objects)"
        return pred_glb, state, status, bbox_label
    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error: {e}", "Error"


def on_generate_scene(data_dir, seed, steps, cfg_strength,
                      gen_texture, layout_mode, state):
    try:
        t_start = time.time()
        state = dict(state) if state else {}
        scene_id = state.get('scene_id', '')
        variant = state.get('variant', 'full')
        voxel_64 = state.get('voxel_64')
        obbs = state.get('obbs')
        camera_center = state.get('camera_center')
        encoded_cond = state.get('encoded_cond')

        if voxel_64 is None or obbs is None:
            return None, None, None, state, "Generate CSG and predict bboxes first."

        log(f"\n--- Shape Generation (layout={layout_mode}) ---")
        t1 = time.time()
        # Reuse shared helper. It expects (data_dir, scene_id, room_id, view_idx)
        # with the patched cubemap loader; we pass scene_id/variant accordingly.
        shape_result = run_stage2_shape_single(
            data_dir, scene_id, variant, 0,
            voxel_64, obbs, camera_center, encoded_cond,
            steps=steps, cfg_strength=cfg_strength, seed=seed,
            layout_mode=layout_mode,
        )
        if shape_result is None:
            return None, None, None, state, "Shape generation failed."

        state['shape_coords'] = shape_result['shape_coords']
        state['shape_feats'] = shape_result['shape_feats']
        state['part_layouts'] = shape_result['part_layouts']
        state['has_layout'] = shape_result.get('has_layout', False)
        has_layout = state['has_layout']
        obbs_for_texture = shape_result.get('obbs_filtered', obbs)

        n_total = shape_result['shape_coords'].shape[0]
        n_parts = len(shape_result['part_layouts'])
        log(f"[Shape] Done ({time.time()-t1:.1f}s): {n_total} voxels, {n_parts} parts")

        tex_feats = None
        if gen_texture:
            log("\n--- Texture Generation ---")
            t2 = time.time()
            tex_result = run_stage2_texture_single(
                data_dir, scene_id, variant, 0,
                shape_result['shape_coords'], shape_result['shape_feats'],
                shape_result['part_layouts'],
                obbs_for_texture, camera_center, has_layout=has_layout,
                steps=steps, cfg_strength=cfg_strength, seed=seed,
            )
            tex_feats = tex_result['tex_feats']
            state['tex_feats'] = tex_feats
            state['has_texture'] = True
            log(f"[Texture] Done ({time.time()-t2:.1f}s)")
        else:
            state['has_texture'] = False

        log("\n--- Decode Meshes ---")
        t3 = time.time()
        asset_names = state.get('asset_names', [])
        meshes_list, trellis_rep_data = decode_meshes_single(
            shape_result['shape_coords'], shape_result['shape_feats'],
            shape_result['part_layouts'],
            tex_feats=tex_feats, asset_names=asset_names,
            has_layout=has_layout,
        )
        state['meshes_list'] = meshes_list
        state['trellis_rep_data'] = trellis_rep_data
        log(f"[Decode] Done ({time.time()-t3:.1f}s)")

        combined_glb = create_scene_glb(meshes_list, asset_names)
        layout_glb = create_layout_glb(meshes_list, has_layout=has_layout)
        exploded_glb = create_exploded_glb(
            meshes_list, asset_names, explosion_scale=0.0, has_layout=has_layout)

        n_valid = sum(1 for m in meshes_list if m is not None)
        tex_label = "with texture" if gen_texture else "shape only"
        status = (f"Generated: {n_total} voxels, {n_parts} parts\n"
                  f"{n_valid}/{len(meshes_list)} meshes ({tex_label})\n"
                  f"Total: {time.time()-t_start:.1f}s")
        return combined_glb, layout_glb, exploded_glb, state, status

    except Exception as e:
        traceback.print_exc()
        return None, None, None, state, f"Error: {e}"


def on_update_explode(explosion_scale, state):
    state = dict(state) if state else {}
    meshes_list = state.get('meshes_list')
    asset_names = state.get('asset_names', [])
    has_layout = state.get('has_layout', False)
    if meshes_list is None:
        return None
    return create_exploded_glb(meshes_list, asset_names, explosion_scale, has_layout=has_layout)


def on_save_all(data_dir, state):
    """Save the full set of result images and meshes under evals/structured3d/<tag>."""
    try:
        state = dict(state) if state else {}
        scene_id = state.get('scene_id')
        variant = state.get('variant')
        if not scene_id or not variant:
            return None, None, None, None, None, None, "Nothing to save.", ""

        light = state.get('light', 'rawlight')
        tag = f"interactive_{light}"
        out_dir = os.path.join(DEFAULT_OUTPUT_ROOT, tag, scene_id, variant)
        meshes_list = state.get('meshes_list')

        viz_dict = save_all_results_structured3d(
            out_dir, state, meshes_list=meshes_list, data_dir=data_dir)

        # Build PIL images for the gradio sidebar; tolerate missing keys.
        def _load(key):
            p = viz_dict.get(key) if isinstance(viz_dict, dict) else None
            if p and os.path.exists(p):
                return Image.open(p)
            return None

        cubemap = _load('cubemap_input')
        bbox = _load('bbox_topdown')
        ss = _load('ss_exterior')
        ss_int = _load('ss_interior')
        geom = _load('geometry_exterior')
        geom_td = _load('geometry_topdown_cam')
        tex_ext = _load('texture_exterior')
        tex_int = _load('texture_interior')
        tex_td = _load('texture_topdown_cam')
        return (cubemap, bbox, ss, ss_int, geom, geom_td, tex_ext, tex_int, tex_td,
                f"Saved to {out_dir}", out_dir)
    except Exception as e:
        traceback.print_exc()
        return (None, None, None, None, None, None, None, None, None,
                f"Save error: {e}", "")


# ============================================================
# UI
# ============================================================

def create_demo(data_dir=DEFAULT_DATA_DIR, light='rawlight'):
    init_dataset(data_dir, light)

    with gr.Blocks(title="InSpace Structured3D") as demo:
        gr.HTML(header_html("Structured3D demo · full / empty variants · windows-doors opening test"))
        state = gr.State({})
        data_dir_state = gr.State(data_dir)

        with gr.Row():
            # =========== LEFT PANEL ===========
            with gr.Column(scale=1):

                gr.Markdown("### Input")
                scene_dd = gr.Dropdown(
                    choices=SCENE_IDS,
                    value=SCENE_IDS[0] if SCENE_IDS else None,
                    label="Scene ID",
                )
                init_variants = SCENE_VARIANTS.get(SCENE_IDS[0], VARIANTS) if SCENE_IDS else VARIANTS
                variant_dd = gr.Dropdown(
                    choices=init_variants, value=init_variants[0],
                    label="Variant (full / empty)",
                )
                light_dd = gr.Dropdown(
                    choices=LIGHT_CHOICES, value=light,
                    label="Lighting",
                )
                load_btn = gr.Button("Load Input", variant="primary")

                gr.Markdown("---")
                gr.Markdown("### Stage 2: Coarse Scene Geometry")
                gr.Markdown("*SS flow → 64^3 voxel (optional SDEdit from GT depth)*")
                use_sdedit_cb = gr.Checkbox(
                    label="SDEdit from GT depth", value=True,
                    info="Use Structured3D depth.png as initial latent via SDEdit")
                alpha_slider = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Alpha (noise level)",
                    info="0=clean depth, 1=pure noise. Recommended 0.3-0.7")
                csg_spatial_mask_cb = gr.Checkbox(
                    label="Spatial Attention Mask", value=True)
                csg_show_cam_cb = gr.Checkbox(label="Show Camera", value=True)
                csg_max_cubes = gr.Slider(
                    1000, 50000, value=30000, step=1000,
                    label="Max Voxel Cubes")
                generate_csg_btn = gr.Button("Generate CSG", variant="primary")
                gr.Markdown(
                    "*To preview the normalized point cloud or remove the ceiling/region, "
                    "use the **Depth Point Cloud** tab.*")

                gr.Markdown("---")
                gr.Markdown("### Stage 3: 3D BBox")
                bbox_source_display = gr.Textbox(
                    label="BBox Source", value="None",
                    interactive=False, max_lines=1)
                pred_bbox_btn = gr.Button("Predict BBox", variant="primary")
                bbox_threshold = gr.Slider(
                    0.1, 0.9, value=0.3, step=0.05, label="Score Threshold")

                gr.Markdown("---")
                gr.Markdown("### Stage 4: Scene Generation")
                gen_texture_cb = gr.Checkbox(
                    label="Generate Texture", value=True)
                layout_mode_radio = gr.Radio(
                    choices=["floor_perimeter", "floor_perimeter_clean"],
                    value="floor_perimeter_clean",
                    label="Layout & Asset Mode")
                gen_scene_btn = gr.Button("Generate Scene", variant="primary")

                with gr.Accordion("Advanced", open=False):
                    seed_slider = gr.Slider(0, 99999, value=42, step=1, label="Seed")
                    steps_s1 = gr.Slider(4, 50, value=12, step=1, label="Stage 1 Steps")
                    steps_s2 = gr.Slider(4, 50, value=12, step=1, label="Stage 2 Steps")
                    cfg_s1 = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Stage 1 CFG")
                    cfg_s2 = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Stage 2 CFG")

                status_box = gr.Textbox(label="Status", lines=3, interactive=False)

            # =========== RIGHT PANEL ===========
            with gr.Column(scale=2):
                with gr.Tabs():

                    with gr.Tab("Input"):
                        with gr.Row():
                            erp_image = gr.Image(label="ERP Panorama", height=200)
                            depth_image = gr.Image(label="GT Depth (16-bit)", height=200)
                        cubemap_image = gr.Image(label="Cubemap (2x3)", height=400)
                        depth_info_box = gr.Textbox(
                            label="Depth / Point cloud", lines=3, interactive=False)

                    with gr.Tab("Depth Point Cloud"):
                        depth_plot = gr.Plot(label="GT Depth → Point Cloud")
                        depth_pc_info = gr.Textbox(
                            label="Point Cloud Info", lines=4, interactive=False)

                        gr.Markdown(
                            "**Step 1: Crop** — Remove ceiling and/or restrict to an XYZ "
                            "bounding box. Useful for excluding through-window points "
                            "that lift far beyond the room.")
                        with gr.Row():
                            depth_remove_ceiling_cb = gr.Checkbox(
                                label="Remove Ceiling", value=True,
                                info="Discard points within `threshold` meters of the highest Y.")
                            depth_ceiling_slider = gr.Slider(
                                0.0, 2.0, value=0.2, step=0.05,
                                label="Ceiling Threshold (m)")
                        with gr.Row():
                            depth_use_crop_cb = gr.Checkbox(
                                label="Enable XYZ Region Crop", value=False,
                                info="Keep only points inside the bbox below.")
                        with gr.Row():
                            depth_crop_x_min = gr.Number(label="X min", value=-10.0, precision=1)
                            depth_crop_x_max = gr.Number(label="X max", value=10.0, precision=1)
                            depth_crop_y_min = gr.Number(label="Y min", value=-10.0, precision=1)
                            depth_crop_y_max = gr.Number(label="Y max", value=10.0, precision=1)
                            depth_crop_z_min = gr.Number(label="Z min", value=-10.0, precision=1)
                            depth_crop_z_max = gr.Number(label="Z max", value=10.0, precision=1)
                        depth_crop_btn = gr.Button("Update Crop", variant="secondary")

                        gr.Markdown(
                            "**Step 2: Normalize** — Map to `[-0.5, 0.5]` (model input space). "
                            "Respects the crop above.")
                        depth_normalize_btn = gr.Button("Normalize", variant="primary")

                        gr.Markdown(
                            "Tip: After normalize, the camera center sits at "
                            "`(0 - center) * scale` in the same `[-0.5, 0.5]` frame.")

                    with gr.Tab("CSG"):
                        csg_viewer = gr.Model3D(
                            label="Predicted CSG (64^3)", height=500,
                            clear_color=WHITE_BG)

                    with gr.Tab("BBox + CSG"):
                        bbox_viewer = gr.Model3D(
                            label="Predicted BBox + CSG", height=500,
                            clear_color=WHITE_BG)

                    with gr.Tab("Mesh"):
                        with gr.Row():
                            with gr.Column():
                                combined_viewer = gr.Model3D(
                                    label="Overall Scene", height=350,
                                    clear_color=WHITE_BG)
                                combined_enlarge_btn = gr.Button("🔍 Enlarge Scene", size="sm")
                            with gr.Column():
                                layout_viewer = gr.Model3D(
                                    label="Layout", height=350,
                                    clear_color=WHITE_BG)
                                layout_enlarge_btn = gr.Button("🔍 Enlarge Layout", size="sm")
                            with gr.Column():
                                exploded_viewer = gr.Model3D(
                                    label="Assets (exploded)", height=350,
                                    clear_color=WHITE_BG)
                                exploded_enlarge_btn = gr.Button("🔍 Enlarge Assets", size="sm")
                        explode_slider = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05, label="Explosion Scale")

                        with gr.Accordion("Enlarged View", open=False) as enlarge_accordion:
                            with gr.Row():
                                enlarge_label = gr.Markdown("*Click an Enlarge button above.*")
                                enlarge_remove_ceiling_cb = gr.Checkbox(
                                    label="Remove Ceiling (top 15%)", value=False,
                                    info="Crop the upper 15% of the mesh by height "
                                         "so you can see inside the room.")
                            enlarge_viewer = gr.Model3D(
                                label="Enlarged", height=750,
                                clear_color=WHITE_BG)
                            # Hidden state remembering the original (un-cropped) GLB
                            enlarge_original_path = gr.State(value=None)

                    with gr.Tab("Save"):
                        gr.Markdown(f"### Save Results (under `{DEFAULT_OUTPUT_ROOT}`)")
                        save_btn = gr.Button("Save All", variant="primary", size="lg")
                        save_dir_box = gr.Textbox(label="Output Directory", interactive=False)
                        with gr.Row():
                            save_cubemap = gr.Image(label="Cubemap", height=200)
                            save_bbox = gr.Image(label="BBox", height=200)
                        with gr.Row():
                            save_ss = gr.Image(label="SS Exterior", height=200)
                            save_ss_int = gr.Image(label="SS Interior", height=200)
                        with gr.Row():
                            save_geom = gr.Image(label="Geometry Exterior", height=200)
                            save_geom_td = gr.Image(label="Geometry Top-Down", height=200)
                        with gr.Row():
                            save_tex_ext = gr.Image(label="Texture Exterior", height=200)
                            save_tex_int = gr.Image(label="Texture Interior", height=200)
                            save_tex_td = gr.Image(label="Texture Top-Down", height=200)

        # ============================================================
        # Wiring
        # ============================================================

        scene_dd.change(on_scene_change, [scene_dd], [variant_dd])

        load_btn.click(
            on_load_input,
            [scene_dd, variant_dd, light_dd, data_dir_state, state],
            [erp_image, depth_image, cubemap_image, depth_plot, state, depth_info_box, status_box],
        )

        depth_crop_btn.click(
            on_update_depth_pointcloud,
            [depth_remove_ceiling_cb, depth_ceiling_slider,
             depth_crop_x_min, depth_crop_x_max,
             depth_crop_y_min, depth_crop_y_max,
             depth_crop_z_min, depth_crop_z_max,
             depth_use_crop_cb, state],
            [depth_plot, state, depth_pc_info],
        )

        depth_normalize_btn.click(
            on_normalize_depth,
            [state],
            [depth_plot, state, depth_pc_info],
        )

        generate_csg_btn.click(
            on_generate_csg,
            [scene_dd, variant_dd, light_dd, data_dir_state,
             use_sdedit_cb, alpha_slider, csg_spatial_mask_cb,
             seed_slider, steps_s1, cfg_s1,
             csg_show_cam_cb, csg_max_cubes, state],
            [csg_viewer, state, status_box],
        )

        pred_bbox_btn.click(
            on_predict_bbox,
            [bbox_threshold, state],
            [bbox_viewer, state, status_box, bbox_source_display],
        )

        gen_scene_btn.click(
            on_generate_scene,
            [data_dir_state, seed_slider, steps_s2, cfg_s2,
             gen_texture_cb, layout_mode_radio, state],
            [combined_viewer, layout_viewer, exploded_viewer, state, status_box],
        )

        explode_slider.change(
            on_update_explode, [explode_slider, state], [exploded_viewer])

        # ---- Enlarge wiring ----
        def _enlarge(glb_path, label_text, remove_ceiling):
            display_path = glb_path
            if remove_ceiling and glb_path is not None:
                cut_path = remove_ceiling_from_glb(glb_path)
                if cut_path is not None:
                    display_path = cut_path
            return [
                gr.update(open=True),
                f"**{label_text}**",
                display_path,
                glb_path,  # remember the original so the toggle below can re-cut
            ]

        enlarge_outputs = [enlarge_accordion, enlarge_label,
                           enlarge_viewer, enlarge_original_path]

        combined_enlarge_btn.click(
            lambda glb, rc: _enlarge(glb, "Overall Scene", rc),
            [combined_viewer, enlarge_remove_ceiling_cb], enlarge_outputs,
        )
        layout_enlarge_btn.click(
            lambda glb, rc: _enlarge(glb, "Layout", rc),
            [layout_viewer, enlarge_remove_ceiling_cb], enlarge_outputs,
        )
        exploded_enlarge_btn.click(
            lambda glb, rc: _enlarge(glb, "Assets (exploded)", rc),
            [exploded_viewer, enlarge_remove_ceiling_cb], enlarge_outputs,
        )

        # Toggle ceiling removal on the already-enlarged model
        def _toggle_enlarge_ceiling(remove_ceiling, original_path):
            if original_path is None:
                return None
            if remove_ceiling:
                cut = remove_ceiling_from_glb(original_path)
                return cut if cut is not None else original_path
            return original_path

        enlarge_remove_ceiling_cb.change(
            _toggle_enlarge_ceiling,
            [enlarge_remove_ceiling_cb, enlarge_original_path],
            [enlarge_viewer],
        )

        save_btn.click(
            on_save_all,
            [data_dir_state, state],
            [save_cubemap, save_bbox, save_ss, save_ss_int, save_geom, save_geom_td,
             save_tex_ext, save_tex_int, save_tex_td, status_box, save_dir_box],
        )

    return demo


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='InSpace Structured3D Demo')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--light', type=str, default='rawlight', choices=LIGHT_CHOICES)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--share', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    ensure_demo_samples(args.data_dir)
    demo = create_demo(args.data_dir, args.light)
    # In gradio 6.x, theme/css moved from Blocks() to launch().
    launch_kwargs = dict(
        server_name='0.0.0.0',
        share=args.share,
        theme=INSPACE_THEME,
        css=INSPACE_CSS,
    )
    if args.port is not None:
        launch_kwargs['server_port'] = args.port
    demo.launch(**launch_kwargs)
