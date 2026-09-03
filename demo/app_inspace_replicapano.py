# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
InSpace ReplicaPano Gradio Interactive Demo

Interactive per-sample inference for the InSpace pipeline on ReplicaPano dataset.
Stages:
    1. Load Input (ERP -> cubemap on-the-fly, DA2 depth -> point cloud)
    2. Coarse Scene Geometry (SS flow -> 64^3 voxel, optional SDEdit from DA2)
    3. 3D BBOX Estimation (CenterPoint prediction, no GT available)
    4. Layout and Asset-Aware Scene Generation (Shape + Texture -> Mesh)

Key differences from demo/app_inspace_erp_front.py:
    - No room hierarchy (scene -> view directly)
    - No pre-computed cubemaps (ERP -> cubemap on-the-fly via py360convert)
    - No GT bboxes, GT voxels, or PSG data
    - Camera center from DA2 depth self-normalization
    - SDEdit uses inline step9+10 (DA2 depth -> voxel -> SS latent)

Usage:
    python demo/app_inspace_replicapano.py --port 7860
"""

import os
import sys
import time
import argparse
import traceback
from tqdm import tqdm

# Select the GPU before importing torch (must be set before the first torch import).
# Defaults to GPU 1; override from the shell, e.g.
#   CUDA_VISIBLE_DEVICES=0 python demo/app_inspace_replicapano.py
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print(f"[DEBUG] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import gradio as gr
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

# Reuse model management, inference, and viz from demo
from demo.app_inspace_utils import (
    model_manager,
    ensure_demo_samples,
    colorize_depth,
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
# ReplicaPano Data Loading
# ============================================================

DEFAULT_DATA_DIR = os.path.join(
    PROJECT_ROOT, 'datasets', 'Replicapano_samples')


def discover_replicapano_samples(data_dir):
    """Discover all ReplicaPano samples.
    Returns list of (scene_name, view_id) tuples.
    """
    samples = []
    for scene_name in sorted(os.listdir(data_dir)):
        scene_dir = os.path.join(data_dir, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        scene_info_dir = os.path.join(scene_dir, 'Scene_Info')
        if not os.path.isdir(scene_info_dir):
            continue
        for view_id in sorted(os.listdir(scene_info_dir)):
            view_dir = os.path.join(scene_info_dir, view_id)
            if not os.path.isdir(view_dir):
                continue
            rgb_path = os.path.join(view_dir, 'rgb.png')
            if os.path.exists(rgb_path):
                samples.append((scene_name, view_id))
    return samples


def erp_to_cubemap(erp_image, face_size=512, fov=120):
    """Convert ERP panorama to 6 cubemap faces."""
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


def load_cubemap_from_erp(data_dir, scene_name, view_id, image_size=512):
    """Load cubemap as [6, 3, H, W] tensor. Uses saved files if available."""
    face_order = ['front', 'right', 'back', 'left', 'top', 'bottom']

    # Check for saved cubemap faces first
    cubic_dir = os.path.join(data_dir, scene_name, 'Scene_Info', view_id, 'cubic_fov_120')
    all_saved = os.path.isdir(cubic_dir) and all(
        os.path.exists(os.path.join(cubic_dir, f'{fn}.png')) for fn in face_order)
    if all_saved:
        return load_cubemap_from_saved(cubic_dir, image_size)

    # Generate from ERP
    import torchvision.transforms as T
    erp_path = os.path.join(data_dir, scene_name, 'Scene_Info', view_id, 'rgb.png')
    erp_img = np.array(Image.open(erp_path).convert('RGB'))
    faces = erp_to_cubemap(erp_img, face_size=image_size, fov=120)

    transform = T.Compose([T.ToTensor()])
    tensors = [transform(faces[n].astype(np.uint8)) for n in face_order]
    return torch.stack(tensors)


def load_cubemap_from_saved(cubic_dir, image_size=512):
    """Load pre-saved cubemap face images, return [6, 3, H, W] tensor."""
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])
    face_order = ['front', 'right', 'back', 'left', 'top', 'bottom']
    tensors = []
    for fname in face_order:
        img = Image.open(os.path.join(cubic_dir, f'{fname}.png')).convert('RGB')
        tensors.append(transform(img))
    return torch.stack(tensors)


def erp_depth_to_point_cloud(rgb, depth, subsample=2, max_points=50000, max_depth=20.0,
                             remove_ceiling=False, ceiling_threshold=0.0):
    """ERP depth + RGB -> 3D point cloud with colors.
    Based on WorldGen/demo_3d_front_pointcloud_dap_da2.py

    Args:
        rgb: (H, W, 3) RGB image uint8
        depth: (H, W) depth map
        subsample: pixel subsample factor (2 = half resolution)
        max_points: max points for visualization (0 = no limit)
        max_depth: filter threshold
        remove_ceiling: remove ceiling points (highest Y region)
        ceiling_threshold: distance from Y_max to remove (meters)

    Returns:
        points: (N, 3) in camera-centered coords
        colors: (N, 3) RGB in 0-255 uint8
    """
    # Subsample for efficiency
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
    colors_flat = rgb.reshape(-1, 3)
    d_flat = depth.flatten()
    valid = np.isfinite(d_flat) & (d_flat > 0.01) & (d_flat < max_depth)

    points = points[valid]
    colors_flat = colors_flat[valid]

    # Remove ceiling (highest Y points in camera space)
    if remove_ceiling and len(points) > 0 and ceiling_threshold > 0:
        y_max = points[:, 1].max()
        ceiling_mask = points[:, 1] < (y_max - ceiling_threshold)
        points = points[ceiling_mask]
        colors_flat = colors_flat[ceiling_mask]

    # Limit point count for browser performance
    if max_points > 0 and len(points) > max_points:
        idx = np.random.RandomState(42).choice(len(points), max_points, replace=False)
        idx = np.sort(idx)
        points = points[idx]
        colors_flat = colors_flat[idx]

    return points, colors_flat.astype(np.uint8)


def self_normalize_point_cloud(points):
    """Self-normalize point cloud to [-0.5, 0.5]. Returns (points, center, scale)."""
    pts_min = points.min(axis=0)
    pts_max = points.max(axis=0)
    center = (pts_min + pts_max) / 2.0
    max_extent = (pts_max - pts_min).max()
    scale = 0.99999 / max_extent if max_extent > 1e-6 else 1.0
    normalized = np.clip((points - center) * scale, -0.5 + 1e-6, 0.5 - 1e-6)
    return normalized, center, scale


def load_da2_raw_point_cloud(data_dir, scene_name, view_id):
    """Step 1: Load DA2 depth + RGB, lift to raw 3D point cloud (no normalization).
    Returns dict with raw points, colors, depth, rgb, or None if unavailable.
    """
    da2_path = os.path.join(data_dir, scene_name, 'Scene_Info', view_id, 'depth_da2.npy')
    erp_path = os.path.join(data_dir, scene_name, 'Scene_Info', view_id, 'rgb.png')
    if not os.path.exists(da2_path):
        return None

    da2_depth = np.load(da2_path)
    erp_rgb = np.array(Image.open(erp_path).convert('RGB')) if os.path.exists(erp_path) else None
    if erp_rgb is None:
        erp_rgb = np.full((*da2_depth.shape, 3), 128, dtype=np.uint8)

    # Lift to raw 3D point cloud (subsampled for visualization)
    points_vis, colors_vis = erp_depth_to_point_cloud(
        erp_rgb, da2_depth, subsample=2, max_points=50000,
        remove_ceiling=False, ceiling_threshold=0.0)

    if len(points_vis) < 100:
        return None

    return {
        'points_raw': points_vis,        # raw camera-space coords, subsampled for viz
        'colors': colors_vis,            # RGB uint8
        'da2_depth': da2_depth,
        'erp_rgb': erp_rgb,
        'is_normalized': False,
        'is_cropped': False,
    }


def crop_da2_point_cloud(da2_data, remove_ceiling=False, ceiling_threshold=0.0,
                          crop_bbox=None):
    """Step 2: Crop the raw point cloud (ceiling removal + region selection).

    Args:
        da2_data: dict from load_da2_raw_point_cloud
        remove_ceiling: remove ceiling points
        ceiling_threshold: distance from Y_max to remove
        crop_bbox: (x_min, x_max, y_min, y_max, z_min, z_max) region to keep, or None

    Returns updated da2_data with cropped full-resolution points.
    """
    if da2_data is None:
        return None

    erp_rgb = da2_data['erp_rgb']
    da2_depth = da2_data['da2_depth']

    # Full-resolution lifting with ceiling removal
    points_full, colors_full = erp_depth_to_point_cloud(
        erp_rgb, da2_depth, subsample=1, max_points=0,
        remove_ceiling=remove_ceiling, ceiling_threshold=ceiling_threshold)

    if len(points_full) < 100:
        return None

    # Apply region crop (bbox filter) if specified
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

    # Subsampled for visualization (with same filters)
    points_vis, colors_vis = erp_depth_to_point_cloud(
        erp_rgb, da2_depth, subsample=2, max_points=50000,
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
    da2_data['points_raw'] = points_vis       # cropped, subsampled for viz
    da2_data['colors'] = colors_vis
    da2_data['points_full_raw'] = points_full # cropped, full resolution
    da2_data['colors_full'] = colors_full
    da2_data['is_cropped'] = True
    da2_data['is_normalized'] = False
    da2_data['crop_settings'] = {
        'remove_ceiling': remove_ceiling,
        'ceiling_threshold': ceiling_threshold,
        'crop_bbox': crop_bbox,
    }
    return da2_data


def cam_to_world_points(points_cam):
    """
    Convert camera-centered coordinates to world coordinates.

    The ERP camera has rotation [pi/2, 0, 0] (Euler XYZ), meaning:
    - Camera forward (-Z) -> World +Y
    - Camera up (+Y) -> World +Z (height)
    - Camera right (+X) -> World +X

    Args:
        points_cam: (N, 3) points in camera space [x_right, y_up, z_backward]
    Returns:
        points_world: (N, 3) points in world coords [x, y, z_height]
    """
    return np.column_stack([
        points_cam[:, 0],    # cam X -> world X
        -points_cam[:, 2],   # cam -Z -> world Y
        points_cam[:, 1],    # cam Y -> world Z (height)
    ])


def normalize_da2_point_cloud(da2_data):
    """Step 3: Normalize the cropped point cloud to [-0.5, 0.5] for generation model.

    Applies cam_to_world rotation so that Z=height (matching step9/step10 pre-saved data
    and the model's expected coordinate system).
    """
    if da2_data is None:
        return None

    points_full = da2_data.get('points_full_raw')
    if points_full is None or len(points_full) < 100:
        return None

    # Convert camera space (Y=up) -> world space (Z=height) to match model input
    points_full_world = cam_to_world_points(points_full)

    # Normalize using full point cloud
    points_norm_full, center, scale = self_normalize_point_cloud(points_full_world)
    camera_center = (np.array([0.0, 0.0, 0.0]) - center) * scale

    # Normalize visualization points with same center/scale
    points_vis = da2_data['points_raw']
    points_vis_world = cam_to_world_points(points_vis)
    points_vis_norm = np.clip((points_vis_world - center) * scale, -0.5 + 1e-6, 0.5 - 1e-6)

    da2_data = dict(da2_data)
    da2_data['points'] = points_vis_norm         # normalized, subsampled for viz
    da2_data['points_full'] = points_norm_full   # normalized, full resolution
    da2_data['camera_center'] = camera_center
    da2_data['center'] = center
    da2_data['scale'] = scale
    da2_data['is_normalized'] = True
    da2_data['n_points_full'] = len(points_norm_full)
    return da2_data


def voxelize_point_cloud(points_norm, grid_size=64):
    """Convert normalized point cloud to voxel grid and PLY-compatible centers.

    Args:
        points_norm: (N, 3) normalized points in [-0.5, 0.5]
        grid_size: voxel resolution

    Returns:
        occ_grid: (1, 1, G, G, G) float32 tensor
        voxel_centers: (M, 3) unique voxel centers in [-0.5, 0.5]
        unique_voxels: (M, 3) voxel index coords
    """
    voxel_indices = ((points_norm + 0.5) * grid_size).astype(np.int32)
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)
    unique_voxels = np.unique(voxel_indices, axis=0)

    occ_grid = torch.zeros(1, 1, grid_size, grid_size, grid_size, dtype=torch.float32)
    for c in unique_voxels:
        occ_grid[0, 0, c[0], c[1], c[2]] = 1.0

    voxel_centers = (unique_voxels + 0.5) / grid_size - 0.5
    return occ_grid, voxel_centers, unique_voxels


def save_depth_voxels_ply(voxel_centers, output_path):
    """Save voxel centers as PLY with position-based RGB colors.
    Compatible with depth_voxels_da2_64 format used by step9.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    colors = ((voxel_centers + 0.5) * 255).clip(0, 255).astype(np.uint8)

    header = f"""ply
format ascii 1.0
element vertex {len(voxel_centers)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(output_path, 'w') as f:
        f.write(header)
        for i in range(len(voxel_centers)):
            f.write(f"{voxel_centers[i,0]:.6f} {voxel_centers[i,1]:.6f} {voxel_centers[i,2]:.6f} "
                    f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")
    return output_path


def load_sdedit_latent_from_saved_voxels(data_dir, scene_name, view_id, device='cuda'):
    """Load pre-saved depth_voxels_da2_64 PLY, encode to SS latent.
    ReplicaPano path: Scene_Info/{view_id}/depth_voxels_da2_64/{view_id}.ply
    Returns (z_numpy [8,16,16,16], camera_center) or (None, None).
    """
    import trimesh
    view_dir = os.path.join(data_dir, scene_name, 'Scene_Info', view_id)
    ply_path = os.path.join(view_dir, 'depth_voxels_da2_64', f'{view_id}.ply')
    norm_path = os.path.join(view_dir, 'depth_voxels_da2_64', 'normalization_info.json')

    if not os.path.exists(ply_path):
        return None, None

    # Load PLY
    pc = trimesh.load(ply_path, process=False)
    verts = np.array(pc.vertices, dtype=np.float32) if hasattr(pc, 'vertices') else None
    if verts is None or len(verts) < 10:
        return None, None

    # Convert to occupancy grid
    coords = ((verts + 0.5) * 64).astype(np.int32)
    coords = np.clip(coords, 0, 63)
    occ_grid = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float32)
    for c in coords:
        occ_grid[0, 0, c[0], c[1], c[2]] = 1.0
    occ_grid = occ_grid.to(device)

    # Encode
    ss_encoder = model_manager.get_ss_encoder()
    with torch.no_grad():
        z = ss_encoder(occ_grid, sample_posterior=False)

    z_np = z[0].cpu().numpy()

    # Load camera center
    camera_center = np.zeros(3)
    if os.path.exists(norm_path):
        import json as json_mod
        with open(norm_path, 'r') as f:
            norm_info = json_mod.load(f)
        camera_center = np.array(norm_info.get('camera_center', [0, 0, 0]), dtype=np.float64)

    log(f"[SDEdit] Loaded saved voxels: {len(verts)} voxels -> SS latent "
        f"range=[{z_np.min():.3f}, {z_np.max():.3f}], std={z_np.std():.4f}")
    return z_np, camera_center


def compute_sdedit_initial_latent(da2_data, device='cuda'):
    """Inline step9+10: DA2 point cloud (pre-computed) -> voxelize -> SS latent.
    Uses full-resolution normalized points from da2_data.
    Returns (z, camera_center) or (None, None).
    """
    if da2_data is None:
        return None, None

    points_norm = da2_data['points_full']
    camera_center = da2_data['camera_center']

    if len(points_norm) < 100:
        return None, None

    occ_grid, voxel_centers, unique_voxels = voxelize_point_cloud(points_norm, grid_size=64)
    n_occupied = int(occ_grid.sum().item())
    log(f"[SDEdit] Voxelized: {len(points_norm)} pts -> {n_occupied} occupied voxels (64^3)")
    occ_grid = occ_grid.to(device)

    ss_encoder = model_manager.get_ss_encoder()
    with torch.no_grad():
        z = ss_encoder(occ_grid, sample_posterior=False)
    log(f"[SDEdit] SS latent: shape={list(z.shape)}, range=[{z.min():.3f}, {z.max():.3f}], std={z.std():.3f}")

    return z[0].cpu().numpy(), camera_center


# ============================================================
# ReplicaPano-specific Stage 1
# ============================================================

@torch.no_grad()
def run_stage1_replicapano(
    data_dir, scene_name, view_id,
    use_sdedit=False, alpha=0.5,
    da2_data=None,
    steps=12, cfg_strength=7.5, seed=42,
    use_spatial_mask=True,
):
    """Stage 1 for ReplicaPano: ERP -> cubemap on-the-fly -> SS latent -> 64^3 voxel."""
    t0 = time.time()
    log(f"[Stage 1] ReplicaPano CSG generation (seed={seed})")
    device = model_manager.device
    config = model_manager._load_config(DEFAULT_STAGE1_CONFIG)
    trainer_config = config['trainer']['args']

    # Load models
    denoiser, ss_decoder = model_manager.get_stage1()
    erp_encoder = model_manager.get_erp_encoder()

    # Encode cubemap (on-the-fly from ERP)
    t1 = time.time()
    log(f"[Stage 1] Converting ERP -> cubemap -> encoding...")
    cond = load_cubemap_from_erp(data_dir, scene_name, view_id).unsqueeze(0).to(device)
    # Debug: log cubemap stats and save visualization
    log(f"[Stage 1] Cubemap tensor: shape={list(cond.shape)}, "
        f"range=[{cond.min():.3f}, {cond.max():.3f}], dtype={cond.dtype}")
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    for i, fn in enumerate(face_names):
        face = cond[0, i]  # [3, H, W]
        log(f"[Stage 1]   {fn}: mean={face.mean():.3f}, std={face.std():.3f}, "
            f"range=[{face.min():.3f}, {face.max():.3f}]")
    # Save cubemap debug image
    try:
        import torchvision.utils as vutils
        debug_dir = os.path.join('/tmp', 'stride_debug_cubemap')
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f'{scene_name}_{view_id}_cubemap.png')
        grid = vutils.make_grid(cond[0], nrow=3, padding=2)  # [3, H, W] grid of 6 faces
        from torchvision.utils import save_image
        save_image(grid, debug_path)
        log(f"[Stage 1] Debug cubemap saved: {debug_path}")
    except Exception as e:
        log(f"[Stage 1] Debug cubemap save failed: {e}")

    encoded_cond = erp_encoder(cond)
    neg_cond = torch.zeros_like(encoded_cond)
    log(f"[Stage 1] Cubemap encoded ({time.time()-t1:.1f}s), cond range=[{encoded_cond.min():.3f}, {encoded_cond.max():.3f}], std={encoded_cond.std():.3f}")

    # Prepare noise
    torch.manual_seed(seed)
    sigma_min = trainer_config.get('sigma_min', 1e-5)

    sdedit_ok = False
    if use_sdedit and da2_data is not None:
        # Check for pre-saved voxels override first
        psg_ss_latent = da2_data.get('_psg_ss_latent_override')
        if psg_ss_latent is not None:
            log(f"[Stage 1] SDEdit from pre-saved voxels (alpha={alpha})")
        else:
            log(f"[Stage 1] SDEdit from DA2 depth inline (alpha={alpha})")
            psg_ss_latent, _ = compute_sdedit_initial_latent(da2_data, device)
        if psg_ss_latent is not None:
            x_init = torch.from_numpy(psg_ss_latent).float().unsqueeze(0).to(device)
            log(f"[Stage 1] x_init range=[{x_init.min():.3f}, {x_init.max():.3f}], "
                f"std={x_init.std():.4f}, mean={x_init.mean():.4f}, "
                f"abs_mean={x_init.abs().mean():.4f}")
            t_val = alpha
            gaussian_noise = torch.randn_like(x_init)
            noise = (1 - t_val) * x_init + (sigma_min + (1 - sigma_min) * t_val) * gaussian_noise
            log(f"[Stage 1] noise range=[{noise.min():.3f}, {noise.max():.3f}], "
                f"std={noise.std():.4f}")
            sdedit_ok = True
        else:
            log("[Stage 1] SDEdit failed, falling back to random noise")
            noise = torch.randn(1, 8, 16, 16, 16, device=device)
    else:
        log("[Stage 1] Using random Gaussian noise")
        noise = torch.randn(1, 8, 16, 16, 16, device=device)

    # Camera center for spatial attention
    camera_center_np = da2_data['camera_center'] if da2_data else np.zeros(3)
    camera_center_t = torch.from_numpy(camera_center_np).float()

    extra_kwargs = {}
    use_spatial = trainer_config.get('use_spatial_attention', False) and use_spatial_mask
    if not use_spatial_mask:
        log("[Stage 1] Spatial attention mask DISABLED by user")
    if use_spatial:
        log("[Stage 1] Creating spatial attention mask...")
        cross_attn_mask = create_spatial_attention_mask(
            camera_center=camera_center_t.unsqueeze(0).to(device),
            voxel_resolution=trainer_config.get('voxel_resolution', 16),
            tokens_per_face=trainer_config.get('tokens_per_face', 1029),
            fov_degrees=trainer_config.get('spatial_attention_fov', 120.0),
            soft_mask=trainer_config.get('spatial_attention_soft', True),
            soft_margin=trainer_config.get('spatial_attention_soft_margin', 0.1),
        )
        extra_kwargs['cross_attn_mask'] = cross_attn_mask

    # Sample
    sampler = model_manager.get_sampler()
    t1 = time.time()
    rescale_t = 5.0

    if sdedit_ok:
        # SDEdit: denoise from t=alpha -> t=0 (no rescale_t).
        # noise was mixed at unrescaled t=alpha (line ~605), so the schedule must
        # also stay unrescaled — otherwise the model is fed a t (e.g. 0.952) that
        # doesn't match the actual noise level (alpha=0.8) and the latent collapses
        # to ~0. Matches training-mixin SDEdit (erp_image_conditioned.py:367-369).
        start_t = alpha
        t_seq = np.linspace(start_t, 0, steps + 1)
        # Rescale the schedule to the model's training-time convention (rescale_t=5.0);
        # feeding un-rescaled t makes the denoiser output collapse toward zero -> 0 voxels.
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        log(f"[Stage 1] SDEdit sampling from t={start_t:.2f} -> 0 ({steps} steps, cfg={cfg_strength}, rescale_t={rescale_t})")
        log(f"[Stage 1]   t_seq (rescaled): [{t_seq[0]:.3f}, ..., {t_seq[-1]:.3f}]")
        # ---- previous (buggy: rescaled schedule with unrescaled noise) ----
        # t_seq = np.linspace(start_t, 0, steps + 1)
        # t_seq_rescaled = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        # t_pairs = list((t_seq_rescaled[i], t_seq_rescaled[i + 1]) for i in range(steps))
        # log(f"[Stage 1]   t_rescaled: [{t_seq_rescaled[0]:.3f}, ..., {t_seq_rescaled[-1]:.3f}]")

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
        # Standard generation: t=1.0 -> t=0
        log(f"[Stage 1] Sampling ({steps} steps, cfg={cfg_strength}, rescale_t={rescale_t})...")
        with torch.autocast('cuda', dtype=torch.bfloat16):
            res = sampler.sample(
                denoiser, noise=noise, cond=encoded_cond, neg_cond=neg_cond,
                steps=steps, rescale_t=rescale_t,
                guidance_strength=cfg_strength,
                guidance_interval=(0.6, 1.0), guidance_rescale=0.7,
                verbose=True, **extra_kwargs,
            )
        z = res.samples

    torch.cuda.synchronize()
    log(f"[Stage 1] Sampling done ({time.time()-t1:.1f}s)")
    log(f"[Stage 1] z range=[{z.min():.3f}, {z.max():.3f}], shape={list(z.shape)}")
    voxel = ss_decoder(z.float())
    log(f"[Stage 1] voxel raw range=[{voxel.min():.3f}, {voxel.max():.3f}], shape={list(voxel.shape)}")
    # Check different thresholds for debug
    for thresh in [0, -1, -2, -5]:
        n = (voxel > thresh).sum().item()
        log(f"[Stage 1]   threshold>{thresh}: {n} voxels")
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
# Monkey-patch load_cubemap_images for ReplicaPano compatibility
# ============================================================

_REPLICAPANO_DATA_DIR = None  # Set during init

_original_load_cubemap_images = _stride_utils.load_cubemap_images


def _patched_load_cubemap_images(data_dir, scene_id, room_id, view_idx=0):
    """ReplicaPano-aware cubemap loader.
    If the standard ERP_3D_FRONT path doesn't exist, try ReplicaPano layout.
    For ReplicaPano, scene_id=scene_name, room_id=view_id, view_idx ignored.
    """
    # Try standard ERP_3D_FRONT path first
    cubic_dir = os.path.join(data_dir, scene_id, room_id, 'cubic_fov_120')
    if os.path.isdir(cubic_dir):
        return _original_load_cubemap_images(data_dir, scene_id, room_id, view_idx)

    # ReplicaPano: scene_id=scene_name, room_id=view_id
    # Check if cubemaps were already saved to disk
    rp_cubic_dir = os.path.join(data_dir, scene_id, 'Scene_Info', room_id, 'cubic_fov_120')
    if os.path.isdir(rp_cubic_dir) and os.path.exists(os.path.join(rp_cubic_dir, 'front.png')):
        return load_cubemap_from_saved(rp_cubic_dir)

    erp_path = os.path.join(data_dir, scene_id, 'Scene_Info', room_id, 'rgb.png')
    if os.path.exists(erp_path):
        return load_cubemap_from_erp(data_dir, scene_id, room_id)

    raise FileNotFoundError(
        f"Cannot find cubemap data for {scene_id}/{room_id} in {data_dir}")


# Apply monkey-patch so run_stage2_shape_single/texture_single use ReplicaPano paths
_stride_utils.load_cubemap_images = _patched_load_cubemap_images


# ============================================================
# Save function (ReplicaPano version)
# ============================================================

def save_all_results_replicapano(output_dir, state, meshes_list=None, data_dir=DEFAULT_DATA_DIR):
    """Save visualization results for ReplicaPano."""
    from demo.app_inspace_utils import save_all_results
    return save_all_results(output_dir, state, meshes_list, data_dir)


# ============================================================
# Dataset Discovery
# ============================================================

SAMPLES = []
SCENE_IDS = []
SCENE_VIEWS = {}  # scene_name -> [view_ids]


def init_dataset(data_dir):
    global SAMPLES, SCENE_IDS, SCENE_VIEWS
    SAMPLES = discover_replicapano_samples(data_dir)
    SCENE_VIEWS = {}
    for scene_name, view_id in SAMPLES:
        if scene_name not in SCENE_VIEWS:
            SCENE_VIEWS[scene_name] = []
        SCENE_VIEWS[scene_name].append(view_id)
    SCENE_IDS = sorted(SCENE_VIEWS.keys())


# ============================================================
# Event Handlers
# ============================================================

def on_scene_change(scene_name):
    """Update view dropdown when scene changes."""
    views = SCENE_VIEWS.get(scene_name, [])
    if views:
        return gr.update(choices=views, value=views[0])
    return gr.update(choices=[], value=None)


def on_load_input(scene_name, view_id, data_dir, state):
    """Load and display ERP image, cubemap, and DA2 point cloud."""
    try:
        t0 = time.time()
        state = dict(state) if state else {}
        state['scene_name'] = scene_name
        state['view_id'] = view_id

        # ERP panorama
        erp_path = os.path.join(data_dir, scene_name, 'Scene_Info', view_id, 'rgb.png')
        erp_img = Image.open(erp_path).convert('RGB') if os.path.exists(erp_path) else None
        log(f"[Load] ERP image: {time.time()-t0:.2f}s")

        # ERP depth map (visualization): DA2 raw npy (colormapped) or pre-rendered png
        depth_img = None
        _dd = os.path.join(data_dir, scene_name, 'Scene_Info', view_id)
        _npy, _png = os.path.join(_dd, 'depth_da2.npy'), os.path.join(_dd, 'depth_da2.png')
        if os.path.exists(_npy):
            depth_img = colorize_depth(np.load(_npy))
        elif os.path.exists(_png):
            depth_img = Image.open(_png).convert('RGB')

        # Cubemap — load from disk if available, otherwise generate and save
        t1 = time.time()
        face_order = ['front', 'right', 'back', 'left', 'top', 'bottom']
        sample_dir = os.path.join(data_dir, scene_name, 'Scene_Info', view_id)
        cubic_dir = os.path.join(sample_dir, 'cubic_fov_120')
        concat_dir = os.path.join(sample_dir, 'cubic_fov_120_concat')

        cubemap_grid = None
        # Check if cubemaps already exist on disk
        all_faces_exist = os.path.isdir(cubic_dir) and all(
            os.path.exists(os.path.join(cubic_dir, f'{fn}.png')) for fn in face_order)

        if all_faces_exist:
            # Load saved faces for display (256px thumbnails)
            faces_display = {}
            for fn in face_order:
                img = Image.open(os.path.join(cubic_dir, f'{fn}.png')).convert('RGB')
                faces_display[fn] = np.array(img.resize((256, 256)))
            log(f"[Load] Cubemap loaded from disk: {time.time()-t1:.2f}s")
        elif erp_img is not None:
            # Generate from ERP and save to disk
            erp_np = np.array(erp_img)
            faces_512 = erp_to_cubemap(erp_np, face_size=512, fov=120)

            # Save individual cubemap faces
            os.makedirs(cubic_dir, exist_ok=True)
            for fname in face_order:
                face_img = Image.fromarray(faces_512[fname].astype(np.uint8))
                face_img.save(os.path.join(cubic_dir, f'{fname}.png'))

            # Save concatenated cubemap
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            os.makedirs(concat_dir, exist_ok=True)
            fig_save, axes_save = plt.subplots(2, 3, figsize=(12, 8))
            for fi, fname in enumerate(face_order):
                r, c = fi // 3, fi % 3
                axes_save[r, c].imshow(faces_512[fname].astype(np.uint8))
                axes_save[r, c].set_title(fname)
                axes_save[r, c].axis('off')
            plt.tight_layout()
            fig_save.savefig(os.path.join(concat_dir, f'{view_id}_concat.png'), bbox_inches='tight', dpi=150)
            plt.close(fig_save)

            # Downsize for display
            faces_display = {fn: np.array(Image.fromarray(
                faces_512[fn].astype(np.uint8)).resize((256, 256)))
                for fn in face_order}
            log(f"[Load] Cubemap generated & saved: {time.time()-t1:.2f}s")
        else:
            faces_display = None

        if faces_display is not None:
            # Build 2x3 concat image with face name labels
            t_concat = time.time()
            from PIL import ImageDraw, ImageFont
            face_size = 256
            label_h = 20  # height for label text
            cell_h = face_size + label_h
            grid_w, grid_h = 3 * face_size, 2 * cell_h
            grid_img = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
            draw = ImageDraw.Draw(grid_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            except (OSError, IOError):
                font = ImageFont.load_default()
            for fi, fname in enumerate(face_order):
                r, c = fi // 3, fi % 3
                x_off = c * face_size
                y_off = r * cell_h
                # Draw label
                draw.text((x_off + face_size // 2, y_off + 2), fname,
                          fill=(0, 0, 0), font=font, anchor='mt')
                # Paste face image below label
                grid_img.paste(
                    Image.fromarray(faces_display[fname]),
                    (x_off, y_off + label_h))
            cubemap_grid = grid_img
            log(f"[Load] Cubemap concat: {time.time()-t_concat:.2f}s")

        # DA2 point cloud — check for pre-saved voxels, otherwise do Step 1: raw lifting
        t2 = time.time()
        sample_dir_da2 = os.path.join(data_dir, scene_name, 'Scene_Info', view_id)
        saved_voxel_dir = os.path.join(sample_dir_da2, 'depth_voxels_da2_64')
        saved_ply = os.path.join(saved_voxel_dir, f'{view_id}.ply')
        saved_norm = os.path.join(saved_voxel_dir, 'normalization_info.json')

        da2_data = None
        psg_fig = None
        da2_status = ""

        if os.path.exists(saved_ply) and os.path.exists(saved_norm):
            # Load pre-saved normalized voxels + normalization info
            import json
            with open(saved_norm, 'r') as f:
                norm_info = json.load(f)
            # Load PLY vertices
            verts = []
            with open(saved_ply, 'r') as f:
                in_header = True
                for line in f:
                    if in_header:
                        if line.strip() == 'end_header':
                            in_header = False
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            verts = np.array(verts, dtype=np.float32) if verts else np.zeros((0, 3))
            colors_ply = ((verts + 0.5) * 255).clip(0, 255).astype(np.uint8)

            center = np.array(norm_info['center'], dtype=np.float64)
            scale = float(norm_info['scale'])
            camera_center = np.array(norm_info['camera_center'], dtype=np.float64)

            # Also load raw depth + rgb for potential re-cropping
            da2_path = os.path.join(sample_dir_da2, 'depth_da2.npy')
            erp_path = os.path.join(sample_dir_da2, 'rgb.png')
            da2_depth = np.load(da2_path) if os.path.exists(da2_path) else None
            erp_rgb = np.array(Image.open(erp_path).convert('RGB')) if os.path.exists(erp_path) else None

            da2_data = {
                'points': verts,             # normalized, for viz
                'colors': colors_ply,        # position-based colors
                'points_full': verts,        # normalized, all voxel centers
                'camera_center': camera_center,
                'center': center,
                'scale': scale,
                'da2_depth': da2_depth,
                'erp_rgb': erp_rgb,
                'is_normalized': True,
                'is_cropped': True,
                'n_points_full': len(verts),
                'crop_settings': {
                    'remove_ceiling': norm_info.get('remove_ceiling', False),
                    'ceiling_threshold': norm_info.get('ceiling_threshold', 0.0),
                    'crop_bbox': norm_info.get('crop_bbox'),
                },
                'loaded_from_saved': True,
            }
            state['da2_data'] = da2_data

            psg_fig = create_psg_plotly_figure(
                verts, colors_ply, camera_center,
                show_camera_center=True, show_coordinates=True,
            )
            cc = camera_center
            da2_status = (f"Loaded saved voxels: {len(verts)} voxels (normalized)\n"
                          f"Camera center: [{cc[0]:.3f}, {cc[1]:.3f}, {cc[2]:.3f}], scale={scale:.4f}\n"
                          f"Source: {saved_ply}")
            log(f"[Load] DA2 loaded from saved: {time.time()-t2:.2f}s ({len(verts)} voxels)")
        else:
            # No pre-saved data — do raw lifting (Step 1)
            da2_data = load_da2_raw_point_cloud(data_dir, scene_name, view_id)
            state['da2_data'] = da2_data
            log(f"[Load] DA2 raw lift: {time.time()-t2:.2f}s")

        t3 = time.time()
        if da2_data is not None and not da2_data.get('is_normalized', False):
            pts = da2_data['points_raw']
            # Camera is at origin in raw camera-space coordinates
            raw_cam = np.array([0.0, 0.0, 0.0])
            psg_fig = create_psg_plotly_figure(
                pts, da2_data['colors'],
                camera_center=raw_cam,
                show_camera_center=True, show_coordinates=True,
            )
            n_pts = len(pts)
            pt_range = (f"X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
                        f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
                        f"Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
            da2_status = (f"Raw point cloud: {n_pts} pts (not normalized)\n"
                          f"Camera center: [0.000, 0.000, 0.000] (origin)\n{pt_range}")
        log(f"[Load] Plotly figure: {time.time()-t3:.2f}s")

        status = f"Loaded: {scene_name}/{view_id} ({time.time()-t0:.1f}s)"
        if da2_status:
            status += f"\n{da2_status}"

        return erp_img, cubemap_grid, depth_img, psg_fig, state, da2_status, status

    except Exception as e:
        traceback.print_exc()
        return None, None, None, None, state, "", f"Error loading input: {e}"


def on_relift_da2(data_dir, state):
    """Re-lift DA2 depth to raw 3D point cloud with RGB colors (overrides saved voxels)."""
    try:
        state = dict(state) if state else {}
        scene_name = state.get('scene_name')
        view_id = state.get('view_id')
        if not scene_name or not view_id:
            return None, state, "Load input first."

        t0 = time.time()
        da2_data = load_da2_raw_point_cloud(data_dir, scene_name, view_id)
        state['da2_data'] = da2_data

        if da2_data is None:
            return None, state, "DA2 depth not available."

        pts = da2_data['points_raw']
        raw_cam = np.array([0.0, 0.0, 0.0])
        psg_fig = create_psg_plotly_figure(
            pts, da2_data['colors'],
            camera_center=raw_cam,
            show_camera_center=True, show_coordinates=True,
        )
        n_pts = len(pts)
        pt_range = (f"X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
                    f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
                    f"Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
        info = (f"Raw RGB point cloud: {n_pts} pts (not normalized)\n"
                f"Camera center: [0.000, 0.000, 0.000] (origin)\n{pt_range}"
                f" ({time.time()-t0:.1f}s)")

        return psg_fig, state, info

    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error: {e}"


def on_update_da2_pointcloud(data_dir, remove_ceiling, ceiling_threshold,
                             crop_x_min, crop_x_max, crop_y_min, crop_y_max,
                             crop_z_min, crop_z_max, use_crop_region,
                             state):
    """Step 2: Crop the raw DA2 point cloud (ceiling removal + region selection).
    Shows the cropped but UN-normalized point cloud so user can inspect.
    """
    try:
        state = dict(state) if state else {}
        da2_data = state.get('da2_data')
        if da2_data is None:
            return None, state, "Load input first."

        t0 = time.time()

        # Build crop bbox if region cropping is enabled
        crop_bbox = None
        if use_crop_region:
            crop_bbox = (crop_x_min, crop_x_max, crop_y_min, crop_y_max, crop_z_min, crop_z_max)

        da2_data = crop_da2_point_cloud(
            da2_data,
            remove_ceiling=remove_ceiling,
            ceiling_threshold=ceiling_threshold,
            crop_bbox=crop_bbox)
        state['da2_data'] = da2_data

        if da2_data is None:
            return None, state, "Too few points after cropping."

        pts = da2_data['points_raw']
        psg_fig = create_psg_plotly_figure(
            pts, da2_data['colors'],
            camera_center=None,
            show_camera_center=False, show_coordinates=True,
        )

        n_vis = len(pts)
        n_full = len(da2_data.get('points_full_raw', []))
        pt_range = (f"X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
                    f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
                    f"Z[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")
        info = f"Cropped: {n_full} pts (vis: {n_vis}), NOT normalized\n{pt_range}"
        if remove_ceiling:
            info += f"\nCeiling removed (threshold={ceiling_threshold:.2f}m)"
        if use_crop_region:
            info += f"\nRegion crop active"
        info += f" ({time.time()-t0:.1f}s)"

        return psg_fig, state, info

    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error: {e}"


def on_normalize_da2(show_camera, state):
    """Step 3: Normalize the cropped point cloud to [-0.5, 0.5] for generation model."""
    try:
        state = dict(state) if state else {}
        da2_data = state.get('da2_data')
        if da2_data is None:
            return None, state, "Load input first."

        if not da2_data.get('is_cropped', False):
            # If user hasn't explicitly cropped, do a default crop (no ceiling, no bbox)
            da2_data = crop_da2_point_cloud(da2_data)
            if da2_data is None:
                return None, state, "Too few points."

        t0 = time.time()
        da2_data = normalize_da2_point_cloud(da2_data)
        state['da2_data'] = da2_data

        if da2_data is None:
            return None, state, "Normalization failed (too few points)."

        psg_fig = create_psg_plotly_figure(
            da2_data['points'], da2_data['colors'],
            da2_data['camera_center'],
            show_camera_center=show_camera, show_coordinates=True,
        )

        n_pts = len(da2_data['points'])
        n_full = da2_data['n_points_full']
        cc = da2_data['camera_center']
        info = (f"Normalized: {n_full} pts (vis: {n_pts})\n"
                f"cam=[{cc[0]:.3f}, {cc[1]:.3f}, {cc[2]:.3f}], scale={da2_data['scale']:.4f}")
        info += f" ({time.time()-t0:.1f}s)"

        return psg_fig, state, info

    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error: {e}"


def on_save_depth_voxels(data_dir, state):
    """Save normalized DA2 point cloud as depth_voxels_da2_64 PLY.
    Requires normalization to have been done first (Step 3).
    """
    try:
        state = dict(state) if state else {}
        scene_name = state.get('scene_name')
        view_id = state.get('view_id')
        da2_data = state.get('da2_data')

        if da2_data is None:
            return "Load input first (need DA2 data)."

        if not da2_data.get('is_normalized', False):
            return "Please normalize the point cloud first (click 'Normalize')."

        # Voxelize at 64^3
        _, voxel_centers, unique_voxels = voxelize_point_cloud(
            da2_data['points_full'], grid_size=64)

        # Save PLY
        sample_dir = os.path.join(data_dir, scene_name, 'Scene_Info', view_id)
        output_dir = os.path.join(sample_dir, 'depth_voxels_da2_64')
        output_path = os.path.join(output_dir, f'{view_id}.ply')
        save_depth_voxels_ply(voxel_centers, output_path)

        # Also save normalization info for later use
        crop_settings = da2_data.get('crop_settings', {})
        norm_info = {
            'center': da2_data['center'].tolist(),
            'scale': float(da2_data['scale']),
            'camera_center': da2_data['camera_center'].tolist(),
            'remove_ceiling': crop_settings.get('remove_ceiling', False),
            'ceiling_threshold': float(crop_settings.get('ceiling_threshold', 0.0)),
            'crop_bbox': crop_settings.get('crop_bbox'),
        }
        import json
        norm_path = os.path.join(output_dir, 'normalization_info.json')
        with open(norm_path, 'w') as f:
            json.dump(norm_info, f, indent=2)

        status = f"Saved: {output_path}\n{len(voxel_centers)} voxels (64^3)"
        status += f"\nNorm info: {norm_path}"
        log(f"[Save] {status}")
        return status

    except Exception as e:
        traceback.print_exc()
        return f"Error saving: {e}"


def on_switch_da2_view(view_mode, data_dir, state):
    """Switch DA2 tab between 3D point cloud view and pre-saved voxel (decoded) view."""
    try:
        state = dict(state) if state else {}
        scene_name = state.get('scene_name')
        view_id = state.get('view_id')

        if scene_name is None or view_id is None:
            return None, state, "Load input first."

        if view_mode == "Pre-saved Voxels (decoded)":
            import trimesh
            view_dir = os.path.join(data_dir, scene_name, 'Scene_Info', view_id)
            ply_path = os.path.join(view_dir, 'depth_voxels_da2_64', f'{view_id}.ply')
            norm_path = os.path.join(view_dir, 'depth_voxels_da2_64', 'normalization_info.json')

            if not os.path.exists(ply_path):
                return None, state, f"No pre-saved voxels at: {ply_path}"

            # Load saved PLY -> encode -> decode to 64^3 voxel for visualization
            pc = trimesh.load(ply_path, process=False)
            verts = np.array(pc.vertices, dtype=np.float32)

            if len(verts) < 10:
                return None, state, "Pre-saved voxels file is nearly empty."

            # Build occupancy grid
            coords = ((verts + 0.5) * 64).astype(np.int32)
            coords = np.clip(coords, 0, 63)
            occ_grid = torch.zeros(1, 1, 64, 64, 64, dtype=torch.float32)
            for c in coords:
                occ_grid[0, 0, c[0], c[1], c[2]] = 1.0

            # Encode + decode to see what the model actually gets
            ss_encoder = model_manager.get_ss_encoder()
            occ_grid_cuda = occ_grid.cuda()
            with torch.no_grad():
                z = ss_encoder(occ_grid_cuda, sample_posterior=False)

            _, ss_decoder = model_manager.get_stage1()
            with torch.no_grad():
                voxel_64 = ss_decoder(z)
            voxel_64 = (voxel_64 > 0).cpu().numpy()[0, 0]  # [64,64,64] bool

            occupied = np.argwhere(voxel_64)
            n_active = len(occupied)
            if n_active == 0:
                return None, state, "Decoded voxel grid is empty."

            voxel_centers = (occupied.astype(np.float32) + 0.5) / 64.0 - 0.5
            colors = ((voxel_centers + 0.5) * 255).clip(0, 255).astype(np.uint8)

            # Camera center
            camera_center = None
            if os.path.exists(norm_path):
                import json as json_mod
                with open(norm_path, 'r') as f:
                    norm_info = json_mod.load(f)
                camera_center = np.array(norm_info.get('camera_center', [0, 0, 0]), dtype=np.float64)

            fig = create_psg_plotly_figure(
                voxel_centers, colors, camera_center,
                show_camera_center=(camera_center is not None),
                show_coordinates=True,
            )
            z_np = z[0].cpu().numpy()
            info = (f"Pre-saved voxels -> encode -> decode: {n_active} active voxels (of {len(verts)} input)\n"
                    f"SS latent range: [{z_np.min():.3f}, {z_np.max():.3f}], std={z_np.std():.4f}")
            return fig, state, info

        else:
            # "3D Point Cloud" mode
            da2_data = state.get('da2_data')
            if da2_data is None:
                return None, state, "No DA2 data loaded. Click 'Load Input' first."

            if da2_data.get('is_normalized', False):
                pts = da2_data['points']
                colors = da2_data['colors']
                camera_center = da2_data.get('camera_center')
                fig = create_psg_plotly_figure(
                    pts, colors, camera_center,
                    show_camera_center=(camera_center is not None),
                    show_coordinates=True,
                )
                info = f"DA2 Point Cloud (normalized): {len(pts)} points"
            elif da2_data.get('is_cropped', False):
                pts = da2_data['points_raw']
                colors = da2_data['colors']
                fig = create_psg_plotly_figure(pts, colors, None,
                    show_camera_center=False, show_coordinates=True)
                info = f"DA2 Point Cloud (cropped): {len(pts)} points"
            else:
                pts = da2_data.get('points_raw', da2_data.get('points'))
                colors = da2_data.get('colors')
                if pts is not None:
                    fig = create_psg_plotly_figure(pts, colors, None,
                        show_camera_center=False, show_coordinates=True)
                    info = f"DA2 Point Cloud (raw): {len(pts)} points"
                else:
                    return None, state, "No DA2 point cloud data available."
            return fig, state, info

    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error: {e}"


def on_generate_csg(scene_name, view_id, data_dir,
                    use_sdedit, sdedit_source, alpha, use_spatial_mask, seed, steps, cfg_strength,
                    show_camera, max_cubes, state):
    """Stage 2: CSG generation."""
    try:
        t_start = time.time()
        state = dict(state) if state else {}

        # Use DA2 data from state (must be normalized for SDEdit)
        da2_data = state.get('da2_data')

        # Handle SDEdit source selection
        psg_ss_latent_override = None
        if use_sdedit and sdedit_source == "pre-saved voxels":
            # Load from saved depth_voxels_da2_64 PLY -> encode to SS latent
            psg_ss_latent_override, saved_cam = load_sdedit_latent_from_saved_voxels(
                data_dir, scene_name, view_id)
            if psg_ss_latent_override is None:
                return None, state, "No pre-saved depth_voxels_da2_64 found. Save voxels first or use 'DA2 inline'."
            # If da2_data not normalized, create minimal da2_data with camera_center from saved
            if da2_data is None or not da2_data.get('is_normalized', False):
                if da2_data is None:
                    da2_data = {'is_normalized': True, 'camera_center': saved_cam}
                else:
                    da2_data = dict(da2_data)
                    da2_data['camera_center'] = saved_cam
                    da2_data['is_normalized'] = True
                state['da2_data'] = da2_data

        if da2_data is not None and not da2_data.get('is_normalized', False):
            if use_sdedit and sdedit_source == "DA2 inline":
                return None, state, "SDEdit requires normalized point cloud. Click 'Normalize' first."
            # For non-SDEdit, we still need camera_center — auto-normalize
            da2_data = crop_da2_point_cloud(da2_data)
            da2_data = normalize_da2_point_cloud(da2_data)
            state['da2_data'] = da2_data

        # If using pre-saved voxels, inject into da2_data for run_stage1_replicapano
        if psg_ss_latent_override is not None:
            da2_data = dict(da2_data) if da2_data else {}
            da2_data['_psg_ss_latent_override'] = psg_ss_latent_override

        result = run_stage1_replicapano(
            data_dir, scene_name, view_id,
            use_sdedit=use_sdedit, alpha=alpha,
            da2_data=da2_data,
            steps=steps, cfg_strength=cfg_strength, seed=seed,
            use_spatial_mask=use_spatial_mask,
        )

        state['scene_name'] = scene_name
        state['view_id'] = view_id
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
    """Stage 3: Predict 3D bboxes."""
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

        pred_glb = create_bbox_with_voxel_glb(
            bbox_result['obbs'], voxel_64, camera_center)

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
    """Stage 4: Shape (+ texture) generation -> mesh."""
    try:
        t_start = time.time()
        state = dict(state) if state else {}
        scene_name = state.get('scene_name', '')
        view_id = state.get('view_id', '')
        voxel_64 = state.get('voxel_64')
        obbs = state.get('obbs')
        camera_center = state.get('camera_center')
        encoded_cond = state.get('encoded_cond')

        if voxel_64 is None or obbs is None:
            return None, None, None, state, "Generate CSG and predict bboxes first."

        # Shape generation
        # run_stage2_shape_single expects (data_dir, scene_id, room_id, view_idx, ...)
        # We pass scene_name as scene_id, view_id as room_id, view_idx=0
        # The function only uses these for logging, actual data comes from voxel_64/obbs/etc.
        log(f"\n--- Shape Generation (layout={layout_mode}) ---")
        t1 = time.time()
        shape_result = run_stage2_shape_single(
            data_dir, scene_name, view_id, 0,
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

        # Optional texture
        tex_feats = None
        if gen_texture:
            log("\n--- Texture Generation ---")
            t2 = time.time()
            tex_result = run_stage2_texture_single(
                data_dir, scene_name, view_id, 0,
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

        # Decode meshes
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

        # Create GLBs
        combined_glb = create_scene_glb(meshes_list, asset_names)
        layout_glb = create_layout_glb(meshes_list, has_layout=has_layout)
        exploded_glb = create_exploded_glb(
            meshes_list, asset_names, explosion_scale=0.0, has_layout=has_layout)

        n_valid = sum(1 for m in meshes_list if m is not None)
        tex_label = "with texture" if gen_texture else "shape only"
        status = f"Generated: {n_total} voxels, {n_parts} parts"
        status += f"\n{n_valid}/{len(meshes_list)} meshes ({tex_label})"
        status += f"\nTotal: {time.time()-t_start:.1f}s"

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
    """Save all results."""
    try:
        state = dict(state) if state else {}
        scene_name = state.get('scene_name', '')
        view_id = state.get('view_id', '')
        meshes_list = state.get('meshes_list')

        if not scene_name:
            return None, None, None, None, None, None, "Select a sample first.", ""

        output_dir = os.path.join(
            PROJECT_ROOT, 'demo_outputs', 'replicapano', scene_name, view_id)

        # Adapt state keys for save_all_results compatibility
        save_state = dict(state)
        save_state['scene_id'] = scene_name
        save_state['room_id'] = view_id
        save_state['view_idx'] = 0

        saved = save_all_results_replicapano(output_dir, save_state, meshes_list, data_dir)

        def _load(key):
            return Image.open(saved[key]) if key in saved else None

        cubemap_img = _load('cubemap_input')
        bbox_img = _load('bbox_topdown')
        ss_ext_img = _load('ss_exterior')
        ss_int_img = _load('ss_interior')
        geom_ext_img = _load('geometry_exterior')
        geom_td_img = _load('geometry_topdown_cam')
        tex_ext_img = _load('texture_exterior')
        tex_int_img = _load('texture_interior')
        tex_td_img = _load('texture_topdown_cam')

        n_vis = sum(1 for v in saved.values() if isinstance(v, str) and v.endswith('.png'))
        status = f"Saved {n_vis} images to: {os.path.join(output_dir, 'vis/')}"

        return (cubemap_img, bbox_img, ss_ext_img, ss_int_img,
                geom_ext_img, geom_td_img, tex_ext_img, tex_int_img, tex_td_img,
                status, output_dir)

    except Exception as e:
        traceback.print_exc()
        return (None, None, None, None, None, None, None, None, None,
                f"Error: {e}", "")


# ============================================================
# Gradio UI
# ============================================================

def create_demo(data_dir=DEFAULT_DATA_DIR):
    init_dataset(data_dir)

    with gr.Blocks(title="InSpace ReplicaPano") as demo:
        gr.HTML(header_html("ReplicaPano demo · real-scan panoramas (13 scenes)"))
        state = gr.State({})

        with gr.Row():
            # ============ LEFT PANEL ============
            with gr.Column(scale=1):

                # --- Input ---
                gr.Markdown("### Input")
                scene_dd = gr.Dropdown(
                    choices=SCENE_IDS,
                    value=SCENE_IDS[0] if SCENE_IDS else None,
                    label="Scene",
                )
                initial_views = SCENE_VIEWS.get(
                    SCENE_IDS[0], ['00000']) if SCENE_IDS else ['00000']
                view_dd = gr.Dropdown(
                    choices=initial_views, value=initial_views[0],
                    label="View ID",
                )
                load_btn = gr.Button("Load Input", variant="primary")

                # --- Stage 2: CSG ---
                gr.Markdown("---")
                gr.Markdown("### Stage 2: Coarse Scene Geometry")
                gr.Markdown("*SS flow -> 64^3 voxel (optional SDEdit from DA2 depth)*")
                use_sdedit_cb = gr.Checkbox(
                    label="SDEdit from DA2", value=True,
                    info="Use DA2 depth as initial latent via SDEdit")
                sdedit_source_radio = gr.Radio(
                    choices=["DA2 inline", "pre-saved voxels"],
                    value="DA2 inline",
                    label="SDEdit Source",
                    info="DA2 inline: use cropped/normalized DA2 point cloud. pre-saved: use saved depth_voxels_da2_64 PLY.")
                alpha_slider = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Alpha (noise level)",
                    info="0=clean DA2, 1=pure noise. Recommended: 0.7-0.9")
                csg_spatial_mask_cb = gr.Checkbox(
                    label="Spatial Attention Mask", value=True,
                    info="Use camera-center-based cross-attention mask. Disable to debug.")
                csg_show_cam_cb = gr.Checkbox(label="Show Camera", value=True)
                csg_max_cubes = gr.Slider(
                    1000, 50000, value=30000, step=1000,
                    label="Max Voxel Cubes",
                    info="Max cubes to render in 3D viewer (more=denser, slower)")
                generate_csg_btn = gr.Button("Generate CSG", variant="primary")

                # --- Stage 3: BBox ---
                gr.Markdown("---")
                gr.Markdown("### Stage 3: 3D BBox Estimation")
                gr.Markdown("*CenterPoint prediction (no GT available)*")
                bbox_source_display = gr.Textbox(
                    label="BBox Source", value="None",
                    interactive=False, max_lines=1)
                pred_bbox_btn = gr.Button("Predict BBox", variant="primary")
                bbox_threshold = gr.Slider(
                    0.1, 0.9, value=0.3, step=0.05, label="Score Threshold")

                # --- Stage 4: Scene Generation ---
                gr.Markdown("---")
                gr.Markdown("### Stage 4: Scene Generation")
                gen_texture_cb = gr.Checkbox(
                    label="Generate Texture", value=True,
                    info="PBR texture (slower)")
                layout_mode_radio = gr.Radio(
                    choices=["floor_perimeter", "floor_perimeter_clean", "no_floor_assets"],
                    value="floor_perimeter",
                    label="Layout & Asset Mode")
                gr.Markdown(
                    "- **floor_perimeter**: Layout=floor+walls. Assets include floor voxels (matches training).\n"
                    "- **floor_perimeter_clean**: Same layout, but assets **exclude floor-layer** voxels. "
                    "Removes floor plane attached to tables/chairs.\n"
                    "- **no_floor_assets**: Alias for floor_perimeter_clean.")
                gen_scene_btn = gr.Button("Generate Scene", variant="primary")

                # --- Advanced ---
                with gr.Accordion("Advanced Settings", open=False):
                    seed_slider = gr.Slider(0, 99999, value=42, step=1, label="Seed")
                    steps_s1 = gr.Slider(4, 50, value=12, step=1, label="Stage 1 Steps")
                    steps_s2 = gr.Slider(4, 50, value=12, step=1, label="Stage 2 Steps")
                    cfg_s1 = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Stage 1 CFG")
                    cfg_s2 = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Stage 2 CFG")

                status_box = gr.Textbox(label="Status", lines=3, interactive=False)

            # ============ RIGHT PANEL ============
            with gr.Column(scale=2):
                with gr.Tabs():

                    with gr.Tab("Input"):
                        erp_image = gr.Image(label="ERP Panorama", height=200)
                        cubemap_image = gr.Image(label="Cubemap (2x3)", height=400)
                        depth_image = gr.Image(label="ERP Depth Map", height=200)

                    with gr.Tab("DA2 Point Cloud"):
                        da2_view_radio = gr.Radio(
                            choices=["3D Point Cloud", "Pre-saved Voxels (decoded)"],
                            value="3D Point Cloud",
                            label="View Mode",
                            info="Switch between DA2 point cloud and pre-saved voxel visualization")
                        da2_plot = gr.Plot(label="DA2 Depth Point Cloud")
                        da2_info_box = gr.Textbox(label="DA2 Info", lines=3, interactive=False)
                        da2_view_btn = gr.Button("Load View", variant="secondary")

                        gr.Markdown("**Step 1: Lift** — Re-lift from ERP depth with RGB colors")
                        da2_relift_btn = gr.Button("Re-lift with RGB Colors", variant="secondary")

                        gr.Markdown("**Step 2: Crop** — Remove ceiling, crop region to exclude points outside room (windows/doors)")
                        with gr.Row():
                            da2_remove_ceiling_cb = gr.Checkbox(
                                label="Remove Ceiling", value=True)
                            da2_ceiling_slider = gr.Slider(
                                0.0, 1.0, value=0.2, step=0.05,
                                label="Ceiling Threshold (m)")
                        with gr.Row():
                            da2_use_crop_cb = gr.Checkbox(
                                label="Enable Region Crop", value=False,
                                info="Crop to XYZ bounding box (for removing outside-room points)")
                        with gr.Row():
                            da2_crop_x_min = gr.Number(label="X min", value=-10.0, precision=1)
                            da2_crop_x_max = gr.Number(label="X max", value=10.0, precision=1)
                            da2_crop_y_min = gr.Number(label="Y min", value=-10.0, precision=1)
                            da2_crop_y_max = gr.Number(label="Y max", value=10.0, precision=1)
                            da2_crop_z_min = gr.Number(label="Z min", value=-10.0, precision=1)
                            da2_crop_z_max = gr.Number(label="Z max", value=10.0, precision=1)
                        da2_crop_btn = gr.Button("Update Crop", variant="secondary")

                        gr.Markdown("**Step 3: Normalize** — Normalize to [-0.5, 0.5] for generation model")
                        with gr.Row():
                            da2_show_cam_cb = gr.Checkbox(
                                label="Show Camera Center", value=True)
                            da2_normalize_btn = gr.Button("Normalize", variant="primary")

                        with gr.Row():
                            da2_save_voxel_btn = gr.Button("Save depth_voxels_da2_64", variant="primary")

                    with gr.Tab("CSG"):
                        csg_viewer = gr.Model3D(label="Predicted CSG (64^3)", height=500)

                    with gr.Tab("BBox + CSG"):
                        bbox_viewer = gr.Model3D(label="Predicted BBox + CSG", height=500)

                    with gr.Tab("Mesh"):
                        with gr.Row():
                            combined_viewer = gr.Model3D(
                                label="Overall Scene", height=350)
                            layout_viewer = gr.Model3D(
                                label="Layout", height=350)
                            exploded_viewer = gr.Model3D(
                                label="Assets (exploded)", height=350)
                        explode_slider = gr.Slider(
                            0.0, 1.0, value=0.0, step=0.05, label="Explosion Scale")

                    with gr.Tab("Save"):
                        gr.Markdown("### Save Results")
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
        # Event Bindings
        # ============================================================

        data_dir_state = gr.State(data_dir)

        scene_dd.change(on_scene_change, [scene_dd], [view_dd])

        load_btn.click(
            on_load_input,
            [scene_dd, view_dd, data_dir_state, state],
            [erp_image, cubemap_image, depth_image, da2_plot, state, da2_info_box, status_box],
        )

        da2_view_btn.click(
            on_switch_da2_view,
            [da2_view_radio, data_dir_state, state],
            [da2_plot, state, da2_info_box],
        )

        da2_relift_btn.click(
            on_relift_da2,
            [data_dir_state, state],
            [da2_plot, state, da2_info_box],
        )

        da2_crop_btn.click(
            on_update_da2_pointcloud,
            [data_dir_state, da2_remove_ceiling_cb, da2_ceiling_slider,
             da2_crop_x_min, da2_crop_x_max, da2_crop_y_min, da2_crop_y_max,
             da2_crop_z_min, da2_crop_z_max, da2_use_crop_cb,
             state],
            [da2_plot, state, da2_info_box],
        )

        da2_normalize_btn.click(
            on_normalize_da2,
            [da2_show_cam_cb, state],
            [da2_plot, state, da2_info_box],
        )

        da2_save_voxel_btn.click(
            on_save_depth_voxels,
            [data_dir_state, state],
            [da2_info_box],
        )

        generate_csg_btn.click(
            on_generate_csg,
            [scene_dd, view_dd, data_dir_state,
             use_sdedit_cb, sdedit_source_radio, alpha_slider, csg_spatial_mask_cb, seed_slider, steps_s1, cfg_s1,
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
    parser = argparse.ArgumentParser(description='InSpace ReplicaPano Demo')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--share', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    ensure_demo_samples(args.data_dir)
    demo = create_demo(args.data_dir)
    launch_kwargs = dict(server_name='0.0.0.0', share=args.share,
                         theme=INSPACE_THEME, css=INSPACE_CSS)
    if args.port is not None:
        launch_kwargs['server_port'] = args.port
    demo.launch(**launch_kwargs)
