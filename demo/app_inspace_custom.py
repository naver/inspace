# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
InSpace Custom (User Data) Gradio Interactive Demo

Interactive per-sample inference for the InSpace pipeline on user-provided custom ERP data.
Stages:
    1. Load Input (cubemap from ERP, DA2 depth -> point cloud)
    2. Coarse Scene Geometry (SS flow -> 64^3 voxel, optional SDEdit from DA2)
    3. 3D BBOX Estimation (CenterPoint prediction)
    4. Layout and Asset-Aware Scene Generation (Shape + Texture -> Mesh)

Usage:
    python demo/app_inspace_custom.py --port 7862
"""

import os
import sys
import time
import json
import argparse
import traceback
from tqdm import tqdm

# Select the GPU before importing torch (must be set before the first torch import).
# Defaults to GPU 1; override from the shell, e.g.
#   CUDA_VISIBLE_DEVICES=0 python demo/app_inspace_custom.py
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

try:
    import py360convert
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "py360convert"])
    import py360convert

# Reuse model management, inference, and viz from demo
from demo.app_inspace_utils import (
    model_manager,
    log,
    find_latest_ckpt,
    load_denoiser,
    load_cubemap_images,
    load_camera_center,
    load_gt_bboxes,
    detect_floor_z,
    detect_layout_from_floor_perimeter,
    assign_voxels_to_obbs,
    construct_stage2_input,
    run_stage1_single,
    run_bbox_gt_single,
    run_bbox_predicted_single,
    run_stage2_shape_single,
    run_stage2_texture_single,
    decode_meshes_single,
    save_all_results,
    DEFAULT_STAGE1_CONFIG,
)
from demo.app_inspace_ui import header_html, INSPACE_THEME, INSPACE_CSS
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


# ============================================================
# ERP_3D_FRONT Data Loading
# ============================================================

DEFAULT_DATA_DIR = os.path.join(
    PROJECT_ROOT, 'datasets', 'custom_samples')

# White background for the 3D (Model3D) viewers, matching the project-page viewer.
WHITE_BG = (1.0, 1.0, 1.0, 1.0)


def discover_erp_front_samples(data_dir):
    """Discover all ERP_3D_FRONT samples.
    Returns dict: {(uuid, room_name): [view_indices]}
    """
    samples = {}
    for uuid in sorted(os.listdir(data_dir)):
        uuid_dir = os.path.join(data_dir, uuid)
        if not os.path.isdir(uuid_dir):
            continue
        for room_name in sorted(os.listdir(uuid_dir)):
            room_dir = os.path.join(uuid_dir, room_name)
            if not os.path.isdir(room_dir):
                continue
            # Check for cubemap views
            cubic_dir = os.path.join(room_dir, 'cubic_fov_120')
            if not os.path.isdir(cubic_dir):
                continue
            views = []
            for v in sorted(os.listdir(cubic_dir)):
                v_dir = os.path.join(cubic_dir, v)
                if os.path.isdir(v_dir) and os.path.exists(os.path.join(v_dir, 'front.png')):
                    views.append(v)
            if views:
                samples[(uuid, room_name)] = views
    return samples


def load_sdedit_latent(data_dir, scene_id, room_id, view_idx=0):
    """Load pre-computed SDEdit initial SS latent from depth_voxels_da2_ss_latent/.
    Returns numpy array [8, 16, 16, 16] or None.
    """
    sample_dir = os.path.join(data_dir, scene_id, room_id)
    latent_dir = os.path.join(sample_dir, 'depth_voxels_da2_ss_latent', 'ss_enc_conv3d_16l8_fp16_64')
    npz_path = os.path.join(latent_dir, f'{view_idx:04d}.npz')

    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path)
    return data['z'].astype(np.float32)  # [8, 16, 16, 16]


# ============================================================
# DA2 Point Cloud Functions (adapted from ReplicaPano app_gradio.py)
# ============================================================

def erp_depth_to_point_cloud(rgb, depth, subsample=2, max_points=50000, max_depth=20.0,
                             remove_ceiling=False, ceiling_threshold=0.0):
    """ERP depth + RGB -> 3D point cloud with colors."""
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
    """Self-normalize point cloud to [-0.5, 0.5]. Returns (points, center, scale)."""
    pts_min = points.min(axis=0)
    pts_max = points.max(axis=0)
    center = (pts_min + pts_max) / 2.0
    max_extent = (pts_max - pts_min).max()
    scale = 0.99999 / max_extent if max_extent > 1e-6 else 1.0
    normalized = np.clip((points - center) * scale, -0.5 + 1e-6, 0.5 - 1e-6)
    return normalized, center, scale


def load_da2_raw_point_cloud(data_dir, scene_id, room_id, view_idx=0):
    """Step 1: Load DA2 depth + RGB, lift to raw 3D point cloud (no normalization).
    ERP_3D_FRONT path: erp/{view_idx:04d}_depth_da2.npy
    """
    sample_dir = os.path.join(data_dir, scene_id, room_id)
    da2_path = os.path.join(sample_dir, 'erp', f'{view_idx:04d}_depth_da2.npy')
    erp_path = os.path.join(sample_dir, 'erp', f'{view_idx:04d}_colors.png')

    if not os.path.exists(da2_path):
        return None

    da2_depth = np.load(da2_path)
    erp_rgb = np.array(Image.open(erp_path).convert('RGB')) if os.path.exists(erp_path) else None
    if erp_rgb is None:
        erp_rgb = np.full((*da2_depth.shape, 3), 128, dtype=np.uint8)

    points_vis, colors_vis = erp_depth_to_point_cloud(
        erp_rgb, da2_depth, subsample=2, max_points=50000,
        remove_ceiling=False, ceiling_threshold=0.0)

    if len(points_vis) < 100:
        return None

    return {
        'points_raw': points_vis,
        'colors': colors_vis,
        'da2_depth': da2_depth,
        'erp_rgb': erp_rgb,
        'is_normalized': False,
        'is_cropped': False,
    }


def _rotate_points_yaw(points, yaw_degrees):
    """Rotate points around Y-axis (height axis in camera space) by yaw_degrees."""
    if yaw_degrees == 0.0:
        return points
    rad = np.radians(yaw_degrees)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    # Rotate around Y-axis: camera space has Y=up
    # x' = x*cos - z*sin, z' = x*sin + z*cos
    rotated = points.copy()
    rotated[:, 0] = points[:, 0] * cos_a - points[:, 2] * sin_a
    rotated[:, 2] = points[:, 0] * sin_a + points[:, 2] * cos_a
    return rotated


def _statistical_outlier_removal(points, colors, nb_neighbors=20, std_ratio=2.0):
    """Remove statistical outliers based on mean distance to k-nearest neighbors.

    Points whose mean distance to their k-nearest neighbors exceeds
    (global_mean + std_ratio * global_std) are removed.
    """
    from scipy.spatial import cKDTree
    if len(points) < nb_neighbors + 1:
        return points, colors
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)  # +1 because first neighbor is self
    mean_dists = dists[:, 1:].mean(axis=1)  # exclude self (dist=0)
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    threshold = global_mean + std_ratio * global_std
    mask = mean_dists < threshold
    return points[mask], colors[mask]


def _apply_filters(points, colors, max_depth_cap, sor_std_ratio, rotation_yaw, crop_bbox):
    """Apply rotation, max depth cap, crop bbox, and statistical outlier removal."""
    # 1) Rotation
    points = _rotate_points_yaw(points, rotation_yaw)

    # 2) Max depth cap (distance from origin = camera position)
    if max_depth_cap > 0:
        dist = np.linalg.norm(points, axis=1)
        mask = dist <= max_depth_cap
        points = points[mask]
        colors = colors[mask]

    # 3) Crop bbox
    if crop_bbox is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bbox
        mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        points = points[mask]
        colors = colors[mask]

    # 4) Statistical outlier removal
    if sor_std_ratio > 0 and len(points) > 100:
        points, colors = _statistical_outlier_removal(points, colors, nb_neighbors=20, std_ratio=sor_std_ratio)

    return points, colors


def crop_da2_point_cloud(da2_data, remove_ceiling=False, ceiling_threshold=0.0,
                          crop_bbox=None, rotation_yaw=0.0,
                          max_depth_cap=0.0, sor_std_ratio=0.0):
    """Step 2: Crop the raw point cloud.

    Pipeline order: ceiling removal -> rotation -> max depth cap -> crop bbox -> SOR.
    """
    if da2_data is None:
        return None

    erp_rgb = da2_data['erp_rgb']
    da2_depth = da2_data['da2_depth']

    points_full, colors_full = erp_depth_to_point_cloud(
        erp_rgb, da2_depth, subsample=1, max_points=0,
        remove_ceiling=remove_ceiling, ceiling_threshold=ceiling_threshold)

    if len(points_full) < 100:
        return None

    points_full, colors_full = _apply_filters(
        points_full, colors_full, max_depth_cap, sor_std_ratio, rotation_yaw, crop_bbox)

    if len(points_full) < 100:
        return None

    points_vis, colors_vis = erp_depth_to_point_cloud(
        erp_rgb, da2_depth, subsample=2, max_points=50000,
        remove_ceiling=remove_ceiling, ceiling_threshold=ceiling_threshold)

    points_vis, colors_vis = _apply_filters(
        points_vis, colors_vis, max_depth_cap, sor_std_ratio, rotation_yaw, crop_bbox)

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
        'rotation_yaw': rotation_yaw,
        'max_depth_cap': max_depth_cap,
        'sor_std_ratio': sor_std_ratio,
    }
    return da2_data


def cam_to_world_points(points_cam):
    """
    Convert camera-centered coordinates to world coordinates.

    The ERP camera has rotation [pi/2, 0, 0] (Euler XYZ), meaning:
    - Camera forward (-Z) -> World +Y
    - Camera up (+Y) -> World +Z (height)
    - Camera right (+X) -> World +X

    Camera origin is at [0,0,0] in camera space, so no translation needed
    (we normalize separately).

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

    points_norm_full, center, scale = self_normalize_point_cloud(points_full_world)
    # Camera is at origin in camera space -> [0, 0, 0] in world space too (no translation)
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


def voxelize_point_cloud(points_norm, grid_size=64):
    """Convert normalized point cloud to voxel grid."""
    voxel_indices = ((points_norm + 0.5) * grid_size).astype(np.int32)
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)
    unique_voxels = np.unique(voxel_indices, axis=0)

    occ_grid = torch.zeros(1, 1, grid_size, grid_size, grid_size, dtype=torch.float32)
    for c in unique_voxels:
        occ_grid[0, 0, c[0], c[1], c[2]] = 1.0

    voxel_centers = (unique_voxels + 0.5) / grid_size - 0.5
    return occ_grid, voxel_centers, unique_voxels


def save_depth_voxels_ply(voxel_centers, output_path):
    """Save voxel centers as PLY with position-based RGB colors."""
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


def compute_sdedit_initial_latent(da2_data, device='cuda'):
    """Inline: DA2 point cloud -> voxelize -> SS latent.
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
# Dataset Discovery
# ============================================================

SAMPLES = {}  # (uuid, room_name) -> [view_indices]
ROOM_IDS = []  # list of "uuid/room_name" strings
ROOM_VIEWS = {}  # "uuid/room_name" -> [view_indices]


def init_dataset(data_dir):
    global SAMPLES, ROOM_IDS, ROOM_VIEWS
    SAMPLES = discover_erp_front_samples(data_dir)
    ROOM_VIEWS = {}
    for (uuid, room_name), views in SAMPLES.items():
        key = f"{uuid}/{room_name}"
        ROOM_VIEWS[key] = views
    ROOM_IDS = sorted(ROOM_VIEWS.keys())
    log(f"[Init] Discovered {len(ROOM_IDS)} rooms, {sum(len(v) for v in ROOM_VIEWS.values())} total views")


# ============================================================
# Custom ERP -> cubemap (upload flow)
# ============================================================

FACE_ORDER = ['front', 'right', 'back', 'left', 'top', 'bottom']


def erp_to_cubemap(erp_image, face_size=512, fov=120):
    """Equirectangular RGB (H, W, 3) -> dict of 6 perspective faces (FOV 120)."""
    face_dirs = {
        'front': (0, 0), 'right': (90, 0), 'back': (180, 0),
        'left': (270, 0), 'top': (0, 90), 'bottom': (0, -90),
    }
    faces = {}
    for name, (yaw, pitch) in face_dirs.items():
        faces[name] = py360convert.e2p(
            erp_image, fov_deg=(fov, fov), u_deg=yaw, v_deg=pitch,
            out_hw=(face_size, face_size), mode='bilinear',
        )
    return faces


def _load_erp_depth(depth_file, target_hw):
    """Load a user-supplied ERP depth map to an (H, W) float array (metric meters).

    Accepts a `.npy` float array (preferred) or a 16-bit single-channel `.png` (mm).
    Resizes to `target_hw` if needed. Returns None on failure.
    """
    if depth_file is None:
        return None
    path = depth_file if isinstance(depth_file, str) else getattr(depth_file, 'name', None)
    if not path or not os.path.exists(path):
        return None
    try:
        if path.lower().endswith('.npy'):
            depth = np.load(path).astype(np.float32)
        else:
            arr = np.array(Image.open(path))
            if arr.ndim == 3:  # colorized visualization, not real depth
                raise ValueError("depth PNG is RGB (colorized); provide a raw .npy depth array")
            depth = arr.astype(np.float32)
            if arr.dtype == np.uint16:  # 16-bit depth stored in millimetres
                depth /= 1000.0
        if depth.ndim != 2:
            raise ValueError(f"expected a 2D depth array, got shape {depth.shape}")
        H, W = target_hw
        if depth.shape != (H, W):
            depth = np.array(Image.fromarray(depth).resize((W, H), Image.NEAREST))
        return depth
    except Exception as e:
        log(f"[Custom] Could not read depth map: {e}")
        return None


def add_custom_erp_sample(erp_pil, data_dir, name=None, depth_file=None, face_size=512):
    """Write an uploaded ERP panorama + its computed cubemap (+ optional depth) into
    `data_dir` as a new sample (uploaded/<name>) so the standard discovery/pipeline can
    use it. Returns (room_key, has_depth).
    """
    if name:
        name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name).strip("_")
    if not name:
        # deterministic fallback name based on existing count (no wall-clock)
        existing = os.path.join(data_dir, 'uploaded')
        n = len(os.listdir(existing)) if os.path.isdir(existing) else 0
        name = f"sample_{n:03d}"

    room_dir = os.path.join(data_dir, 'uploaded', name)
    erp_dir = os.path.join(room_dir, 'erp')
    cubic_dir = os.path.join(room_dir, 'cubic_fov_120', '0000')
    os.makedirs(erp_dir, exist_ok=True)
    os.makedirs(cubic_dir, exist_ok=True)

    erp_pil = erp_pil.convert('RGB')
    erp_pil.save(os.path.join(erp_dir, '0000_colors.png'))

    faces = erp_to_cubemap(np.array(erp_pil), face_size=face_size, fov=120)
    for fname in FACE_ORDER:
        Image.fromarray(faces[fname].astype(np.uint8)).save(
            os.path.join(cubic_dir, f'{fname}.png'))

    # optional ERP depth -> erp/0000_depth_da2.npy (enables SDEdit / PSG)
    W, H = erp_pil.size
    depth = _load_erp_depth(depth_file, (H, W))
    has_depth = depth is not None
    if has_depth:
        np.save(os.path.join(erp_dir, '0000_depth_da2.npy'), depth)

    log(f"[Custom] Added uploaded sample: uploaded/{name} (depth={'yes' if has_depth else 'no'})")
    return f"uploaded/{name}", has_depth


def colorize_depth(depth):
    """Turn an (H, W) depth array into a turbo-colormapped RGB PIL image for display."""
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return None
    lo, hi = np.percentile(d[valid], [2, 98])
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    import matplotlib.cm as cm
    rgb = (cm.turbo(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb)


def load_depth_for_view(data_dir, scene_id, room_id, view_idx):
    """Return a depth visualization (PIL RGB) for the Input tab, or None.

    Prefers the raw `*_depth_da2.npy` (colormapped); falls back to a pre-rendered
    `*_depth_vis_da2.png` if present.
    """
    base = os.path.join(data_dir, scene_id, room_id, 'erp')
    npy = os.path.join(base, f'{view_idx:04d}_depth_da2.npy')
    if os.path.exists(npy):
        try:
            vis = colorize_depth(np.load(npy))
            if vis is not None:
                return vis
        except Exception as e:
            log(f"[Load] depth colorize failed: {e}")
    vis_png = os.path.join(base, f'{view_idx:04d}_depth_vis_da2.png')
    if os.path.exists(vis_png):
        return Image.open(vis_png).convert('RGB')
    return None


# ============================================================
# Event Handlers
# ============================================================

def on_room_change(room_key):
    """Update view dropdown when room changes."""
    views = ROOM_VIEWS.get(room_key, [])
    if views:
        return gr.update(choices=views, value=views[0])
    return gr.update(choices=[], value=None)


def on_generate_cubemap(erp_pil, depth_file, name, data_dir, state):
    """Upload flow: ERP panorama (+ optional depth) -> FOV-120 cubemap -> new sample,
    then immediately load & display it (ERP / cubemap / depth / DA2 point cloud)."""
    if erp_pil is None:
        return (gr.update(), gr.update(), None, None, None, None,
                state, "", "Please upload an ERP panorama first.")
    try:
        room_key, has_depth = add_custom_erp_sample(
            erp_pil, data_dir, name=name or None, depth_file=depth_file)
        init_dataset(data_dir)
        views = ROOM_VIEWS.get(room_key, ['0000'])
        # reuse the load handler to populate all Input-tab viewers right away
        erp_img, cubemap_grid, depth_vis, psg_fig, new_state, da2_status, _ = \
            on_load_input(room_key, '0000', data_dir, state)
        extra = ("Depth added -> SDEdit / PSG available." if has_depth
                 else "No depth -> Stage 2 uses random noise (add an ERP depth .npy for SDEdit).")
        status = f"Added '{room_key}' and loaded it. {extra}"
        return (gr.update(choices=ROOM_IDS, value=room_key),
                gr.update(choices=views, value=views[0]),
                erp_img, cubemap_grid, depth_vis, psg_fig, new_state, da2_status, status)
    except Exception as e:
        traceback.print_exc()
        return (gr.update(), gr.update(), None, None, None, None,
                state, "", f"Error generating cubemap: {e}")


def on_load_input(room_key, view_id, data_dir, state):
    """Load and display ERP image, cubemap, and DA2 point cloud."""
    try:
        t0 = time.time()
        state = dict(state) if state else {}

        parts = room_key.split('/', 1)
        scene_id, room_id = parts[0], parts[1]
        view_idx = int(view_id)
        state['scene_id'] = scene_id
        state['room_id'] = room_id
        state['view_idx'] = view_idx
        state['room_key'] = room_key

        # ERP panorama
        erp_path = os.path.join(data_dir, scene_id, room_id, 'erp', f'{view_idx:04d}_colors.png')
        erp_img = Image.open(erp_path).convert('RGB') if os.path.exists(erp_path) else None
        log(f"[Load] ERP image: {erp_path}")

        # ERP depth map (visualization)
        depth_vis = load_depth_for_view(data_dir, scene_id, room_id, view_idx)

        # Cubemap (pre-computed)
        t1 = time.time()
        face_order = ['front', 'right', 'back', 'left', 'top', 'bottom']
        cubic_dir = os.path.join(data_dir, scene_id, room_id, 'cubic_fov_120', f'{view_idx:04d}')

        cubemap_grid = None
        if os.path.isdir(cubic_dir):
            from PIL import ImageDraw, ImageFont
            face_size = 256
            label_h = 20
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
                img = Image.open(os.path.join(cubic_dir, f'{fname}.png')).convert('RGB')
                img_small = img.resize((face_size, face_size))
                x_off = c * face_size
                y_off = r * cell_h
                draw.text((x_off + face_size // 2, y_off + 2), fname,
                          fill=(0, 0, 0), font=font, anchor='mt')
                grid_img.paste(img_small, (x_off, y_off + label_h))
            cubemap_grid = grid_img
            log(f"[Load] Cubemap loaded: {time.time()-t1:.2f}s")

        # DA2 point cloud — check for pre-saved voxels, otherwise do Step 1: raw lifting
        t2 = time.time()
        sample_dir = os.path.join(data_dir, scene_id, room_id)
        saved_voxel_dir = os.path.join(sample_dir, 'depth_voxels_da2_64')
        saved_ply = os.path.join(saved_voxel_dir, f'{view_idx:04d}.ply')
        saved_norm = os.path.join(saved_voxel_dir, 'normalization_info.json')

        da2_data = None
        psg_fig = None
        da2_status = ""

        if os.path.exists(saved_ply) and os.path.exists(saved_norm):
            # Load pre-saved normalized voxels + normalization info
            import trimesh
            with open(saved_norm, 'r') as f:
                norm_info = json.load(f)

            pc = trimesh.load(saved_ply, process=False)
            verts = np.array(pc.vertices, dtype=np.float32) if hasattr(pc, 'vertices') else np.zeros((0, 3))
            colors_ply = ((verts + 0.5) * 255).clip(0, 255).astype(np.uint8)

            center = np.array(norm_info['center'], dtype=np.float64)
            scale = float(norm_info['scale'])
            camera_center = np.array(norm_info['camera_center'], dtype=np.float64)

            # Also load raw depth + rgb for potential re-cropping
            da2_path = os.path.join(sample_dir, 'erp', f'{view_idx:04d}_depth_da2.npy')
            erp_rgb_path = os.path.join(sample_dir, 'erp', f'{view_idx:04d}_colors.png')
            da2_depth = np.load(da2_path) if os.path.exists(da2_path) else None
            erp_rgb = np.array(Image.open(erp_rgb_path).convert('RGB')) if os.path.exists(erp_rgb_path) else None

            da2_data = {
                'points': verts,
                'colors': colors_ply,
                'points_full': verts,
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
            da2_data = load_da2_raw_point_cloud(data_dir, scene_id, room_id, view_idx)
            state['da2_data'] = da2_data
            log(f"[Load] DA2 raw lift: {time.time()-t2:.2f}s")

        t3 = time.time()
        if da2_data is not None and not da2_data.get('is_normalized', False):
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
            da2_status = (f"Raw point cloud: {n_pts} pts (not normalized)\n"
                          f"Camera center: [0.000, 0.000, 0.000] (origin)\n{pt_range}")
        elif da2_data is None:
            da2_status = "No DA2 depth data found."
        log(f"[Load] Plotly figure: {time.time()-t3:.2f}s")

        status = f"Loaded: {room_key} / view {view_id} ({time.time()-t0:.1f}s)"
        if da2_status:
            status += f"\n{da2_status}"

        return erp_img, cubemap_grid, depth_vis, psg_fig, state, da2_status, status

    except Exception as e:
        traceback.print_exc()
        return None, None, None, None, state, "", f"Error loading input: {e}"


def on_relift_da2(data_dir, state):
    """Re-lift DA2 depth to raw 3D point cloud with RGB colors (overrides saved voxels)."""
    try:
        state = dict(state) if state else {}
        scene_id = state.get('scene_id')
        room_id = state.get('room_id')
        view_idx = state.get('view_idx', 0)
        if not scene_id or not room_id:
            return None, state, "Load input first."

        t0 = time.time()
        da2_data = load_da2_raw_point_cloud(data_dir, scene_id, room_id, view_idx)
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
                             rotation_yaw, max_depth_cap, sor_std_ratio,
                             crop_x_min, crop_x_max, crop_y_min, crop_y_max,
                             crop_z_min, crop_z_max, use_crop_region,
                             state):
    """Step 2: Crop the raw DA2 point cloud."""
    try:
        state = dict(state) if state else {}
        da2_data = state.get('da2_data')
        if da2_data is None:
            return None, state, "Load input first."

        t0 = time.time()

        crop_bbox = None
        if use_crop_region:
            crop_bbox = (crop_x_min, crop_x_max, crop_y_min, crop_y_max, crop_z_min, crop_z_max)

        da2_data = crop_da2_point_cloud(
            da2_data,
            remove_ceiling=remove_ceiling,
            ceiling_threshold=ceiling_threshold,
            crop_bbox=crop_bbox,
            rotation_yaw=rotation_yaw,
            max_depth_cap=max_depth_cap,
            sor_std_ratio=sor_std_ratio)
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
        if rotation_yaw != 0.0:
            info += f"\nRotation: {rotation_yaw:.0f} deg"
        if max_depth_cap > 0:
            info += f"\nMax depth cap: {max_depth_cap:.1f}m"
        if sor_std_ratio > 0:
            info += f"\nSOR: std_ratio={sor_std_ratio:.1f}"
        if use_crop_region:
            info += f"\nRegion crop active"
        info += f" ({time.time()-t0:.1f}s)"

        return psg_fig, state, info

    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error: {e}"


def on_normalize_da2(show_camera, state):
    """Step 3: Normalize the cropped point cloud to [-0.5, 0.5]."""
    try:
        state = dict(state) if state else {}
        da2_data = state.get('da2_data')
        if da2_data is None:
            return None, state, "Load input first."

        if not da2_data.get('is_cropped', False):
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
    """Save normalized DA2 point cloud as depth_voxels_da2_64 PLY."""
    try:
        state = dict(state) if state else {}
        scene_id = state.get('scene_id')
        room_id = state.get('room_id')
        view_idx = state.get('view_idx', 0)
        da2_data = state.get('da2_data')

        if da2_data is None:
            return "Load input first (need DA2 data)."

        if not da2_data.get('is_normalized', False):
            return "Please normalize the point cloud first (click 'Normalize')."

        _, voxel_centers, _ = voxelize_point_cloud(da2_data['points_full'], grid_size=64)

        sample_dir = os.path.join(data_dir, scene_id, room_id)
        output_dir = os.path.join(sample_dir, 'depth_voxels_da2_64')
        output_path = os.path.join(output_dir, f'{view_idx:04d}.ply')
        save_depth_voxels_ply(voxel_centers, output_path)

        crop_settings = da2_data.get('crop_settings', {})
        norm_info = {
            'center': da2_data['center'].tolist(),
            'scale': float(da2_data['scale']),
            'camera_center': da2_data['camera_center'].tolist(),
            'remove_ceiling': crop_settings.get('remove_ceiling', False),
            'ceiling_threshold': float(crop_settings.get('ceiling_threshold', 0.0)),
            'rotation_yaw': float(crop_settings.get('rotation_yaw', 0.0)),
            'max_depth_cap': float(crop_settings.get('max_depth_cap', 0.0)),
            'sor_std_ratio': float(crop_settings.get('sor_std_ratio', 0.0)),
            'crop_bbox': crop_settings.get('crop_bbox'),
        }
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
    """Switch DA2 tab between 3D point cloud view and pre-saved SS latent voxel view."""
    try:
        state = dict(state) if state else {}
        scene_id = state.get('scene_id')
        room_id = state.get('room_id')
        view_idx = state.get('view_idx', 0)

        if scene_id is None or room_id is None:
            return None, state, "Load input first."

        if view_mode == "Pre-saved SS Latent (voxel)":
            # Load pre-saved SS latent and decode to 64^3 voxel for visualization
            latent = load_sdedit_latent(data_dir, scene_id, room_id, view_idx)
            if latent is None:
                return None, state, "No pre-saved SS latent found at depth_voxels_da2_ss_latent/"

            # Decode latent [8,16,16,16] -> voxel [64,64,64]
            _, ss_decoder = model_manager.get_stage1()
            latent_t = torch.from_numpy(latent).unsqueeze(0).cuda()  # [1,8,16,16,16]
            with torch.no_grad():
                voxel_64 = ss_decoder(latent_t)  # [1,1,64,64,64]
            voxel_64 = (voxel_64 > 0).cpu().numpy()[0, 0]  # [64,64,64] bool

            # Get voxel centers for visualization
            occupied = np.argwhere(voxel_64)  # [N, 3]
            n_active = len(occupied)
            if n_active == 0:
                return None, state, "Pre-saved SS latent decoded to empty voxel grid."

            # Normalize to [-0.5, 0.5]
            voxel_centers = (occupied.astype(np.float32) + 0.5) / 64.0 - 0.5
            colors = ((voxel_centers + 0.5) * 255).clip(0, 255).astype(np.uint8)

            # Load camera center for display
            camera_center = None
            try:
                cc_data = load_camera_center(data_dir, scene_id, room_id, view_idx)
                camera_center = cc_data.numpy() if isinstance(cc_data, torch.Tensor) else np.array(cc_data)
            except Exception:
                pass

            fig = create_psg_plotly_figure(
                voxel_centers, colors, camera_center,
                show_camera_center=(camera_center is not None),
                show_coordinates=True,
            )
            info = (f"Pre-saved SS Latent -> decoded voxel: {n_active} active voxels\n"
                    f"Latent range: [{latent.min():.3f}, {latent.max():.3f}], std={latent.std():.4f}")
            return fig, state, info

        else:
            # "3D Point Cloud" mode — show current DA2 data
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
                pts = da2_data['points']
                colors = da2_data['colors']
                fig = create_psg_plotly_figure(pts, colors, None,
                    show_camera_center=False, show_coordinates=True)
                info = f"DA2 Point Cloud (cropped): {len(pts)} points"
            else:
                pts = da2_data.get('points_raw', da2_data.get('points'))
                colors = da2_data.get('colors_raw', da2_data.get('colors'))
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


def on_generate_csg(room_key, view_id, data_dir,
                    use_sdedit, alpha, seed, steps, cfg_strength,
                    show_camera, max_cubes, state):
    """Stage 2: CSG generation using standard run_stage1_single."""
    try:
        t_start = time.time()
        state = dict(state) if state else {}

        parts = room_key.split('/', 1)
        scene_id, room_id = parts[0], parts[1]
        view_idx = int(view_id)

        # SDEdit latent (custom data has no pre-saved latent): compute inline from the
        # normalized DA2 point cloud produced in the "DA2 Point Cloud" tab.
        psg_ss_latent = None
        if use_sdedit:
            da2_data = state.get('da2_data')
            if da2_data is not None and da2_data.get('is_normalized', False):
                log("[Stage 1] Computing SDEdit latent inline from DA2 point cloud...")
                psg_ss_latent, _ = compute_sdedit_initial_latent(da2_data)
                if psg_ss_latent is not None:
                    log(f"[Stage 1] Inline SDEdit latent: "
                        f"range=[{psg_ss_latent.min():.3f}, {psg_ss_latent.max():.3f}], "
                        f"std={psg_ss_latent.std():.4f}")
                else:
                    log("[Stage 1] Inline computation failed, falling back to random noise")
            else:
                log("[Stage 1] DA2 data not normalized. Please run Normalize first. Falling back to random noise")

        # Get camera_center from DA2 normalization (no camera_poses.json for custom data)
        da2_data = state.get('da2_data')
        cam_center_for_stage1 = None
        if da2_data is not None and da2_data.get('is_normalized', False):
            cam_center_for_stage1 = da2_data['camera_center']

        result = run_stage1_single(
            data_dir, scene_id, room_id, view_idx=view_idx,
            use_psg=(use_sdedit and psg_ss_latent is not None),
            alpha=alpha,
            psg_ss_latent=psg_ss_latent,
            steps=steps, cfg_strength=cfg_strength, seed=seed,
            camera_center_override=cam_center_for_stage1,
        )

        voxel_64 = result['voxel_64']
        camera_center = result['camera_center']
        n_active = voxel_64.sum()

        state['scene_id'] = scene_id
        state['room_id'] = room_id
        state['view_idx'] = view_idx
        state['voxel_64'] = voxel_64
        state['ss_latent'] = result['ss_latent']
        state['encoded_cond'] = result['encoded_cond']
        state['camera_center'] = camera_center
        state['scene_name'] = room_key

        # Create GLB
        csg_glb = create_voxel_glb(
            voxel_64, camera_center,
            show_camera=show_camera,
            max_cubes=max_cubes)

        status = f"CSG: {int(n_active)} active voxels ({time.time()-t_start:.1f}s)"
        if use_sdedit and psg_ss_latent is not None:
            status += f" [SDEdit alpha={alpha}]"
        else:
            status += " [random noise]"

        return csg_glb, state, status

    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error: {e}"


def on_load_gt_bbox(room_key, view_id, data_dir, state):
    """Load GT bounding boxes."""
    try:
        state = dict(state) if state else {}
        parts = room_key.split('/', 1)
        scene_id, room_id = parts[0], parts[1]
        view_idx = int(view_id)

        voxel_64 = state.get('voxel_64')
        camera_center = state.get('camera_center')
        if voxel_64 is None:
            return None, state, "Generate CSG first.", "None"

        result = run_bbox_gt_single(data_dir, scene_id, room_id, view_idx)
        if result is None:
            return None, state, "No GT bboxes found.", "None"

        state['obbs'] = result['obbs']
        state['asset_names'] = result.get('asset_names', [])
        state['asset_filenames'] = result.get('asset_filenames', [])
        state['bbox_source'] = 'gt'

        gt_glb = create_bbox_with_voxel_glb(
            result['obbs'], voxel_64, camera_center)

        n_obbs = len(result['obbs'])
        bbox_label = f"GT ({n_obbs} objects)"
        status = f"GT: {n_obbs} bounding boxes"

        return gt_glb, state, status, bbox_label

    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error: {e}", "Error"


def on_predict_bbox(bbox_threshold, state):
    """Predict bounding boxes using CenterPoint model."""
    try:
        state = dict(state) if state else {}
        voxel_64 = state.get('voxel_64')
        camera_center = state.get('camera_center')

        if voxel_64 is None:
            return None, state, "Generate CSG first.", "None"

        bbox_result = run_bbox_predicted_single(
            voxel_64, score_threshold=bbox_threshold)

        if bbox_result is None:
            return None, state, "BBox prediction failed.", "None"

        state['obbs'] = bbox_result['obbs']
        state['asset_names'] = bbox_result.get('asset_names', [])
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
        scene_id = state.get('scene_id', '')
        room_id = state.get('room_id', '')
        view_idx = state.get('view_idx', 0)
        voxel_64 = state.get('voxel_64')
        obbs = state.get('obbs')
        camera_center = state.get('camera_center')
        encoded_cond = state.get('encoded_cond')

        if voxel_64 is None or obbs is None:
            return None, None, None, state, "Generate CSG and predict/load bboxes first."

        log(f"\n--- Shape Generation (layout={layout_mode}) ---")
        t1 = time.time()
        shape_result = run_stage2_shape_single(
            data_dir, scene_id, room_id, view_idx,
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
                data_dir, scene_id, room_id, view_idx,
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
        scene_id = state.get('scene_id', '')
        room_id = state.get('room_id', '')
        view_idx = state.get('view_idx', 0)
        meshes_list = state.get('meshes_list')

        if not scene_id:
            return None, None, None, None, None, None, "Select a sample first.", ""

        output_dir = os.path.join(
            PROJECT_ROOT, 'demo_outputs', 'custom', scene_id, room_id, f'view{view_idx:04d}')

        save_state = dict(state)
        save_state['scene_id'] = scene_id
        save_state['room_id'] = room_id
        save_state['view_idx'] = view_idx

        saved = save_all_results(output_dir, save_state, meshes_list, data_dir)

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

    with gr.Blocks(title="InSpace Custom") as demo:
        gr.HTML(header_html("Custom demo · run InSpace on your own 360° panorama"))
        state = gr.State({})

        with gr.Row():
            # ============ LEFT PANEL ============
            with gr.Column(scale=1):

                # --- Input ---
                gr.Markdown("### Input")
                room_dd = gr.Dropdown(
                    choices=ROOM_IDS,
                    value=ROOM_IDS[0] if ROOM_IDS else None,
                    label="Room (uuid/room_name)",
                )
                initial_views = ROOM_VIEWS.get(
                    ROOM_IDS[0], ['0000']) if ROOM_IDS else ['0000']
                view_dd = gr.Dropdown(
                    choices=initial_views, value=initial_views[0],
                    label="View Index",
                )
                load_btn = gr.Button("Load Input", variant="primary")

                # --- Custom data: upload your own ERP panorama (+ optional depth) ---
                with gr.Accordion("Use your own data", open=False):
                    gr.Markdown("Upload an **ERP panorama** (and, optionally, its **ERP depth map** "
                                "to enable SDEdit / PSG), then click **Generate Cubemap & Add**. "
                                "The new sample loads immediately in the viewers on the right.")
                    custom_erp_upload = gr.Image(
                        label="ERP panorama (equirectangular RGB)", type="pil", height=180)
                    custom_depth_upload = gr.File(
                        label="ERP depth map — optional (.npy, HxW float, metric meters)",
                        file_types=[".npy", ".png"])
                    custom_name_box = gr.Textbox(label="Name (optional)", placeholder="my_room")
                    gen_cubemap_btn = gr.Button("Generate Cubemap & Add", variant="secondary")
                    gr.Markdown(
                        "> **Note:** we do not ship a depth estimator. The cubemap alone runs "
                        "Stage 2 from random noise; for **SDEdit / PSG** (layout-guided) inference "
                        "you must supply the ERP depth yourself, e.g. from **DA2** (Depth Anything "
                        "in Any Direction) or **PaGeR** (Unified Panoramic Geometry Estimation). "
                        "The depth must be a raw per-pixel array (`.npy`, same H×W as the ERP, "
                        "depth in meters), not a colorized image.")

                # --- Stage 2: CSG ---
                gr.Markdown("---")
                gr.Markdown("### Stage 2: Coarse Scene Geometry")
                gr.Markdown("*SS flow -> 64^3 voxel (optional SDEdit from DA2 depth)*")
                use_sdedit_cb = gr.Checkbox(
                    label="SDEdit from DA2", value=True,
                    info="Use the DA2 depth point cloud (from the DA2 Point Cloud tab) as the "
                         "initial latent via SDEdit. Uncheck for random-noise generation.")
                alpha_slider = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05,
                    label="Alpha (noise level)",
                    info="0=clean DA2, 1=pure noise. Recommended: 0.5-0.8")
                csg_show_cam_cb = gr.Checkbox(label="Show Camera", value=True)
                csg_max_cubes = gr.Slider(
                    1000, 50000, value=30000, step=1000,
                    label="Max Voxel Cubes",
                    info="Max cubes to render in 3D viewer (more=denser, slower)")
                generate_csg_btn = gr.Button("Generate CSG", variant="primary")

                # --- Stage 3: BBox ---
                gr.Markdown("---")
                gr.Markdown("### Stage 3: 3D BBox Estimation")
                bbox_source_display = gr.Textbox(
                    label="BBox Source", value="None",
                    interactive=False, max_lines=1)
                pred_bbox_btn = gr.Button("Predict BBox", variant="primary")
                bbox_threshold = gr.Slider(
                    0.1, 0.9, value=0.3, step=0.05, label="Score Threshold (predicted)")

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
                            choices=["3D Point Cloud", "Pre-saved SS Latent (voxel)"],
                            value="3D Point Cloud",
                            label="View Mode",
                            info="Switch between DA2 point cloud and pre-saved SS latent voxel visualization")
                        da2_plot = gr.Plot(label="DA2 Depth Point Cloud")
                        da2_info_box = gr.Textbox(label="DA2 Info", lines=3, interactive=False)
                        da2_view_btn = gr.Button("Load View", variant="secondary")

                        gr.Markdown("**Step 1: Lift** — Re-lift from ERP depth with RGB colors")
                        da2_relift_btn = gr.Button("Re-lift with RGB Colors", variant="secondary")

                        gr.Markdown("**Step 2: Crop** — Remove ceiling, crop region to exclude points outside room")
                        with gr.Row():
                            da2_remove_ceiling_cb = gr.Checkbox(
                                label="Remove Ceiling", value=False)
                            da2_ceiling_slider = gr.Slider(
                                0.0, 1.0, value=0.0, step=0.05,
                                label="Ceiling Threshold (m)")
                            da2_rotation_slider = gr.Slider(
                                -180, 180, value=0, step=5,
                                label="Rotation Yaw (deg)",
                                info="Rotate point cloud around vertical axis")
                        with gr.Row():
                            da2_max_depth_slider = gr.Slider(
                                0.0, 20.0, value=0.0, step=0.5,
                                label="Max Depth Cap (m)",
                                info="0=off. Remove points farther than this from camera. Cuts door/window leaks.")
                            da2_sor_slider = gr.Slider(
                                0.0, 5.0, value=0.0, step=0.5,
                                label="SOR Std Ratio",
                                info="0=off. Statistical outlier removal. Lower=more aggressive (try 1.0-2.0).")
                        with gr.Row():
                            da2_use_crop_cb = gr.Checkbox(
                                label="Enable Region Crop", value=False,
                                info="Crop to XYZ bounding box")
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
                        csg_viewer = gr.Model3D(label="Predicted CSG (64^3)", height=500,
                                                clear_color=WHITE_BG)

                    with gr.Tab("BBox + CSG"):
                        bbox_viewer = gr.Model3D(label="BBox + CSG", height=500,
                                                 clear_color=WHITE_BG)

                    with gr.Tab("Mesh"):
                        with gr.Row():
                            combined_viewer = gr.Model3D(
                                label="Overall Scene", height=350, clear_color=WHITE_BG)
                            layout_viewer = gr.Model3D(
                                label="Layout", height=350, clear_color=WHITE_BG)
                            exploded_viewer = gr.Model3D(
                                label="Assets (exploded)", height=350, clear_color=WHITE_BG)
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

        room_dd.change(on_room_change, [room_dd], [view_dd])

        load_btn.click(
            on_load_input,
            [room_dd, view_dd, data_dir_state, state],
            [erp_image, cubemap_image, depth_image, da2_plot, state, da2_info_box, status_box],
        )

        # Custom-data: upload ERP (+ optional depth) -> cubemap -> new sample, then display
        gen_cubemap_btn.click(
            on_generate_cubemap,
            [custom_erp_upload, custom_depth_upload, custom_name_box, data_dir_state, state],
            [room_dd, view_dd, erp_image, cubemap_image, depth_image, da2_plot,
             state, da2_info_box, status_box],
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
             da2_rotation_slider, da2_max_depth_slider, da2_sor_slider,
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
            [room_dd, view_dd, data_dir_state,
             use_sdedit_cb, alpha_slider, seed_slider, steps_s1, cfg_s1,
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
    parser = argparse.ArgumentParser(description='InSpace Custom Demo')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--share', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    demo = create_demo(args.data_dir)
    launch_kwargs = dict(server_name='0.0.0.0', share=args.share,
                         theme=INSPACE_THEME, css=INSPACE_CSS)
    if args.port is not None:
        launch_kwargs['server_port'] = args.port
    demo.launch(**launch_kwargs)
