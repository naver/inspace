# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Backend utilities for the InSpace Gradio demo.

Contains:
- ModelManager: lazy model loading/unloading for GPU memory management
- Single-sample inference functions for each pipeline stage
- PSG loading from pre-computed data
- Save functions for visualizations and meshes
"""

import os
import sys
import json
import glob
import time
import shutil
import colorsys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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
)

def log(msg):
    """Print with flush for real-time output in Gradio."""
    print(msg, flush=True)


from .app_inspace_viz import (
    create_psg_plotly_figure,
    create_voxel_glb,
    create_csg_comparison_glb,
    create_bbox_with_voxel_glb,
    create_scene_glb,
    create_exploded_glb,
    # New rendering functions for save
    reconstruct_trellis_rep,
    render_ss_exterior,
    render_ss_topdown_cam,
    render_ss_interior,
    render_mesh_exterior,
    render_mesh_topdown_cam,
    render_mesh_interior,
    render_bbox_topdown,
    render_cubemap_input,
)


# ============================================================
# Default paths
# ============================================================

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'ERP_3D_FRONT_test_samples')

# Hugging Face repo that also hosts the demo sample datasets (under datasets/).
DEMO_SAMPLES_REPO = "GwanHyeong/InSpace"


def colorize_depth(depth):
    """Turn an (H, W) depth array into a turbo-colormapped RGB PIL image, or None."""
    from PIL import Image as _Image
    d = np.asarray(depth).astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return None
    lo, hi = np.percentile(d[valid], [2, 98])
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    import matplotlib.cm as cm
    rgb = (cm.turbo(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return _Image.fromarray(rgb)


def ensure_demo_samples(data_dir, repo_id=DEMO_SAMPLES_REPO):
    """Auto-download the demo sample set for `data_dir` from Hugging Face if it is
    missing or empty. The samples live at `datasets/<basename(data_dir)>/` in the
    model repo, and are placed into `datasets/<basename(data_dir)>/` locally.

    Returns True if data is available (already present or freshly downloaded).
    """
    name = os.path.basename(os.path.normpath(data_dir))
    if os.path.isdir(data_dir) and any(os.scandir(data_dir)):
        return True
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[demo] huggingface_hub not installed; cannot auto-download samples.")
        return False
    print(f"[demo] '{name}' not found locally -> downloading from {repo_id} (datasets/{name}) ...")
    tmp = os.path.join(PROJECT_ROOT, 'datasets', '_hf_download_tmp')
    try:
        snapshot_download(
            repo_id=repo_id, repo_type="model",
            allow_patterns=[f"datasets/{name}/*"],
            local_dir=tmp,
        )
        src = os.path.join(tmp, 'datasets', name)
        if not os.path.isdir(src):
            print(f"[demo] samples for '{name}' not found in {repo_id}.")
            return False
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        shutil.move(src, data_dir)
        print(f"[demo] samples ready at {data_dir}")
        return True
    except Exception as e:
        print(f"[demo] auto-download failed: {e}\n"
              f"       Manually run: hf download {repo_id} --include 'datasets/{name}/*' --local-dir .")
        return False


DEFAULT_STAGE1_CONFIG = os.path.join(PROJECT_ROOT, 'configs/gen/erp_ss_flow_img_dit_L_16l8_bf16_spatial.json')
DEFAULT_STAGE1_CKPT_DIR = os.path.join(PROJECT_ROOT, 'ckpts/erp_ss_flow_img_dit_L_16l8_bf16_spatial')

DEFAULT_BBOX_CONFIG = os.path.join(PROJECT_ROOT, 'configs/bbox/erp_bbox_centerpoint_v2.json')
DEFAULT_BBOX_CKPT_DIR = os.path.join(PROJECT_ROOT, 'ckpts/bbox_centerpoint')

DEFAULT_SHAPE_CONFIG = os.path.join(PROJECT_ROOT, 'configs/gen/erp_slat_flow_img2shape_asset_aware_bf16.json')
DEFAULT_SHAPE_CKPT_DIR = os.path.join(PROJECT_ROOT, 'ckpts/erp_slat_flow_img2shape_asset_aware_bf16')

DEFAULT_TEX_CONFIG = os.path.join(PROJECT_ROOT, 'configs/gen/erp_slat_flow_imgshape2tex_asset_aware_bf16.json')
DEFAULT_TEX_CKPT_DIR = os.path.join(PROJECT_ROOT, 'ckpts/erp_slat_flow_imgshape2tex_asset_aware_bf16')


# ============================================================
# Utility functions (from eval_pipeline.py)
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
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    denoiser.load_state_dict(state_dict)
    denoiser.eval()
    return denoiser


def discover_samples(data_dir):
    """Discover all test samples. Returns list of (scene_id, room_id)."""
    samples = []
    for scene_id in sorted(os.listdir(data_dir)):
        scene_dir = os.path.join(data_dir, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        for room_id in sorted(os.listdir(scene_dir)):
            room_dir = os.path.join(scene_dir, room_id)
            if not os.path.isdir(room_dir):
                continue
            
            # Check if cubic_fov_120 directory exists and has at least one view subdirectory
            cubic_base = os.path.join(room_dir, 'cubic_fov_120')
            erp_dir = os.path.join(room_dir, 'erp')
            
            has_cubic = False
            has_erp = False
            
            if os.path.isdir(cubic_base):
                # Check if there's at least one view directory (e.g., 0000, 0001, etc.)
                view_dirs = [d for d in os.listdir(cubic_base) 
                            if os.path.isdir(os.path.join(cubic_base, d))]
                if view_dirs:
                    has_cubic = True
            
            if os.path.isdir(erp_dir):
                # Check if there's at least one ERP image file
                erp_files = [f for f in os.listdir(erp_dir) 
                            if f.endswith('_colors.png')]
                if erp_files:
                    has_erp = True
            
            # Add sample if it has either cubic or erp data
            if has_cubic or has_erp:
                samples.append((scene_id, room_id))
    return samples


def load_cubemap_images(data_dir, scene_id, room_id, view_idx=0, image_size=512):
    """Load 6 cubemap face images as [6, 3, H, W] tensor."""
    from PIL import Image
    import torchvision.transforms as T

    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    cubic_dir = os.path.join(data_dir, scene_id, room_id, 'cubic_fov_120', f'{view_idx:04d}')

    transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    faces = []
    for face_name in face_names:
        img_path = os.path.join(cubic_dir, f'{face_name}.png')
        img = Image.open(img_path).convert('RGB')
        faces.append(transform(img))
    return torch.stack(faces)  # [6, 3, H, W]


def load_camera_center(data_dir, scene_id, room_id, view_idx=0):
    """Load and normalize camera center from camera_poses.json + normalization_info.json."""
    sample_dir = os.path.join(data_dir, scene_id, room_id)

    with open(os.path.join(sample_dir, 'camera_poses.json'), 'r') as f:
        camera_data = json.load(f)
    cam_location = camera_data['views'][view_idx]['location']

    norm_info_path = os.path.join(sample_dir, 'mesh_dumps', 'normalization_info.json')
    if not os.path.exists(norm_info_path):
        norm_info_path = os.path.join(sample_dir, 'dual_grid_512', 'normalization_info.json')
    with open(norm_info_path, 'r') as f:
        norm_info = json.load(f)

    center = np.array(norm_info['center'])
    scale = norm_info['scale']
    cam_normalized = (np.array(cam_location) - center) * scale
    return torch.from_numpy(cam_normalized).float()


def load_gt_bboxes(data_dir, scene_id, room_id):
    """Load GT bounding boxes from NPZ file."""
    sample_dir = os.path.join(data_dir, scene_id, room_id)
    bbox_dir = os.path.join(sample_dir, '3d_bounding_box')
    npz_files = glob.glob(os.path.join(bbox_dir, '*_scene_data.npz'))
    if not npz_files:
        return None
    data = np.load(npz_files[0], allow_pickle=True)
    return {
        'obbs': data['obbs'].astype(np.float32),
        'asset_filenames': list(data['asset_filenames']),
        'asset_names': list(data['asset_names']),
        'asset_categories': list(data['asset_categories']) if 'asset_categories' in data else [],
        'n_assets': int(data['n_assets']),
    }

#NOTE: assign_voxels_to_obbs
def assign_voxels_to_obbs(coords_norm, obbs):
    """Assign voxels to OBBs. Returns list of boolean masks."""
    masks = []
    for i in range(obbs.shape[0]):
        obb = obbs[i]
        cx, cy, cz = obb[0], obb[1], obb[2]
        sx, sy, sz = obb[3], obb[4], obb[5]
        yaw = obb[6]
        local = coords_norm - torch.tensor([cx, cy, cz], device=coords_norm.device)
        cos_a, sin_a = torch.cos(-yaw), torch.sin(-yaw)
        rx = local[:, 0] * cos_a - local[:, 1] * sin_a
        ry = local[:, 0] * sin_a + local[:, 1] * cos_a
        rz = local[:, 2]
        inside = (rx.abs() <= sx/2) & (ry.abs() <= sy/2) & (rz.abs() <= sz/2)
        masks.append(inside)
    return masks

#NOTE: detect floor Z layer from overall voxel coords
def detect_floor_z(coords_32):
    """Find floor Z = Z layer with the most voxels (robust to outlier Z layers)."""
    xyz = coords_32[:, 1:4].cpu().numpy().astype(int)
    unique_z, counts = np.unique(xyz[:, 2], return_counts=True)
    return int(unique_z[np.argmax(counts)])


#NOTE: detect layout (floor + walls) from overall voxel grid using floor perimeter
def detect_layout_from_floor_perimeter(coords_32):
    """
    Detect layout voxels (floor + walls) from overall voxel coords.

    Method:
    1. Floor = Z layer with the most voxels (robust to outlier Z layers from furniture legs)
    2. Floor perimeter = floor voxels with at least 1 missing 4-neighbor
    3. Wall = any voxel whose (X,Y) is on the floor perimeter AND Z > floor
    4. Layout = floor voxels + wall voxels

    This detects both exterior walls AND interior partition walls,
    matching the training data where layout comes from layout_wo_ceiling.obj.

    Args:
        coords_32: [N, 4] int tensor (batch_idx, x, y, z)

    Returns:
        layout_mask: [N] boolean mask over coords_32
    """
    xyz = coords_32[:, 1:4].cpu().numpy().astype(int)
    floor_z = detect_floor_z(coords_32)

    # Floor XY footprint
    floor_mask_z = xyz[:, 2] == floor_z
    floor_xy_set = set(map(tuple, xyz[floor_mask_z, :2]))

    # Floor perimeter: floor voxels with at least 1 missing 4-neighbor
    perimeter_xy = set()
    for x, y in floor_xy_set:
        if any((x + dx, y + dy) not in floor_xy_set for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            perimeter_xy.add((x, y))

    # Layout = floor layer + voxels above perimeter (walls)
    layout_mask = torch.zeros(coords_32.shape[0], dtype=torch.bool)
    for i, (x, y, z) in enumerate(xyz):
        if z == floor_z or (z > floor_z and (x, y) in perimeter_xy):
            layout_mask[i] = True

    return layout_mask


#NOTE: seperate overall scene/layout/assets from the CSG voxel grid (64^3)
def construct_stage2_input(voxel_64, obbs, device='cuda', layout_mode='floor_perimeter'):
    """
    Convert 64³ voxel + OBBs to SparseTensor with part_layouts for Stage 2.
    64³ → max_pool3d → 32³ → extract active coords.

    Structure: [overall, layout, asset0, asset1, ...]

    Args:
        voxel_64: [1, 64, 64, 64] binary voxel grid
        obbs: [M, 7] oriented bounding boxes
        device: torch device
        layout_mode: layout detection and asset assignment mode
            - 'floor_perimeter': layout = floor + walls (perimeter detection).
              Assets include all bbox voxels (floor overlap allowed, matches training).
            - 'floor_perimeter_clean': same layout, but assets EXCLUDE floor-layer
              voxels. Reduces floor artifacts on assets like dining tables.
            - 'no_floor_assets': same layout, assets exclude floor-layer voxels.
              (alias for floor_perimeter_clean)

    Returns: (noise_st, part_layouts, coords_32, has_layout, valid_obb_indices)
             or (None, None, None, False, [])
    valid_obb_indices: list of original OBB indices that have >0 voxels.
    """
    voxel_float = voxel_64.float().unsqueeze(0)  # [1, 1, 64, 64, 64]
    voxel_32 = F.max_pool3d(voxel_float, 2, 2, 0) > 0.5

    active = torch.argwhere(voxel_32)
    coords_32 = active[:, [0, 2, 3, 4]].int()
    N_overall = coords_32.shape[0]

    if N_overall == 0:
        return None, None, None, False, []

    coords_norm = (coords_32[:, 1:4].float() + 0.5) / 32.0 - 0.5
    obbs_tensor = torch.from_numpy(obbs).float()
    voxel_masks = assign_voxels_to_obbs(coords_norm, obbs_tensor)

    # Layout = floor + walls (floor perimeter detection, all modes use this)
    layout_mask = detect_layout_from_floor_perimeter(coords_32)
    log(f"  layout_mode={layout_mode}: detected {layout_mask.sum().item()} layout voxels")

    # Determine whether to exclude floor-layer voxels from assets
    exclude_floor_from_assets = layout_mode in ('floor_perimeter_clean', 'no_floor_assets')
    floor_z = detect_floor_z(coords_32) if exclude_floor_from_assets else None

    layout_coords = coords_32[layout_mask]
    n_layout = layout_coords.shape[0]

    all_coords = [coords_32]  # overall scene
    part_layouts = [slice(0, N_overall)]  # overall scene at index 0
    current_idx = N_overall

    # Layout at index 1
    has_layout = n_layout > 0
    if has_layout:
        all_coords.append(layout_coords)
        part_layouts.append(slice(current_idx, current_idx + n_layout))
        current_idx += n_layout

    # Assets at index 2+ (or 1+ if no layout)
    # Skip OBBs with 0 voxels to avoid empty slices that cause batch ID mismatch
    valid_obb_indices = []
    for obb_idx, mask in enumerate(voxel_masks):
        # Optionally exclude floor-layer voxels from asset
        if exclude_floor_from_assets:
            floor_mask = coords_32[:, 3] == floor_z  # Z column
            mask = mask & ~floor_mask

        asset_coords = coords_32[mask]
        n_asset = asset_coords.shape[0]
        if n_asset == 0:
            continue  # Skip empty OBBs entirely
        valid_obb_indices.append(obb_idx)
        all_coords.append(asset_coords)
        part_layouts.append(slice(current_idx, current_idx + n_asset))
        current_idx += n_asset

    all_coords_cat = torch.cat(all_coords, dim=0).to(device)
    noise_feats = torch.randn(all_coords_cat.shape[0], 32, device=device)
    noise_st = SparseTensor(coords=all_coords_cat, feats=noise_feats)

    n_valid = len(valid_obb_indices)
    n_total = len(obbs)
    log(f"  construct_stage2_input: {N_overall} overall, {n_layout} layout, "
        f"{sum(m.sum().item() for m in voxel_masks)} asset voxels, "
        f"{n_valid}/{n_total} OBBs with voxels, has_layout={has_layout}")

    return noise_st, part_layouts, coords_32, has_layout, valid_obb_indices


def inverse_normalize(z, normalization):
    """Apply inverse normalization: z * std + mean."""
    mean = torch.tensor(normalization['mean']).reshape(1, -1).to(z.device)
    std = torch.tensor(normalization['std']).reshape(1, -1).to(z.device)
    return z.replace(feats=z.feats * std + mean)


# ============================================================
# ModelManager: Lazy loading with GPU memory management
# ============================================================

class ModelManager:
    """Manages model loading/unloading to fit within GPU memory."""

    def __init__(self, device='cuda'):
        self.device = device
        self.erp_encoder = None
        self.stage1_denoiser = None
        self.ss_decoder = None
        self.ss_encoder = None
        self.bbox_model = None
        self.stage2_shape_denoiser = None
        self.stage2_tex_denoiser = None
        self.shape_dec = None
        self.pbr_dec = None
        self.tex_shape_dec = None
        self._configs = {}
        self._sampler = None

    def _unload(self, *names):
        """Unload specified models from GPU."""
        for name in names:
            obj = getattr(self, name, None)
            if obj is not None:
                del obj
                setattr(self, name, None)
        torch.cuda.empty_cache()

    def _load_config(self, config_path):
        if config_path not in self._configs:
            with open(config_path, 'r') as f:
                self._configs[config_path] = json.load(f)
        return self._configs[config_path]

    def get_sampler(self, sigma_min=1e-5):
        if self._sampler is None:
            self._sampler = FlowEulerGuidanceIntervalSampler(sigma_min=sigma_min)
        return self._sampler

    def get_erp_encoder(self, config_path=DEFAULT_STAGE1_CONFIG):
        """Load or return cached ERP encoder (persistent across stages)."""
        if self.erp_encoder is None:
            config = self._load_config(config_path)
            trainer_config = config['trainer']['args']
            self.erp_encoder = ERPImageEncoder(
                image_cond_model=trainer_config['image_cond_model'],
                feature_dim=1024,
            ).to(self.device)
        return self.erp_encoder

    def get_stage1(self, config_path=DEFAULT_STAGE1_CONFIG, ckpt_dir=DEFAULT_STAGE1_CKPT_DIR):
        """Load Stage 1 denoiser + SS decoder. Unloads Stage 2 if needed."""
        if self.stage1_denoiser is None:
            self._unload('stage2_shape_denoiser', 'stage2_tex_denoiser', 'bbox_model',
                         'shape_dec', 'pbr_dec', 'tex_shape_dec')
            config = self._load_config(config_path)
            ckpt_step = find_latest_ckpt(ckpt_dir)
            self.stage1_denoiser = load_denoiser(config, ckpt_dir, ckpt_step, self.device)

            pretrained_ss_dec = config['dataset']['args'].get(
                'pretrained_ss_dec',
                'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16'
            )
            self.ss_decoder = models.from_pretrained(pretrained_ss_dec).to(self.device).eval()
        return self.stage1_denoiser, self.ss_decoder

    def get_ss_encoder(self, config_path=DEFAULT_STAGE1_CONFIG):
        """Load SS encoder (for SDEdit inline step9+10)."""
        if self.ss_encoder is None:
            config = self._load_config(config_path)
            pretrained_ss_enc = config['dataset']['args'].get(
                'pretrained_ss_enc',
                'microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16'
            )
            self.ss_encoder = models.from_pretrained(pretrained_ss_enc).to(self.device).eval()
        return self.ss_encoder

    def get_bbox_model(self, config_path=DEFAULT_BBOX_CONFIG, ckpt_dir=DEFAULT_BBOX_CKPT_DIR):
        """Load BBox CenterPoint model."""
        if self.bbox_model is None:
            from trellis2.models.bbox_centerpoint import BBoxCenterPoint
            config = self._load_config(config_path)
            model_args = config['models']['bbox_centerpoint']['args']
            self.bbox_model = BBoxCenterPoint(**model_args).to(self.device).eval()

            ckpt_step = find_latest_ckpt(ckpt_dir, prefix='bbox_centerpoint')
            ckpt_path = os.path.join(ckpt_dir, 'ckpts',
                                     f'bbox_centerpoint_ema0.9999_step{ckpt_step:07d}.pt')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(ckpt_dir, 'ckpts',
                                         f'bbox_centerpoint_step{ckpt_step:07d}.pt')
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.bbox_model.load_state_dict(ckpt, strict=True)
        return self.bbox_model

    def get_stage2_shape(self, config_path=DEFAULT_SHAPE_CONFIG, ckpt_dir=DEFAULT_SHAPE_CKPT_DIR):
        """Load Stage 2 shape denoiser. Unloads Stage 1 denoiser."""
        if self.stage2_shape_denoiser is None:
            self._unload('stage1_denoiser', 'ss_decoder', 'stage2_tex_denoiser')
            config = self._load_config(config_path)
            ckpt_step = find_latest_ckpt(ckpt_dir)
            self.stage2_shape_denoiser = load_denoiser(config, ckpt_dir, ckpt_step, self.device)
        return self.stage2_shape_denoiser

    def get_stage2_texture(self, config_path=DEFAULT_TEX_CONFIG, ckpt_dir=DEFAULT_TEX_CKPT_DIR):
        """Load Stage 2 texture denoiser. Unloads Stage 1 and shape denoiser."""
        if self.stage2_tex_denoiser is None:
            self._unload('stage1_denoiser', 'ss_decoder', 'stage2_shape_denoiser')
            config = self._load_config(config_path)
            ckpt_step = find_latest_ckpt(ckpt_dir)
            self.stage2_tex_denoiser = load_denoiser(config, ckpt_dir, ckpt_step, self.device)
        return self.stage2_tex_denoiser

    def get_shape_decoder(self, config_path=DEFAULT_SHAPE_CONFIG):
        """Load shape SLat decoder."""
        if self.shape_dec is None:
            self._unload('stage1_denoiser', 'ss_decoder',
                         'stage2_shape_denoiser', 'stage2_tex_denoiser')
            config = self._load_config(config_path)
            pretrained = config['dataset']['args'].get(
                'pretrained_slat_dec',
                'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16'
            )
            # Use DATASET resolution (512), not denoiser resolution (32).
            # The decoder upsamples from 32³ to 512³ internally.
            resolution = config['dataset']['args'].get('resolution', 512)
            self.shape_dec = models.from_pretrained(pretrained)
            self.shape_dec.set_resolution(resolution)
            self.shape_dec = self.shape_dec.to(self.device).eval()
            log(f"  Shape decoder loaded: pretrained={pretrained}, resolution={resolution}")
        return self.shape_dec

    def get_texture_decoders(self, config_path=DEFAULT_TEX_CONFIG):
        """Load texture shape decoder + PBR decoder."""
        if self.tex_shape_dec is None or self.pbr_dec is None:
            self._unload('stage1_denoiser', 'ss_decoder',
                         'stage2_shape_denoiser', 'stage2_tex_denoiser', 'shape_dec')
            config = self._load_config(config_path)
            ds_args = config['dataset']['args']
            # Use DATASET resolution (512), not denoiser resolution (32).
            resolution = ds_args.get('resolution', 512)

            pretrained_shape = ds_args.get(
                'pretrained_shape_slat_dec',
                'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16'
            )
            self.tex_shape_dec = models.from_pretrained(pretrained_shape)
            self.tex_shape_dec.set_resolution(resolution)
            self.tex_shape_dec = self.tex_shape_dec.to(self.device).eval()

            pretrained_pbr = ds_args.get(
                'pretrained_pbr_slat_dec',
                'microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16'
            )
            self.pbr_dec = models.from_pretrained(pretrained_pbr)
            self.pbr_dec = self.pbr_dec.to(self.device).eval()
            log(f"  Texture decoders loaded: resolution={resolution}")

        return self.tex_shape_dec, self.pbr_dec


# Global model manager
model_manager = ModelManager()


# ============================================================
# PSG Loading (pre-computed DA2 data)
# ============================================================

def load_psg_data(data_dir, scene_id, room_id, view_idx=0):
    """
    Load pre-computed PSG (Partial Scene Geometry) data.

    Returns:
        dict with: points, colors, camera_center, psg_ss_latent
        or None if data not available
    """
    sample_dir = os.path.join(data_dir, scene_id, room_id)

    # Load PLY point cloud for visualization
    ply_path = os.path.join(sample_dir, 'depth_voxels_da2_64', f'{view_idx:04d}.ply')
    if not os.path.exists(ply_path):
        return None

    mesh = trimesh.load(ply_path)
    points = np.array(mesh.vertices)  # [N, 3] in normalized space
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        colors = np.array(mesh.visual.vertex_colors)[:, :3]
    else:
        colors = ((points + 0.5) * 255).clip(0, 255).astype(np.uint8)

    # Load camera center
    camera_center = load_camera_center(data_dir, scene_id, room_id, view_idx)

    # Load pre-computed SS latent for SDEdit
    latent_dir = os.path.join(
        sample_dir, 'depth_voxels_da2_ss_latent', 'ss_enc_conv3d_16l8_fp16_64'
    )
    latent_path = os.path.join(latent_dir, f'{view_idx:04d}.npz')
    psg_ss_latent = None
    if os.path.exists(latent_path):
        psg_ss_latent = np.load(latent_path)['z']  # [8, 16, 16, 16]

    return {
        'points': points,
        'colors': colors,
        'camera_center': camera_center.numpy(),
        'psg_ss_latent': psg_ss_latent,
    }


# ============================================================
# GT CSG Loading (for comparison)
# ============================================================

@torch.no_grad()
@torch.no_grad()
def load_gt_voxel_64(data_dir, scene_id, room_id):
    """
    Load GT voxel 64³ grid by decoding the GT SS latent.

    Returns:
        np.ndarray [1, 64, 64, 64] binary occupancy, or None
    """
    sample_dir = os.path.join(data_dir, scene_id, room_id)
    latent_dir = os.path.join(sample_dir, 'ss_latents', 'ss_enc_conv3d_16l8_fp16_64')
    latent_path = os.path.join(latent_dir, 'full_room_wo_ceiling.npz')
    if not os.path.exists(latent_path):
        log(f"[GT] SS latent not found: {latent_path}")
        return None

    device = model_manager.device
    z = np.load(latent_path)['z']  # [8, 16, 16, 16]
    z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)

    # Need SS decoder (should already be loaded from Stage 1)
    _, ss_decoder = model_manager.get_stage1()
    voxel = ss_decoder(z_tensor)  # [1, 1, 64, 64, 64]
    torch.cuda.synchronize()
    voxel_binary = (voxel > 0).cpu().numpy()
    return voxel_binary[0]  # [1, 64, 64, 64]


# ============================================================
# Stage 1: Indoor Scene Layout Estimation → CSG
# ============================================================

@torch.no_grad()
def run_stage1_single(
    data_dir, scene_id, room_id, view_idx=0,
    use_psg=False, alpha=0.5,
    psg_ss_latent=None,
    steps=12, cfg_strength=7.5, seed=42,
    camera_center_override=None,
):
    """
    Run Stage 1: Coarse Scene Geometry Generation.

    Returns:
        dict with: ss_latent, voxel_64, encoded_cond, camera_center
    """
    t0 = time.time()
    log(f"[Stage 1] Starting CSG generation (seed={seed}, steps={steps}, cfg={cfg_strength})")
    device = model_manager.device
    config = model_manager._load_config(DEFAULT_STAGE1_CONFIG)
    trainer_config = config['trainer']['args']

    # Load models
    log("[Stage 1] Loading denoiser + SS decoder...")
    denoiser, ss_decoder = model_manager.get_stage1()
    log(f"[Stage 1] Denoiser + SS decoder loaded ({time.time()-t0:.1f}s)")
    t1 = time.time()
    erp_encoder = model_manager.get_erp_encoder()
    log(f"[Stage 1] ERP encoder loaded ({time.time()-t1:.1f}s)")

    # Load cubemap
    t1 = time.time()
    log(f"[Stage 1] Encoding cubemap images ({scene_id}/{room_id}/view {view_idx})...")
    cond = load_cubemap_images(data_dir, scene_id, room_id, view_idx).unsqueeze(0).to(device)
    encoded_cond = erp_encoder(cond)
    torch.cuda.synchronize()
    neg_cond = torch.zeros_like(encoded_cond)
    log(f"[Stage 1] Cubemap encoded ({time.time()-t1:.1f}s)")

    # Prepare noise
    torch.manual_seed(seed)
    sdedit_start_t = None
    if use_psg and psg_ss_latent is not None:
        log(f"[Stage 1] Using PSG initial latent (SDEdit alpha={alpha})")
        x_init = torch.from_numpy(psg_ss_latent).float().unsqueeze(0).to(device)
        sigma_min = trainer_config.get('sigma_min', 1e-5)
        t = alpha
        gaussian_noise = torch.randn_like(x_init)
        noise = (1 - t) * x_init + (sigma_min + (1 - sigma_min) * t) * gaussian_noise
        sdedit_start_t = alpha  # Start denoising from t=alpha, not t=1.0
    else:
        log("[Stage 1] Using random Gaussian noise")
        noise = torch.randn(1, 8, 16, 16, 16, device=device)

    # Spatial attention mask
    extra_kwargs = {}
    use_spatial = trainer_config.get('use_spatial_attention', False)
    if use_spatial:
        log("[Stage 1] Creating spatial attention mask...")
        if camera_center_override is not None:
            camera_center = torch.from_numpy(camera_center_override).float() if isinstance(camera_center_override, np.ndarray) else camera_center_override
        else:
            camera_center = load_camera_center(data_dir, scene_id, room_id, view_idx)
        cross_attn_mask = create_spatial_attention_mask(
            camera_center=camera_center.unsqueeze(0).to(device),
            voxel_resolution=trainer_config.get('voxel_resolution', 16),
            tokens_per_face=trainer_config.get('tokens_per_face', 1029),
            fov_degrees=trainer_config.get('spatial_attention_fov', 120.0),
            soft_mask=trainer_config.get('spatial_attention_soft', True),
            soft_margin=trainer_config.get('spatial_attention_soft_margin', 0.1),
        )
        extra_kwargs['cross_attn_mask'] = cross_attn_mask

    sampler = model_manager.get_sampler()
    rescale_t = 5.0

    t1 = time.time()
    if sdedit_start_t is not None:
        # SDEdit: denoise from t=alpha -> t=0 (not from t=1.0)
        t_seq = np.linspace(sdedit_start_t, 0, steps + 1)
        t_seq_rescaled = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq_rescaled[i], t_seq_rescaled[i + 1]) for i in range(steps))
        log(f"[Stage 1] SDEdit sampling from t={sdedit_start_t} ({steps} steps)")

        sample = noise
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for t_cur, t_prev in t_pairs:
                out = sampler.sample_once(
                    denoiser, sample, t_cur, t_prev, encoded_cond,
                    neg_cond=neg_cond,
                    guidance_strength=cfg_strength,
                    guidance_interval=(0.6, 1.0),
                    guidance_rescale=0.7,
                    **extra_kwargs,
                )
                sample = out.pred_x_prev
        z = sample
    else:
        log(f"[Stage 1] Sampling SS latent ({steps} steps)...")
        with torch.autocast('cuda', dtype=torch.bfloat16):
            res = sampler.sample(
                denoiser,
                noise=noise,
                cond=encoded_cond,
                neg_cond=neg_cond,
                steps=steps,
                rescale_t=rescale_t,
                guidance_strength=cfg_strength,
                guidance_interval=(0.6, 1.0),
                guidance_rescale=0.7,
                verbose=True,
                **extra_kwargs,
            )
        z = res.samples
    torch.cuda.synchronize()
    log(f"[Stage 1] Sampling done ({time.time()-t1:.1f}s)")
    # z: [1, 8, 16, 16, 16]
    t1 = time.time()
    log("[Stage 1] Decoding SS latent -> 64³ voxel grid...")
    voxel = ss_decoder(z.float())  # [1, 1, 64, 64, 64]
    torch.cuda.synchronize()
    voxel_binary = (voxel > 0).cpu()
    n_active = voxel_binary.sum().item()
    log(f"[Stage 1] Done: {n_active} active voxels in 64³ grid ({time.time()-t1:.1f}s)")
    log(f"[Stage 1] Total time: {time.time()-t0:.1f}s")

    if camera_center_override is not None:
        camera_center = torch.from_numpy(camera_center_override).float() if isinstance(camera_center_override, np.ndarray) else camera_center_override
    else:
        camera_center = load_camera_center(data_dir, scene_id, room_id, view_idx)

    return {
        'ss_latent': z[0].cpu().numpy(),
        'voxel_64': voxel_binary[0].numpy(),
        'encoded_cond': encoded_cond.cpu().numpy(),
        'camera_center': camera_center.numpy(),
    }


# ============================================================
# Stage 3: 3D BBOX Estimator
# ============================================================

@torch.no_grad()
def run_bbox_gt_single(data_dir, scene_id, room_id, view_idx=0):
    """Load GT bounding boxes with visibility filtering."""
    bbox_data = load_gt_bboxes(data_dir, scene_id, room_id)
    if bbox_data is None:
        return None

    camera_center = load_camera_center(data_dir, scene_id, room_id, view_idx)

    obbs_tensor = torch.from_numpy(bbox_data['obbs']).float()
    visible_indices, _ = filter_visible_assets(
        obbs_tensor, camera_center,
        visibility_threshold=0.5, fov_degrees=120.0, image_size=512,
    )

    return {
        'obbs': bbox_data['obbs'][visible_indices],
        'asset_filenames': [bbox_data['asset_filenames'][i] for i in visible_indices],
        'asset_names': [bbox_data['asset_names'][i] for i in visible_indices],
        'confidences': np.ones(len(visible_indices)),
        'source': 'gt',
    }


@torch.no_grad()
def run_bbox_predicted_single(voxel_64, score_threshold=0.3):
    """Predict 3D bounding boxes from voxel grid."""
    device = model_manager.device
    bbox_model = model_manager.get_bbox_model()

    voxel_grid = torch.from_numpy(voxel_64).float().unsqueeze(0).to(device)

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = bbox_model(voxel_grid)
    outputs = {k: v.float() for k, v in outputs.items()}

    detections = bbox_model.decode_detections(
        outputs, score_threshold=score_threshold,
        nms_kernel=7, iou_nms_threshold=0.3,
    )

    det = detections[0]
    n_pred = det['pred_centers'].shape[0]
    if n_pred > 0:
        pred_obbs = torch.cat([
            det['pred_centers'], det['pred_sizes'], det['pred_rotations'],
        ], dim=-1).cpu().numpy()
        pred_conf = det['pred_confidences'][:, 0].cpu().numpy()
        sort_idx = np.argsort(-pred_conf)
        pred_obbs = pred_obbs[sort_idx]
        pred_conf = pred_conf[sort_idx]
    else:
        pred_obbs = np.zeros((0, 7), dtype=np.float32)
        pred_conf = np.zeros(0, dtype=np.float32)

    pred_names = [f'asset_{i:03d}' for i in range(len(pred_obbs))]

    return {
        'obbs': pred_obbs,
        'asset_filenames': pred_names,
        'asset_names': pred_names,
        'confidences': pred_conf,
        'source': 'predicted',
    }


# ============================================================
# Stage 4: Layout and Asset-Aware Scene Generation
# ============================================================

@torch.no_grad()
def run_stage2_shape_single(
    data_dir, scene_id, room_id, view_idx,
    voxel_64, obbs, camera_center, encoded_cond,
    steps=50, cfg_strength=3.0, seed=42,
    layout_mode='floor_perimeter',
):
    """Run Stage 2 shape generation with asset-aware attention."""
    log(f"[Stage 2 Shape] Starting (seed={seed}, steps={steps}, cfg={cfg_strength})")
    device = model_manager.device
    config = model_manager._load_config(DEFAULT_SHAPE_CONFIG)
    trainer_config = config['trainer']['args']

    log("[Stage 2 Shape] Loading shape denoiser...")
    denoiser = model_manager.get_stage2_shape()
    erp_encoder = model_manager.get_erp_encoder()

    # Re-encode if needed (erp_encoder may have been reloaded)
    log("[Stage 2 Shape] Encoding cubemap images...")
    cond = load_cubemap_images(data_dir, scene_id, room_id, view_idx).unsqueeze(0).to(device)
    encoded_cond = erp_encoder(cond)
    neg_cond = torch.zeros_like(encoded_cond)

    # Construct input
    log("[Stage 2 Shape] Constructing stage2 input (64³ → 32³ + part layouts)...")
    voxel_t = torch.from_numpy(voxel_64)
    torch.manual_seed(seed)
    noise_st, part_layouts, coords_32, has_layout, valid_obb_indices = construct_stage2_input(voxel_t, obbs, device, layout_mode=layout_mode)

    if noise_st is None:
        return None

    # Filter OBBs to only those with voxels (empty OBBs cause batch ID mismatch)
    obbs_filtered = obbs[valid_obb_indices] if len(valid_obb_indices) > 0 else obbs[:0]

    # Attention masks
    log("[Stage 2 Shape] Computing cross-attention masks...")
    tokens_per_face = trainer_config.get('tokens_per_face', 1029)
    fov_degrees = trainer_config.get('fov_degrees', 120.0)
    expand_pixels = trainer_config.get('expand_pixels', 28)
    overlap_margin = trainer_config.get('overlap_margin', 0.02)
    voxel_resolution = config['models']['denoiser']['args'].get('resolution', 32)

    obbs_tensor = torch.from_numpy(obbs_filtered).float()
    cam_center_t = torch.from_numpy(camera_center).float()

    # Layout voxel coords for cross-attention mask (if layout exists)
    layout_voxel_coords = None
    if has_layout:
        layout_voxel_coords = noise_st.coords[part_layouts[1], 1:4]

    overlap_groups = compute_overlap_groups(obbs_tensor, margin=overlap_margin) if len(obbs_filtered) > 0 else []
    overall_voxel_coords = noise_st.coords[part_layouts[0], 1:4]
    masks = create_per_part_cross_attn_masks(
        obbs=obbs_tensor if len(obbs_filtered) > 0 else None,
        camera_center=cam_center_t,
        num_parts=len(part_layouts),
        tokens_per_face=tokens_per_face,
        fov_degrees=fov_degrees,
        expand_pixels=expand_pixels,
        overall_voxel_coords=overall_voxel_coords,
        layout_voxel_coords=layout_voxel_coords,
        has_layout=has_layout,
        voxel_resolution=voxel_resolution,
    )

    sampler = model_manager.get_sampler()

    log(f"[Stage 2 Shape] Sampling shape latent ({steps} steps, {noise_st.feats.shape[0]} voxels, {len(part_layouts)} parts)...")
    with torch.autocast('cuda', dtype=torch.bfloat16):
        res = sampler.sample(
            denoiser,
            noise=noise_st,
            cond=encoded_cond,
            neg_cond=neg_cond,
            part_layouts=[part_layouts],
            overlap_groups=[overlap_groups],
            cross_attn_masks=[masks],
            has_layout=[has_layout],
            steps=steps,
            guidance_strength=cfg_strength,
            verbose=True,
        )

    shape_latent = res.samples
    log(f"[Stage 2 Shape] Done: output shape = {shape_latent.feats.shape}")

    return {
        'shape_coords': shape_latent.coords.cpu().numpy(),
        'shape_feats': shape_latent.feats.cpu().numpy(),
        'part_layouts': [(s.start, s.stop) for s in part_layouts],
        'has_layout': has_layout,
        'obbs_filtered': obbs_filtered,
    }


@torch.no_grad()
def run_stage2_texture_single(
    data_dir, scene_id, room_id, view_idx,
    shape_coords, shape_feats, part_layouts_raw,
    obbs, camera_center, has_layout=False,
    steps=50, cfg_strength=3.0, seed=42,
):
    """Run Stage 2 texture generation with shape as concat_cond."""
    log(f"[Stage 2 Texture] Starting (seed={seed}, steps={steps}, cfg={cfg_strength})")
    device = model_manager.device
    config = model_manager._load_config(DEFAULT_TEX_CONFIG)
    trainer_config = config['trainer']['args']
    dataset_config = config['dataset']['args']

    log("[Stage 2 Texture] Loading texture denoiser...")
    denoiser = model_manager.get_stage2_texture()
    erp_encoder = model_manager.get_erp_encoder()

    log("[Stage 2 Texture] Encoding cubemap images...")
    cond = load_cubemap_images(data_dir, scene_id, room_id, view_idx).unsqueeze(0).to(device)
    encoded_cond = erp_encoder(cond)
    neg_cond = torch.zeros_like(encoded_cond)

    coords_t = torch.from_numpy(shape_coords).int().to(device)
    feats_t = torch.from_numpy(shape_feats).float().to(device)
    part_layouts = [slice(s, e) for s, e in part_layouts_raw]

    # Shape feats from sampler are already in NORMALIZED form.
    # Official TRELLIS.2 pipeline: sample_shape_slat() denormalizes → sample_tex_slat()
    # re-normalizes → round-trip back to normalized. So we can use directly.
    log(f"[Stage 2 Texture] Using shape latent as concat_cond ({feats_t.shape[0]} voxels, {feats_t.shape[1]}ch)")
    concat_cond = SparseTensor(coords=coords_t, feats=feats_t)

    torch.manual_seed(seed)
    tex_noise_feats = torch.randn(coords_t.shape[0], 32, device=device)
    tex_noise = SparseTensor(coords=coords_t, feats=tex_noise_feats)

    # Attention masks
    log("[Stage 2 Texture] Computing cross-attention masks...")
    tokens_per_face = trainer_config.get('tokens_per_face', 1029)
    fov_degrees = trainer_config.get('fov_degrees', 120.0)
    expand_pixels = trainer_config.get('expand_pixels', 28)
    overlap_margin = trainer_config.get('overlap_margin', 0.02)
    voxel_resolution = config['models']['denoiser']['args'].get('resolution', 32)

    obbs_tensor = torch.from_numpy(obbs).float()
    cam_center_t = torch.from_numpy(camera_center).float()

    # Layout voxel coords for cross-attention mask
    layout_voxel_coords = None
    if has_layout:
        layout_voxel_coords = tex_noise.coords[part_layouts[1], 1:4]

    overlap_groups = compute_overlap_groups(obbs_tensor, margin=overlap_margin) if len(obbs) > 0 else []
    overall_voxel_coords = tex_noise.coords[part_layouts[0], 1:4]
    masks = create_per_part_cross_attn_masks(
        obbs=obbs_tensor if len(obbs) > 0 else None,
        camera_center=cam_center_t,
        num_parts=len(part_layouts),
        tokens_per_face=tokens_per_face,
        fov_degrees=fov_degrees,
        expand_pixels=expand_pixels,
        overall_voxel_coords=overall_voxel_coords,
        layout_voxel_coords=layout_voxel_coords,
        has_layout=has_layout,
        voxel_resolution=voxel_resolution,
    )

    sampler = model_manager.get_sampler()

    log(f"[Stage 2 Texture] Sampling texture latent ({steps} steps, {tex_noise.feats.shape[0]} voxels)...")
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
            has_layout=[has_layout],
            steps=steps,
            guidance_strength=cfg_strength,
            verbose=True,
        )

    tex_latent = res.samples
    log(f"[Stage 2 Texture] Done: output shape = {tex_latent.feats.shape}")

    return {
        'tex_feats': tex_latent.feats.cpu().numpy(),
    }


# ============================================================
# Mesh Decoding
# ============================================================

PBR_ATTR_LAYOUT = {
    'base_color': slice(0, 3),
    'metallic': slice(3, 4),
    'roughness': slice(4, 5),
    'alpha': slice(5, 6),
}


def _mesh_to_trimesh_gray(mesh_obj, color=None):
    """Convert a Mesh to trimesh with uniform face colors.

    Uses ColorVisuals (face colors) instead of PBR materials for
    maximum compatibility with GLB export and Gradio Model3D viewer.

    Args:
        mesh_obj: Mesh with .vertices and .faces
        color: [R, G, B, A] ints in [0, 255]. Defaults to light gray.
    """
    v = mesh_obj.vertices.cpu().numpy()
    f = mesh_obj.faces.cpu().numpy()
    if v.shape[0] == 0 or f.shape[0] == 0:
        return None
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if color is None:
        color = [200, 200, 200, 255]
    face_colors = np.tile(np.array(color, dtype=np.uint8), (f.shape[0], 1))
    tm.visual.face_colors = face_colors
    return tm


def _mesh_to_glb_textured(mesh_obj, vox_st, resolution, part_type='asset'):
    """Convert Mesh + PBR voxel SparseTensor to textured trimesh via to_glb().

    Args:
        mesh_obj: Mesh with .vertices and .faces (fill_holes already called)
        vox_st: SparseTensor with PBR attrs (feats[:, 0:3]=baseColor, etc.)
        resolution: decoder grid resolution (e.g. 512)
        part_type: 'scene', 'layout', or 'asset' — controls decimation_target

    Returns:
        trimesh.Trimesh with baked PBR texture in Z-up space, or None on failure.
    """
    # Decimation targets matching eval_pipeline.py / official TRELLIS2 pipeline
    dec_targets = {'scene': 500000, 'layout': 200000, 'asset': 100000}
    decimation_target = dec_targets.get(part_type, 100000)

    try:
        import o_voxel.postprocess
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh_obj.vertices,
            faces=mesh_obj.faces,
            attr_volume=vox_st.feats,
            coords=vox_st.coords[:, 1:],
            attr_layout=PBR_ATTR_LAYOUT,
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=1024,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        # to_glb() internally applies Z-up→Y-up: y',z' = z,-y
        # Undo it so all meshes stay in Z-up O-Voxel space consistently.
        # _scene_to_glb() will apply the single Z-up→Y-up transform for export.
        v = glb.vertices
        v_y = v[:, 1].copy()
        v[:, 1] = -v[:, 2]
        v[:, 2] = v_y
        glb.vertices = v
        if hasattr(glb, 'vertex_normals') and glb.vertex_normals is not None and len(glb.vertex_normals) > 0:
            n = glb.vertex_normals.copy()
            n_y = n[:, 1].copy()
            n[:, 1] = -n[:, 2]
            n[:, 2] = n_y
            glb.vertex_normals = n
        return glb
    except Exception as e:
        log(f"  Warning: to_glb() failed: {e}")
        import traceback; traceback.print_exc()
        # Fallback to gray material
        return _mesh_to_trimesh_gray(mesh_obj)


@torch.no_grad()
def decode_meshes_single(
    shape_coords, shape_feats, part_layouts_raw,
    tex_feats=None, asset_names=None, has_layout=False,
):
    """
    Decode latents to trimesh meshes.

    Structure: meshes_list[0] = overall, [1] = layout (if has_layout), [2+] = assets.
    Combined Scene uses ONLY meshes_list[0] (overall already has all voxels).
    Exploded View uses asset parts only (skips overall and layout).

    Returns:
        (meshes_list, trellis_rep_data)
        - meshes_list: list of trimesh.Trimesh (index 0 = overall, rest = parts)
        - trellis_rep_data: dict with TRELLIS rep CPU data for the overall mesh (for rendering)
    """
    mode = "textured" if tex_feats is not None else "shape-only"
    log(f"[Decode] Decoding meshes ({mode}, {len(part_layouts_raw)} parts)...")
    device = model_manager.device
    has_texture = tex_feats is not None

    coords_t = torch.from_numpy(shape_coords).int()
    feats_t = torch.from_numpy(shape_feats).float()
    part_layouts = [slice(s, e) for s, e in part_layouts_raw]

    meshes_list = []
    trellis_rep_data = None  # Will store overall TRELLIS rep for rendering

    def _part_label(idx):
        """Return a human-readable label for part index."""
        if idx == 0:
            return "Overall"
        if has_layout and idx == 1:
            return "Layout"
        asset_start = 2 if has_layout else 1
        asset_idx = idx - asset_start
        if asset_names and asset_idx < len(asset_names):
            return f"Asset {asset_idx} ({asset_names[asset_idx]})"
        return f"Asset {asset_idx}"

    if has_texture:
        config_tex = model_manager._load_config(DEFAULT_TEX_CONFIG)
        tex_normalization = config_tex['dataset']['args'].get('normalization', None)
        tex_shape_normalization = config_tex['dataset']['args'].get('shape_normalization', None)
        resolution = config_tex['dataset']['args'].get('resolution', 512)
        tex_feats_t = torch.from_numpy(tex_feats).float()

        # Build tex_layout from config attrs
        tex_attrs = config_tex['dataset']['args'].get('attrs', ['base_color', 'metallic', 'roughness', 'alpha'])
        channels = {'base_color': 3, 'metallic': 1, 'roughness': 1, 'emissive': 3, 'alpha': 1}
        tex_layout = {}
        start = 0
        for attr in tex_attrs:
            tex_layout[attr] = slice(start, start + channels[attr])
            start += channels[attr]

        tex_shape_dec, pbr_dec = model_manager.get_texture_decoders()

        for part_idx, part_slice in enumerate(part_layouts):
            n_voxels = part_slice.stop - part_slice.start
            if n_voxels < 10:
                log(f"  Part {part_idx} [{_part_label(part_idx)}]: skipped (only {n_voxels} voxels)")
                meshes_list.append(None)
                continue

            log(f"  Part {part_idx} [{_part_label(part_idx)}]: decoding {n_voxels} voxels (shape+texture)...")
            part_shape = SparseTensor(
                coords=coords_t[part_slice], feats=feats_t[part_slice],
            ).to(device)
            part_tex = SparseTensor(
                coords=coords_t[part_slice], feats=tex_feats_t[part_slice],
            ).to(device)

            if tex_shape_normalization:
                sm = torch.tensor(tex_shape_normalization['mean']).reshape(1,-1).to(device)
                ss = torch.tensor(tex_shape_normalization['std']).reshape(1,-1).to(device)
                part_shape = part_shape.replace(feats=part_shape.feats * ss + sm)
            if tex_normalization:
                part_tex = inverse_normalize(part_tex, tex_normalization)

            try:
                mesh_list, subs = tex_shape_dec(part_shape, return_subs=True)
                vox = pbr_dec(part_tex, guide_subs=subs) * 0.5 + 0.5
                if mesh_list and len(mesh_list) > 0:
                    # fill_holes() before texture baking (matches official TRELLIS2 pipeline)
                    mesh_list[0].fill_holes()
                    n_v = mesh_list[0].vertices.shape[0]
                    n_f = mesh_list[0].faces.shape[0]
                    log(f"  Part {part_idx} [{_part_label(part_idx)}]: {n_v} verts, {n_f} faces")

                    # Store TRELLIS rep data for overall mesh (for PBR rendering during save)
                    if part_idx == 0:
                        trellis_rep_data = {
                            'is_textured': True,
                            'vertices': mesh_list[0].vertices.cpu().numpy(),
                            'faces': mesh_list[0].faces.cpu().numpy(),
                            'vox_coords': vox[0].coords[:, 1:].cpu().numpy(),
                            'vox_attrs': vox[0].feats.cpu().numpy(),
                            'vox_shape': list(vox[0].shape) + list(vox[0].spatial_shape),
                            'resolution': resolution,
                            'layout': {k: (v.start, v.stop) for k, v in tex_layout.items()},
                        }

                    # Determine part type for decimation target
                    if part_idx == 0:
                        part_type = 'scene'
                    elif has_layout and part_idx == 1:
                        part_type = 'layout'
                    else:
                        part_type = 'asset'

                    # All parts: use to_glb() for proper PBR texture baking
                    log(f"  Part {part_idx} [{_part_label(part_idx)}]: baking PBR texture (to_glb, {part_type})...")
                    tm = _mesh_to_glb_textured(mesh_list[0], vox[0], resolution, part_type=part_type)
                    meshes_list.append(tm)
                else:
                    log(f"  Part {part_idx} [{_part_label(part_idx)}]: decoder returned empty mesh list")
                    meshes_list.append(None)
            except Exception as e:
                log(f"  Error decoding Part {part_idx} [{_part_label(part_idx)}]: {e}")
                import traceback; traceback.print_exc()
                meshes_list.append(None)
    else:
        config_shape = model_manager._load_config(DEFAULT_SHAPE_CONFIG)
        shape_normalization = config_shape['dataset']['args'].get('normalization', None)

        shape_dec = model_manager.get_shape_decoder()

        for part_idx, part_slice in enumerate(part_layouts):
            n_voxels = part_slice.stop - part_slice.start
            if n_voxels < 10:
                log(f"  Part {part_idx} [{_part_label(part_idx)}]: skipped (only {n_voxels} voxels)")
                meshes_list.append(None)
                continue

            log(f"  Part {part_idx} [{_part_label(part_idx)}]: decoding {n_voxels} voxels (shape)...")
            part_z = SparseTensor(
                coords=coords_t[part_slice], feats=feats_t[part_slice],
            ).to(device)

            if shape_normalization:
                part_z = inverse_normalize(part_z, shape_normalization)

            try:
                reps = shape_dec(part_z)
                if reps and len(reps) > 0:
                    # fill_holes() before export (matches official TRELLIS2 pipeline)
                    reps[0].fill_holes()
                    n_v = reps[0].vertices.shape[0]
                    n_f = reps[0].faces.shape[0]
                    log(f"  Part {part_idx} [{_part_label(part_idx)}]: {n_v} verts, {n_f} faces")

                    # Store TRELLIS rep data for overall mesh (for normal rendering during save)
                    if part_idx == 0:
                        trellis_rep_data = {
                            'is_textured': False,
                            'vertices': reps[0].vertices.cpu().numpy(),
                            'faces': reps[0].faces.cpu().numpy(),
                        }

                    if part_idx == 0:
                        # Overall: gray material
                        tm = _mesh_to_trimesh_gray(reps[0])
                    else:
                        # Assets/layout: colored material
                        asset_start = 2 if has_layout else 1
                        hue = (part_idx - asset_start) / max(len(part_layouts) - asset_start, 1)
                        r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.85)
                        tm = _mesh_to_trimesh_gray(reps[0], color=[int(r*255), int(g*255), int(b*255), 255])
                    meshes_list.append(tm)
                else:
                    log(f"  Part {part_idx} [{_part_label(part_idx)}]: decoder returned empty mesh list")
                    meshes_list.append(None)
            except Exception as e:
                log(f"  Error decoding Part {part_idx} [{_part_label(part_idx)}]: {e}")
                import traceback; traceback.print_exc()
                meshes_list.append(None)

    return meshes_list, trellis_rep_data


# ============================================================
# Save Functions
# ============================================================

def save_all_results(
    output_dir, state,
    meshes_list=None,
    data_dir=DEFAULT_DATA_DIR,
):
    """
    Save all visualization results and meshes.

    Outputs:
        {output_dir}/vis/          - Rendered visualization images
            bbox_topdown.png       - BBox top-down view
            cubemap_input.png      - 2x3 cubemap input grid
            ss_exterior_ccm.png    - SS voxel exterior (4 views, CCM colormap)
            ss_topdown_cam_ccm.png - SS voxel top-down with camera center
            ss_interior_ccm.png    - SS voxel interior (cubemap vs rendered)
            geometry_exterior.png  - Mesh geometry exterior (normal map)
            geometry_topdown_cam.png
            geometry_interior.png
            texture_exterior.png   - Mesh texture exterior (PBR, if textured)
            texture_topdown_cam.png
            texture_interior.png
        {output_dir}/meshes/       - Exported mesh files
            scene.obj              - Overall scene
            layout.obj             - Layout (floor+walls) if available
            assets/                - Per-asset OBJs

    Returns:
        dict with paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    saved = {}

    scene_id = state.get('scene_id', '')
    room_id = state.get('room_id', '')
    view_idx = state.get('view_idx', 0)
    camera_center = state.get('camera_center')
    has_texture = state.get('has_texture', False)
    voxel_64 = state.get('voxel_64')
    obbs = state.get('obbs')
    asset_names = state.get('asset_names', [])
    bbox_source = state.get('bbox_source', 'gt')
    trellis_rep_data = state.get('trellis_rep_data')
    has_layout = state.get('has_layout', False)

    cubemap_dir = os.path.join(data_dir, scene_id, room_id, 'cubic_fov_120', f'{view_idx:04d}')

    # Load cubemap images as tensor for interior rendering
    cubemap_tensor = None
    try:
        cubemap_tensor = load_cubemap_images(data_dir, scene_id, room_id, view_idx, image_size=512)
    except Exception as e:
        log(f"  Warning: Could not load cubemap images: {e}")

    tile = 512

    # --- 1. BBox Top-Down ---
    if obbs is not None:
        try:
            log("[Save] Rendering bbox_topdown...")
            img = render_bbox_topdown(obbs, asset_names, camera_center, bbox_source)
            path = os.path.join(vis_dir, 'bbox_topdown.png')
            img.save(path)
            saved['bbox_topdown'] = path
        except Exception as e:
            log(f"  Save bbox_topdown error: {e}")

    # --- 2. Cubemap Input ---
    try:
        log("[Save] Rendering cubemap_input...")
        img = render_cubemap_input(cubemap_dir, f'{scene_id}/{room_id}')
        path = os.path.join(vis_dir, 'cubemap_input.png')
        img.save(path)
        saved['cubemap_input'] = path
    except Exception as e:
        log(f"  Save cubemap_input error: {e}")

    # --- 3. SS Voxel Views (if voxel_64 available) ---
    if voxel_64 is not None:
        try:
            log("[Save] Rendering ss_exterior_ccm...")
            img = render_ss_exterior(voxel_64, tile=tile)
            if img:
                path = os.path.join(vis_dir, 'ss_exterior_ccm.png')
                img.save(path)
                saved['ss_exterior'] = path
        except Exception as e:
            log(f"  Save ss_exterior error: {e}")

        try:
            log("[Save] Rendering ss_topdown_cam_ccm...")
            img = render_ss_topdown_cam(voxel_64, camera_center, tile=tile)
            if img:
                path = os.path.join(vis_dir, 'ss_topdown_cam_ccm.png')
                img.save(path)
                saved['ss_topdown_cam'] = path
        except Exception as e:
            log(f"  Save ss_topdown_cam error: {e}")

        if camera_center is not None:
            try:
                log("[Save] Rendering ss_interior_ccm...")
                img = render_ss_interior(voxel_64, camera_center,
                                         cubemap_images=cubemap_tensor, tile=tile)
                if img:
                    path = os.path.join(vis_dir, 'ss_interior_ccm.png')
                    img.save(path)
                    saved['ss_interior'] = path
            except Exception as e:
                log(f"  Save ss_interior error: {e}")

    # --- 4. Mesh Geometry + Texture Views (if trellis_rep_data available) ---
    if trellis_rep_data is not None:
        try:
            rep = reconstruct_trellis_rep(trellis_rep_data)

            # Geometry views (normal map) - always available
            try:
                log("[Save] Rendering geometry_exterior...")
                img = render_mesh_exterior(rep, tile=tile, use_pbr=False)
                path = os.path.join(vis_dir, 'geometry_exterior.png')
                img.save(path)
                saved['geometry_exterior'] = path
            except Exception as e:
                log(f"  Save geometry_exterior error: {e}")

            try:
                log("[Save] Rendering geometry_topdown_cam...")
                img = render_mesh_topdown_cam(rep, camera_center, tile=tile, use_pbr=False)
                path = os.path.join(vis_dir, 'geometry_topdown_cam.png')
                img.save(path)
                saved['geometry_topdown_cam'] = path
            except Exception as e:
                log(f"  Save geometry_topdown_cam error: {e}")

            if camera_center is not None:
                try:
                    log("[Save] Rendering geometry_interior...")
                    img = render_mesh_interior(rep, camera_center,
                                              cubemap_images=cubemap_tensor, tile=tile,
                                              use_pbr=False)
                    path = os.path.join(vis_dir, 'geometry_interior.png')
                    img.save(path)
                    saved['geometry_interior'] = path
                except Exception as e:
                    log(f"  Save geometry_interior error: {e}")

            # Texture views (PBR) - only if textured
            if has_texture and trellis_rep_data.get('is_textured'):
                try:
                    log("[Save] Rendering texture_exterior...")
                    img = render_mesh_exterior(rep, tile=tile, use_pbr=True)
                    path = os.path.join(vis_dir, 'texture_exterior.png')
                    img.save(path)
                    saved['texture_exterior'] = path
                except Exception as e:
                    log(f"  Save texture_exterior error: {e}")

                try:
                    log("[Save] Rendering texture_topdown_cam...")
                    img = render_mesh_topdown_cam(rep, camera_center, tile=tile, use_pbr=True)
                    path = os.path.join(vis_dir, 'texture_topdown_cam.png')
                    img.save(path)
                    saved['texture_topdown_cam'] = path
                except Exception as e:
                    log(f"  Save texture_topdown_cam error: {e}")

                if camera_center is not None:
                    try:
                        log("[Save] Rendering texture_interior...")
                        img = render_mesh_interior(rep, camera_center,
                                                  cubemap_images=cubemap_tensor, tile=tile,
                                                  use_pbr=True)
                        path = os.path.join(vis_dir, 'texture_interior.png')
                        img.save(path)
                        saved['texture_interior'] = path
                    except Exception as e:
                        log(f"  Save texture_interior error: {e}")

            # Clean up GPU memory
            del rep
            torch.cuda.empty_cache()

        except Exception as e:
            log(f"  Save mesh rendering error: {e}")
            import traceback; traceback.print_exc()

    # --- 5. OBJ Meshes ---
    if meshes_list is not None:
        meshes_dir = os.path.join(output_dir, 'meshes')
        os.makedirs(meshes_dir, exist_ok=True)

        # Scene (overall)
        if len(meshes_list) > 0 and meshes_list[0] is not None:
            scene_path = os.path.join(meshes_dir, 'scene.obj')
            meshes_list[0].export(scene_path)
            saved['scene_obj'] = scene_path

        # Layout (if available)
        if has_layout and len(meshes_list) > 1 and meshes_list[1] is not None:
            layout_path = os.path.join(meshes_dir, 'layout.obj')
            meshes_list[1].export(layout_path)
            saved['layout_obj'] = layout_path

        # Per-asset OBJs
        assets_dir = os.path.join(meshes_dir, 'assets')
        os.makedirs(assets_dir, exist_ok=True)
        asset_start = 2 if has_layout else 1
        for i in range(asset_start, len(meshes_list)):
            if meshes_list[i] is None:
                continue
            asset_idx = i - asset_start
            name = asset_names[asset_idx] if asset_idx < len(asset_names) else f'asset_{asset_idx:03d}'
            safe_name = name.replace('/', '_').replace(' ', '_')
            path = os.path.join(assets_dir, f'{asset_idx:03d}_{safe_name}.obj')
            meshes_list[i].export(path)

        saved['assets_dir'] = assets_dir

    saved['vis_dir'] = vis_dir
    n_saved = len([k for k in saved if k.endswith('.png') or k.endswith('_obj') or k == 'assets_dir'])
    log(f"[Save] Done. {len(saved)} items saved to {output_dir}")
    return saved
