# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
ERP Structured Latent Dataset for Asset-Aware Generation

This dataset is designed for training structured latent flow models (shape/texture)
with asset-aware cross-attention using ERP (cubemap) image conditions.

Key features:
1. Loads overall scene latent + individual asset latents
2. Creates part_layouts for OmniPart-style generation
3. Filters assets by visibility (>= 50% visible in cubemap)
4. Provides 3D bounding box info for cross-attention mask generation
5. Supports cubemap (FOV 120°) image conditioning

Data structure:
    {sample}/
    ├── 3d_bounding_box/{room_name}_scene_data.npz  # OBBs + asset info
    ├── shape_latents/{encoder}_{resolution}/
    │   ├── full_room_wo_ceiling.npz               # Overall scene latent
    │   ├── layout_wo_ceiling.npz                  # Layout latent
    │   └── individual_assets_room_coord/*.npz     # Per-asset latents
    ├── pbr_latents/{encoder}_{resolution}/        # (for texture generation)
    │   └── ...
    ├── cubic_fov_120/{view_idx}/                  # Cubemap images
    │   ├── front.png, right.png, back.png, left.png, top.png, bottom.png
    └── camera_poses.json                          # Camera center info
"""

import os
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import torch
from PIL import Image

from .. import models
from ..modules.sparse import SparseTensor
from ..modules.sparse.basic import sparse_cat
from ..utils.data_utils import load_balanced_group_indices
from ..utils.asset_attention_mask import filter_visible_assets, create_per_part_cross_attn_masks
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics
from tqdm import tqdm

def extract_instance_info(filename: str) -> Optional[Tuple[str, int]]:
    """
    Extract category name and instance number from asset filename.

    Examples:
        'dining_chair_inst005.npz' -> ('dining_chair', 5)
        'dining_chair_dining_chair_77dcc4f8_inst005.glb' -> ('dining_chair', 5)
        'tv_stand_tv_stand_7a57580e_inst000.glb' -> ('tv_stand', 0)
        'three-seat___multi-seat_sofa_inst012.npz' -> ('three-seat___multi-seat_sofa', 12)

    Returns:
        Tuple of (category, instance_number) or None if pattern not found.
    """
    # Extract instance number
    inst_match = re.search(r'_inst(\d+)', filename)
    if not inst_match:
        return None

    inst_num = int(inst_match.group(1))

    # Get everything before _inst as the category candidate
    prefix = filename[:inst_match.start()]

    # For bbox files with format: category_category_uid_inst
    # e.g., "dining_chair_dining_chair_77dcc4f8" -> extract "dining_chair"
    # Check if there's an 8-char hex UID
    uid_match = re.search(r'_([a-f0-9]{8})$', prefix)
    if uid_match:
        prefix = prefix[:uid_match.start()]
        # Now prefix might be "dining_chair_dining_chair"
        # Try to find repeated category name
        half_len = len(prefix) // 2
        if half_len > 0:
            first_half = prefix[:half_len]
            second_half = prefix[half_len + 1:]  # +1 for underscore
            if first_half == second_half:
                return (first_half, inst_num)

    # For latent files with format: category_inst (simpler)
    # e.g., "dining_chair" directly
    return (prefix, inst_num)


def align_bbox_with_latents(
    bbox_filenames: List[str],
    latent_files: List[str],
) -> List[Tuple[int, str]]:
    """
    Align 3d_bbox asset filenames with latent files using instance number matching.

    The matching uses instance number as the primary key, since instance numbers
    are unique within each sample.

    Examples:
        Bbox: 'dining_chair_dining_chair_77dcc4f8_inst005.glb'
        Latent: 'dining_chair_inst005.npz'
        -> Matched by instance number 5

    Returns:
        List of (bbox_idx, latent_filename) tuples for matched assets.
    """
    # Build map of instance number -> latent filename
    latent_inst_map = {}
    for lf in latent_files: # lf = 'bed_king-size_bed_12c0a7d0_inst000.npz'
        info = extract_instance_info(lf) # ('bed_king-size_bed', 0)
        if info:
            category, inst_num = info
            latent_inst_map[inst_num] = lf

    # Match bbox files by instance number
    aligned = []
    for bbox_idx, bbox_filename in enumerate(bbox_filenames):
        info = extract_instance_info(bbox_filename)
        if info:
            _, inst_num = info
            if inst_num in latent_inst_map:
                aligned.append((bbox_idx, latent_inst_map[inst_num]))

    return aligned # if bbox has 8 assets, and latent has 7 assets, then the bbox that doesn't have corresponding latent will be ignored.

# latent_inst_map = {0: 'bed_king-size_bed_12c0a7d0_inst000.npz', 4: 'cabinet_floor-based_cabinet_232df588_inst004.npz', 6: 'lighting_pendant_light_3c35f189_inst006.npz', 5: 'plants_plants_-_floor-based_50e050e8_inst005.npz', 3: 'storage_unit_armoire_1ddf606e_inst003.npz', 1: 'table_night_table_6d9ca14e_inst001.npz', 2: 'table_night_table_6d9ca14e_inst002.npz'}
class ERPStructuredLatentBase:
    """
    Base class for ERP structured latent dataset.

    Handles loading of:
    - Overall scene latent (full_room_wo_ceiling.npz)
    - Individual asset latents (individual_assets_room_coord/*.npz)
    - 3D bounding box info
    - Camera center for cross-attention mask
    - Visibility filtering (>= 50% visible assets only)
    """

    def __init__(
        self,
        root: str,
        *,
        latent_type: str = 'shape',  # 'shape' or 'texture'
        latent_encoder: str = 'shape_enc_next_dc_f16c32_fp16_256',
        aug_bbox_range: Optional[Tuple[int, int]] = None,
        max_num_voxels: int = 32768,
        max_assets: int = 10,
        visibility_threshold: float = 0.5,
        fov_degrees: float = 120.0,
        image_size: int = 512,
        normalization: Optional[dict] = None,
        save_visibility_visualization: bool = False,
        # Texture generation params (only used when latent_type == 'texture')
        shape_latent_encoder: Optional[str] = None,
        shape_normalization: Optional[dict] = None,
        pretrained_shape_slat_dec: Optional[str] = None,
        pretrained_pbr_slat_dec: Optional[str] = None,
        attrs: Optional[List[str]] = None,
    ):
        """
        Args:
            root: Dataset root directory
            latent_type: 'shape' for shape generation, 'texture' for texture generation
            latent_encoder: Encoder name (e.g., 'shape_enc_next_dc_f16c32_fp16_256')
            aug_bbox_range: Optional bbox augmentation range (min, max)
            max_num_voxels: Maximum number of voxels to load
            max_assets: Maximum number of individual assets per sample (default: 10).
                If a scene has more visible assets, randomly sample this many.
            visibility_threshold: Minimum visibility fraction (default: 0.5 = 50%)
            fov_degrees: Cubemap FOV (default: 120)
            image_size: Cubemap image size (default: 512)
            normalization: Normalization stats {'mean': [...], 'std': [...]}
            save_visibility_visualization: Whether to save visibility debug PNG (default: False)
            shape_latent_encoder: Shape encoder name (for texture gen concat_cond)
            shape_normalization: Shape normalization stats {'mean': [...], 'std': [...]}
            pretrained_shape_slat_dec: Pretrained shape decoder path (for texture gen visualization)
            pretrained_pbr_slat_dec: Pretrained PBR/texture decoder path (for texture gen visualization)
            attrs: PBR attribute names (for texture gen, e.g., ['base_color', 'metallic', 'roughness', 'alpha'])
        """
        self.root = root
        self.latent_type = latent_type
        self.latent_encoder = latent_encoder
        self.aug_bbox_range = aug_bbox_range
        self.max_num_voxels = max_num_voxels
        self.max_assets = max_assets
        self.visibility_threshold = visibility_threshold
        self.fov_degrees = fov_degrees
        self.image_size = image_size
        self.normalization = normalization
        self.save_visibility_visualization = save_visibility_visualization

        # Determine latent folder based on type
        if latent_type == 'shape':
            self.latent_folder = 'shape_latents'
        elif latent_type == 'texture':
            self.latent_folder = 'pbr_latents'
        else:
            raise ValueError(f"Unknown latent_type: {latent_type}")

        # Set up normalization
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)

        # Texture generation: shape latent params for concat_cond
        if latent_type == 'texture':
            self.shape_latent_encoder = shape_latent_encoder or 'shape_enc_next_dc_f16c32_fp16_512'
            self.shape_latent_folder = 'shape_latents'
            self.shape_normalization = shape_normalization
            if self.shape_normalization is not None:
                self.shape_mean = torch.tensor(self.shape_normalization['mean']).reshape(1, -1)
                self.shape_std = torch.tensor(self.shape_normalization['std']).reshape(1, -1)
            self.pretrained_shape_slat_dec = pretrained_shape_slat_dec or 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16'
            self.pretrained_pbr_slat_dec = pretrained_pbr_slat_dec or 'microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16'
            self.attrs = attrs or ['base_color', 'metallic', 'roughness', 'alpha']
            # Build layout dict for MeshWithVoxel
            channels = {
                'base_color': 3, 'metallic': 1, 'roughness': 1,
                'emissive': 3, 'alpha': 1,
            }
            self.layout = {}
            start = 0
            for attr in self.attrs:
                self.layout[attr] = slice(start, start + channels[attr])
                start += channels[attr]

        self.value_range = (0, 1)

        # Find all valid samples
        self.samples = self._find_samples()
        print(f"[ERPStructuredLatent] Found {len(self.samples)} samples")

        # Compute loads (number of voxels per sample) for BalancedResumableSampler
        self.loads = self._compute_loads()

        # Compute sample weights (based on room area) for WeightedResumableSampler
        self.sample_weights = self._compute_sample_weights()

    def _find_samples(self) -> List[str]:
        """Find all valid sample directories."""
        samples = []

        for uuid in os.listdir(self.root):
            uuid_dir = os.path.join(self.root, uuid)
            if not os.path.isdir(uuid_dir):
                continue

            for room_name in os.listdir(uuid_dir):
                room_dir = os.path.join(uuid_dir, room_name)
                if not os.path.isdir(room_dir):
                    continue

                # Check required files exist
                latent_dir = os.path.join(room_dir, self.latent_folder, self.latent_encoder)
                overall_latent = os.path.join(latent_dir, 'full_room_wo_ceiling.npz')
                assets_dir = os.path.join(latent_dir, 'individual_assets_room_coord')
                bbox_dir = os.path.join(room_dir, '3d_bounding_box')
                camera_poses = os.path.join(room_dir, 'camera_poses.json')
                norm_info = os.path.join(room_dir, 'mesh_dumps', 'normalization_info.json')

                required_exist = (
                    os.path.exists(overall_latent) and
                    os.path.exists(assets_dir) and
                    os.path.exists(bbox_dir) and
                    os.path.exists(camera_poses) and
                    os.path.exists(norm_info)
                )

                # For texture gen, also check shape latent exists
                if required_exist and self.latent_type == 'texture':
                    shape_latent_dir = os.path.join(room_dir, self.shape_latent_folder, self.shape_latent_encoder)
                    shape_overall = os.path.join(shape_latent_dir, 'full_room_wo_ceiling.npz')
                    shape_assets_dir = os.path.join(shape_latent_dir, 'individual_assets_room_coord')
                    required_exist = os.path.exists(shape_overall) and os.path.exists(shape_assets_dir)

                if required_exist:
                    samples.append(f"{uuid}/{room_name}")

        return sorted(samples)

    def _compute_loads(self) -> List[int]:
        """
        Compute load (number of voxels) for each sample.
        Used by BalancedResumableSampler for balanced batching.
        """
        loads = []
        for sample_path in self.samples:
            sample_dir = os.path.join(self.root, sample_path)
            latent_dir = os.path.join(sample_dir, self.latent_folder, self.latent_encoder)
            overall_path = os.path.join(latent_dir, 'full_room_wo_ceiling.npz')

            try:
                # Load only coords to count voxels (faster than loading feats too)
                data = np.load(overall_path)
                num_voxels = data['coords'].shape[0]
                loads.append(num_voxels)
            except Exception as e:
                print(f"Warning: Could not load {overall_path}: {e}")
                loads.append(10000)  # Default load estimate

        print(f"[ERPStructuredLatent] Computed loads for {len(loads)} samples, "
              f"avg={np.mean(loads):.0f}, min={np.min(loads)}, max={np.max(loads)}")
        return loads

    def _compute_sample_weights(self) -> List[float]:
        """
        Compute per-sample weights based on room area for WeightedResumableSampler.
        Larger rooms get higher weights to compensate for their underrepresentation.

        Weight bins (based on dataset distribution analysis):
            area < 20 m²:  weight = 1.0  (67.8% of data)
            area >= 20 m²: weight = 3.0  (32.2% of data)
        """
        weights = []
        for sample_path in self.samples:
            sample_dir = os.path.join(self.root, sample_path)
            room_info_path = os.path.join(sample_dir, 'room_info.json')

            area = None
            if os.path.exists(room_info_path):
                try:
                    with open(room_info_path) as f:
                        room_info = json.load(f)
                    area = room_info.get('area')
                except Exception:
                    pass

            if area is None:
                weights.append(1.0)
            elif area < 20.0:
                weights.append(1.0)
            elif area < 30.0:
                weights.append(3.0)
            else:
                weights.append(4.0)

        n_total = len(weights)
        n_small = sum(1 for w in weights if w == 1.0)
        n_medium = sum(1 for w in weights if w == 3.0)
        n_large = sum(1 for w in weights if w == 4.0)
        print(f"[ERPStructuredLatent] Sample weights: "
              f"small(w=1.0)={n_small}, medium(w=3.0)={n_medium}, large(w=4.0)={n_large}, "
              f"total={n_total}")
        return weights

    def __len__(self):
        return len(self.samples)

    def _coords_to_ids(self, coords: np.ndarray, max_val: int = 64) -> np.ndarray:
        """Convert 3D coordinates to unique IDs for matching."""
        return coords[:, 0] * max_val**2 + coords[:, 1] * max_val + coords[:, 2]

    def _load_latent(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load latent from npz file."""
        data = np.load(path)
        coords = data['coords'].astype(np.int32)
        feats = data['feats'].astype(np.float32)
        return coords, feats

    def _load_bbox_data(self, sample_dir: str, room_name: str) -> Dict:
        """Load 3D bounding box data."""
        bbox_path = os.path.join(
            sample_dir, '3d_bounding_box', f'{room_name}_scene_data.npz'
        )
        data = np.load(bbox_path, allow_pickle=True)
        return {
            'obbs': data['obbs'],
            'asset_filenames': list(data['asset_filenames']),
            'asset_names': list(data['asset_names']),
            'norm_center': data['norm_center'],
            'norm_scale': data['norm_scale'],
        }

    def _load_camera_center(self, sample_dir: str) -> np.ndarray:
        """Load and normalize camera center."""
        poses_path = os.path.join(sample_dir, 'camera_poses.json')
        with open(poses_path) as f:
            poses = json.load(f)

        norm_path = os.path.join(sample_dir, 'mesh_dumps', 'normalization_info.json')
        with open(norm_path) as f:
            norm_info = json.load(f)

        cam_world = np.array(poses['views'][0]['location'])
        center = np.array(norm_info['center'])
        scale = norm_info['scale']

        cam_normalized = (cam_world - center) * scale
        return cam_normalized.astype(np.float32)

    def get_instance(self, index: int) -> Dict[str, Any]:
        """
        Load a sample with overall scene + visible individual assets.

        Visibility filtering:
        - Only includes assets with >= visibility_threshold visibility
        - Excludes assets where camera is inside their bounding box

        Returns dict with:
            - coords: [N, 3] combined coordinates
            - feats: [N, C+1] combined features (with noise mask channel)
            - part_layouts: List of slices for [overall, asset0, asset1, ...]
            - obbs: [M, 7] oriented bounding boxes for visible assets
            - asset_names: List of asset names (visible only)
            - camera_center: [3] normalized camera center
            - n_visible_assets: number of visible assets
            - shape_feats: [N, 32] shape features (only for texture generation)
        """
        sample_path = self.samples[index]

        sample_dir = os.path.join(self.root, sample_path)
        room_name = sample_path.split('/')[-1]

        # Load latent paths
        latent_dir = os.path.join(sample_dir, self.latent_folder, self.latent_encoder)
        overall_path = os.path.join(latent_dir, 'full_room_wo_ceiling.npz')
        assets_dir = os.path.join(latent_dir, 'individual_assets_room_coord')

        # Load overall scene latent
        overall_coords, overall_feats = self._load_latent(overall_path)

        # For texture gen: also load shape latents
        is_texture = self.latent_type == 'texture'
        if is_texture:
            shape_latent_dir = os.path.join(sample_dir, self.shape_latent_folder, self.shape_latent_encoder)
            shape_overall_path = os.path.join(shape_latent_dir, 'full_room_wo_ceiling.npz')
            shape_assets_dir = os.path.join(shape_latent_dir, 'individual_assets_room_coord')
            shape_overall_coords, shape_overall_feats = self._load_latent(shape_overall_path)
            assert np.array_equal(overall_coords, shape_overall_coords), \
                f"Texture and shape overall coords mismatch: {overall_coords.shape} vs {shape_overall_coords.shape}"

        # Load bbox data and camera center
        bbox_data = self._load_bbox_data(sample_dir, room_name)
        camera_center = self._load_camera_center(sample_dir)
        camera_center_tensor = torch.from_numpy(camera_center).float()

        # Filter assets by visibility
        obbs_tensor = torch.from_numpy(bbox_data['obbs']).float()
        visible_indices, visible_visibilities = filter_visible_assets(
            obbs_tensor, camera_center_tensor,
            visibility_threshold=self.visibility_threshold,
            fov_degrees=self.fov_degrees,
            image_size=self.image_size,
            save_visualization=getattr(self, 'save_visibility_visualization', False),
            sample_dir=sample_dir,
            asset_names=bbox_data['asset_filenames'],
        )

        # Align bbox with latent files
        asset_latent_files = sorted(os.listdir(assets_dir))
        aligned_assets = align_bbox_with_latents(
            bbox_data['asset_filenames'],
            asset_latent_files
        )
        # Filter to only visible assets
        visible_set = set(visible_indices)
        visible_aligned = [(bbox_idx, latent_file)
                          for bbox_idx, latent_file in aligned_assets
                          if bbox_idx in visible_set]

        # Cap number of assets to prevent OOM in cross-attention
        if self.max_assets is not None and len(visible_aligned) > self.max_assets:
            import random as _random
            visible_aligned = _random.sample(visible_aligned, self.max_assets)

        # Prepare outputs
        all_coords = []
        all_feats = []
        all_shape_feats = [] if is_texture else None
        part_layouts = []
        matched_obbs = []
        matched_names = []

        start_idx = 0

        # Add overall scene (first segment, part_layouts[0])
        all_coords.append(torch.from_numpy(overall_coords))
        all_feats.append(torch.from_numpy(overall_feats))
        if is_texture:
            all_shape_feats.append(torch.from_numpy(shape_overall_feats))
        part_layouts.append(slice(start_idx, start_idx + overall_coords.shape[0]))
        start_idx += overall_coords.shape[0]

        # Add layout (floor + walls, no furniture) as part_layouts[1]
        layout_path = os.path.join(latent_dir, 'layout_wo_ceiling.npz')
        has_layout = os.path.exists(layout_path)
        if has_layout:
            layout_coords, layout_feats = self._load_latent(layout_path)
            all_coords.append(torch.from_numpy(layout_coords))
            all_feats.append(torch.from_numpy(layout_feats))
            if is_texture:
                shape_layout_path = os.path.join(shape_latent_dir, 'layout_wo_ceiling.npz')
                if os.path.exists(shape_layout_path):
                    shape_layout_coords, shape_layout_feats = self._load_latent(shape_layout_path)
                    assert np.array_equal(layout_coords, shape_layout_coords), \
                        f"Texture and shape layout coords mismatch"
                    all_shape_feats.append(torch.from_numpy(shape_layout_feats))
                else:
                    # Fallback: use zeros if shape layout doesn't exist yet
                    all_shape_feats.append(torch.zeros(layout_coords.shape[0], shape_overall_feats.shape[1]))
            part_layouts.append(slice(start_idx, start_idx + layout_coords.shape[0]))
            start_idx += layout_coords.shape[0]

        # Add each visible asset
        for bbox_idx, latent_filename in visible_aligned:
            # Load asset latent
            asset_path = os.path.join(assets_dir, latent_filename)
            asset_coords, asset_feats = self._load_latent(asset_path)

            # For texture gen: load corresponding shape latent
            if is_texture:
                shape_asset_path = os.path.join(shape_assets_dir, latent_filename)
                shape_asset_coords, shape_asset_feats = self._load_latent(shape_asset_path)
                assert np.array_equal(asset_coords, shape_asset_coords), \
                    f"Texture and shape asset coords mismatch for {latent_filename}"

            # Add asset voxels directly (no noise voxels)
            all_coords.append(torch.from_numpy(asset_coords))
            all_feats.append(torch.from_numpy(asset_feats))

            if is_texture:
                all_shape_feats.append(torch.from_numpy(shape_asset_feats))

            part_layouts.append(slice(start_idx, start_idx + asset_coords.shape[0]))
            start_idx += asset_coords.shape[0]

            # Store matched OBB and name
            matched_obbs.append(bbox_data['obbs'][bbox_idx])
            matched_names.append(bbox_data['asset_names'][bbox_idx])

        # Concatenate all
        combined_coords = torch.cat(all_coords, dim=0).int()
        combined_feats = torch.cat(all_feats, dim=0).float()

        # Apply normalization
        if self.normalization is not None:
            combined_feats = (combined_feats - self.mean) / self.std

        result = {
            'coords': combined_coords,
            'feats': combined_feats,
            'part_layouts': part_layouts,
            'obbs': torch.from_numpy(np.array(matched_obbs)).float() if matched_obbs else torch.zeros((0, 7)),
            'asset_names': matched_names,
            'camera_center': camera_center_tensor,
            'sample_path': sample_path,
            'n_visible_assets': len(matched_obbs),
            'has_layout': has_layout,
        }

        # For texture gen: add shape features as concat_cond
        if is_texture:
            combined_shape_feats = torch.cat(all_shape_feats, dim=0).float()
            # Normalize shape feats
            if self.shape_normalization is not None:
                combined_shape_feats = (combined_shape_feats - self.shape_mean) / self.shape_std
            result['shape_feats'] = combined_shape_feats

        return result

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            return self.get_instance(index) # -> line 568: get_instance
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            import traceback
            traceback.print_exc()
            return self.__getitem__(np.random.randint(0, len(self)))

    # collate_fn merges 4 batch items into one
    @staticmethod
    def collate_fn(batch, split_size=None):
        """
        Collate function for batching samples.
        """
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices(
                [b['coords'].shape[0] for b in batch], split_size
            )
        # batch[0].keys() = dict_keys(['coords', 'feats', 'part_layouts', 'obbs', 'asset_names', 'camera_center', 'sample_path', 'n_visible_assets', 'cond'])
        packs = []
        for group in group_idx: # group_idx = [[0, 1, 2, 3]] # the number of assets, group = [0, 1, 2, 3]

            sub_batch = [batch[i] for i in group] # sub_batch[0].keys() = dict_keys(['coords', 'feats', 'part_layouts', 'obbs', 'asset_names', 'camera_center', 'sample_path', 'n_visible_assets', 'cond'])
            pack = {}

            # Part layouts per sample
            pack['part_layouts'] = [b['part_layouts'] for b in sub_batch]
            # pack['part_layouts'] = [[slice(0, 1088, None), slice(1088, 1232, None) ...]]
            # OBBs and asset names per sample
            pack['obbs'] = [b['obbs'] for b in sub_batch]
            pack['asset_names'] = [b['asset_names'] for b in sub_batch]

            # Camera centers
            pack['camera_center'] = torch.stack([b['camera_center'] for b in sub_batch])

            # Cubemap images (if present) - key name 'cond' matches vis_cond/training_losses
            if 'cond' in sub_batch[0]:
                pack['cond'] = torch.stack([b['cond'] for b in sub_batch])

            # Build sparse tensor
            coords = []
            feats = []
            layout = []
            start = 0
            # pack.keys() = dict_keys(['part_layouts', 'obbs', 'asset_names', 'camera_center', 'cond'])
            for i, b in enumerate(sub_batch): # stack and concatenate the asset coords one by one
                coords.append(torch.cat([
                    torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), # [1498, 1]
                    b['coords'] # [1498, 3]
                ], dim=-1))
                feats.append(b['feats']) # b['feats'].shape = [1498, 32], b['coords'].shape[0] = [1498, 3]
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]

            coords = torch.cat(coords) # [5855, 4]
            feats = torch.cat(feats) # [5855, 32]

            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)

            # Build concat_cond SparseTensor for texture generation
            if 'shape_feats' in sub_batch[0]:
                shape_feats_list = []
                for i, b in enumerate(sub_batch):
                    shape_feats_list.append(b['shape_feats'])
                shape_feats_cat = torch.cat(shape_feats_list)  # same N as feats
                pack['concat_cond'] = SparseTensor(
                    coords=coords.clone(),  # same coords as x_0
                    feats=shape_feats_cat,
                )
                pack['concat_cond']._shape = torch.Size([len(group), *sub_batch[0]['shape_feats'].shape[1:]])
                pack['concat_cond'].register_spatial_cache('layout', [slice(s.start, s.stop) for s in layout])

            # Sample paths for debugging
            pack['sample_paths'] = [b['sample_path'] for b in sub_batch]
            pack['n_visible_assets'] = [b['n_visible_assets'] for b in sub_batch]
            pack['has_layout'] = [b.get('has_layout', False) for b in sub_batch]

            packs.append(pack)

        if split_size is None:
            return packs[0]
        return packs


class ERPCubemapConditionedMixin:
    """
    Mixin for adding cubemap image conditioning.
    """

    def __init__(
        self,
        *args,
        cubemap_image_size: int = 512,
        cubemap_folder: str = 'cubic_fov_120',
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.cubemap_image_size = cubemap_image_size
        self.cubemap_folder = cubemap_folder

    def _load_cubemap_images(self, sample_dir: str, view_idx: int = 0) -> torch.Tensor:
        """
        Load 6 cubemap images and stack them.

        Returns:
            Tensor of shape [6, 3, H, W]
        """
        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        cubemap_dir = os.path.join(
            sample_dir, self.cubemap_folder, f'{view_idx:04d}'
        )

        images = []
        for face_name in face_names:
            img_path = os.path.join(cubemap_dir, f'{face_name}.png')
            img = Image.open(img_path).convert('RGB')

            if img.size[0] != self.cubemap_image_size:
                img = img.resize(
                    (self.cubemap_image_size, self.cubemap_image_size),
                    Image.LANCZOS
                )

            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            images.append(img_tensor)

        return torch.stack(images)  # [6, 3, H, W]

    def get_instance(self, index: int) -> Dict[str, Any]:
        """Override to add cubemap images."""
        data = super().get_instance(index) # -> line 291: get_instance 

        sample_dir = os.path.join(self.root, data['sample_path'])

        # Load cubemap images (use view 0 by default)
        data['cond'] = self._load_cubemap_images(sample_dir, view_idx=0)

        return data

    @staticmethod
    def collate_fn(batch, split_size=None):
        """Override to handle cubemap images (stored as 'cond' for trainer compatibility)."""
        result = ERPStructuredLatentBase.collate_fn(batch, split_size)

        if isinstance(result, list):
            for i, pack in enumerate(result):
                # Get indices for this pack
                group_size = len(pack['sample_paths'])
                start = sum(len(result[j]['sample_paths']) for j in range(i))
                pack['cond'] = torch.stack([
                    batch[start + k]['cond'] for k in range(group_size)
                ])
        else:
            result['cond'] = torch.stack([
                b['cond'] for b in batch
            ])

        return result


class ERPSLatVisMixin:
    """
    Mixin class for visualization of ERP structured latent representations.
    Handles loading of latent decoders and rendering 3D structures from latent codes.
    """

    def __init__(
        self,
        *args,
        pretrained_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        resolution: int = 256,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.slat_dec = None
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt
        self.resolution = resolution

    def _loading_slat_dec(self):
        """Load the structured latent decoder model(s) if not already loaded."""
        if self.latent_type == 'texture':
            # Texture mode: load both shape decoder and PBR decoder
            if getattr(self, 'shape_slat_dec', None) is not None and getattr(self, 'pbr_slat_dec', None) is not None:
                return
            # Shape decoder
            shape_dec = models.from_pretrained(self.pretrained_shape_slat_dec)
            shape_dec.set_resolution(self.resolution)
            self.shape_slat_dec = shape_dec.cuda().eval()
            # PBR/texture decoder
            pbr_dec = models.from_pretrained(self.pretrained_pbr_slat_dec)
            self.pbr_slat_dec = pbr_dec.cuda().eval()
        else:
            # Shape mode: load single decoder
            if self.slat_dec is not None:
                return
            if self.slat_dec_path is not None:
                cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
                decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
                ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_{self.slat_dec_ckpt}.pt')
                decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
            else:
                decoder = models.from_pretrained(self.pretrained_slat_dec)
            decoder.set_resolution(self.resolution)
            self.slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        """Delete the decoder model(s) to free up memory."""
        if getattr(self, 'slat_dec', None) is not None:
            del self.slat_dec
            self.slat_dec = None
        if getattr(self, 'shape_slat_dec', None) is not None:
            del self.shape_slat_dec
            self.shape_slat_dec = None
        if getattr(self, 'pbr_slat_dec', None) is not None:
            del self.pbr_slat_dec
            self.pbr_slat_dec = None

    def _remove_noise_voxels(self, z_batch: SparseTensor) -> Optional[SparseTensor]:
        """
        Legacy compatibility wrapper. With noise mask channel removed,
        this simply returns the input unchanged.
        """
        return z_batch

    def _extract_overall_only(self, data: dict) -> SparseTensor:
        """
        Extract only the overall scene (part_layouts[0]) from each sample
        in the batched SparseTensor. This avoids feeding duplicate voxels
        (overall + parts) into the decoder.

        NOTE: Layout is excluded here because its voxels overlap with overall,
        causing duplicate coords that produce flickering artifacts in the decoded mesh.

        Returns a new SparseTensor containing only the overall voxels per sample.
        """
        x_0 = data['x_0']
        part_layouts = data.get('part_layouts', None)

        if part_layouts is None:
            return x_0

        overall_parts = []
        for sample_idx, sample_layouts in enumerate(part_layouts):
            sample_z = x_0[sample_idx]
            overall_slice = sample_layouts[0]
            overall_z = SparseTensor(
                coords=sample_z.coords[overall_slice],
                feats=sample_z.feats[overall_slice],
            )
            overall_parts.append(overall_z)

        return sparse_cat(overall_parts)

    def _extract_parts_separately(self, data: dict) -> Tuple[SparseTensor, List[List[str]]]:
        """
        Extract each part (overall, layout, asset0, asset1, ...) as a separate batch element.
        This is the OmniPart-style decoding approach.

        When has_layout is True:
            part_layouts = [overall, layout, asset0, asset1, ...]
            Labels = ['overall', 'layout', asset_name0, asset_name1, ...]
        When has_layout is False:
            part_layouts = [overall, asset0, asset1, ...]
            Labels = ['overall', asset_name0, asset_name1, ...]

        Returns:
            parts_sparse: SparseTensor where each batch element is one part
            part_labels: List of lists of labels per sample
        """
        x_0 = data['x_0']
        part_layouts = data.get('part_layouts', None)
        asset_names = data.get('asset_names', None)
        has_layout_list = data.get('has_layout', None)

        if part_layouts is None:
            return x_0, [['scene']] * x_0.shape[0]

        all_parts = []
        part_labels = []
        for sample_idx, sample_layouts in enumerate(part_layouts): # holds 4 batch items
            has_layout = has_layout_list[sample_idx] if has_layout_list is not None else False
            asset_start_idx = 2 if has_layout else 1  # Assets start after overall (+ layout)

            sample_path = data['sample_paths'][sample_idx]
            n_visible_assets = data['n_visible_assets'][sample_idx]
            print(f"sample_path: {sample_path}, n_visible_assets: {n_visible_assets}, has_layout: {has_layout}")

            sample_z = x_0[sample_idx]
            sample_labels = []
            for part_idx, part_slice in tqdm(enumerate(sample_layouts), desc='Extracting parts'):
                part_z = SparseTensor(
                    coords=sample_z.coords[part_slice],
                    feats=sample_z.feats[part_slice],
                )
                all_parts.append(part_z)
                if part_idx == 0:
                    sample_labels.append('overall')
                elif has_layout and part_idx == 1:
                    sample_labels.append('layout')
                elif asset_names is not None and sample_idx < len(asset_names):
                    asset_idx = part_idx - asset_start_idx
                    if asset_idx < len(asset_names[sample_idx]):
                        sample_labels.append(asset_names[sample_idx][asset_idx])
                    else:
                        sample_labels.append(f'asset_{asset_idx}')
                else:
                    sample_labels.append(f'asset_{part_idx - asset_start_idx}')
            part_labels.append(sample_labels)

        return sparse_cat(all_parts), part_labels

    def _inverse_normalize(self, z: SparseTensor) -> SparseTensor:
        """Apply inverse normalization to latent features."""
        if self.normalization is None:
            return z
        mean = torch.tensor(self.normalization['mean']).reshape(1, -1).to(z.device)
        std = torch.tensor(self.normalization['std']).reshape(1, -1).to(z.device)
        return z.replace(feats=z.feats * std + mean)

    def _inverse_normalize_shape(self, z: SparseTensor) -> SparseTensor:
        """Apply inverse normalization to shape latent features (for texture gen concat_cond)."""
        if not hasattr(self, 'shape_normalization') or self.shape_normalization is None:
            return z
        mean = torch.tensor(self.shape_normalization['mean']).reshape(1, -1).to(z.device)
        std = torch.tensor(self.shape_normalization['std']).reshape(1, -1).to(z.device)
        return z.replace(feats=z.feats * std + mean)

    @torch.no_grad()
    def decode_latent(self, z: SparseTensor, shape_z: Optional[SparseTensor] = None, batch_size: int = 4):
        """
        Decode latent vectors into 3D representations.

        For shape mode: z → shape_dec → List[Mesh]
        For texture mode: z (texture) + shape_z (shape) → dual decoders → List[MeshWithVoxel]

        Args:
            z: Texture/shape latent SparseTensor
            shape_z: Shape latent SparseTensor (required for texture mode)
            batch_size: Batch size for decoding
        """
        self._loading_slat_dec()
        reps = []

        if shape_z is not None and self.latent_type == 'texture':
            # Texture mode: dual decoder path
            from ..representations import MeshWithVoxel
            z = self._inverse_normalize(z)
            shape_z = self._inverse_normalize_shape(shape_z)

            for i in range(0, z.shape[0], batch_size):
                tex_batch = z[i:i+batch_size]
                shape_batch = shape_z[i:i+batch_size]

                # Remove noise voxels from both (same mask since coords are identical)
                tex_batch = self._remove_noise_voxels(tex_batch)
                shape_batch = self._remove_noise_voxels(shape_batch)

                if tex_batch is not None and shape_batch is not None:
                    mesh, subs = self.shape_slat_dec(shape_batch, return_subs=True)
                    vox = self.pbr_slat_dec(tex_batch, guide_subs=subs) * 0.5 + 0.5
                    reps.extend([
                        MeshWithVoxel(
                            m.vertices, m.faces,
                            origin=[-0.5, -0.5, -0.5],
                            voxel_size=1 / self.resolution,
                            coords=v.coords[:, 1:],
                            attrs=v.feats,
                            voxel_shape=torch.Size([*v.shape, *v.spatial_shape]),
                            layout=self.layout,
                        )
                        for m, v in zip(mesh, vox)
                    ])
        else:
            # Shape mode: single decoder path
            z = self._inverse_normalize(z)
            for i in range(0, z.shape[0], batch_size):
                z_batch = z[i:i+batch_size]
                z_batch = self._remove_noise_voxels(z_batch)
                if z_batch is not None:
                    reps.append(self.slat_dec(z_batch))
            reps = sum(reps, [])

        self._delete_slat_dec()
        return reps

    @torch.no_grad()
    def decode_latent_parts(self, z: SparseTensor, shape_z: SparseTensor = None, min_voxels: int = 10):
        """
        Decode parts one-by-one, returning an aligned list where each index
        corresponds to the input batch index. Returns None for parts that are
        too small or fail to decode.

        Unlike decode_latent() which can drop parts via _remove_noise_voxels
        and break alignment, this method guarantees len(result) == z.shape[0].

        Args:
            z: SparseTensor with shape[0] = total number of parts
            shape_z: Optional shape SparseTensor for texture mode (dual decoder)
            min_voxels: Minimum voxels after noise removal to attempt decoding

        Returns:
            List of length z.shape[0], with decoded representation or None per part.
        """
        self._loading_slat_dec()
        z = self._inverse_normalize(z)
        use_dual_decoder = (shape_z is not None and self.latent_type == 'texture')
        if use_dual_decoder:
            from ..representations import MeshWithVoxel
            shape_z = self._inverse_normalize_shape(shape_z)

        reps = []
        for i in tqdm(range(z.shape[0]), desc='remove noise voxels -> Decoding latent parts'):
            z_single = z[i:i+1]
            z_filtered = self._remove_noise_voxels(z_single)

            if z_filtered is None or z_filtered.feats.shape[0] < min_voxels:
                reps.append(None)
                continue

            try:
                if use_dual_decoder:
                    shape_single = shape_z[i:i+1]
                    shape_filtered = self._remove_noise_voxels(shape_single)
                    if shape_filtered is None or shape_filtered.feats.shape[0] < min_voxels:
                        reps.append(None)
                        continue
                    mesh, subs = self.shape_slat_dec(shape_filtered, return_subs=True)
                    vox = self.pbr_slat_dec(z_filtered, guide_subs=subs) * 0.5 + 0.5
                    rep = MeshWithVoxel(
                        mesh[0].vertices, mesh[0].faces,
                        origin=[-0.5, -0.5, -0.5],
                        voxel_size=1 / self.resolution,
                        coords=vox[0].coords[:, 1:],
                        attrs=vox[0].feats,
                        voxel_shape=torch.Size([*vox[0].shape, *vox[0].spatial_shape]),
                        layout=self.layout,
                    )
                    reps.append(rep)
                else:
                    rep = self.slat_dec(z_filtered)
                    reps.append(rep[0])
            except Exception as e:
                print(f'  Warning: Failed to decode part {i} ({z_filtered.feats.shape[0]} voxels): {e}')
                reps.append(None)

        self._delete_slat_dec()
        return reps

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        """
        Generate multi-view renderings of a 3D representation.
        Uses only the overall scene (excludes individual asset parts to avoid
        duplicate voxels).

        For shape mode: renders normal maps → Tensor [B, 3, 1024, 1024]
        For texture mode: renders PBR shaded + normal → dict {'shaded': Tensor, 'normal': Tensor}

        Args:
            x_0: SparseTensor or dict containing SparseTensor + part_layouts

        Returns:
            Tensor or dict of rendered images from multiple viewpoints
        """
        if isinstance(x_0, dict) and 'part_layouts' in x_0:
            tex_z = self._extract_overall_only(x_0)
        elif isinstance(x_0, SparseTensor):
            tex_z = x_0
        elif isinstance(x_0, dict):
            tex_z = x_0['x_0']
        else:
            raise ValueError(f"Unsupported input type: {type(x_0)}")

        # For texture mode: also extract shape concat_cond
        shape_z = None
        if self.latent_type == 'texture' and isinstance(x_0, dict) and 'concat_cond' in x_0:
            shape_data = {'x_0': x_0['concat_cond'], 'part_layouts': x_0.get('part_layouts')}
            shape_z = self._extract_overall_only(shape_data)

        reps = self.decode_latent(tex_z.cuda(), shape_z=shape_z.cuda() if shape_z is not None else None)

        if len(reps) == 0:
            if self.latent_type == 'texture':
                return {'shaded': torch.zeros(1, 3, 1024, 1024).cuda(), 'normal': torch.zeros(1, 3, 1024, 1024).cuda()}
            return torch.zeros(1, 3, 1024, 1024).cuda()

        # Build camera parameters
        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        if self.latent_type == 'texture':
            # PBR rendering with environment map
            return self._render_pbr_multiview(reps, exts, ints)
        else:
            # Normal map rendering (shape mode)
            renderer = get_renderer(reps[0])
            images = []
            for representation in reps:
                image = torch.zeros(3, 1024, 1024).cuda()
                tile = [2, 2]
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    res = renderer.render(representation, ext, intr)
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1),
                          512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['normal']
                images.append(image)
            return torch.stack(images)

    def _render_pbr_multiview(self, reps, exts, ints):
        """
        Render MeshWithVoxel representations using PBR renderer with HDRI environment map.

        Args:
            reps: List of MeshWithVoxel representations
            exts: List of extrinsic matrices
            ints: List of intrinsic matrices

        Returns:
            dict with 'shaded' and 'normal' tensors of shape [B, 3, 1024, 1024]
        """
        import os
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        import cv2
        from ..renderers import PbrMeshRenderer, EnvMap

        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8

        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/interior.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))

        result_images = {}
        for representation in reps:
            image = {}
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, envmap=envmap)
                for k, v in res.items():
                    if k not in result_images:
                        result_images[k] = []
                    if k not in image:
                        image[k] = torch.zeros(3, 1024, 1024).cuda()
                    image[k][:, 512 * (j // tile[1]):512 * (j // tile[1] + 1),
                             512 * (j % tile[1]):512 * (j % tile[1] + 1)] = v
            for k in result_images.keys():
                result_images[k].append(image[k])

        for k in result_images.keys():
            result_images[k] = torch.stack(result_images[k], dim=0)

        return result_images

    def _render_pbr_single_view(self, reps, ext, intr, resolution=512):
        """
        Render MeshWithVoxel representations from a single view using PBR renderer.
        Returns the 'shaded' channel as tensor [B, 3, resolution, resolution].
        """
        import os
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        import cv2
        from ..renderers import PbrMeshRenderer, EnvMap

        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = resolution
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8

        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/interior.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))

        images = []
        for representation in reps:
            res = renderer.render(representation, ext, intr, envmap=envmap)
            images.append(res['shaded'])

        return torch.stack(images)

    def _render_base_color_single_view(self, reps, ext, intr, resolution=512):
        """
        Render MeshWithVoxel representations from a single view using PBR renderer.
        Returns the 'base_color' channel (no lighting) as tensor [B, 3, resolution, resolution].
        """
        import os
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        import cv2
        from ..renderers import PbrMeshRenderer, EnvMap

        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = resolution
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8

        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/interior.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))

        images = []
        for representation in reps:
            res = renderer.render(representation, ext, intr, envmap=envmap)
            images.append(res['base_color'])

        return torch.stack(images)

    @torch.no_grad()
    def _extract_tex_and_shape(self, data: Union[SparseTensor, dict]) -> Tuple[SparseTensor, Optional[SparseTensor]]:
        """
        Extract texture and shape SparseTensors from data dict.
        For shape mode, returns (tex_z, None).
        For texture mode, returns (tex_z, shape_z).
        """
        if isinstance(data, dict) and 'part_layouts' in data:
            tex_z = self._extract_overall_only(data)
        elif isinstance(data, SparseTensor):
            tex_z = data
        elif isinstance(data, dict):
            tex_z = data['x_0']
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")

        shape_z = None
        if self.latent_type == 'texture' and isinstance(data, dict) and 'concat_cond' in data:
            shape_data = {'x_0': data['concat_cond'], 'part_layouts': data.get('part_layouts')}
            shape_z = self._extract_overall_only(shape_data)

        return tex_z, shape_z

    @torch.no_grad()
    def visualize_sample_topdown(self, x_0: Union[SparseTensor, dict]):
        """
        Visualize structured latents from a top-down view (pitch=90, looking straight down).
        Uses only the overall scene.

        Returns [B, 3, 512, 512] tensor of rendered images.
        """
        tex_z, shape_z = self._extract_tex_and_shape(x_0)
        reps = self.decode_latent(tex_z.cuda(), shape_z=shape_z.cuda() if shape_z is not None else None)

        if len(reps) == 0:
            return torch.zeros(1, 3, 512, 512).cuda()

        # Top-down camera: yaw=0, pitch=90 degrees (straight down)
        yaw = [0]
        pitch = [90 / 180 * np.pi]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        if self.latent_type == 'texture':
            # PBR rendering for texture mode — return dict with shaded + base_color
            return {
                'shaded': self._render_pbr_single_view(reps, exts[0], ints[0], resolution=512),
                'base_color': self._render_base_color_single_view(reps, exts[0], ints[0], resolution=512),
            }
        else:
            renderer = get_renderer(reps[0])
            renderer.rendering_options.resolution = 512
            images = []
            for representation in reps:
                res = renderer.render(representation, exts[0], ints[0])
                images.append(res['normal'])
            return torch.stack(images)

    @torch.no_grad()
    def visualize_sample_topdown_camera_center(self, data: dict):
        """
        Visualize top-down view with camera center marked as a cyan circle.
        Uses only the overall scene.

        Returns [B, 3, 512, 512], or None if camera_center is missing.
        """
        from PIL import ImageDraw

        if not isinstance(data, dict) or 'camera_center' not in data:
            return None

        camera_centers = data['camera_center']  # [B, 3]

        tex_z, shape_z = self._extract_tex_and_shape(data)
        reps = self.decode_latent(tex_z.cuda(), shape_z=shape_z.cuda() if shape_z is not None else None)

        if len(reps) == 0:
            return torch.zeros(1, 3, 512, 512).cuda()

        render_res = 512
        yaw = [0]
        pitch = [90 / 180 * np.pi]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        if self.latent_type == 'texture':
            # PBR rendering for texture mode
            import os
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
            import cv2
            from ..renderers import PbrMeshRenderer, EnvMap
            renderer = PbrMeshRenderer()
            renderer.rendering_options.resolution = render_res
            renderer.rendering_options.near = 1
            renderer.rendering_options.far = 100
            renderer.rendering_options.ssaa = 2
            renderer.rendering_options.peel_layers = 8
            envmap = EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread('assets/hdri/interior.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            ))
            render_keys = ['shaded', 'base_color']
        else:
            renderer = get_renderer(reps[0])
            renderer.rendering_options.resolution = render_res
            envmap = None
            render_keys = ['normal']

        images_dict = {k: [] for k in render_keys}
        for i, representation in enumerate(reps):
            if envmap is not None:
                res = renderer.render(representation, exts[0], ints[0], envmap=envmap)
            else:
                res = renderer.render(representation, exts[0], ints[0])

            # Project camera center onto the top-down image
            cam_3d = camera_centers[i].float().cuda()  # [3]
            point_h = torch.cat([cam_3d, torch.ones(1, device='cuda')])  # [4]
            point_cam = exts[0] @ point_h  # [4]
            point_proj = ints[0] @ point_cam[:3]  # [3]

            px, py = None, None
            if point_proj[2].abs() > 1e-6:
                u = (point_proj[0] / point_proj[2]).item()
                v = (point_proj[1] / point_proj[2]).item()
                px = u * render_res
                py = v * render_res
                if not (-20 < px < render_res + 20 and -20 < py < render_res + 20):
                    px, py = None, None

            for rk in render_keys:
                face_img = res[rk]  # [3, H, W]
                if px is not None and py is not None:
                    img_np = (face_img.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                    draw = ImageDraw.Draw(pil_img)
                    radius = 8
                    draw.ellipse(
                        [px - radius, py - radius, px + radius, py + radius],
                        fill=(0, 255, 255),
                        outline=(255, 255, 255),
                        width=2,
                    )
                    face_img = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float().cuda() / 255.0
                images_dict[rk].append(face_img)

        result = {k: torch.stack(v) for k, v in images_dict.items()}
        # For shape mode (single key 'normal'), return tensor directly for backward compat
        if len(render_keys) == 1:
            return result[render_keys[0]]
        return result

    def _make_label_strip(self, labels: list, tile_size: int, label_height: int = 24) -> torch.Tensor:
        """Create a white strip with text labels as a tensor [3, label_height, len(labels)*tile_size]."""
        from PIL import ImageDraw, ImageFont
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
        strip_tensor = torch.tensor(np.array(strip)).permute(2, 0, 1).float() / 255.0
        return strip_tensor

    @torch.no_grad()
    def visualize_sample_interior(self, data: dict, tile_size: int = 256):
        """
        Visualize interior views from camera center position.

        Returns a composite per sample:
            Row 0: Face labels (white strip with face names)
            Row 1: 6 actual cubemap images (front, right, back, left, top, bottom)
            Row 2: 6 rendered mesh views from the same camera center and directions

        Returns [B, 3, label_h + 2*tile_size, 6*tile_size], or None if camera_center/cond missing.
        """
        import torch.nn.functional as F
        import utils3d.torch

        if not isinstance(data, dict) or 'camera_center' not in data or 'cond' not in data:
            return None

        camera_centers = data['camera_center']  # [B, 3]
        cubemap_images = data['cond']  # [B, 6, 3, H, W]

        tex_z, shape_z = self._extract_tex_and_shape(data)
        reps = self.decode_latent(tex_z.cuda(), shape_z=shape_z.cuda() if shape_z is not None else None)

        if len(reps) == 0:
            return None

        # Set up renderer based on mode
        if self.latent_type == 'texture':
            import os
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
            import cv2
            from ..renderers import PbrMeshRenderer, EnvMap
            renderer = PbrMeshRenderer()
            renderer.rendering_options.resolution = tile_size
            renderer.rendering_options.ssaa = 4
            renderer.rendering_options.near = 0.01
            renderer.rendering_options.far = 2.0
            renderer.rendering_options.peel_layers = 8
            envmap = EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread('assets/hdri/interior.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            ))
            render_keys = ['shaded', 'base_color']
        else:
            renderer = get_renderer(reps[0])
            renderer.rendering_options.resolution = tile_size
            renderer.rendering_options.ssaa = 4
            renderer.rendering_options.near = 0.01
            renderer.rendering_options.far = 2.0
            envmap = None
            render_keys = ['normal']

        # Face directions matching cubemap convention
        face_labels = ['front (+Y)', 'right (+X)', 'back (-Y)', 'left (-X)', 'top (+Z)', 'bottom (-Z)']
        face_directions = [
            [0.0, 1.0, 0.0],   # front
            [1.0, 0.0, 0.0],   # right
            [0.0, -1.0, 0.0],  # back
            [-1.0, 0.0, 0.0],  # left
            [0.0, 0.0, 1.0],   # top
            [0.0, 0.0, -1.0],  # bottom
        ]
        fov = torch.deg2rad(torch.tensor(120.0)).cuda()

        # Create label strip once
        label_h = max(20, tile_size // 10)
        label_strip = self._make_label_strip(face_labels, tile_size, label_h).cuda()

        composites_dict = {k: [] for k in render_keys}
        for i, representation in enumerate(reps):
            # Render 6 interior faces
            rendered_faces_dict = {k: [] for k in render_keys}
            cam = camera_centers[i].float().cuda()

            for face_idx, fd in enumerate(face_directions):
                look_at = cam + torch.tensor(fd, dtype=torch.float32).cuda()
                if face_idx == 4:  # top (+Z): up=-Y to match py360convert
                    up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32).cuda()
                elif face_idx == 5:  # bottom (-Z): up=+Y
                    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()
                else:  # horizontal faces: up=+Z
                    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda()
                ext = utils3d.torch.extrinsics_look_at(cam, look_at, up)
                intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
                if envmap is not None:
                    res = renderer.render(representation, ext, intr, envmap=envmap)
                else:
                    res = renderer.render(representation, ext, intr)
                for rk in render_keys:
                    rendered_faces_dict[rk].append(res[rk])

            # Resize cubemap images to tile_size
            cubemap_resized = F.interpolate(
                cubemap_images[i].cuda(),  # [6, 3, H, W]
                size=(tile_size, tile_size), mode='bilinear', align_corners=False,
            )  # [6, 3, tile_size, tile_size]

            # Row 0: labels, Row 1: cubemap images, Row 2: rendered interior
            row1 = torch.cat([cubemap_resized[j] for j in range(6)], dim=2)  # [3, tile_size, 6*tile_size]
            for rk in render_keys:
                row2 = torch.cat(rendered_faces_dict[rk], dim=2)  # [3, tile_size, 6*tile_size]
                composite = torch.cat([label_strip, row1, row2], dim=1)  # [3, label_h+2*tile_size, 6*tile_size]
                composites_dict[rk].append(composite)

        result = {k: torch.stack(v) for k, v in composites_dict.items()}
        # For shape mode (single key 'normal'), return tensor directly for backward compat
        if len(render_keys) == 1:
            return result[render_keys[0]]
        return result


    @torch.no_grad()
    def visualize_sample_parts_topdown(self, data: dict, tile_size: int = 256, min_voxels: int = 1):
        """
        Visualize each part (overall + individual assets) from top-down view.

        For each sample, produces one row:
            [overall | layout | asset_0 | asset_1 | ... ]
        with text labels on top. Parts too small to decode show as gray tiles.

        For shape mode: renders normal maps.
        For texture mode: renders PBR shaded images using dual decoders.

        Returns a list of per-sample composite images, since each sample
        may have a different number of parts. Caller is responsible for saving.
        """
        if not isinstance(data, dict) or 'part_layouts' not in data:
            return None

        parts_sparse, part_labels = self._extract_parts_separately(data)

        # For texture mode, also extract shape parts
        shape_parts_sparse = None
        if self.latent_type == 'texture' and 'concat_cond' in data:
            shape_data = {'x_0': data['concat_cond'], 'part_layouts': data.get('part_layouts'),
                          'asset_names': data.get('asset_names'), 'has_layout': data.get('has_layout'),
                          'sample_paths': data.get('sample_paths', ['']*len(data.get('part_layouts', []))),
                          'n_visible_assets': data.get('n_visible_assets', [0]*len(data.get('part_layouts', [])))}
            shape_parts_sparse, _ = self._extract_parts_separately(shape_data)

        # Decode each part individually — aligned: len(reps) == parts_sparse.shape[0]
        reps = self.decode_latent_parts(
            parts_sparse.cuda(),
            shape_z=shape_parts_sparse.cuda() if shape_parts_sparse is not None else None,
            min_voxels=min_voxels,
        )

        # Find the first successfully decoded part to initialize renderer
        first_valid = next((r for r in reps if r is not None), None)
        if first_valid is None:
            return None

        # Top-down camera
        yaw = [0]
        pitch = [90 / 180 * np.pi]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        # Check if we need PBR rendering (texture mode with MeshWithVoxel)
        from ..representations import MeshWithVoxel
        use_pbr = isinstance(first_valid, MeshWithVoxel)

        if use_pbr:
            import os
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
            import cv2
            from ..renderers import PbrMeshRenderer, EnvMap
            renderer = PbrMeshRenderer()
            renderer.rendering_options.resolution = tile_size
            renderer.rendering_options.near = 1
            renderer.rendering_options.far = 100
            renderer.rendering_options.ssaa = 2
            renderer.rendering_options.peel_layers = 8
            envmap = EnvMap(torch.tensor(
                cv2.cvtColor(cv2.imread('assets/hdri/interior.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            ))
        else:
            renderer = get_renderer(first_valid)
            renderer.rendering_options.resolution = tile_size

        # Render all parts (gray placeholder for None)
        gray_tile = torch.full((3, tile_size, tile_size), 0.3, device='cuda')
        rendered = []
        for rep in reps:
            if rep is not None:
                try:
                    if use_pbr:
                        res = renderer.render(rep, exts[0], ints[0], envmap=envmap)
                        rendered.append(res['shaded'])
                    else:
                        res = renderer.render(rep, exts[0], ints[0])
                        rendered.append(res['normal'])
                except Exception:
                    rendered.append(gray_tile)
            else:
                rendered.append(gray_tile)

        # Build per-sample composites
        label_h = max(20, tile_size // 10)
        composites = []
        rep_idx = 0
        for sample_idx, sample_labels in enumerate(part_labels):
            n_parts = len(sample_labels)
            sample_rendered = rendered[rep_idx:rep_idx + n_parts]
            rep_idx += n_parts

            # Create label strip for this sample
            label_strip = self._make_label_strip(sample_labels, tile_size, label_h).cuda()

            # Concat rendered tiles horizontally
            row = torch.cat(sample_rendered, dim=2)  # [3, tile_size, n_parts*tile_size]
            composite = torch.cat([label_strip, row], dim=1)  # [3, label_h+tile_size, n_parts*tile_size]
            composites.append(composite)

        return composites

    @torch.no_grad()
    def visualize_sample_parts_topdown_camera_center(
        self,
        composites: list,
        camera_centers: torch.Tensor,
        tile_size: int = 256,
    ):
        """
        Overlay camera center dots on already-rendered parts_topdown composites.

        Takes the composites from visualize_sample_parts_topdown() and draws
        a cyan circle on the overall tile (first tile) at the projected camera position.

        Args:
            composites: list of per-sample composite tensors from visualize_sample_parts_topdown
            camera_centers: [B, 3] camera centers in normalized coordinates
            tile_size: must match the tile_size used in visualize_sample_parts_topdown

        Returns:
            list of per-sample composite tensors with camera center overlay
        """
        from PIL import ImageDraw

        if composites is None or camera_centers is None:
            return None

        # Same top-down camera as visualize_sample_parts_topdown
        yaw = [0]
        pitch = [90 / 180 * np.pi]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        label_h = max(20, tile_size // 10)

        result = []
        for i, comp in enumerate(composites):
            if i >= camera_centers.shape[0]:
                result.append(comp)
                continue

            cam_3d = camera_centers[i].float().cuda()  # [3]
            point_h = torch.cat([cam_3d, torch.ones(1, device='cuda')])  # [4]
            point_cam = exts[0] @ point_h  # [4]
            point_proj = ints[0] @ point_cam[:3]  # [3]

            # Convert composite to PIL, draw, convert back
            comp_np = (comp.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(comp_np)

            if point_proj[2].abs() > 1e-6:
                u = (point_proj[0] / point_proj[2]).item()
                v = (point_proj[1] / point_proj[2]).item()
                # Map to the overall tile (first tile), offset by label_h
                px = u * tile_size
                py = v * tile_size + label_h

                if -20 < px < tile_size + 20 and label_h - 20 < py < label_h + tile_size + 20:
                    draw = ImageDraw.Draw(pil_img)
                    radius = 8
                    draw.ellipse(
                        [px - radius, py - radius, px + radius, py + radius],
                        fill=(0, 255, 255),
                        outline=(255, 255, 255),
                        width=2,
                    )

            result_tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            result.append(result_tensor.to(comp.device))

        return result

    @torch.no_grad()
    def visualize_bbox_projection(
        self,
        data: dict,
        image_size: int = 512,
        fov_degrees: float = 120.0,
        patch_size: int = 16,
    ) -> Optional[List[Tuple[str, Image.Image]]]:
        """
        Visualize 3D bounding box projections on cubemap images.

        For each sample, produces a 2x3 matplotlib figure showing all 6 cubemap faces
        with projected OBB bounding boxes drawn on them.

        Args:
            data: dict with 'cond', 'camera_center', 'obbs', 'asset_names'
            image_size: cubemap image size
            fov_degrees: cubemap FOV
            patch_size: DINO patch size

        Returns:
            List of (sample_name, PIL.Image) tuples, or None if required data missing.
        """
        if not isinstance(data, dict):
            return None
        for key in ['cond', 'camera_center', 'obbs', 'asset_names']:
            if key not in data:
                return None

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from data_toolkit.erp.visualize_bbox_cubemap_projection import (
                project_bbox_to_cubemap,
                get_display_name,
            )
        except ImportError:
            return None

        cubemap_images_batch = data['cond']  # [B, 6, 3, H, W]
        camera_centers = data['camera_center']  # [B, 3]
        obbs_list = data['obbs']  # List of [N_i, 7] per sample
        asset_names_list = data['asset_names']  # List of List[str] per sample
        sample_paths = data.get('sample_paths', None)

        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        asset_colors = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
            (1.0, 0.5, 0.0), (0.5, 0.0, 1.0), (0.0, 0.5, 1.0), (1.0, 0.0, 0.5),
        ]

        B = cubemap_images_batch.shape[0]
        results = []

        for b in range(B):
            cam_center = camera_centers[b].cpu().numpy()
            obbs = obbs_list[b]
            if isinstance(obbs, torch.Tensor):
                obbs = obbs.cpu().numpy()
            asset_names = asset_names_list[b]
            cubemap_tensors = cubemap_images_batch[b]  # [6, 3, H, W]

            # Convert cubemap tensors to PIL images
            cubemap_pil = []
            for f_idx in range(6):
                img_np = (cubemap_tensors[f_idx].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cubemap_pil.append(img_np)

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            face_axes = {}
            for f_idx, fname in enumerate(face_names):
                row, col = f_idx // 3, f_idx % 3
                face_axes[fname] = axes[row, col]

            # Draw cubemap images
            for f_idx, fname in enumerate(face_names):
                ax = face_axes[fname]
                ax.imshow(cubemap_pil[f_idx])
                ax.set_title(f'{fname.upper()}', fontsize=12, fontweight='bold')
                ax.set_xlim(0, image_size)
                ax.set_ylim(image_size, 0)

            # Project and draw bboxes
            for i, (obb, name) in enumerate(zip(obbs, asset_names)):
                color = asset_colors[i % len(asset_colors)]
                projections = project_bbox_to_cubemap(obb, cam_center, fov_degrees, image_size)

                for face_name, proj in projections.items():
                    ax = face_axes[face_name]
                    bbox = proj['bbox']
                    u_min, v_min, u_max, v_max = bbox

                    u_min_c = max(0, u_min)
                    u_max_c = min(image_size, u_max)
                    v_min_c = max(0, v_min)
                    v_max_c = min(image_size, v_max)

                    width = u_max_c - u_min_c
                    height = v_max_c - v_min_c

                    if width > 0 and height > 0:
                        lw = 3 if proj['in_fov'] else 1
                        ls = '-' if proj['in_fov'] else '--'
                        rect = mpatches.Rectangle(
                            (u_min_c, v_min_c), width, height,
                            linewidth=lw, linestyle=ls,
                            edgecolor=color, facecolor=(*color, 0.1),
                        )
                        ax.add_patch(rect)
                        label = get_display_name(name)
                        vis_pct = proj.get('visibility_pct', 100.0)
                        ax.text(u_min_c + 2, v_min_c + 12, f'{label} {vis_pct:.0f}%',
                                fontsize=7, color='white',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))

            fig.suptitle(
                f'3D Bbox Projection | cam=({cam_center[0]:.3f}, {cam_center[1]:.3f}, {cam_center[2]:.3f})',
                fontsize=14,
            )
            plt.tight_layout()

            # Convert to PIL
            from io import BytesIO
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            pil_img = Image.open(buf).convert('RGB')
            plt.close(fig)

            path_name = sample_paths[b] if sample_paths and b < len(sample_paths) else f'sample_{b}'
            if isinstance(path_name, str):
                path_name = path_name.replace('/', '_')
            results.append((path_name, pil_img))

        return results

    @torch.no_grad()
    def visualize_cross_attn_mask(
        self,
        data: dict,
        image_size: int = 512,
        fov_degrees: float = 120.0,
        patch_size: int = 16,
        tokens_per_face: int = 1029,
        expand_pixels: int = 28,
    ) -> Optional[List[Tuple[str, Image.Image]]]:
        """
        Visualize cross-attention token selection per asset per cubemap face.

        Uses create_per_part_cross_attn_masks() from asset_attention_mask.py —
        the same function used during training — to ensure visualization matches
        the actual masks applied in the forward pass.

        For each sample, produces a grid where:
        - Row 0: "overall" -> all tokens active (uniform color)
        - Row 1..N: each asset -> only bbox-projected tokens active
        - Columns: 6 cubemap faces

        All active tokens shown in a single uniform color.

        Args:
            data: dict with 'camera_center', 'obbs', 'asset_names'
            image_size: cubemap image size
            fov_degrees: cubemap FOV
            patch_size: DINO patch size
            tokens_per_face: DINO tokens per face
            expand_pixels: expand projected bbox by this many pixels

        Returns:
            List of (sample_name, PIL.Image) tuples, or None if data missing.
        """
        if not isinstance(data, dict):
            return None
        for key in ['camera_center', 'obbs', 'asset_names']:
            if key not in data:
                return None

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        except ImportError:
            return None

        camera_centers = data['camera_center']  # [B, 3]
        obbs_list = data['obbs']  # List of [N_i, 7] per sample
        asset_names_list = data['asset_names']  # List of List[str] per sample
        sample_paths = data.get('sample_paths', None)

        face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
        tokens_per_row = int(np.ceil(image_size / patch_size))

        # Uniform color: inactive=light gray, active=blue
        cmap = ListedColormap(['#f0f0f0', '#2196F3'])

        B = camera_centers.shape[0]
        results = []

        for b in range(B):
            cam_center = camera_centers[b].float()  # [3]
            obbs = obbs_list[b]
            if not isinstance(obbs, torch.Tensor):
                obbs = torch.tensor(obbs, dtype=torch.float32)
            asset_names = asset_names_list[b]
            num_parts = 1 + len(asset_names)  # overall + assets

            # Use the ACTUAL training function to get masks
            masks = create_per_part_cross_attn_masks(
                obbs=obbs,
                camera_center=cam_center,
                num_parts=num_parts,
                tokens_per_face=tokens_per_face,
                fov_degrees=fov_degrees,
                image_size=image_size,
                patch_size=patch_size,
                expand_pixels=expand_pixels,
            )
            # masks: List of [total_tokens] boolean tensors
            # masks[0] = overall (all True), masks[1..N] = per-asset

            # Build row labels
            row_labels = ['overall']
            for name in asset_names:
                # Shorten asset name for display
                short = name.split('/')[-1].replace('.npz', '').replace('.glb', '')
                if len(short) > 25:
                    short = short[:22] + '...'
                row_labels.append(short)

            n_rows = len(masks)
            fig, axes = plt.subplots(n_rows, 6, figsize=(18, 3 * n_rows + 1))
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for row_idx, mask in enumerate(masks):
                # mask: [6 * tokens_per_face] boolean
                for face_idx, fname in enumerate(face_names):
                    ax = axes[row_idx, face_idx]

                    # Extract per-face mask
                    face_start = face_idx * tokens_per_face
                    face_end = (face_idx + 1) * tokens_per_face
                    face_mask = mask[face_start:face_end]  # [tokens_per_face]

                    # Convert flat mask to 2D grid
                    # DINOv3 token layout: [CLS, reg0, reg1, reg2, reg3, patch_0_0, ...]
                    # Token 0 = CLS, tokens 1-4 = registers, tokens 5..1028 = patch tokens
                    num_special_tokens = 5  # 1 CLS + 4 registers
                    grid = np.zeros((tokens_per_row, tokens_per_row))
                    for t_v in range(tokens_per_row):
                        for t_u in range(tokens_per_row):
                            token_idx = t_v * tokens_per_row + t_u + num_special_tokens
                            if token_idx < tokens_per_face and face_mask[token_idx].item():
                                grid[t_v, t_u] = 1.0

                    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
                    if face_idx == 0:
                        ax.set_ylabel(row_labels[row_idx], fontsize=7)
                    if row_idx == 0:
                        ax.set_title(fname.upper(), fontsize=10, pad=5)
                    ax.set_xticks([])
                    ax.set_yticks([])

            cam_np = cam_center.cpu().numpy()
            fig.suptitle(
                f'Cross-Attn Token Selection (actual training masks)\n'
                f'cam=({cam_np[0]:.3f}, {cam_np[1]:.3f}, {cam_np[2]:.3f}) | '
                f'{tokens_per_row}x{tokens_per_row} tokens, {tokens_per_face}/face',
                fontsize=14, y=0.98,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            from io import BytesIO
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            pil_img = Image.open(buf).convert('RGB')
            plt.close(fig)

            path_name = sample_paths[b] if sample_paths and b < len(sample_paths) else f'sample_{b}'
            if isinstance(path_name, str):
                path_name = path_name.replace('/', '_')
            results.append((path_name, pil_img))

        return results


class ERPStructuredLatentShape(ERPSLatVisMixin, ERPCubemapConditionedMixin, ERPStructuredLatentBase):
    """
    ERP structured latent dataset for shape generation.
    """

    def __init__(
        self,
        root: str,
        *,
        latent_encoder: str = 'shape_enc_next_dc_f16c32_fp16_512',
        pretrained_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        resolution: int = 512,
        **kwargs
    ):
        super().__init__(
            root,
            latent_type='shape',
            latent_encoder=latent_encoder,
            pretrained_slat_dec=pretrained_slat_dec,
            resolution=resolution,
            **kwargs
        )


class ERPStructuredLatentTexture(ERPSLatVisMixin, ERPCubemapConditionedMixin, ERPStructuredLatentBase):
    """
    ERP structured latent dataset for texture generation.

    Uses dual decoders for visualization:
    - shape_slat_dec: decodes shape latent → mesh + guide_subs
    - pbr_slat_dec: decodes texture latent with guide_subs → PBR voxel attributes
    """

    def __init__(
        self,
        root: str,
        *,
        latent_encoder: str = 'tex_enc_next_dc_f16c32_fp16_512',
        pretrained_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        resolution: int = 512,
        shape_latent_encoder: str = 'shape_enc_next_dc_f16c32_fp16_512',
        shape_normalization: Optional[dict] = None,
        pretrained_shape_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        pretrained_pbr_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        attrs: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            root,
            latent_type='texture',
            latent_encoder=latent_encoder,
            pretrained_slat_dec=pretrained_slat_dec,
            resolution=resolution,
            shape_latent_encoder=shape_latent_encoder,
            shape_normalization=shape_normalization,
            pretrained_shape_slat_dec=pretrained_shape_slat_dec,
            pretrained_pbr_slat_dec=pretrained_pbr_slat_dec,
            attrs=attrs or ['base_color', 'metallic', 'roughness', 'alpha'],
            **kwargs
        )


# Aliases for consistency
ERPImageConditionedSLatShape = ERPStructuredLatentShape
ERPImageConditionedSLatTexture = ERPStructuredLatentTexture
