# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

## ERP Asset-Aware Conditioned Mixin for Structured Latent Flow (TRELLIS 2)
## This file implements the ERP cubemap conditioning with asset-aware attention
## for shape/texture generation models (SLat Flow)
##
## Key features:
## 1. ERP image encoding with DINOv3 (same as sparse structure)
## 2. OmniPart-style part layouts support
## 3. Grouped self-attention based on 3D bbox overlap
## 4. Per-part cross-attention with cubemap projection masks

from typing import *
import copy
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import utils3d.torch

from .... import models
from ....modules import sparse as sp
from ....modules.sparse.basic import SparseTensor, sparse_cat
from ....utils import dist_utils
from ....utils.general_utils import dict_reduce
from ....utils.data_utils import recursive_to_device
from ....utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics
from ....utils.asset_attention_mask import (
    compute_overlap_groups,
    create_per_part_cross_attn_masks,
    filter_visible_assets,
)
from .erp_image_conditioned import ERPImageEncoder


class ERPSLatConditionedMixin:
    """
    Mixin for ERP cubemap-conditioned structured latent flow models.

    This mixin provides:
    1. ERP image encoding with DINOv3 + view position embeddings
    2. Support for OmniPart-style part_layouts
    3. Asset-aware cross-attention with cubemap projection masks

    Data format:
    - x_0: SparseTensor with shape latent features
    - cond: [B, 6, 3, 512, 512] cubemap images
    - part_layouts: List of part slices per batch
    - obbs: List of [num_assets, 7] oriented bounding boxes per batch
    - camera_center: [B, 3] normalized camera center

    Args:
        image_cond_model: dict with 'name' and 'args' for feature extractor
        use_asset_aware_attention: Whether to use asset-aware cross-attention masks
        tokens_per_face: Number of DINO tokens per cubemap face (default: 1029 for DINOv3)
        fov_degrees: Cubemap FOV (default: 120)
        expand_pixels: Expand projected bbox by this many pixels
    """
    def __init__(
        self,
        *args,
        image_cond_model: dict,
        use_asset_aware_attention: bool = True,
        tokens_per_face: int = 1029,
        fov_degrees: float = 120.0,
        expand_pixels: int = 28,
        overlap_margin: float = 0.02,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_cond_model_config = image_cond_model
        self.use_asset_aware_attention = use_asset_aware_attention
        self.tokens_per_face = tokens_per_face
        self.fov_degrees = fov_degrees
        self.expand_pixels = expand_pixels
        self.overlap_margin = overlap_margin
        self.erp_encoder = None

        # Set voxel_resolution from denoiser model's resolution (e.g. 32 for dual_grid_512)
        if hasattr(self, 'models') and 'denoiser' in self.models:
            self.voxel_resolution = getattr(self.models['denoiser'], 'resolution', 32)
        else:
            self.voxel_resolution = 32

    def _init_erp_encoder(self):
        """Initialize ERP encoder lazily"""
        if self.erp_encoder is not None:
            return

        self.erp_encoder = ERPImageEncoder(
            image_cond_model=self.image_cond_model_config,
            feature_dim=1024
        ).cuda()

    @torch.no_grad()
    def encode_erp_image(self, cubemap: torch.Tensor) -> torch.Tensor:
        """
        Encode ERP cubemap images.

        Args:
            cubemap: [B, 6, 3, H, W] tensor of 6 cubemap faces

        Returns:
            features: [B, 6*N, 1024] encoded features
        """
        self._init_erp_encoder()
        return self.erp_encoder(cubemap)

    def compute_overlap_groups_for_batch(
        self,
        obbs_list: List[torch.Tensor]
    ) -> List[List[List[int]]]:
        """
        Compute overlap groups for each batch sample.

        Args:
            obbs_list: List of [num_assets, 7] OBB tensors per batch

        Returns:
            List of overlap group lists per batch
        """
        batch_overlap_groups = []
        for obbs in obbs_list:
            if obbs is not None and obbs.shape[0] > 0:
                groups = compute_overlap_groups(obbs, margin=self.overlap_margin)
            else:
                groups = []
            batch_overlap_groups.append(groups)
        return batch_overlap_groups

    def compute_cross_attn_masks_for_batch(
        self,
        obbs_list: List[torch.Tensor],
        camera_centers: torch.Tensor,
        part_layouts: List[List[slice]],
        x_0_coords: torch.Tensor = None,
        has_layout_list: List[bool] = None,
    ) -> List[List[torch.Tensor]]:
        """
        Compute per-part cross-attention masks for each batch sample.

        Args:
            obbs_list: List of [num_assets, 7] OBB tensors per batch
            camera_centers: [B, 3] camera centers
            part_layouts: List of part slices per batch
            x_0_coords: [total_voxels, 4] SparseTensor coords (batch_idx, x, y, z).
                If provided, overall/layout masks use per-voxel spatial attention.
            has_layout_list: List of booleans indicating layout presence per batch

        Returns:
            List of per-part mask lists per batch
        """
        batch_masks = []
        for batch_idx, obbs in enumerate(obbs_list):
            num_parts = len(part_layouts[batch_idx])
            camera_center = camera_centers[batch_idx]
            has_layout = has_layout_list[batch_idx] if has_layout_list is not None else False

            # Extract overall voxel coords from SparseTensor
            overall_voxel_coords = None
            layout_voxel_coords = None
            if x_0_coords is not None:
                overall_slice = part_layouts[batch_idx][0]
                # SparseTensor coords: [N, 4] = (batch_idx, x, y, z)
                overall_voxel_coords = x_0_coords[overall_slice, 1:4]  # [N_overall, 3]

                # Extract layout voxel coords if layout is present
                if has_layout and len(part_layouts[batch_idx]) > 1:
                    layout_slice = part_layouts[batch_idx][1]
                    layout_voxel_coords = x_0_coords[layout_slice, 1:4]  # [N_layout, 3]

            # Overall mask + (layout mask) + asset masks
            masks = create_per_part_cross_attn_masks(
                obbs=obbs if obbs is not None and obbs.shape[0] > 0 else None,
                camera_center=camera_center,
                num_parts=num_parts,
                tokens_per_face=self.tokens_per_face,
                fov_degrees=self.fov_degrees,
                expand_pixels=self.expand_pixels,
                overall_voxel_coords=overall_voxel_coords,
                layout_voxel_coords=layout_voxel_coords,
                has_layout=has_layout,
                voxel_resolution=self.voxel_resolution,
            )

            batch_masks.append(masks)
        return batch_masks

    def run_snapshot(self, num_samples, batch_size, verbose=False):
        """
        Override run_snapshot to extract metadata for cross-attn mask visualization.

        The parent SparseFlowMatchingTrainer.run_snapshot() stores the full data dict
        inside sample_gt/sample values, but doesn't add _train_* metadata entries.
        We extract camera_center, obbs, asset_names, paths for visualization.
        """
        sample_dict = super().run_snapshot(num_samples, batch_size, verbose)

        # Extract metadata from train sample_gt
        sample_gt_value = sample_dict.get('train_sample_gt', {}).get('value', {})
        if isinstance(sample_gt_value, dict):
            if 'camera_center' in sample_gt_value:
                cc = sample_gt_value['camera_center']
                if isinstance(cc, torch.Tensor):
                    cc = cc.cpu()
                sample_dict['_train_camera_center'] = {'value': cc, 'type': 'metadata'}
            if 'obbs' in sample_gt_value:
                sample_dict['_train_obbs'] = {'value': sample_gt_value['obbs'], 'type': 'metadata'}
            if 'asset_names' in sample_gt_value:
                sample_dict['_train_asset_names'] = {'value': sample_gt_value['asset_names'], 'type': 'metadata'}
            if 'sample_paths' in sample_gt_value:
                sample_dict['_train_paths'] = {'value': sample_gt_value['sample_paths'], 'type': 'paths'}
            if 'part_layouts' in sample_gt_value:
                sample_dict['_train_part_layouts'] = {'value': sample_gt_value['part_layouts'], 'type': 'metadata'}
            if 'has_layout' in sample_gt_value:
                sample_dict['_train_has_layout'] = {'value': sample_gt_value['has_layout'], 'type': 'metadata'}

        # Extract metadata from eval sample_gt (if eval dataset was processed)
        eval_gt_value = sample_dict.get('eval_sample_gt', {}).get('value', {})
        if isinstance(eval_gt_value, dict):
            if 'camera_center' in eval_gt_value:
                cc = eval_gt_value['camera_center']
                if isinstance(cc, torch.Tensor):
                    cc = cc.cpu()
                sample_dict['_eval_camera_center'] = {'value': cc, 'type': 'metadata'}
            if 'obbs' in eval_gt_value:
                sample_dict['_eval_obbs'] = {'value': eval_gt_value['obbs'], 'type': 'metadata'}
            if 'asset_names' in eval_gt_value:
                sample_dict['_eval_asset_names'] = {'value': eval_gt_value['asset_names'], 'type': 'metadata'}
            if 'sample_paths' in eval_gt_value:
                sample_dict['_eval_paths'] = {'value': eval_gt_value['sample_paths'], 'type': 'paths'}
            if 'part_layouts' in eval_gt_value:
                sample_dict['_eval_part_layouts'] = {'value': eval_gt_value['part_layouts'], 'type': 'metadata'}
            if 'has_layout' in eval_gt_value:
                sample_dict['_eval_has_layout'] = {'value': eval_gt_value['has_layout'], 'type': 'metadata'}

        return sample_dict

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.

        Args:
            cond: [B, 6, 3, H, W] cubemap images

        Returns:
            Encoded conditioning tensor [B, 6*N, 1024]
        """
        cond = self.encode_erp_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond

    def get_inference_cond(
        self,
        cond,
        part_layouts=None,
        obbs=None,
        camera_center=None,
        x_0_coords=None,
        has_layout=None,
        **kwargs
    ):
        """
        Get the conditioning data for inference.

        Args:
            cond: [B, 6, 3, H, W] cubemap images
            part_layouts: Optional list of part slices per batch
            obbs: Optional list of OBB tensors per batch
            camera_center: Optional [B, 3] camera centers
            x_0_coords: Optional [N, 4] SparseTensor coords for per-voxel spatial mask
            has_layout: Optional list of booleans indicating layout presence per batch
        """
        cond = self.encode_erp_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)

        # Add part-aware kwargs if provided
        if part_layouts is not None:
            kwargs['part_layouts'] = part_layouts

            if self.use_asset_aware_attention and obbs is not None and camera_center is not None:
                # Compute overlap groups
                kwargs['overlap_groups'] = self.compute_overlap_groups_for_batch(obbs)
                # Compute cross-attention masks (with per-voxel spatial mask for overall/layout)
                kwargs['cross_attn_masks'] = self.compute_cross_attn_masks_for_batch(
                    obbs, camera_center.cuda(), part_layouts,
                    x_0_coords=x_0_coords,
                    has_layout_list=has_layout,
                )

        return super().get_inference_cond(cond, **kwargs)

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        part_layouts=None,
        obbs=None,
        camera_center=None,
        has_layout=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses with asset-aware attention support.

        Args:
            x_0: SparseTensor with features
            cond: [B, 6, 3, H, W] cubemap images
            part_layouts: List of part slices per batch
            obbs: List of OBB tensors per batch
            camera_center: [B, 3] camera centers
            has_layout: List of booleans indicating layout presence per batch
        """
        # Add part-aware kwargs if provided
        if part_layouts is not None:
            kwargs['part_layouts'] = part_layouts

            # Pass has_layout through to the model's forward()
            if has_layout is not None:
                kwargs['has_layout'] = has_layout

            if self.use_asset_aware_attention and obbs is not None and camera_center is not None:
                device = x_0.device
                camera_center = camera_center.to(device)

                # Compute overlap groups
                kwargs['overlap_groups'] = self.compute_overlap_groups_for_batch(obbs)

                # Compute cross-attention masks (with per-voxel spatial mask for overall/layout)
                kwargs['cross_attn_masks'] = self.compute_cross_attn_masks_for_batch(
                    [ob.to(device) if ob is not None else None for ob in obbs],
                    camera_center,
                    part_layouts,
                    x_0_coords=x_0.coords,
                    has_layout_list=has_layout,
                )

        return super().training_losses(x_0=x_0, cond=cond, **kwargs)

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        Returns first 4 cubemap faces arranged in 2x2 grid for visualization.
        """
        B = cond.shape[0]

        grid_images = []
        for b in range(B):
            top = torch.cat([cond[b, 0], cond[b, 1]], dim=2)
            bottom = torch.cat([cond[b, 2], cond[b, 3]], dim=2)
            grid = torch.cat([top, bottom], dim=1)
            grid_images.append(grid)

        grid_images = torch.stack(grid_images, dim=0)

        return {'cubemap': {'value': grid_images, 'type': 'image'}}

    # ===================== Visualization Methods =====================
    # NOTE: Commented out - visualization is handled by Dataset (trellis2/datasets/erp_sparse_structure_latent.py.py)
    # The BasicTrainer.visualize_sample() delegates to self.dataset.visualize_sample()
    # Keeping these here would override that delegation due to MRO.
    #
    # def _loading_slat_dec(self):
    #     """Load the structured latent decoder model if not already loaded."""
    #     ...
    #
    # def _delete_slat_dec(self):
    #     """Delete the decoder model to free up memory."""
    #     ...
    #
    # def _remove_noise_voxels(self, z_batch: SparseTensor) -> SparseTensor:
    #     """Remove noise voxels from latent vectors."""
    #     ...
    #
    # def decode_latent(self, z: SparseTensor, batch_size: int = 4):
    #     """Decode latent vectors into 3D representations."""
    #     ...
    #
    # def visualize_sample(self, x_0):
    #     """Generate multi-view renderings of a 3D representation."""
    #     ...


class ERPSLatSimpleMixin:
    """
    Simple mixin for ERP cubemap-conditioned SLat flow models without asset-awareness.

    This is for standard TRELLIS-style generation (no part layouts).
    Uses same ERP encoding as ERPSLatConditionedMixin but without part-aware attention.

    Args:
        image_cond_model: dict with 'name' and 'args' for feature extractor
    """
    def __init__(
        self,
        *args,
        image_cond_model: dict,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.image_cond_model_config = image_cond_model
        self.erp_encoder = None

    def _init_erp_encoder(self):
        """Initialize ERP encoder lazily"""
        if self.erp_encoder is not None:
            return

        self.erp_encoder = ERPImageEncoder(
            image_cond_model=self.image_cond_model_config,
            feature_dim=1024
        ).cuda()

    @torch.no_grad()
    def encode_erp_image(self, cubemap: torch.Tensor) -> torch.Tensor:
        """Encode ERP cubemap images."""
        self._init_erp_encoder()
        return self.erp_encoder(cubemap)

    def get_cond(self, cond, **kwargs):
        """Get the conditioning data."""
        cond = self.encode_erp_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond

    def get_inference_cond(self, cond, **kwargs):
        """Get the conditioning data for inference."""
        cond = self.encode_erp_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        return super().get_inference_cond(cond, **kwargs)

    def vis_cond(self, cond, **kwargs):
        """Visualize the conditioning data."""
        B = cond.shape[0]
        grid_images = []
        for b in range(B):
            top = torch.cat([cond[b, 0], cond[b, 1]], dim=2)
            bottom = torch.cat([cond[b, 2], cond[b, 3]], dim=2)
            grid = torch.cat([top, bottom], dim=1)
            grid_images.append(grid)
        grid_images = torch.stack(grid_images, dim=0)
        return {'cubemap': {'value': grid_images, 'type': 'image'}}

    # ===================== Visualization Methods =====================
    # NOTE: Commented out - visualization is handled by Dataset (erp_structured_latent.py)
    # The BasicTrainer.visualize_sample() delegates to self.dataset.visualize_sample()
    # Keeping these here would override that delegation due to MRO.
    #
    # def _loading_slat_dec(self):
    #     ...
    #
    # def _delete_slat_dec(self):
    #     ...
    #
    # def decode_latent(self, z: SparseTensor, batch_size: int = 4):
    #     ...
    #
    # def visualize_sample(self, x_0):
    #     ...
