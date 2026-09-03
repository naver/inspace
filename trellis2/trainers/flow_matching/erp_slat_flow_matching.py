# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

## ERP Structured Latent Flow Trainers for TRELLIS 2
## This file defines trainer classes for ERP-to-3D shape/texture generation
##
## Trainers:
## - ERPSLatFlowMatchingCFGTrainer: Simple ERP conditioning (no part awareness)
## - ERPAssetAwareSLatFlowMatchingCFGTrainer: Asset-aware with OmniPart-style generation
##
## These trainers work with the structured latent flow model (SLatFlowModel/ElasticSLatFlowModel)
## which generates shape or texture features on the sparse voxel grid.

from typing import *

from .sparse_flow_matching import SparseFlowMatchingCFGTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.erp_slat_conditioned import ERPSLatConditionedMixin, ERPSLatSimpleMixin


class ERPSLatFlowMatchingCFGTrainer(ERPSLatSimpleMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for ERP cubemap-conditioned structured latent flow model.

    This trainer uses 6 cubemap faces from ERP images as conditioning for
    shape/texture generation on sparse voxel grids.

    Training mode: Standard TRELLIS (no part layouts)

    Data format:
    - x_0: SparseTensor with shape/texture latent features
    - cond: [B, 6, 3, 512, 512] 6 cubemap faces

    Args:
        models (dict[str, nn.Module]): Models to train (denoiser)
        dataset: SLat dataset with cubemap conditioning
        output_dir (str): Output directory
        max_steps (int): Max training steps
        batch_size_per_gpu (int): Batch size per GPU
        optimizer (dict): Optimizer config
        ema_rate (float or list): EMA rates
        fp16_mode (str): FP16 mode ('amp' recommended)
        grad_clip (dict): Gradient clipping config
        i_log (int): Logging interval
        i_sample (int): Sampling interval
        i_save (int): Save interval
        p_uncond (float): Probability of dropping condition (for CFG)
        t_schedule (dict): Timestep schedule
        sigma_min (float): Minimum noise level
        image_cond_model (dict): DINOv3 feature extractor config
    """
    pass


class ERPAssetAwareSLatFlowMatchingCFGTrainer(ERPSLatConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for ERP cubemap-conditioned structured latent flow model with asset-awareness.

    This trainer implements OmniPart-style generation with:
    1. Part layouts: Overall scene + individual assets concatenated in latent space
    2. Grouped self-attention: Assets with overlapping 3D bboxes attend to each other
    3. Per-part cross-attention: Assets only attend to cubemap tokens where they project

    Training mode: OmniPart-style asset-aware generation

    Data format:
    - x_0: SparseTensor with concatenated [overall, asset0, asset1, ...] latents
    - cond: [B, 6, 3, 512, 512] 6 cubemap faces
    - part_layouts: List of part slices per batch [[overall_slice, asset0_slice, ...], ...]
    - obbs: List of [num_assets, 7] oriented bounding boxes per batch
    - camera_center: [B, 3] normalized camera center in O-Voxel space

    Self-attention behavior:
    - Overall scene: self-attends only to itself
    - Assets in same overlap group: attend to each other
    - Assets not overlapping: self-attend only

    Cross-attention behavior:
    - Overall scene: attends to ALL cubemap tokens
    - Each asset: attends only to tokens where its 3D bbox projects

    Args:
        models (dict[str, nn.Module]): Models to train (denoiser)
        dataset: SLat dataset with asset-aware data loading
        output_dir (str): Output directory
        max_steps (int): Max training steps
        batch_size_per_gpu (int): Batch size per GPU
        optimizer (dict): Optimizer config
        ema_rate (float or list): EMA rates
        fp16_mode (str): FP16 mode
        grad_clip (dict): Gradient clipping config
        i_log (int): Logging interval
        i_sample (int): Sampling interval
        i_save (int): Save interval
        p_uncond (float): Probability of dropping condition
        t_schedule (dict): Timestep schedule
        sigma_min (float): Minimum noise level
        image_cond_model (dict): DINOv3 feature extractor config
        use_asset_aware_attention (bool): Whether to use asset-aware attention
        tokens_per_face (int): DINO tokens per face (1029 for DINOv3)
        fov_degrees (float): Cubemap FOV (120)
        expand_pixels (int): Expand projected bbox by this many pixels
        overlap_margin (float): Margin for OBB overlap detection
    """
    pass
