# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

## ERP Flow Matching Trainers for TRELLIS 2
## This file defines trainer classes for ERP-to-3D scene generation
##
## Trainers:
## - ERPImageConditionedFlowMatchingCFGTrainer: Cubemap conditioning with random Gaussian noise
## - ERPInitialVoxelFlowMatchingCFGTrainer: Cubemap conditioning with initial voxel latent support
## - ERPSpatialAttentionFlowMatchingCFGTrainer: With spatial attention mask support

from typing import *

from .flow_matching import FlowMatchingCFGTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.erp_image_conditioned import (
    ERPImageConditionedMixin,
    ERPInitialVoxelMixin,
    SpatialAttentionMixin,
    create_spatial_attention_mask
)


class ERPImageConditionedFlowMatchingCFGTrainer(ERPImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for ERP cubemap-conditioned diffusion model with flow matching and CFG.

    This trainer uses 6 cubemap faces from ERP images as conditioning.
    The conditioning is encoded using DINOv3 with learnable view position embeddings.

    Training mode: Random Gaussian noise (standard flow matching)

    Data format:
    - x_0: [B, 8, 16, 16, 16] GT sparse structure latent
    - cond: [B, 6, 3, 512, 512] 6 cubemap faces

    Dataset args:
    - gt_latent_folder: 'voxels_ss_latent' (GT latents)
    - use_initial_voxel: False

    Args:
        models (dict[str, nn.Module]): Models to train (denoiser)
        dataset: ERP dataset (ERPCubemapConditionedSparseStructureLatent)
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


class ERPInitialVoxelFlowMatchingCFGTrainer(ERPInitialVoxelMixin, ERPImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for ERP cubemap-conditioned diffusion model with initial voxel latent support.

    This trainer supports two training modes:
    1. Random Gaussian noise (when initial_voxel_latent not provided)
    2. Initial voxel latent mode (when initial_voxel_latent provided):
       - Uses pre-encoded initial voxels (from dap_depth_voxels_ss_latent)
       - Learns to refine towards GT

    Data format:
    - x_0: [B, 8, 16, 16, 16] GT sparse structure latent
    - cond: [B, 6, 3, 512, 512] 6 cubemap faces
    - initial_voxel_latent: [B, 8, 16, 16, 16] pre-encoded initial voxel latent

    Dataset args:
    - gt_latent_folder: 'voxels_ss_latent' (GT latents)
    - use_initial_voxel: True
    - initial_voxel_latent_folder: 'dap_depth_voxels_ss_latent'

    Args:
        models (dict[str, nn.Module]): Models to train (denoiser)
        dataset: ERP dataset with initial voxel support
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
        use_initial_voxel_latent (bool): Whether to use initial voxel latent mode
        initial_voxel_t_range (tuple): Timestep range for initial voxel mode
    """
    pass


class ERPSpatialAttentionFlowMatchingCFGTrainer(
    SpatialAttentionMixin,
    ERPImageConditionedMixin,
    FlowMatchingCFGTrainer
):
    """
    Trainer for ERP cubemap-conditioned diffusion model with spatial attention mask.

    This trainer adds spatially-aware cross-attention where each cubemap face
    only attends to relevant voxel regions based on camera position and viewing direction.

    Features:
    1. Creates attention mask based on camera center position
    2. Each cubemap face attends only to voxels visible from that viewing direction
    3. Supports FOV 90 and 120 configurations

    Data format:
    - x_0: [B, 8, 16, 16, 16] GT sparse structure latent
    - cond: [B, 6, 3, 512, 512] 6 cubemap faces
    - camera_center: [B, 3] normalized camera center in voxel space [-1, 1]

    Args:
        models (dict[str, nn.Module]): Models to train (denoiser)
        dataset: ERP dataset with camera center support
        output_dir (str): Output directory
        ... (same as ERPImageConditionedFlowMatchingCFGTrainer)
        use_spatial_attention (bool): Whether to use spatial attention mask
        spatial_attention_fov (float): FOV degrees (90 or 120)
        spatial_attention_soft (bool): Use soft mask instead of hard mask
        spatial_attention_soft_margin (float): Margin for soft mask transition
        voxel_resolution (int): Voxel latent resolution
        tokens_per_face (int): Number of tokens per cubemap face
    """
    pass


class ERPSpatialAttentionInitialVoxelFlowMatchingCFGTrainer(
    SpatialAttentionMixin,
    ERPInitialVoxelMixin,
    ERPImageConditionedMixin,
    FlowMatchingCFGTrainer
):
    """
    Trainer combining spatial attention mask and initial voxel latent support.

    This trainer combines:
    1. Spatially-aware cross-attention (from SpatialAttentionMixin)
    2. Initial voxel latent mode (from ERPInitialVoxelMixin)

    Data format:
    - x_0: [B, 8, 16, 16, 16] GT sparse structure latent
    - cond: [B, 6, 3, 512, 512] 6 cubemap faces
    - camera_center: [B, 3] normalized camera center
    - initial_voxel_latent: [B, 8, 16, 16, 16] pre-encoded initial voxel latent
    """
    pass
