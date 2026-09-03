# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license
#
# Modified from TRELLIS (https://github.com/microsoft/TRELLIS)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

## ERP Image Conditioned Mixin for Flow Matching Trainer (TRELLIS 2)
## This file implements the ERP cubemap conditioning with DINOv3 encoding
## and optional initial voxel support for ERP-to-3D scene generation
##
## Adapted from TRELLIS 1 for TRELLIS 2 architecture (using DINOv3 instead of DINOv2)
##
## Features:
## 1. Encodes 6 cubemap faces with DINOv3 -> [B, 6*1029, 1024]
## 2. Adds learnable view position embeddings to distinguish cubemap faces
## 3. Supports two training modes:
##    - Random Gaussian noise (default)
##    - Initial voxel mode (starting from depth-lifted point cloud)
## 4. Optional spatial attention mask for FOV-aware cross-attention

from typing import *
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

from ....utils import dist_utils
from ....utils.general_utils import dict_reduce
from .image_conditioned import DinoV3FeatureExtractor, DinoV2FeatureExtractor


class ERPImageEncoder(nn.Module):
    """
    ERP Image Encoder that encodes 6 cubemap faces with DINOv3 and view position embeddings.

    Architecture:
    - Input: [B, 6, 3, 512, 512] (6 cubemap faces)
    - DINOv3 encoding: each face -> [B, 1029, 1024]
    - Add learnable view position embedding for each face
    - Concatenate: [B, 6174, 1024] (6 x 1029 = 6174)

    Face order: front, right, back, left, top, bottom
    """
    def __init__(
        self,
        image_cond_model: dict,
        feature_dim: int = 1024
    ):
        super().__init__()
        self.image_cond_model_config = image_cond_model
        self.feature_dim = feature_dim
        self.feature_extractor = None

        # Learnable view position embeddings: [6, 1024]
        # Order: front, right, back, left, top, bottom
        self.view_pos_emb = nn.Parameter(torch.randn(6, feature_dim) * 0.02)

    def _init_feature_extractor(self):
        """Initialize feature extractor lazily"""
        if self.feature_extractor is not None:
            return

        with dist_utils.local_master_first():
            model_name = self.image_cond_model_config['name']
            model_args = self.image_cond_model_config.get('args', {})

            if model_name == 'DinoV3FeatureExtractor':
                self.feature_extractor = DinoV3FeatureExtractor(**model_args)
            elif model_name == 'DinoV2FeatureExtractor':
                self.feature_extractor = DinoV2FeatureExtractor(**model_args)
            else:
                raise ValueError(f"Unknown feature extractor: {model_name}")

            self.feature_extractor.cuda()

    @torch.no_grad()
    def encode_single_face(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode a single cubemap face with DINOv3/DINOv2.

        Args:
            image: [B, 3, H, W] tensor

        Returns:
            features: [B, N, 1024] tensor (N=1029 for DINOv3, N=1374 for DINOv2)
        """
        features = self.feature_extractor(image)
        return features

    def forward(self, cubemap: torch.Tensor) -> torch.Tensor:
        """
        Encode 6 cubemap faces.

        Args:
            cubemap: [B, 6, 3, H, W] tensor of 6 cubemap faces

        Returns:
            features: [B, 6*N, 1024] concatenated features with view position embeddings
        """
        self._init_feature_extractor()

        B = cubemap.shape[0]
        device = cubemap.device

        features_list = []
        for i in range(6):
            face = cubemap[:, i]  # [B, 3, H, W]
            feat = self.encode_single_face(face)  # [B, N, 1024]

            # Add view position embedding (broadcast to all tokens)
            view_emb = self.view_pos_emb[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 1024]
            feat = feat + view_emb.to(device)  # [B, N, 1024]

            features_list.append(feat)

        # Concatenate all faces
        features = torch.cat(features_list, dim=1)  # [B, 6*N, 1024]

        return features


class ERPImageConditionedMixin:
    """
    Mixin for ERP cubemap-conditioned models in TRELLIS 2.

    This mixin provides:
    1. ERP image encoding with DINOv3 + view position embeddings
    2. Support for classifier-free guidance

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
        """
        Encode ERP cubemap images.

        Args:
            cubemap: [B, 6, 3, H, W] tensor of 6 cubemap faces

        Returns:
            features: [B, 6*N, 1024] encoded features
        """
        self._init_erp_encoder()
        return self.erp_encoder(cubemap)

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

    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_erp_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        Returns first 4 cubemap faces arranged in 2x2 grid for visualization.
        """
        # cond: [B, 6, 3, H, W]
        B = cond.shape[0]

        # Create 2x2 grid of first 4 faces (front, right, back, left)
        grid_images = []
        for b in range(B):
            # Stack front, right on top; back, left on bottom
            top = torch.cat([cond[b, 0], cond[b, 1]], dim=2)  # [3, H, 2W]
            bottom = torch.cat([cond[b, 2], cond[b, 3]], dim=2)  # [3, H, 2W]
            grid = torch.cat([top, bottom], dim=1)  # [3, 2H, 2W]
            grid_images.append(grid)

        grid_images = torch.stack(grid_images, dim=0)  # [B, 3, 2H, 2W]

        return {'cubemap': {'value': grid_images, 'type': 'image'}}


class ERPInitialVoxelMixin:
    """
    Mixin for supporting initial voxel latent mode in training.

    This mixin modifies the diffusion process to start from pre-encoded initial voxel latents
    instead of random Gaussian noise when initial_voxel_latent is provided in the data.

    Two training modes:
    1. Random Gaussian noise (when 'initial_voxel_latent' not in data)
    2. Initial voxel latent mode (when 'initial_voxel_latent' in data):
       - Use pre-encoded initial voxel latent (from dap_depth_voxels_ss_latent)
       - Add noise to create x_t (similar to SDEdit inversion)
    """
    def __init__(
        self,
        *args,
        use_initial_voxel_latent: bool = False,
        initial_voxel_t_range: Tuple[float, float] = (0.3, 0.7),
        **kwargs
    ):
        """
        Args:
            use_initial_voxel_latent: Whether to use initial voxel latent mode.
            initial_voxel_t_range: Range of timesteps to use for initial voxel mode.
                                  Lower t means less noise (closer to initial voxel).
                                  Higher t means more noise (closer to random).
        """
        super().__init__(*args, **kwargs)
        self.use_initial_voxel_latent = use_initial_voxel_latent
        self.initial_voxel_t_range = initial_voxel_t_range

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        initial_voxel_latent=None,
        **kwargs
    ):
        """
        Compute training losses with optional initial voxel latent support.

        When initial_voxel_latent is provided:
        - Use pre-encoded initial voxel latent as x_init
        - Sample t from initial_voxel_t_range
        - Create x_t by diffusing from x_init
        - Model learns to go from noisy initial voxel to GT

        Args:
            x_0: [B, 8, 16, 16, 16] GT latent
            cond: [B, 6, 3, H, W] cubemap images
            initial_voxel_latent: [B, 8, 16, 16, 16] pre-encoded initial voxel latent (optional)
        """
        if initial_voxel_latent is not None and self.use_initial_voxel_latent:
            # Initial voxel latent mode - use pre-encoded latent directly
            x_init = initial_voxel_latent.to(x_0.device)

            # Sample timestep from restricted range
            t_min, t_max = self.initial_voxel_t_range
            t = torch.rand(x_0.shape[0], device=x_0.device) * (t_max - t_min) + t_min

            # Generate noise
            noise = torch.randn_like(x_0)

            # Diffuse from initial voxel (not from x_0)
            t_view = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
            x_t = (1 - t_view) * x_init + (self.sigma_min + (1 - self.sigma_min) * t_view) * noise

            # Get conditioning
            cond = self.get_cond(cond, **kwargs)

            # Model prediction
            pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)

            # Target: velocity from x_init to x_0
            target = self.get_v(x_0, noise, t)

            from easydict import EasyDict as edict
            terms = edict()
            terms["mse"] = F.mse_loss(pred, target)
            terms["loss"] = terms["mse"]

            return terms, {}

        else:
            # Default: random Gaussian noise mode
            return super().training_losses(x_0=x_0, cond=cond, **kwargs)

    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Run snapshot with both random noise and initial voxel inference.

        Produces:
        - sample_gt: Ground truth latent
        - sample: Generated from random Gaussian noise
        - sample_init_voxel: Generated starting from initial voxel latent (SDEdit-style)
        - eval_* variants if eval_dataset is available

        SDEdit-style inference:
        1. Forward-diffuse initial_voxel_latent to t_start by mixing with Gaussian noise
        2. Denoise from t_start -> 0 (partial denoising, preserving spatial structure)
        """
        # Get base results from parent (random noise + eval if available)
        sample_dict = super().run_snapshot(num_samples, batch_size, verbose=verbose)

        if not self.use_initial_voxel_latent:
            return sample_dict

        # Generate additional samples from initial voxel latent
        sampler = self.get_sampler()

        # Use the midpoint of the training t_range as noise level
        t_noise = (self.initial_voxel_t_range[0] + self.initial_voxel_t_range[1]) / 2.0

        # Helper to run initial voxel inference on a dataset
        def _run_init_voxel_inference(dataset, prefix=''):
            dataloader = DataLoader(
                copy.deepcopy(dataset),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
            )

            init_voxel_samples = []
            has_init_voxel = False
            for i in range(0, num_samples, batch_size):
                batch = min(batch_size, num_samples - i)
                data = next(iter(dataloader))
                data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}

                if 'initial_voxel_latent' not in data:
                    break

                has_init_voxel = True
                x_init = data['initial_voxel_latent']

                # Add noise to initial voxel latent (same formula as training)
                # x_t = (1 - t) * x_init + (sigma_min + (1 - sigma_min) * t) * gaussian_noise
                # At t_noise=0.5: roughly half signal, half noise
                gaussian_noise = torch.randn_like(x_init)
                noise = (1 - t_noise) * x_init + (self.sigma_min + (1 - self.sigma_min) * t_noise) * gaussian_noise

                del data['x_0']
                del data['initial_voxel_latent']
                data.pop('_data_path', None)  # Remove non-tensor metadata
                args = self.get_inference_cond(**data)

                # SDEdit: denoise from t=t_noise -> 0 (not from t=1.0)
                # Build custom time schedule starting from t_noise
                import numpy as _np
                sdedit_steps = 50
                t_seq = _np.linspace(t_noise, 0, sdedit_steps + 1).tolist()
                t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(sdedit_steps)]
                sample = noise
                for t_cur, t_prev in t_pairs:
                    out = sampler.sample_once(
                        self.models['denoiser'], sample, t_cur, t_prev,
                        **args, guidance_strength=3.0,
                    )
                    sample = out.pred_x_prev
                init_voxel_samples.append(sample)

            if has_init_voxel and len(init_voxel_samples) > 0:
                key = f'{prefix}sample_init_voxel' if prefix else 'train_sample_init_voxel'
                sample_dict[key] = {
                    'value': torch.cat(init_voxel_samples, dim=0),
                    'type': 'sample'
                }

        # Run on training dataset
        _run_init_voxel_inference(self.dataset)

        # Run on eval dataset if available
        if self.eval_dataset is not None:
            _run_init_voxel_inference(self.eval_dataset, prefix='eval_')

        return sample_dict


## Spatial Attention Mask Generation

def create_spatial_attention_mask(
    camera_center: torch.Tensor,
    voxel_resolution: int = 16,
    tokens_per_face: int = 1029,
    fov_degrees: float = 120.0,
    soft_mask: bool = True,
    soft_margin: float = 0.1,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create spatial attention mask for voxel-cubemap cross-attention.

    This mask allows each cubemap face's tokens to only attend to voxels
    that are visible from that face's viewing direction.

    Cubemap face order: front, right, back, left, top, bottom
    Face viewing directions (in normalized voxel space):
    - front: +Y direction
    - back: -Y direction
    - right: +X direction
    - left: -X direction
    - top: +Z direction
    - bottom: -Z direction

    Args:
        camera_center: [B, 3] or [3] normalized camera center in voxel space [-1, 1]
        voxel_resolution: Resolution of voxel grid (default: 16 for latent space)
        tokens_per_face: Number of feature tokens per cubemap face (default: 1029 for DINOv3)
        fov_degrees: Field of view in degrees (90 or 120)
        soft_mask: If True, use soft (smooth) mask instead of hard mask
        soft_margin: Margin for soft mask transition (in cosine similarity)
        device: Device to create mask on

    Returns:
        attention_mask: [B, num_voxels, num_cubemap_tokens] or [num_voxels, num_cubemap_tokens]
            Value of 0 means attend, large negative value means don't attend
            Shape: [B, 4096, 6*tokens_per_face] for 16x16x16 voxels and 6 cubemap faces
    """
    # Handle batch dimension
    if camera_center.dim() == 1:
        camera_center = camera_center.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    B = camera_center.shape[0]
    camera_center = camera_center.to(device)

    # Create voxel grid positions in normalized space [-1, 1]
    coords = torch.linspace(-1 + 1/voxel_resolution, 1 - 1/voxel_resolution, voxel_resolution, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
    voxel_positions = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # [4096, 3]

    # Compute direction from camera to voxel
    camera_center_expanded = camera_center.unsqueeze(1)  # [B, 1, 3]
    voxel_positions_expanded = voxel_positions.unsqueeze(0)  # [1, 4096, 3]

    directions = voxel_positions_expanded - camera_center_expanded  # [B, 4096, 3]
    directions = F.normalize(directions, p=2, dim=-1)

    # Define cubemap face viewing directions
    face_directions = torch.tensor([
        [0.0, 1.0, 0.0],   # front: +Y
        [1.0, 0.0, 0.0],   # right: +X
        [0.0, -1.0, 0.0],  # back: -Y
        [-1.0, 0.0, 0.0],  # left: -X
        [0.0, 0.0, 1.0],   # top: +Z
        [0.0, 0.0, -1.0],  # bottom: -Z
    ], device=device, dtype=torch.float32)  # [6, 3]

    # Compute cosine similarity
    cos_sim = torch.einsum('bnd,fd->bnf', directions, face_directions)  # [B, 4096, 6]

    # FOV threshold
    fov_rad = np.radians(fov_degrees)
    cos_threshold = np.cos(fov_rad / 2)

    # Use FP16-safe large negative value
    MASK_NEG_INF = -1e4

    if soft_mask:
        scale = 10.0 / soft_margin
        mask_scores = torch.sigmoid(scale * (cos_sim - cos_threshold + soft_margin / 2))
        attention_mask = (1 - mask_scores) * MASK_NEG_INF
    else:
        visible = (cos_sim >= cos_threshold).float()
        attention_mask = (1 - visible) * MASK_NEG_INF

    # Expand mask to match token dimensions
    attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, -1, tokens_per_face)
    attention_mask = attention_mask.reshape(B, voxel_resolution**3, -1)  # [B, 4096, 6*tokens_per_face]

    if squeeze_batch:
        attention_mask = attention_mask.squeeze(0)

    return attention_mask


class SpatialAttentionMixin:
    """
    Mixin for spatially-aware cross-attention in ERP-to-3D generation.

    This mixin modifies the cross-attention to use a spatial mask that
    restricts each cubemap face to only attend to relevant voxel regions.

    Args:
        use_spatial_attention: Whether to use spatial attention mask
        spatial_attention_fov: FOV degrees (90 or 120)
        spatial_attention_soft: Use soft mask instead of hard mask
        voxel_resolution: Voxel latent resolution (default: 16)
        tokens_per_face: Number of tokens per cubemap face (default: 1029 for DINOv3)
    """
    def __init__(
        self,
        *args,
        use_spatial_attention: bool = True,
        spatial_attention_fov: float = 120.0,
        spatial_attention_soft: bool = True,
        spatial_attention_soft_margin: float = 0.1,
        voxel_resolution: int = 16,
        tokens_per_face: int = 1029,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_spatial_attention = use_spatial_attention
        self.spatial_attention_fov = spatial_attention_fov
        self.spatial_attention_soft = spatial_attention_soft
        self.spatial_attention_soft_margin = spatial_attention_soft_margin
        self.voxel_resolution = voxel_resolution
        self.tokens_per_face = tokens_per_face

    def get_spatial_attention_mask(
        self,
        camera_center: torch.Tensor,
        voxel_resolution: int = None,
        tokens_per_face: int = None
    ) -> torch.Tensor:
        """
        Get spatial attention mask for cross-attention.
        """
        if voxel_resolution is None:
            voxel_resolution = self.voxel_resolution
        if tokens_per_face is None:
            tokens_per_face = self.tokens_per_face

        return create_spatial_attention_mask(
            camera_center=camera_center,
            voxel_resolution=voxel_resolution,
            tokens_per_face=tokens_per_face,
            fov_degrees=self.spatial_attention_fov,
            soft_mask=self.spatial_attention_soft,
            soft_margin=self.spatial_attention_soft_margin,
            device=camera_center.device
        )

    def get_inference_cond(self, cond, camera_center=None, **kwargs):
        """
        Get the conditioning data for inference with spatial attention mask.
        """
        if camera_center is not None and self.use_spatial_attention:
            cross_attn_mask = self.get_spatial_attention_mask(
                camera_center=camera_center.cuda()
            )
            kwargs['cross_attn_mask'] = cross_attn_mask

        return super().get_inference_cond(cond, **kwargs)

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        camera_center=None,
        **kwargs
    ):
        """
        Compute training losses with spatial attention mask support.
        """
        cross_attn_mask = None
        if camera_center is not None and self.use_spatial_attention:
            cross_attn_mask = self.get_spatial_attention_mask(
                camera_center=camera_center.to(x_0.device)
            )

        if cross_attn_mask is not None:
            kwargs['cross_attn_mask'] = cross_attn_mask

        return super().training_losses(x_0=x_0, cond=cond, **kwargs)


# Convenience function to get trainable parameters from ERP encoder
def get_erp_encoder_params(trainer):
    """
    Get trainable parameters from ERP encoder (only view position embeddings).

    The DINOv3 backbone is frozen, so only view_pos_emb is trainable.
    """
    if hasattr(trainer, 'erp_encoder') and trainer.erp_encoder is not None:
        return [trainer.erp_encoder.view_pos_emb]
    return []
