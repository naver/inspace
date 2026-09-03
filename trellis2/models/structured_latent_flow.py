# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license
#
# Modified from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

from typing import *
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to, manual_cast, str_to_dtype, zero_module
from ..modules.transformer import AbsolutePositionEmbedder
from ..modules.norm import LayerNorm32
from ..modules import sparse as sp
from ..modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
from .sparse_structure_flow import TimestepEmbedder
from .sparse_elastic_mixin import SparseTransformerElasticMixin


# Maximum number of parts (assets) per sample for part embedding
PART_MAX_SIZE = 10

# FP16-safe large negative value for attention masking
MASK_NEG_INF = -1e4


class SparseResBlock3d(nn.Module):
    """
    3D Sparse Residual Block with time embedding conditioning.

    This block performs normalization, convolution operations on sparse tensors,
    and incorporates time embeddings via adaptive layer normalization.
    Supports optional up/downsampling.

    From OmniPart: Used for spatial downsampling/upsampling in structured latent flow.
    """
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample

        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        # First normalization and convolution
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)

        # Second convolution initialized to zero for stable training
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))

        # Time embedding projection for adaptive layer norm
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )

        # Skip connection with linear projection if channel dimensions change
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()

        # Optional up/downsampling
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """Apply up/downsampling if configured"""
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        """
        Forward pass of the residual block.

        Args:
            x: Input sparse tensor
            emb: Time embedding tensor

        Returns:
            Processed sparse tensor
        """
        # Project embedding to scale and shift factors
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        # Apply up/downsampling if needed
        x = self._updown(x)

        # Main processing path
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        # Apply adaptive layer norm using scale and shift from time embedding
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)

        # Residual connection
        h = h + self.skip_connection(x)
        return h
    

class SLatFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        dtype: str = 'float32',
        use_checkpoint: bool = False,
        share_mod: bool = False,
        initialization: str = 'vanilla',
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        # IO block parameters for sparse downsampling/upsampling (OmniPart-style)
        io_block_channels: Optional[List[int]] = None,
        patch_size: int = 1,
        num_io_res_blocks: int = 2,
        use_skip_connection: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = str_to_dtype(dtype)

        # IO block settings
        self.io_block_channels = io_block_channels
        self.patch_size = patch_size
        self.num_io_res_blocks = num_io_res_blocks
        self.use_skip_connection = use_skip_connection

        # Validate IO block settings
        if self.io_block_channels is not None:
            assert np.log2(patch_size) == len(io_block_channels), \
                f"Number of IO ResBlocks ({len(io_block_channels)}) must match log2(patch_size) ({np.log2(patch_size)})"

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)
            # Part positional embedding for OmniPart-style generation
            # Index 0 is for overall scene, 1-N for individual assets
            self.part_pe = nn.Embedding(PART_MAX_SIZE + 1, model_channels)
            self.part_pe_proj = nn.Linear(model_channels, model_channels)

        # Input layer: depends on whether IO blocks are used
        input_layer_out_channels = model_channels if io_block_channels is None else io_block_channels[0]
        self.input_layer = sp.SparseLinear(in_channels, input_layer_out_channels)

        # Input processing blocks (downsampling path) - OmniPart-style
        self.input_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
                # Add regular residual blocks at current resolution
                self.input_blocks.extend([
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks - 1)
                ])
                # Add downsampling block at the end of each resolution level
                self.input_blocks.append(
                    SparseResBlock3d(
                        chs,
                        model_channels,
                        out_channels=next_chs,
                        downsample=True,
                    )
                )

        # Core transformer blocks
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                rope_freq=rope_freq,
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        # Output processing blocks (upsampling path) - OmniPart-style
        self.out_blocks = nn.ModuleList([])
        if io_block_channels is not None:
            for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
                # Add upsampling block at the beginning of each resolution level
                in_chs = prev_chs * 2 if self.use_skip_connection else prev_chs
                self.out_blocks.append(
                    SparseResBlock3d(
                        in_chs,
                        model_channels,
                        out_channels=chs,
                        upsample=True,
                    )
                )
                # Add regular residual blocks at current resolution
                self.out_blocks.extend([
                    SparseResBlock3d(
                        chs * 2 if self.use_skip_connection else chs,
                        model_channels,
                        out_channels=chs,
                    )
                    for _ in range(num_io_res_blocks - 1)
                ])

        # Output layer
        out_layer_in_channels = model_channels if io_block_channels is None else io_block_channels[0]
        self.out_layer = sp.SparseLinear(out_layer_in_channels, out_channels)

        self.initialize_weights()
        self.convert_to(self.dtype)

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to(self, dtype: torch.dtype) -> None:
        """
        Convert the torso of the model to the specified dtype.
        """
        self.dtype = dtype
        self.blocks.apply(partial(convert_module_to, dtype=dtype))
        # Also convert IO blocks if they exist
        if len(self.input_blocks) > 0:
            self.input_blocks.apply(partial(convert_module_to, dtype=dtype))
        if len(self.out_blocks) > 0:
            self.out_blocks.apply(partial(convert_module_to, dtype=dtype))

    def initialize_weights(self) -> None:
        if self.initialization == 'vanilla':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)

            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)
            
        elif self.initialization == 'scaled':
            # Initialize transformer layers:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=np.sqrt(2.0 / (5.0 * self.model_channels)))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
            
            # Scaled init for to_out and ffn2
            def _scaled_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, std=1.0 / np.sqrt(5 * self.num_blocks * self.model_channels))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            for block in self.blocks:
                block.self_attn.to_out.apply(_scaled_init)
                block.cross_attn.to_out.apply(_scaled_init)
                block.mlp.mlp[2].apply(_scaled_init)
            
            # Initialize input layer to make the initial representation have variance 1
            nn.init.normal_(self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels))
            nn.init.zeros_(self.input_layer.bias)
            
            # Initialize timestep embedding MLP:
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            
            # Zero-out adaLN modulation layers in DiT blocks:
            if self.share_mod:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
            else:
                for block in self.blocks:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

        # Initialize part embedding layers (if using APE)
        if self.pe_mode == "ape":
            nn.init.zeros_(self.part_pe_proj.weight)
            nn.init.zeros_(self.part_pe_proj.bias)
            self.part_pe.weight.data.normal_(mean=0.0, std=0.02)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, List[torch.Tensor]],
        concat_cond: Optional[sp.SparseTensor] = None,
        **kwargs
    ) -> sp.SparseTensor:
        """
        Forward pass of the Structured Latent Flow model.

        Args:
            x: Input sparse tensor [N, C] where C is in_channels
            t: Timestep tensor [B]
            cond: Condition tensor [B, L, D] or list of tensors
            concat_cond: Optional concatenated condition (e.g., shape latent for texture)
            **kwargs:
                part_layouts: Optional list of lists of slices for OmniPart-style generation
                             [[slice(0, overall_end), slice(...), ...], ...]
                overlap_groups: Optional list of lists of overlap groups per batch
                               [[group0, group1, ...], ...]
                               where each group is a list of asset indices that overlap
                cross_attn_masks: Optional list of per-part cross-attention masks per batch
                                 [[overall_mask, asset0_mask, ...], ...]
                                 Each mask is [cond_len] or [N_voxels, cond_len] bool, True = attend

        Returns:
            Output sparse tensor [N, out_channels]
        """
        # Check if part_layouts is provided - determines OmniPart vs TRELLIS mode
        part_layouts = kwargs.pop('part_layouts', None)
        overlap_groups = kwargs.pop('overlap_groups', None)
        cross_attn_masks = kwargs.pop('cross_attn_masks', None)
        has_layout = kwargs.pop('has_layout', None)

        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        # TRELLIS mode: simple forward without part-aware processing
        if part_layouts is None:
            return self._forward_trellis(x, t, cond)

        # ERP OmniPart mode: part-aware forward with overlap groups and cross-attention masks
        if overlap_groups is not None or cross_attn_masks is not None:
            return self._forward_omnipart_erp(
                x, t, cond, part_layouts, overlap_groups, cross_attn_masks,
                has_layout=has_layout,
            )

        # Standard OmniPart mode: part-aware forward with part_layouts
        return self._forward_omnipart(x, t, cond, part_layouts)

    def _forward_trellis(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, sp.VarLenTensor],
    ) -> sp.SparseTensor:
        """
        TRELLIS-style forward pass without part-aware processing.
        Used when part_layouts is not provided.

        If io_block_channels is set, uses sparse downsampling/upsampling (OmniPart-style).
        """
        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)
        cond = manual_cast(cond, self.dtype)

        # ---- Input blocks (downsampling) ----
        skips = []
        if self.io_block_channels is not None:
            for block in self.input_blocks:
                h = block(h, t_emb)
                if self.use_skip_connection:
                    skips.append(h)

        # ---- Positional embeddings (at downsampled resolution) ----
        if self.pe_mode == "ape":
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)

        # ---- Transformer blocks ----
        for block in self.blocks:
            h = block(h, t_emb, cond)

        # ---- Output blocks (upsampling) ----
        if self.io_block_channels is not None:
            skips = skips[::-1]  # Reverse for upsampling order
            skip_idx = 0
            for block in self.out_blocks:
                if self.use_skip_connection and skip_idx < len(skips):
                    # Concatenate skip connection
                    h = sp.sparse_cat([h, skips[skip_idx]], dim=-1)
                    skip_idx += 1
                h = block(h, t_emb)

        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h

    def _forward_omnipart(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, sp.VarLenTensor],
        part_layouts: List[List[slice]],
    ) -> sp.SparseTensor:
        """
        OmniPart-style forward pass with part-aware processing.
        Used when part_layouts is provided.

        Each sample has multiple parts:
        - part_layouts[batch_idx][0]: overall scene slice
        - part_layouts[batch_idx][1:]: individual asset slices

        The model:
        1. Assigns unique batch IDs to each part for isolated processing
        2. Input blocks: each part processed independently (part-wise batch IDs)
        3. Transformer blocks: parts within same batch attend to each other (batch-wise IDs)
        4. Output blocks: each part processed independently again (part-wise batch IDs)
        5. Adds part positional embeddings to distinguish parts
        """
        input_dtype = x.dtype

        # Store original batch IDs for later restoration
        original_batch_ids = x.coords[:, 0].clone()

        # Create new batch IDs to represent individual parts
        new_batch_ids = torch.zeros_like(original_batch_ids)

        # Track which part_id each token belongs to (for part PE)
        part_ids_per_token = torch.zeros_like(original_batch_ids)

        # Assign unique IDs to each part across all batches
        part_id = 0
        len_before = 0
        batch_last_partid = []
        for batch_idx, part_layout in enumerate(part_layouts):
            for layout_idx, layout in enumerate(part_layout):
                adjusted_layout = slice(
                    layout.start + len_before,
                    layout.stop + len_before,
                    layout.step
                )
                new_batch_ids[adjusted_layout] = part_id
                # layout_idx: 0=overall, 1+=assets
                part_ids_per_token[adjusted_layout] = layout_idx
                part_id += 1

            batch_last_partid.append(part_id)
            len_before += part_layout[-1].stop

        # Project input to model dimensions
        h = self.input_layer(x)
        h = manual_cast(h, self.dtype)

        # Replace batch IDs with part IDs (part-wise processing)
        h = sp.SparseTensor(
            feats=h.feats,
            coords=torch.cat([new_batch_ids.view(-1, 1), h.coords[:, 1:]], dim=1),
        )

        # Process timestep embedding
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype)

        cond = manual_cast(cond, self.dtype)

        # ---- Input blocks (downsampling) with part-wise batch IDs ----
        # Create expanded t_emb for IO blocks (one per part)
        t_emb_updown = []
        for batch_idx, part_layout in enumerate(part_layouts):
            t_emb_updown_batch = t_emb[batch_idx:batch_idx+1].repeat(len(part_layout), 1)
            t_emb_updown.append(t_emb_updown_batch)
        t_emb_updown = torch.cat(t_emb_updown, dim=0)
        t_emb_updown = manual_cast(t_emb_updown, self.dtype)

        skips = []
        if self.io_block_channels is not None:
            for block in self.input_blocks:
                h = block(h, t_emb_updown)
                if self.use_skip_connection:
                    skips.append(h.feats)

        # Store part-wise batch IDs before transformer processing
        part_wise_batch_ids = h.coords[:, 0].clone()

        # Convert to batch-wise IDs for transformer blocks
        # This allows parts within the same sample to attend to each other
        new_transformer_batch_ids = torch.zeros_like(part_wise_batch_ids)
        part_ids_in_batch = torch.zeros_like(part_wise_batch_ids)
        start_reform = 0
        last_part_id = 0
        for part_id in batch_last_partid:
            mask = (part_wise_batch_ids >= last_part_id) & (part_wise_batch_ids < part_id)
            new_transformer_batch_ids[mask] = start_reform
            # Relative part ID within the batch (for part PE)
            part_ids_in_batch[mask] = part_wise_batch_ids[mask] - last_part_id
            last_part_id = part_id
            start_reform += 1

        # Update coordinates with batch-wise IDs for transformer processing
        h = sp.SparseTensor(
            feats=h.feats,
            coords=torch.cat([new_transformer_batch_ids.view(-1, 1), h.coords[:, 1:]], dim=1),
        )

        # ---- Add positional embeddings (at downsampled resolution) ----
        if self.pe_mode == "ape":
            # Spatial positional embedding
            pe = self.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.dtype)

            # Part positional embedding (0=overall, 1+=assets)
            part_pe = self.part_pe(part_ids_in_batch)
            part_pe = self.part_pe_proj(part_pe)
            h = h + manual_cast(part_pe, self.dtype)

        # ---- Transformer blocks ----
        for block in self.blocks:
            h = block(h, t_emb, cond)

        # Restore part-wise batch IDs for output blocks
        h = sp.SparseTensor(
            feats=h.feats,
            coords=torch.cat([part_wise_batch_ids.view(-1, 1), h.coords[:, 1:]], dim=1),
        )

        # ---- Output blocks (upsampling) with skip connections ----
        if self.io_block_channels is not None:
            skips = skips[::-1]  # Reverse for upsampling order
            for block_idx, block in enumerate(self.out_blocks):
                if self.use_skip_connection and block_idx < len(skips):
                    # Concatenate skip connection
                    h = h.replace(torch.cat([h.feats, skips[block_idx]], dim=1))
                h = block(h, t_emb_updown)

        # Final output
        h = manual_cast(h, input_dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)

        # Restore original batch IDs
        h = sp.SparseTensor(
            feats=h.feats,
            coords=torch.cat([original_batch_ids.view(-1, 1), h.coords[:, 1:]], dim=1),
        )

        return h

    def _forward_omnipart_erp(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        cond: Union[torch.Tensor, sp.VarLenTensor],
        part_layouts: List[List[slice]],
        overlap_groups: Optional[List[List[List[int]]]] = None,
        cross_attn_masks: Optional[List[List[torch.Tensor]]] = None,
        has_layout: Optional[List[bool]] = None,
    ) -> sp.SparseTensor:
        """
        ERP-style OmniPart forward pass with:
        1. Grouped self-attention based on 3D bbox overlap
        2. Per-voxel cross-attention masks via SDPA backend

        Args:
            x: Input sparse tensor
            t: Timestep tensor [B]
            cond: Condition tensor (VarLenTensor with [B, L, D] features)
            part_layouts: List of slices per batch [[overall, asset0, ...], ...]
            overlap_groups: Per-batch overlap groups [[group0, group1, ...], ...]
                           where each group is list of asset indices (0-indexed from assets)
            cross_attn_masks: Per-batch per-part attention masks
                             [[overall_mask, asset0_mask, ...], ...]
                             Each mask is [cond_len] or [N_voxels, cond_len] bool (True=attend)

        Self-attention strategy:
        - Overall scene has its own batch ID (self-attends only)
        - Assets in same overlap group share batch ID (attend to each other)
        - Singleton assets have own batch ID (self-attends only)
        - Uses flash_attn (no mask needed — batch IDs handle grouping)

        Cross-attention strategy:
        - ALL groups receive the SAME full condition tokens (e.g., 6174 tokens)
        - A flat per-voxel mask [total_Q_tokens, cond_len] differentiates:
          * Overall voxels: per-voxel spatial mask (horizontal stripes, like Stage 1)
          * Asset voxels: per-asset bbox projection mask (same for all voxels in part)
        - Even within an overlap group, different assets get different mask rows
        - Uses SDPA backend when mask is provided, flash_attn when not
        """
        input_dtype = x.dtype
        device = x.device
        batch_size = len(part_layouts)

        # Store original batch IDs for later restoration
        original_batch_ids = x.coords[:, 0].clone()

        # =========================================================================
        # Phase 1: Compute self-attention batch IDs based on overlap groups
        # =========================================================================

        # Create new batch IDs for grouped self-attention
        new_self_attn_batch_ids = torch.zeros_like(original_batch_ids)
        part_ids_per_token = torch.zeros_like(original_batch_ids)

        current_batch_id = 0
        len_before = 0
        batch_info = []  # Track batch boundaries for later

        for batch_idx, part_layout in enumerate(part_layouts):
            batch_start_id = current_batch_id
            batch_has_layout = has_layout[batch_idx] if has_layout is not None else False
            asset_start = 2 if batch_has_layout else 1  # Assets start after overall (+ layout)
            num_assets = len(part_layout) - asset_start

            # Get overlap groups for this batch (default: each asset separate)
            if overlap_groups is not None and batch_idx < len(overlap_groups):
                groups = overlap_groups[batch_idx]
            else:
                groups = [[i] for i in range(num_assets)]  # Each asset separate

            # Determine which group each asset belongs to
            asset_to_group = {}
            for group_idx, group in enumerate(groups):
                for asset_idx in group:
                    asset_to_group[asset_idx] = group_idx

            # Assign batch IDs:
            # - Overall (index 0) gets its own batch ID
            # - Layout (index 1, if present) gets its own batch ID
            # - Each overlap group gets its own batch ID
            # - Singleton assets (not in any group) get own batch ID

            # Overall scene (part PE = 0)
            overall_slice = part_layout[0]
            adjusted_overall = slice(
                overall_slice.start + len_before,
                overall_slice.stop + len_before
            )
            new_self_attn_batch_ids[adjusted_overall] = current_batch_id
            part_ids_per_token[adjusted_overall] = 0  # Overall = 0
            overall_batch_id = current_batch_id
            current_batch_id += 1

            # Layout (part PE = 1, own batch ID, no grouping)
            if batch_has_layout:
                layout_slice = part_layout[1]
                adjusted_layout = slice(
                    layout_slice.start + len_before,
                    layout_slice.stop + len_before
                )
                new_self_attn_batch_ids[adjusted_layout] = current_batch_id
                part_ids_per_token[adjusted_layout] = 1  # Layout = 1
                current_batch_id += 1

            # Assign batch IDs to groups
            group_batch_ids = {}
            for group_idx in range(len(groups)):
                group_batch_ids[group_idx] = current_batch_id
                current_batch_id += 1

            # Assets (part PE = asset_start + asset_idx)
            for asset_idx in range(num_assets):
                asset_slice = part_layout[asset_idx + asset_start]
                adjusted_asset = slice(
                    asset_slice.start + len_before,
                    asset_slice.stop + len_before
                )

                # Get group batch ID
                if asset_idx in asset_to_group:
                    batch_id = group_batch_ids[asset_to_group[asset_idx]]
                else:
                    # Singleton (not in any group)
                    batch_id = current_batch_id
                    current_batch_id += 1

                new_self_attn_batch_ids[adjusted_asset] = batch_id
                part_ids_per_token[adjusted_asset] = asset_idx + asset_start  # 2+ if layout, 1+ if not

            # Build group_details: per-group type info for sparse cross-attention
            # Order must match batch_id assignment: overall, layout?, groups..., ungrouped...
            group_details = []
            group_details.append({'type': 'overall', 'part_idx': 0})
            if batch_has_layout:
                group_details.append({'type': 'layout', 'part_idx': 1})
            for group_idx in range(len(groups)):
                group = groups[group_idx]
                if len(group) == 1:
                    a_idx = group[0]
                    group_details.append({
                        'type': 'singleton',
                        'asset_idx': a_idx,
                        'mask_idx': a_idx + asset_start,
                    })
                else:
                    group_details.append({
                        'type': 'overlap',
                        'asset_indices': group,
                        'mask_indices': [a + asset_start for a in group],
                    })
            # Safety: handle assets not in any group (line 731-733 fallback)
            for asset_idx in range(num_assets):
                if asset_idx not in asset_to_group:
                    group_details.append({
                        'type': 'singleton',
                        'asset_idx': asset_idx,
                        'mask_idx': asset_idx + asset_start,
                    })

            batch_info.append({
                'batch_idx': batch_idx,
                'start_id': batch_start_id,
                'end_id': current_batch_id,
                'overall_batch_id': overall_batch_id,
                'len_before': len_before,
                'part_layout': part_layout,
                'has_layout': batch_has_layout,
                'asset_start': asset_start,
                'group_details': group_details,
            })
            len_before += part_layout[-1].stop

        # =========================================================================
        # Phase 2: Project input and apply IO blocks
        # =========================================================================

        h = self.input_layer(x) # x.shape = SparseTensor[4, 32] -> h.shape = SparseTensor[4, 1536]
        h = manual_cast(h, self.dtype) # h.shape = SparseTensor[4, 1536], h.coords.shape = [17143, 4], h.feats.shape = [17143, 1536]

        # Apply self-attention batch IDs
        h = sp.SparseTensor( 
            feats=h.feats, # h.feats.shape = [17143, 1536] # features stay unchanged
            coords=torch.cat([new_self_attn_batch_ids.view(-1, 1),  # column 0: new batch ID (0-20), assigns a group for all 17143 tokens
            h.coords[:, 1:]], dim=1), # [17143, 4] # columns 1-3: spatial coordinates stay unchanged # the 4 batch items sum to 17143
        ) # => self-attention runs only among tokens sharing the same batch ID
        # h.shape=SparseTensor[21, 1536], h[0].coords.shape=[1807,4], h[1].coords.shape=[196,4], h[2].coords.shape=[37,4]...
        # Process timestep embedding
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.dtype) # t_emb.shape=[4, 9216]

        cond = manual_cast(cond, self.dtype) # cond.shape=[4, 6174, 1024]

        # Create expanded t_emb for IO blocks (one per unique self-attention batch ID)
        # The number of unique batch IDs is current_batch_id
        t_emb_updown = []
        for info in batch_info:
            batch_idx = info['batch_idx']
            num_batch_ids = info['end_id'] - info['start_id'] # 6
            for _ in range(num_batch_ids):
                t_emb_updown.append(t_emb[batch_idx:batch_idx+1])
        t_emb_updown = torch.cat(t_emb_updown, dim=0)
        t_emb_updown = manual_cast(t_emb_updown, self.dtype) # [21, 9216] # t_emb is [4, 9216] originally, but becomes 21 rows depending on the part groups

        # ---- Input blocks (downsampling) ----
        # skips = []
        # if self.io_block_channels is not None:
        #     for block in self.input_blocks:
        #         h = block(h, t_emb_updown)
        #         if self.use_skip_connection:
        #             skips.append(h.feats)

        # Store batch IDs after downsampling for later restoration
        downsampled_batch_ids = h.coords[:, 0].clone() # [17143]

        # ---- Add positional embeddings (at downsampled resolution) ----
        if self.pe_mode == "ape": # self == ElasticSlatFlowModel, pe => potision embedding
            pe = self.pos_embedder(h.coords[:, 1:]) # pe.shape = [17143, 1536] # h.coords.shape = [17143, 4], h.coords[:, 1:].shape = [17143, 3] drop the leading group id
            h = h + manual_cast(pe, self.dtype)

            # Part PE: clamp to valid range
            # Note: part_ids_per_token needs to be recomputed for downsampled coords
            # For now, we use the part embedding based on the batch ID mapping
            clamped_part_ids = torch.clamp(part_ids_per_token, 0, PART_MAX_SIZE) # part_ids_per_token.shape = [17143], tensor([0, 0, 0,  ..., 8, 8, 8]
            # If downsampling happened, we need to map back - but spatial downsampling
            # doesn't change the order, just reduces points. Use original mapping.

            # if self.io_block_channels is not None:
            #     # Recompute part_ids for downsampled voxels
            #     # Since downsampling preserves batch_id, we can use that
            #     downsampled_part_ids = torch.zeros(h.feats.shape[0], dtype=torch.long, device=device)
            #     # Map batch_id back to part_id
            #     for info in batch_info:
            #         for part_idx, part_slice in enumerate(info['part_layout']):
            #             # This is approximate - for exact mapping we'd need to track indices
            #             pass
            #     # Fallback: use zeros (overall) for all - the batch ID already encodes grouping
            #     clamped_part_ids = torch.zeros(h.feats.shape[0], dtype=torch.long, device=device)

            part_pe = self.part_pe(clamped_part_ids) # part_pe.shape = [17143, 1536] # clamped_part_ids.shape = [17143], tensor([0, 0, 0,  ..., 8, 8, 8])
            part_pe = self.part_pe_proj(part_pe) # part_pe.shape = [17143, 1536]
            h = h + manual_cast(part_pe, self.dtype)

        # =========================================================================
        # Phase 3: Build condition tensor and per-voxel cross-attention mask
        # =========================================================================
        # Sparse Cross-Attention: pre-filter condition tokens for asset groups.
        # - Overall/Layout: full 6174 tokens + per-voxel spatial mask
        # - Singleton asset: only bbox-projected tokens (~300-500), no mask needed
        # - Overlap group: union of assets' tokens (~600-800), remapped per-voxel mask
        # Uses VarLenTensor for variable KV length per group.

        # Extract per-sample condition features
        if isinstance(cond, sp.VarLenTensor):
            cond_feats_list = []
            for i in range(batch_size):
                start, end = cond.layout[i].start, cond.layout[i].stop
                cond_feats_list.append(cond.feats[start:end])
        else:
            cond_feats_list = [cond[i] for i in range(batch_size)]

        cross_attn_mask_flat = None

        if cross_attn_masks is not None:
            # ---- SPARSE MODE: VarLenTensor condition with pre-filtered tokens ----
            total_q_tokens = h.feats.shape[0]
            cond_tensors_list = []  # Per-group condition tensors (variable length)
            group_cond_info = []    # Per-group metadata for mask construction

            for info in batch_info:
                batch_idx = info['batch_idx']
                cond_feats = cond_feats_list[batch_idx]  # [6174, D]
                masks = cross_attn_masks[batch_idx] if batch_idx < len(cross_attn_masks) else None

                for gd in info['group_details']:
                    if gd['type'] in ('overall', 'layout'):
                        # Full condition tokens for overall/layout
                        cond_tensors_list.append(cond_feats)
                        group_cond_info.append({'mode': 'full', 'nkv': cond_feats.shape[0]})

                    elif gd['type'] == 'singleton':
                        # Pre-filter to bbox-projected tokens only
                        mask_idx = gd['mask_idx']
                        if masks is not None and mask_idx < len(masks) and masks[mask_idx] is not None:
                            asset_mask = masks[mask_idx].to(device=device, dtype=torch.bool)
                            if asset_mask.dim() == 1 and asset_mask.any():
                                filtered = cond_feats[asset_mask]  # [K_i, D]
                                cond_tensors_list.append(filtered)
                                group_cond_info.append({'mode': 'filtered', 'nkv': filtered.shape[0]})
                                continue
                        # Fallback: full condition tokens
                        cond_tensors_list.append(cond_feats)
                        group_cond_info.append({'mode': 'full', 'nkv': cond_feats.shape[0]})

                    elif gd['type'] == 'overlap':
                        # Union of all assets' projected tokens in this overlap group
                        mask_indices = gd['mask_indices']
                        if masks is not None:
                            union_mask = torch.zeros(cond_feats.shape[0], dtype=torch.bool, device=device)
                            valid = False
                            for mi in mask_indices:
                                if mi < len(masks) and masks[mi] is not None:
                                    m = masks[mi].to(device=device, dtype=torch.bool)
                                    if m.dim() == 1:
                                        union_mask |= m
                                        valid = True
                            if valid and union_mask.any():
                                union_indices = union_mask.nonzero(as_tuple=True)[0]  # [K_union]
                                filtered = cond_feats[union_mask]  # [K_union, D]
                                cond_tensors_list.append(filtered)
                                group_cond_info.append({
                                    'mode': 'union',
                                    'nkv': filtered.shape[0],
                                    'union_indices': union_indices,
                                })
                                continue
                        # Fallback: full condition tokens
                        cond_tensors_list.append(cond_feats)
                        group_cond_info.append({'mode': 'full', 'nkv': cond_feats.shape[0]})

            cond_full = sp.VarLenTensor.from_tensor_list(cond_tensors_list)
            # cond_feats already in self.dtype from line 770 manual_cast

            # ---- Build per-voxel cross-attention mask [total_Q, max_kv_len] ----
            # Initialized to False. SDPA loop slices mask_g = mask[q_off:q_off+nq, :nkv]
            # so columns beyond nkv for a group are irrelevant.
            max_kv_len = max(gi['nkv'] for gi in group_cond_info)
            cross_attn_mask_flat = torch.zeros(
                total_q_tokens, max_kv_len, dtype=torch.bool, device=device
            )

            group_global_idx = 0
            for info in batch_info:
                batch_idx = info['batch_idx']
                part_layout = info['part_layout']
                len_before_mask = info['len_before']
                asset_start = info['asset_start']
                masks = cross_attn_masks[batch_idx] if batch_idx < len(cross_attn_masks) else None

                for gd in info['group_details']:
                    gi = group_cond_info[group_global_idx]
                    nkv = gi['nkv']

                    if gd['type'] == 'overall':
                        # Per-voxel spatial mask
                        s = part_layout[0]
                        adj = slice(s.start + len_before_mask, s.stop + len_before_mask)
                        if masks is not None and len(masks) > 0 and masks[0] is not None:
                            m = masks[0].to(device=device, dtype=torch.bool)
                            if m.dim() == 2:
                                cross_attn_mask_flat[adj, :nkv] = m[:, :nkv]
                            elif m.dim() == 1 and m.any():
                                cross_attn_mask_flat[adj, :nkv] = m[:nkv].unsqueeze(0)
                            else:
                                cross_attn_mask_flat[adj, :nkv] = True
                        else:
                            cross_attn_mask_flat[adj, :nkv] = True

                    elif gd['type'] == 'layout':
                        # Per-voxel spatial mask (same as overall)
                        s = part_layout[1]
                        adj = slice(s.start + len_before_mask, s.stop + len_before_mask)
                        if masks is not None and len(masks) > 1 and masks[1] is not None:
                            m = masks[1].to(device=device, dtype=torch.bool)
                            if m.dim() == 2:
                                cross_attn_mask_flat[adj, :nkv] = m[:, :nkv]
                            elif m.dim() == 1 and m.any():
                                cross_attn_mask_flat[adj, :nkv] = m[:nkv].unsqueeze(0)
                            else:
                                cross_attn_mask_flat[adj, :nkv] = True
                        else:
                            cross_attn_mask_flat[adj, :nkv] = True

                    elif gd['type'] == 'singleton':
                        s = part_layout[gd['asset_idx'] + asset_start]
                        adj = slice(s.start + len_before_mask, s.stop + len_before_mask)
                        if gi['mode'] == 'filtered':
                            # Pre-filtered: all tokens are relevant → all True
                            cross_attn_mask_flat[adj, :nkv] = True
                        else:
                            # Full fallback: apply original mask
                            mi = gd['mask_idx']
                            if masks is not None and mi < len(masks) and masks[mi] is not None:
                                m = masks[mi].to(device=device, dtype=torch.bool)
                                if m.dim() == 1 and m.any():
                                    cross_attn_mask_flat[adj, :nkv] = m[:nkv].unsqueeze(0)
                                elif m.dim() == 2:
                                    cross_attn_mask_flat[adj, :nkv] = m[:, :nkv]
                                else:
                                    cross_attn_mask_flat[adj, :nkv] = True
                            else:
                                cross_attn_mask_flat[adj, :nkv] = True

                    elif gd['type'] == 'overlap':
                        asset_indices = gd['asset_indices']
                        mask_indices = gd['mask_indices']

                        if gi['mode'] == 'union':
                            # Remap each asset's original mask to union token positions
                            union_indices = gi['union_indices']  # [K_union]
                            for a_idx, mi in zip(asset_indices, mask_indices):
                                s = part_layout[a_idx + asset_start]
                                adj = slice(s.start + len_before_mask, s.stop + len_before_mask)
                                if masks is not None and mi < len(masks) and masks[mi] is not None:
                                    orig = masks[mi].to(device=device, dtype=torch.bool)
                                    if orig.dim() == 1:
                                        # remapped[j] = orig[union_indices[j]]
                                        remapped = orig[union_indices]  # [K_union]
                                        if remapped.any():
                                            cross_attn_mask_flat[adj, :nkv] = remapped.unsqueeze(0)
                                        else:
                                            cross_attn_mask_flat[adj, :nkv] = True
                                    else:
                                        cross_attn_mask_flat[adj, :nkv] = True
                                else:
                                    cross_attn_mask_flat[adj, :nkv] = True
                        else:
                            # Full fallback: apply original masks
                            for a_idx, mi in zip(asset_indices, mask_indices):
                                s = part_layout[a_idx + asset_start]
                                adj = slice(s.start + len_before_mask, s.stop + len_before_mask)
                                if masks is not None and mi < len(masks) and masks[mi] is not None:
                                    m = masks[mi].to(device=device, dtype=torch.bool)
                                    if m.dim() == 1 and m.any():
                                        cross_attn_mask_flat[adj, :nkv] = m[:nkv].unsqueeze(0)
                                    elif m.dim() == 2:
                                        cross_attn_mask_flat[adj, :nkv] = m[:, :nkv]
                                    else:
                                        cross_attn_mask_flat[adj, :nkv] = True
                                else:
                                    cross_attn_mask_flat[adj, :nkv] = True

                    group_global_idx += 1

        else:
            # ---- DENSE MODE: no mask, unchanged (uses flash_attn) ----
            cond_full_list = []
            for info in batch_info:
                cond_feats = cond_feats_list[info['batch_idx']]
                num_groups = info['end_id'] - info['start_id']
                cond_full_list.append(cond_feats.unsqueeze(0).expand(num_groups, -1, -1))
            cond_full = torch.cat(cond_full_list, dim=0)
            cond_full = manual_cast(cond_full, self.dtype)

        # =========================================================================
        # Phase 4: Process with transformer blocks
        # =========================================================================

        # Hybrid attention strategy:
        # - Self-attention: whole-scale (all parts in same sample attend to each other)
        #   Uses original_batch_ids (one per sample) → flash_attn, O(n) memory
        # - Cross-attention: grouped (each group gets its own pre-filtered KV tokens)
        #   Uses grouped batch IDs in h.coords → VarLenTensor + SDPA with mask
        for block in self.blocks:
            h = block(h, t_emb_updown, cond_full,
                      cross_attn_mask=cross_attn_mask_flat,
                      self_attn_batch_ids=original_batch_ids)
                      
        # =========================================================================
        # Phase 5: Output blocks and restore original batch IDs
        # =========================================================================

        # Restore downsampled batch IDs for output blocks
        h = sp.SparseTensor(
            feats=h.feats,
            coords=torch.cat([downsampled_batch_ids.view(-1, 1), h.coords[:, 1:]], dim=1),
        ) # downsampled_batch_ids.shape = [17143]

        # ---- Output blocks (upsampling) with skip connections ----
        # if self.io_block_channels is not None:
        #     skips = skips[::-1]  # Reverse for upsampling order
        #     for block_idx, block in enumerate(self.out_blocks):
        #         if self.use_skip_connection and block_idx < len(skips):
        #             # Concatenate skip connection
        #             h = h.replace(torch.cat([h.feats, skips[block_idx]], dim=1))
        #         h = block(h, t_emb_updown)

        h = manual_cast(h, input_dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)

        # Restore original batch IDs
        h = sp.SparseTensor(
            feats=h.feats,
            coords=torch.cat([original_batch_ids.view(-1, 1), h.coords[:, 1:]], dim=1),
        )

        return h


class ElasticSLatFlowModel(SparseTransformerElasticMixin, SLatFlowModel):
    """
    SLat Flow Model with elastic memory management.
    Used for training with low VRAM.
    """
    pass
