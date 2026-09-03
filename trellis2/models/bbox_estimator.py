# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
3D Bounding Box Estimator using 3D CNN encoder + DETR-style decoder.

Takes decoded voxel grid [B, 1, 64, 64, 64] (binary occupancy) and predicts
oriented bounding boxes (OBBs) for furniture in the scene.

Output format: [cx, cy, cz, sx, sy, sz, rotation_yaw] in O-Voxel normalized space [-0.5, 0.5].

Encoder variants:
- VoxelEncoder3D: Plain CNN (v2/v3). Multiscale extracts 16^3+8^3+4^3 = 4672 tokens.
- VoxelEncoderFPN3D: CNN + Feature Pyramid Network (v4). Propagates 32^3 spatial detail
  into 16^3 features via top-down pathway. Same 4672 DETR tokens but much richer features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple multi-layer perceptron."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResBlock3d(nn.Module):
    """3D residual block with GroupNorm."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class VoxelEncoder3D(nn.Module):
    """
    3D CNN encoder that extracts multi-scale features from a binary voxel grid.

    [B, 1, 64, 64, 64]
      -> Conv3d(1->32, s=2)  + ResBlock -> [B, 32, 32, 32]
      -> Conv3d(32->64, s=2) + ResBlock -> [B, 64, 16, 16, 16]
      -> Conv3d(64->128, s=2)+ ResBlock -> [B, 128, 8, 8, 8]
      -> Conv3d(128->D, s=2) + ResBlock -> [B, D, 4, 4, 4]
      -> flatten -> [B, 64, D]

    When multiscale=True, returns features from 16^3, 8^3, and 4^3 stages.
    """
    def __init__(self, d_model=256, multiscale=False):
        super().__init__()
        self.multiscale = multiscale
        self.encoder = nn.Sequential(
            # 64 -> 32
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResBlock3d(32),
            # 32 -> 16
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock3d(64),
            # 16 -> 8
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock3d(128),
            # 8 -> 4
            nn.Conv3d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            ResBlock3d(d_model),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, 64, 64, 64] binary voxel grid
        Returns:
            If multiscale=False: [B, 64, D] flattened feature tokens
            If multiscale=True: tuple of (feat_16, feat_8, feat_4)
                feat_16: [B, 4096, 64]  (16^3 positions, 64 channels)
                feat_8:  [B, 512, 128]  (8^3 positions, 128 channels)
                feat_4:  [B, 64, D]     (4^3 positions, D channels)
        """
        # Stage 1: 64 -> 32 (layers 0-3)
        x = self.encoder[0:4](x)   # [B, 32, 32, 32]
        # Stage 2: 32 -> 16 (layers 4-7)
        x = self.encoder[4:8](x)   # [B, 64, 16, 16, 16]
        if self.multiscale:
            feat_16 = x
        # Stage 3: 16 -> 8 (layers 8-11)
        x = self.encoder[8:12](x)  # [B, 128, 8, 8, 8]
        if self.multiscale:
            feat_8 = x
        # Stage 4: 8 -> 4 (layers 12-15)
        x = self.encoder[12:16](x) # [B, D, 4, 4, 4]

        B = x.shape[0]
        if self.multiscale:
            # Flatten each scale: [B, C, H, W, D] -> [B, H*W*D, C]
            f16 = feat_16.reshape(B, 64, -1).permute(0, 2, 1)   # [B, 4096, 64]
            f8 = feat_8.reshape(B, 128, -1).permute(0, 2, 1)    # [B, 512, 128]
            f4 = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)  # [B, 64, D]
            return f16, f8, f4
        else:
            D = x.shape[1]
            return x.reshape(B, D, -1).permute(0, 2, 1)  # [B, 64, D]


class VoxelEncoderFPN3D(nn.Module):
    """
    3D CNN encoder with Feature Pyramid Network (FPN) neck + C2 fusion.

    Extracts multi-scale features at 32^3, 16^3, 8^3, 4^3 and enriches them
    through a top-down FPN pathway. Additionally, C2 (32^3) features are
    pooled to 16^3 and fused directly into P3, ensuring 32^3 spatial detail
    is always available at the 16^3 level.

    Bottom-up encoder (same as VoxelEncoder3D):
      [B, 1, 64^3] -> C2[32, 32^3] -> C3[64, 16^3] -> C4[128, 8^3] -> C5[D, 4^3]

    Top-down FPN + C2 fusion:
      P5 = C5                                                         [D, 4^3]
      P4 = smooth(upsample(P5) + lateral(C4))                         [D, 8^3]
      P3 = smooth(upsample(P4) + lateral(C3) + pool(c2_proj(C2)))     [D, 16^3]
      P2 = smooth(upsample(P3) + c2_proj(C2))                         [D, 32^3]  (only if include_level_32)

    Key: C2 (32^3) is projected once via c2_proj, then:
    - Always pooled 2x and fused into P3, giving 16^3 features 32^3 spatial detail
    - Optionally used to build P2 tokens when include_level_32=True

    Args:
        d_model: Output feature dimension (default: 256)
        include_level_32: Include 32^3 tokens in output (adds 32768 tokens).
            Default False = only output P3(16^3)+P4(8^3)+P5(4^3) = 4672 tokens.
            P3 still contains 32^3 info via pooled C2 fusion.
    """
    def __init__(self, d_model=256, include_level_32=False):
        super().__init__()
        self.d_model = d_model
        self.include_level_32 = include_level_32

        # Bottom-up encoder stages
        self.stage1 = nn.Sequential(  # 64 -> 32
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResBlock3d(32),
        )
        self.stage2 = nn.Sequential(  # 32 -> 16
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock3d(64),
        )
        self.stage3 = nn.Sequential(  # 16 -> 8
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock3d(128),
        )
        self.stage4 = nn.Sequential(  # 8 -> 4
            nn.Conv3d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            ResBlock3d(d_model),
        )

        # FPN lateral connections (1x1 conv to project to d_model)
        self.lateral_c4 = nn.Sequential(
            nn.Conv3d(128, d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
        )
        self.lateral_c3 = nn.Sequential(
            nn.Conv3d(64, d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
        )

        # C2 projection: used for BOTH P3 fusion (pooled) and P2 pathway
        self.c2_proj = nn.Sequential(
            nn.Conv3d(32, d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
        )

        # FPN smooth convolutions (3x3 to reduce upsampling aliasing)
        self.smooth_p4 = nn.Sequential(
            nn.Conv3d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv3d(d_model, d_model, kernel_size=3, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
        )

        if include_level_32:
            self.smooth_p2 = nn.Sequential(
                nn.Conv3d(d_model, d_model, kernel_size=3, padding=1),
                nn.GroupNorm(8, d_model),
                nn.GELU(),
            )

    def forward(self, x):
        """
        Args:
            x: [B, 1, 64, 64, 64] binary voxel grid
        Returns:
            If include_level_32=False: (p3_tokens, p4_tokens, p5_tokens)
                p3_tokens: [B, 4096, D]  (16^3, enriched with 32^3 + 8^3 + 4^3 info)
                p4_tokens: [B, 512, D]   (8^3, enriched with 4^3 info)
                p5_tokens: [B, 64, D]    (4^3)
            If include_level_32=True: (p2_tokens, p3_tokens, p4_tokens, p5_tokens)
                p2_tokens: [B, 32768, D] (32^3, enriched)
                + above
        """
        B = x.shape[0]

        # Bottom-up
        c2 = self.stage1(x)   # [B, 32, 32, 32, 32]
        c3 = self.stage2(c2)  # [B, 64, 16, 16, 16]
        c4 = self.stage3(c3)  # [B, 128, 8, 8, 8]
        c5 = self.stage4(c4)  # [B, D, 4, 4, 4]

        # Project C2 once (reused for P3 fusion and optionally P2)
        c2_proj = self.c2_proj(c2)  # [B, D, 32, 32, 32]

        # Top-down FPN
        p5 = c5  # [B, D, 4, 4, 4]

        p4 = self.smooth_p4(
            F.interpolate(p5, size=c4.shape[2:], mode='trilinear', align_corners=False)
            + self.lateral_c4(c4)
        )  # [B, D, 8, 8, 8]

        # Fuse C2 (32^3) into P3 by pooling to 16^3
        c2_pooled = F.avg_pool3d(c2_proj, kernel_size=2, stride=2)  # [B, D, 16, 16, 16]
        p3 = self.smooth_p3(
            F.interpolate(p4, size=c3.shape[2:], mode='trilinear', align_corners=False)
            + self.lateral_c3(c3)
            + c2_pooled
        )  # [B, D, 16, 16, 16]

        # Flatten to tokens: [B, C, X, Y, Z] -> [B, XYZ, C]
        def to_tokens(feat):
            return feat.flatten(2).permute(0, 2, 1)

        p5_tokens = to_tokens(p5)  # [B, 64, D]
        p4_tokens = to_tokens(p4)  # [B, 512, D]
        p3_tokens = to_tokens(p3)  # [B, 4096, D]

        if self.include_level_32:
            p2 = self.smooth_p2(
                F.interpolate(p3, size=c2.shape[2:], mode='trilinear', align_corners=False)
                + c2_proj
            )  # [B, D, 32, 32, 32]
            p2_tokens = to_tokens(p2)  # [B, 32768, D]
            return p2_tokens, p3_tokens, p4_tokens, p5_tokens
        else:
            return p3_tokens, p4_tokens, p5_tokens


def sinusoidal_positional_encoding_3d(resolution, d_model):
    """
    Generate 3D sinusoidal positional encoding.

    Args:
        resolution: Grid resolution (e.g., 4 for 4^3 grid)
        d_model: Feature dimension

    Returns:
        [resolution^3, d_model] positional encoding
    """
    pe = torch.zeros(resolution ** 3, d_model)

    coords = torch.stack(torch.meshgrid(
        torch.arange(resolution),
        torch.arange(resolution),
        torch.arange(resolution),
        indexing='ij'
    ), dim=-1).reshape(-1, 3).float()

    coords = 2.0 * coords / (resolution - 1) - 1.0

    d_per_axis = d_model // 3
    remainder = d_model - 3 * d_per_axis

    for axis in range(3):
        d = d_per_axis + (1 if axis < remainder else 0)
        offset = axis * d_per_axis + min(axis, remainder)

        div_term = torch.exp(
            torch.arange(0, d, 2).float() * -(math.log(10000.0) / d)
        )

        pe[:, offset:offset + d:2] = torch.sin(coords[:, axis:axis+1] * div_term)
        if d > 1:
            pe[:, offset + 1:offset + d:2] = torch.cos(coords[:, axis:axis+1] * div_term[:d // 2])

    return pe


class BBoxEstimator(nn.Module):
    """
    3D CNN + DETR-style bounding box estimator.

    Takes a binary voxel grid [B, 1, 64, 64, 64] (decoded from Stage 1) and
    predicts OBBs for all furniture items. A 3D CNN extracts multi-scale spatial
    features, then DETR object queries cross-attend to these features.

    Args:
        voxel_resolution: Input voxel grid resolution (default: 64)
        d_model: Hidden dimension for transformer (default: 256)
        nhead: Number of attention heads (default: 8)
        num_decoder_layers: Number of DETR decoder layers (default: 6)
        dim_feedforward: FFN intermediate dimension (default: 1024)
        num_queries: Maximum number of object queries (default: 50)
        dropout: Dropout rate (default: 0.1)
        num_categories: Number of furniture categories (0 = no category prediction)
        multiscale: Use multi-scale features (16^3+8^3+4^3 = 4672 tokens) instead of single-scale (4^3 = 64 tokens)
        use_fpn: Use FPN encoder (v4). Overrides multiscale.
        include_level_32: Include 32^3 tokens in DETR memory (only with use_fpn=True).
            Adds 32768 tokens; requires reduced batch_size.
    """
    def __init__(
        self,
        voxel_resolution: int = 64,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        num_queries: int = 50,
        dropout: float = 0.1,
        num_categories: int = 0,
        multiscale: bool = False,
        use_fpn: bool = False,
        include_level_32: bool = False,
    ):
        super().__init__()
        self.voxel_resolution = voxel_resolution
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_categories = num_categories
        self.multiscale = multiscale
        self.use_fpn = use_fpn
        self.include_level_32 = include_level_32

        # Choose encoder
        if use_fpn:
            self.voxel_encoder = VoxelEncoderFPN3D(
                d_model=d_model,
                include_level_32=include_level_32,
            )
            # FPN outputs are already d_model channels — just need LayerNorm + PE + scale embed
            if include_level_32:
                n_levels = 4  # 32^3 + 16^3 + 8^3 + 4^3
                resolutions = [32, 16, 8, 4]
            else:
                n_levels = 3  # 16^3 + 8^3 + 4^3
                resolutions = [16, 8, 4]

            self.fpn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_levels)])
            self.scale_embed = nn.Embedding(n_levels, d_model)
            for i, res in enumerate(resolutions):
                pe = sinusoidal_positional_encoding_3d(res, d_model)
                self.register_buffer(f'fpn_pos_embed_{i}', pe.unsqueeze(0))

        elif multiscale:
            self.voxel_encoder = VoxelEncoder3D(d_model=d_model, multiscale=True)
            self.proj_16 = nn.Sequential(nn.Linear(64, d_model), nn.LayerNorm(d_model))
            self.proj_8 = nn.Sequential(nn.Linear(128, d_model), nn.LayerNorm(d_model))
            self.proj_4 = nn.LayerNorm(d_model)
            self.scale_embed = nn.Embedding(3, d_model)
            pe_16 = sinusoidal_positional_encoding_3d(voxel_resolution // 4, d_model)
            pe_8 = sinusoidal_positional_encoding_3d(voxel_resolution // 8, d_model)
            pe_4 = sinusoidal_positional_encoding_3d(voxel_resolution // 16, d_model)
            self.register_buffer('pos_embed_16', pe_16.unsqueeze(0))
            self.register_buffer('pos_embed_8', pe_8.unsqueeze(0))
            self.register_buffer('pos_embed_4', pe_4.unsqueeze(0))
        else:
            self.voxel_encoder = VoxelEncoder3D(d_model=d_model, multiscale=False)
            feature_resolution = voxel_resolution // 16
            pe = sinusoidal_positional_encoding_3d(feature_resolution, d_model)
            self.register_buffer('pos_embed', pe.unsqueeze(0))

        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # DETR decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Prediction heads
        self.center_head = MLP(d_model, d_model, 3, num_layers=3)
        self.size_head = MLP(d_model, d_model, 3, num_layers=3)
        self.rotation_head = MLP(d_model, d_model, 2, num_layers=3)  # sin, cos
        self.confidence_head = nn.Linear(d_model, 1)

        if num_categories > 0:
            self.category_head = nn.Linear(d_model, num_categories)

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.confidence_head.bias, -4.6)
        nn.init.normal_(self.query_embed.weight, std=0.02)
        for head in [self.center_head, self.size_head, self.rotation_head]:
            nn.init.xavier_uniform_(head.net[-1].weight, gain=0.01)
            nn.init.zeros_(head.net[-1].bias)

        # Initialize FPN lateral/smooth layers
        if self.use_fpn:
            for name, m in self.voxel_encoder.named_modules():
                if isinstance(m, nn.Conv3d) and ('lateral' in name or 'smooth' in name):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _apply_heads(self, hs):
        """Apply prediction heads to decoder hidden states."""
        pred_centers = self.center_head(hs).sigmoid() - 0.5  # [-0.5, 0.5]
        pred_sizes = self.size_head(hs).sigmoid() * 0.5  # [0, 0.5]
        pred_rot_sincos = self.rotation_head(hs)  # [B, Q, 2]
        pred_rotations = torch.atan2(
            pred_rot_sincos[..., 0:1],
            pred_rot_sincos[..., 1:2]
        )  # [B, Q, 1]
        pred_confidences = self.confidence_head(hs).sigmoid()  # [B, Q, 1]

        output = {
            'pred_centers': pred_centers,
            'pred_sizes': pred_sizes,
            'pred_rotations': pred_rotations,
            'pred_rot_sincos': pred_rot_sincos,
            'pred_confidences': pred_confidences,
        }
        if self.num_categories > 0:
            output['pred_categories'] = self.category_head(hs)
        return output

    def _build_memory(self, voxel_grid):
        """Build DETR memory from voxel grid using the configured encoder."""
        B = voxel_grid.shape[0]

        if self.use_fpn:
            feat_levels = self.voxel_encoder(voxel_grid)
            # Each level: [B, N_i, D] — already d_model channels from FPN
            mem_parts = []
            for i, feat in enumerate(feat_levels):
                pe = getattr(self, f'fpn_pos_embed_{i}')
                mem = self.fpn_norms[i](feat) + pe + self.scale_embed.weight[i]
                mem_parts.append(mem)
            memory = torch.cat(mem_parts, dim=1)

        elif self.multiscale:
            feat_16, feat_8, feat_4 = self.voxel_encoder(voxel_grid)
            mem_16 = self.proj_16(feat_16) + self.pos_embed_16 + self.scale_embed.weight[0]
            mem_8 = self.proj_8(feat_8) + self.pos_embed_8 + self.scale_embed.weight[1]
            mem_4 = self.proj_4(feat_4) + self.pos_embed_4 + self.scale_embed.weight[2]
            memory = torch.cat([mem_16, mem_8, mem_4], dim=1)

        else:
            memory = self.voxel_encoder(voxel_grid)
            memory = memory + self.pos_embed

        return memory

    def forward(self, voxel_grid):
        """
        Args:
            voxel_grid: [B, 1, 64, 64, 64] binary occupancy grid

        Returns:
            dict with:
                pred_centers: [B, Q, 3] in [-0.5, 0.5]
                pred_sizes: [B, Q, 3] positive values
                pred_rotations: [B, Q, 1] angle in radians
                pred_rot_sincos: [B, Q, 2] sin/cos of rotation
                pred_confidences: [B, Q, 1] in [0, 1]
                aux_outputs: list of dicts (intermediate decoder layer predictions)
        """
        B = voxel_grid.shape[0]

        # Build memory from encoder
        memory = self._build_memory(voxel_grid)

        # Object queries: [B, Q, D]
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Manual decoder loop to collect intermediate outputs
        hs = queries
        intermediates = []
        for layer in self.decoder.layers:
            hs = layer(hs, memory)
            intermediates.append(hs)

        # Apply norm to final layer
        if self.decoder.norm is not None:
            hs = self.decoder.norm(hs)
            intermediates[-1] = hs

        # Final layer predictions (top-level keys for backward compatibility)
        output = self._apply_heads(hs)

        # Auxiliary outputs from intermediate layers (all except last)
        output['aux_outputs'] = [self._apply_heads(h) for h in intermediates[:-1]]

        return output
