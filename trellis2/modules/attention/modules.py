# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license
#
# Modified from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
from .rope import RotaryPositionEmbedder


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)
    

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: Tuple[float, float] = (1.0, 10000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias) # channels: 1536 (query channels), ctx_channels: 1024 (context channels)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, phases: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [N, L, C] input tensor
            context: [N, L_ctx, C_ctx] context tensor for cross-attention
            phases: [L, C] RoPE phases
            attn_mask: [N, L, L_ctx] or [N, H, L, L_ctx] attention mask for cross-attention.
                      Values of 0 mean attend, large negative values mean don't attend.
        """
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)

            if self.attn_mode == "full":
                if self.qk_rms_norm or self.use_rope:
                    q, k, v = qkv.unbind(dim=2)
                    if self.qk_rms_norm:
                        q = self.q_rms_norm(q)
                        k = self.k_rms_norm(k)
                    if self.use_rope:
                        assert phases is not None, "Phases must be provided for RoPE"
                        q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                        k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
                    h = scaled_dot_product_attention(q, k, v)
                else:
                    h = scaled_dot_product_attention(qkv)
            elif self.attn_mode == "windowed":
                raise NotImplementedError("Windowed attention is not yet implemented")
        else:
            Lkv = context.shape[1] # 1029
            q = self.to_q(x) # [1, 4096, 1536]
            kv = self.to_kv(context) # kv: [1, 1029, 3072], context: [1, 1029, 1024]
            q = q.reshape(B, L, self.num_heads, -1) # [1, 4096, 12, 128]
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1) # [1, 1029, 2, 12, 128]
            if self.qk_rms_norm: # True
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=2)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            else:
                h = scaled_dot_product_attention(q, kv, attn_mask=attn_mask) #NOTE:
        h = h.reshape(B, L, -1) # [1, 4096, 12, 128] => [1, 4096, 1536]
        h = self.to_out(h)
        return h # [1, 4096, 1536] # [4, 4096, 1536]
