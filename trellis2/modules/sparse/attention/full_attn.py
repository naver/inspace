# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license
#
# Modified from TRELLIS.2 (https://github.com/microsoft/TRELLIS.2)
# Copyright (c) Microsoft Corporation. Licensed under the MIT License.

from typing import *
import torch
import torch.nn.functional as F
from .. import VarLenTensor
from .. import config


__all__ = [
    'sparse_scaled_dot_product_attention',
]


@overload
def sparse_scaled_dot_product_attention(qkv: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (VarLenTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, kv: Union[VarLenTensor, torch.Tensor]) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (VarLenTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, C] dense tensor containing Qs.
        kv (VarLenTensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: VarLenTensor, v: VarLenTensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: VarLenTensor, k: torch.Tensor, v: torch.Tensor) -> VarLenTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: VarLenTensor, v: VarLenTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (VarLenTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (VarLenTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    # Extract attn_mask before counting positional arguments.
    # When provided, cross-attention uses per-group SDPA (supports arbitrary per-token masks).
    # When None, uses the configured backend (flash_attn, xformers, etc.).
    attn_mask = kwargs.pop('attn_mask', None)

    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1: # self-attention
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, VarLenTensor), f"qkv must be a VarLenTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])] # [1807, 196, 37, 1070, 34, 144, 1100, 162, 203, 168, 4580, 1737, 74, 72, 984, 184, 2784, 180, 41, 1305, 281]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C] # qkv.shape = [17143, 3, 12, 128]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, VarLenTensor) and isinstance(kv, (VarLenTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, VarLenTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, VarLenTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3: # cross-attention
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, VarLenTensor) and isinstance(k, (VarLenTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, VarLenTensor) and isinstance(v, VarLenTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, VarLenTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])] # [1807, 196, 37, 1070, 34, 144, 1100, 162, 203, 168, 4580, 1737, 74, 72, 984, 184, 2784, 180, 41, 1305, 281]
            q = q.feats     # [T_Q, H, Ci] # q.shape = [17143, 12, 128]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, VarLenTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    # When attn_mask is provided, use per-group SDPA (supports arbitrary per-token masks).
    # Flash Attention doesn't support arbitrary masks — only block-diagonal via cu_seqlens.
    # attn_mask shape: [total_Q_tokens, max_kv_len], bool (True=attend, False=block).
    if attn_mask is not None:
        assert num_all_args >= 2, "attn_mask only supported for cross-attention (2 or 3 args)"
        if num_all_args == 2:
            k, v = kv.unbind(dim=1)

        num_heads = q.shape[-2]  # q: [T, H, C]
        outputs = []
        q_offset = 0
        kv_offset = 0

        for i in range(len(q_seqlen)): # len(q_seqlen)=21, q_seqlen=[1807, 196, 37, 1070, 34, 144, 1100, 162, 203, 168, 4580, 1737, 74, 72, 984, 184, 2784, 180, 41, 1305, 281]
            nq = q_seqlen[i]
            nkv = kv_seqlen[i]

            q_g = q[q_offset:q_offset+nq]    # [nq, H, C] # [1807, 12, 128]
            k_g = k[kv_offset:kv_offset+nkv]  # [nkv, H, C] # [6174, 12, 128]
            v_g = v[kv_offset:kv_offset+nkv]  # [nkv, H, C] # [6174, 12, 128]

            # Extract mask for this group's Q tokens: [nq, nkv]
            mask_g = attn_mask[q_offset:q_offset+nq, :nkv] # attn_mask.shape = [17143, 6174]

            # Reshape for SDPA: [1, H, N, C]
            q_t = q_g.permute(1, 0, 2).unsqueeze(0)  # [1, H, nq, C] # torch.Size([1, 12, 1807, 128])
            k_t = k_g.permute(1, 0, 2).unsqueeze(0)  # [1, H, nkv, C] # torch.Size([1, 12, 6174, 128])
            v_t = v_g.permute(1, 0, 2).unsqueeze(0)  # [1, H, nkv, C] # torch.Size([1, 12, 6174, 128])
            # Bool mask [1, 1, nq, nkv] broadcasts to [1, H, nq, nkv]
            mask_t = mask_g.unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 1807, 6174])

            out_g = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_t)
            out_g = out_g.squeeze(0).permute(1, 0, 2)  # [nq, H, C] # torch.Size([1807, 12, 128]) # [37, 12, 128]...
            outputs.append(out_g)

            q_offset += nq
            kv_offset += nkv

        out = torch.cat(outputs, dim=0) # torch.Size([17143, 12, 128])

    elif config.ATTN == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif config.ATTN == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    elif config.ATTN == 'flash_attn_3':
        if 'flash_attn_3' not in globals():
            import flash_attn_interface as flash_attn_3
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
            cu_seqlens_kv = cu_seqlens_q.clone()
            max_q_seqlen = max_kv_seqlen = max(q_seqlen)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        elif num_all_args == 3:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        out = flash_attn_3.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen)
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")
    
    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
