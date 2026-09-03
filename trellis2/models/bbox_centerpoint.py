# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
CenterPoint-style 3D Bounding Box Estimator (v2).

Uses a 3D U-Net to produce per-voxel predictions:
- Center heatmap: Gaussian peaks at object centers
- Offset: subvoxel center refinement
- Size: bbox extents
- Rotation: multi-bin classification + residual regression (v2)

Input: binary voxel grid [B, 1, 64, 64, 64]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3d(nn.Module):
    """3D residual block with GroupNorm."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class DownBlock(nn.Module):
    """Downsample + ResBlock."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()
        self.res = ResBlock3d(out_ch)

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return self.res(x)


class UpBlock(nn.Module):
    """Upsample + concat skip + Conv + ResBlock."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()
        self.res = ResBlock3d(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm(self.conv(x)))
        return self.res(x)


class PredictionHead(nn.Module):
    """Two-layer Conv3d prediction head."""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, mid_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, mid_ch), mid_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv3d(mid_ch, out_ch, 1)

    def forward(self, x):
        return self.conv2(self.act(self.norm(self.conv1(x))))


def _iou_nms_3d(centers, sizes, scores, iou_threshold=0.3):
    """
    IoU-based NMS for 3D axis-aligned bounding boxes.

    Args:
        centers: [K, 3] bbox centers
        sizes: [K, 3] bbox extents (full width)
        scores: [K] confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: list of indices to keep
    """
    if len(centers) == 0:
        return []

    # Convert to min/max format
    half = sizes / 2
    mins = centers - half  # [K, 3]
    maxs = centers + half  # [K, 3]

    # Sort by score descending
    order = scores.argsort(descending=True)

    keep = []
    while len(order) > 0:
        i = order[0].item()
        keep.append(i)
        if len(order) == 1:
            break

        # Compute IoU of current box with remaining
        remaining = order[1:]
        inter_min = torch.max(mins[i].unsqueeze(0), mins[remaining])  # [N, 3]
        inter_max = torch.min(maxs[i].unsqueeze(0), maxs[remaining])  # [N, 3]
        inter_size = (inter_max - inter_min).clamp(min=0)  # [N, 3]
        inter_vol = inter_size[:, 0] * inter_size[:, 1] * inter_size[:, 2]

        vol_i = sizes[i, 0] * sizes[i, 1] * sizes[i, 2]
        vol_rem = sizes[remaining, 0] * sizes[remaining, 1] * sizes[remaining, 2]
        union_vol = vol_i + vol_rem - inter_vol

        iou = inter_vol / (union_vol + 1e-8)

        # Keep boxes with IoU below threshold
        mask = iou <= iou_threshold
        order = remaining[mask]

    return keep


class BBoxCenterPoint(nn.Module):
    """
    CenterPoint-style 3D bounding box estimator (v2).

    Uses a 3D U-Net backbone to produce dense per-voxel predictions.
    At inference, local maxima of the center heatmap yield bbox detections.

    v2 changes:
    - Multi-bin rotation: num_rot_bins classification + 1 residual per bin
    - IoU-based NMS post-processing to eliminate duplicate boxes

    Args:
        voxel_resolution: Input voxel grid resolution (default: 64)
        base_channels: Base channel count for U-Net (default: 32)
        max_detections: Maximum number of detections at inference (default: 50)
        num_rot_bins: Number of rotation bins (default: 12, i.e., 30 deg each)
    """
    def __init__(
        self,
        voxel_resolution: int = 64,
        base_channels: int = 32,
        max_detections: int = 50,
        num_rot_bins: int = 12,
    ):
        super().__init__()
        self.voxel_resolution = voxel_resolution
        self.max_detections = max_detections
        self.num_rot_bins = num_rot_bins
        C = base_channels

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, C, 3, padding=1),
            nn.GroupNorm(min(8, C), C),
            nn.GELU(),
            ResBlock3d(C),
        )  # [C, 64, 64, 64]
        self.enc2 = DownBlock(C, C * 2)      # [2C, 32, 32, 32]
        self.enc3 = DownBlock(C * 2, C * 4)  # [4C, 16, 16, 16]
        self.enc4 = DownBlock(C * 4, C * 8)  # [8C, 8, 8, 8]

        # Bottleneck
        self.bottleneck = ResBlock3d(C * 8)

        # Decoder
        self.dec3 = UpBlock(C * 8, C * 4, C * 4)  # [4C, 16, 16, 16]
        self.dec2 = UpBlock(C * 4, C * 2, C * 2)  # [2C, 32, 32, 32]
        self.dec1 = UpBlock(C * 2, C, C)            # [C, 64, 64, 64]

        # Prediction heads (all at full resolution)
        self.heatmap_head = PredictionHead(C, C, 1)     # center heatmap
        self.offset_head = PredictionHead(C, C, 3)      # subvoxel offset
        self.size_head = PredictionHead(C, C, 3)         # bbox size

        # v2: Multi-bin rotation head
        # num_rot_bins for bin classification + num_rot_bins for per-bin residual
        self.rotation_bin_head = PredictionHead(C, C, num_rot_bins)      # bin logits
        self.rotation_res_head = PredictionHead(C, C, num_rot_bins)      # per-bin residual

        # Bin centers: evenly spaced in [-pi, pi)
        bin_size = 2 * math.pi / num_rot_bins
        bin_centers = torch.arange(num_rot_bins).float() * bin_size - math.pi + bin_size / 2
        self.register_buffer('bin_centers', bin_centers)  # [num_rot_bins]
        self.bin_size = bin_size

        self._init_weights()

    def _init_weights(self):
        # Initialize heatmap bias so initial predictions are low confidence
        # sigmoid(-4.6) ~ 0.01
        nn.init.constant_(self.heatmap_head.conv2.bias, -4.6)

    def forward(self, voxel_grid):
        """
        Args:
            voxel_grid: [B, 1, R, R, R] binary occupancy grid

        Returns:
            dict with dense predictions:
                heatmap: [B, 1, R, R, R] center heatmap (sigmoid applied)
                offset: [B, 3, R, R, R] subvoxel center offset
                size: [B, 3, R, R, R] bbox extents (positive)
                rot_bin_logits: [B, num_rot_bins, R, R, R] bin classification logits
                rot_bin_res: [B, num_rot_bins, R, R, R] per-bin residual
        """
        # Encoder
        e1 = self.enc1(voxel_grid)  # [B, C, 64, 64, 64]
        e2 = self.enc2(e1)          # [B, 2C, 32, 32, 32]
        e3 = self.enc3(e2)          # [B, 4C, 16, 16, 16]
        e4 = self.enc4(e3)          # [B, 8C, 8, 8, 8]

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d3 = self.dec3(b, e3)   # [B, 4C, 16, 16, 16]
        d2 = self.dec2(d3, e2)  # [B, 2C, 32, 32, 32]
        d1 = self.dec1(d2, e1)  # [B, C, 64, 64, 64]

        # Prediction heads
        heatmap_logits = self.heatmap_head(d1)        # [B, 1, R, R, R] raw logits
        heatmap = heatmap_logits.float().sigmoid()     # [B, 1, R, R, R] for inference/viz
        offset = self.offset_head(d1)               # [B, 3, R, R, R]
        size = self.size_head(d1).float().clamp(max=0.0).exp()  # positive, bounded (clamp before exp to prevent overflow in bf16)
        rot_bin_logits = self.rotation_bin_head(d1)  # [B, num_rot_bins, R, R, R]
        rot_bin_res = self.rotation_res_head(d1)     # [B, num_rot_bins, R, R, R]

        return {
            'heatmap': heatmap,
            'heatmap_logits': heatmap_logits,
            'offset': offset,
            'size': size,
            'rot_bin_logits': rot_bin_logits,
            'rot_bin_res': rot_bin_res,
        }

    def decode_rotation(self, rot_bin_logits, rot_bin_res):
        """
        Decode multi-bin rotation to angle.

        Args:
            rot_bin_logits: [..., num_rot_bins] bin classification logits
            rot_bin_res: [..., num_rot_bins] per-bin residual

        Returns:
            angle: [...] in [-pi, pi]
        """
        # Select best bin
        best_bin = rot_bin_logits.argmax(dim=-1)  # [...]

        # Gather the residual for the best bin
        res = torch.gather(rot_bin_res, dim=-1, index=best_bin.unsqueeze(-1)).squeeze(-1)

        # Clamp residual to half bin size
        half_bin = self.bin_size / 2
        res = res.tanh() * half_bin

        # Final angle = bin center + residual
        angle = self.bin_centers[best_bin] + res
        # Wrap to [-pi, pi]
        angle = torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
        return angle

    @torch.no_grad()
    def decode_detections(self, outputs, score_threshold=0.3, nms_kernel=3, iou_nms_threshold=0.3):
        """
        Post-process dense predictions into a list of bounding boxes.

        v2: Added IoU-based NMS after max_pool peak detection.

        Args:
            outputs: dict from forward()
            score_threshold: minimum heatmap score
            nms_kernel: kernel size for 3D NMS via max_pool3d
            iou_nms_threshold: IoU threshold for bbox-level NMS

        Returns:
            List of dicts (one per batch), each with:
                pred_centers: [K, 3] in [-0.5, 0.5]
                pred_sizes: [K, 3]
                pred_rotations: [K, 1] radians
                pred_confidences: [K, 1]
        """
        heatmap = outputs['heatmap']            # [B, 1, R, R, R]
        offset = outputs['offset']               # [B, 3, R, R, R]
        size = outputs['size']                   # [B, 3, R, R, R]
        rot_bin_logits = outputs['rot_bin_logits']  # [B, num_rot_bins, R, R, R]
        rot_bin_res = outputs['rot_bin_res']        # [B, num_rot_bins, R, R, R]

        B, _, R, _, _ = heatmap.shape
        results = []

        for b in range(B):
            hm = heatmap[b, 0]  # [R, R, R]

            # 3D NMS: keep only local maxima
            pad = nms_kernel // 2
            hm_max = F.max_pool3d(
                hm.unsqueeze(0).unsqueeze(0),
                kernel_size=nms_kernel, stride=1, padding=pad
            ).squeeze(0).squeeze(0)
            peaks = (hm == hm_max) & (hm >= score_threshold)

            # Get peak locations
            peak_indices = peaks.nonzero(as_tuple=False)  # [K, 3]
            if len(peak_indices) == 0:
                results.append({
                    'pred_centers': torch.zeros(0, 3, device=hm.device),
                    'pred_sizes': torch.zeros(0, 3, device=hm.device),
                    'pred_rotations': torch.zeros(0, 1, device=hm.device),
                    'pred_confidences': torch.zeros(0, 1, device=hm.device),
                })
                continue

            # Sort by score, keep top-K candidates (generous limit before IoU NMS)
            scores = hm[peak_indices[:, 0], peak_indices[:, 1], peak_indices[:, 2]]
            max_candidates = self.max_detections * 3  # allow more candidates for IoU NMS
            if len(scores) > max_candidates:
                topk = scores.topk(max_candidates)
                peak_indices = peak_indices[topk.indices]
                scores = topk.values

            ix, iy, iz = peak_indices[:, 0], peak_indices[:, 1], peak_indices[:, 2]

            # Compute centers: voxel index -> normalized coords + subvoxel offset
            off = offset[b, :, ix, iy, iz].T  # [K, 3]
            centers = (peak_indices.float() + 0.5 + off) / R - 0.5  # [-0.5, 0.5]

            # Size
            sz = size[b, :, ix, iy, iz].T  # [K, 3]

            # Rotation: multi-bin decode
            rbl = rot_bin_logits[b, :, ix, iy, iz].T  # [K, num_rot_bins]
            rbr = rot_bin_res[b, :, ix, iy, iz].T      # [K, num_rot_bins]
            rot_angle = self.decode_rotation(rbl, rbr)  # [K]

            # IoU-based NMS to remove overlapping boxes
            keep = _iou_nms_3d(centers, sz, scores, iou_threshold=iou_nms_threshold)
            if len(keep) == 0:
                results.append({
                    'pred_centers': torch.zeros(0, 3, device=hm.device),
                    'pred_sizes': torch.zeros(0, 3, device=hm.device),
                    'pred_rotations': torch.zeros(0, 1, device=hm.device),
                    'pred_confidences': torch.zeros(0, 1, device=hm.device),
                })
                continue

            keep = keep[:self.max_detections]  # limit after NMS
            keep_t = torch.tensor(keep, device=hm.device, dtype=torch.long)

            results.append({
                'pred_centers': centers[keep_t],
                'pred_sizes': sz[keep_t],
                'pred_rotations': rot_angle[keep_t].unsqueeze(-1),
                'pred_confidences': scores[keep_t].unsqueeze(-1),
            })

        return results
