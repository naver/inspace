# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
CenterPoint-style 3D Bounding Box Estimation Trainer (v2).

Uses dense per-voxel supervision:
- Center heatmap: Gaussian focal loss
- Offset/Size: L1 loss at GT center voxels only
- Rotation: multi-bin classification + residual regression (v2)
  - Dense supervision within Gaussian radius (v2)
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils

from .basic import BasicTrainer
from ..utils.bbox_loss import bbox3d_iou

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_3d_gaussian_heatmap(gt_bboxes, gt_mask, resolution=64, min_sigma=2.0, max_sigma=4.0):
    """
    Generate 3D Gaussian heatmap from GT bounding boxes.

    For each GT object, places a Gaussian blob at its center voxel.
    Sigma is proportional to object size (larger objects -> larger blobs).

    v2: increased default min_sigma (1.0->2.0) and max_sigma (3.0->4.0) for
    broader Gaussians that reduce spurious peaks.

    Also returns dense rotation targets for all voxels within Gaussian radius
    (not just center voxels), enabling dense rotation supervision.

    Args:
        gt_bboxes: [M, 7] (cx, cy, cz, sx, sy, sz, rotation)
        gt_mask: [M] bool
        resolution: Voxel grid resolution (default 64)
        min_sigma: Minimum Gaussian sigma in voxel units
        max_sigma: Maximum Gaussian sigma in voxel units

    Returns:
        heatmap: [1, R, R, R] float32 heatmap in [0, 1]
        center_indices: [N, 3] int64 voxel indices of GT centers
        gt_offsets: [N, 3] subvoxel offsets for each GT center
        gt_sizes: [N, 3] bbox extents
        gt_rotations: [N] rotation angle in radians
        dense_rot_map: [1, R, R, R] rotation angle per voxel (from nearest GT)
        dense_rot_weight: [1, R, R, R] Gaussian weight for dense rotation supervision
    """
    R = resolution
    n_gt = gt_mask.sum().item()

    heatmap = torch.zeros(1, R, R, R, dtype=torch.float32)
    dense_rot_map = torch.zeros(1, R, R, R, dtype=torch.float32)
    dense_rot_weight = torch.zeros(1, R, R, R, dtype=torch.float32)
    center_indices = []
    gt_offsets = []
    gt_sizes = []
    gt_rotations = []

    if n_gt == 0:
        return (heatmap,
                torch.zeros(0, 3, dtype=torch.long),
                torch.zeros(0, 3),
                torch.zeros(0, 3),
                torch.zeros(0),
                dense_rot_map,
                dense_rot_weight)

    for i in range(n_gt):
        if not gt_mask[i]:
            continue

        cx, cy, cz = gt_bboxes[i, 0].item(), gt_bboxes[i, 1].item(), gt_bboxes[i, 2].item()
        sx, sy, sz = gt_bboxes[i, 3].item(), gt_bboxes[i, 4].item(), gt_bboxes[i, 5].item()
        rot = gt_bboxes[i, 6].item()

        # Convert center from [-0.5, 0.5] to voxel coordinates [0, R)
        vx = (cx + 0.5) * R
        vy = (cy + 0.5) * R
        vz = (cz + 0.5) * R

        # Integer voxel index (clamped)
        ix = int(min(max(round(vx - 0.5), 0), R - 1))
        iy = int(min(max(round(vy - 0.5), 0), R - 1))
        iz = int(min(max(round(vz - 0.5), 0), R - 1))

        # Subvoxel offset: how far the true center is from the voxel center
        off_x = vx - (ix + 0.5)
        off_y = vy - (iy + 0.5)
        off_z = vz - (iz + 0.5)

        center_indices.append(torch.tensor([ix, iy, iz], dtype=torch.long))
        gt_offsets.append(torch.tensor([off_x, off_y, off_z]))
        gt_sizes.append(torch.tensor([sx, sy, sz]))
        gt_rotations.append(rot)

        # Gaussian sigma proportional to object size (in voxel units)
        size_voxels = max(sx, sy, sz) * R
        sigma = min(max(size_voxels / 3.0, min_sigma), max_sigma)

        # Generate Gaussian in a local window (3*sigma radius)
        radius = int(math.ceil(3 * sigma))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    nx, ny, nz = ix + dx, iy + dy, iz + dz
                    if 0 <= nx < R and 0 <= ny < R and 0 <= nz < R:
                        dist_sq = dx * dx + dy * dy + dz * dz
                        val = math.exp(-dist_sq / (2 * sigma * sigma))
                        # Heatmap: take max (for overlapping objects)
                        heatmap[0, nx, ny, nz] = max(heatmap[0, nx, ny, nz].item(), val)
                        # Dense rotation: higher Gaussian weight wins
                        if val > dense_rot_weight[0, nx, ny, nz].item():
                            dense_rot_weight[0, nx, ny, nz] = val
                            dense_rot_map[0, nx, ny, nz] = rot

    center_indices = torch.stack(center_indices)
    gt_offsets = torch.stack(gt_offsets)
    gt_sizes = torch.stack(gt_sizes)
    gt_rotations = torch.tensor(gt_rotations, dtype=torch.float32)

    return heatmap, center_indices, gt_offsets, gt_sizes, gt_rotations, dense_rot_map, dense_rot_weight


def gaussian_focal_loss(pred, target, alpha=2.0, beta=4.0, logits=None):
    """
    Modified focal loss for center heatmap (CornerNet / CenterNet style).

    For positive locations (target == 1):
        loss = -(1 - pred)^alpha * log(pred)
    For negative locations (target < 1):
        loss = -(1 - target)^beta * pred^alpha * log(1 - pred)

    When `logits` is provided, uses numerically stable logsigmoid computation
    instead of log(pred)/log(1-pred) which can produce -inf/NaN when pred
    saturates to 0 or 1.

    Args:
        pred: [B, 1, R, R, R] predicted heatmap (after sigmoid), used for focal weights
        target: [B, 1, R, R, R] GT Gaussian heatmap
        logits: [B, 1, R, R, R] raw logits (before sigmoid). If provided, uses
                stable log computation: log(pred) = logsigmoid(logits),
                log(1-pred) = logsigmoid(-logits).

    Returns:
        Scalar loss (mean over batch)
    """
    # Cast to float32
    pred = pred.float()
    target = target.float()

    if logits is not None:
        logits = logits.float()
        # Numerically stable: logsigmoid never produces -inf
        log_pred = F.logsigmoid(logits)        # = log(sigmoid(x))
        log_1m_pred = F.logsigmoid(-logits)     # = log(1 - sigmoid(x))
    else:
        # Fallback: clamp-based (less stable)
        eps = 1e-6
        pred = pred.clamp(eps, 1 - eps)
        log_pred = torch.log(pred)
        log_1m_pred = torch.log(1 - pred)

    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()

    pos_loss = -((1 - pred) ** alpha) * log_pred * pos_mask
    neg_loss = -((1 - target) ** beta) * (pred ** alpha) * log_1m_pred * neg_mask

    # Normalize by number of positive samples
    num_pos = pos_mask.sum()
    if num_pos == 0:
        return neg_loss.sum() / max(pred.shape[0], 1)

    loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss


def angle_to_bin_and_residual(angle, num_bins, bin_size):
    """
    Convert angle to bin index and residual.

    Args:
        angle: [...] angle in radians [-pi, pi]
        num_bins: number of bins
        bin_size: size of each bin in radians

    Returns:
        bin_idx: [...] int64 bin index
        residual: [...] residual within bin [-bin_size/2, bin_size/2]
    """
    # Normalize to [0, 2*pi)
    angle_pos = torch.remainder(angle + math.pi, 2 * math.pi)  # [0, 2*pi)
    bin_idx = (angle_pos / bin_size).long().clamp(0, num_bins - 1)

    # Bin center
    bin_center = bin_idx.float() * bin_size + bin_size / 2  # center in [0, 2*pi)
    residual = angle_pos - bin_center  # residual in [-bin_size/2, bin_size/2]

    return bin_idx, residual


class BBoxCenterPointTrainer(BasicTrainer):
    """
    Trainer for CenterPoint-style 3D bbox estimation (v2).

    Uses dense per-voxel supervision:
    - Gaussian focal loss for center heatmap
    - L1 loss for offset/size at GT center locations only
    - Multi-bin rotation: cross-entropy for bin classification + smooth L1 for residual (v2)
    - Dense rotation supervision at all voxels within Gaussian radius (v2)

    Additional trainer args:
        lambda_heatmap: Weight for heatmap focal loss
        lambda_offset: Weight for subvoxel offset L1 loss
        lambda_size: Weight for size L1 loss
        lambda_rot_bin: Weight for rotation bin classification loss (v2)
        lambda_rot_res: Weight for rotation residual regression loss (v2)
        lambda_dense_rot: Weight for dense rotation supervision (v2)
        dense_rot_weight_threshold: Gaussian weight threshold for dense supervision (v2)
        score_threshold: Confidence threshold for detection at inference
        nms_kernel: Max pool kernel size for 3D NMS
        iou_nms_threshold: IoU threshold for bbox-level NMS (v2)
        min_sigma: Minimum Gaussian sigma for heatmap generation (v2)
        max_sigma: Maximum Gaussian sigma for heatmap generation (v2)
    """
    def __init__(self, *args,
                 lambda_heatmap=1.0,
                 lambda_offset=1.0,
                 lambda_size=5.0,
                 lambda_rot_bin=2.0,
                 lambda_rot_res=2.0,
                 lambda_dense_rot=1.0,
                 dense_rot_weight_threshold=0.1,
                 score_threshold=0.3,
                 nms_kernel=7,
                 confidence_threshold=0.3,
                 iou_nms_threshold=0.3,
                 min_sigma=2.0,
                 max_sigma=4.0,
                 # backward compat: ignore old rotation param
                 lambda_rotation=None,
                 **kwargs):
        self.lambda_heatmap = lambda_heatmap
        self.lambda_offset = lambda_offset
        self.lambda_size = lambda_size
        self.lambda_rot_bin = lambda_rot_bin
        self.lambda_rot_res = lambda_rot_res
        self.lambda_dense_rot = lambda_dense_rot
        self.dense_rot_weight_threshold = dense_rot_weight_threshold
        self.score_threshold = score_threshold
        self.nms_kernel = nms_kernel
        self.confidence_threshold = confidence_threshold
        self.iou_nms_threshold = iou_nms_threshold
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        super().__init__(*args, **kwargs)

    def training_losses(self, voxel_grid, gt_bboxes, gt_mask, **kwargs):
        """
        Compute CenterPoint v2 training losses.

        Args:
            voxel_grid: [B, 1, 64, 64, 64]
            gt_bboxes: [B, M, 7]
            gt_mask: [B, M]

        Returns:
            loss_dict, status_dict
        """
        model = self.training_models['bbox_centerpoint']
        R = model.voxel_resolution
        num_bins = model.num_rot_bins
        bin_size = model.bin_size

        outputs = model(voxel_grid)
        pred_heatmap = outputs['heatmap']         # [B, 1, R, R, R] (after sigmoid)
        pred_heatmap_logits = outputs['heatmap_logits']  # [B, 1, R, R, R] (raw logits)
        pred_offset = outputs['offset']           # [B, 3, R, R, R]
        pred_size = outputs['size']               # [B, 3, R, R, R]
        pred_rot_bin = outputs['rot_bin_logits']  # [B, num_bins, R, R, R]
        pred_rot_res = outputs['rot_bin_res']     # [B, num_bins, R, R, R]

        B = voxel_grid.shape[0]
        device = voxel_grid.device

        # Generate GT heatmaps and regression targets
        all_gt_heatmaps = []
        all_dense_rot_maps = []
        all_dense_rot_weights = []
        total_offset_loss = torch.tensor(0.0, device=device)
        total_size_loss = torch.tensor(0.0, device=device)
        total_rot_bin_loss = torch.tensor(0.0, device=device)
        total_rot_res_loss = torch.tensor(0.0, device=device)
        total_centers = 0

        for b in range(B):
            gt_hm, center_idx, gt_off, gt_sz, gt_rot, dense_rot_map, dense_rot_weight = \
                generate_3d_gaussian_heatmap(
                    gt_bboxes[b], gt_mask[b], resolution=R,
                    min_sigma=self.min_sigma, max_sigma=self.max_sigma,
                )
            all_gt_heatmaps.append(gt_hm)
            all_dense_rot_maps.append(dense_rot_map)
            all_dense_rot_weights.append(dense_rot_weight)

            n_centers = center_idx.shape[0]
            if n_centers == 0:
                continue

            total_centers += n_centers
            ix, iy, iz = center_idx[:, 0], center_idx[:, 1], center_idx[:, 2]

            # Offset loss at GT centers
            pred_off_at_centers = pred_offset[b, :, ix, iy, iz].T  # [N, 3]
            total_offset_loss = total_offset_loss + F.l1_loss(
                pred_off_at_centers, gt_off.to(device), reduction='sum'
            )

            # Size loss at GT centers
            pred_sz_at_centers = pred_size[b, :, ix, iy, iz].T  # [N, 3]
            total_size_loss = total_size_loss + F.l1_loss(
                pred_sz_at_centers, gt_sz.to(device), reduction='sum'
            )

            # Rotation bin classification + residual at GT centers
            pred_bin_at_centers = pred_rot_bin[b, :, ix, iy, iz].T  # [N, num_bins]
            pred_res_at_centers = pred_rot_res[b, :, ix, iy, iz].T  # [N, num_bins]

            gt_rot_dev = gt_rot.to(device)  # [N]
            gt_bin_idx, gt_residual = angle_to_bin_and_residual(gt_rot_dev, num_bins, bin_size)

            # Bin classification: cross-entropy
            total_rot_bin_loss = total_rot_bin_loss + F.cross_entropy(
                pred_bin_at_centers, gt_bin_idx, reduction='sum'
            )

            # Residual regression: smooth L1 only for the GT bin
            pred_res_for_gt_bin = torch.gather(
                pred_res_at_centers, dim=1, index=gt_bin_idx.unsqueeze(1)
            ).squeeze(1)  # [N]
            # Target residual normalized to [-1, 1] (divided by half_bin)
            half_bin = bin_size / 2
            gt_residual_norm = gt_residual / half_bin  # [-1, 1]
            pred_res_tanh = pred_res_for_gt_bin.tanh()  # model outputs through tanh
            total_rot_res_loss = total_rot_res_loss + F.smooth_l1_loss(
                pred_res_tanh, gt_residual_norm, reduction='sum'
            )

        # Stack GT heatmaps
        gt_heatmap = torch.stack(all_gt_heatmaps).to(device)  # [B, 1, R, R, R]

        # Heatmap focal loss
        heatmap_loss = gaussian_focal_loss(pred_heatmap, gt_heatmap, logits=pred_heatmap_logits)

        # Normalize regression losses by number of GT centers
        if total_centers > 0:
            offset_loss = total_offset_loss / total_centers
            size_loss = total_size_loss / total_centers
            rot_bin_loss = total_rot_bin_loss / total_centers
            rot_res_loss = total_rot_res_loss / total_centers
        else:
            offset_loss = total_offset_loss
            size_loss = total_size_loss
            rot_bin_loss = total_rot_bin_loss
            rot_res_loss = total_rot_res_loss

        # Dense rotation supervision (v2): supervise rotation at ALL voxels within Gaussian
        dense_rot_map = torch.stack(all_dense_rot_maps).to(device)      # [B, 1, R, R, R]
        dense_rot_weight = torch.stack(all_dense_rot_weights).to(device)  # [B, 1, R, R, R]

        # Mask: only voxels with meaningful Gaussian weight
        dense_mask = dense_rot_weight > self.dense_rot_weight_threshold  # [B, 1, R, R, R]
        n_dense = dense_mask.sum().item()

        if n_dense > 0:
            # Flatten spatial dims for masked indexing
            # Get rotation predictions at dense supervised voxels
            dense_angles = dense_rot_map[dense_mask]  # [N_dense]
            dense_weights = dense_rot_weight[dense_mask]  # [N_dense]

            # Get bin targets for dense supervision
            dense_gt_bin, dense_gt_res = angle_to_bin_and_residual(dense_angles, num_bins, bin_size)

            # Extract predictions at those voxels
            # dense_mask is [B, 1, R, R, R], we need indices for [B, num_bins, R, R, R]
            mask_3d = dense_mask.squeeze(1)  # [B, R, R, R]
            pred_bin_flat = pred_rot_bin.permute(0, 2, 3, 4, 1)[mask_3d]  # [N_dense, num_bins]
            pred_res_flat = pred_rot_res.permute(0, 2, 3, 4, 1)[mask_3d]  # [N_dense, num_bins]

            # Weighted cross-entropy for bin classification
            ce_per_voxel = F.cross_entropy(pred_bin_flat, dense_gt_bin, reduction='none')
            dense_rot_loss = (ce_per_voxel * dense_weights).sum() / dense_weights.sum()
        else:
            dense_rot_loss = torch.tensor(0.0, device=device)

        # Total loss
        loss = (self.lambda_heatmap * heatmap_loss
                + self.lambda_offset * offset_loss
                + self.lambda_size * size_loss
                + self.lambda_rot_bin * rot_bin_loss
                + self.lambda_rot_res * rot_res_loss
                + self.lambda_dense_rot * dense_rot_loss)

        loss_dict = {
            'loss': loss,
            'heatmap_loss': heatmap_loss.detach(),
            'offset_loss': offset_loss.detach(),
            'size_loss': size_loss.detach(),
            'rot_bin_loss': rot_bin_loss.detach(),
            'rot_res_loss': rot_res_loss.detach(),
            'dense_rot_loss': dense_rot_loss.detach(),
        }

        status = {
            'num_gt_centers': total_centers,
            'num_dense_rot_voxels': n_dense,
            'heatmap_max': pred_heatmap.max().item(),
            'heatmap_mean': pred_heatmap.mean().item(),
        }

        return loss_dict, status

    def run_step(self, data_list):
        """Override to flatten loss dict for tqdm display and write log.txt."""
        step_log = super().run_step(data_list)

        if self.is_master:
            flat = {}
            if 'loss' in step_log:
                for k, v in step_log['loss'].items():
                    flat[k] = v
            if 'status' in step_log:
                for k, v in step_log['status'].items():
                    flat[k] = v
            step_log.update(flat)

            if self.step % self.i_print == 0:
                log_path = os.path.join(self.output_dir, 'log.txt')
                log_entry = {k: round(v, 6) if isinstance(v, float) else v
                             for k, v in flat.items()}
                with open(log_path, 'a') as f:
                    f.write(f'step {self.step}: {json.dumps(log_entry)}\n')

        return step_log

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=16, batch_size=4):
        """Visualize GT bboxes from the dataset (init snapshot)."""
        os.makedirs(os.path.join(self.output_dir, 'samples', 'init'), exist_ok=True)

        for prefix, dataset in [('train', self.dataset), ('eval', self.eval_dataset)]:
            if dataset is None:
                continue

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=2,
                shuffle=True,
                collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
            )

            all_vis = []
            sample_count = 0
            for batch in dataloader:
                if sample_count >= num_samples:
                    break
                gt_bboxes = batch['gt_bboxes']
                gt_mask = batch['gt_mask']
                sample_ids = batch['sample_id']
                for i in range(gt_bboxes.shape[0]):
                    if sample_count >= num_samples:
                        break
                    fig = self._visualize_bboxes_topdown(
                        gt_bboxes[i], gt_mask[i],
                        pred_centers=torch.zeros(0, 3),
                        pred_sizes=torch.zeros(0, 3),
                        pred_rots=torch.zeros(0, 1),
                        pred_confs=torch.zeros(0, 1),
                        title=f'[{prefix}] {sample_ids[i]}',
                    )
                    all_vis.append(fig)
                    sample_count += 1

            if all_vis:
                grid = torch.stack(all_vis)
                nrow = int(math.ceil(math.sqrt(len(all_vis))))
                save_path = os.path.join(
                    self.output_dir, 'samples', 'init',
                    f'{prefix}_dataset_gt_bboxes.png'
                )
                vutils.save_image(grid, save_path, nrow=nrow, normalize=False)
                print(f'Saved {prefix} dataset GT bboxes: {save_path}')

    @torch.no_grad()
    def run_snapshot(self, num_samples=16, batch_size=4, verbose=False, **kwargs):
        """
        Generate bbox visualization: GT vs CenterPoint detections.
        """
        model = self.models['bbox_centerpoint']
        model.eval()

        samples = {}

        for prefix, dataset in [('train', self.dataset), ('eval', self.eval_dataset)]:
            if dataset is None:
                continue

            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=2,
                shuffle=True,
                collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None,
            )

            all_vis = []
            heatmap_vis = []
            sample_count = 0

            for batch in dataloader:
                if sample_count >= num_samples:
                    break

                voxel_grid = batch['voxel_grid'].to(self.device)
                gt_bboxes = batch['gt_bboxes'].to(self.device)
                gt_mask = batch['gt_mask'].to(self.device)
                sample_ids = batch['sample_id']

                outputs = model(voxel_grid)
                # Cast to float32 for post-processing (max_pool3d, atan2 don't support bf16)
                outputs = {k: v.float() for k, v in outputs.items()}
                detections = model.decode_detections(
                    outputs,
                    score_threshold=self.score_threshold,
                    nms_kernel=self.nms_kernel,
                    iou_nms_threshold=self.iou_nms_threshold,
                )

                for i in range(voxel_grid.shape[0]):
                    if sample_count >= num_samples:
                        break

                    det = detections[i]
                    title = sample_ids[i] if i < len(sample_ids) else f'{prefix}_{sample_count}'

                    # Bbox comparison visualization
                    fig = self._visualize_bboxes_topdown(
                        gt_bboxes[i].float().cpu(), gt_mask[i].cpu(),
                        det['pred_centers'].cpu(), det['pred_sizes'].cpu(),
                        det['pred_rotations'].cpu(), det['pred_confidences'].cpu(),
                        title=f'[{prefix}] {title}',
                    )
                    all_vis.append(fig)

                    # Heatmap visualization (max projection along Z axis)
                    hm_fig = self._visualize_heatmap_topdown(
                        outputs['heatmap'][i, 0].cpu(),
                        voxel_grid[i, 0].float().cpu(),
                        gt_bboxes[i].float().cpu(), gt_mask[i].cpu(),
                        title=f'[{prefix}] {title}',
                    )
                    heatmap_vis.append(hm_fig)

                    sample_count += 1

            if all_vis:
                grid = torch.stack(all_vis)
                samples[f'{prefix}_bbox_prediction'] = {'value': grid, 'type': 'image'}

            if heatmap_vis:
                grid = torch.stack(heatmap_vis)
                samples[f'{prefix}_heatmap'] = {'value': grid, 'type': 'image'}

        # Compute metrics
        self._compute_and_log_metrics(num_eval=min(200, len(self.dataset)))

        model.train()
        return samples

    def _visualize_heatmap_topdown(self, heatmap, voxel_grid, gt_bboxes, gt_mask, title=''):
        """
        Visualize predicted heatmap via max projection along Z, overlaid on voxel occupancy.

        Args:
            heatmap: [R, R, R] predicted center heatmap
            voxel_grid: [R, R, R] binary occupancy
            gt_bboxes: [M, 7]
            gt_mask: [M]
            title: str

        Returns: [3, H, W] tensor
        """
        # Max projection along Z axis (height) -> [R, R]
        hm_topdown = heatmap.max(dim=2).values  # [R, R]
        vg_topdown = voxel_grid.max(dim=2).values  # [R, R]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=80)

        # Left: voxel occupancy
        axes[0].imshow(vg_topdown.numpy().T, origin='lower', cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Voxel Occupancy (top-down)', fontsize=8)

        # Right: heatmap overlay
        axes[1].imshow(vg_topdown.numpy().T, origin='lower', cmap='gray', vmin=0, vmax=1, alpha=0.3)
        im = axes[1].imshow(hm_topdown.numpy().T, origin='lower', cmap='hot', vmin=0, vmax=1, alpha=0.7)
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Draw GT centers
        n_gt = gt_mask.sum().item()
        R = heatmap.shape[0]
        for i in range(n_gt):
            cx, cy = gt_bboxes[i, 0].item(), gt_bboxes[i, 1].item()
            vx = (cx + 0.5) * R
            vy = (cy + 0.5) * R
            axes[1].plot(vx, vy, 'g+', markersize=8, markeredgewidth=2)

        axes[1].set_title(f'Heatmap + GT centers ({n_gt})', fontsize=8)
        fig.suptitle(title, fontsize=9)
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        tensor = torch.from_numpy(buf).permute(2, 0, 1).float() / 255.0
        return tensor

    def _visualize_bboxes_topdown(self, gt_bboxes, gt_mask, pred_centers,
                                   pred_sizes, pred_rots, pred_confs, title=''):
        """
        Create a top-down (XY plane) visualization of GT vs predicted bboxes.

        Returns: [3, H, W] tensor
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)

        # Draw GT bboxes (green)
        n_gt = gt_mask.sum().item()
        for i in range(n_gt):
            cx, cy, cz, sx, sy, sz, rot = gt_bboxes[i].numpy()
            self._draw_rotated_box(ax, cx, cy, sx, sy, rot,
                                    color='green', alpha=0.5, label='GT' if i == 0 else None)

        # Draw predicted bboxes (red, only above threshold)
        n_pred = 0
        for i in range(pred_centers.shape[0]):
            conf = pred_confs[i, 0].item()
            if conf < self.confidence_threshold:
                continue
            cx, cy = pred_centers[i, 0].item(), pred_centers[i, 1].item()
            sx, sy = pred_sizes[i, 0].item(), pred_sizes[i, 1].item()
            rot = pred_rots[i, 0].item()
            label = f'Pred (conf>={self.confidence_threshold})' if n_pred == 0 else None
            self._draw_rotated_box(ax, cx, cy, sx, sy, rot,
                                    color='red', alpha=0.3, label=label)
            ax.text(cx, cy, f'{conf:.2f}', ha='center', va='center',
                    fontsize=6, color='red')
            n_pred += 1

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.55, 0.55)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\nGT: {n_gt}, Pred: {n_pred}', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        tensor = torch.from_numpy(buf).permute(2, 0, 1).float() / 255.0
        return tensor

    @staticmethod
    def _draw_rotated_box(ax, cx, cy, w, h, angle_rad, color='green', alpha=0.5, label=None):
        """Draw a rotated rectangle on matplotlib axis."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        hw, hh = w / 2, h / 2
        corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        corners_world = []
        for lx, ly in corners_local:
            wx = cx + lx * cos_a - ly * sin_a
            wy = cy + lx * sin_a + ly * cos_a
            corners_world.append([wx, wy])

        polygon = plt.Polygon(corners_world, closed=True,
                               facecolor=color, edgecolor=color,
                               alpha=alpha, linewidth=1.5, label=label)
        ax.add_patch(polygon)

    @torch.no_grad()
    def _compute_and_log_metrics(self, num_eval=200):
        """Compute and log IoU metrics using CenterPoint detections."""
        model = self.models['bbox_centerpoint']
        model.eval()

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=16,
            num_workers=4,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        all_ious = []
        all_center_errors = []
        all_size_errors = []
        all_rot_errors = []
        total_gt = 0
        total_pred = 0
        total_tp_25 = 0
        total_tp_50 = 0

        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_eval:
                break

            voxel_grid = batch['voxel_grid'].to(self.device)
            gt_bboxes = batch['gt_bboxes'].to(self.device)
            gt_mask = batch['gt_mask'].to(self.device)

            outputs = model(voxel_grid)
            # Cast to float32 for post-processing (max_pool3d doesn't support bf16)
            outputs = {k: v.float() for k, v in outputs.items()}
            detections = model.decode_detections(
                outputs,
                score_threshold=self.score_threshold,
                nms_kernel=self.nms_kernel,
                iou_nms_threshold=self.iou_nms_threshold,
            )

            B = voxel_grid.shape[0]
            for b in range(B):
                if sample_count >= num_eval:
                    break
                sample_count += 1

                n_gt = gt_mask[b].sum().item()
                if n_gt == 0:
                    continue

                det = detections[b]
                n_pred = det['pred_centers'].shape[0]

                total_gt += n_gt
                total_pred += n_pred

                if n_pred == 0:
                    continue

                pred_c = det['pred_centers']   # [K, 3]
                pred_s = det['pred_sizes']     # [K, 3]
                pred_r = det['pred_rotations'][:, 0]  # [K]

                gt_c = gt_bboxes[b, :n_gt, :3]
                gt_s = gt_bboxes[b, :n_gt, 3:6]
                gt_r = gt_bboxes[b, :n_gt, 6]

                # IoU matrix (AABB approximation)
                pred_box6 = torch.cat([pred_c, pred_s], dim=-1)
                gt_box6 = torch.cat([gt_c, gt_s], dim=-1)
                iou_mat = bbox3d_iou(pred_box6, gt_box6)  # [K, n_gt]

                # Greedy matching by IoU
                max_iou_per_gt, best_pred_per_gt = iou_mat.max(dim=0)
                all_ious.extend(max_iou_per_gt.cpu().tolist())
                total_tp_25 += (max_iou_per_gt >= 0.25).sum().item()
                total_tp_50 += (max_iou_per_gt >= 0.5).sum().item()

                for j in range(n_gt):
                    pi = best_pred_per_gt[j]
                    if max_iou_per_gt[j] >= 0.25:
                        all_center_errors.append((pred_c[pi] - gt_c[j]).norm().item())
                        all_size_errors.append(F.l1_loss(pred_s[pi], gt_s[j]).item())
                        all_rot_errors.append(
                            torch.abs(torch.remainder(pred_r[pi] - gt_r[j] + math.pi, 2 * math.pi) - math.pi).item()
                        )

        # Log metrics
        metrics = {}
        if total_gt > 0:
            metrics['recall@0.25'] = total_tp_25 / total_gt
            metrics['recall@0.50'] = total_tp_50 / total_gt
        if total_pred > 0:
            metrics['precision@0.25'] = total_tp_25 / total_pred
            metrics['precision@0.50'] = total_tp_50 / total_pred
        if all_ious:
            metrics['mean_iou'] = np.mean(all_ious)
        if all_center_errors:
            metrics['mean_center_error'] = np.mean(all_center_errors)
        if all_size_errors:
            metrics['mean_size_error'] = np.mean(all_size_errors)
        if all_rot_errors:
            metrics['mean_rot_error_deg'] = np.degrees(np.mean(all_rot_errors))

        if self.is_master and hasattr(self, 'writer'):
            for k, v in metrics.items():
                self.writer.add_scalar(f'metrics/{k}', v, self.step)
            print(f'\n[Step {self.step}] Metrics: {metrics}')

        model.train()
