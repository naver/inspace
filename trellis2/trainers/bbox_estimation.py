# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Trainer for 3D bounding box estimation.

Extends BasicTrainer with:
- Hungarian matching loss computation
- Bbox visualization in run_snapshot
"""

import os
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils

from .basic import BasicTrainer
from ..utils.bbox_loss import BBoxCriterion, bbox3d_iou

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch


class BBoxEstimationTrainer(BasicTrainer):
    """
    Trainer for 3D bounding box estimation using DETR-style decoder.

    Extends BasicTrainer with:
    - BBoxCriterion loss (Hungarian matching + regression + focal)
    - Top-down bbox visualization in snapshot

    Additional trainer args (passed via config):
        lambda_center, lambda_size, lambda_rotation,
        lambda_confidence, lambda_giou: Loss weights
        cost_center, cost_size, cost_confidence, cost_giou: Matching cost weights
    """
    def __init__(self, *args,
                 lambda_center=5.0,
                 lambda_size=5.0,
                 lambda_rotation=2.0,
                 lambda_confidence=1.0,
                 lambda_giou=2.0,
                 cost_center=5.0,
                 cost_size=5.0,
                 cost_confidence=2.0,
                 cost_giou=2.0,
                 confidence_threshold=0.5,
                 use_diou=False,
                 aux_loss_weight=0.0,
                 **kwargs):
        self.confidence_threshold = confidence_threshold
        self.criterion = BBoxCriterion(
            lambda_center=lambda_center,
            lambda_size=lambda_size,
            lambda_rotation=lambda_rotation,
            lambda_confidence=lambda_confidence,
            lambda_giou=lambda_giou,
            cost_center=cost_center,
            cost_size=cost_size,
            cost_confidence=cost_confidence,
            cost_giou=cost_giou,
            use_diou=use_diou,
            aux_loss_weight=aux_loss_weight,
        )
        super().__init__(*args, **kwargs)

    def training_losses(self, voxel_grid, gt_bboxes, gt_mask, **kwargs):
        """
        Compute training losses for bbox estimation.

        Args:
            voxel_grid: [B, 1, 64, 64, 64] binary occupancy grid
            gt_bboxes: [B, M, 7]
            gt_mask: [B, M]

        Returns:
            loss_dict: dict with 'loss' and components
            status_dict: dict with additional info
        """
        model = self.training_models['bbox_estimator']
        predictions = model(voxel_grid)
        loss_dict = self.criterion(predictions, gt_bboxes, gt_mask)

        status = {
            'num_matched': loss_dict.pop('num_matched').item(),
            'mean_confidence': predictions['pred_confidences'].mean().item(),
        }

        return loss_dict, status

    def run_step(self, data_list):
        """Override to flatten loss dict for tqdm display and write log.txt."""
        step_log = super().run_step(data_list)

        if self.is_master:
            # Flatten loss/status for tqdm postfix display
            flat = {}
            if 'loss' in step_log:
                for k, v in step_log['loss'].items():
                    flat[k] = v
            if 'status' in step_log:
                for k, v in step_log['status'].items():
                    flat[k] = v
            step_log.update(flat)

            # Write to log.txt every i_print steps
            if self.step % self.i_print == 0:
                log_path = os.path.join(self.output_dir, 'log.txt')
                log_entry = {k: round(v, 6) if isinstance(v, float) else v
                             for k, v in flat.items()}
                with open(log_path, 'a') as f:
                    f.write(f'step {self.step}: {json.dumps(log_entry)}\n')

        return step_log

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=16, batch_size=4):
        """
        Override BasicTrainer.snapshot_dataset() to visualize GT bboxes from the dataset.
        The base class version calls self.dataset.visualize_sample() which doesn't exist
        for ERPBBoxDataset.
        """
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
                        torch.zeros(0, 3), torch.zeros(0, 3),
                        torch.zeros(0, 1), torch.zeros(0, 1),
                        title=f'[{prefix}] {sample_ids[i]}' if i < len(sample_ids) else f'{prefix}_{sample_count}',
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
        Generate bbox visualization comparing GT vs predicted bounding boxes.
        """
        model = self.models['bbox_estimator']
        model.eval()

        # Collect samples
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=2,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        all_vis = []
        sample_count = 0

        for batch in dataloader:
            if sample_count >= num_samples:
                break

            voxel_grid = batch['voxel_grid'].to(self.device)
            gt_bboxes = batch['gt_bboxes'].to(self.device)
            gt_mask = batch['gt_mask'].to(self.device)
            sample_ids = batch['sample_id']

            predictions = model(voxel_grid)

            # Visualize each sample
            for i in range(voxel_grid.shape[0]):
                if sample_count >= num_samples:
                    break

                fig = self._visualize_bboxes_topdown(
                    gt_bboxes[i].cpu(),
                    gt_mask[i].cpu(),
                    predictions['pred_centers'][i].cpu(),
                    predictions['pred_sizes'][i].cpu(),
                    predictions['pred_rotations'][i].cpu(),
                    predictions['pred_confidences'][i].cpu(),
                    title=sample_ids[i] if i < len(sample_ids) else f'sample_{sample_count}',
                )
                all_vis.append(fig)
                sample_count += 1

        # Build samples dict for base class to save
        samples = {}
        if all_vis:
            grid = torch.stack(all_vis)  # [N, 3, H, W]
            samples['train_bbox_prediction'] = {'value': grid, 'type': 'image'}

        # Also run on eval dataset if available
        if self.eval_dataset is not None:
            eval_vis = []
            eval_count = 0
            eval_loader = torch.utils.data.DataLoader(
                self.eval_dataset,
                batch_size=batch_size,
                num_workers=2,
                shuffle=True,
                collate_fn=self.eval_dataset.collate_fn if hasattr(self.eval_dataset, 'collate_fn') else None,
            )
            for batch in eval_loader:
                if eval_count >= num_samples:
                    break
                voxel_grid = batch['voxel_grid'].to(self.device)
                gt_bboxes = batch['gt_bboxes'].to(self.device)
                gt_mask = batch['gt_mask'].to(self.device)
                sample_ids = batch['sample_id']
                predictions = model(voxel_grid)
                for i in range(voxel_grid.shape[0]):
                    if eval_count >= num_samples:
                        break
                    fig = self._visualize_bboxes_topdown(
                        gt_bboxes[i].cpu(), gt_mask[i].cpu(),
                        predictions['pred_centers'][i].cpu(),
                        predictions['pred_sizes'][i].cpu(),
                        predictions['pred_rotations'][i].cpu(),
                        predictions['pred_confidences'][i].cpu(),
                        title=f'[eval] {sample_ids[i]}' if i < len(sample_ids) else f'eval_{eval_count}',
                    )
                    eval_vis.append(fig)
                    eval_count += 1

            if eval_vis:
                grid = torch.stack(eval_vis)
                samples['eval_bbox_prediction'] = {'value': grid, 'type': 'image'}

        # Compute metrics on a larger sample
        self._compute_and_log_metrics(num_eval=min(200, len(self.dataset)))

        model.train()

        return samples

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
            label = f'Pred (conf≥{self.confidence_threshold})' if n_pred == 0 else None
            self._draw_rotated_box(ax, cx, cy, sx, sy, rot,
                                    color='red', alpha=0.3, label=label)
            # Annotate confidence
            ax.text(cx, cy, f'{conf:.2f}', ha='center', va='center',
                    fontsize=6, color='red')
            n_pred += 1

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.55, 0.55)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\nGT: {n_gt}, Pred: {n_pred}', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Convert to tensor
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
        plt.close(fig)

        tensor = torch.from_numpy(buf).permute(2, 0, 1).float() / 255.0
        return tensor

    @staticmethod
    def _draw_rotated_box(ax, cx, cy, w, h, angle_rad, color='green', alpha=0.5, label=None):
        """Draw a rotated rectangle on matplotlib axis."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 4 corners relative to center
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
        """Compute and log IoU metrics."""
        model = self.models['bbox_estimator']
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

            predictions = model(voxel_grid)

            B = voxel_grid.shape[0]
            for b in range(B):
                if sample_count >= num_eval:
                    break
                sample_count += 1

                n_gt = gt_mask[b].sum().item()
                if n_gt == 0:
                    continue

                # Get confident predictions
                conf_mask = predictions['pred_confidences'][b, :, 0] >= self.confidence_threshold
                n_pred = conf_mask.sum().item()

                total_gt += n_gt
                total_pred += n_pred

                if n_pred == 0:
                    continue

                pred_c = predictions['pred_centers'][b][conf_mask]  # [n_pred, 3]
                pred_s = predictions['pred_sizes'][b][conf_mask]    # [n_pred, 3]
                pred_r = predictions['pred_rotations'][b][conf_mask, 0]  # [n_pred]

                gt_c = gt_bboxes[b, :n_gt, :3]
                gt_s = gt_bboxes[b, :n_gt, 3:6]
                gt_r = gt_bboxes[b, :n_gt, 6]

                # IoU matrix
                pred_box6 = torch.cat([pred_c, pred_s], dim=-1)
                gt_box6 = torch.cat([gt_c, gt_s], dim=-1)
                iou_mat = bbox3d_iou(pred_box6, gt_box6)  # [n_pred, n_gt]

                # Greedy matching by IoU
                max_iou_per_gt, best_pred_per_gt = iou_mat.max(dim=0)
                all_ious.extend(max_iou_per_gt.cpu().tolist())
                total_tp_25 += (max_iou_per_gt >= 0.25).sum().item()
                total_tp_50 += (max_iou_per_gt >= 0.5).sum().item()

                # Per-matched-pair errors
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
