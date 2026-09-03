# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
3D Bounding Box Estimation - Inference and Evaluation (CenterPoint, Voxel-64 input).

Uses dense heatmap prediction + 3D NMS for detection.

Usage:
    python eval/bbox_inference_centerpoint.py \
        --config configs/bbox/erp_bbox_centerpoint.json \
        --ckpt results/bbox_centerpoint/ckpts/bbox_centerpoint_ema0.9999_step0050000.pt \
        --data_dir datasets/ERP_3D_FRONT_test \
        --output_dir evals/bbox_centerpoint \
        --num_vis 20 \
        --score_threshold 0.3
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from trellis2.models.bbox_centerpoint import BBoxCenterPoint
from trellis2.datasets.erp_bbox_estimation import ERPBBoxDataset
from trellis2.utils.bbox_loss import bbox3d_iou


def load_model(config_path, ckpt_path, device='cuda'):
    """Load trained CenterPoint model."""
    with open(config_path) as f:
        config = json.load(f)

    model_args = config['models']['bbox_centerpoint']['args']
    model = BBoxCenterPoint(**model_args).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    print(f'Loaded model from {ckpt_path}')
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)')

    return model, config


@torch.no_grad()
def run_inference(model, dataset, device='cuda', batch_size=32,
                  score_threshold=0.3, nms_kernel=7, iou_nms_threshold=0.3):
    """Run CenterPoint inference on all samples."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    results = []
    all_heatmaps = []

    for batch in tqdm(dataloader, desc='Inference'):
        voxel_grid = batch['voxel_grid'].to(device)
        gt_bboxes = batch['gt_bboxes']
        gt_mask = batch['gt_mask']
        sample_ids = batch['sample_id']

        # Match training: run model under bf16 autocast (same as training snapshot)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(voxel_grid)
        # Cast to float32 for post-processing
        outputs = {k: v.float() for k, v in outputs.items()}
        detections = model.decode_detections(
            outputs, score_threshold=score_threshold, nms_kernel=nms_kernel,
            iou_nms_threshold=iou_nms_threshold)

        B = voxel_grid.shape[0]
        for b in range(B):
            n_gt = gt_mask[b].sum().item()
            gt = gt_bboxes[b, :n_gt].numpy()

            det = detections[b]
            n_pred = det['pred_centers'].shape[0]

            if n_pred > 0:
                pred_bboxes = torch.cat([
                    det['pred_centers'],
                    det['pred_sizes'],
                    det['pred_rotations'],
                ], dim=-1).cpu().numpy()  # [K, 7]
                pred_conf = det['pred_confidences'][:, 0].cpu().numpy()  # [K]
            else:
                pred_bboxes = np.zeros((0, 7))
                pred_conf = np.zeros(0)

            # Sort by confidence
            if len(pred_conf) > 0:
                sort_idx = np.argsort(-pred_conf)
                pred_bboxes = pred_bboxes[sort_idx]
                pred_conf = pred_conf[sort_idx]

            results.append({
                'sample_id': sample_ids[b],
                'gt_bboxes': gt,
                'pred_bboxes': pred_bboxes,
                'pred_confidences': pred_conf,
                'heatmap_topdown': outputs['heatmap'][b, 0].cpu().max(dim=2).values.numpy(),
                'voxel_topdown': voxel_grid[b, 0].float().cpu().max(dim=2).values.numpy(),
            })

    return results


def bbox2d_iou(boxes1, boxes2):
    """
    Compute pairwise 2D IoU on XY plane (top-down) between two sets of bounding boxes.

    Args:
        boxes1: [N, 6] (cx, cy, cz, sx, sy, sz)
        boxes2: [M, 6] (cx, cy, cz, sx, sy, sz)

    Returns:
        iou: [N, M] 2D IoU matrix (using only X, Y dimensions)
    """
    # Use only X (dim 0) and Y (dim 1)
    min1 = boxes1[:, :2] - boxes1[:, 3:5] / 2  # [N, 2]
    max1 = boxes1[:, :2] + boxes1[:, 3:5] / 2  # [N, 2]
    min2 = boxes2[:, :2] - boxes2[:, 3:5] / 2  # [M, 2]
    max2 = boxes2[:, :2] + boxes2[:, 3:5] / 2  # [M, 2]

    inter_min = torch.max(min1.unsqueeze(1), min2.unsqueeze(0))  # [N, M, 2]
    inter_max = torch.min(max1.unsqueeze(1), max2.unsqueeze(0))  # [N, M, 2]
    inter_size = (inter_max - inter_min).clamp(min=0)  # [N, M, 2]
    inter_area = inter_size.prod(-1)  # [N, M]

    area1 = boxes1[:, 3] * boxes1[:, 4]  # [N]
    area2 = boxes2[:, 3] * boxes2[:, 4]  # [M]
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area  # [N, M]

    return inter_area / union_area.clamp(min=1e-8)


def compute_metrics(results, iou_thresholds=[0.25, 0.5, 0.75], data_dir=None):
    """Compute detection metrics over all results (both 3D and 2D IoU)."""
    metrics = {}

    # Load normalization scales for real-world distance computation
    scale_cache = {}
    if data_dir is not None:
        for r in results:
            sid = r['sample_id']
            ninfo_path = os.path.join(data_dir, sid, 'dual_grid_512', 'normalization_info.json')
            if os.path.exists(ninfo_path):
                with open(ninfo_path) as f:
                    scale_cache[sid] = json.load(f)['scale']

    # Collect per-match statistics (using lowest threshold for matching)
    all_ious_3d = []
    all_ious_2d = []
    all_center_errors_3d = []
    all_center_errors_2d = []
    all_center_errors_3d_m = []
    all_center_errors_2d_m = []
    all_size_errors = []
    all_size_errors_m = []
    all_rot_errors = []

    # ---- 3D IoU metrics ----
    for thresh in iou_thresholds:
        total_gt = 0
        total_pred = 0
        total_tp = 0

        for r in results:
            n_gt = len(r['gt_bboxes'])
            n_pred = len(r['pred_bboxes'])
            total_gt += n_gt
            total_pred += n_pred

            if n_gt == 0 or n_pred == 0:
                continue

            pred_t = torch.from_numpy(r['pred_bboxes'][:, :6]).float()
            gt_t = torch.from_numpy(r['gt_bboxes'][:, :6]).float()
            iou_mat_3d = bbox3d_iou(pred_t, gt_t).numpy()
            iou_mat_2d = bbox2d_iou(pred_t, gt_t).numpy()

            gt_matched = np.zeros(n_gt, dtype=bool)
            for pi in range(n_pred):
                best_gt = iou_mat_3d[pi].argmax()
                if iou_mat_3d[pi, best_gt] >= thresh and not gt_matched[best_gt]:
                    gt_matched[best_gt] = True
                    total_tp += 1

                    if thresh == iou_thresholds[0]:
                        pred_box = r['pred_bboxes'][pi]
                        gt_box = r['gt_bboxes'][best_gt]
                        all_ious_3d.append(iou_mat_3d[pi, best_gt])
                        all_ious_2d.append(iou_mat_2d[pi, best_gt])
                        ce_3d = np.linalg.norm(pred_box[:3] - gt_box[:3])
                        ce_2d = np.linalg.norm(pred_box[:2] - gt_box[:2])
                        se = np.mean(np.abs(pred_box[3:6] - gt_box[3:6]))
                        all_center_errors_3d.append(ce_3d)
                        all_center_errors_2d.append(ce_2d)
                        all_size_errors.append(se)
                        # Real-world distance (meters)
                        scale = scale_cache.get(r['sample_id'])
                        if scale is not None and scale > 0:
                            all_center_errors_3d_m.append(ce_3d / scale)
                            all_center_errors_2d_m.append(ce_2d / scale)
                            all_size_errors_m.append(se / scale)
                        angle_diff = (pred_box[6] - gt_box[6] + np.pi) % (2 * np.pi) - np.pi
                        all_rot_errors.append(np.abs(angle_diff))

        if total_gt > 0:
            metrics[f'3d_recall@{thresh}'] = total_tp / total_gt
        if total_pred > 0:
            metrics[f'3d_precision@{thresh}'] = total_tp / total_pred
        if total_gt > 0 and total_pred > 0:
            prec = total_tp / total_pred
            rec = total_tp / total_gt
            if prec + rec > 0:
                metrics[f'3d_f1@{thresh}'] = 2 * prec * rec / (prec + rec)

    # ---- 2D IoU metrics (top-down XY plane) ----
    for thresh in iou_thresholds:
        total_gt = 0
        total_pred = 0
        total_tp = 0

        for r in results:
            n_gt = len(r['gt_bboxes'])
            n_pred = len(r['pred_bboxes'])
            total_gt += n_gt
            total_pred += n_pred

            if n_gt == 0 or n_pred == 0:
                continue

            pred_t = torch.from_numpy(r['pred_bboxes'][:, :6]).float()
            gt_t = torch.from_numpy(r['gt_bboxes'][:, :6]).float()
            iou_mat_2d = bbox2d_iou(pred_t, gt_t).numpy()

            gt_matched = np.zeros(n_gt, dtype=bool)
            for pi in range(n_pred):
                best_gt = iou_mat_2d[pi].argmax()
                if iou_mat_2d[pi, best_gt] >= thresh and not gt_matched[best_gt]:
                    gt_matched[best_gt] = True
                    total_tp += 1

        if total_gt > 0:
            metrics[f'2d_recall@{thresh}'] = total_tp / total_gt
        if total_pred > 0:
            metrics[f'2d_precision@{thresh}'] = total_tp / total_pred
        if total_gt > 0 and total_pred > 0:
            prec = total_tp / total_pred
            rec = total_tp / total_gt
            if prec + rec > 0:
                metrics[f'2d_f1@{thresh}'] = 2 * prec * rec / (prec + rec)

    # ---- Per-match statistics ----
    if all_ious_3d:
        metrics['mean_iou_3d'] = float(np.mean(all_ious_3d))
        metrics['median_iou_3d'] = float(np.median(all_ious_3d))
    if all_ious_2d:
        metrics['mean_iou_2d'] = float(np.mean(all_ious_2d))
        metrics['median_iou_2d'] = float(np.median(all_ious_2d))
    if all_center_errors_3d:
        metrics['mean_center_error_3d'] = float(np.mean(all_center_errors_3d))
    if all_center_errors_2d:
        metrics['mean_center_error_2d'] = float(np.mean(all_center_errors_2d))
    if all_center_errors_3d_m:
        metrics['mean_center_error_3d_m'] = float(np.mean(all_center_errors_3d_m))
        metrics['median_center_error_3d_m'] = float(np.median(all_center_errors_3d_m))
    if all_center_errors_2d_m:
        metrics['mean_center_error_2d_m'] = float(np.mean(all_center_errors_2d_m))
        metrics['median_center_error_2d_m'] = float(np.median(all_center_errors_2d_m))
    if all_size_errors:
        metrics['mean_size_error'] = float(np.mean(all_size_errors))
    if all_size_errors_m:
        metrics['mean_size_error_m'] = float(np.mean(all_size_errors_m))
        metrics['median_size_error_m'] = float(np.median(all_size_errors_m))
    if all_rot_errors:
        metrics['mean_rot_error_deg'] = float(np.degrees(np.mean(all_rot_errors)))

    total_gt = sum(len(r['gt_bboxes']) for r in results)
    total_pred = sum(len(r['pred_bboxes']) for r in results)
    metrics['total_gt_boxes'] = total_gt
    metrics['total_pred_boxes'] = total_pred
    metrics['num_scenes'] = len(results)
    metrics['avg_gt_per_scene'] = total_gt / max(len(results), 1)
    metrics['avg_pred_per_scene'] = total_pred / max(len(results), 1)

    return metrics


def _draw_rotated_box(ax, cx, cy, w, h, angle_rad, color='green', alpha=0.5, label=None):
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    world_corners = [[cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a] for lx, ly in corners]
    polygon = plt.Polygon(world_corners, closed=True, facecolor=color, edgecolor=color,
                           alpha=alpha, linewidth=1, label=label)
    ax.add_patch(polygon)


def visualize_topdown_with_heatmap(result, output_path):
    """Create visualization with: voxel occupancy, heatmap, GT vs Pred bboxes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)

    R = result['heatmap_topdown'].shape[0]

    # 1. Voxel occupancy (top-down)
    ax = axes[0]
    ax.imshow(result['voxel_topdown'].T, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax.set_title('Voxel Occupancy (top-down)', fontsize=9)
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    # 2. Heatmap + GT centers
    ax = axes[1]
    ax.imshow(result['voxel_topdown'].T, origin='lower', cmap='gray', vmin=0, vmax=1, alpha=0.3)
    im = ax.imshow(result['heatmap_topdown'].T, origin='lower', cmap='hot', vmin=0, vmax=1, alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for gt in result['gt_bboxes']:
        vx = (gt[0] + 0.5) * R
        vy = (gt[1] + 0.5) * R
        ax.plot(vx, vy, 'g+', markersize=10, markeredgewidth=2)
    ax.set_title(f'Center Heatmap + GT ({len(result["gt_bboxes"])} objs)', fontsize=9)

    # 3. Bbox comparison (normalized coords) - pred first, GT on top
    ax = axes[2]
    for i, (pred, conf) in enumerate(zip(result['pred_bboxes'], result['pred_confidences'])):
        _draw_rotated_box(ax, pred[0], pred[1], pred[3], pred[4], pred[6],
                          color='red', alpha=0.3, label='Pred' if i == 0 else None)
        ax.text(pred[0], pred[1], f'{conf:.2f}', ha='center', va='center', fontsize=5, color='darkred')
    for i, gt in enumerate(result['gt_bboxes']):
        _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6],
                          color='green', alpha=0.4, label='GT' if i == 0 else None)
    ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55); ax.set_aspect('equal')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    ax.set_title(f'GT: {len(result["gt_bboxes"])}, Pred: {len(result["pred_bboxes"])}', fontsize=9)

    short_id = result['sample_id'].split('/')[-1][:40]
    fig.suptitle(f'CenterPoint: {short_id}', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def visualize_concat(results, output_dir, num_samples=50):
    """
    Create a concatenated comparison image: each row = one sample with
    GT (left), Heatmap (center), Pred (right).
    """
    n = min(num_samples, len(results))
    cell_size = 4
    fig, axes = plt.subplots(n, 3, figsize=(cell_size * 3, cell_size * n), dpi=200)
    if n == 1:
        axes = axes[np.newaxis, :]

    for idx in range(n):
        r = results[idx]
        short_id = r['sample_id'].split('/')[-1][:30]
        R = r['heatmap_topdown'].shape[0]

        # Left: GT only (green)
        ax = axes[idx, 0]
        for i, gt in enumerate(r['gt_bboxes']):
            _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6],
                              color='green', alpha=0.5, label='GT' if i == 0 else None)
        ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55); ax.set_aspect('equal')
        ax.set_title(f'GT ({len(r["gt_bboxes"])} objs)\n{short_id}', fontsize=7)
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=5)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

        # Center: Heatmap + GT centers
        ax = axes[idx, 1]
        ax.imshow(r['voxel_topdown'].T, origin='lower', cmap='gray', vmin=0, vmax=1, alpha=0.3)
        ax.imshow(r['heatmap_topdown'].T, origin='lower', cmap='hot', vmin=0, vmax=1, alpha=0.7)
        for gt in r['gt_bboxes']:
            vx, vy = (gt[0] + 0.5) * R, (gt[1] + 0.5) * R
            ax.plot(vx, vy, 'g+', markersize=8, markeredgewidth=1.5)
        ax.set_title('Heatmap', fontsize=7)
        ax.tick_params(labelsize=5)

        # Right: Pred (red) first, then GT (green) on top
        ax = axes[idx, 2]
        for i, (pred, conf) in enumerate(zip(r['pred_bboxes'], r['pred_confidences'])):
            _draw_rotated_box(ax, pred[0], pred[1], pred[3], pred[4], pred[6],
                              color='red', alpha=0.3, label='Pred' if i == 0 else None)
            ax.text(pred[0], pred[1], f'{conf:.2f}', ha='center', va='center',
                    fontsize=5, color='darkred', weight='bold')
        for i, gt in enumerate(r['gt_bboxes']):
            _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6],
                              color='green', alpha=0.4, label='GT' if i == 0 else None)
        ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55); ax.set_aspect('equal')
        ax.set_title(f'Pred ({len(r["pred_bboxes"])} objs)', fontsize=7)
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=5)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'concat_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved concat comparison ({n} samples): {save_path}')


def visualize_grid(results, output_dir, per_page=36):
    """Save all samples as paginated 6x6 grid images in output_dir/grid_comparison/."""
    grid_dir = os.path.join(output_dir, 'grid_comparison')
    os.makedirs(grid_dir, exist_ok=True)

    total = len(results)
    num_pages = math.ceil(total / per_page)
    cols = int(math.ceil(math.sqrt(per_page)))
    rows = int(math.ceil(per_page / cols))

    for page in tqdm(range(num_pages), desc='Visualizing grid'):
        start = page * per_page
        end = min(start + per_page, total)
        page_results = results[start:end]
        n = len(page_results)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=200)
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for idx in range(n):
            r = page_results[idx]
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            # Draw pred (red) first, then GT (green) on top
            for i, (pred, conf) in enumerate(zip(r['pred_bboxes'], r['pred_confidences'])):
                _draw_rotated_box(ax, pred[0], pred[1], pred[3], pred[4], pred[6], color='red', alpha=0.3)
            for i, gt in enumerate(r['gt_bboxes']):
                _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6], color='green', alpha=0.4)
            ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55); ax.set_aspect('equal')
            short_id = r['sample_id'].split('/')[-1][:25]
            ax.set_title(f'{short_id}\nGT:{len(r["gt_bboxes"])} P:{len(r["pred_bboxes"])}', fontsize=7)
            ax.grid(True, alpha=0.2); ax.tick_params(labelsize=5)

        for idx in range(n, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')

        fig.suptitle(f'Grid Comparison (page {page}, samples {start}-{end-1})', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(grid_dir, f'grid_comparison_{page}.png'), bbox_inches='tight')
        plt.close(fig)

    print(f'Saved {num_pages} grid pages ({total} samples) to {grid_dir}')


def visualize_heatmap_grid(results, output_dir, per_page=36):
    """Save all samples as paginated 6x6 heatmap grid images in output_dir/grid_heatmap/."""
    grid_dir = os.path.join(output_dir, 'grid_heatmap')
    os.makedirs(grid_dir, exist_ok=True)

    total = len(results)
    num_pages = math.ceil(total / per_page)
    cols = int(math.ceil(math.sqrt(per_page)))
    rows = int(math.ceil(per_page / cols))

    for page in tqdm(range(num_pages), desc='Visualizing heatmap grid'):
        start = page * per_page
        end = min(start + per_page, total)
        page_results = results[start:end]
        n = len(page_results)

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=200)
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for idx in range(n):
            r = page_results[idx]
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            R = r['heatmap_topdown'].shape[0]
            ax.imshow(r['voxel_topdown'].T, origin='lower', cmap='gray', vmin=0, vmax=1, alpha=0.3)
            ax.imshow(r['heatmap_topdown'].T, origin='lower', cmap='hot', vmin=0, vmax=1, alpha=0.7)
            for gt in r['gt_bboxes']:
                vx, vy = (gt[0] + 0.5) * R, (gt[1] + 0.5) * R
                ax.plot(vx, vy, 'g+', markersize=6, markeredgewidth=1.5)
            short_id = r['sample_id'].split('/')[-1][:25]
            ax.set_title(f'{short_id}\n{len(r["gt_bboxes"])} objs', fontsize=7)
            ax.tick_params(labelsize=5)

        for idx in range(n, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')

        fig.suptitle(f'CenterPoint Heatmaps (page {page}, samples {start}-{end-1})', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(grid_dir, f'grid_heatmap_{page}.png'), bbox_inches='tight')
        plt.close(fig)

    print(f'Saved {num_pages} heatmap grid pages ({total} samples) to {grid_dir}')


def main():
    parser = argparse.ArgumentParser(description='3D BBox Eval - CenterPoint')
    parser.add_argument('--config', type=str, default='configs/bbox/erp_bbox_centerpoint.json')
    parser.add_argument('--ckpt', type=str, default='results/bbox_centerpoint/ckpts/bbox_centerpoint_ema0.9999_step0001500.pt')
    parser.add_argument('--data_dir', type=str, default='datasets/ERP_3D_FRONT_test')
    parser.add_argument('--output_dir', type=str, default='evals/bbox_centerpoint')
    parser.add_argument('--num_vis', type=int, default=20)
    parser.add_argument('--score_threshold', type=float, default=0.3)
    parser.add_argument('--nms_kernel', type=int, default=7)
    parser.add_argument('--iou_nms_threshold', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    args.num_vis = 100
    # args.ckpt = 'results/bbox_centerpoint_v2/ckpts/bbox_centerpoint_ema0.9999_step0018000.pt'
    args.ckpt = 'results/bbox_centerpoint_v2/ckpts/bbox_centerpoint_ema0.9999_step0024500.pt'
    args.config = 'configs/bbox/erp_bbox_centerpoint_v2.json'
    args.data_dir = 'datasets/ERP_3D_FRONT_test'
    args.output_dir = 'evals/bbox_centerpoint_v2'
    args.batch_size = 32
    args.device = 'cuda:5'
    args.score_threshold = 0.1
    args.nms_kernel = 7

    # Extract checkpoint name for output subdirectory
    import re
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    step_match = re.search(r'step(\d+)', ckpt_name)
    ckpt_tag = f'ckpt_step{step_match.group(1)}' if step_match else f'ckpt_{ckpt_name}'
    args.output_dir = os.path.join(args.output_dir, ckpt_tag)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per_sample'), exist_ok=True)

    model, config = load_model(args.config, args.ckpt, args.device)

    dataset_args = config['dataset']['args']
    dataset = ERPBBoxDataset(args.data_dir, **dataset_args)
    print(f'Dataset: {len(dataset)} samples')

    results = run_inference(model, dataset, device=args.device,
                            batch_size=args.batch_size,
                            score_threshold=args.score_threshold,
                            nms_kernel=args.nms_kernel,
                            iou_nms_threshold=args.iou_nms_threshold)

    # Metrics
    metrics = compute_metrics(results, data_dir=args.data_dir)

    print('\n' + '=' * 70)
    print('CenterPoint Metrics')
    print('=' * 70)

    print('\n  --- 3D IoU (volume-based) ---')
    for thresh in [0.25, 0.5, 0.75]:
        r = metrics.get(f'3d_recall@{thresh}', 0)
        p = metrics.get(f'3d_precision@{thresh}', 0)
        f = metrics.get(f'3d_f1@{thresh}', 0)
        print(f'  @{thresh}  recall={r:.4f}  precision={p:.4f}  f1={f:.4f}')

    print('\n  --- 2D IoU (top-down XY plane) ---')
    for thresh in [0.25, 0.5, 0.75]:
        r = metrics.get(f'2d_recall@{thresh}', 0)
        p = metrics.get(f'2d_precision@{thresh}', 0)
        f = metrics.get(f'2d_f1@{thresh}', 0)
        print(f'  @{thresh}  recall={r:.4f}  precision={p:.4f}  f1={f:.4f}')

    print('\n  --- Per-match Statistics ---')
    print(f'  mean_iou_3d:          {metrics.get("mean_iou_3d", 0):.4f}')
    print(f'  median_iou_3d:        {metrics.get("median_iou_3d", 0):.4f}')
    print(f'  mean_iou_2d:          {metrics.get("mean_iou_2d", 0):.4f}')
    print(f'  median_iou_2d:        {metrics.get("median_iou_2d", 0):.4f}')
    print(f'  mean_center_error_3d: {metrics.get("mean_center_error_3d", 0):.4f}')
    print(f'  mean_center_error_2d: {metrics.get("mean_center_error_2d", 0):.4f}')
    if 'mean_center_error_3d_m' in metrics:
        print(f'  mean_center_error_3d (m):   {metrics["mean_center_error_3d_m"]:.4f}  ({metrics["mean_center_error_3d_m"]*100:.1f} cm)')
        print(f'  median_center_error_3d (m): {metrics["median_center_error_3d_m"]:.4f}  ({metrics["median_center_error_3d_m"]*100:.1f} cm)')
        print(f'  mean_center_error_2d (m):   {metrics["mean_center_error_2d_m"]:.4f}  ({metrics["mean_center_error_2d_m"]*100:.1f} cm)')
        print(f'  median_center_error_2d (m): {metrics["median_center_error_2d_m"]:.4f}  ({metrics["median_center_error_2d_m"]*100:.1f} cm)')
    print(f'  mean_size_error:      {metrics.get("mean_size_error", 0):.4f}')
    if 'mean_size_error_m' in metrics:
        print(f'  mean_size_error (m):        {metrics["mean_size_error_m"]:.4f}  ({metrics["mean_size_error_m"]*100:.1f} cm)')
        print(f'  median_size_error (m):      {metrics["median_size_error_m"]:.4f}  ({metrics["median_size_error_m"]*100:.1f} cm)')
    print(f'  mean_rot_error_deg:   {metrics.get("mean_rot_error_deg", 0):.4f}')

    print(f'\n  --- Counts ---')
    print(f'  total_gt: {metrics["total_gt_boxes"]}  total_pred: {metrics["total_pred_boxes"]}')
    print(f'  scenes: {metrics["num_scenes"]}  avg_gt/scene: {metrics["avg_gt_per_scene"]:.1f}  avg_pred/scene: {metrics["avg_pred_per_scene"]:.1f}')
    print('=' * 70)

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Per-sample visualization (with heatmap)
    print(f'\nVisualizing {args.num_vis} samples...')
    for i in range(min(args.num_vis, len(results))):
        r = results[i]
        safe_name = r['sample_id'].replace('/', '_')
        visualize_topdown_with_heatmap(
            r, os.path.join(args.output_dir, 'per_sample', f'{safe_name}.png'))

    # Grid visualizations (6x6 pages, all samples)
    visualize_grid(results, args.output_dir)
    visualize_heatmap_grid(results, args.output_dir)

    # Concat comparison (GT | Heatmap | Pred)
    visualize_concat(results, args.output_dir, num_samples=args.num_vis)

    # Save predictions
    predictions_data = [{
        'sample_id': r['sample_id'],
        'num_gt': len(r['gt_bboxes']),
        'num_pred': len(r['pred_bboxes']),
        'gt_bboxes': r['gt_bboxes'].tolist(),
        'pred_bboxes': r['pred_bboxes'].tolist(),
        'pred_confidences': r['pred_confidences'].tolist(),
    } for r in results]
    with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
        json.dump(predictions_data, f)

    print(f'\nDone! Results in {args.output_dir}')


if __name__ == '__main__':
    main()
