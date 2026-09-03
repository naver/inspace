# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
3D Bounding Box Estimation - Inference and Evaluation (DETR v2, Voxel-64 input).

Usage:
    python eval/bbox_inference_detr_voxel64.py \
        --config configs/bbox/legacy/erp_bbox_estimator.json \
        --ckpt results/bbox_estimator_voxel64/ckpts/bbox_estimator_ema0.9999_step0050000.pt \
        --data_dir datasets/ERP_3D_FRONT_test \
        --output_dir evals/bbox_detr_voxel64 \
        --num_vis 20 \
        --confidence_threshold 0.5
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from trellis2.models.bbox_estimator import BBoxEstimator
from trellis2.datasets.erp_bbox_estimation import ERPBBoxDataset
from trellis2.utils.bbox_loss import bbox3d_iou


def load_model(config_path, ckpt_path, device='cuda'):
    """Load trained BBox estimator model."""
    with open(config_path) as f:
        config = json.load(f)

    model_args = config['models']['bbox_estimator']['args']
    model = BBoxEstimator(**model_args).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    print(f'Loaded model from {ckpt_path}')
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)')

    return model, config


@torch.no_grad()
def run_inference(model, dataset, device='cuda', batch_size=64, conf_threshold=0.5):
    """Run inference on all samples."""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    results = []
    for batch in tqdm(dataloader, desc='Inference'):
        voxel_grid = batch['voxel_grid'].to(device)
        gt_bboxes = batch['gt_bboxes']
        gt_mask = batch['gt_mask']
        sample_ids = batch['sample_id']

        predictions = model(voxel_grid)

        B = voxel_grid.shape[0]
        for b in range(B):
            n_gt = gt_mask[b].sum().item()
            gt = gt_bboxes[b, :n_gt].numpy()

            pred_c = predictions['pred_centers'][b].cpu().float().numpy()
            pred_s = predictions['pred_sizes'][b].cpu().float().numpy()
            pred_r = predictions['pred_rotations'][b].cpu().float().numpy()
            pred_conf = predictions['pred_confidences'][b, :, 0].cpu().float().numpy()

            all_pred = np.concatenate([pred_c, pred_s, pred_r], axis=-1)

            mask = pred_conf >= conf_threshold
            filtered_pred = all_pred[mask]
            filtered_conf = pred_conf[mask]

            sort_idx = np.argsort(-filtered_conf)
            filtered_pred = filtered_pred[sort_idx]
            filtered_conf = filtered_conf[sort_idx]

            results.append({
                'sample_id': sample_ids[b],
                'gt_bboxes': gt,
                'pred_bboxes': filtered_pred,
                'pred_confidences': filtered_conf,
            })

    return results


def compute_metrics(results, iou_thresholds=[0.25, 0.5, 0.75]):
    """Compute detection metrics over all results."""
    metrics = {}
    all_ious = []
    all_center_errors = []
    all_size_errors = []
    all_rot_errors = []

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
            iou_mat = bbox3d_iou(pred_t, gt_t).numpy()

            gt_matched = np.zeros(n_gt, dtype=bool)
            for pi in range(n_pred):
                best_gt = iou_mat[pi].argmax()
                if iou_mat[pi, best_gt] >= thresh and not gt_matched[best_gt]:
                    gt_matched[best_gt] = True
                    total_tp += 1

                    if thresh == iou_thresholds[0]:
                        pred_box = r['pred_bboxes'][pi]
                        gt_box = r['gt_bboxes'][best_gt]
                        all_ious.append(iou_mat[pi, best_gt])
                        all_center_errors.append(np.linalg.norm(pred_box[:3] - gt_box[:3]))
                        all_size_errors.append(np.mean(np.abs(pred_box[3:6] - gt_box[3:6])))
                        angle_diff = (pred_box[6] - gt_box[6] + np.pi) % (2 * np.pi) - np.pi
                        all_rot_errors.append(np.abs(angle_diff))

        if total_gt > 0:
            metrics[f'recall@{thresh}'] = total_tp / total_gt
        if total_pred > 0:
            metrics[f'precision@{thresh}'] = total_tp / total_pred
        if total_gt > 0 and total_pred > 0:
            prec = total_tp / total_pred
            rec = total_tp / total_gt
            if prec + rec > 0:
                metrics[f'f1@{thresh}'] = 2 * prec * rec / (prec + rec)

    if all_ious:
        metrics['mean_iou'] = float(np.mean(all_ious))
        metrics['median_iou'] = float(np.median(all_ious))
    if all_center_errors:
        metrics['mean_center_error'] = float(np.mean(all_center_errors))
    if all_size_errors:
        metrics['mean_size_error'] = float(np.mean(all_size_errors))
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


def visualize_topdown(result, output_path, conf_threshold=0.5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    ax = axes[0]
    ax.set_title(f'Top-down (XY)\n{result["sample_id"]}', fontsize=9)
    for i, gt in enumerate(result['gt_bboxes']):
        _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6],
                          color='green', alpha=0.4, label='GT' if i == 0 else None)
    for i, (pred, conf) in enumerate(zip(result['pred_bboxes'], result['pred_confidences'])):
        _draw_rotated_box(ax, pred[0], pred[1], pred[3], pred[4], pred[6],
                          color='red', alpha=0.3, label=f'Pred' if i == 0 else None)
        ax.text(pred[0], pred[1], f'{conf:.2f}', ha='center', va='center', fontsize=5, color='darkred')
    ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55); ax.set_aspect('equal')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.set_title(f'Side view (XZ)\nGT: {len(result["gt_bboxes"])}, Pred: {len(result["pred_bboxes"])}', fontsize=9)
    for i, gt in enumerate(result['gt_bboxes']):
        rect = plt.Rectangle((gt[0] - gt[3]/2, gt[2] - gt[5]/2), gt[3], gt[5],
                              facecolor='green', edgecolor='green', alpha=0.4, label='GT' if i == 0 else None)
        ax.add_patch(rect)
    for i, (pred, conf) in enumerate(zip(result['pred_bboxes'], result['pred_confidences'])):
        rect = plt.Rectangle((pred[0] - pred[3]/2, pred[2] - pred[5]/2), pred[3], pred[5],
                              facecolor='red', edgecolor='red', alpha=0.3, label='Pred' if i == 0 else None)
        ax.add_patch(rect)
    ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55); ax.set_aspect('equal')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


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

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=300)
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
            for i, gt in enumerate(r['gt_bboxes']):
                _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6], color='green', alpha=0.4)
            for i, (pred, conf) in enumerate(zip(r['pred_bboxes'], r['pred_confidences'])):
                _draw_rotated_box(ax, pred[0], pred[1], pred[3], pred[4], pred[6], color='red', alpha=0.3)
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


def visualize_concat(results, output_dir, num_samples=50):
    """
    Create a concatenated comparison image: each row = one sample with GT (left) vs Pred (right).
    Saves one large image for quick overview.
    """
    n = min(num_samples, len(results))
    cell_size = 4  # inches per cell
    fig, axes = plt.subplots(n, 2, figsize=(cell_size * 2, cell_size * n), dpi=300)
    if n == 1:
        axes = axes[np.newaxis, :]

    for idx in tqdm(range(n), desc='Visualizing concat'):
        r = results[idx]
        short_id = r['sample_id'].split('/')[-1][:30]

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

        # Right: GT (green, faint) + Pred (red)
        ax = axes[idx, 1]
        for i, gt in enumerate(r['gt_bboxes']):
            _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6],
                              color='green', alpha=0.2)
        for i, (pred, conf) in enumerate(zip(r['pred_bboxes'], r['pred_confidences'])):
            _draw_rotated_box(ax, pred[0], pred[1], pred[3], pred[4], pred[6],
                              color='red', alpha=0.4, label='Pred' if i == 0 else None)
            ax.text(pred[0], pred[1], f'{conf:.2f}', ha='center', va='center',
                    fontsize=5, color='darkred', weight='bold')
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


def main():
    parser = argparse.ArgumentParser(description='3D BBox Eval - DETR Voxel-64')
    parser.add_argument('--config', type=str, default='configs/bbox/legacy/erp_bbox_estimator.json')
    parser.add_argument('--ckpt', type=str, default='results/bbox_estimator_voxel64/ckpts/bbox_estimator_ema0.9999_step0050000.pt')
    parser.add_argument('--data_dir', type=str, default='datasets/ERP_3D_FRONT_test')
    parser.add_argument('--output_dir', type=str, default='evals/bbox_detr_voxel64')
    parser.add_argument('--num_vis', type=int, default=20)
    parser.add_argument('--confidence_threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    args.num_vis = 5
    args.ckpt = 'results/bbox_estimator_voxel64_v3/ckpts/bbox_estimator_ema0.9999_step0050000.pt'
    args.config = 'configs/bbox/legacy/erp_bbox_estimator_v3.json'
    args.data_dir = 'datasets/ERP_3D_FRONT_test'
    args.output_dir = 'evals/bbox_detr_voxel64_v3'
    args.batch_size = 64
    args.device = 'cuda:5'
    args.confidence_threshold = 0.5

    # Extract checkpoint name for output subdirectory
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]  # e.g. bbox_estimator_ema0.9999_step0050000
    # Shorten: extract step part
    import re
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
                            batch_size=args.batch_size, conf_threshold=args.confidence_threshold)

    # Metrics
    print('\n' + '=' * 60)
    print('DETR Voxel-64 Metrics')
    print('=' * 60)
    metrics = compute_metrics(results)
    for k, v in tqdm(sorted(metrics.items()), desc='Computing metrics'):
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    print('=' * 60)

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Visualize
    print(f'\nVisualizing {args.num_vis} samples...')
    for i in tqdm(range(min(args.num_vis, len(results))), desc='Visualizing'):
        r = results[i]
        safe_name = r['sample_id'].replace('/', '_')
        visualize_topdown(r, os.path.join(args.output_dir, 'per_sample', f'{safe_name}.png'),
                          conf_threshold=args.confidence_threshold)

    print(f'\nVisualizing grid...')
    visualize_grid(results, args.output_dir)

    print(f'\nVisualizing concat...')
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
