# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
3D Bounding Box Estimation - Inference and Evaluation.

Usage:
    python eval/bbox_inference.py \
        --config configs/bbox/legacy/erp_bbox_estimator.json \
        --ckpt results/bbox_estimator/ckpts/bbox_estimator_step0050000.pt \
        --data_dir datasets/ERP_3D_FRONT_test \
        --output_dir evals/bbox_estimator \
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
import matplotlib.patches as patches

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from trellis2.models.bbox_estimator import BBoxEstimator
from trellis2.datasets.erp_bbox_estimation import ERPBBoxDataset
from trellis2.utils.bbox_loss import bbox3d_iou, bbox3d_giou


def load_model(config_path, ckpt_path, device='cuda'):
    """Load trained BBox estimator model."""
    with open(config_path) as f:
        config = json.load(f)

    model_args = config['models']['bbox_estimator']['args']
    model = BBoxEstimator(**model_args).to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    print(f'Loaded model from {ckpt_path}')
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)')

    return model, config


@torch.no_grad()
def run_inference(model, dataset, device='cuda', batch_size=64, conf_threshold=0.5):
    """
    Run inference on all samples and collect predictions + GT.

    Returns:
        results: list of dicts with keys:
            - sample_id: str
            - gt_bboxes: [N, 7] numpy
            - pred_bboxes: [M, 7] numpy (filtered by confidence)
            - pred_confidences: [M] numpy
            - all_pred_bboxes: [Q, 7] numpy (all queries, unfiltered)
            - all_pred_confidences: [Q] numpy
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    results = []
    for batch in tqdm(dataloader, desc='Inference'):
        ss_latent = batch['ss_latent'].to(device)
        gt_bboxes = batch['gt_bboxes']
        gt_mask = batch['gt_mask']
        sample_ids = batch['sample_id']

        predictions = model(ss_latent)

        B = ss_latent.shape[0]
        for b in range(B):
            # GT
            n_gt = gt_mask[b].sum().item()
            gt = gt_bboxes[b, :n_gt].numpy()

            # All predictions
            pred_c = predictions['pred_centers'][b].cpu().numpy()   # [Q, 3]
            pred_s = predictions['pred_sizes'][b].cpu().numpy()     # [Q, 3]
            pred_r = predictions['pred_rotations'][b].cpu().numpy() # [Q, 1]
            pred_conf = predictions['pred_confidences'][b, :, 0].cpu().numpy()  # [Q]

            all_pred = np.concatenate([pred_c, pred_s, pred_r], axis=-1)  # [Q, 7]

            # Filter by confidence
            mask = pred_conf >= conf_threshold
            filtered_pred = all_pred[mask]
            filtered_conf = pred_conf[mask]

            # Sort by confidence (descending)
            sort_idx = np.argsort(-filtered_conf)
            filtered_pred = filtered_pred[sort_idx]
            filtered_conf = filtered_conf[sort_idx]

            results.append({
                'sample_id': sample_ids[b],
                'gt_bboxes': gt,
                'pred_bboxes': filtered_pred,
                'pred_confidences': filtered_conf,
                'all_pred_bboxes': all_pred,
                'all_pred_confidences': pred_conf,
            })

    return results


def compute_metrics(results, iou_thresholds=[0.25, 0.5]):
    """
    Compute detection metrics over all results.

    Returns:
        metrics: dict with metric name → value
    """
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

            # Compute IoU matrix
            pred_t = torch.from_numpy(r['pred_bboxes'][:, :6]).float()
            gt_t = torch.from_numpy(r['gt_bboxes'][:, :6]).float()
            iou_mat = bbox3d_iou(pred_t, gt_t).numpy()  # [n_pred, n_gt]

            # Greedy matching
            gt_matched = np.zeros(n_gt, dtype=bool)
            for pi in range(n_pred):
                best_gt = iou_mat[pi].argmax()
                if iou_mat[pi, best_gt] >= thresh and not gt_matched[best_gt]:
                    gt_matched[best_gt] = True
                    total_tp += 1

                    # Collect errors for matched pairs (only at lowest threshold)
                    if thresh == iou_thresholds[0]:
                        pred_box = r['pred_bboxes'][pi]
                        gt_box = r['gt_bboxes'][best_gt]
                        all_ious.append(iou_mat[pi, best_gt])
                        all_center_errors.append(
                            np.linalg.norm(pred_box[:3] - gt_box[:3])
                        )
                        all_size_errors.append(
                            np.mean(np.abs(pred_box[3:6] - gt_box[3:6]))
                        )
                        # Angle error (handle wrap-around)
                        angle_diff = pred_box[6] - gt_box[6]
                        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
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

    # Aggregate per-match errors
    if all_ious:
        metrics['mean_iou'] = float(np.mean(all_ious))
        metrics['median_iou'] = float(np.median(all_ious))
    if all_center_errors:
        metrics['mean_center_error'] = float(np.mean(all_center_errors))
        metrics['median_center_error'] = float(np.median(all_center_errors))
    if all_size_errors:
        metrics['mean_size_error'] = float(np.mean(all_size_errors))
    if all_rot_errors:
        metrics['mean_rot_error_deg'] = float(np.degrees(np.mean(all_rot_errors)))
        metrics['median_rot_error_deg'] = float(np.degrees(np.median(all_rot_errors)))

    # Count stats
    total_gt = sum(len(r['gt_bboxes']) for r in results)
    total_pred = sum(len(r['pred_bboxes']) for r in results)
    metrics['total_gt_boxes'] = total_gt
    metrics['total_pred_boxes'] = total_pred
    metrics['num_scenes'] = len(results)
    metrics['avg_gt_per_scene'] = total_gt / max(len(results), 1)
    metrics['avg_pred_per_scene'] = total_pred / max(len(results), 1)

    return metrics


def visualize_topdown(result, output_path, conf_threshold=0.5):
    """
    Create a top-down (XY plane) visualization with GT (green) vs Pred (red).
    Also includes a side view (XZ plane) for height verification.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # --- Top-down view (XY) ---
    ax = axes[0]
    ax.set_title(f'Top-down (XY)\n{result["sample_id"]}', fontsize=9)

    for i, gt in enumerate(result['gt_bboxes']):
        _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6],
                          color='green', alpha=0.4,
                          label='GT' if i == 0 else None)

    for i, (pred, conf) in enumerate(zip(result['pred_bboxes'], result['pred_confidences'])):
        _draw_rotated_box(ax, pred[0], pred[1], pred[3], pred[4], pred[6],
                          color='red', alpha=0.3,
                          label=f'Pred (≥{conf_threshold})' if i == 0 else None)
        ax.text(pred[0], pred[1], f'{conf:.2f}', ha='center', va='center',
                fontsize=5, color='darkred', weight='bold')

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # --- Side view (XZ) ---
    ax = axes[1]
    ax.set_title(f'Side view (XZ)\nGT: {len(result["gt_bboxes"])}, Pred: {len(result["pred_bboxes"])}',
                 fontsize=9)

    for i, gt in enumerate(result['gt_bboxes']):
        rect = plt.Rectangle(
            (gt[0] - gt[3]/2, gt[2] - gt[5]/2), gt[3], gt[5],
            facecolor='green', edgecolor='green', alpha=0.4,
            label='GT' if i == 0 else None
        )
        ax.add_patch(rect)

    for i, (pred, conf) in enumerate(zip(result['pred_bboxes'], result['pred_confidences'])):
        rect = plt.Rectangle(
            (pred[0] - pred[3]/2, pred[2] - pred[5]/2), pred[3], pred[5],
            facecolor='red', edgecolor='red', alpha=0.3,
            label=f'Pred' if i == 0 else None
        )
        ax.add_patch(rect)

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect('equal')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Z (height)')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def visualize_grid(results, output_path, num_samples=16, conf_threshold=0.5):
    """Create a grid visualization of multiple samples."""
    n = min(num_samples, len(results))
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=300)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(n):
        r = results[idx]
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        for i, gt in enumerate(r['gt_bboxes']):
            _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6],
                              color='green', alpha=0.4)

        for i, (pred, conf) in enumerate(zip(r['pred_bboxes'], r['pred_confidences'])):
            _draw_rotated_box(ax, pred[0], pred[1], pred[3], pred[4], pred[6],
                              color='red', alpha=0.3)

        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.55, 0.55)
        ax.set_aspect('equal')
        short_id = r['sample_id'].split('/')[-1][:25]
        ax.set_title(f'{short_id}\nGT:{len(r["gt_bboxes"])} P:{len(r["pred_bboxes"])}',
                     fontsize=7)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=5)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved grid visualization: {output_path}')


def _draw_rotated_box(ax, cx, cy, w, h, angle_rad, color='green', alpha=0.5, label=None):
    """Draw a rotated rectangle on matplotlib axis."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    hw, hh = w / 2, h / 2
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    world_corners = []
    for lx, ly in corners:
        wx = cx + lx * cos_a - ly * sin_a
        wy = cy + lx * sin_a + ly * cos_a
        world_corners.append([wx, wy])
    polygon = plt.Polygon(world_corners, closed=True,
                           facecolor=color, edgecolor=color,
                           alpha=alpha, linewidth=1, label=label)
    ax.add_patch(polygon)


def main():
    parser = argparse.ArgumentParser(description='3D BBox Estimation - Inference & Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Test dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_vis', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--batch_size', type=int, default=64, help='Inference batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'per_sample'), exist_ok=True)

    # Load model
    model, config = load_model(args.config, args.ckpt, args.device)

    # Load dataset
    dataset_args = config['dataset']['args']
    dataset_args['noise_augmentation'] = 0.0  # No noise during inference
    dataset = ERPBBoxDataset(args.data_dir, **dataset_args)
    print(f'Dataset: {dataset}')

    # Run inference
    results = run_inference(
        model, dataset,
        device=args.device,
        batch_size=args.batch_size,
        conf_threshold=args.confidence_threshold,
    )

    # Compute metrics
    print('\n' + '=' * 60)
    print('Computing metrics...')
    metrics = compute_metrics(results, iou_thresholds=[0.25, 0.5, 0.75])
    print('=' * 60)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
    print('=' * 60)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Saved metrics to {metrics_path}')

    # Visualize individual samples
    print(f'\nVisualizing {args.num_vis} samples...')
    for i in range(min(args.num_vis, len(results))):
        r = results[i]
        safe_name = r['sample_id'].replace('/', '_')
        vis_path = os.path.join(args.output_dir, 'per_sample', f'{safe_name}.png')
        visualize_topdown(r, vis_path, conf_threshold=args.confidence_threshold)

    # Grid visualization
    visualize_grid(
        results[:min(args.num_vis, 16)],
        os.path.join(args.output_dir, 'grid_comparison.png'),
        num_samples=min(args.num_vis, 16),
        conf_threshold=args.confidence_threshold,
    )

    # Save all predictions as JSON
    predictions_data = []
    for r in results:
        predictions_data.append({
            'sample_id': r['sample_id'],
            'num_gt': len(r['gt_bboxes']),
            'num_pred': len(r['pred_bboxes']),
            'gt_bboxes': r['gt_bboxes'].tolist(),
            'pred_bboxes': r['pred_bboxes'].tolist(),
            'pred_confidences': r['pred_confidences'].tolist(),
        })
    pred_path = os.path.join(args.output_dir, 'predictions.json')
    with open(pred_path, 'w') as f:
        json.dump(predictions_data, f)
    print(f'Saved predictions to {pred_path}')

    print(f'\nDone! Results in {args.output_dir}')


if __name__ == '__main__':
    main()
