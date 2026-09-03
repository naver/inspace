# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Plot loss curves and metrics for BBox Estimator training logs.

The bbox log.txt has two line formats:
  1. Summary lines (every i_print steps): "step N: {json}"
  2. Per-step lines: "N: {json}"

It also handles multiple restarts (multiple "step 0:" entries) by
taking only the last contiguous run.

Usage:
    python plot_bbox_loss_curve.py results/bbox_estimator_voxel64_v3/log.txt
    python plot_bbox_loss_curve.py results/bbox_estimator_voxel64_v3/log.txt --smoothing 200
    python plot_bbox_loss_curve.py results/bbox_estimator_voxel64_v3/log.txt --eval-dir evals/bbox_detr_voxel64

    # Run eval on multiple checkpoints and plot metrics
    python plot_bbox_loss_curve.py results/bbox_estimator_voxel64_v3/log.txt \
        --run-eval --config configs/bbox/legacy/erp_bbox_estimator_v3.json \
        --eval-steps 10000 20000 30000 40000 49900
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_bbox_log(path):
    """Load bbox training log.txt with restart handling.

    Returns list of dicts with keys: step, loss, loss_center, loss_size,
    loss_rotation, loss_iou, loss_aux, loss_confidence, mean_confidence,
    num_matched, grad_norm, lr
    """
    all_entries = []
    last_restart_idx = 0

    with open(path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Parse "step N: {json}" summary lines
            if line.startswith('step '):
                colon_pos = line.index(':')
                step = int(line[5:colon_pos].strip())
                json_str = line[colon_pos + 1:].strip()
                try:
                    data = json.loads(json_str)
                    data['_step'] = step
                    data['_type'] = 'summary'
                    # Detect restarts
                    if step == 0 and all_entries:
                        last_restart_idx = len(all_entries)
                    all_entries.append(data)
                except json.JSONDecodeError:
                    continue
            # Parse "N: {json}" per-step lines
            else:
                parts = line.split(':', 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    step = int(parts[0].strip())
                    json_str = parts[1].strip()
                    try:
                        data = json.loads(json_str)
                        data['_step'] = step
                        data['_type'] = 'per_step'
                        # Detect restarts
                        if step == 0 and all_entries:
                            last_restart_idx = len(all_entries)
                        all_entries.append(data)
                    except json.JSONDecodeError:
                        continue

    # Take only the last run (after final restart)
    entries = all_entries[last_restart_idx:]
    print(f"Loaded {len(entries)} entries from last run (total {len(all_entries)} entries, "
          f"{last_restart_idx} entries from previous runs skipped)")

    return entries


def smooth(values, window):
    """Exponential moving average smoothing."""
    if window <= 1 or len(values) == 0:
        return values
    smoothed = []
    ema = values[0]
    alpha = 2.0 / (window + 1)
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed


def extract_series(entries):
    """Extract time series from log entries."""
    series = defaultdict(lambda: {'steps': [], 'values': []})

    loss_keys = ['loss', 'loss_center', 'loss_size', 'loss_rotation',
                 'loss_iou', 'loss_aux', 'loss_confidence']
    metric_keys = ['mean_confidence', 'num_matched', 'grad_norm', 'lr']

    for entry in entries:
        step = entry['_step']

        for key in loss_keys:
            val = entry.get(key)
            if val is not None:
                series[key]['steps'].append(step)
                series[key]['values'].append(val)

        for key in metric_keys:
            val = entry.get(key)
            if val is None:
                val = entry.get('status', {}).get(key)
            if val is not None:
                series[key]['steps'].append(step)
                series[key]['values'].append(val)

    return series


def load_eval_metrics(eval_dir):
    """Load evaluation metrics from eval checkpoint directories.

    Looks for: {eval_dir}/ckpt_step{N}/metrics.json
    Returns dict: {step: metrics_dict}
    """
    eval_dir = Path(eval_dir)
    metrics_by_step = {}

    for d in sorted(eval_dir.iterdir()):
        if d.is_dir() and d.name.startswith('ckpt_step'):
            metrics_file = d / 'metrics.json'
            if metrics_file.exists():
                step_str = d.name.replace('ckpt_step', '')
                step = int(step_str)
                with open(metrics_file) as f:
                    metrics_by_step[step] = json.load(f)

    return metrics_by_step


def plot_bbox_training(log_path, smoothing=100, output=None, eval_dir=None):
    """Plot bbox estimator training curves."""
    entries = load_bbox_log(log_path)
    if not entries:
        print("No entries found!")
        return

    series = extract_series(entries)

    # Load eval metrics if available
    eval_metrics = {}
    if eval_dir and Path(eval_dir).exists():
        eval_metrics = load_eval_metrics(eval_dir)
        if eval_metrics:
            print(f"Loaded eval metrics for steps: {sorted(eval_metrics.keys())}")

    has_eval = bool(eval_metrics)

    # Determine layout
    n_rows = 4 if has_eval else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 5 * n_rows))

    # ============================================================
    # Plot 1: Total loss
    # ============================================================
    ax = axes[0]
    s = series['loss']
    if s['values']:
        ax.plot(s['steps'], s['values'], alpha=0.1, color='C0', linewidth=0.3)
        ax.plot(s['steps'], smooth(s['values'], smoothing), color='C0',
                linewidth=2, label=f'Total loss (EMA w={smoothing})')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Step')
    ax.set_title('Total Training Loss')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ============================================================
    # Plot 2: Component losses
    # ============================================================
    ax = axes[1]
    component_keys = ['loss_center', 'loss_size', 'loss_rotation',
                      'loss_iou', 'loss_confidence']
    component_labels = {
        'loss_center': 'Center (L1)',
        'loss_size': 'Size (L1)',
        'loss_rotation': 'Rotation (1-cos)',
        'loss_iou': 'DIoU',
        'loss_confidence': 'Confidence (Focal)',
    }
    colors_comp = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for i, key in enumerate(component_keys):
        s = series[key]
        if s['values']:
            ax.plot(s['steps'], smooth(s['values'], smoothing),
                    color=colors_comp[i], linewidth=1.5, alpha=0.9,
                    label=component_labels.get(key, key))

    # Also plot aux loss on secondary y-axis
    s_aux = series['loss_aux']
    if s_aux['values']:
        ax2 = ax.twinx()
        ax2.plot(s_aux['steps'], smooth(s_aux['values'], smoothing),
                 color='gray', linewidth=1.5, alpha=0.6, linestyle='--',
                 label='Aux loss (right axis)')
        ax2.set_ylabel('Aux Loss', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.legend(fontsize=8, loc='upper right')

    ax.set_ylabel('Loss')
    ax.set_xlabel('Step')
    ax.set_title('Component Losses')
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ============================================================
    # Plot 3: Training metrics (grad_norm, mean_confidence, num_matched)
    # ============================================================
    ax = axes[2]

    # Grad norm
    s_gn = series['grad_norm']
    if s_gn['values']:
        ax.plot(s_gn['steps'], smooth(s_gn['values'], smoothing),
                color='C3', linewidth=1.5, alpha=0.8, label='Grad norm')
    ax.set_ylabel('Grad Norm', color='C3')
    ax.tick_params(axis='y', labelcolor='C3')
    ax.set_xlabel('Step')
    ax.set_title('Training Metrics')
    ax.grid(True, alpha=0.3)

    # Mean confidence + num_matched on secondary axis
    ax3 = ax.twinx()
    s_conf = series['mean_confidence']
    if s_conf['values']:
        ax3.plot(s_conf['steps'], smooth(s_conf['values'], smoothing),
                 color='C2', linewidth=1.5, alpha=0.8, label='Mean confidence')
    ax3.set_ylabel('Mean Confidence', color='C2')
    ax3.tick_params(axis='y', labelcolor='C2')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper right')

    # ============================================================
    # Plot 4: Eval metrics (if available)
    # ============================================================
    if has_eval:
        ax = axes[3]

        steps_sorted = sorted(eval_metrics.keys())

        # Recall/Precision/F1 at different IoU thresholds
        iou_thresholds = ['0.25', '0.5', '0.75']
        metric_colors = {
            '0.25': '#2ca02c',   # green
            '0.5': '#1f77b4',    # blue
            '0.75': '#d62728',   # red
        }

        for thresh in iou_thresholds:
            recall_key = f'recall@{thresh}'
            precision_key = f'precision@{thresh}'
            f1_key = f'f1@{thresh}'

            recalls = [eval_metrics[s].get(recall_key, 0) for s in steps_sorted]
            precisions = [eval_metrics[s].get(precision_key, 0) for s in steps_sorted]
            f1s = [eval_metrics[s].get(f1_key, 0) for s in steps_sorted]

            color = metric_colors[thresh]
            ax.plot(steps_sorted, recalls, 'o-', color=color, linewidth=1.5,
                    markersize=5, label=f'Recall@{thresh}')
            ax.plot(steps_sorted, precisions, 's--', color=color, linewidth=1.0,
                    markersize=4, alpha=0.6, label=f'Precision@{thresh}')
            ax.plot(steps_sorted, f1s, '^:', color=color, linewidth=1.0,
                    markersize=4, alpha=0.6, label=f'F1@{thresh}')

        # Mean IoU on secondary axis
        ax4 = ax.twinx()
        mean_ious = [eval_metrics[s].get('mean_iou', 0) for s in steps_sorted]
        ax4.plot(steps_sorted, mean_ious, 'D-', color='purple', linewidth=2,
                 markersize=6, label='Mean IoU')
        ax4.set_ylabel('Mean IoU', color='purple')
        ax4.tick_params(axis='y', labelcolor='purple')
        ax4.set_ylim(0, 1)

        ax.set_ylabel('Recall / Precision / F1')
        ax.set_xlabel('Step')
        ax.set_title('Evaluation Metrics')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax4.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
                  ncol=4, loc='lower right')

    plt.tight_layout()

    if output:
        out_path = output
    else:
        out_path = str(Path(log_path).parent / 'loss_curve.png')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.close()

    # Print final stats
    s = series['loss']
    if s['values']:
        final_loss = smooth(s['values'], smoothing)[-1]
        print(f"\nFinal smoothed loss: {final_loss:.4f}")
        print(f"Steps: {s['steps'][0]} → {s['steps'][-1]}")

    if has_eval:
        last_step = max(eval_metrics.keys())
        m = eval_metrics[last_step]
        print(f"\nEval metrics at step {last_step}:")
        for key in ['recall@0.25', 'recall@0.5', 'recall@0.75',
                     'mean_iou', 'mean_center_error', 'mean_size_error',
                     'mean_rot_error_deg']:
            if key in m:
                print(f"  {key}: {m[key]:.4f}")


def run_eval_on_checkpoints(config_path, ckpt_dir, data_dir, output_dir, steps, conf_threshold=0.5):
    """Run bbox eval on multiple checkpoints."""
    import subprocess
    import sys

    ckpt_dir = Path(ckpt_dir)
    output_dir = Path(output_dir)

    for step in steps:
        ckpt_name = f"bbox_estimator_ema0.9999_step{step:07d}.pt"
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}, skipping")
            continue

        eval_out = output_dir / f"ckpt_step{step:07d}"
        metrics_file = eval_out / "metrics.json"
        if metrics_file.exists():
            print(f"Step {step}: metrics already exist, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Running eval for step {step}: {ckpt_path}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, 'eval/bbox_inference_detr_voxel64.py',
            '--config', str(config_path),
            '--ckpt', str(ckpt_path),
            '--data_dir', str(data_dir),
            '--output_dir', str(output_dir),
            '--num_vis', '0',
            '--confidence_threshold', str(conf_threshold),
        ]
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot BBox Estimator training curves')
    parser.add_argument('log', nargs='?',
                        default='results/bbox_estimator_voxel64_v3/log.txt',
                        help='Path to log.txt')
    parser.add_argument('--smoothing', type=int, default=100,
                        help='EMA smoothing window (default: 100)')
    parser.add_argument('--output', '-o', type=str, help='Output image path')
    parser.add_argument('--eval-dir', type=str, help='Directory with eval metrics (ckpt_stepN/metrics.json)')

    # Run eval on multiple checkpoints
    parser.add_argument('--run-eval', action='store_true',
                        help='Run eval on checkpoints before plotting')
    parser.add_argument('--config', type=str, help='Config JSON for eval')
    parser.add_argument('--data-dir', type=str, default='datasets/ERP_3D_FRONT_test',
                        help='Test dataset directory')
    parser.add_argument('--eval-steps', type=int, nargs='+',
                        help='Steps to evaluate (e.g., 10000 20000 30000)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold for eval')

    args = parser.parse_args()

    # args.log = 'results/bbox_centerpoint_v2/log.txt'
    # args.smoothing = 50
    # args.output = 'results/bbox_centerpoint_v2/loss_curve.png'
    # args.eval_dir = 'results/bbox_centerpoint_v2/eval_metrics'
    # args.run_eval = True
    # args.config = 'configs/bbox/erp_bbox_centerpoint_v2.json'
    # args.data_dir = 'datasets/ERP_3D_FRONT_test'
    # args.eval_steps = [1750]
    # args.conf_threshold = 0.5

    args.log = 'results/bbox_estimator_voxel64_v3/log.txt'
    args.smoothing = 50
    args.output = 'results/bbox_estimator_voxel64_v3/loss_curve.png'
    args.eval_dir = 'results/bbox_estimator_voxel64_v3/eval_metrics'
    args.run_eval = True
    args.config = 'configs/bbox/legacy/erp_bbox_estimator_v3.json'
    args.data_dir = 'datasets/ERP_3D_FRONT_test'
    args.eval_steps = [30000, 40000, 50000]
    args.conf_threshold = 0.5

    # Run eval if requested
    if args.run_eval:
        if not args.config:
            print("Error: --config required for --run-eval")
            exit(1)
        if not args.eval_steps:
            print("Error: --eval-steps required for --run-eval")
            exit(1)

        ckpt_dir = str(Path(args.log).parent / 'ckpts')
        eval_dir = args.eval_dir or str(Path(args.log).parent / 'eval_metrics')

        run_eval_on_checkpoints(
            args.config, ckpt_dir, args.data_dir, eval_dir,
            args.eval_steps, args.conf_threshold
        )
        args.eval_dir = eval_dir

    plot_bbox_training(
        args.log,
        smoothing=args.smoothing,
        output=args.output,
        eval_dir=args.eval_dir,
    )
