# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Plot loss curves from training log.json files.

Usage:
    python plot_loss_curve.py results/erp_ss_flow_img_dit_L_16l8_bf16_spatial/log.json
    python plot_loss_curve.py results/erp_ss_flow_img_dit_L_16l8_bf16_spatial/log.json --smoothing 100
    python plot_loss_curve.py results/erp_ss_flow_img_dit_L_16l8_bf16_spatial/log.json --no-bins
    python plot_loss_curve.py log1.json log2.json --labels "Run A" "Run B"

Save: 
    results/erp_ss_flow_img_dit_L_16l8_bf16_spatial/loss_curve.png
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_log(path):
    """Load log.json, one JSON object per line."""
    steps = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Handle "N: {json}" format (line number prefix)
            if ':' in line and line.split(':')[0].strip().isdigit():
                line = ':'.join(line.split(':')[1:])
            try:
                entry = json.loads(line)
                entry['_step'] = i + 1
                steps.append(entry)
            except json.JSONDecodeError:
                continue
    return steps


def smooth(values, window):
    """Exponential moving average smoothing."""
    if window <= 1:
        return values
    smoothed = []
    ema = values[0]
    alpha = 2.0 / (window + 1)
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed


def plot_loss_curves(log_paths, labels=None, smoothing=50, show_bins=True,
                     show_grad_norm=True, output=None):
    fig_height = 5
    n_plots = 1 + (1 if show_bins else 0) + (1 if show_grad_norm else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, fig_height * n_plots),
                              squeeze=False)
    axes = axes.flatten()

    colors = plt.cm.tab10.colors

    for log_idx, log_path in enumerate(log_paths):
        entries = load_log(log_path)
        if not entries:
            print(f"No entries found in {log_path}")
            continue

        label = labels[log_idx] if labels and log_idx < len(labels) else Path(log_path).parent.name
        color = colors[log_idx % len(colors)]

        # Extract overall loss
        steps = []
        losses = []
        grad_norms = []
        bin_data = defaultdict(lambda: {'steps': [], 'values': []})

        for entry in entries:
            step = entry['_step']
            loss_dict = entry.get('loss', {})

            # Overall MSE loss
            mse = loss_dict.get('mse') or loss_dict.get('loss')
            if mse is not None:
                steps.append(step)
                losses.append(mse)

            # Grad norm
            gn = entry.get('status', {}).get('grad_norm')
            if gn is not None:
                grad_norms.append((step, gn))

            # Per-bin losses
            for key, val in loss_dict.items():
                if key.startswith('bin_') and isinstance(val, dict):
                    bin_mse = val.get('mse')
                    if bin_mse is not None:
                        bin_data[key]['steps'].append(step)
                        bin_data[key]['values'].append(bin_mse)

        # --- Plot 1: Overall loss ---
        ax = axes[0]
        ax.plot(steps, losses, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(steps, smooth(losses, smoothing), color=color, linewidth=1.5,
                label=f'{label} (smoothed, w={smoothing})')
        ax.set_ylabel('MSE Loss')
        ax.set_xlabel('Step')
        ax.set_title('Overall Training Loss')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Per-bin losses ---
        if show_bins:
            ax = axes[1]
            bin_colors = plt.cm.viridis(np.linspace(0, 1, 10))
            for bin_idx in range(10):
                key = f'bin_{bin_idx}'
                if key in bin_data and len(bin_data[key]['values']) > 10:
                    bsteps = bin_data[key]['steps']
                    bvals = bin_data[key]['values']
                    # Use larger smoothing window for sparse bin data
                    sw = max(smoothing // 3, 10)
                    ax.plot(bsteps, smooth(bvals, sw), color=bin_colors[bin_idx],
                            linewidth=1.2, alpha=0.8,
                            label=f'bin_{bin_idx} (t∈[{bin_idx/10:.1f},{(bin_idx+1)/10:.1f})')
            ax.set_ylabel('MSE Loss')
            ax.set_xlabel('Step')
            ax.set_title('Loss by Timestep Bin (bin_0=clean → bin_9=noisy)')
            ax.set_yscale('log')
            ax.legend(fontsize=8, ncol=2, loc='upper right')
            ax.grid(True, alpha=0.3)

        # --- Plot 3: Grad norm ---
        if show_grad_norm:
            ax = axes[-1]
            gn_steps, gn_vals = zip(*grad_norms) if grad_norms else ([], [])
            ax.plot(gn_steps, gn_vals, alpha=0.15, color=color, linewidth=0.5)
            ax.plot(gn_steps, smooth(list(gn_vals), smoothing), color=color,
                    linewidth=1.5, label=label)
            ax.set_ylabel('Gradient Norm')
            ax.set_xlabel('Step')
            ax.set_title('Gradient Norm')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output:
        out_path = output
    else:
        out_path = str(Path(log_paths[0]).parent / 'loss_curve.png')

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training loss curves')
    # parser.add_argument('logs', nargs='+', help='Path(s) to log.json file(s)')
    # parser.add_argument('--labels', nargs='+', help='Labels for each log file')
    parser.add_argument('--smoothing', type=int, default=50, help='EMA smoothing window (default: 50)')
    parser.add_argument('--no-bins', action='store_true', help='Hide per-bin loss plot')
    parser.add_argument('--no-grad-norm', action='store_true', help='Hide gradient norm plot')
    parser.add_argument('--output', '-o', type=str, help='Output image path')
    args = parser.parse_args()

    # args.logs = ['results/erp_ss_flow_img_dit_L_16l8_bf16_spatial/log.json']
    # args.labels = ['erp_ss_flow_img_dit_L_16l8_bf16_spatial']
    # args.smoothing = 50
    # args.no_bins = False
    # args.no_grad_norm = False
    # args.output = 'results/erp_ss_flow_img_dit_L_16l8_bf16_spatial/loss_curve.png'

    args.logs = ['results/erp_slat_flow_img2shape_asset_aware_bf16/log.json']
    args.labels = ['erp_slat_flow_img2shape_asset_aware_bf16']
    args.smoothing = 50
    args.no_bins = False
    args.no_grad_norm = False
    args.output = 'results/erp_slat_flow_img2shape_asset_aware_bf16/loss_curve.png'

    # args.logs = ['results/bbox_centerpoint_v2/log.json']
    # args.labels = ['bbox_centerpoint_v2']
    # args.smoothing = 50
    # args.no_bins = False
    # args.no_grad_norm = False
    # args.output = 'results/bbox_centerpoint_v2/loss_curve.png'

    plot_loss_curves(
        args.logs,
        labels=args.labels,
        smoothing=args.smoothing,
        show_bins=not args.no_bins,
        show_grad_norm=not args.no_grad_norm,
        output=args.output,
    )
