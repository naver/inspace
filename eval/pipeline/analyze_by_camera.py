# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Analyze metrics by camera-center-to-room-center distance.

Reads existing metrics.json (per-sample results) + camera_poses.json + normalization_info.json,
bins samples by normalized camera distance, and plots metric curves.

Usage:
    # Single experiment
    python eval/pipeline/analyze_by_camera.py \
        --pred_dirs evals/stage12_pipeline/random_gt \
        --data_dir datasets/ERP_3D_FRONT_test \
        --n_bins 8

    # Compare multiple experiments
    python eval/pipeline/analyze_by_camera.py \
        --pred_dirs evals/stage12_pipeline/random_gt \
                    evals/stage12_pipeline/random_predicted \
                    evals/stage12_pipeline/sdedit0.5_gt \
                    evals/stage12_pipeline/sdedit0.5_predicted \
        --data_dir datasets/ERP_3D_FRONT_test \
        --n_bins 8
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def get_camera_distance(data_dir, scene_id, room_id):
    """
    Compute normalized distance from camera center to room center (origin).

    Returns: float distance, or None if data missing.
    """
    cam_path = os.path.join(data_dir, scene_id, room_id, 'camera_poses.json')
    norm_path = os.path.join(data_dir, scene_id, room_id, 'dual_grid_512', 'normalization_info.json')

    if not os.path.exists(cam_path) or not os.path.exists(norm_path):
        return None

    with open(cam_path) as f:
        cam_data = json.load(f)
    with open(norm_path) as f:
        norm_data = json.load(f)

    center = np.array(norm_data['center'])
    scale = norm_data['scale']

    # Use first view (ERP has 1 view per sample)
    loc = np.array(cam_data['views'][0]['location'])
    norm_cam = (loc - center) * scale

    return float(np.linalg.norm(norm_cam))


def load_metrics_with_camera(pred_dir, data_dir):
    """
    Load per-sample metrics and attach camera distance.
    Uses camera_distance from metrics.json if available, otherwise computes from data_dir.

    Returns: list of (camera_dist, metrics_dict) tuples.
    """
    metrics_path = os.path.join(pred_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        print(f"Warning: {metrics_path} not found")
        return []

    with open(metrics_path) as f:
        data = json.load(f)

    per_sample = data.get('per_sample', {})
    results = []

    for sample_key, metrics in per_sample.items():
        # Try reading camera_distance directly from metrics
        dist = metrics.get('camera_distance', None)

        # Fallback: compute from data_dir
        if dist is None:
            parts = sample_key.split('/')
            if len(parts) != 2:
                continue
            scene_id, room_id = parts
            dist = get_camera_distance(data_dir, scene_id, room_id)

        if dist is None:
            continue

        results.append((dist, metrics))

    return results


def bin_metrics(samples, n_bins=8, bin_mode='quantile'):
    """
    Bin samples by camera distance and compute mean metrics per bin.

    Args:
        samples: list of (camera_dist, metrics_dict)
        n_bins: number of bins
        bin_mode: 'quantile' (equal count) or 'uniform' (equal width)

    Returns:
        bin_edges: list of (low, high) tuples
        bin_metrics: list of dicts {metric_name: mean_value}
        bin_counts: list of int
    """
    if not samples:
        return [], [], []

    distances = np.array([s[0] for s in samples])

    if bin_mode == 'quantile':
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(distances, percentiles)
    else:  # uniform
        edges = np.linspace(distances.min(), distances.max(), n_bins + 1)

    # Ensure unique edges
    edges = np.unique(edges)
    actual_bins = len(edges) - 1

    bin_edges = []
    bin_metrics_list = []
    bin_counts = []

    for i in range(actual_bins):
        low, high = edges[i], edges[i + 1]
        if i == actual_bins - 1:
            mask = (distances >= low) & (distances <= high)
        else:
            mask = (distances >= low) & (distances < high)

        bin_samples = [samples[j] for j in range(len(samples)) if mask[j]]
        if not bin_samples:
            continue

        bin_edges.append((float(low), float(high)))
        bin_counts.append(len(bin_samples))

        # Aggregate metrics
        agg = defaultdict(list)
        for _, metrics in bin_samples:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    agg[k].append(v)

        bin_metrics_list.append({k: float(np.mean(v)) for k, v in agg.items()})

    return bin_edges, bin_metrics_list, bin_counts


def plot_metrics_by_distance(all_experiments, output_dir, n_bins=8):
    """
    Plot metrics vs camera distance for all experiments.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Key metrics to plot
    metric_groups = {
        'Scene-level 3D': {
            'chamfer_distance': ('Chamfer Distance', True),  # (label, lower_is_better)
            'f1@0.02': ('F1@0.02', False),
            'voxel_iou': ('Voxel IoU', False),
        },
        'Asset-level 3D': {
            'asset_chamfer_distance': ('Asset CD', True),
            'asset_f1@0.02': ('Asset F1@0.02', False),
            'asset_voxel_iou': ('Asset Voxel IoU', False),
        },
        '2D Rendering': {
            'psnr_geometry_exterior': ('PSNR Ext', False),
            'psnr_geometry_topdown': ('PSNR Top', False),
            'lpips_geometry_exterior': ('LPIPS Ext', True),
            'lpips_geometry_topdown': ('LPIPS Top', True),
        },
    }

    # Nice experiment names
    name_map = {
        'random_gt': 'Random + GT SS',
        'random_predicted': 'Random + Pred SS',
        'sdedit0.5_gt': 'SDEdit + GT SS',
        'sdedit0.5_predicted': 'SDEdit + Pred SS',
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    os.makedirs(output_dir, exist_ok=True)

    for group_name, metrics_info in metric_groups.items():
        # Filter metrics that actually exist in data
        available = {}
        for mk, (label, lower_better) in metrics_info.items():
            for exp_name, (bin_edges, bin_mets, _) in all_experiments.items():
                if any(mk in bm for bm in bin_mets):
                    available[mk] = (label, lower_better)
                    break

        if not available:
            continue

        n_metrics = len(available)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

        for ax, (mk, (label, lower_better)) in zip(axes, available.items()):
            for ci, (exp_name, (bin_edges, bin_mets, bin_counts)) in enumerate(all_experiments.items()):
                display_name = name_map.get(os.path.basename(exp_name), os.path.basename(exp_name))
                color = colors[ci % len(colors)]

                x_centers = [(e[0] + e[1]) / 2 for e in bin_edges]
                y_vals = [bm.get(mk, float('nan')) for bm in bin_mets]

                # Filter out nan
                valid = [(x, y) for x, y in zip(x_centers, y_vals) if not np.isnan(y)]
                if not valid:
                    continue
                xs, ys = zip(*valid)

                ax.plot(xs, ys, 'o-', color=color, label=display_name, markersize=4, linewidth=1.5)

            ax.set_xlabel('Camera Distance to Room Center')
            ax.set_ylabel(label)
            ax.set_title(label)
            arrow = r'$\downarrow$' if lower_better else r'$\uparrow$'
            ax.set_title(f'{label} {arrow}')
            ax.grid(True, alpha=0.3)

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=min(len(handles), 4),
                       bbox_to_anchor=(0.5, 1.08), fontsize=9)

        fig.suptitle(f'{group_name} Metrics vs. Camera Distance', y=1.12, fontsize=13)
        plt.tight_layout()

        safe_name = group_name.lower().replace(' ', '_').replace('-', '_')
        fig_path = os.path.join(output_dir, f'camera_dist_{safe_name}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    # Also plot bin distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    # Use first experiment for distribution
    first_exp = list(all_experiments.values())[0]
    bin_edges, _, bin_counts = first_exp
    x_labels = [f'{e[0]:.2f}-{e[1]:.2f}' for e in bin_edges]
    ax.bar(range(len(bin_counts)), bin_counts, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Camera Distance Bin')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Distribution by Camera Distance')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'camera_dist_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {os.path.join(output_dir, 'camera_dist_distribution.png')}")


def main():
    parser = argparse.ArgumentParser(description='Analyze metrics by camera distance')
    parser.add_argument('--pred_dirs', nargs='+', required=True,
                        help='Prediction dirs (each must have metrics.json)')
    parser.add_argument('--data_dir', type=str, default='datasets/ERP_3D_FRONT_test')
    parser.add_argument('--n_bins', type=int, default=8)
    parser.add_argument('--bin_mode', choices=['quantile', 'uniform'], default='quantile',
                        help='Binning strategy: quantile (equal count) or uniform (equal width)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output dir for plots/json (default: first pred_dir parent / analysis_camera)')
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.join(os.path.dirname(args.pred_dirs[0]), 'analysis_camera')

    print(f"Analyzing {len(args.pred_dirs)} experiments")
    print(f"Bins: {args.n_bins} ({args.bin_mode})")

    all_experiments = {}
    all_json_data = {}

    for pred_dir in args.pred_dirs:
        exp_name = os.path.basename(pred_dir)
        print(f"\nLoading: {exp_name}")

        samples = load_metrics_with_camera(pred_dir, args.data_dir)
        print(f"  {len(samples)} samples with camera info")

        if not samples:
            continue

        bin_edges, bin_mets, bin_counts = bin_metrics(samples, args.n_bins, args.bin_mode)
        all_experiments[exp_name] = (bin_edges, bin_mets, bin_counts)

        # Store for JSON
        all_json_data[exp_name] = {
            'num_samples': len(samples),
            'distance_stats': {
                'min': float(min(s[0] for s in samples)),
                'max': float(max(s[0] for s in samples)),
                'mean': float(np.mean([s[0] for s in samples])),
                'median': float(np.median([s[0] for s in samples])),
            },
            'bins': [
                {
                    'range': [float(e[0]), float(e[1])],
                    'count': int(c),
                    'metrics': m,
                }
                for e, c, m in zip(bin_edges, bin_counts, bin_mets)
            ],
        }

        for i, (edge, count, mets) in enumerate(zip(bin_edges, bin_counts, bin_mets)):
            print(f"  Bin {i}: [{edge[0]:.3f}, {edge[1]:.3f}] n={count}")

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, 'analysis_by_camera.json')
    with open(json_path, 'w') as f:
        json.dump(all_json_data, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Plot
    print("\nGenerating plots...")
    plot_metrics_by_distance(all_experiments, args.output_dir, args.n_bins)

    print(f"\nDone! Results in: {args.output_dir}")


if __name__ == '__main__':
    main()
