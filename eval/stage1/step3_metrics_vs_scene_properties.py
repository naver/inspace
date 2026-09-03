# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 3: Analyze metrics vs scene properties (camera-room center distance, room size).

Computes per-sample:
  - d: Euclidean distance from camera center to room center (in O-Voxel normalized space)
  - s: Room floor area (m²) from room_info.json

Then saves all per-sample data to JSON and plots metric-vs-property graphs
with one line per model.

Usage:
    python eval/step3_metrics_vs_scene_properties.py \
        --pred_dirs \
            evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16 \
            evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial \
            evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial_da2_inversion_0.3 \
            evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial_da2_inversion_0.5 \
            evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial_da2_inversion_0.7 \
        --model_names wo_spatial spatial Da2_inv_0.3 Da2_inv_0.5 Da2_inv_0.7 \
        --data_dir datasets/ERP_3D_FRONT_test \
        --output_dir evals/results/metrics_vs_scene
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import open3d as o3d
except ImportError:
    o3d = None


# ---------------------------------------------------------------------------
# Metrics (same as step2)
# ---------------------------------------------------------------------------

def compute_voxel_metrics(gt: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    gt = gt.bool().flatten()
    pred = pred.bool().flatten()
    tp = (gt & pred).sum().float()
    fp = (~gt & pred).sum().float()
    fn = (gt & ~pred).sum().float()
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item(),
    }


def compute_chamfer_distance(gt: torch.Tensor, pred: torch.Tensor, resolution: int = 64) -> float:
    gt_coords = torch.nonzero(gt.squeeze(), as_tuple=False).float()
    pred_coords = torch.nonzero(pred.squeeze(), as_tuple=False).float()
    if gt_coords.shape[0] == 0 or pred_coords.shape[0] == 0:
        return float('inf')
    gt_coords = gt_coords / resolution
    pred_coords = pred_coords / resolution
    chunk = 4096

    def _min_dists(src, tgt):
        dists = []
        for i in range(0, src.shape[0], chunk):
            c = src[i:i + chunk]
            d = torch.cdist(c.unsqueeze(0), tgt.unsqueeze(0)).squeeze(0).min(dim=1).values
            dists.append(d)
        return torch.cat(dists).mean().item()

    return (_min_dists(gt_coords, pred_coords) + _min_dists(pred_coords, gt_coords)) / 2.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_pred_samples(pred_dir: str) -> List[dict]:
    samples = []
    if not os.path.isdir(pred_dir):
        return samples
    for uuid_dir in sorted(os.listdir(pred_dir)):
        uuid_path = os.path.join(pred_dir, uuid_dir)
        if not os.path.isdir(uuid_path) or uuid_dir.startswith('.'):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            if not os.path.isdir(room_path):
                continue
            for npz_file in sorted(glob(os.path.join(room_path, '*.npz'))):
                view_idx = int(os.path.splitext(os.path.basename(npz_file))[0])
                samples.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'view_idx': view_idx,
                    'pred_path': npz_file,
                })
    return samples


def build_lookup(samples: List[dict]) -> Dict[str, str]:
    lookup = {}
    for s in samples:
        key = f"{s['uuid']}/{s['room_name']}/{s['view_idx']:04d}"
        lookup[key] = s['pred_path']
    return lookup


def load_pred_voxel(pred_path: str) -> Optional[torch.Tensor]:
    if not os.path.exists(pred_path):
        return None
    data = np.load(pred_path)
    return torch.tensor(data['voxel'])


def load_gt_voxel(data_dir, uuid, room_name, gt_voxel_folder='voxels_64'):
    path = os.path.join(data_dir, uuid, room_name, gt_voxel_folder, 'full_room_wo_ceiling.ply')
    if not os.path.exists(path):
        return None
    if o3d is None:
        return None
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        return None
    resolution = 64
    indices = np.round((pts + 0.5) * resolution - 0.5).astype(np.int64)
    indices = np.clip(indices, 0, resolution - 1)
    grid = torch.zeros(1, resolution, resolution, resolution, dtype=torch.bool)
    grid[0, indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return grid


# ---------------------------------------------------------------------------
# Scene property computation
# ---------------------------------------------------------------------------

def load_scene_properties(data_dir: str, uuid: str, room_name: str, view_idx: int) -> Optional[dict]:
    """Load camera center distance d and room size s for a sample.

    d: Euclidean distance from camera center to room center in O-Voxel normalized space.
       Room center in normalized space = origin (0,0,0) by construction.
       So d = ||normalized_camera_center||.

    s: Room floor area in m² from room_info.json.

    Returns dict with 'd', 's', 'd_xy' (horizontal only), 'room_diagonal'.
    """
    room_path = os.path.join(data_dir, uuid, room_name)

    cp_path = os.path.join(room_path, 'camera_poses.json')
    ni_path = os.path.join(room_path, 'mesh_dumps', 'normalization_info.json')
    ri_path = os.path.join(room_path, 'room_info.json')

    if not os.path.exists(cp_path) or not os.path.exists(ni_path):
        return None

    with open(cp_path) as f:
        cp = json.load(f)
    with open(ni_path) as f:
        ni = json.load(f)

    # Find camera location for this view
    loc = None
    for v in cp.get('views', []):
        if v.get('view_idx') == view_idx:
            loc = v.get('location')
            break
    if loc is None:
        return None

    # Camera center in normalized space
    center = np.array(ni['center'], dtype=np.float64)
    scale = float(ni['scale'])
    cam_world = np.array(loc, dtype=np.float64)
    cam_norm = (cam_world - center) * scale  # in [-0.5, 0.5]

    # Distance in normalized space
    d = float(np.linalg.norm(cam_norm))
    d_xy = float(np.linalg.norm(cam_norm[:2]))

    # Room size
    s = None
    room_diagonal = None
    if os.path.exists(ri_path):
        with open(ri_path) as f:
            ri = json.load(f)
        s = float(ri.get('area', 0))
        min_c = np.array(ri.get('min_corner', [0, 0, 0]), dtype=np.float64)
        max_c = np.array(ri.get('max_corner', [0, 0, 0]), dtype=np.float64)
        # Floor diagonal length (XY plane)
        room_diagonal = float(np.linalg.norm(max_c[:2] - min_c[:2]))
    else:
        # Fallback: compute from normalization_info bbox
        bbox_min = np.array(ni.get('bbox_min', [0, 0, 0]), dtype=np.float64)
        bbox_max = np.array(ni.get('bbox_max', [0, 0, 0]), dtype=np.float64)
        # Approximate area from XY extent
        extent = bbox_max - bbox_min
        s = float(extent[0] * extent[1])
        room_diagonal = float(np.linalg.norm(extent[:2]))

    return {
        'd': d,
        'd_xy': d_xy,
        's': s,
        'room_diagonal': room_diagonal,
        'cam_norm': cam_norm.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

MODEL_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
MODEL_MARKERS = ['o', 's', '^', 'D', 'v']


def plot_metric_vs_property(
    records: List[dict],
    model_names: List[str],
    prop_key: str,
    prop_label: str,
    metric_key: str,
    metric_label: str,
    output_path: str,
    n_bins: int = 10,
    figsize: tuple = (8, 5),
    prop_max: float = None,
):
    """Plot binned average of metric vs scene property for all models.

    Groups samples into n_bins equal-width bins along prop_key,
    computes mean+std of metric_key per bin per model.
    """
    n_models = len(model_names)

    # Collect property values to determine bin edges
    prop_vals = [r[prop_key] for r in records if r[prop_key] is not None]
    if prop_max is not None:
        prop_vals = [v for v in prop_vals if v <= prop_max]
    if len(prop_vals) < 5:
        return
    pmin, pmax = min(prop_vals), max(prop_vals)
    if pmax - pmin < 1e-8:
        return
    bin_edges = np.linspace(pmin, pmax, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig, ax = plt.subplots(figsize=figsize)

    for mi in range(n_models):
        mk = f'm{mi}_{metric_key}'
        bin_means = []
        bin_stds = []
        bin_counts = []
        valid_centers = []

        for bi in range(n_bins):
            lo, hi = bin_edges[bi], bin_edges[bi + 1]
            if bi == n_bins - 1:
                vals = [r[mk] for r in records
                        if r.get(prop_key) is not None and lo <= r[prop_key] <= hi
                        and mk in r and r[mk] != float('inf') and not np.isnan(r[mk])]
            else:
                vals = [r[mk] for r in records
                        if r.get(prop_key) is not None and lo <= r[prop_key] < hi
                        and mk in r and r[mk] != float('inf') and not np.isnan(r[mk])]

            if len(vals) >= 3:  # minimum samples per bin
                # Trim top/bottom 5% outliers
                vals_arr = np.array(vals)
                lo_pct, hi_pct = np.percentile(vals_arr, [5, 95])
                trimmed = vals_arr[(vals_arr >= lo_pct) & (vals_arr <= hi_pct)]
                if len(trimmed) >= 3:
                    bin_means.append(np.mean(trimmed))
                    bin_stds.append(np.std(trimmed))
                else:
                    bin_means.append(np.mean(vals))
                    bin_stds.append(np.std(vals))
                bin_counts.append(len(vals))
                valid_centers.append(bin_centers[bi])

        if not valid_centers:
            continue

        valid_centers = np.array(valid_centers)
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)

        color = MODEL_COLORS[mi % len(MODEL_COLORS)]
        marker = MODEL_MARKERS[mi % len(MODEL_MARKERS)]

        ax.plot(valid_centers, bin_means, color=color, marker=marker,
                markersize=7, linewidth=2.5, label=model_names[mi])

    ax.set_xlabel(prop_label, fontsize=16)
    ax.set_ylabel(metric_label, fontsize=16)
    ax.set_title(f'{metric_label} vs {prop_label}', fontsize=17)
    ax.legend(fontsize=10, loc='upper right')
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close(fig)


def plot_scatter_with_trend(
    records: List[dict],
    model_names: List[str],
    prop_key: str,
    prop_label: str,
    metric_key: str,
    metric_label: str,
    output_path: str,
    figsize: tuple = (8, 5),
):
    """Scatter plot + polynomial trend line of metric vs property per model."""
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=figsize)

    for mi in range(n_models):
        mk = f'm{mi}_{metric_key}'
        xs = []
        ys = []
        for r in records:
            if (r.get(prop_key) is not None and mk in r
                    and r[mk] != float('inf') and not np.isnan(r[mk])):
                xs.append(r[prop_key])
                ys.append(r[mk])

        if len(xs) < 5:
            continue

        xs = np.array(xs)
        ys = np.array(ys)
        color = MODEL_COLORS[mi % len(MODEL_COLORS)]

        # Scatter (small, semi-transparent)
        ax.scatter(xs, ys, color=color, alpha=0.15, s=8, edgecolors='none')

        # Polynomial trend (degree 2)
        try:
            coeffs = np.polyfit(xs, ys, deg=2)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_line, color=color, linewidth=2.5, label=model_names[mi])
        except Exception:
            pass

    ax.set_xlabel(prop_label, fontsize=16)
    ax.set_ylabel(metric_label, fontsize=16)
    ax.set_title(f'{metric_label} vs {prop_label}', fontsize=17)
    ax.legend(fontsize=10, loc='best')
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close(fig)


def plot_combined_metrics_grid(
    records: List[dict],
    model_names: List[str],
    prop_key: str,
    prop_label: str,
    output_path: str,
    n_bins: int = 10,
):
    """Plot a 2x3 grid of all metrics vs one property."""
    metrics = [
        ('iou', 'Voxel IoU'),
        ('dice', 'Dice'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('chamfer', 'Chamfer Distance'),
    ]
    n_models = len(model_names)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes_flat = axes.flatten()

    # Bin edges from property values
    prop_vals = [r[prop_key] for r in records if r[prop_key] is not None]
    if len(prop_vals) < 5:
        plt.close(fig)
        return
    pmin, pmax = min(prop_vals), max(prop_vals)
    if pmax - pmin < 1e-8:
        plt.close(fig)
        return
    bin_edges = np.linspace(pmin, pmax, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for ai, (mkey, mlabel) in enumerate(metrics):
        ax = axes_flat[ai]

        for mi in range(n_models):
            mk = f'm{mi}_{mkey}'
            bin_means = []
            valid_centers = []

            for bi in range(n_bins):
                lo, hi = bin_edges[bi], bin_edges[bi + 1]
                if bi == n_bins - 1:
                    vals = [r[mk] for r in records
                            if r.get(prop_key) is not None and lo <= r[prop_key] <= hi
                            and mk in r and r[mk] != float('inf') and not np.isnan(r[mk])]
                else:
                    vals = [r[mk] for r in records
                            if r.get(prop_key) is not None and lo <= r[prop_key] < hi
                            and mk in r and r[mk] != float('inf') and not np.isnan(r[mk])]

                if len(vals) >= 3:
                    vals_arr = np.array(vals)
                    lo_pct, hi_pct = np.percentile(vals_arr, [5, 95])
                    trimmed = vals_arr[(vals_arr >= lo_pct) & (vals_arr <= hi_pct)]
                    if len(trimmed) >= 3:
                        bin_means.append(np.mean(trimmed))
                    else:
                        bin_means.append(np.mean(vals))
                    valid_centers.append(bin_centers[bi])

            if not valid_centers:
                continue

            color = MODEL_COLORS[mi % len(MODEL_COLORS)]
            marker = MODEL_MARKERS[mi % len(MODEL_MARKERS)]
            ax.plot(valid_centers, bin_means, color=color, marker=marker,
                    markersize=6, linewidth=2.5, label=model_names[mi])

        ax.set_xlabel(prop_label, fontsize=14)
        ax.set_ylabel(mlabel, fontsize=14)
        ax.set_title(mlabel, fontsize=15)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3)
        if ai == 0:
            ax.legend(fontsize=11, loc='best')

    # Hide last subplot if odd number of metrics
    if len(metrics) < len(axes_flat):
        for ai in range(len(metrics), len(axes_flat)):
            axes_flat[ai].set_visible(False)

    # Add sample count histogram in the last subplot
    last_ax = axes_flat[-1]
    last_ax.set_visible(True)
    counts = []
    for bi in range(n_bins):
        lo, hi = bin_edges[bi], bin_edges[bi + 1]
        if bi == n_bins - 1:
            c = sum(1 for r in records if r.get(prop_key) is not None and lo <= r[prop_key] <= hi)
        else:
            c = sum(1 for r in records if r.get(prop_key) is not None and lo <= r[prop_key] < hi)
        counts.append(c)
    last_ax.bar(bin_centers, counts, width=(pmax - pmin) / n_bins * 0.8,
                color='gray', alpha=0.6, edgecolor='black', linewidth=0.5)
    last_ax.set_xlabel(prop_label, fontsize=14)
    last_ax.set_ylabel('Sample Count', fontsize=14)
    last_ax.set_title('Sample Distribution', fontsize=15)
    last_ax.tick_params(axis='both', labelsize=12)
    last_ax.grid(True, alpha=0.3)

    fig.suptitle(f'Metrics vs {prop_label}', fontsize=18, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Analyze metrics vs scene properties')
    parser.add_argument('--pred_dirs', type=str, nargs='+', default=[])
    parser.add_argument('--model_names', type=str, nargs='*', default=[])
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=0, help='Max samples (0=all)')
    parser.add_argument('--gt_voxel_folder', type=str, default='voxels_64')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of bins for binned plots')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--skip_metrics', action='store_true',
                        help='Load metrics from existing JSON instead of recomputing')
    args = parser.parse_args()
    args.skip_metrics = True
    args.num_samples = 200
    args.output_dir = "evals/results/metrics_vs_scene"

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Defaults
    if not args.data_dir:
        args.data_dir = "datasets/ERP_3D_FRONT_test"
    if not args.pred_dirs:
        args.pred_dirs = [
            "evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16",
            "evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial",
            "evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial_da2_inversion_0.3",
            "evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial_da2_inversion_0.5",
            "evals/ss_generated/erp_ss_flow_img_dit_L_16l8_bf16_spatial_da2_inversion_0.7",
        ]
    if not args.model_names:
        args.model_names = [
            "w/o VS-CrossAttn",
            "w/ VS-CrossAttn",
            r"w/ VS-CrossAttn + $t_0$=0.3",
            r"w/ VS-CrossAttn + $t_0$=0.5",
            r"w/ VS-CrossAttn + $t_0$=0.7",
        ]
    if not args.output_dir:
        args.output_dir = "evals/results/metrics_vs_scene"

    n_models = len(args.pred_dirs)
    model_names = args.model_names[:n_models]
    while len(model_names) < n_models:
        model_names.append(os.path.basename(args.pred_dirs[len(model_names)].rstrip('/')))

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, 'per_sample_data.json')

    # ------------------------------------------------------------------
    # Either load existing data or compute from scratch
    # ------------------------------------------------------------------
    if args.skip_metrics and os.path.exists(json_path):
        print(f"Loading existing data from {json_path}...")
        with open(json_path) as f:
            saved = json.load(f)
        records = saved['records']
        n_models = saved.get('n_models', len(model_names))
        # Use args model_names (allow renaming without recomputing)
        print(f"Loaded {len(records)} records, {n_models} models.")
    else:
        # Find samples
        all_lookups = []
        ref_samples = None
        for mi, pred_dir in enumerate(args.pred_dirs):
            print(f"Finding predictions for model {mi}: {model_names[mi]}...")
            samples = find_pred_samples(pred_dir)
            print(f"  {len(samples)} samples")
            all_lookups.append(build_lookup(samples))
            if ref_samples is None:
                ref_samples = samples

        if not ref_samples:
            print("No samples found.")
            return

        if args.num_samples > 0:
            ref_samples = ref_samples[:args.num_samples]

        # Compute metrics + scene properties
        records = []
        gt_cache = {}
        scene_prop_cache = {}

        for sample in tqdm(ref_samples, desc="Computing metrics & scene properties"):
            uuid = sample['uuid']
            room_name = sample['room_name']
            view_idx = sample['view_idx']
            key = f"{uuid}/{room_name}/{view_idx:04d}"

            # Load GT voxel (cached per room)
            gt_key = f"{uuid}/{room_name}"
            if gt_key not in gt_cache:
                gt_cache[gt_key] = load_gt_voxel(
                    args.data_dir, uuid, room_name,
                    gt_voxel_folder=args.gt_voxel_folder,
                )
            gt_voxel = gt_cache[gt_key]
            if gt_voxel is None:
                continue

            # Load scene properties (cached per room+view)
            sp_key = f"{uuid}/{room_name}/{view_idx}"
            if sp_key not in scene_prop_cache:
                scene_prop_cache[sp_key] = load_scene_properties(
                    args.data_dir, uuid, room_name, view_idx)
            scene_props = scene_prop_cache[sp_key]

            record = {
                'uuid': uuid,
                'room_name': room_name,
                'view_idx': view_idx,
                'key': key,
                'd': scene_props['d'] if scene_props else None,
                'd_xy': scene_props['d_xy'] if scene_props else None,
                's': scene_props['s'] if scene_props else None,
                'room_diagonal': scene_props['room_diagonal'] if scene_props else None,
            }

            any_valid = False
            for mi in range(n_models):
                prefix = f'm{mi}'
                pred_path = all_lookups[mi].get(key)
                if pred_path is None:
                    continue
                pred_voxel = load_pred_voxel(pred_path)
                if pred_voxel is None:
                    continue
                any_valid = True
                metrics = compute_voxel_metrics(gt_voxel, pred_voxel)
                cd = compute_chamfer_distance(gt_voxel, pred_voxel)
                record[f'{prefix}_iou'] = metrics['iou']
                record[f'{prefix}_dice'] = metrics['dice']
                record[f'{prefix}_precision'] = metrics['precision']
                record[f'{prefix}_recall'] = metrics['recall']
                record[f'{prefix}_chamfer'] = cd

            if not any_valid:
                continue
            records.append(record)

        del gt_cache
        torch.cuda.empty_cache()

        if len(records) == 0:
            print("No valid samples found.")
            return

        # Save to JSON
        save_data = {
            'model_names': model_names,
            'n_models': n_models,
            'n_records': len(records),
            'records': records,
        }
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved {len(records)} records to {json_path}")

    # ------------------------------------------------------------------
    # Print summary statistics for d and s
    # ------------------------------------------------------------------
    d_vals = [r['d'] for r in records if r['d'] is not None]
    s_vals = [r['s'] for r in records if r['s'] is not None]
    print(f"\nScene property statistics:")
    if d_vals:
        print(f"  d (cam-room center dist): min={min(d_vals):.4f}, max={max(d_vals):.4f}, "
              f"mean={np.mean(d_vals):.4f}, std={np.std(d_vals):.4f}")
    if s_vals:
        print(f"  s (room area m²):         min={min(s_vals):.1f}, max={max(s_vals):.1f}, "
              f"mean={np.mean(s_vals):.1f}, std={np.std(s_vals):.1f}")

    # Print per-model metric means
    print(f"\nOverall metric means:")
    for mi, mn in enumerate(model_names):
        mk_iou = f'm{mi}_iou'
        iou_vals = [r[mk_iou] for r in records if mk_iou in r]
        mk_cd = f'm{mi}_chamfer'
        cd_vals = [r[mk_cd] for r in records if mk_cd in r and r[mk_cd] != float('inf')]
        print(f"  {mn}: IoU={np.mean(iou_vals):.4f}, Chamfer={np.mean(cd_vals):.6f}" if cd_vals else
              f"  {mn}: IoU={np.mean(iou_vals):.4f}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plot_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    metrics_to_plot = [
        ('iou', 'Voxel IoU'),
        ('dice', 'Dice'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('chamfer', 'Chamfer Distance'),
    ]

    # prop_max: upper clamp for each property (None = no clamp)
    properties_to_plot = [
        ('d', 'Camera-Room Center Distance (normalized)', 0.3),
        ('d_xy', 'Camera-Room Center Distance XY (normalized)', 0.3),
        ('s', 'Room Floor Area (m²)', 50.0),
        ('room_diagonal', 'Room Floor Diagonal (m)', 12.0),
    ]

    # 1. Combined grid plots (one per property)
    print("\nGenerating combined grid plots...")
    for prop_key, prop_label, pmax in properties_to_plot:
        plot_combined_metrics_grid(
            records, model_names, prop_key, prop_label,
            os.path.join(plot_dir, f'grid_{prop_key}.png'),
            n_bins=args.n_bins,
        )

    # 2. Individual binned line plots
    print("Generating individual binned plots...")
    for prop_key, prop_label, pmax in properties_to_plot:
        for mkey, mlabel in metrics_to_plot:
            plot_metric_vs_property(
                records, model_names, prop_key, prop_label, mkey, mlabel,
                os.path.join(plot_dir, f'binned_{prop_key}_vs_{mkey}.png'),
                n_bins=args.n_bins,
                prop_max=pmax,
            )

    # 3. Scatter + trend plots
    print("Generating scatter + trend plots...")
    for prop_key, prop_label, _pmax in properties_to_plot:
        for mkey, mlabel in metrics_to_plot:
            plot_scatter_with_trend(
                records, model_names, prop_key, prop_label, mkey, mlabel,
                os.path.join(plot_dir, f'scatter_{prop_key}_vs_{mkey}.png'),
            )

    print(f"\nDone! Results saved to: {args.output_dir}/")
    print(f"  Data: {json_path}")
    print(f"  Plots: {plot_dir}/")


if __name__ == '__main__':
    main()
