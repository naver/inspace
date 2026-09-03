# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
3D Bounding Box Estimation - Inference and Evaluation (CenterPoint, Voxel-64 input).

OBB-aware version: in addition to AABB-based 3D/2D IoU metrics (which ignore yaw),
this script also reports OBB-based 3D/2D IoU metrics using:
  - 3D OBB IoU: pytorch3d.ops.box3d_overlap (analytical fallback when unavailable)
  - 2D OBB IoU: top-down rotated polygon intersection (Sutherland-Hodgman clipping)

Usage:
    python eval/bbox_inference_centerpoint_obb.py \
        --config configs/bbox/erp_bbox_centerpoint.json \
        --ckpt results/bbox_centerpoint/ckpts/bbox_centerpoint_ema0.9999_step0050000.pt \
        --data_dir datasets/ERP_3D_FRONT_test \
        --output_dir evals/bbox_centerpoint_obb \
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


# ---------------------------------------------------------------------------
# OBB IoU: pytorch3d (3D, general) + Sutherland-Hodgman (2D, top-down)
# ---------------------------------------------------------------------------

try:
    from pytorch3d.ops import box3d_overlap as _pt3d_box3d_overlap
    _HAS_PYTORCH3D = True
except Exception as _e:
    _HAS_PYTORCH3D = False
    print(f'[OBB] pytorch3d not available ({_e}). Falling back to analytical 3D OBB IoU '
          f'(exact for yaw-only rotation).')


def boxes_to_corners_3d(boxes):
    """
    Convert OBB [N, 7] (cx, cy, cz, sx, sy, sz, yaw) -> corners [N, 8, 3].

    Corner ordering follows pytorch3d's box3d_overlap convention:
            (4) +---------+ (5)
                | ` .     |  ` .
                | (0) +---+-----+ (1)
                |     |   |     |
            (7) +-----+---+ (6) |
                ` .   |     ` . |
                (3) ` +---------+ (2)
        0,1,2,3 = bottom face (z = -sz/2),  4,5,6,7 = top face (z = +sz/2).
        0->4, 1->5, 2->6, 3->7 along +z.
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes).float()
    boxes = boxes.float()

    centers = boxes[:, :3]            # [N, 3]
    sizes = boxes[:, 3:6]             # [N, 3]
    yaws = boxes[:, 6]                # [N]

    template = torch.tensor([
        [-0.5, -0.5, -0.5],  # 0
        [+0.5, -0.5, -0.5],  # 1
        [+0.5, +0.5, -0.5],  # 2
        [-0.5, +0.5, -0.5],  # 3
        [-0.5, -0.5, +0.5],  # 4
        [+0.5, -0.5, +0.5],  # 5
        [+0.5, +0.5, +0.5],  # 6
        [-0.5, +0.5, +0.5],  # 7
    ], dtype=boxes.dtype, device=boxes.device)  # [8, 3]

    local = template.unsqueeze(0) * sizes.unsqueeze(1)  # [N, 8, 3]

    cos_y = torch.cos(yaws).view(-1, 1)  # [N, 1]
    sin_y = torch.sin(yaws).view(-1, 1)  # [N, 1]

    rx = local[..., 0] * cos_y - local[..., 1] * sin_y
    ry = local[..., 0] * sin_y + local[..., 1] * cos_y
    rz = local[..., 2]
    rotated = torch.stack([rx, ry, rz], dim=-1)         # [N, 8, 3]
    corners = rotated + centers.unsqueeze(1)            # [N, 8, 3]
    return corners


# ---- 2D OBB IoU on top-down (XY) plane via Sutherland-Hodgman -------------

def _polygon_clip_sh(subject, clipper):
    """
    Sutherland-Hodgman convex polygon clipping.
    Both `subject` and `clipper` are lists of (x, y) tuples in CCW order.
    Returns the clipped polygon (list of (x, y)).
    """
    def inside(p, a, b):
        # True iff p is on the left of edge a->b (CCW convention)
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

    def line_intersection(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = a;  x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-12:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = list(subject)
    n_clip = len(clipper)
    for i in range(n_clip):
        if not output:
            break
        input_polygon = output
        output = []
        a = clipper[i]
        b = clipper[(i + 1) % n_clip]
        for j in range(len(input_polygon)):
            curr = input_polygon[j]
            prev = input_polygon[j - 1]
            curr_in = inside(curr, a, b)
            prev_in = inside(prev, a, b)
            if curr_in:
                if not prev_in:
                    output.append(line_intersection(prev, curr, a, b))
                output.append(curr)
            elif prev_in:
                output.append(line_intersection(prev, curr, a, b))
    return output


def _polygon_area(poly):
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def _obb_corners_xy(box):
    """box: array-like [7] -> 4 corners on XY plane in CCW order."""
    cx = float(box[0]); cy = float(box[1])
    sx = float(box[3]); sy = float(box[4])
    yaw = float(box[6])
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    hw, hh = sx * 0.5, sy * 0.5
    locals_ = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + lx * cos_y - ly * sin_y, cy + lx * sin_y + ly * cos_y) for lx, ly in locals_]


def obb_iou_2d_xy_pair(box1, box2):
    """OBB IoU on XY (top-down) plane for a single pair."""
    p1 = _obb_corners_xy(box1)
    p2 = _obb_corners_xy(box2)
    inter = _polygon_area(_polygon_clip_sh(p1, p2))
    a1 = _polygon_area(p1)
    a2 = _polygon_area(p2)
    union = a1 + a2 - inter
    return float(inter / union) if union > 1e-12 else 0.0


def obb_iou_2d_xy_inter_area_pair(box1, box2):
    """Returns the intersection AREA (not IoU) on XY plane for a pair."""
    p1 = _obb_corners_xy(box1)
    p2 = _obb_corners_xy(box2)
    return _polygon_area(_polygon_clip_sh(p1, p2))


def obb_iou_2d_matrix(boxes1, boxes2):
    """[N, 7], [M, 7] -> [N, M] 2D OBB IoU matrix on XY plane (top-down)."""
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    N, M = boxes1.shape[0], boxes2.shape[0]
    iou = np.zeros((N, M), dtype=np.float32)
    # Precompute corners and areas
    corners1 = [_obb_corners_xy(boxes1[i]) for i in range(N)]
    corners2 = [_obb_corners_xy(boxes2[j]) for j in range(M)]
    areas1 = np.array([_polygon_area(c) for c in corners1])
    areas2 = np.array([_polygon_area(c) for c in corners2])
    for i in range(N):
        for j in range(M):
            inter = _polygon_area(_polygon_clip_sh(corners1[i], corners2[j]))
            union = areas1[i] + areas2[j] - inter
            iou[i, j] = float(inter / union) if union > 1e-12 else 0.0
    return iou


def obb_iou_3d_matrix(boxes1, boxes2):
    """
    [N, 7], [M, 7] -> [N, M] 3D OBB IoU.

    Primary: pytorch3d.ops.box3d_overlap (general 3D oriented-box IoU).
    Fallback: yaw-only analytical formula
                 inter_volume = (2D XY OBB intersection area) * (1D Z overlap length)
                 union_volume = vol1 + vol2 - inter_volume
              This is exact when boxes only rotate around the Z axis (yaw),
              which is the case for our scene representation.
    """
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    N, M = boxes1.shape[0], boxes2.shape[0]

    if _HAS_PYTORCH3D:
        try:
            c1 = boxes_to_corners_3d(torch.from_numpy(boxes1).float())
            c2 = boxes_to_corners_3d(torch.from_numpy(boxes2).float())
            # pytorch3d returns (intersection_volume, iou), each [N, M]
            _, iou = _pt3d_box3d_overlap(c1, c2)
            return iou.cpu().numpy().astype(np.float32)
        except Exception as e:
            # Numerical issues (e.g., non-coplanar faces under fp). Fall through.
            print(f'[OBB] pytorch3d.box3d_overlap failed ({e}); using analytical fallback.')

    # Analytical fallback: yaw-only -> exact via XY OBB area * Z overlap
    z1_min = boxes1[:, 2] - boxes1[:, 5] * 0.5
    z1_max = boxes1[:, 2] + boxes1[:, 5] * 0.5
    z2_min = boxes2[:, 2] - boxes2[:, 5] * 0.5
    z2_max = boxes2[:, 2] + boxes2[:, 5] * 0.5
    z_inter = np.clip(
        np.minimum(z1_max[:, None], z2_max[None, :]) -
        np.maximum(z1_min[:, None], z2_min[None, :]),
        a_min=0.0, a_max=None,
    )  # [N, M]

    # 2D XY OBB intersection area (not IoU); also compute footprint areas
    corners1 = [_obb_corners_xy(boxes1[i]) for i in range(N)]
    corners2 = [_obb_corners_xy(boxes2[j]) for j in range(M)]
    areas1 = np.array([_polygon_area(c) for c in corners1])  # XY footprint
    areas2 = np.array([_polygon_area(c) for c in corners2])  # XY footprint

    inter_xy = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            inter_xy[i, j] = _polygon_area(_polygon_clip_sh(corners1[i], corners2[j]))

    inter_vol = inter_xy * z_inter
    vol1 = areas1 * (z1_max - z1_min)  # [N]
    vol2 = areas2 * (z2_max - z2_min)  # [M]
    union_vol = vol1[:, None] + vol2[None, :] - inter_vol
    iou = np.where(union_vol > 1e-12, inter_vol / np.maximum(union_vol, 1e-12), 0.0)
    return iou.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference + AABB 2D IoU helper (kept from original)
# ---------------------------------------------------------------------------

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

    for batch in tqdm(dataloader, desc='Inference'):
        voxel_grid = batch['voxel_grid'].to(device)
        gt_bboxes = batch['gt_bboxes']
        gt_mask = batch['gt_mask']
        sample_ids = batch['sample_id']

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(voxel_grid)
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
    """AABB 2D IoU on XY plane (rotation-ignored)."""
    min1 = boxes1[:, :2] - boxes1[:, 3:5] / 2
    max1 = boxes1[:, :2] + boxes1[:, 3:5] / 2
    min2 = boxes2[:, :2] - boxes2[:, 3:5] / 2
    max2 = boxes2[:, :2] + boxes2[:, 3:5] / 2

    inter_min = torch.max(min1.unsqueeze(1), min2.unsqueeze(0))
    inter_max = torch.min(max1.unsqueeze(1), max2.unsqueeze(0))
    inter_size = (inter_max - inter_min).clamp(min=0)
    inter_area = inter_size.prod(-1)

    area1 = boxes1[:, 3] * boxes1[:, 4]
    area2 = boxes2[:, 3] * boxes2[:, 4]
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    return inter_area / union_area.clamp(min=1e-8)


# ---------------------------------------------------------------------------
# Metrics: AABB 3D/2D + OBB 3D/2D
# ---------------------------------------------------------------------------

def _greedy_match_metrics(results, iou_mat_fn, iou_thresholds, key_prefix,
                          collect_per_match=False, scale_cache=None):
    """
    Generic greedy-matching metric loop. Reuses prediction/GT ordering already
    sorted by confidence in run_inference.

    Args:
        iou_mat_fn(pred_arr, gt_arr) -> [N_pred, N_gt] IoU matrix (numpy float).
    Returns dict: {f'{key_prefix}_recall@t', f'{key_prefix}_precision@t', f'{key_prefix}_f1@t'}
        plus per-match stats if collect_per_match.
    """
    out = {}
    per_match = {
        'iou': [],
        'center_3d': [], 'center_2d': [],
        'center_3d_m': [], 'center_2d_m': [],
        'size': [], 'size_m': [],
        'rot': [],
    }

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

            iou_mat = iou_mat_fn(r['pred_bboxes'], r['gt_bboxes'])
            gt_matched = np.zeros(n_gt, dtype=bool)
            for pi in range(n_pred):
                best_gt = int(iou_mat[pi].argmax())
                if iou_mat[pi, best_gt] >= thresh and not gt_matched[best_gt]:
                    gt_matched[best_gt] = True
                    total_tp += 1

                    if collect_per_match and thresh == iou_thresholds[0]:
                        pred_box = r['pred_bboxes'][pi]
                        gt_box = r['gt_bboxes'][best_gt]
                        per_match['iou'].append(float(iou_mat[pi, best_gt]))
                        ce_3d = float(np.linalg.norm(pred_box[:3] - gt_box[:3]))
                        ce_2d = float(np.linalg.norm(pred_box[:2] - gt_box[:2]))
                        se = float(np.mean(np.abs(pred_box[3:6] - gt_box[3:6])))
                        per_match['center_3d'].append(ce_3d)
                        per_match['center_2d'].append(ce_2d)
                        per_match['size'].append(se)
                        if scale_cache is not None:
                            scale = scale_cache.get(r['sample_id'])
                            if scale is not None and scale > 0:
                                per_match['center_3d_m'].append(ce_3d / scale)
                                per_match['center_2d_m'].append(ce_2d / scale)
                                per_match['size_m'].append(se / scale)
                        angle_diff = (pred_box[6] - gt_box[6] + np.pi) % (2 * np.pi) - np.pi
                        per_match['rot'].append(float(np.abs(angle_diff)))

        if total_gt > 0:
            out[f'{key_prefix}_recall@{thresh}'] = total_tp / total_gt
        if total_pred > 0:
            out[f'{key_prefix}_precision@{thresh}'] = total_tp / total_pred
        if total_gt > 0 and total_pred > 0:
            prec = total_tp / total_pred
            rec = total_tp / total_gt
            if prec + rec > 0:
                out[f'{key_prefix}_f1@{thresh}'] = 2 * prec * rec / (prec + rec)

    return out, per_match


def compute_metrics(results, iou_thresholds=[0.25, 0.5, 0.75], data_dir=None):
    """Compute AABB & OBB detection metrics (3D/2D)."""
    metrics = {}

    # Real-world scale cache for center/size errors in meters
    scale_cache = {}
    if data_dir is not None:
        for r in results:
            sid = r['sample_id']
            ninfo_path = os.path.join(data_dir, sid, 'dual_grid_512', 'normalization_info.json')
            if os.path.exists(ninfo_path):
                with open(ninfo_path) as f:
                    scale_cache[sid] = json.load(f)['scale']

    # ---- AABB 3D IoU ----
    def _aabb_3d(pred, gt):
        return bbox3d_iou(torch.from_numpy(pred[:, :6]).float(),
                          torch.from_numpy(gt[:, :6]).float()).numpy()

    aabb3d_out, aabb3d_pm = _greedy_match_metrics(
        results, _aabb_3d, iou_thresholds, key_prefix='3d',
        collect_per_match=True, scale_cache=scale_cache)
    metrics.update(aabb3d_out)

    # ---- AABB 2D IoU (top-down) ----
    def _aabb_2d(pred, gt):
        return bbox2d_iou(torch.from_numpy(pred[:, :6]).float(),
                          torch.from_numpy(gt[:, :6]).float()).numpy()

    aabb2d_out, _ = _greedy_match_metrics(
        results, _aabb_2d, iou_thresholds, key_prefix='2d',
        collect_per_match=False)
    metrics.update(aabb2d_out)

    # ---- OBB 3D IoU (rotation-aware) ----
    obb3d_out, obb3d_pm = _greedy_match_metrics(
        results, obb_iou_3d_matrix, iou_thresholds, key_prefix='3d_obb',
        collect_per_match=True, scale_cache=scale_cache)
    metrics.update(obb3d_out)

    # ---- OBB 2D IoU (top-down rotated polygon) ----
    obb2d_out, obb2d_pm = _greedy_match_metrics(
        results, obb_iou_2d_matrix, iou_thresholds, key_prefix='2d_obb',
        collect_per_match=True, scale_cache=scale_cache)
    metrics.update(obb2d_out)

    # ---- Per-match statistics: AABB-3D matches (legacy fields kept for compatibility) ----
    if aabb3d_pm['iou']:
        metrics['mean_iou_3d'] = float(np.mean(aabb3d_pm['iou']))
        metrics['median_iou_3d'] = float(np.median(aabb3d_pm['iou']))
    if aabb3d_pm['center_3d']:
        metrics['mean_center_error_3d'] = float(np.mean(aabb3d_pm['center_3d']))
    if aabb3d_pm['center_2d']:
        metrics['mean_center_error_2d'] = float(np.mean(aabb3d_pm['center_2d']))
    if aabb3d_pm['center_3d_m']:
        metrics['mean_center_error_3d_m'] = float(np.mean(aabb3d_pm['center_3d_m']))
        metrics['median_center_error_3d_m'] = float(np.median(aabb3d_pm['center_3d_m']))
    if aabb3d_pm['center_2d_m']:
        metrics['mean_center_error_2d_m'] = float(np.mean(aabb3d_pm['center_2d_m']))
        metrics['median_center_error_2d_m'] = float(np.median(aabb3d_pm['center_2d_m']))
    if aabb3d_pm['size']:
        metrics['mean_size_error'] = float(np.mean(aabb3d_pm['size']))
    if aabb3d_pm['size_m']:
        metrics['mean_size_error_m'] = float(np.mean(aabb3d_pm['size_m']))
        metrics['median_size_error_m'] = float(np.median(aabb3d_pm['size_m']))
    if aabb3d_pm['rot']:
        metrics['mean_rot_error_deg'] = float(np.degrees(np.mean(aabb3d_pm['rot'])))

    # ---- Per-match statistics: OBB-3D matches ----
    if obb3d_pm['iou']:
        metrics['mean_iou_3d_obb'] = float(np.mean(obb3d_pm['iou']))
        metrics['median_iou_3d_obb'] = float(np.median(obb3d_pm['iou']))
    if obb3d_pm['rot']:
        metrics['mean_rot_error_deg_obb_match'] = float(np.degrees(np.mean(obb3d_pm['rot'])))

    # ---- Per-match statistics: OBB-2D matches ----
    if obb2d_pm['iou']:
        metrics['mean_iou_2d_obb'] = float(np.mean(obb2d_pm['iou']))
        metrics['median_iou_2d_obb'] = float(np.median(obb2d_pm['iou']))

    # Note: AABB 2D IoU per-match stats are derived alongside AABB 3D matches in the
    # original script. For consistency with the OBB metrics, we keep the 3D-matched
    # 2D AABB IoU only if needed; not recomputed here.

    total_gt = sum(len(r['gt_bboxes']) for r in results)
    total_pred = sum(len(r['pred_bboxes']) for r in results)
    metrics['total_gt_boxes'] = total_gt
    metrics['total_pred_boxes'] = total_pred
    metrics['num_scenes'] = len(results)
    metrics['avg_gt_per_scene'] = total_gt / max(len(results), 1)
    metrics['avg_pred_per_scene'] = total_pred / max(len(results), 1)
    metrics['used_pytorch3d_for_obb_3d'] = bool(_HAS_PYTORCH3D)

    return metrics


# ---------------------------------------------------------------------------
# Visualization (unchanged from original)
# ---------------------------------------------------------------------------

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
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
    R = result['heatmap_topdown'].shape[0]

    ax = axes[0]
    ax.imshow(result['voxel_topdown'].T, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax.set_title('Voxel Occupancy (top-down)', fontsize=9)
    ax.set_xlabel('X'); ax.set_ylabel('Y')

    ax = axes[1]
    ax.imshow(result['voxel_topdown'].T, origin='lower', cmap='gray', vmin=0, vmax=1, alpha=0.3)
    im = ax.imshow(result['heatmap_topdown'].T, origin='lower', cmap='hot', vmin=0, vmax=1, alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for gt in result['gt_bboxes']:
        vx = (gt[0] + 0.5) * R
        vy = (gt[1] + 0.5) * R
        ax.plot(vx, vy, 'g+', markersize=10, markeredgewidth=2)
    ax.set_title(f'Center Heatmap + GT ({len(result["gt_bboxes"])} objs)', fontsize=9)

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
    n = min(num_samples, len(results))
    cell_size = 4
    fig, axes = plt.subplots(n, 3, figsize=(cell_size * 3, cell_size * n), dpi=200)
    if n == 1:
        axes = axes[np.newaxis, :]

    for idx in range(n):
        r = results[idx]
        short_id = r['sample_id'].split('/')[-1][:30]
        R = r['heatmap_topdown'].shape[0]

        ax = axes[idx, 0]
        for i, gt in enumerate(r['gt_bboxes']):
            _draw_rotated_box(ax, gt[0], gt[1], gt[3], gt[4], gt[6],
                              color='green', alpha=0.5, label='GT' if i == 0 else None)
        ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55); ax.set_aspect('equal')
        ax.set_title(f'GT ({len(r["gt_bboxes"])} objs)\n{short_id}', fontsize=7)
        ax.grid(True, alpha=0.2); ax.tick_params(labelsize=5)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right')

        ax = axes[idx, 1]
        ax.imshow(r['voxel_topdown'].T, origin='lower', cmap='gray', vmin=0, vmax=1, alpha=0.3)
        ax.imshow(r['heatmap_topdown'].T, origin='lower', cmap='hot', vmin=0, vmax=1, alpha=0.7)
        for gt in r['gt_bboxes']:
            vx, vy = (gt[0] + 0.5) * R, (gt[1] + 0.5) * R
            ax.plot(vx, vy, 'g+', markersize=8, markeredgewidth=1.5)
        ax.set_title('Heatmap', fontsize=7)
        ax.tick_params(labelsize=5)

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='3D BBox Eval - CenterPoint (OBB-aware)')
    parser.add_argument('--config', type=str, default='configs/bbox/erp_bbox_centerpoint.json')
    parser.add_argument('--ckpt', type=str, default='results/bbox_centerpoint/ckpts/bbox_centerpoint_ema0.9999_step0001500.pt')
    parser.add_argument('--data_dir', type=str, default='datasets/ERP_3D_FRONT_test')
    parser.add_argument('--output_dir', type=str, default='evals/bbox_centerpoint_obb')
    parser.add_argument('--num_vis', type=int, default=20)
    parser.add_argument('--score_threshold', type=float, default=0.3)
    parser.add_argument('--nms_kernel', type=int, default=7)
    parser.add_argument('--iou_nms_threshold', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    args.num_vis = 100
    args.ckpt = 'results/bbox_centerpoint_v2/ckpts/bbox_centerpoint_ema0.9999_step0024500.pt'
    args.config = 'configs/bbox/erp_bbox_centerpoint_v2.json'
    args.data_dir = 'datasets/ERP_3D_FRONT_test'
    args.output_dir = 'evals/bbox_centerpoint_v2_obb'
    args.batch_size = 32
    args.device = 'cuda:5'
    args.score_threshold = 0.1
    args.nms_kernel = 7

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

    metrics = compute_metrics(results, data_dir=args.data_dir)

    print('\n' + '=' * 70)
    print('CenterPoint Metrics  (AABB + OBB)')
    print('=' * 70)
    print(f'  pytorch3d available for 3D OBB IoU: {metrics["used_pytorch3d_for_obb_3d"]}')

    print('\n  --- AABB 3D IoU (rotation-ignored, volume-based) ---')
    for thresh in [0.25, 0.5, 0.75]:
        r = metrics.get(f'3d_recall@{thresh}', 0)
        p = metrics.get(f'3d_precision@{thresh}', 0)
        f = metrics.get(f'3d_f1@{thresh}', 0)
        print(f'  @{thresh}  recall={r:.4f}  precision={p:.4f}  f1={f:.4f}')

    print('\n  --- AABB 2D IoU (rotation-ignored, top-down XY) ---')
    for thresh in [0.25, 0.5, 0.75]:
        r = metrics.get(f'2d_recall@{thresh}', 0)
        p = metrics.get(f'2d_precision@{thresh}', 0)
        f = metrics.get(f'2d_f1@{thresh}', 0)
        print(f'  @{thresh}  recall={r:.4f}  precision={p:.4f}  f1={f:.4f}')

    print('\n  --- OBB 3D IoU (rotation-aware) ---')
    for thresh in [0.25, 0.5, 0.75]:
        r = metrics.get(f'3d_obb_recall@{thresh}', 0)
        p = metrics.get(f'3d_obb_precision@{thresh}', 0)
        f = metrics.get(f'3d_obb_f1@{thresh}', 0)
        print(f'  @{thresh}  recall={r:.4f}  precision={p:.4f}  f1={f:.4f}')

    print('\n  --- OBB 2D IoU (rotation-aware, top-down XY) ---')
    for thresh in [0.25, 0.5, 0.75]:
        r = metrics.get(f'2d_obb_recall@{thresh}', 0)
        p = metrics.get(f'2d_obb_precision@{thresh}', 0)
        f = metrics.get(f'2d_obb_f1@{thresh}', 0)
        print(f'  @{thresh}  recall={r:.4f}  precision={p:.4f}  f1={f:.4f}')

    print('\n  --- Per-match Statistics ---')
    print(f'  mean_iou_3d (AABB):       {metrics.get("mean_iou_3d", 0):.4f}')
    print(f'  median_iou_3d (AABB):     {metrics.get("median_iou_3d", 0):.4f}')
    print(f'  mean_iou_3d_obb:          {metrics.get("mean_iou_3d_obb", 0):.4f}')
    print(f'  median_iou_3d_obb:        {metrics.get("median_iou_3d_obb", 0):.4f}')
    print(f'  mean_iou_2d_obb:          {metrics.get("mean_iou_2d_obb", 0):.4f}')
    print(f'  median_iou_2d_obb:        {metrics.get("median_iou_2d_obb", 0):.4f}')
    print(f'  mean_center_error_3d:     {metrics.get("mean_center_error_3d", 0):.4f}')
    print(f'  mean_center_error_2d:     {metrics.get("mean_center_error_2d", 0):.4f}')
    if 'mean_center_error_3d_m' in metrics:
        print(f'  mean_center_error_3d (m):   {metrics["mean_center_error_3d_m"]:.4f}  ({metrics["mean_center_error_3d_m"]*100:.1f} cm)')
        print(f'  median_center_error_3d (m): {metrics["median_center_error_3d_m"]:.4f}  ({metrics["median_center_error_3d_m"]*100:.1f} cm)')
        print(f'  mean_center_error_2d (m):   {metrics["mean_center_error_2d_m"]:.4f}  ({metrics["mean_center_error_2d_m"]*100:.1f} cm)')
        print(f'  median_center_error_2d (m): {metrics["median_center_error_2d_m"]:.4f}  ({metrics["median_center_error_2d_m"]*100:.1f} cm)')
    print(f'  mean_size_error:          {metrics.get("mean_size_error", 0):.4f}')
    if 'mean_size_error_m' in metrics:
        print(f'  mean_size_error (m):        {metrics["mean_size_error_m"]:.4f}  ({metrics["mean_size_error_m"]*100:.1f} cm)')
        print(f'  median_size_error (m):      {metrics["median_size_error_m"]:.4f}  ({metrics["median_size_error_m"]*100:.1f} cm)')
    print(f'  mean_rot_error_deg (AABB-match): {metrics.get("mean_rot_error_deg", 0):.4f}')
    if 'mean_rot_error_deg_obb_match' in metrics:
        print(f'  mean_rot_error_deg (OBB-match):  {metrics["mean_rot_error_deg_obb_match"]:.4f}')

    print(f'\n  --- Counts ---')
    print(f'  total_gt: {metrics["total_gt_boxes"]}  total_pred: {metrics["total_pred_boxes"]}')
    print(f'  scenes: {metrics["num_scenes"]}  avg_gt/scene: {metrics["avg_gt_per_scene"]:.1f}  avg_pred/scene: {metrics["avg_pred_per_scene"]:.1f}')
    print('=' * 70)

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\nVisualizing {args.num_vis} samples...')
    for i in range(min(args.num_vis, len(results))):
        r = results[i]
        safe_name = r['sample_id'].replace('/', '_')
        visualize_topdown_with_heatmap(
            r, os.path.join(args.output_dir, 'per_sample', f'{safe_name}.png'))

    visualize_grid(results, args.output_dir)
    visualize_heatmap_grid(results, args.output_dir)
    visualize_concat(results, args.output_dir, num_samples=args.num_vis)

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
