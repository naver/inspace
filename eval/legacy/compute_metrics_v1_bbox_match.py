# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Compute 3D and 2D metrics for Stage 2 evaluation.

Scene-level 3D Metrics (from NPZ, NO GLB needed):
  - Voxel IoU: GT vs Pred occupancy on 32³ grid (fast, no decoder)
  - Chamfer Distance (CD): shape_dec() → mesh → sample points (accurate, ~1s/sample)
  - F1 Score @threshold: precision/recall on point-to-point distances

Asset-level 3D Metrics:
  - Per-asset Voxel IoU, CD, F1: match pred assets to GT via bbox center, compare individually
  - Aggregated as asset_voxel_iou, asset_chamfer_distance, asset_f1@t etc.

2D Metrics (from vis images):
  - PSNR, SSIM, LPIPS on rendered images (exterior, topdown, interior)

Usage:
    # Voxel IoU only (fastest, no GPU needed for decoder)
    python eval/pipeline/compute_metrics.py \
        --pred_dir evals/stage12_pipeline/random_gt \
        --data_dir datasets/ERP_3D_FRONT_test \
        --metrics voxel_iou

    # All 3D metrics (needs GPU for shape_dec)
    python eval/pipeline/compute_metrics.py \
        --pred_dir evals/stage12_pipeline/random_gt \
        --data_dir datasets/ERP_3D_FRONT_test \
        --metrics voxel_iou chamfer f1

    # Asset-level metrics only
    python eval/pipeline/compute_metrics.py \
        --pred_dir evals/stage12_pipeline/random_gt \
        --data_dir datasets/ERP_3D_FRONT_test \
        --metrics asset_voxel_iou asset_chamfer asset_f1

    # All metrics (scene + asset + 2D)
    python eval/pipeline/compute_metrics.py \
        --pred_dir evals/stage12_pipeline/random_gt \
        --data_dir datasets/ERP_3D_FRONT_test \
        --metrics all
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================
# 3D Metrics: Voxel-level
# ============================================================

def compute_voxel_iou(gt_coords, pred_coords, resolution=32):
    """
    Compute voxel-level IoU between GT and prediction.

    Args:
        gt_coords: [N, 3] int array, voxel coordinates in [0, resolution-1]
        pred_coords: [M, 3] int array, voxel coordinates in [0, resolution-1]
        resolution: voxel grid resolution

    Returns:
        dict with iou, precision, recall, f1
    """
    # Convert to flat indices for fast set operations
    def coords_to_set(coords):
        c = coords.astype(np.int64)
        return set(
            c[:, 0] * resolution * resolution +
            c[:, 1] * resolution +
            c[:, 2]
        )

    gt_set = coords_to_set(gt_coords)
    pred_set = coords_to_set(pred_coords)

    intersection = len(gt_set & pred_set)
    union = len(gt_set | pred_set)

    iou = intersection / max(union, 1)
    precision = intersection / max(len(pred_set), 1)
    recall = intersection / max(len(gt_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'voxel_iou': iou,
        'voxel_precision': precision,
        'voxel_recall': recall,
        'voxel_f1': f1,
        'gt_voxels': len(gt_set),
        'pred_voxels': len(pred_set),
    }


# ============================================================
# 3D Metrics: Mesh-level (Chamfer Distance, F1)
# ============================================================

def sample_points_from_mesh(vertices, faces, num_points=10000):
    """Sample points uniformly from mesh surface using trimesh."""
    import trimesh
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if len(mesh.faces) == 0:
        return None

    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def chamfer_distance_np(pts1, pts2):
    """
    Compute Chamfer Distance between two point clouds using scipy KDTree.
    Returns: cd (mean), cd_p2g (pred→gt), cd_g2p (gt→pred)
    """
    from scipy.spatial import cKDTree

    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    dist_1to2, _ = tree1.query(pts2)  # for each pt in pts2, distance to nearest in pts1
    dist_2to1, _ = tree2.query(pts1)  # for each pt in pts1, distance to nearest in pts2

    cd_p2g = np.mean(dist_2to1 ** 2)  # pred → gt
    cd_g2p = np.mean(dist_1to2 ** 2)  # gt → pred
    cd = (cd_p2g + cd_g2p) / 2

    return {
        'chamfer_distance': float(cd),
        'cd_pred_to_gt': float(cd_p2g),
        'cd_gt_to_pred': float(cd_g2p),
    }


def f1_score_3d(pts_gt, pts_pred, threshold=0.01):
    """
    Compute F1 score at given threshold.
    A predicted point is correct if nearest GT point < threshold, and vice versa.
    """
    from scipy.spatial import cKDTree

    tree_gt = cKDTree(pts_gt)
    tree_pred = cKDTree(pts_pred)

    # Precision: for each predicted point, is nearest GT < threshold?
    dist_pred_to_gt, _ = tree_gt.query(pts_pred)
    precision = np.mean(dist_pred_to_gt < threshold)

    # Recall: for each GT point, is nearest predicted < threshold?
    dist_gt_to_pred, _ = tree_pred.query(pts_gt)
    recall = np.mean(dist_gt_to_pred < threshold)

    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        f'f1@{threshold}': float(f1),
        f'precision@{threshold}': float(precision),
        f'recall@{threshold}': float(recall),
    }


@torch.no_grad()
def decode_latent_to_mesh(coords, feats, shape_dec, device, normalization=None):
    """Decode shape latent to mesh vertices/faces. Fast (~1s), no GLB pipeline."""
    from trellis2.modules.sparse import SparseTensor

    # Inverse normalize if needed
    if normalization is not None:
        mean = torch.tensor(normalization['mean'], dtype=torch.float32)
        std = torch.tensor(normalization['std'], dtype=torch.float32)
        feats = feats * std + mean

    batch_idx = torch.zeros(coords.shape[0], 1, dtype=torch.int32)
    if coords.shape[1] == 3:
        coords_4d = torch.cat([batch_idx, coords], dim=1)
    else:
        coords_4d = coords  # already has batch dim

    z = SparseTensor(coords=coords_4d, feats=feats).to(device)
    reps = shape_dec(z)

    if reps and len(reps) > 0:
        return reps[0].vertices, reps[0].faces
    return None, None


# ============================================================
# Asset-level: Matching pred assets to GT assets
# ============================================================

import re

def find_gt_asset_file_by_inst(gt_ia_dir, inst_num):
    """Find GT individual asset NPZ file by instance number."""
    pattern = f'_inst{inst_num:03d}.npz'
    for fname in os.listdir(gt_ia_dir):
        if fname.endswith(pattern):
            return os.path.join(gt_ia_dir, fname)
    return None


def match_pred_assets_to_gt(pred_bboxes_path, gt_bbox_path, gt_ia_dir):
    """
    Match predicted assets to GT assets by bbox center proximity.

    Returns list of dicts:
        [{'pred_idx': int, 'gt_inst': int, 'gt_file': str, 'asset_name': str, 'dist': float}, ...]
    Only matched pairs are returned.
    """
    if not os.path.exists(pred_bboxes_path) or not os.path.exists(gt_bbox_path):
        return []
    if not os.path.exists(gt_ia_dir):
        return []

    pred_bbox = np.load(pred_bboxes_path, allow_pickle=True)
    gt_bbox = np.load(gt_bbox_path, allow_pickle=True)

    pred_obbs = pred_bbox['obbs']  # [N_pred, 7]
    gt_obbs = gt_bbox['obbs']      # [N_gt, 7]
    pred_names = list(pred_bbox['asset_names'])

    if len(pred_obbs) == 0 or len(gt_obbs) == 0:
        return []

    pred_centers = pred_obbs[:, :3]  # [N_pred, 3]
    gt_centers = gt_obbs[:, :3]      # [N_gt, 3]

    # Greedy nearest-neighbor matching (pred → GT)
    matches = []
    used_gt = set()
    # Compute all pairwise distances
    dists = np.linalg.norm(pred_centers[:, None] - gt_centers[None, :], axis=2)  # [N_pred, N_gt]

    for _ in range(min(len(pred_obbs), len(gt_obbs))):
        # Mask used GT
        mask = np.ones_like(dists) * 1e10
        for gi in range(len(gt_obbs)):
            if gi not in used_gt:
                mask[:, gi] = 0
        masked_dists = dists + mask

        pi, gi = np.unravel_index(np.argmin(masked_dists), masked_dists.shape)
        d = dists[pi, gi]
        if d > 0.3:  # skip if too far (threshold in normalized [-0.5, 0.5] space)
            break

        gt_file = find_gt_asset_file_by_inst(gt_ia_dir, gi)
        if gt_file is not None:
            matches.append({
                'pred_idx': int(pi),
                'gt_inst': int(gi),
                'gt_file': gt_file,
                'asset_name': pred_names[pi] if pi < len(pred_names) else f'asset_{pi}',
                'dist': float(d),
            })

        used_gt.add(gi)
        dists[pi, :] = 1e10  # mark pred as used

    return matches


def compute_asset_metrics_for_sample(
    pred_data, part_layouts, matches,
    gt_shape_folder, data_dir, scene_id, room_id,
    metrics, shape_dec, device, shape_normalization,
    num_points, f1_thresholds, resolution=32,
):
    """
    Compute per-asset metrics for one sample.

    Returns:
        asset_results: list of dicts, one per matched asset
    """
    pred_coords = pred_data['coords']  # [N, 4] with batch_id
    pred_feats = torch.from_numpy(pred_data['feats']).float()

    asset_start = 2  # part_layouts: [overall(0), layout(1), asset0(2), ...]
    asset_results = []

    for match in matches:
        pi = match['pred_idx']
        part_idx = pi + asset_start

        if part_idx >= len(part_layouts):
            continue

        ps, pe = int(part_layouts[part_idx][0]), int(part_layouts[part_idx][1])
        n_pred = pe - ps
        if n_pred < 1:
            continue

        pred_asset_coords = pred_coords[ps:pe, 1:]  # drop batch_id, [n, 3]
        pred_asset_feats = pred_feats[ps:pe]

        # Load GT asset latent
        gt_asset_data = np.load(match['gt_file'])
        gt_asset_coords = gt_asset_data['coords']  # [m, 3]
        gt_asset_feats = torch.from_numpy(gt_asset_data['feats']).float()

        if len(gt_asset_coords) < 1:
            continue

        result = {
            'asset_name': match['asset_name'],
            'pred_voxels': n_pred,
            'gt_voxels': len(gt_asset_coords),
            'match_dist': match['dist'],
        }

        # Voxel IoU
        if any(m in metrics for m in ['asset_voxel_iou', 'all']):
            viou = compute_voxel_iou(gt_asset_coords, pred_asset_coords, resolution)
            result['asset_voxel_iou'] = viou['voxel_iou']
            result['asset_voxel_f1'] = viou['voxel_f1']
            result['asset_voxel_precision'] = viou['voxel_precision']
            result['asset_voxel_recall'] = viou['voxel_recall']

        # Chamfer Distance & F1
        need_asset_decoder = any(m in metrics for m in ['asset_chamfer', 'asset_f1', 'all'])
        if need_asset_decoder and shape_dec is not None and n_pred >= 10 and len(gt_asset_coords) >= 10:
            try:
                gt_v, gt_f = decode_latent_to_mesh(
                    torch.from_numpy(gt_asset_coords).int(), gt_asset_feats,
                    shape_dec, device, normalization=None)

                pred_v, pred_f = decode_latent_to_mesh(
                    torch.from_numpy(pred_asset_coords).int(), pred_asset_feats,
                    shape_dec, device, normalization=shape_normalization)

                if gt_v is not None and pred_v is not None:
                    gt_pts = sample_points_from_mesh(gt_v, gt_f, num_points)
                    pred_pts = sample_points_from_mesh(pred_v, pred_f, num_points)

                    if gt_pts is not None and pred_pts is not None:
                        if any(m in metrics for m in ['asset_chamfer', 'all']):
                            cd = chamfer_distance_np(gt_pts, pred_pts)
                            result['asset_chamfer_distance'] = cd['chamfer_distance']

                        if any(m in metrics for m in ['asset_f1', 'all']):
                            for thresh in f1_thresholds:
                                f1 = f1_score_3d(gt_pts, pred_pts, threshold=thresh)
                                result[f'asset_f1@{thresh}'] = f1[f'f1@{thresh}']
            except Exception:
                pass

        asset_results.append(result)

    return asset_results


# ============================================================
# 2D Metrics
# ============================================================

def compute_psnr(img1, img2):
    """PSNR between two images (numpy HWC uint8 or float32)."""
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
    if img2.dtype == np.uint8:
        img2 = img2.astype(np.float32) / 255.0

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(1.0 / mse))


def compute_ssim_value(img1, img2):
    """SSIM between two images."""
    try:
        from skimage.metrics import structural_similarity
        if img1.dtype == np.uint8:
            img1 = img1.astype(np.float32) / 255.0
        if img2.dtype == np.uint8:
            img2 = img2.astype(np.float32) / 255.0

        # Ensure same shape
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]

        if img1.ndim == 3:
            return float(structural_similarity(img1, img2, channel_axis=2, data_range=1.0))
        return float(structural_similarity(img1, img2, data_range=1.0))
    except ImportError:
        print("Warning: skimage not available for SSIM, skipping")
        return None


def compute_lpips_value(img1, img2, lpips_model=None):
    """LPIPS between two images. img1/img2: HWC uint8 numpy."""
    if lpips_model is None:
        return None

    def to_tensor(img):
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        t = t * 2 - 1  # [0,1] -> [-1,1]
        return t

    # Ensure same shape
    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:min_h, :min_w]
    img2 = img2[:min_h, :min_w]

    t1 = to_tensor(img1)
    t2 = to_tensor(img2)

    with torch.no_grad():
        val = lpips_model(t1, t2)
    return float(val.item())


def load_gt_and_pred_vis_images(pred_dir, scene_id, room_id, image_name):
    """
    Load GT and Pred images from vis_concat (split left/right halves).
    vis_concat images are [GT | Pred] concatenated horizontally.
    Returns: (gt_img, pred_img) as numpy HWC uint8, or (None, None).
    """
    from PIL import Image

    concat_path = os.path.join(pred_dir, scene_id, room_id, 'vis_concat', image_name)
    if not os.path.exists(concat_path):
        return None, None

    img = np.array(Image.open(concat_path).convert('RGB'))
    h, w, c = img.shape
    mid = w // 2
    gt_img = img[:, :mid]
    pred_img = img[:, mid:]

    return gt_img, pred_img


# ============================================================
# Main
# ============================================================

def discover_samples(pred_dir):
    """Discover all samples that have shape_latent.npz in pred_dir."""
    samples = []
    for scene_id in sorted(os.listdir(pred_dir)):
        scene_dir = os.path.join(pred_dir, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        for room_id in sorted(os.listdir(scene_dir)):
            room_dir = os.path.join(scene_dir, room_id)
            if not os.path.isdir(room_dir):
                continue
            if os.path.exists(os.path.join(room_dir, 'shape_latent.npz')):
                samples.append((scene_id, room_id))
    return samples


def get_camera_info(data_dir, scene_id, room_id):
    """
    Get normalized camera center and distance to room center.

    In O-Voxel normalized space, room center = origin (0,0,0).
    Camera center = (world_location - center) * scale.

    Returns: dict with camera_center, room_center, camera_distance, or None.
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
    loc = np.array(cam_data['views'][0]['location'])
    norm_cam = (loc - center) * scale

    return {
        'camera_center': [round(float(x), 6) for x in norm_cam],
        'room_center': [0.0, 0.0, 0.0],  # origin in normalized space
        'camera_distance': round(float(np.linalg.norm(norm_cam)), 6),
    }


def main():
    parser = argparse.ArgumentParser(description='Compute 3D/2D metrics for Stage 2 evaluation')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Prediction output dir (e.g., evals/stage12_pipeline/random_gt)')
    parser.add_argument('--data_dir', type=str, default='datasets/ERP_3D_FRONT_test',
                        help='GT dataset dir')
    parser.add_argument('--metrics', nargs='+', default=['all'],
                        choices=['all', 'voxel_iou', 'chamfer', 'f1',
                                 'asset_voxel_iou', 'asset_chamfer', 'asset_f1',
                                 'psnr', 'ssim', 'lpips'],
                        help='Which metrics to compute')
    parser.add_argument('--max_samples', type=int, default=-1, help='-1 for all')
    parser.add_argument('--num_points', type=int, default=10000,
                        help='Points to sample from mesh for CD/F1')
    parser.add_argument('--f1_thresholds', nargs='+', type=float, default=[0.005, 0.01, 0.02],
                        help='F1 score thresholds')
    parser.add_argument('--stage2_shape_config', type=str,
                        default='configs/gen/erp_slat_flow_img2shape_asset_aware_bf16.json')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--output_json', type=str, default='',
                        help='Output JSON path (default: {pred_dir}/metrics.json)')
    # 2D metric image types to evaluate
    parser.add_argument('--vis_images', nargs='+',
                        default=['geometry_exterior.png', 'geometry_topdown.png', 'geometry_interior.png'],
                        help='vis_concat image names for 2D metrics')
    args = parser.parse_args()

    # CUDA_VISIBLE_DEVICES=1 python eval/pipeline/compute_metrics.py --pred_dir evals/stage12_pipeline/random_gt --data_dir datasets/ERP_3D_FRONT_test --metrics all
    # CUDA_VISIBLE_DEVICES=1 python eval/pipeline/compute_metrics.py --pred_dir evals/stage12_pipeline/sdedit0.5_gt --data_dir datasets/ERP_3D_FRONT_test --metrics all
    # CUDA_VISIBLE_DEVICES=2 python eval/pipeline/compute_metrics.py --pred_dir evals/stage12_pipeline/random_predicted --data_dir datasets/ERP_3D_FRONT_test --metrics all
    # CUDA_VISIBLE_DEVICES=2 python eval/pipeline/compute_metrics.py --pred_dir evals/stage12_pipeline/sdedit0.5_predicted --data_dir datasets/ERP_3D_FRONT_test --metrics all

    # CUDA_VISIBLE_DEVICES=1 python eval/pipeline/compute_metrics.py --pred_dir evals/stage12_pipeline/random_predicted --data_dir datasets/ERP_3D_FRONT_test --metrics all --max_samples 200
    # CUDA_VISIBLE_DEVICES=1 python eval/pipeline/compute_metrics.py --pred_dir evals/stage12_pipeline/sdedit0.3_predicted --data_dir datasets/ERP_3D_FRONT_test --metrics all 
    # CUDA_VISIBLE_DEVICES=2 python eval/pipeline/compute_metrics.py --pred_dir evals/stage12_pipeline/sdedit0.5_predicted --data_dir datasets/ERP_3D_FRONT_test --metrics all --max_samples 200
    # CUDA_VISIBLE_DEVICES=2 python eval/pipeline/compute_metrics.py --pred_dir evals/stage12_pipeline/sdedit0.7_predicted --data_dir datasets/ERP_3D_FRONT_test --metrics all

    if 'all' in args.metrics:
        args.metrics = ['voxel_iou', 'chamfer', 'f1',
                        'asset_voxel_iou', 'asset_chamfer', 'asset_f1',
                        'psnr', 'ssim', 'lpips']

    if not args.output_json:
        args.output_json = os.path.join(args.pred_dir, 'metrics.json')

    device = f'cuda:{args.gpu_id}'

    # Discover samples
    samples = discover_samples(args.pred_dir)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Found {len(samples)} samples in {args.pred_dir}")
    print(f"Metrics: {args.metrics}")

    # GT latent info
    gt_shape_folder = 'shape_latents/shape_enc_next_dc_f16c32_fp16_512'

    # ── Load models if needed ──
    shape_dec = None
    shape_normalization = None
    need_decoder = any(m in args.metrics for m in ['chamfer', 'f1', 'asset_chamfer', 'asset_f1'])

    if need_decoder:
        from trellis2 import models
        with open(args.stage2_shape_config, 'r') as f:
            shape_config = json.load(f)
        shape_normalization = shape_config['dataset']['args'].get('normalization', None)
        data_resolution = shape_config['dataset']['args'].get('resolution', 512)

        pretrained_slat_dec = shape_config['dataset']['args'].get(
            'pretrained_slat_dec', 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16')
        print(f"Loading shape decoder: {pretrained_slat_dec}")
        shape_dec = models.from_pretrained(pretrained_slat_dec)
        shape_dec.set_resolution(data_resolution)
        shape_dec = shape_dec.to(device).eval()

    # ── Load LPIPS model if needed ──
    lpips_model = None
    if 'lpips' in args.metrics:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex')
            print("Loaded LPIPS (alex) model")
        except ImportError:
            print("Warning: lpips not installed, skipping LPIPS metric. Install: pip install lpips")
            args.metrics.remove('lpips')

    # ── Compute metrics ──
    all_metrics = defaultdict(list)
    per_sample_metrics = {}

    for scene_id, room_id in tqdm(samples, desc="Computing metrics"):
        sample_key = f"{scene_id}/{room_id}"
        sample_metrics = {}

        # ── Camera info ──
        cam_info = get_camera_info(args.data_dir, scene_id, room_id)
        if cam_info is not None:
            sample_metrics['camera_center'] = cam_info['camera_center']
            sample_metrics['room_center'] = cam_info['room_center']
            sample_metrics['camera_distance'] = cam_info['camera_distance']
            all_metrics['camera_distance'].append(cam_info['camera_distance'])

        # Load prediction shape latent (overall part only)
        pred_path = os.path.join(args.pred_dir, scene_id, room_id, 'shape_latent.npz')
        pred_data = np.load(pred_path, allow_pickle=True)
        pred_coords = pred_data['coords']  # [N, 4] with batch_id
        pred_feats = torch.from_numpy(pred_data['feats']).float()

        # Extract overall part (part_layouts[0])
        part_layouts = pred_data['part_layouts']
        overall_start, overall_end = int(part_layouts[0][0]), int(part_layouts[0][1])
        pred_overall_coords = pred_coords[overall_start:overall_end, 1:]  # drop batch_id, [N, 3]
        pred_overall_feats = pred_feats[overall_start:overall_end]

        # Load GT shape latent
        gt_path = os.path.join(args.data_dir, scene_id, room_id,
                               gt_shape_folder, 'full_room_wo_ceiling.npz')
        if not os.path.exists(gt_path):
            continue
        gt_data = np.load(gt_path)
        gt_coords = gt_data['coords']  # [M, 3]
        gt_feats = torch.from_numpy(gt_data['feats']).float()

        # ── Voxel IoU ──
        if 'voxel_iou' in args.metrics:
            viou = compute_voxel_iou(gt_coords, pred_overall_coords)
            sample_metrics.update(viou)
            for k, v in viou.items():
                all_metrics[k].append(v)

        # ── Chamfer Distance & F1 ──
        if need_decoder:
            try:
                gt_verts, gt_faces = decode_latent_to_mesh(
                    torch.from_numpy(gt_coords).int(), gt_feats,
                    shape_dec, device, normalization=None)  # GT is in original space

                pred_verts, pred_faces = decode_latent_to_mesh(
                    torch.from_numpy(pred_overall_coords).int(), pred_overall_feats,
                    shape_dec, device, normalization=shape_normalization)

                if gt_verts is not None and pred_verts is not None:
                    gt_pts = sample_points_from_mesh(gt_verts, gt_faces, args.num_points)
                    pred_pts = sample_points_from_mesh(pred_verts, pred_faces, args.num_points)

                    if gt_pts is not None and pred_pts is not None:
                        if 'chamfer' in args.metrics:
                            cd = chamfer_distance_np(gt_pts, pred_pts)
                            sample_metrics.update(cd)
                            for k, v in cd.items():
                                all_metrics[k].append(v)

                        if 'f1' in args.metrics:
                            for thresh in args.f1_thresholds:
                                f1 = f1_score_3d(gt_pts, pred_pts, threshold=thresh)
                                sample_metrics.update(f1)
                                for k, v in f1.items():
                                    all_metrics[k].append(v)
                    else:
                        print(f"  Warning: mesh sampling failed for {sample_key}")
                else:
                    print(f"  Warning: decode failed for {sample_key}")
            except Exception as e:
                print(f"  Warning: mesh decode failed for {sample_key}: {e}")

        # ── Asset-level Metrics ──
        need_asset = any(m in args.metrics for m in ['asset_voxel_iou', 'asset_chamfer', 'asset_f1'])
        if need_asset:
            pred_bboxes_path = os.path.join(args.pred_dir, scene_id, room_id, 'bboxes.npz')
            gt_bbox_path = os.path.join(args.data_dir, scene_id, room_id,
                                        '3d_bounding_box', f'{room_id}_scene_data.npz')
            gt_ia_dir = os.path.join(args.data_dir, scene_id, room_id,
                                     gt_shape_folder, 'individual_assets_room_coord')

            matches = match_pred_assets_to_gt(pred_bboxes_path, gt_bbox_path, gt_ia_dir)

            if matches:
                asset_results = compute_asset_metrics_for_sample(
                    pred_data, part_layouts, matches,
                    gt_shape_folder, args.data_dir, scene_id, room_id,
                    args.metrics, shape_dec, device, shape_normalization,
                    args.num_points, args.f1_thresholds,
                )

                # Aggregate per-asset metrics into sample-level averages
                if asset_results:
                    sample_metrics['num_matched_assets'] = len(asset_results)
                    all_metrics['num_matched_assets'].append(len(asset_results))

                    # Collect all asset metric keys
                    asset_metric_keys = set()
                    for ar in asset_results:
                        for k in ar:
                            if k.startswith('asset_'):
                                asset_metric_keys.add(k)

                    for k in sorted(asset_metric_keys):
                        vals = [ar[k] for ar in asset_results if k in ar and isinstance(ar[k], (int, float))]
                        if vals:
                            mean_val = float(np.mean(vals))
                            sample_metrics[k] = mean_val
                            all_metrics[k].append(mean_val)

                    # Store per-asset detail
                    sample_metrics['asset_details'] = asset_results

        # ── 2D Metrics ──
        need_2d = any(m in args.metrics for m in ['psnr', 'ssim', 'lpips'])
        if need_2d:
            for img_name in args.vis_images:
                gt_img, pred_img = load_gt_and_pred_vis_images(
                    args.pred_dir, scene_id, room_id, img_name)
                if gt_img is None:
                    continue

                base = img_name.replace('.png', '')

                if 'psnr' in args.metrics:
                    val = compute_psnr(gt_img, pred_img)
                    k = f'psnr_{base}'
                    sample_metrics[k] = val
                    all_metrics[k].append(val)

                if 'ssim' in args.metrics:
                    val = compute_ssim_value(gt_img, pred_img)
                    if val is not None:
                        k = f'ssim_{base}'
                        sample_metrics[k] = val
                        all_metrics[k].append(val)

                if 'lpips' in args.metrics:
                    val = compute_lpips_value(gt_img, pred_img, lpips_model)
                    if val is not None:
                        k = f'lpips_{base}'
                        sample_metrics[k] = val
                        all_metrics[k].append(val)

        per_sample_metrics[sample_key] = sample_metrics

    # ── Aggregate & Print ──
    print("\n" + "=" * 70)
    print(f"Metrics Summary ({len(per_sample_metrics)} samples)")
    print("=" * 70)

    summary = {}
    for metric_name, values in sorted(all_metrics.items()):
        values = np.array(values)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        median_val = float(np.median(values))
        summary[metric_name] = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'count': len(values),
        }
        print(f"  {metric_name:30s}: {mean_val:.6f} ± {std_val:.6f}  (median: {median_val:.6f}, n={len(values)})")

    # ── Save ──
    output = {
        'config': {
            'pred_dir': args.pred_dir,
            'data_dir': args.data_dir,
            'metrics': args.metrics,
            'num_points': args.num_points,
            'f1_thresholds': args.f1_thresholds,
            'num_samples': len(per_sample_metrics),
        },
        'summary': summary,
        'per_sample': per_sample_metrics,
    }

    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {args.output_json}")


if __name__ == '__main__':
    main()
