# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Sparse Structure VAE evaluation script for 3D-FRONT room-scale PLYs.

This script:
1. Reads 3D-FRONT room-scale ply files and converts them to 64x64x64 voxel grids
2. Runs them through the pretrained VAE encoder/decoder for reconstruction
3. Evaluates the difference between original and reconstructed voxels (IoU, Dice, Accuracy, etc.)
4. Saves the results (original ply, reconstructed ply, visualization images)
5. Visualizes GT vs Reconstruction comparisons from multiple viewpoints
"""

import os
import sys
import argparse
import json
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# TRELLIS imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import utils3d
from trellis import models
from trellis.representations.octree import DfsOctree as Octree
from trellis.renderers import OctreeRenderer


def load_vae_models(model_path: str, device: str = 'cuda') -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load the pretrained VAE encoder/decoder.

    Args:
        model_path: path to pretrained checkpoints (e.g., microsoft/TRELLIS-image-large/ckpts)
        device: cuda or cpu

    Returns:
        encoder, decoder models
    """
    encoder_path = os.path.join(model_path, 'ss_enc_conv3d_16l8_fp16')
    decoder_path = os.path.join(model_path, 'ss_dec_conv3d_16l8_fp16')

    print(f"Loading encoder from: {encoder_path}")
    encoder = models.from_pretrained(encoder_path)
    encoder = encoder.to(device).eval()

    print(f"Loading decoder from: {decoder_path}")
    decoder = models.from_pretrained(decoder_path)
    decoder = decoder.to(device).eval()

    return encoder, decoder


def ply_to_voxel(ply_path: str, resolution: int = 64) -> torch.Tensor:
    """
    Read a PLY file and convert it to a voxel grid.

    3D-FUTURE convention: position values lie in [-0.5, 0.5] and are
    mapped via (position + 0.5) * resolution.

    Args:
        ply_path: path to the ply file
        resolution: voxel grid resolution (default 64)

    Returns:
        binary voxel tensor of shape [1, resolution, resolution, resolution]
    """
    position = utils3d.io.read_ply(ply_path)[0]  # shape: (N, 3)

    # Convert coordinates to integer indices in [0, resolution)
    coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()

    # Clip out-of-range coordinates
    coords = torch.clamp(coords, 0, resolution - 1)

    # Build the voxel grid
    voxel = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    voxel[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    return voxel


def voxel_to_ply_positions(voxel: torch.Tensor, resolution: int = 64) -> np.ndarray:
    """
    Convert a voxel grid to a position array for PLY export.

    Args:
        voxel: binary voxel tensor of shape [1, resolution, resolution, resolution]
        resolution: voxel grid resolution

    Returns:
        position array of shape (N, 3), values in [-0.5, 0.5]
    """
    # Extract active voxel coordinates
    coords = torch.nonzero(voxel[0], as_tuple=False)  # (N, 3)

    # Map coordinates to the [-0.5, 0.5] range
    positions = (coords.float() / resolution) - 0.5 + (0.5 / resolution)

    return positions.numpy()


def save_ply(positions: np.ndarray, output_path: str):
    """
    Save a position array to a PLY file.

    Args:
        positions: position array of shape (N, 3)
        output_path: output ply file path
    """
    utils3d.io.write_ply(output_path, positions)


@torch.no_grad()
def visualize_voxel_multiview(
    voxel: torch.Tensor,
    resolution: int = 64,
    num_views: int = 8,
    image_size: int = 512,
    elevation: float = 30.0,
    camera_distance: float = 2.0,
    look_at: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Render a voxel grid from multiple viewpoints.

    Args:
        voxel: binary voxel tensor of shape [1, resolution, resolution, resolution]
        resolution: voxel resolution
        num_views: number of views to render (evenly spaced horizontally)
        image_size: size of each rendered image
        elevation: camera elevation (degrees)
        camera_distance: distance between the camera and the look_at point
        look_at: point the camera looks at [x, y, z]; [0, 0, 0] if None

    Returns:
        rendered images of shape [num_views, 3, image_size, image_size]
    """
    # Setup renderer
    renderer = OctreeRenderer()
    renderer.rendering_options.resolution = image_size
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (0, 0, 0)
    renderer.rendering_options.ssaa = 4
    renderer.pipe.primitive = 'voxel'

    # Build representation
    voxel = voxel.cuda()
    coords = torch.nonzero(voxel[0], as_tuple=False)

    if coords.shape[0] == 0:
        # Empty voxel grid
        return torch.zeros(num_views, 3, image_size, image_size)

    representation = Octree(
        depth=10,
        aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
        device='cuda',
        primitive='voxel',
        sh_degree=0,
        primitive_config={'solid': True},
    )
    representation.position = coords.float() / resolution
    representation.depth = torch.full(
        (representation.position.shape[0], 1),
        int(np.log2(resolution)),
        dtype=torch.uint8,
        device='cuda'
    )

    # Look at point
    if look_at is None:
        look_at_point = torch.tensor([0, 0, 0]).float().cuda()
    else:
        look_at_point = torch.tensor(look_at).float().cuda()

    # Generate camera poses
    pitch = np.deg2rad(elevation)
    yaws = [2 * np.pi * i / num_views for i in range(num_views)]

    images = []
    for yaw in yaws:
        orig = torch.tensor([
            np.sin(yaw) * np.cos(pitch),
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
        ]).float().cuda() * camera_distance + look_at_point

        fov = torch.deg2rad(torch.tensor(30)).cuda()
        extrinsics = utils3d.torch.extrinsics_look_at(
            orig,
            look_at_point,
            torch.tensor([0, 0, 1]).float().cuda()
        )
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)

        res = renderer.render(representation, extrinsics, intrinsics,
                              colors_overwrite=representation.position)
        images.append(res['color'])

    return torch.stack(images)  # [num_views, 3, H, W]


@torch.no_grad()
def visualize_comparison(
    gt_voxel: torch.Tensor,
    recon_voxel: torch.Tensor,
    resolution: int = 64,
    num_views: int = 8,
    image_size: int = 512,
    elevation: float = 30.0,
    camera_distance: float = 2.0,
    look_at: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a side-by-side visualization comparing GT and reconstruction.

    Args:
        gt_voxel: GT voxel [1, resolution, resolution, resolution]
        recon_voxel: reconstruction voxel [1, resolution, resolution, resolution]
        resolution: voxel resolution
        num_views: number of views to render
        image_size: size of each image
        elevation: camera elevation
        camera_distance: camera distance
        look_at: point the camera looks at

    Returns:
        gt_images: GT renderings [num_views, 3, H, W]
        recon_images: reconstruction renderings [num_views, 3, H, W]
        diff_images: difference visualization (GT only: green, recon only: red, both: white)
    """
    gt_images = visualize_voxel_multiview(
        gt_voxel, resolution, num_views, image_size, elevation, camera_distance, look_at
    )
    recon_images = visualize_voxel_multiview(
        recon_voxel, resolution, num_views, image_size, elevation, camera_distance, look_at
    )

    # Difference visualization
    # GT only: Green, Recon only: Red, Both: White
    diff_images = torch.zeros_like(gt_images)
    gt_mask = gt_images.sum(dim=1, keepdim=True) > 0.1
    recon_mask = recon_images.sum(dim=1, keepdim=True) > 0.1

    # Both (white)
    both_mask = gt_mask & recon_mask
    diff_images[:, 0:1] = both_mask.float()
    diff_images[:, 1:2] = both_mask.float()
    diff_images[:, 2:3] = both_mask.float()

    # GT only (green)
    gt_only_mask = gt_mask & ~recon_mask
    diff_images[:, 1:2] = torch.where(gt_only_mask, torch.ones_like(diff_images[:, 1:2]), diff_images[:, 1:2])

    # Recon only (red)
    recon_only_mask = ~gt_mask & recon_mask
    diff_images[:, 0:1] = torch.where(recon_only_mask, torch.ones_like(diff_images[:, 0:1]), diff_images[:, 0:1])

    return gt_images, recon_images, diff_images


def save_comparison_grid(
    gt_images: torch.Tensor,
    recon_images: torch.Tensor,
    diff_images: torch.Tensor,
    output_path: str,
    room_name: str,
    metrics: Dict[str, float],
):
    """
    Combine GT, reconstruction, and diff renderings into a single labeled image.
    Row labels are drawn on the left column, angle labels on the top row.

    Rows: GT, Reconstruction, Difference
    Columns: one per viewing angle
    """
    num_views = gt_images.shape[0]
    img_size = gt_images.shape[2]  # H or W (square assumed)

    # Stack vertically: [3*num_views, 3, H, W]
    all_images = torch.cat([gt_images, recon_images, diff_images], dim=0)

    # Create grid
    grid = vutils.make_grid(all_images, nrow=num_views, padding=2, normalize=False)

    # Convert to PIL
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    grid_img = Image.fromarray(grid_np)

    # Add labels
    label_width = 150
    label_height = 50

    # Create new image with space for labels
    new_width = grid_img.width + label_width
    new_height = grid_img.height + label_height
    labeled_img = Image.new('RGB', (new_width, new_height), color=(255, 255, 255))

    # Paste grid image (offset by label dimensions)
    labeled_img.paste(grid_img, (label_width, label_height))

    # Draw text
    draw = ImageDraw.Draw(labeled_img)

    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Row labels (left side)
    row_labels = ['GT', 'Pred', 'Difference']
    row_height = grid_img.height // 3

    for i, label in enumerate(row_labels):
        y_pos = label_height + row_height * i + row_height // 2
        # Draw text centered vertically in each row
        bbox = draw.textbbox((0, 0), label, font=font)
        text_height = bbox[3] - bbox[1]
        text_width = bbox[2] - bbox[0]
        draw.text(
            ((label_width - text_width) // 2, y_pos - text_height // 2),
            label,
            fill=(0, 0, 0),
            font=font
        )

    # Column labels (top)
    col_width = grid_img.width // num_views
    for i in range(num_views):
        angle = int(360 * i / num_views)
        angle_text = f"{angle}°"
        x_pos = label_width + col_width * i + col_width // 2

        bbox = draw.textbbox((0, 0), angle_text, font=small_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(
            (x_pos - text_width // 2, (label_height - text_height) // 2),
            angle_text,
            fill=(0, 0, 0),
            font=small_font
        )

    # Add metrics text in top-left corner
    metrics_text = f"IoU: {metrics['iou']:.4f}"
    draw.text((5, 5), metrics_text, fill=(0, 0, 0), font=small_font)

    # Save
    labeled_img.save(output_path)


def save_individual_views(
    gt_images: torch.Tensor,
    recon_images: torch.Tensor,
    output_dir: str,
):
    """
    Save GT and reconstruction renderings as individual files per angle.
    """
    num_views = gt_images.shape[0]

    for i in range(num_views):
        angle = int(360 * i / num_views)

        # GT
        gt_img = (gt_images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(gt_img).save(os.path.join(output_dir, f'gt_angle_{angle:03d}.png'))

        # Reconstruction
        recon_img = (recon_images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(recon_img).save(os.path.join(output_dir, f'recon_angle_{angle:03d}.png'))

        # Side by side comparison
        comparison = np.concatenate([gt_img, recon_img], axis=1)
        Image.fromarray(comparison).save(os.path.join(output_dir, f'comparison_angle_{angle:03d}.png'))


def compute_metrics(gt: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics between ground truth and prediction.

    Args:
        gt: ground truth voxel grid
        pred: predicted voxel grid

    Returns:
        metrics dictionary (IoU, Dice, Accuracy, Precision, Recall)
    """
    gt = gt.bool().flatten()
    pred = pred.bool().flatten()

    # True Positives, False Positives, False Negatives, True Negatives
    tp = (gt & pred).sum().float()
    fp = (~gt & pred).sum().float()
    fn = (gt & ~pred).sum().float()
    tn = (~gt & ~pred).sum().float()

    # Metrics
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    # Voxel counts
    gt_count = gt.sum().item()
    pred_count = pred.sum().item()

    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'gt_voxel_count': gt_count,
        'pred_voxel_count': pred_count,
        'voxel_count_diff': pred_count - gt_count,
    }


@torch.no_grad()
def evaluate_single(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    ply_path: str,
    resolution: int = 64,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Run VAE reconstruction and evaluation for a single PLY file.

    Args:
        encoder: VAE encoder
        decoder: VAE decoder
        ply_path: input ply file path
        resolution: voxel resolution
        device: cuda or cpu

    Returns:
        gt_voxel: original voxel
        recon_voxel: reconstructed voxel
        latent: latent representation
        metrics: evaluation metrics
    """
    # PLY -> Voxel
    gt_voxel = ply_to_voxel(ply_path, resolution)  # [1, 64, 64, 64]

    # Add batch dimension: [1, 1, 64, 64, 64]
    gt_input = gt_voxel.unsqueeze(0).float().to(device)

    # Encode
    latent = encoder(gt_input, sample_posterior=False)  # [1, 8, 16, 16, 16]

    # Decode
    logits = decoder(latent)  # [1, 1, 64, 64, 64]
    recon_voxel = (logits > 0).long().squeeze(0)  # [1, 64, 64, 64]

    # Compute metrics
    metrics = compute_metrics(gt_voxel, recon_voxel.cpu())

    return gt_voxel, recon_voxel.cpu(), latent.cpu(), metrics


def find_3dfront_rooms(base_path: str) -> List[str]:
    """
    Find room directories in the 3D-FRONT dataset.

    Args:
        base_path: base path of the 3D-FRONT data

    Returns:
        list of ply file paths
    """
    ply_paths = []
    for room_dir in sorted(os.listdir(base_path)):
        room_path = os.path.join(base_path, room_dir)
        if os.path.isdir(room_path):
            ply_path = os.path.join(room_path, 'voxels', 'full_room_wo_ceiling.ply')
            if os.path.exists(ply_path):
                ply_paths.append(ply_path)
    return ply_paths


def main():
    parser = argparse.ArgumentParser(description='Evaluate Sparse Structure VAE on 3D-FRONT dataset')
    parser.add_argument('--data_path', type=str,
                        default='/path/to/ERP_3D_FRONT',
                        help='Path to 3D-FRONT data')
    parser.add_argument('--model_path', type=str,
                        default='/path/to/TRELLIS-image-large/ckpts',
                        help='Path to pretrained VAE checkpoints')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/3dfront_vae_eval',
                        help='Output directory for results')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Voxel grid resolution')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_ply', action='store_true',
                        help='Save reconstructed PLY files')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate multi-view visualization comparing GT and reconstruction')
    parser.add_argument('--num_views', type=int, default=8,
                        help='Number of views for visualization (default: 8)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Size of each rendered image (default: 512)')
    parser.add_argument('--elevation', type=float, default=30.0,
                        help='Camera elevation angle in degrees (default: 30)')
    parser.add_argument('--save_individual', action='store_true',
                        help='Save individual view images (in addition to grid)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed metrics for each room')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("=" * 60)
    print("Loading VAE models...")
    print("=" * 60)
    encoder, decoder = load_vae_models(args.model_path, args.device)

    # Find all 3D-FRONT rooms
    print("\n" + "=" * 60)
    print("Finding 3D-FRONT rooms...")
    print("=" * 60)
    ply_paths = find_3dfront_rooms(args.data_path)
    print(f"Found {len(ply_paths)} rooms")

    # Evaluate each room
    print("\n" + "=" * 60)
    print("Evaluating VAE reconstruction...")
    print("=" * 60)

    all_metrics = []

    for ply_path in tqdm(ply_paths, desc="Processing rooms"):
        room_name = os.path.basename(os.path.dirname(os.path.dirname(ply_path)))

        # Evaluate
        gt_voxel, recon_voxel, latent, metrics = evaluate_single(
            encoder, decoder, ply_path, args.resolution, args.device
        )
        metrics['room_name'] = room_name
        all_metrics.append(metrics)

        if args.verbose:
            print(f"\n{room_name}:")
            print(f"  IoU: {metrics['iou']:.4f}")
            print(f"  Dice: {metrics['dice']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  GT voxels: {metrics['gt_voxel_count']}")
            print(f"  Pred voxels: {metrics['pred_voxel_count']}")

        # Save PLY files
        if args.save_ply or args.visualize:
            room_output_dir = os.path.join(args.output_dir, room_name)
            os.makedirs(room_output_dir, exist_ok=True)

        if args.save_ply:
            # Save original (copy)
            gt_positions = voxel_to_ply_positions(gt_voxel, args.resolution)
            save_ply(gt_positions, os.path.join(room_output_dir, 'gt.ply'))

            # Save reconstruction
            recon_positions = voxel_to_ply_positions(recon_voxel, args.resolution)
            save_ply(recon_positions, os.path.join(room_output_dir, 'recon.ply'))

            # Save latent
            torch.save(latent, os.path.join(room_output_dir, 'latent.pt'))

        # Generate visualization
        if args.visualize:
            # Exterior view (bird's eye view)
            gt_images_ext, recon_images_ext, diff_images_ext = visualize_comparison(
                gt_voxel, recon_voxel, args.resolution,
                num_views=args.num_views,
                image_size=args.image_size,
                elevation=args.elevation,
                camera_distance=2.0,
                look_at=None
            )

            # Save exterior comparison grid
            save_comparison_grid(
                gt_images_ext, recon_images_ext, diff_images_ext,
                os.path.join(room_output_dir, 'comparison_grid_exterior.png'),
                room_name, metrics
            )

            # Interior view (inside the room, lower elevation, closer)
            gt_images_int, recon_images_int, diff_images_int = visualize_comparison(
                gt_voxel, recon_voxel, args.resolution,
                num_views=args.num_views,
                image_size=args.image_size,
                elevation=10.0,  # Lower elevation to look more horizontally
                camera_distance=1.2,  # Closer to the scene
                look_at=[0.0, 0.0, 0.0]  # Look at center
            )

            # Save interior comparison grid
            save_comparison_grid(
                gt_images_int, recon_images_int, diff_images_int,
                os.path.join(room_output_dir, 'comparison_grid_interior.png'),
                room_name, metrics
            )

            # Save individual views if requested
            if args.save_individual:
                # Exterior views
                ext_dir = os.path.join(room_output_dir, 'exterior')
                os.makedirs(ext_dir, exist_ok=True)
                save_individual_views(gt_images_ext, recon_images_ext, ext_dir)

                # Interior views
                int_dir = os.path.join(room_output_dir, 'interior')
                os.makedirs(int_dir, exist_ok=True)
                save_individual_views(gt_images_int, recon_images_int, int_dir)

    # Aggregate metrics
    print("\n" + "=" * 60)
    print("Aggregate Results")
    print("=" * 60)

    avg_metrics = {
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
    }

    std_metrics = {
        'iou_std': np.std([m['iou'] for m in all_metrics]),
        'dice_std': np.std([m['dice'] for m in all_metrics]),
        'accuracy_std': np.std([m['accuracy'] for m in all_metrics]),
        'precision_std': np.std([m['precision'] for m in all_metrics]),
        'recall_std': np.std([m['recall'] for m in all_metrics]),
    }

    print(f"\nAverage IoU:       {avg_metrics['iou']:.4f} ± {std_metrics['iou_std']:.4f}")
    print(f"Average Dice:      {avg_metrics['dice']:.4f} ± {std_metrics['dice_std']:.4f}")
    print(f"Average Accuracy:  {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision_std']:.4f}")
    print(f"Average Recall:    {avg_metrics['recall']:.4f} ± {std_metrics['recall_std']:.4f}")

    # Per-room results
    print("\n" + "-" * 60)
    print("Per-room Results:")
    print("-" * 60)
    print(f"{'Room Name':<30} {'IoU':>8} {'Dice':>8} {'GT Vox':>10} {'Pred Vox':>10}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['room_name']:<30} {m['iou']:>8.4f} {m['dice']:>8.4f} {m['gt_voxel_count']:>10} {m['pred_voxel_count']:>10}")

    # Save results
    results = {
        'average_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'per_room_metrics': all_metrics,
        'config': {
            'data_path': args.data_path,
            'model_path': args.model_path,
            'resolution': args.resolution,
            'num_rooms': len(ply_paths),
        }
    }

    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Recommendation
    print("\n" + "=" * 60)
    print("Recommendation")
    print("=" * 60)

    if avg_metrics['iou'] < 0.8:
        print("⚠️  IoU is below 0.8.")
        print("   Retraining the VAE on 3D-FRONT room-scale data is recommended.")
        print("\n   How to retrain:")
        print("   1. Create a dataset class for 3D-FRONT")
        print("   2. Update the dataset path in the config file")
        print("   3. Run: python train.py --config configs/vae/ss_vae_conv3d_16l8_fp16.json")
    else:
        print("✅ IoU is 0.8 or higher. The pretrained VAE appears to work well.")


if __name__ == '__main__':
    main()
