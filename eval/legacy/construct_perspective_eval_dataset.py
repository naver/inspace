#!/usr/bin/env python3
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Construct perspective image evaluation dataset from ERP_3D_FRONT_test.

Selects high-quality perspective images for evaluation by:
1. Projecting 3D furniture bounding boxes onto each perspective view
2. Verifying asset visibility using depth maps (handles occlusion)
3. Scoring views based on number of visible assets, depth variety, etc.

Output: JSON file with selected (ERP image, perspective image, mesh) triplets.

Usage:
    python eval/construct_perspective_eval_dataset.py \
        --root datasets/ERP_3D_FRONT_test \
        --output evals/perspective_eval_dataset.json \
        --min_visible_assets 3

    # With visualization of selected vs rejected views:
    python eval/construct_perspective_eval_dataset.py \
        --root datasets/ERP_3D_FRONT_test \
        --output evals/perspective_eval_dataset.json \
        --visualize --vis_count 20
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Camera constants (fixed across all perspective images)
# ──────────────────────────────────────────────────────────────
IMG_W, IMG_H = 640, 480
FOV_DEG = 90.0
FX = FY = (IMG_W / 2) / np.tan(np.radians(FOV_DEG / 2))  # = 320.0
CX, CY = IMG_W / 2, IMG_H / 2  # = 320.0, 240.0

# Blender uses 1e10 for "no intersection" (background) in depth maps
BG_DEPTH_THRESHOLD = 100.0

# Depth tolerance for visibility verification
# actual_depth should be within [expected * (1 - tol), expected * (1 + tol)]
DEPTH_TOLERANCE = 0.5


# ──────────────────────────────────────────────────────────────
# Coordinate transforms
# ──────────────────────────────────────────────────────────────
# Camera convention (Blender with Euler XYZ pitch=π/2, yaw=0):
#   Camera forward = World +Y
#   Camera right   = World +X
#   Camera up      = World +Z

def world_to_camera(points_world, cam_loc):
    """
    Transform world-space points to camera-space coordinates.

    Args:
        points_world: [N, 3] array in world coordinates (X=floor, Y=floor, Z=height)
        cam_loc: [3] camera location in world coordinates

    Returns:
        cam_points: [N, 3] in camera space (x=right, y=up, z=backward)
    """
    rel = points_world - cam_loc
    cam_x = rel[:, 0]         # World X → Camera right
    cam_y = rel[:, 2]         # World Z → Camera up
    cam_z = -rel[:, 1]        # World -Y → Camera backward (+Z = behind camera)
    return np.stack([cam_x, cam_y, cam_z], axis=-1)


def project_to_image(cam_points):
    """
    Project camera-space points to image pixel coordinates.

    Args:
        cam_points: [N, 3] in camera space

    Returns:
        u: [N] horizontal pixel coordinates
        v: [N] vertical pixel coordinates
        depth: [N] forward depth (positive = in front of camera)
    """
    depth = -cam_points[:, 2]  # Depth = -cam_z (positive when in front)

    valid = depth > 0.01
    u = np.full(len(cam_points), -1.0)
    v = np.full(len(cam_points), -1.0)

    u[valid] = CX + FX * cam_points[valid, 0] / depth[valid]
    v[valid] = CY - FY * cam_points[valid, 1] / depth[valid]

    return u, v, depth


def normalized_to_world(points_norm, center, scale):
    """Convert O-Voxel normalized coordinates [-0.5, 0.5] to world coordinates."""
    return points_norm / scale + center


# ──────────────────────────────────────────────────────────────
# 3D bounding box utilities
# ──────────────────────────────────────────────────────────────

def get_obb_corners(cx, cy, cz, sx, sy, sz, yaw):
    """
    Get 8 corners of an oriented bounding box.

    OBB format: center (cx,cy,cz), full extents (sx,sy,sz), rotation yaw (around Z axis).
    Returns: [8, 3] array of corner positions.
    """
    hx, hy, hz = sx / 2, sy / 2, sz / 2

    offsets = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ])

    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos_y, -sin_y, 0],
                  [sin_y,  cos_y, 0],
                  [0,      0,     1]])

    corners = offsets @ R.T + np.array([cx, cy, cz])
    return corners


# ──────────────────────────────────────────────────────────────
# Per-view scoring
# ──────────────────────────────────────────────────────────────

def score_perspective_view(cam_loc, obbs_world, depth_map, depth_tolerance=DEPTH_TOLERANCE):
    """
    Score a perspective view based on visible assets and scene overview quality.

    Args:
        cam_loc: [3] camera world position
        obbs_world: list of [7] OBBs in world coordinates [cx,cy,cz,sx,sy,sz,yaw]
        depth_map: [H, W] depth map from perspective rendering
        depth_tolerance: relative tolerance for depth-based visibility check

    Returns:
        score: composite quality score (higher = better)
        metrics: dict with detailed metrics
    """
    n_assets = len(obbs_world)

    # ── Depth-based scene metrics (on valid pixels only) ──
    valid_mask = depth_map < BG_DEPTH_THRESHOLD
    valid_depths = depth_map[valid_mask]

    if len(valid_depths) == 0:
        return 0.0, {
            'n_visible_assets': 0, 'n_in_fov': 0, 'n_total_assets': n_assets,
            'depth_mean': 0, 'depth_std': 0, 'depth_min': 0,
            'valid_pixel_ratio': 0, 'max_asset_coverage': 0, 'score': 0,
        }

    depth_mean = float(np.mean(valid_depths))
    depth_std = float(np.std(valid_depths))
    depth_min = float(np.min(valid_depths))
    valid_pixel_ratio = float(valid_mask.sum()) / (IMG_W * IMG_H)

    # ── Per-asset projection and visibility ──
    visible_assets = []
    in_fov_assets = []
    projected_areas = []

    for i, obb_w in enumerate(obbs_world):
        cx, cy, cz, sx, sy, sz, yaw = obb_w
        center_world = np.array([[cx, cy, cz]])
        corners_world = get_obb_corners(*obb_w)  # [8, 3]

        # Project center
        cam_center = world_to_camera(center_world, cam_loc)
        u_c, v_c, depth_c = project_to_image(cam_center)

        if depth_c[0] <= 0.01:
            continue  # Behind camera

        # Project all 8 corners
        cam_corners = world_to_camera(corners_world, cam_loc)
        u_corners, v_corners, depth_corners = project_to_image(cam_corners)

        # Check if at least some corners are within image bounds
        in_image = (
            (u_corners >= 0) & (u_corners < IMG_W) &
            (v_corners >= 0) & (v_corners < IMG_H) &
            (depth_corners > 0.01)
        )

        if not np.any(in_image):
            continue  # Completely outside FOV

        in_fov_assets.append(i)

        # Compute projected bounding rectangle area (clipped to image)
        u_vis = u_corners[in_image]
        v_vis = v_corners[in_image]
        u_min = np.clip(u_vis.min(), 0, IMG_W)
        u_max = np.clip(u_vis.max(), 0, IMG_W)
        v_min = np.clip(v_vis.min(), 0, IMG_H)
        v_max = np.clip(v_vis.max(), 0, IMG_H)
        area_ratio = (u_max - u_min) * (v_max - v_min) / (IMG_W * IMG_H)

        # ── Depth-based visibility verification ──
        # Sample multiple points in the projected bbox for robustness
        sample_us = [u_c[0]]  # Center
        sample_vs = [v_c[0]]
        sample_depths = [depth_c[0]]

        # Add corner samples that are in image
        for j in range(8):
            if in_image[j]:
                sample_us.append(u_corners[j])
                sample_vs.append(v_corners[j])
                sample_depths.append(depth_corners[j])

        n_visible_samples = 0
        for su, sv, sd in zip(sample_us, sample_vs, sample_depths):
            if su < 0 or su >= IMG_W or sv < 0 or sv >= IMG_H or sd <= 0.01:
                continue
            ui, vi = int(np.clip(su, 0, IMG_W - 1)), int(np.clip(sv, 0, IMG_H - 1))
            actual_d = depth_map[vi, ui]

            if actual_d >= BG_DEPTH_THRESHOLD:
                continue  # Background pixel

            # Visibility check: actual depth should be close to expected
            # If actual << expected → occluded by something closer
            # If actual ≈ expected → asset surface is visible
            # Allow generous tolerance for bbox center vs surface offset
            if actual_d >= sd * (1.0 - depth_tolerance) and actual_d <= sd * (1.0 + depth_tolerance):
                n_visible_samples += 1

        # Asset is visible if at least 1 sample point has matching depth
        if n_visible_samples > 0:
            visible_assets.append(i)
            projected_areas.append(area_ratio)

    n_visible = len(visible_assets)
    n_in_fov = len(in_fov_assets)
    max_coverage = max(projected_areas) if projected_areas else 0.0

    # ── Composite score ──
    # Rewards: more visible assets, higher depth variety, moderate distance
    # Penalties: single asset dominating the view, very close views
    asset_score = n_visible
    depth_variety_bonus = 1.0 + min(depth_std, 2.0)  # Cap at 2.0
    distance_factor = min(1.0, depth_mean / 1.5)      # Penalize very close views
    dominance_penalty = 1.0 - max_coverage * 0.5       # Penalize if one asset > 50% area
    close_penalty = min(1.0, depth_min / 0.3)          # Penalize if anything < 30cm

    score = asset_score * depth_variety_bonus * distance_factor * dominance_penalty * close_penalty

    metrics = {
        'n_visible_assets': n_visible,
        'n_in_fov': n_in_fov,
        'n_total_assets': n_assets,
        'depth_mean': round(depth_mean, 3),
        'depth_std': round(depth_std, 3),
        'depth_min': round(depth_min, 3),
        'valid_pixel_ratio': round(valid_pixel_ratio, 3),
        'max_asset_coverage': round(max_coverage, 4),
        'score': round(score, 3),
    }

    return score, metrics


# ──────────────────────────────────────────────────────────────
# Per-room processing
# ──────────────────────────────────────────────────────────────

def load_normalization_info(room_dir):
    """Load O-Voxel normalization parameters (center, scale)."""
    for subdir in ['mesh_dumps', 'dual_grid_512', 'dual_grid_256']:
        path = room_dir / subdir / 'normalization_info.json'
        if path.exists():
            with open(path) as f:
                info = json.load(f)
            return np.array(info['center']), info['scale']
    return None, None


def load_obbs(room_dir, room_name):
    """Load oriented bounding boxes from scene_data.npz."""
    bbox_path = room_dir / '3d_bounding_box' / f'{room_name}_scene_data.npz'
    if not bbox_path.exists():
        return None
    data = np.load(bbox_path, allow_pickle=True)
    return data['obbs']  # [N, 7] in O-Voxel normalized space


def process_room(room_dir, depth_tolerance=DEPTH_TOLERANCE):
    """
    Process a single room: score all perspective views.

    Returns:
        list of view results sorted by score (descending), or None if room is invalid.
    """
    room_dir = Path(room_dir)
    room_name = room_dir.name

    # Load camera poses
    camera_poses_path = room_dir / 'camera_poses.json'
    if not camera_poses_path.exists():
        return None
    with open(camera_poses_path) as f:
        camera_data = json.load(f)

    # Load normalization info
    center, scale = load_normalization_info(room_dir)
    if center is None:
        return None

    # Load 3D bounding boxes
    obbs_norm = load_obbs(room_dir, room_name)
    if obbs_norm is None or len(obbs_norm) == 0:
        return None

    # Convert OBBs from normalized space to world coordinates
    obbs_world = []
    for obb in obbs_norm:
        cx, cy, cz, sx, sy, sz, yaw = obb
        c_world = normalized_to_world(np.array([cx, cy, cz]), center, scale)
        s_world = np.array([sx, sy, sz]) / scale
        obbs_world.append([*c_world, *s_world, yaw])

    # Score each perspective view
    results = []
    for view in camera_data['views']:
        view_idx = view['view_idx']
        cam_loc = np.array(view['location'])

        persp_path = room_dir / 'perspective' / f'{view_idx:04d}_colors.png'
        depth_path = room_dir / 'perspective' / f'{view_idx:04d}_depth.npy'
        erp_path = room_dir / 'erp' / f'{view_idx:04d}_colors.png'

        if not persp_path.exists() or not depth_path.exists():
            continue

        depth_map = np.load(depth_path)
        score, metrics = score_perspective_view(cam_loc, obbs_world, depth_map, depth_tolerance)

        results.append({
            'view_idx': view_idx,
            'camera_location': cam_loc.tolist(),
            'has_erp': erp_path.exists(),
            'score': score,
            'metrics': metrics,
        })

    if not results:
        return None

    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ──────────────────────────────────────────────────────────────
# Visualization (optional)
# ──────────────────────────────────────────────────────────────

def visualize_selected_views(selected_samples, root, output_dir, count=20):
    """Generate comparison grids showing selected perspective views with projected bboxes."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from PIL import Image
    except ImportError:
        print("Warning: matplotlib/PIL not available, skipping visualization")
        return

    os.makedirs(output_dir, exist_ok=True)
    root = Path(root)

    for idx, sample in enumerate(selected_samples[:count]):
        room_path = root / sample['room_path']
        view_idx = sample['view_idx']

        # Load perspective image
        persp_img_path = room_path / 'perspective' / f'{view_idx:04d}_colors.png'
        if not persp_img_path.exists():
            continue
        img = np.array(Image.open(persp_img_path))

        # Load ERP image if available
        erp_img_path = room_path / 'erp' / f'{view_idx:04d}_colors.png'
        has_erp = erp_img_path.exists()

        # Load data for projection overlay
        center, scale = load_normalization_info(room_path)
        obbs_norm = load_obbs(room_path, room_path.name)
        cam_loc = np.array(sample['camera_location'])

        if center is None or obbs_norm is None:
            continue

        # Convert OBBs to world
        obbs_world = []
        for obb in obbs_norm:
            cx, cy, cz, sx, sy, sz, yaw = obb
            c_world = normalized_to_world(np.array([cx, cy, cz]), center, scale)
            s_world = np.array([sx, sy, sz]) / scale
            obbs_world.append([*c_world, *s_world, yaw])

        # Create figure
        n_cols = 2 if has_erp else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4.5))
        if n_cols == 1:
            axes = [axes]

        # Plot perspective image with bbox projections
        ax = axes[0]
        ax.imshow(img)

        # Project and draw bboxes
        depth_map = np.load(room_path / 'perspective' / f'{view_idx:04d}_depth.npy')

        for obb_w in obbs_world:
            corners = get_obb_corners(*obb_w)
            cam_corners = world_to_camera(corners, cam_loc)
            u_corners, v_corners, depth_corners = project_to_image(cam_corners)

            in_image = (
                (u_corners >= 0) & (u_corners < IMG_W) &
                (v_corners >= 0) & (v_corners < IMG_H) &
                (depth_corners > 0.01)
            )

            if not np.any(in_image):
                continue

            # Check visibility (center point)
            center_world = np.array([[obb_w[0], obb_w[1], obb_w[2]]])
            cam_c = world_to_camera(center_world, cam_loc)
            u_c, v_c, d_c = project_to_image(cam_c)

            visible = False
            if 0 <= u_c[0] < IMG_W and 0 <= v_c[0] < IMG_H and d_c[0] > 0.01:
                ui, vi = int(u_c[0]), int(v_c[0])
                actual_d = depth_map[vi, ui]
                if actual_d < BG_DEPTH_THRESHOLD:
                    if abs(actual_d - d_c[0]) <= d_c[0] * DEPTH_TOLERANCE:
                        visible = True

            # Draw projected bbox (2D bounding rect of projected corners)
            u_vis = u_corners[in_image]
            v_vis = v_corners[in_image]
            u_min, u_max = u_vis.min(), u_vis.max()
            v_min, v_max = v_vis.min(), v_vis.max()

            color = 'lime' if visible else 'red'
            rect = patches.Rectangle(
                (u_min, v_min), u_max - u_min, v_max - v_min,
                linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.7
            )
            ax.add_patch(rect)

        ax.set_title(
            f"Perspective (score={sample['score']:.1f}, "
            f"vis={sample['n_visible_assets']}, fov={sample['n_in_fov']})",
            fontsize=9
        )
        ax.axis('off')

        # Plot ERP image
        if has_erp:
            erp_img = np.array(Image.open(erp_img_path))
            axes[1].imshow(erp_img)
            axes[1].set_title("ERP Panorama", fontsize=9)
            axes[1].axis('off')

        fig.suptitle(
            f"{sample['room_path']} (view {view_idx})\n"
            f"depth: mean={sample['depth_mean']:.1f}m, std={sample['depth_std']:.2f}m, "
            f"min={sample['depth_min']:.2f}m",
            fontsize=8
        )
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'{idx:04d}_{Path(sample["room_path"]).name}_v{view_idx}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"  Saved {min(count, len(selected_samples))} visualizations to {output_dir}/")


# ──────────────────────────────────────────────────────────────
# Grid examination (5x5 grids for manual review)
# ──────────────────────────────────────────────────────────────

def generate_examination_grids(json_path, output_dir=None, grid_size=5):
    """
    Generate 5x5 grid images for manual examination of selected perspective views.

    Each cell has a text header (white bg, black text) showing the sample index,
    followed by the perspective image below it.

    Also saves a text file per grid page listing the indices and room info.

    Args:
        json_path: path to the perspective_eval_dataset.json
        output_dir: output directory (default: same dir as json / examination_grids)
        grid_size: number of rows and cols per grid page (default: 5)
    """
    from PIL import Image, ImageDraw, ImageFont

    with open(json_path) as f:
        data = json.load(f)
    samples = data['samples']

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(json_path), 'examination_grids')
    os.makedirs(output_dir, exist_ok=True)

    n_per_page = grid_size * grid_size  # 25
    n_pages = (len(samples) + n_per_page - 1) // n_per_page

    # Target cell size
    cell_w, cell_h = IMG_W, IMG_H  # 640 x 480
    header_h = 40  # text header height

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    print(f"  Generating {n_pages} grid pages ({grid_size}x{grid_size}) for {len(samples)} samples...")

    for page_idx in range(n_pages):
        start = page_idx * n_per_page
        end = min(start + n_per_page, len(samples))
        page_samples = samples[start:end]

        # Create grid image
        grid_w = grid_size * cell_w
        grid_h = grid_size * (cell_h + header_h)
        grid_img = Image.new('RGB', (grid_w, grid_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(grid_img)

        # Text file content for this page
        text_lines = [f"Page {page_idx} (indices {start}-{end-1})\n{'='*60}\n"]

        for i, sample in enumerate(page_samples):
            global_idx = start + i
            row = i // grid_size
            col = i % grid_size

            x_offset = col * cell_w
            y_offset = row * (cell_h + header_h)

            # Draw text header
            header_text = f"#{global_idx}"
            detail_text = f"vis={sample['n_visible_assets']} score={sample['score']:.1f}"
            draw.rectangle([x_offset, y_offset, x_offset + cell_w, y_offset + header_h],
                           fill=(255, 255, 255))
            draw.text((x_offset + 5, y_offset + 2), header_text, fill=(0, 0, 0), font=font)
            draw.text((x_offset + 80, y_offset + 6), detail_text, fill=(100, 100, 100), font=font_small)

            # Load and paste perspective image
            persp_path = sample['perspective_image']
            try:
                persp_img = Image.open(persp_path).convert('RGB')
                persp_img = persp_img.resize((cell_w, cell_h), Image.LANCZOS)
                grid_img.paste(persp_img, (x_offset, y_offset + header_h))
            except Exception as e:
                # Draw placeholder
                draw.rectangle([x_offset, y_offset + header_h,
                                x_offset + cell_w, y_offset + header_h + cell_h],
                               fill=(200, 200, 200))
                draw.text((x_offset + 10, y_offset + header_h + 10),
                          f"Failed: {e}", fill=(255, 0, 0), font=font_small)

            # Add to text file
            text_lines.append(
                f"#{global_idx}: {sample['room_name']} (view {sample['view_idx']})\n"
                f"  vis={sample['n_visible_assets']}, fov={sample['n_in_fov']}, "
                f"score={sample['score']:.2f}\n"
                f"  depth: mean={sample['depth_mean']:.1f}, std={sample['depth_std']:.2f}, "
                f"min={sample['depth_min']:.2f}\n"
                f"  path: {sample['perspective_image']}\n"
            )

        # Save grid image
        grid_path = os.path.join(output_dir, f'grid_page_{page_idx:03d}.jpg')
        grid_img.save(grid_path, quality=90)

        # Save text file
        text_path = os.path.join(output_dir, f'grid_page_{page_idx:03d}.txt')
        with open(text_path, 'w') as f:
            f.write('\n'.join(text_lines))

    print(f"  Saved {n_pages} grid pages to {output_dir}/")
    print(f"  Review the grids and note the index numbers to keep/reject.")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct perspective image eval dataset from ERP_3D_FRONT_test"
    )
    parser.add_argument('--root', type=str, default='datasets/ERP_3D_FRONT_test',
                        help='Root directory of ERP_3D_FRONT_test dataset')
    parser.add_argument('--output', type=str, default='evals/perspective_eval_dataset.json',
                        help='Output JSON file path')
    parser.add_argument('--min_visible_assets', type=int, default=3,
                        help='Minimum number of depth-verified visible assets')
    parser.add_argument('--min_depth_mean', type=float, default=1.0,
                        help='Minimum mean depth in meters (reject wall-facing views)')
    parser.add_argument('--min_depth_min', type=float, default=0.3,
                        help='Minimum closest depth in meters (reject extreme close-ups)')
    parser.add_argument('--depth_tolerance', type=float, default=0.5,
                        help='Relative tolerance for depth-based visibility verification')
    parser.add_argument('--max_per_room', type=int, default=1,
                        help='Maximum selected views per room')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of selected views')
    parser.add_argument('--vis_count', type=int, default=20,
                        help='Number of views to visualize')
    parser.add_argument('--examine', action='store_true',
                        help='Generate 5x5 grid images for manual examination')
    parser.add_argument('--examine_from_json', type=str, default=None,
                        help='Generate grids from existing JSON (skip scoring)')
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)

    # ── Discover all rooms ──
    rooms = []
    for uuid_dir in sorted(root.iterdir()):
        if not uuid_dir.is_dir():
            continue
        for room_dir in sorted(uuid_dir.iterdir()):
            if not room_dir.is_dir():
                continue
            if (room_dir / 'perspective').exists():
                rooms.append(room_dir)

    print(f"Found {len(rooms)} rooms with perspective images")

    # ── Process all rooms ──
    selected = []
    rejected_reasons = {'no_data': 0, 'no_good_views': 0}
    all_scores = []

    for room_dir in tqdm(rooms, desc="Scoring perspective views"):
        results = process_room(room_dir, depth_tolerance=args.depth_tolerance)

        if results is None:
            rejected_reasons['no_data'] += 1
            continue

        # Track all scores for statistics
        for r in results:
            all_scores.append(r['score'])

        # Filter by quality criteria
        good_views = [
            r for r in results
            if r['metrics']['n_visible_assets'] >= args.min_visible_assets
            and r['metrics']['depth_mean'] >= args.min_depth_mean
            and r['metrics']['depth_min'] >= args.min_depth_min
        ]

        if not good_views:
            rejected_reasons['no_good_views'] += 1
            continue

        # Select top N views per room
        abs_room = str(room_dir.resolve())
        for view_info in good_views[:args.max_per_room]:
            view_idx = view_info['view_idx']
            sample = {
                'room_path': abs_room,
                'room_name': room_dir.name,
                'uuid': room_dir.parent.name,
                'erp_image': f'{abs_room}/erp/{view_idx:04d}_colors.png',
                'perspective_image': f'{abs_room}/perspective/{view_idx:04d}_colors.png',
                'perspective_depth': f'{abs_room}/perspective/{view_idx:04d}_depth.npy',
                'mesh_dir': f'{abs_room}/mesh/',
                'view_idx': view_idx,
                'camera_location': view_info['camera_location'],
                'has_erp': view_info['has_erp'],
                **view_info['metrics'],
            }
            selected.append(sample)

    # Sort by score descending and assign index
    selected.sort(key=lambda x: x['score'], reverse=True)
    for i, s in enumerate(selected):
        s['idx'] = i

    # ── Statistics ──
    all_scores = np.array(all_scores)
    n_rooms_with_good = len(rooms) - rejected_reasons['no_data'] - rejected_reasons['no_good_views']

    stats = {
        'total_rooms': len(rooms),
        'rooms_with_valid_data': len(rooms) - rejected_reasons['no_data'],
        'rooms_with_good_views': n_rooms_with_good,
        'rooms_no_data': rejected_reasons['no_data'],
        'rooms_no_good_views': rejected_reasons['no_good_views'],
        'total_selected': len(selected),
        'score_distribution': {
            'all_views_mean': round(float(all_scores.mean()), 2) if len(all_scores) > 0 else 0,
            'all_views_median': round(float(np.median(all_scores)), 2) if len(all_scores) > 0 else 0,
            'all_views_std': round(float(all_scores.std()), 2) if len(all_scores) > 0 else 0,
            'selected_mean': round(float(np.mean([s['score'] for s in selected])), 2) if selected else 0,
            'selected_min': round(float(min(s['score'] for s in selected)), 2) if selected else 0,
        },
        'visible_assets_distribution': {},
    }

    # Distribution of n_visible_assets in selected samples
    if selected:
        vis_counts = [s['n_visible_assets'] for s in selected]
        for n in sorted(set(vis_counts)):
            stats['visible_assets_distribution'][str(n)] = vis_counts.count(n)

    # ── Save output ──
    output = {
        'dataset_info': {
            'source': str(root),
            'description': 'Selected perspective images for evaluation (ERP, perspective, mesh triplets)',
            'selection_criteria': {
                'min_visible_assets': args.min_visible_assets,
                'min_depth_mean': args.min_depth_mean,
                'min_depth_min': args.min_depth_min,
                'depth_tolerance': args.depth_tolerance,
                'max_per_room': args.max_per_room,
            },
            'camera_intrinsics': {
                'width': IMG_W, 'height': IMG_H,
                'fov_horizontal_deg': FOV_DEG,
                'fx': FX, 'fy': FY, 'cx': CX, 'cy': CY,
            },
            'statistics': stats,
        },
        'samples': selected,
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"  Total rooms scanned:      {stats['total_rooms']}")
    print(f"  Rooms with valid data:    {stats['rooms_with_valid_data']}")
    print(f"  Rooms with good views:    {stats['rooms_with_good_views']}")
    print(f"  Rooms rejected (no data): {stats['rooms_no_data']}")
    print(f"  Rooms rejected (quality): {stats['rooms_no_good_views']}")
    print(f"  Total selected images:    {stats['total_selected']}")
    print(f"{'='*60}")
    if selected:
        print(f"  Score - mean: {stats['score_distribution']['selected_mean']:.2f}, "
              f"min: {stats['score_distribution']['selected_min']:.2f}")
        print(f"  Visible assets distribution: {stats['visible_assets_distribution']}")
    print(f"\n  Saved to: {args.output}")

    # ── Optional visualization ──
    if args.visualize and selected:
        vis_dir = os.path.join(os.path.dirname(args.output), 'perspective_selection_viz')
        print(f"\nGenerating visualizations...")
        visualize_selected_views(selected, root, vis_dir, count=args.vis_count)

    # ── Optional examination grids ──
    if args.examine and selected:
        print(f"\nGenerating examination grids...")
        generate_examination_grids(args.output)

    return output


if __name__ == '__main__':
    args = parse_args()

    # Shortcut: generate grids from existing JSON without re-scoring
    if args.examine_from_json:
        print(f"Generating examination grids from {args.examine_from_json}")
        generate_examination_grids(args.examine_from_json)
    else:
        main()
