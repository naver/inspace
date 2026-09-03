# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Asset-Aware Cross-Attention Mask Generation for TRELLIS 2

This module is used ONLY in Stage 2 (Structured Latent Flow: Shape/Texture Generation).
Stage 1 (Sparse Structure Generation) uses its own spatial attention in
  trellis2/trainers/flow_matching/mixins/erp_image_conditioned.py

================================================================================
  Stage 2 Usage Summary
================================================================================

  [Data Loading] — erp_structured_latent.py (dataset __getitem__)
    filter_visible_assets()      : Filter assets by visibility (>50%) during loading
      └── is_point_inside_obb()  : Check if camera is inside OBB
      └── calculate_asset_visibility() : Per-face visibility calculation
          └── obb_to_corners()   : Convert OBB [7] → 8 corner vertices

  [Training / Inference] — erp_slat_conditioned.py (trainer mixin)
    compute_overlap_groups()     : Group overlapping assets via Union-Find (→ self-attn mask)
      └── check_obb_overlap_sat(): SAT algorithm for OBB overlap detection
          └── obb_to_corners(), get_obb_axes(), project_obb_to_axis()
    create_per_part_cross_attn_masks() : Per-part cross-attn masks (→ cubemap projection)
      └── _compute_asset_token_mask()  : Project single OBB → cubemap token mask
          └── obb_to_corners()

  [Visualization] — erp_structured_latent.py (dataset visualize_cross_attn_mask)
    create_per_part_cross_attn_masks() : Same function as training (ensures consistency)

  [Visualization-only helpers] — NOT used in training/inference
    save_visibility_visualization()
    _save_token_selection_visualization()
    _save_bbox_projection_visualization()
    _get_display_name()
    _compute_asset_token_mask_for_viz()

================================================================================
  Coordinate Systems
================================================================================
- O-Voxel space: normalized [-0.5, 0.5]^3 (center at 0)
- Camera space: relative to camera center, same orientation as O-Voxel
- Cubemap: 6 faces (front, right, back, left, top, bottom), FOV=120

  Face Directions (in O-Voxel space):
  - front: +Y    - right: +X    - back: -Y
  - left:  -X    - top:   +Z    - bottom: -Z
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt


# Cubemap face directions
FACE_DIRECTIONS = torch.tensor([
    [0.0, 1.0, 0.0],   # front: +Y
    [1.0, 0.0, 0.0],   # right: +X
    [0.0, -1.0, 0.0],  # back: -Y
    [-1.0, 0.0, 0.0],  # left: -X
    [0.0, 0.0, 1.0],   # top: +Z
    [0.0, 0.0, -1.0],  # bottom: -Z
], dtype=torch.float32)

# Face up vectors for proper orientation
FACE_UP_VECTORS = torch.tensor([
    [0.0, 0.0, 1.0],   # front: Z up
    [0.0, 0.0, 1.0],   # right: Z up
    [0.0, 0.0, 1.0],   # back: Z up
    [0.0, 0.0, 1.0],   # left: Z up
    [0.0, -1.0, 0.0],  # top: -Y up (looking down)
    [0.0, 1.0, 0.0],   # bottom: +Y up (looking up)
], dtype=torch.float32)

# FP16-safe large negative value for masking
MASK_NEG_INF = -1e4


def obb_to_corners(obb: torch.Tensor) -> torch.Tensor:
    """
    Convert OBB (oriented bounding box) to 8 corner vertices.

    Args:
        obb: [7] tensor [x, y, z, sx, sy, sz, rotation]
             - x, y, z: center position (in O-Voxel space [-0.5, 0.5])
             - sx, sy, sz: FULL extents (full dimensions, NOT half-extents)
             - rotation: yaw angle in radians (around Z axis)

    Returns:
        corners: [8, 3] tensor of corner positions
    """
    cx, cy, cz = obb[0], obb[1], obb[2]
    sx, sy, sz = obb[3], obb[4], obb[5]
    rot = obb[6]

    # Create 8 corners of axis-aligned box centered at origin
    hx, hy, hz = sx / 2, sy / 2, sz / 2

    # Corner offsets
    signs = torch.tensor([
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ], dtype=obb.dtype, device=obb.device)

    corners = signs * torch.tensor([hx, hy, hz], dtype=obb.dtype, device=obb.device)

    # Apply rotation around Z axis (yaw)
    cos_r = torch.cos(rot)
    sin_r = torch.sin(rot)
    rot_matrix = torch.tensor([
        [cos_r, -sin_r, 0],
        [sin_r, cos_r, 0],
        [0, 0, 1]
    ], dtype=obb.dtype, device=obb.device)

    corners = corners @ rot_matrix.T

    # Translate to center
    corners = corners + torch.tensor([cx, cy, cz], dtype=obb.dtype, device=obb.device)

    return corners


# Stage 2
def create_overall_spatial_mask_sparse(
    voxel_coords: torch.Tensor,
    camera_center: torch.Tensor,
    voxel_resolution: int = 32,
    tokens_per_face: int = 1029,
    fov_degrees: float = 120.0,
) -> torch.Tensor:
    """
    Create per-voxel spatial cross-attention mask for Stage 2 overall scene.

    Unlike create_overall_spatial_attention_mask() which generates a dense grid,
    this takes actual sparse voxel coordinates from the SparseTensor.

    Each voxel's 3D position determines which cubemap face(s) it's visible from
    (based on direction from camera center). Boolean True = attend, False = block.

    Args:
        voxel_coords: [N, 3] integer voxel coordinates (from SparseTensor, 0-indexed)
        camera_center: [3] camera center in O-Voxel normalized space [-0.5, 0.5]
        voxel_resolution: Voxel grid resolution (default: 32 for Stage 2)
        tokens_per_face: DINO tokens per face (default: 1029 for DINOv3)
        fov_degrees: Cubemap FOV in degrees (default: 120)

    Returns:
        mask: [N, 6 * tokens_per_face] boolean mask (True = attend)
    """
    device = voxel_coords.device
    N = voxel_coords.shape[0]
    total_tokens = 6 * tokens_per_face

    # Convert integer voxel coords to normalized positions [-0.5, 0.5]
    # Voxel coord i maps to center: (i + 0.5) / resolution - 0.5
    voxel_positions = (voxel_coords.float() + 0.5) / voxel_resolution - 0.5  # [N, 3]

    # Direction from camera to each voxel
    cam = camera_center.to(device=device, dtype=torch.float32).unsqueeze(0)
    directions = voxel_positions - cam  # [N, 3]
    directions = F.normalize(directions, p=2, dim=-1)

    # Cosine similarity with each face direction
    face_dirs = FACE_DIRECTIONS.to(device=device, dtype=torch.float32)  # [6, 3]
    cos_sim = torch.matmul(directions, face_dirs.T)  # [N, 6]

    # FOV threshold: voxel is visible from face if cos_sim >= cos(fov/2)
    fov_rad = np.radians(fov_degrees)
    cos_threshold = np.cos(fov_rad / 2)
    visible = cos_sim >= cos_threshold  # [N, 6] boolean

    # Expand face-level visibility to token-level mask
    # [N, 6] -> [N, 6, tokens_per_face] -> [N, 6 * tokens_per_face]
    mask = visible.unsqueeze(-1).expand(-1, -1, tokens_per_face)
    mask = mask.reshape(N, total_tokens)

    # Safety: if a voxel is outside all FOVs, enable all tokens as fallback
    no_visibility = ~mask.any(dim=1)  # [N]
    if no_visibility.any():
        mask[no_visibility] = True

    return mask


# =============================================================================
# 3D Bounding Box Overlap Detection
# =============================================================================

def get_obb_axes(obb: torch.Tensor) -> torch.Tensor:
    """
    Get the 3 principal axes of an OBB (rotated by yaw angle).

    Args:
        obb: [7] tensor [x, y, z, sx, sy, sz, rotation]

    Returns:
        axes: [3, 3] tensor where each row is an axis direction
    """
    rot = obb[6]
    cos_r = torch.cos(rot)
    sin_r = torch.sin(rot)

    # Rotated X, Y axes; Z axis unchanged
    axis_x = torch.tensor([cos_r, sin_r, 0], dtype=obb.dtype, device=obb.device)
    axis_y = torch.tensor([-sin_r, cos_r, 0], dtype=obb.dtype, device=obb.device)
    axis_z = torch.tensor([0, 0, 1], dtype=obb.dtype, device=obb.device)

    return torch.stack([axis_x, axis_y, axis_z], dim=0)


def project_obb_to_axis(corners: torch.Tensor, axis: torch.Tensor) -> Tuple[float, float]:
    """
    Project OBB corners onto an axis and return min/max projection values.

    Args:
        corners: [8, 3] corner positions
        axis: [3] axis direction (should be normalized)

    Returns:
        (min_proj, max_proj) projection extents
    """
    projections = torch.matmul(corners, axis)
    return projections.min().item(), projections.max().item()


def check_obb_overlap_sat(obb1: torch.Tensor, obb2: torch.Tensor,
                          margin: float = 0.02) -> bool:
    """
    Check if two OBBs overlap using the Separating Axis Theorem (SAT).

    The SAT states that two convex objects do NOT overlap if there exists
    a separating axis (a line) onto which their projections do not overlap.
    For two OBBs, we need to test 15 potential separating axes:
    - 3 face normals of OBB1
    - 3 face normals of OBB2
    - 9 edge-edge cross products

    Args:
        obb1: [7] first OBB [x, y, z, sx, sy, sz, rotation]
        obb2: [7] second OBB
        margin: expansion margin to consider "close" boxes as overlapping

    Returns:
        True if OBBs overlap (or are within margin)
    """
    device = obb1.device
    dtype = obb1.dtype

    # Get corners with margin expansion
    obb1_expanded = obb1.clone()
    obb2_expanded = obb2.clone()
    obb1_expanded[3:6] = obb1_expanded[3:6] + margin
    obb2_expanded[3:6] = obb2_expanded[3:6] + margin

    corners1 = obb_to_corners(obb1_expanded)  # [8, 3]
    corners2 = obb_to_corners(obb2_expanded)  # [8, 3]

    # Get axes
    axes1 = get_obb_axes(obb1)  # [3, 3]
    axes2 = get_obb_axes(obb2)  # [3, 3]

    # Collect all axes to test
    test_axes = []

    # Face normals from both OBBs
    for i in range(3):
        test_axes.append(axes1[i])
        test_axes.append(axes2[i])

    # Edge-edge cross products
    for i in range(3):
        for j in range(3):
            cross = torch.linalg.cross(axes1[i], axes2[j])
            norm = torch.norm(cross)
            if norm > 1e-6:
                test_axes.append(cross / norm)

    # Test each axis
    for axis in test_axes:
        min1, max1 = project_obb_to_axis(corners1, axis)
        min2, max2 = project_obb_to_axis(corners2, axis)

        # Check for separation
        if max1 < min2 or max2 < min1:
            return False  # Found separating axis, no overlap

    return True  # No separating axis found, boxes overlap


def compute_overlap_groups(
    obbs: torch.Tensor,
    margin: float = 0.02
) -> List[List[int]]:
    """
    Group OBBs that overlap with each other using Union-Find algorithm.

    Creates groups where all OBBs in a group have at least one overlap chain
    connecting them. This is useful for intra-asset attention grouping.

    Args:
        obbs: [N, 7] oriented bounding boxes
        margin: expansion margin for overlap detection

    Returns:
        List of groups, each group is a list of OBB indices
    """
    n = obbs.shape[0]
    if n == 0:
        return []

    # Union-Find data structure
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        # Union by rank
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1

    # Check all pairs for overlap
    for i in range(n):
        for j in range(i + 1, n):
            if check_obb_overlap_sat(obbs[i], obbs[j], margin=margin):
                union(i, j)

    # Collect groups
    groups_dict = {}
    for i in range(n):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)

    return list(groups_dict.values())


def create_intra_asset_attention_mask(
    part_layouts: List[slice],
    overlap_groups: List[List[int]],
    total_voxels: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create self-attention mask for intra-asset attention.

    Within each overlap group, all parts can attend to each other.
    Parts in different groups cannot attend to each other.
    Overall (index 0) always attends only to itself.

    Args:
        part_layouts: list of slices [overall, asset0, asset1, ...]
        overlap_groups: list of groups of asset indices (0-indexed from assets, not including overall)
        total_voxels: total number of voxels
        device: torch device
        dtype: torch dtype

    Returns:
        attention_mask: [total_voxels, total_voxels]
            0 means attend, MASK_NEG_INF means don't attend
    """
    # Initialize mask: block everything
    mask = torch.full((total_voxels, total_voxels), MASK_NEG_INF, dtype=dtype, device=device)

    # Overall (part 0) attends only to itself
    overall_slice = part_layouts[0]
    overall_start, overall_end = overall_slice.start, overall_slice.stop
    mask[overall_start:overall_end, overall_start:overall_end] = 0.0

    # For each overlap group, all assets in the group can attend to each other
    for group in overlap_groups:
        # Group indices are asset indices (0-indexed),
        # but in part_layouts they are at index+1 (since 0 is overall)
        group_slices = []
        for asset_idx in group:
            part_idx = asset_idx + 1
            if part_idx < len(part_layouts):
                group_slices.append(part_layouts[part_idx])

        # Allow attention within this group
        for slice_i in group_slices:
            for slice_j in group_slices:
                mask[slice_i.start:slice_i.stop, slice_j.start:slice_j.stop] = 0.0

    # Assets not in any group (singletons) attend only to themselves
    grouped_assets = set()
    for group in overlap_groups:
        grouped_assets.update(group)

    for asset_idx in range(len(part_layouts) - 1):
        if asset_idx not in grouped_assets:
            part_idx = asset_idx + 1
            if part_idx < len(part_layouts):
                s = part_layouts[part_idx]
                mask[s.start:s.stop, s.start:s.stop] = 0.0

    return mask


# =============================================================================
# Visibility Filtering Functions
# =============================================================================

def is_point_inside_obb(point: torch.Tensor, obb: torch.Tensor) -> bool:
    """
    Check if a point is inside an oriented bounding box.

    Args:
        point: [3] point coordinates
        obb: [7] oriented bounding box [x, y, z, sx, sy, sz, rotation]

    Returns:
        True if point is inside the OBB
    """
    cx, cy, cz = obb[0].item(), obb[1].item(), obb[2].item()
    sx, sy, sz = obb[3].item(), obb[4].item(), obb[5].item()
    rot = obb[6].item()

    # Transform point to OBB local coordinates (undo rotation and translation)
    local_point = point - torch.tensor([cx, cy, cz], dtype=point.dtype, device=point.device)

    # Inverse rotation (rotate by -rot)
    cos_r = np.cos(-rot)
    sin_r = np.sin(-rot)
    rot_matrix_inv = torch.tensor([
        [cos_r, -sin_r, 0],
        [sin_r, cos_r, 0],
        [0, 0, 1]
    ], dtype=point.dtype, device=point.device)
    local_point = rot_matrix_inv @ local_point

    # Check if inside axis-aligned box
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    return (
        -hx <= local_point[0].item() <= hx and
        -hy <= local_point[1].item() <= hy and
        -hz <= local_point[2].item() <= hz
    )


def calculate_asset_visibility(
    obb: torch.Tensor,
    camera_center: torch.Tensor,
    fov_degrees: float = 120.0,
    image_size: int = 512,
) -> Tuple[float, Optional[str]]:
    """
    Calculate the maximum visibility percentage of an asset across all cubemap faces.

    Uses frustum clipping and edge-touching heuristics to estimate visibility.

    Args:
        obb: [7] oriented bounding box [x, y, z, sx, sy, sz, rotation]
        camera_center: [3] camera center in O-Voxel normalized space
        fov_degrees: cubemap FOV (default: 120)
        image_size: cubemap image size (default: 512)

    Returns:
        Tuple of (max_visibility_fraction, best_face_name)
        max_visibility_fraction is in range [0, 1]
        best_face_name is the face with highest visibility, or None if not visible
    """
    device = obb.device
    dtype = obb.dtype

    # Get bbox corners
    corners = obb_to_corners(obb)  # [8, 3]

    fov_rad = np.radians(fov_degrees)
    cos_threshold = np.cos(fov_rad / 2)
    tan_half_fov = np.tan(fov_rad / 2)
    k = tan_half_fov  # Frustum half-angle tangent

    face_dirs = FACE_DIRECTIONS.to(device=device, dtype=dtype)
    face_ups = FACE_UP_VECTORS.to(device=device, dtype=dtype)
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']

    max_visibility = 0.0
    best_face = None

    for face_idx in range(6):
        face_dir = face_dirs[face_idx]
        face_up = face_ups[face_idx]

        # Face right vector
        face_right = torch.linalg.cross(face_dir, face_up)
        face_right = face_right / torch.norm(face_right)
        face_up_actual = torch.linalg.cross(face_right, face_dir)

        # Project corners that are in front of camera (relative to this face)
        u_pixels = []
        v_pixels = []
        in_frustum_count = 0

        for corner in corners:
            direction = corner - camera_center
            distance = torch.norm(direction)
            if distance < 1e-6:
                continue

            # Check if corner is in front of this face's view
            forward_dist = torch.dot(direction, face_dir)
            if forward_dist <= 0:
                continue

            direction_norm = direction / distance
            cos_angle = torch.dot(direction_norm, face_dir)

            # Project to face plane
            u_coord = torch.dot(direction, face_right) / forward_dist
            v_coord = torch.dot(direction, face_up_actual) / forward_dist

            # Normalize by tan(fov/2)
            u_normalized = u_coord / tan_half_fov
            v_normalized = v_coord / tan_half_fov

            # Check if within frustum
            if abs(u_normalized.item()) <= 1 and abs(v_normalized.item()) <= 1:
                in_frustum_count += 1

            # Convert to pixel coordinates
            u_pixel = ((u_normalized + 1) / 2 * image_size).item()
            v_pixel = ((1 - v_normalized) / 2 * image_size).item()

            u_pixels.append(u_pixel)
            v_pixels.append(v_pixel)

        if len(u_pixels) == 0:
            continue

        # Compute full bbox (before clipping)
        u_min_full = min(u_pixels)
        u_max_full = max(u_pixels)
        v_min_full = min(v_pixels)
        v_max_full = max(v_pixels)

        full_width = u_max_full - u_min_full
        full_height = v_max_full - v_min_full
        full_area = full_width * full_height

        if full_area <= 0:
            continue

        # Compute clipped bbox (within image bounds)
        u_min_clip = max(0, u_min_full)
        u_max_clip = min(image_size, u_max_full)
        v_min_clip = max(0, v_min_full)
        v_max_clip = min(image_size, v_max_full)

        clipped_width = max(0, u_max_clip - u_min_clip)
        clipped_height = max(0, v_max_clip - v_min_clip)
        clipped_area = clipped_width * clipped_height

        if clipped_area <= 0:
            continue

        # Base visibility from area comparison
        base_visibility = clipped_area / full_area

        # Edge adjustment: if bbox touches image boundary, reduce visibility
        edge_tolerance = 3.0
        edge_length = 0.0
        tight_perimeter = 2 * (clipped_width + clipped_height)

        if u_min_clip <= edge_tolerance:
            edge_length += clipped_height
        if u_max_clip >= image_size - edge_tolerance:
            edge_length += clipped_height
        if v_min_clip <= edge_tolerance:
            edge_length += clipped_width
        if v_max_clip >= image_size - edge_tolerance:
            edge_length += clipped_width

        if edge_length > 0 and tight_perimeter > 0:
            edge_fraction = edge_length / tight_perimeter
            edge_adjustment = 1.0 - (edge_fraction * 0.5)  # Max 50% reduction
            visibility = base_visibility * edge_adjustment
        else:
            visibility = base_visibility

        if visibility > max_visibility:
            max_visibility = visibility
            best_face = face_names[face_idx]

    return max_visibility, best_face


def create_per_part_cross_attn_masks(
    obbs: torch.Tensor,
    camera_center: torch.Tensor,
    num_parts: int,
    tokens_per_face: int = 1029,
    fov_degrees: float = 120.0,
    image_size: int = 512,
    patch_size: int = 16,
    expand_pixels: int = 28,
    overall_voxel_coords: torch.Tensor = None,
    layout_voxel_coords: torch.Tensor = None,
    has_layout: bool = False,
    voxel_resolution: int = 32,
) -> List[torch.Tensor]:
    """
    Create per-part cross-attention masks for ERP-style OmniPart forward.

    Args:
        obbs: [num_assets, 7] oriented bounding boxes
        camera_center: [3] camera center in O-Voxel normalized space
        num_parts: total number of parts (overall + [layout] + assets)
        tokens_per_face: DINO tokens per face (default: 1029 = 1 CLS + 4 registers + 1024 patches)
        fov_degrees: cubemap FOV (default: 120)
        image_size: cubemap image size (default: 512)
        patch_size: DINO patch size (default: 16 for DINOv3 vitl16)
        expand_pixels: expand bbox by this many pixels
        overall_voxel_coords: [N_overall, 3] integer voxel coords for overall part.
            If provided, creates per-voxel spatial mask [N_overall, total_tokens].
            If None, creates broadcast mask [total_tokens] (all True).
        layout_voxel_coords: [N_layout, 3] integer voxel coords for layout part.
            If provided (and has_layout=True), creates per-voxel spatial mask [N_layout, total_tokens].
        has_layout: Whether layout part is present (index 1 in part_layouts).
        voxel_resolution: Voxel grid resolution (default: 32 for Stage 2)

    Returns:
        List of boolean masks [overall_mask, (layout_mask), asset0_mask, asset1_mask, ...]
        - overall_mask: [N_overall, total_tokens] if overall_voxel_coords provided,
                       else [total_tokens] (broadcast)
        - layout_mask: [N_layout, total_tokens] if layout_voxel_coords provided,
                      else [total_tokens] (broadcast). Only present if has_layout=True.
        - asset masks: [total_tokens] each (broadcast to all voxels in asset)
    """
    device = camera_center.device
    total_tokens = 6 * tokens_per_face
    num_assets = obbs.shape[0] if obbs is not None else 0

    masks = []

    # Overall mask (index 0): per-voxel spatial mask or broadcast all-True
    if overall_voxel_coords is not None:
        overall_mask = create_overall_spatial_mask_sparse(
            voxel_coords=overall_voxel_coords,
            camera_center=camera_center,
            voxel_resolution=voxel_resolution,
            tokens_per_face=tokens_per_face,
            fov_degrees=fov_degrees,
        )  # [N_overall, total_tokens] boolean
    else:
        overall_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
    masks.append(overall_mask)

    # Layout mask (index 1, if present): per-voxel spatial mask (same style as overall)
    if has_layout:
        if layout_voxel_coords is not None:
            layout_mask = create_overall_spatial_mask_sparse(
                voxel_coords=layout_voxel_coords,
                camera_center=camera_center,
                voxel_resolution=voxel_resolution,
                tokens_per_face=tokens_per_face,
                fov_degrees=fov_degrees,
            )  # [N_layout, total_tokens] boolean
        else:
            layout_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
        masks.append(layout_mask)

    # Asset masks: attend only to projected bbox region
    for i in range(num_assets):
        obb = obbs[i]
        asset_mask = _compute_asset_token_mask(
            obb=obb,
            camera_center=camera_center,
            tokens_per_face=tokens_per_face,
            fov_degrees=fov_degrees,
            image_size=image_size,
            patch_size=patch_size,
            expand_pixels=expand_pixels,
        )
        masks.append(asset_mask)

    # Pad with full masks if num_parts > len(masks)
    while len(masks) < num_parts:
        masks.append(overall_mask.clone())

    return masks


def _compute_asset_token_mask(
    obb: torch.Tensor,
    camera_center: torch.Tensor,
    tokens_per_face: int = 1029,
    fov_degrees: float = 120.0,
    image_size: int = 512,
    patch_size: int = 16,
    expand_pixels: int = 28,
) -> torch.Tensor:
    """
    Compute boolean mask of which cubemap tokens an asset's bbox projects to.

    DINOv3 token layout per face (1029 tokens):
      - Token 0: CLS token
      - Tokens 1-4: 4 register tokens
      - Tokens 5-1028: 1024 patch tokens (32x32 grid for 512px / 16px patch)

    Args:
        obb: [7] oriented bounding box
        camera_center: [3] camera center
        tokens_per_face: DINO tokens per face (1029 = 1 CLS + 4 registers + 1024 patches)
        fov_degrees: cubemap FOV
        image_size: cubemap image size
        patch_size: DINO patch size (16 for DINOv3 vitl16)
        expand_pixels: bbox expansion

    Returns:
        mask: [6 * tokens_per_face] boolean, True = token is in projected bbox
    """
    device = obb.device
    dtype = obb.dtype
    total_tokens = 6 * tokens_per_face

    # Get bbox corners
    corners = obb_to_corners(obb)  # [8, 3]

    tokens_per_row = int(np.ceil(image_size / patch_size))
    fov_rad = np.radians(fov_degrees)
    tan_half_fov = np.tan(fov_rad / 2)

    face_dirs = FACE_DIRECTIONS.to(device=device, dtype=dtype)
    face_ups = FACE_UP_VECTORS.to(device=device, dtype=dtype)

    # Token mask
    token_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)

    for face_idx in range(6):
        face_dir = face_dirs[face_idx]
        face_up = face_ups[face_idx]

        # Face right vector
        face_right = torch.linalg.cross(face_dir, face_up)
        face_right = face_right / torch.norm(face_right)
        face_up_actual = torch.linalg.cross(face_right, face_dir)

        # Project corners to this face
        u_pixels = []
        v_pixels = []

        for corner in corners:
            direction = corner - camera_center
            distance = torch.norm(direction)
            if distance < 1e-6:
                continue

            # Forward distance to face plane
            forward_dist = torch.dot(direction, face_dir)
            if forward_dist <= 0:
                continue

            # Project to face tangent plane
            u_coord = torch.dot(direction, face_right) / forward_dist
            v_coord = torch.dot(direction, face_up_actual) / forward_dist

            u_normalized = u_coord / tan_half_fov
            v_normalized = v_coord / tan_half_fov

            u_pixel = ((u_normalized + 1) / 2 * image_size).item()
            v_pixel = ((1 - v_normalized) / 2 * image_size).item()

            u_pixels.append(u_pixel)
            v_pixels.append(v_pixel)

        if len(u_pixels) == 0:
            continue

        # Compute bounding rectangle with expansion
        u_min = max(0, min(u_pixels) - expand_pixels)
        u_max = min(image_size, max(u_pixels) + expand_pixels)
        v_min = max(0, min(v_pixels) - expand_pixels)
        v_max = min(image_size, max(v_pixels) + expand_pixels)

        # Convert to token indices
        scale = tokens_per_row / image_size
        t_u_min = int(np.floor(u_min * scale))
        t_u_max = int(np.ceil(u_max * scale))
        t_v_min = int(np.floor(v_min * scale))
        t_v_max = int(np.ceil(v_max * scale))

        t_u_min = max(0, t_u_min)
        t_u_max = min(tokens_per_row, t_u_max)
        t_v_min = max(0, t_v_min)
        t_v_max = min(tokens_per_row, t_v_max)

        # Set token mask
        # DINOv3: patch tokens start at index 5 (skip 1 CLS + 4 registers)
        num_special_tokens = 5
        for t_v in range(t_v_min, t_v_max):
            for t_u in range(t_u_min, t_u_max):
                token_idx = t_v * tokens_per_row + t_u + num_special_tokens
                if 0 <= token_idx < tokens_per_face:
                    global_idx = face_idx * tokens_per_face + token_idx
                    token_mask[global_idx] = True

    # If no tokens found, return full mask as fallback
    if not token_mask.any():
        token_mask[:] = True

    return token_mask


def filter_visible_assets(
    obbs: torch.Tensor,
    camera_center: torch.Tensor,
    visibility_threshold: float = 0.5,
    fov_degrees: float = 120.0,
    image_size: int = 512,
    save_visualization: bool = False,
    sample_dir: Optional[str] = None,
    asset_names: Optional[List[str]] = None,
) -> Tuple[List[int], List[float]]:
    """
    Filter assets based on visibility threshold.

    An asset is included if:
    1. Camera is NOT inside the asset's bounding box
    2. Asset has >= visibility_threshold visibility on at least one cubemap face

    Args:
        obbs: [N, 7] oriented bounding boxes
        camera_center: [3] camera center in O-Voxel normalized space
        visibility_threshold: minimum visibility fraction (default: 0.5 = 50%)
        fov_degrees: cubemap FOV (default: 120)
        image_size: cubemap image size (default: 512)
        save_visualization: whether to save visibility visualization to sample_dir
        sample_dir: directory to save visualization (required if save_visualization=True)
        asset_names: list of asset names for labeling (optional)

    Returns:
        Tuple of:
        - List of indices of visible assets
        - List of max visibility values for each visible asset
    """
    visible_indices = []
    visible_visibilities = []
    visibility_info = []  # Store info for visualization

    for i, obb in enumerate(obbs):
        # Check if camera is inside this bbox
        cam_inside = is_point_inside_obb(camera_center, obb)
        if cam_inside:
            # Camera is inside this asset's bbox — asset is clearly visible
            # (common in indoor scenes: camera on/above bed, inside large furniture)
            visible_indices.append(i)
            visible_visibilities.append(1.0)
            visibility_info.append({
                'index': i,
                'max_visibility': 1.0,
                'best_face': 'all',
                'cam_inside': True,
                'included': True,
            })
            continue

        # Calculate visibility
        max_vis, best_face = calculate_asset_visibility(
            obb, camera_center, fov_degrees, image_size
        )

        included = max_vis >= visibility_threshold
        if included:
            visible_indices.append(i)
            visible_visibilities.append(max_vis)

        visibility_info.append({
            'index': i,
            'max_visibility': max_vis,
            'best_face': best_face,
            'cam_inside': False,
            'included': included,
        })

    # Save visualization if requested
    if save_visualization and sample_dir is not None:
        save_visibility_visualization(
            obbs=obbs,
            camera_center=camera_center,
            visibility_info=visibility_info,
            sample_dir=sample_dir,
            asset_names=asset_names,
            fov_degrees=fov_degrees,
            image_size=image_size,
        )

    return visible_indices, visible_visibilities


def save_visibility_visualization(
    obbs: torch.Tensor,
    camera_center: torch.Tensor,
    visibility_info: List[Dict],
    sample_dir: str,
    asset_names: Optional[List[str]] = None,
    fov_degrees: float = 120.0,
    image_size: int = 512,
    patch_size: int = 16,
    tokens_per_face: int = 1029,
    cubemap_folder: str = 'cubic_fov_120',
    view_idx: int = 0,
):
    """
    Save visibility and token selection visualization to sample directory.

    Creates two PNGs:
    1. asset_token_selection.png - DINO token selection per asset
    2. bbox_projection.png - Bbox projections on cubemap images

    Args:
        obbs: [N, 7] oriented bounding boxes
        camera_center: [3] camera center in O-Voxel normalized space
        visibility_info: list of visibility info dicts from filter_visible_assets
        sample_dir: directory to save visualization
        asset_names: list of asset names for labeling
        fov_degrees: cubemap FOV (default: 120)
        image_size: cubemap image size (default: 512)
        patch_size: DINO patch size (default: 16 for DINOv3 vitl16)
        tokens_per_face: DINO tokens per face (default: 1029)
        cubemap_folder: folder name for cubemap images (default: 'cubic_fov_120')
        view_idx: view index for cubemap images (default: 0)
    """
    # Check if visualization already exists
    vis_dir = os.path.join(sample_dir, 'visibility_debug')
    token_output_path = os.path.join(vis_dir, 'asset_token_selection.png')
    bbox_output_path = os.path.join(vis_dir, 'bbox_projection.png')

    # Skip if both already exist
    if os.path.exists(token_output_path) and os.path.exists(bbox_output_path):
        return

    # Create directory if needed
    os.makedirs(vis_dir, exist_ok=True)

    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    tokens_per_row = image_size // patch_size  # 512/16 = 32 for DINOv3

    n_assets = len(obbs)
    if n_assets == 0:
        return

    # ========== 1. Save asset_token_selection.png ==========
    if not os.path.exists(token_output_path):
        _save_token_selection_visualization(
            obbs=obbs,
            camera_center=camera_center,
            visibility_info=visibility_info,
            asset_names=asset_names,
            output_path=token_output_path,
            fov_degrees=fov_degrees,
            image_size=image_size,
            patch_size=patch_size,
            tokens_per_face=tokens_per_face,
        )

    # ========== 2. Save bbox_projection.png ==========
    if not os.path.exists(bbox_output_path):
        _save_bbox_projection_visualization(
            obbs=obbs,
            camera_center=camera_center,
            visibility_info=visibility_info,
            asset_names=asset_names,
            sample_dir=sample_dir,
            output_path=bbox_output_path,
            fov_degrees=fov_degrees,
            image_size=image_size,
            cubemap_folder=cubemap_folder,
            view_idx=view_idx,
        )


def _save_token_selection_visualization(
    obbs: torch.Tensor,
    camera_center: torch.Tensor,
    visibility_info: List[Dict],
    asset_names: Optional[List[str]],
    output_path: str,
    fov_degrees: float = 120.0,
    image_size: int = 512,
    patch_size: int = 16,
    tokens_per_face: int = 1029,
):
    """Save DINO token selection visualization."""
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    n_assets = len(obbs)

    # Create figure
    fig, axes = plt.subplots(n_assets, 6, figsize=(15, 2.5 * n_assets + 1))
    if n_assets == 1:
        axes = axes.reshape(1, -1)

    for asset_idx, info in enumerate(visibility_info):
        obb = obbs[info['index']]

        # Get asset name for labeling
        if asset_names is not None and info['index'] < len(asset_names):
            asset_name = _get_display_name(asset_names[info['index']])
        else:
            asset_name = f"Asset {info['index']}"

        # Compute token mask for each face
        for face_idx, face_name in enumerate(face_names):
            ax = axes[asset_idx, face_idx]

            # Create token mask
            token_mask = _compute_asset_token_mask_for_viz(
                obb=obb,
                camera_center=camera_center,
                face_idx=face_idx,
                tokens_per_face=tokens_per_face,
                fov_degrees=fov_degrees,
                image_size=image_size,
                patch_size=patch_size,
            )

            # Reshape to 2D grid
            # DINOv3: 1 CLS + 4 registers + (image_size/patch_size)^2 patches
            num_special_tokens = 5  # 1 CLS + 4 registers
            grid_size = image_size // patch_size  # 512/16 = 32
            token_grid = np.zeros((grid_size, grid_size))

            for patch_idx in range(grid_size * grid_size):
                token_idx = patch_idx + num_special_tokens
                if token_idx < tokens_per_face and token_mask[token_idx]:
                    row = patch_idx // grid_size
                    col = patch_idx % grid_size
                    token_grid[row, col] = 1.0

            # Color based on status
            if info['cam_inside']:
                cmap = 'Greys'
                alpha = 0.5
            elif info['included']:
                cmap = 'YlOrRd'
                alpha = 1.0
            else:
                cmap = 'Blues'
                alpha = 0.7

            ax.imshow(token_grid, cmap=cmap, vmin=0, vmax=1, alpha=alpha)

            if asset_idx == 0:
                ax.set_title(face_name.upper(), fontsize=10)

            if face_idx == 0:
                status = ""
                if info['cam_inside']:
                    status = "[cam inside]"
                elif info['included']:
                    status = f"✓ {info['max_visibility']*100:.0f}%"
                else:
                    status = f"✗ {info['max_visibility']*100:.0f}%"

                ax.set_ylabel(f"{asset_name}\n{status}", fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f'Asset Token Selection (DINO tokens per face)\n'
        f'Orange=included, Blue=excluded (low vis), Gray=camera inside',
        fontsize=12,
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[Visibility] Saved: {output_path}")
    except Exception as e:
        print(f"[Visibility] Failed to save: {e}")
    finally:
        plt.close(fig)


def _save_bbox_projection_visualization(
    obbs: torch.Tensor,
    camera_center: torch.Tensor,
    visibility_info: List[Dict],
    asset_names: Optional[List[str]],
    sample_dir: str,
    output_path: str,
    fov_degrees: float = 120.0,
    image_size: int = 512,
    cubemap_folder: str = 'cubic_fov_120',
    view_idx: int = 0,
):
    """Save bbox projection visualization on cubemap images."""
    from PIL import Image
    import matplotlib.patches as patches

    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']

    # Asset colors
    ASSET_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
        (0, 128, 255), (255, 0, 128),
    ]

    # Load cubemap images
    cubemap_dir = os.path.join(sample_dir, cubemap_folder, f'{view_idx:04d}')
    cubemap_images = {}
    for face in face_names:
        img_path = os.path.join(cubemap_dir, f'{face}.png')
        if os.path.exists(img_path):
            cubemap_images[face] = Image.open(img_path).convert('RGB')
        else:
            # Create blank image if not found
            cubemap_images[face] = Image.new('RGB', (image_size, image_size), (50, 50, 50))

    # Create figure with 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    face_axes = {
        'front': axes[0, 0], 'right': axes[0, 1], 'back': axes[0, 2],
        'left': axes[1, 0], 'top': axes[1, 1], 'bottom': axes[1, 2],
    }

    # Draw cubemap images
    for face, ax in face_axes.items():
        ax.imshow(cubemap_images[face])
        ax.set_title(f'{face.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, image_size)
        ax.set_ylim(image_size, 0)

    # Project and draw bboxes for each asset
    tan_half_fov = np.tan(np.radians(fov_degrees) / 2)

    for info in visibility_info:
        asset_idx = info['index']
        obb = obbs[asset_idx]
        color = ASSET_COLORS[asset_idx % len(ASSET_COLORS)]
        color_normalized = tuple(c / 255.0 for c in color)

        if asset_names is not None and asset_idx < len(asset_names):
            asset_name = _get_display_name(asset_names[asset_idx])
        else:
            asset_name = f"Asset {asset_idx}"

        # Get corners
        corners = obb_to_corners(obb)

        for face_idx, face_name in enumerate(face_names):
            face_dir = FACE_DIRECTIONS[face_idx].to(obb.device, dtype=obb.dtype)
            face_up = FACE_UP_VECTORS[face_idx].to(obb.device, dtype=obb.dtype)
            face_right = torch.linalg.cross(face_dir, face_up)
            face_right = face_right / torch.norm(face_right)
            face_up_actual = torch.linalg.cross(face_right, face_dir)

            # Project corners
            u_pixels = []
            v_pixels = []

            for corner in corners:
                direction = corner - camera_center
                forward_dist = torch.dot(direction, face_dir)
                if forward_dist <= 1e-6:
                    continue

                u_coord = torch.dot(direction, face_right) / forward_dist
                v_coord = torch.dot(direction, face_up_actual) / forward_dist

                u_normalized = u_coord / tan_half_fov
                v_normalized = v_coord / tan_half_fov

                u_pixel = ((u_normalized + 1) / 2 * image_size).item()
                v_pixel = ((1 - v_normalized) / 2 * image_size).item()

                u_pixels.append(u_pixel)
                v_pixels.append(v_pixel)

            if len(u_pixels) == 0:
                continue

            # Compute bbox
            u_min = max(0, min(u_pixels))
            u_max = min(image_size, max(u_pixels))
            v_min = max(0, min(v_pixels))
            v_max = min(image_size, max(v_pixels))

            width = u_max - u_min
            height = v_max - v_min

            if width > 0 and height > 0:
                ax = face_axes[face_name]

                # Line style based on inclusion
                if info['cam_inside']:
                    linewidth = 1
                    linestyle = ':'
                    alpha = 0.3
                elif info['included']:
                    linewidth = 3
                    linestyle = '-'
                    alpha = 0.7
                else:
                    linewidth = 1
                    linestyle = '--'
                    alpha = 0.4

                rect = patches.Rectangle(
                    (u_min, v_min), width, height,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    edgecolor=color_normalized,
                    facecolor=(*color_normalized, 0.1 * alpha),
                )
                ax.add_patch(rect)

                # Add label
                visibility_pct = info['max_visibility'] * 100
                if info['cam_inside']:
                    label = f"{asset_name} [cam]"
                elif info['included']:
                    label = f"{asset_name} ✓{visibility_pct:.0f}%"
                else:
                    label = f"{asset_name} ✗{visibility_pct:.0f}%"

                ax.text(
                    u_min + 2, v_min + 12,
                    label,
                    fontsize=7,
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color_normalized, alpha=0.7)
                )

    # Add legend
    legend_elements = [
        patches.Patch(
            facecolor=tuple(c/255 for c in ASSET_COLORS[i % len(ASSET_COLORS)]),
            edgecolor='black',
            label=f'{i}: {_get_display_name(asset_names[i]) if asset_names and i < len(asset_names) else f"Asset {i}"}'
        )
        for i in range(len(obbs))
    ]
    fig.legend(
        handles=legend_elements,
        loc='center right',
        bbox_to_anchor=(1.12, 0.5),
        fontsize=9,
    )

    cam_np = camera_center.cpu().numpy()
    fig.suptitle(
        f'3D Bbox → Cubemap Projection\n'
        f'Camera: ({cam_np[0]:.3f}, {cam_np[1]:.3f}, {cam_np[2]:.3f}), '
        f'FOV: {fov_degrees}°, {len(obbs)} assets\n'
        f'✓=included (vis≥50%), ✗=excluded, [cam]=camera inside',
        fontsize=12
    )

    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[Visibility] Saved: {output_path}")
    except Exception as e:
        print(f"[Visibility] Failed to save: {e}")
    finally:
        plt.close(fig)


def _get_display_name(name: str) -> str:
    """Get clean display name from asset filename."""
    # Remove _inst suffix
    if '_inst' in name:
        name = name.split('_inst')[0]
    else:
        name = name.rsplit('.', 1)[0]

    # Remove UID suffix (8-char hex)
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and len(parts[1]) == 8:
        try:
            int(parts[1], 16)
            name = parts[0]
        except ValueError:
            pass

    # Truncate if too long
    if len(name) > 25:
        name = name[:22] + "..."

    return name


def _compute_asset_token_mask_for_viz(
    obb: torch.Tensor,
    camera_center: torch.Tensor,
    face_idx: int,
    tokens_per_face: int = 1029,
    fov_degrees: float = 120.0,
    image_size: int = 512,
    patch_size: int = 16,
    expand_pixels: int = 28,
) -> np.ndarray:
    """
    Compute token mask for a single face (for visualization).

    Returns numpy array of shape [tokens_per_face] with True for selected tokens.
    """
    device = obb.device
    dtype = obb.dtype

    # Get bbox corners
    corners = obb_to_corners(obb)

    fov_rad = np.radians(fov_degrees)
    tan_half_fov = np.tan(fov_rad / 2)

    face_dirs = FACE_DIRECTIONS.to(device=device, dtype=dtype)
    face_ups = FACE_UP_VECTORS.to(device=device, dtype=dtype)

    face_dir = face_dirs[face_idx]
    face_up = face_ups[face_idx]

    # Face right vector
    face_right = torch.linalg.cross(face_dir, face_up)
    face_right = face_right / torch.norm(face_right)
    face_up_actual = torch.linalg.cross(face_right, face_dir)

    # Project corners
    u_pixels = []
    v_pixels = []

    for corner in corners:
        direction = corner - camera_center
        forward_dist = torch.dot(direction, face_dir)
        if forward_dist <= 1e-6:
            continue

        u_coord = torch.dot(direction, face_right) / forward_dist
        v_coord = torch.dot(direction, face_up_actual) / forward_dist

        u_normalized = u_coord / tan_half_fov
        v_normalized = v_coord / tan_half_fov

        u_pixel = ((u_normalized + 1) / 2 * image_size).item()
        v_pixel = ((1 - v_normalized) / 2 * image_size).item()

        u_pixels.append(u_pixel)
        v_pixels.append(v_pixel)

    token_mask = np.zeros(tokens_per_face, dtype=bool)

    if len(u_pixels) == 0:
        return token_mask

    # Compute bounding rectangle with expansion
    u_min = max(0, min(u_pixels) - expand_pixels)
    u_max = min(image_size, max(u_pixels) + expand_pixels)
    v_min = max(0, min(v_pixels) - expand_pixels)
    v_max = min(image_size, max(v_pixels) + expand_pixels)

    # Convert to token indices
    # DINOv3: 1 CLS + 4 registers + (image_size/patch_size)^2 patches
    num_special_tokens = 5  # 1 CLS + 4 registers
    tokens_per_row = image_size // patch_size  # 512/16 = 32
    scale = tokens_per_row / image_size

    t_u_min = max(0, int(np.floor(u_min * scale)))
    t_u_max = min(tokens_per_row, int(np.ceil(u_max * scale)))
    t_v_min = max(0, int(np.floor(v_min * scale)))
    t_v_max = min(tokens_per_row, int(np.ceil(v_max * scale)))

    # Set token mask
    for t_v in range(t_v_min, t_v_max):
        for t_u in range(t_u_min, t_u_max):
            token_idx = t_v * tokens_per_row + t_u + num_special_tokens
            if 0 <= token_idx < tokens_per_face:
                token_mask[token_idx] = True

    return token_mask