# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Visualization utilities for the InSpace Gradio demo.

Functions for creating:
- Plotly 3D scatter plots (PSG point cloud)
- GLB meshes for Gradio Model3D (voxels, bboxes, scene meshes)
- Rendered views for saving (exterior, interior, topdown)
"""

import os
import math
import tempfile
import colorsys
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Tuple

import plotly.graph_objects as go


# ============================================================
# Plotly: PSG Point Cloud
# ============================================================

def create_psg_plotly_figure(
    points: np.ndarray,
    colors: np.ndarray,
    camera_center: np.ndarray,
    room_bbox: Optional[np.ndarray] = None,
    show_camera_center: bool = True,
    show_coordinates: bool = True,
) -> go.Figure:
    """
    Create Plotly 3D scatter figure for PSG point cloud visualization.
    Style reference: WorldGen demo_3d_front_pointcloud_da2_260212.py

    Args:
        points: [N, 3] point positions in normalized [-0.5, 0.5] space
        colors: [N, 3] RGB colors (0-255)
        camera_center: [3] normalized camera center
        room_bbox: Optional [2, 3] min/max corners for room wireframe
        show_camera_center: Whether to show the camera center marker
        show_coordinates: Whether to show coordinate axes

    Returns:
        Plotly Figure for gr.Plot
    """
    traces = []

    # Point cloud
    traces.append(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=[f'rgb({r},{g},{b})' for r, g, b in colors.astype(int)],
            opacity=1.0,
            line=dict(width=0),
        ),
        name='PSG Point Cloud',
    ))

    # Camera center (red circle)
    if show_camera_center and camera_center is not None:
        cc = camera_center
        traces.append(go.Scatter3d(
            x=[cc[0]], y=[cc[1]], z=[cc[2]],
            mode='markers',
            marker=dict(size=6, color='red', symbol='circle'),
            name='Camera Center',
            hovertext=f"Camera: ({cc[0]:.3f}, {cc[1]:.3f}, {cc[2]:.3f})",
            hoverinfo='text',
        ))

    # Room bbox wireframe
    if room_bbox is not None:
        mn, mx = room_bbox[0], room_bbox[1]
        edges = _bbox_wireframe_edges(mn, mx)
        x_lines, y_lines, z_lines = [], [], []
        for edge in edges:
            x_lines += [edge[0, 0], edge[1, 0], None]
            y_lines += [edge[0, 1], edge[1, 1], None]
            z_lines += [edge[0, 2], edge[1, 2], None]
        traces.append(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color='red', width=3),
            name='Room BBox',
            hoverinfo='skip',
            opacity=0.7,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=show_coordinates, title='X'),
            yaxis=dict(visible=show_coordinates, title='Y'),
            zaxis=dict(visible=show_coordinates, title='Z'),
            bgcolor='white',
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        autosize=True,
        showlegend=True,
        legend=dict(x=0, y=1),
    )
    return fig


def _bbox_wireframe_edges(mn, mx):
    """Generate 12 edges of an axis-aligned bounding box."""
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ])
    edge_pairs = [
        (0,1),(1,2),(2,3),(3,0),  # bottom
        (4,5),(5,6),(6,7),(7,4),  # top
        (0,4),(1,5),(2,6),(3,7),  # vertical
    ]
    return [np.array([corners[i], corners[j]]) for i, j in edge_pairs]


# ============================================================
# GLB: CSG Voxel Grid
# ============================================================

def _voxel_to_cubes(
    voxel_64: np.ndarray,
    color_rgba: Optional[Tuple[int,int,int,int]] = None,
    max_cubes: int = 20000,
) -> List[trimesh.Trimesh]:
    """Convert voxel grid to a single concatenated mesh of colored cubes.

    Uses vectorized numpy operations instead of per-voxel trimesh.creation.box()
    loop, which is 100x+ faster for thousands of voxels.

    Args:
        voxel_64: [64, 64, 64] binary occupancy (squeezed)
        color_rgba: If None, use position-based RGB coloring. Otherwise, fixed color.
        max_cubes: Max voxel cubes to render (subsample if more)

    Returns:
        List with a single concatenated trimesh (or empty list)
    """
    active = np.argwhere(voxel_64 > 0)  # [N, 3]
    if len(active) == 0:
        return []

    # Subsample if too many
    if len(active) > max_cubes:
        indices = np.random.choice(len(active), max_cubes, replace=False)
        active = active[indices]

    N = len(active)
    centers = (active.astype(float) + 0.5) / 64.0 - 0.5  # [N, 3]
    half = 0.5 / 64.0  # half cube size

    # Unit cube: 8 vertices, 12 triangles
    unit_verts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64) * half  # [8, 3]

    unit_faces = np.array([
        [0,1,2], [0,2,3],  # bottom
        [4,6,5], [4,7,6],  # top
        [0,4,5], [0,5,1],  # front
        [2,6,7], [2,7,3],  # back
        [0,3,7], [0,7,4],  # left
        [1,5,6], [1,6,2],  # right
    ], dtype=np.int64)  # [12, 3]

    # Vectorized: replicate and translate all cubes at once
    # vertices: [N*8, 3]
    all_verts = np.tile(unit_verts, (N, 1))  # [N*8, 3]
    offsets = np.repeat(centers, 8, axis=0)  # [N*8, 3]
    all_verts += offsets

    # faces: [N*12, 3] with offset indices
    face_offsets = np.arange(N).reshape(-1, 1, 1) * 8  # [N, 1, 1]
    all_faces = np.tile(unit_faces, (N, 1, 1)) + face_offsets  # [N, 12, 3]
    all_faces = all_faces.reshape(-1, 3)  # [N*12, 3]

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)

    # Per-face colors: [N*12, 4]
    if color_rgba is not None:
        mesh.visual.face_colors = np.tile(
            np.array(color_rgba, dtype=np.uint8), (N * 12, 1)
        )
    else:
        # Position-based RGB per cube (all 12 faces of each cube get same color)
        rgb = ((centers + 0.5) * 255).clip(0, 255).astype(np.uint8)  # [N, 3]
        alpha = np.full((N, 1), 220, dtype=np.uint8)
        rgba = np.hstack([rgb, alpha])  # [N, 4]
        face_colors = np.repeat(rgba, 12, axis=0)  # [N*12, 4]
        mesh.visual.face_colors = face_colors

    return [mesh]


def _add_camera_center_sphere(meshes: list, camera_center: np.ndarray, color=(255, 0, 0, 255)):
    """Add a red sphere at camera center position."""
    if camera_center is not None:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.02)
        sphere.apply_translation(camera_center)
        sphere.visual.face_colors = color
        meshes.append(sphere)


def _scene_to_glb(meshes: list) -> Optional[str]:
    """Combine meshes into a Scene, apply Z-up→Y-up transform, export as GLB.

    Uses trimesh.Scene to preserve per-mesh PBR materials (unlike concatenation
    which merges everything into one mesh and loses individual materials).
    """
    if not meshes:
        return None
    # Apply Z-up → Y-up transform for glTF (Y-up, right-handed)
    # O-Voxel: X=right, Y=forward, Z=up  →  glTF: X=right, Y=up, Z=-forward
    # v @ [[1,0,0],[0,0,-1],[0,1,0]]  =>  x'=x, y'=z, z'=-y
    transform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        if mesh is None:
            continue
        m = mesh.copy()
        m.vertices = m.vertices @ transform
        # Also transform vertex normals if present (from to_glb() textured meshes)
        if hasattr(m, 'vertex_normals') and m.vertex_normals is not None and len(m.vertex_normals) > 0:
            try:
                m.vertex_normals = m.vertex_normals @ transform
            except Exception:
                pass
        scene.add_geometry(m, node_name=f'part_{i}')

    if len(scene.geometry) == 0:
        return None

    temp_path = tempfile.mktemp(suffix='.glb')
    scene.export(temp_path, file_type='glb')
    return temp_path


def create_voxel_glb(
    voxel_64: np.ndarray,
    camera_center: Optional[np.ndarray] = None,
    show_camera: bool = True,
    max_cubes: int = 8000,
) -> Optional[str]:
    """
    Create GLB from 64³ binary voxel grid for Model3D display.

    Args:
        voxel_64: [1, 64, 64, 64] or [64, 64, 64] binary occupancy
        camera_center: [3] normalized camera center
        show_camera: Whether to show camera center sphere
        max_cubes: Max voxel cubes to render (subsample if more)

    Returns:
        Path to temporary GLB file
    """
    if voxel_64.ndim == 4:
        voxel_64 = voxel_64[0]

    meshes = _voxel_to_cubes(voxel_64, max_cubes=max_cubes)
    if not meshes:
        return None

    if show_camera and camera_center is not None:
        _add_camera_center_sphere(meshes, camera_center)

    return _scene_to_glb(meshes)


def create_csg_comparison_glb(
    pred_voxel_64: np.ndarray,
    gt_voxel_64: Optional[np.ndarray],
    camera_center: Optional[np.ndarray] = None,
    show_camera: bool = True,
    max_cubes: int = 20000,
    color_mode: str = "ccm",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Create GLBs for predicted CSG and GT CSG side-by-side comparison.

    Args:
        color_mode: "ccm" for position-based RGB coloring, "gray" for uniform gray.

    Returns:
        (pred_glb_path, gt_glb_path)
    """
    color_rgba = [180, 180, 180, 220] if color_mode == "gray" else None

    pred_vox = pred_voxel_64[0] if pred_voxel_64.ndim == 4 else pred_voxel_64

    # Predicted CSG
    pred_meshes = _voxel_to_cubes(pred_vox, color_rgba=color_rgba, max_cubes=max_cubes)
    if show_camera and camera_center is not None:
        _add_camera_center_sphere(pred_meshes, camera_center)
    pred_glb = _scene_to_glb(pred_meshes)

    # GT CSG
    gt_glb = None
    if gt_voxel_64 is not None:
        gt_vox = gt_voxel_64[0] if gt_voxel_64.ndim == 4 else gt_voxel_64
        gt_meshes = _voxel_to_cubes(gt_vox, color_rgba=color_rgba, max_cubes=max_cubes)
        if show_camera and camera_center is not None:
            _add_camera_center_sphere(gt_meshes, camera_center)
        gt_glb = _scene_to_glb(gt_meshes)

    return pred_glb, gt_glb


# ============================================================
# GLB: BBox + CSG Overlay
# ============================================================

def create_bbox_with_voxel_glb(
    obbs: np.ndarray,
    voxel_64: np.ndarray,
    camera_center: np.ndarray,
    asset_names: Optional[List[str]] = None,
    max_voxel_cubes: int = 20000,
) -> Optional[str]:
    """
    Create GLB showing 3D bounding boxes overlaid on CSG voxel cubes.

    Args:
        obbs: [M, 7] oriented bounding boxes (cx,cy,cz,sx,sy,sz,yaw)
        voxel_64: [1, 64, 64, 64] or [64, 64, 64] binary voxel grid
        camera_center: [3] normalized camera center
        asset_names: optional list of asset names
        max_voxel_cubes: max voxel cubes to show

    Returns:
        Path to temporary GLB file
    """
    meshes = []

    # 1. Voxel cubes (gray, semi-transparent)
    if voxel_64 is not None:
        vox = voxel_64[0] if voxel_64.ndim == 4 else voxel_64
        meshes.extend(_voxel_to_cubes(vox, color_rgba=[180, 180, 180, 100], max_cubes=max_voxel_cubes))

    # 2. OBB semi-transparent boxes (colored)
    if obbs is not None and len(obbs) > 0:
        num_assets = len(obbs)
        for i, obb in enumerate(obbs):
            center = obb[0:3]
            size = obb[3:6]
            yaw = obb[6]

            hue = i / max(num_assets, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            color = [int(r*255), int(g*255), int(b*255), 150]

            box = trimesh.creation.box(extents=size)
            rotation = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
            box.apply_transform(rotation)
            box.apply_translation(center)
            box.visual.face_colors = color
            meshes.append(box)

    # 3. Camera center sphere (red)
    _add_camera_center_sphere(meshes, camera_center)

    return _scene_to_glb(meshes)


# ============================================================
# GLB: Scene Mesh (Combined + Exploded)
# ============================================================

def create_scene_glb(
    meshes_list: List[trimesh.Trimesh],
    asset_names: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Create combined scene GLB using ONLY the overall mesh (index 0).

    The overall mesh already contains all voxels. Including asset parts would
    create duplicate geometry (overall + assets overlap).

    Args:
        meshes_list: List of trimesh.Trimesh, index 0 = overall scene
        asset_names: Optional list of names

    Returns:
        Path to temporary GLB file
    """
    if not meshes_list:
        return None

    # Use only the overall mesh (index 0) — it already has all voxels.
    # The materials are set in decode_meshes_single (gray PBR or textured GLB).
    overall = meshes_list[0]
    if overall is None:
        return None

    return _scene_to_glb([overall])


def create_layout_glb(
    meshes_list: List[trimesh.Trimesh],
    has_layout: bool = False,
) -> Optional[str]:
    """
    Create GLB showing only the layout mesh (floor + walls).

    Args:
        meshes_list: List of trimesh.Trimesh. Index 1 = layout (if has_layout).
        has_layout: Whether layout is at index 1.

    Returns:
        Path to temporary GLB file, or None.
    """
    if not meshes_list or not has_layout or len(meshes_list) < 2:
        return None

    layout_mesh = meshes_list[1]
    if layout_mesh is None:
        return None

    return _scene_to_glb([layout_mesh])


def create_exploded_glb(
    meshes_list: List[trimesh.Trimesh],
    asset_names: Optional[List[str]] = None,
    explosion_scale: float = 0.3,
    has_layout: bool = False,
) -> Optional[str]:
    """
    Create exploded view GLB showing only individual asset parts.

    Skips overall (index 0) and layout (index 1 if has_layout).
    Each asset is translated away from scene center.

    Args:
        meshes_list: List of trimesh.Trimesh.
        asset_names: Optional names
        explosion_scale: How far to push parts apart
        has_layout: Whether layout is at index 1

    Returns:
        Path to temporary GLB file
    """
    if not meshes_list:
        return None

    # Determine asset start index (skip overall and layout)
    asset_start = 2 if has_layout else 1
    asset_meshes = meshes_list[asset_start:]

    if not asset_meshes or all(m is None for m in asset_meshes):
        # Fall back to showing overall
        return create_scene_glb(meshes_list, asset_names)

    def _center(m):
        # Robust centroid via vertex mean (works for any mesh with .vertices,
        # unlike Trimesh-only .centroid).
        return np.asarray(m.vertices, dtype=np.float64).mean(axis=0)

    # Compute scene center from overall mesh
    overall = meshes_list[0]
    try:
        scene_center = _center(overall) if overall is not None else np.zeros(3)
    except Exception:
        scene_center = np.zeros(3)

    exploded = []
    for mesh in asset_meshes:
        if mesh is None:
            continue
        m = mesh.copy()
        try:
            direction = _center(m) - scene_center
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
            # translate by assigning vertices (avoids Trimesh-only apply_translation)
            m.vertices = np.asarray(m.vertices, dtype=np.float64) + direction * explosion_scale
        except Exception:
            pass
        exploded.append(m)

    if not exploded:
        return create_scene_glb(meshes_list, asset_names)

    return _scene_to_glb(exploded)


# ============================================================
# Cubemap Grid Image
# ============================================================

def create_cubemap_grid_image(cubemap_dir: str, size: int = 256) -> Optional[Image.Image]:
    """
    Create 2x3 grid image from 6 cubemap face images.

    Layout:
        front  right  back
        left   top    bottom
    """
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    faces = []
    for name in face_names:
        path = os.path.join(cubemap_dir, f'{name}.png')
        if not os.path.exists(path):
            return None
        img = Image.open(path).convert('RGB').resize((size, size))
        faces.append(img)

    grid = Image.new('RGB', (size * 3, size * 2))
    for i, face in enumerate(faces):
        col = i % 3
        row = i // 3
        grid.paste(face, (col * size, row * size))

    return grid


# ============================================================
# Rendering Helpers
# ============================================================

def _trimesh_to_trellis_mesh(tm_mesh):
    """Convert a trimesh.Trimesh to TRELLIS Mesh for rendering."""
    import torch
    from trellis2.representations.mesh.base import Mesh as TrellisMesh
    v = torch.from_numpy(np.asarray(tm_mesh.vertices)).float().cuda()
    f = torch.from_numpy(np.asarray(tm_mesh.faces)).int().cuda()
    return TrellisMesh(v, f)


def _ccm_colormap(coords):
    """Map voxel (x, y, z) to (R, G, B), normalized to occupied range per axis."""
    import torch
    c = coords.float()
    c_min = c.min(dim=0).values
    c_max = c.max(dim=0).values
    span = (c_max - c_min).clamp(min=1.0)
    return (c - c_min) / span


def _make_label_strip(labels, tile_size, label_height=24):
    """Create a white strip with centered text labels."""
    total_w = len(labels) * tile_size
    strip = Image.new('RGB', (total_w, label_height), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", label_height - 8)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        x = i * tile_size + (tile_size - tw) // 2
        draw.text((x, 2), label, fill=(0, 0, 0), font=font)
    return strip


def _make_row_labels(labels, row_height, label_width=80):
    """Create a vertical strip with row labels."""
    total_h = len(labels) * row_height
    strip = Image.new('RGB', (label_width, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        tw = bbox[2] - bbox[0]
        x = (label_width - tw) // 2
        y = i * row_height + (row_height - th) // 2
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
    return strip


def _tensor_to_pil(t):
    """Convert [3, H, W] float tensor [0,1] to PIL Image."""
    img_np = (t.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def _draw_camera_marker_on_topdown(face_tensor, camera_center, tile):
    """Draw cyan camera center marker on topdown image tensor [3,H,W]. Returns modified tensor."""
    import torch
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

    exts, ints_mat = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [np.pi / 2], 2, 30)
    cam_np = camera_center.numpy() if hasattr(camera_center, 'numpy') else np.array(camera_center)
    cam_3d = torch.tensor(cam_np, dtype=torch.float32).cuda()
    point_h = torch.cat([cam_3d, torch.ones(1, device='cuda')])
    point_cam = exts[0] @ point_h
    point_proj = ints_mat[0] @ point_cam[:3]

    if point_proj[2].abs() > 1e-6:
        px = (point_proj[0] / point_proj[2]).item() * tile
        py = (point_proj[1] / point_proj[2]).item() * tile
        if -20 < px < tile + 20 and -20 < py < tile + 20:
            pil_img = _tensor_to_pil(face_tensor)
            draw = ImageDraw.Draw(pil_img)
            r = 8
            draw.ellipse([px-r, py-r, px+r, py+r],
                         fill=(0, 255, 255), outline=(255, 255, 255), width=2)
            face_tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
    return face_tensor


def _compose_interior_grid(rendered_faces, cubemap_images, face_labels, tile):
    """Compose interior view grid: face labels + cubemap row + rendered row with row labels.

    Args:
        rendered_faces: list of 6 tensors [3, tile, tile]
        cubemap_images: [6, 3, H, W] tensor
        face_labels: list of 6 face name strings
        tile: per-face resolution

    Returns:
        PIL Image composite
    """
    import torch
    import torch.nn.functional as F

    label_h = max(20, tile // 10)
    label_strip = _make_label_strip(face_labels, tile, label_h)
    label_tensor = torch.tensor(np.array(label_strip)).permute(2, 0, 1).float() / 255.0

    cubemap_resized = F.interpolate(
        cubemap_images, size=(tile, tile), mode='bilinear', align_corners=False,
    )

    row1 = torch.cat([cubemap_resized[j] for j in range(6)], dim=2)
    row2 = torch.cat([rendered_faces[j] for j in range(6)], dim=2)

    row_label_w = 80
    row_labels = _make_row_labels(['Input', 'Predicted'], tile, row_label_w)
    row_labels_t = torch.tensor(np.array(row_labels)).permute(2, 0, 1).float() / 255.0
    label_pad = torch.ones(3, label_h, row_label_w)
    label_full = torch.cat([label_pad, label_tensor], dim=2)
    rows = torch.cat([row1, row2], dim=1)
    rows_full = torch.cat([row_labels_t, rows], dim=2)
    composite = torch.cat([label_full, rows_full], dim=1)

    return _tensor_to_pil(composite)


def _get_envmap():
    """Load interior HDRI environment map for PBR rendering."""
    import torch
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    import cv2
    from trellis2.renderers import EnvMap
    hdri_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'assets', 'hdri', 'interior.exr')
    img = cv2.imread(hdri_path, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return EnvMap(torch.tensor(img_rgb, dtype=torch.float32, device='cuda'))


def reconstruct_trellis_rep(rep_data):
    """Reconstruct TRELLIS representation from stored CPU numpy data.

    Args:
        rep_data: dict with 'is_textured', 'vertices', 'faces', and optionally
                  'vox_coords', 'vox_attrs', 'vox_shape', 'resolution', 'layout'

    Returns:
        TRELLIS Mesh or MeshWithVoxel on CUDA
    """
    import torch
    v = torch.from_numpy(rep_data['vertices']).float().cuda()
    f = torch.from_numpy(rep_data['faces']).int().cuda()

    if rep_data.get('is_textured'):
        from trellis2.representations import MeshWithVoxel
        coords = torch.from_numpy(rep_data['vox_coords']).int().cuda()
        attrs = torch.from_numpy(rep_data['vox_attrs']).float().cuda()
        layout = {k: slice(s, e) for k, (s, e) in rep_data['layout'].items()}
        return MeshWithVoxel(
            v, f,
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1.0 / rep_data['resolution'],
            coords=coords, attrs=attrs,
            voxel_shape=torch.Size(rep_data['vox_shape']),
            layout=layout,
        )
    else:
        from trellis2.representations.mesh.base import Mesh as TrellisMesh
        return TrellisMesh(v, f)


# ============================================================
# SS Voxel Rendering (VoxelRenderer)
# ============================================================

def render_ss_exterior(voxel_64, tile=512):
    """Render 4 exterior SS voxel views in 2x2 grid. Returns PIL Image."""
    import torch
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    from trellis2.renderers import VoxelRenderer
    from trellis2.representations import Voxel

    grid = torch.tensor(voxel_64).squeeze() if not isinstance(voxel_64, torch.Tensor) \
        else voxel_64.squeeze()
    coords = torch.nonzero(grid > 0, as_tuple=False)
    if coords.shape[0] == 0:
        return None

    color = _ccm_colormap(coords)
    resolution = grid.shape[0]
    rep = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / resolution,
        coords=coords.cuda(),
        attrs=color.cuda(),
        layout={'color': slice(0, 3)},
    )

    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = tile
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.ssaa = 4

    yaw_offset = -16 / 180 * np.pi
    yaws = [i * np.pi / 2 + yaw_offset for i in range(4)]
    pitch = [60 / 180 * np.pi] * 4
    exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 30)

    image = torch.zeros(3, 2 * tile, 2 * tile).cuda()
    for j, (ext, intr) in enumerate(zip(exts, ints)):
        res = renderer.render(rep, ext, intr, colors_overwrite=color.cuda())
        r, c = j // 2, j % 2
        image[:, r * tile:(r+1) * tile, c * tile:(c+1) * tile] = res['color']

    return _tensor_to_pil(image.cpu())


def render_ss_topdown_cam(voxel_64, camera_center, tile=512):
    """Render top-down SS voxel view with camera center marker. Returns PIL Image."""
    import torch
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    from trellis2.renderers import VoxelRenderer
    from trellis2.representations import Voxel

    grid = torch.tensor(voxel_64).squeeze() if not isinstance(voxel_64, torch.Tensor) \
        else voxel_64.squeeze()
    coords = torch.nonzero(grid > 0, as_tuple=False)
    if coords.shape[0] == 0:
        return None

    color = _ccm_colormap(coords)
    resolution = grid.shape[0]
    rep = Voxel(
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1.0 / resolution,
        coords=coords.cuda(),
        attrs=color.cuda(),
        layout={'color': slice(0, 3)},
    )

    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = tile
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.ssaa = 4

    exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [np.pi / 2], 2, 30)
    res = renderer.render(rep, exts[0], ints[0], colors_overwrite=color.cuda())
    face = res['color'].cpu()

    if camera_center is not None:
        face = _draw_camera_marker_on_topdown(face, camera_center, tile)
    return _tensor_to_pil(face)


def render_ss_interior(voxel_64, camera_center, cubemap_images=None, tile=512):
    """Render 6 interior SS voxel views from camera center. Returns PIL Image composite."""
    import torch
    import utils3d.torch
    from trellis2.renderers import VoxelRenderer
    from trellis2.representations import Voxel

    grid = torch.tensor(voxel_64).squeeze() if not isinstance(voxel_64, torch.Tensor) \
        else voxel_64.squeeze()
    coords = torch.nonzero(grid > 0, as_tuple=False)
    if coords.shape[0] == 0:
        return None

    color = _ccm_colormap(coords)
    resolution = grid.shape[0]
    world_scale = 10.0

    rep = Voxel(
        origin=[-0.5 * world_scale, -0.5 * world_scale, -0.5 * world_scale],
        voxel_size=world_scale / resolution,
        coords=coords.cuda(),
        attrs=color.cuda(),
        layout={'color': slice(0, 3)},
    )

    renderer = VoxelRenderer()
    renderer.rendering_options.resolution = tile
    renderer.rendering_options.near = 0.01 * world_scale
    renderer.rendering_options.far = 2.0 * world_scale
    renderer.rendering_options.ssaa = 4

    cam_np = camera_center.numpy() if hasattr(camera_center, 'numpy') else np.array(camera_center)
    cam = torch.tensor(cam_np, dtype=torch.float32).cuda() * world_scale
    fov = torch.deg2rad(torch.tensor(120.0)).cuda()

    face_dirs = [
        [0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
    ]
    face_labels = ['front (+Y)', 'right (+X)', 'back (-Y)', 'left (-X)', 'top (+Z)', 'bottom (-Z)']

    rendered_faces = []
    for fi, fd in enumerate(face_dirs):
        look_at = cam + torch.tensor(fd, dtype=torch.float32).cuda()
        if fi == 4:
            up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32).cuda()
        elif fi == 5:
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()
        else:
            up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda()
        ext = utils3d.torch.extrinsics_look_at(cam, look_at, up)
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        res = renderer.render(rep, ext, intr, colors_overwrite=color.cuda())
        rendered_faces.append(res['color'].cpu())

    if cubemap_images is not None:
        return _compose_interior_grid(rendered_faces, cubemap_images, face_labels, tile)
    else:
        grid_img = Image.new('RGB', (tile * 6, tile))
        for i, face in enumerate(rendered_faces):
            grid_img.paste(_tensor_to_pil(face), (i * tile, 0))
        return grid_img


# ============================================================
# Mesh Rendering (Geometry: Normal Map / Texture: PBR)
# ============================================================

def render_mesh_exterior(rep, tile=512, use_pbr=True):
    """Render 4 exterior mesh views in 2x2 grid. Returns PIL Image.

    Args:
        rep: TRELLIS Mesh (normal maps) or MeshWithVoxel (PBR)
        tile: per-view resolution
        use_pbr: if True and rep is MeshWithVoxel, use PbrMeshRenderer
    """
    import torch
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    from trellis2.representations import MeshWithVoxel

    yaw_offset = -16 / 180 * np.pi
    yaws = [i * np.pi / 2 + yaw_offset for i in range(4)]
    pitch = [60 / 180 * np.pi] * 4
    exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, 2, 30)

    if use_pbr and isinstance(rep, MeshWithVoxel):
        from trellis2.renderers import PbrMeshRenderer
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8
        envmap = _get_envmap()
        render_key = 'base_color'
    else:
        from trellis2.renderers.mesh_renderer import MeshRenderer
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        envmap = None
        render_key = 'normal'

    image = torch.zeros(3, 2 * tile, 2 * tile).cuda()
    for j, (ext, intr) in enumerate(zip(exts, ints)):
        if envmap is not None:
            res = renderer.render(rep, ext, intr, envmap=envmap)
        else:
            res = renderer.render(rep, ext, intr)
        r, c = j // 2, j % 2
        image[:, r * tile:(r+1) * tile, c * tile:(c+1) * tile] = res[render_key]

    return _tensor_to_pil(image.cpu())


def render_mesh_topdown_cam(rep, camera_center, tile=512, use_pbr=True):
    """Render top-down mesh view with camera center marker. Returns PIL Image."""
    import torch
    from trellis2.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics
    from trellis2.representations import MeshWithVoxel

    exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [np.pi / 2], 2, 30)

    if use_pbr and isinstance(rep, MeshWithVoxel):
        from trellis2.renderers import PbrMeshRenderer
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8
        envmap = _get_envmap()
        res = renderer.render(rep, exts[0], ints[0], envmap=envmap)
        face = res['base_color'].cpu()
    else:
        from trellis2.renderers.mesh_renderer import MeshRenderer
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        res = renderer.render(rep, exts[0], ints[0])
        face = res['normal'].cpu()

    if camera_center is not None:
        face = _draw_camera_marker_on_topdown(face, camera_center, tile)
    return _tensor_to_pil(face)


def render_mesh_interior(rep, camera_center, cubemap_images=None, tile=512, use_pbr=True):
    """Render 6 interior mesh views from camera center. Returns PIL Image composite."""
    import torch
    import utils3d.torch
    from trellis2.representations import MeshWithVoxel

    if use_pbr and isinstance(rep, MeshWithVoxel):
        from trellis2.renderers import PbrMeshRenderer
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.ssaa = 4
        renderer.rendering_options.near = 0.01
        renderer.rendering_options.far = 2.0
        renderer.rendering_options.peel_layers = 8
        envmap = _get_envmap()
        render_key = 'base_color'
    else:
        from trellis2.renderers.mesh_renderer import MeshRenderer
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = tile
        renderer.rendering_options.ssaa = 4
        renderer.rendering_options.near = 0.01
        renderer.rendering_options.far = 2.0
        envmap = None
        render_key = 'normal'

    cam_np = camera_center.numpy() if hasattr(camera_center, 'numpy') else np.array(camera_center)
    cam = torch.tensor(cam_np, dtype=torch.float32).cuda()
    fov = torch.deg2rad(torch.tensor(120.0)).cuda()

    face_dirs = [
        [0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
    ]
    face_labels = ['front (+Y)', 'right (+X)', 'back (-Y)', 'left (-X)', 'top (+Z)', 'bottom (-Z)']

    rendered_faces = []
    for fi, fd in enumerate(face_dirs):
        look_at = cam + torch.tensor(fd, dtype=torch.float32).cuda()
        if fi == 4:
            up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32).cuda()
        elif fi == 5:
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).cuda()
        else:
            up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32).cuda()
        ext = utils3d.torch.extrinsics_look_at(cam, look_at, up)
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        if envmap is not None:
            res = renderer.render(rep, ext, intr, envmap=envmap)
        else:
            res = renderer.render(rep, ext, intr)
        rendered_faces.append(res[render_key].cpu())

    if cubemap_images is not None:
        return _compose_interior_grid(rendered_faces, cubemap_images, face_labels, tile)
    else:
        grid_img = Image.new('RGB', (tile * 6, tile))
        for i, face in enumerate(rendered_faces):
            grid_img.paste(_tensor_to_pil(face), (i * tile, 0))
        return grid_img


# ============================================================
# BBox Top-Down (matplotlib)
# ============================================================

def render_bbox_topdown(obbs, asset_names=None, camera_center=None, bbox_source='gt'):
    """Render top-down BBox visualization using matplotlib. Returns PIL Image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)

    color = 'green' if bbox_source == 'gt' else 'red'
    if obbs is not None and len(obbs) > 0:
        for i, obb in enumerate(obbs):
            cx, cy = obb[0], obb[1]
            sx, sy = obb[3], obb[4]
            yaw = obb[6]
            cos_a = math.cos(yaw)
            sin_a = math.sin(yaw)
            hw, hh = sx / 2, sy / 2
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            world_corners = [
                [cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a]
                for lx, ly in corners
            ]
            polygon = plt.Polygon(world_corners, closed=True,
                                  facecolor=color, edgecolor=color,
                                  alpha=0.4, linewidth=1)
            ax.add_patch(polygon)
            name = asset_names[i][:15] if asset_names and i < len(asset_names) else ''
            if name:
                ax.text(cx, cy, name, ha='center', va='center', fontsize=5, color='white')

    if camera_center is not None:
        cc = np.array(camera_center)
        ax.plot(cc[0], cc[1], 'o', color='cyan', markersize=8,
                markeredgecolor='blue', markeredgewidth=2)

    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    n_obbs = len(obbs) if obbs is not None else 0
    ax.set_title(f'{bbox_source.upper()} BBox: {n_obbs} objects')

    plt.tight_layout()
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return img


# ============================================================
# Cubemap Input (matplotlib)
# ============================================================

def render_cubemap_input(cubemap_dir, scene_label=''):
    """Render 2x3 cubemap input grid using matplotlib. Returns PIL Image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=80)

    for i, name in enumerate(face_names):
        ax = axes[i // 3, i % 3]
        path = os.path.join(cubemap_dir, f'{name}.png')
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            ax.imshow(np.array(img))
        ax.set_title(name)
        ax.axis('off')

    if scene_label:
        plt.suptitle(f'Input Cubemap: {scene_label}', fontsize=10)
    plt.tight_layout()
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    return img
