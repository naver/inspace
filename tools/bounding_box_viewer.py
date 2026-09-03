# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Gradio viewer for 3D-FRONT floorplan bounding boxes.

Displays:
- 2D floorplan images (PNG)
- 3D bounding boxes from NPY files

Usage:
    python bbox_viewer_gradio.py --port 7860

from ubuntu_desktop:
"""

import os
import glob
import tempfile
import colorsys
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import socket

import gradio as gr
import numpy as np
from PIL import Image

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("trimesh not installed")


# Base directory for 3D-FRONT floorplan data
# BASE_DIR = "datasets/ERP_3D_FRONT"
BASE_DIR = "datasets/ERP_3D_FRONT_test"
# BASE_DIR = "figure_sample"



def get_scene_folders(base_dir: str) -> List[str]:
    """Get list of house IDs (scene folders) from ERP_3D_FRONT dataset."""
    base = Path(base_dir)
    if not base.exists():
        return []

    # ERP_3D_FRONT structure: base_dir/{house_id}/{room_name}/
    houses = []
    for folder in sorted(base.iterdir()):
        if folder.is_dir():
            # Check if it has room subdirectories with 3d_bounding_box data
            has_rooms = False
            for room_folder in folder.iterdir():
                if room_folder.is_dir():
                    bbox_dir = room_folder / '3d_bounding_box'
                    if bbox_dir.exists() and list(bbox_dir.glob('*.npz')):
                        has_rooms = True
                        break
            if has_rooms:
                houses.append(folder.name)

    return houses


def get_room_samples(base_dir: str, house_id: str) -> List[str]:
    """Get list of room names for a house in ERP_3D_FRONT dataset."""
    house_dir = Path(base_dir) / house_id
    if not house_dir.exists():
        return []

    # ERP_3D_FRONT structure: {house_id}/{room_name}/3d_bounding_box/*.npz
    rooms = []
    for room_folder in sorted(house_dir.iterdir()):
        if room_folder.is_dir():
            bbox_dir = room_folder / '3d_bounding_box'
            if bbox_dir.exists() and list(bbox_dir.glob('*.npz')):
                rooms.append(room_folder.name)

    return rooms


def load_floorplan_image(base_dir: str, house_id: str, room_name: str) -> Optional[Image.Image]:
    """Load representative image from ERP_3D_FRONT dataset."""
    room_dir = Path(base_dir) / house_id / room_name

    # Priority 1: Camera trajectory (top-down view)
    trajectory_img = room_dir / "trajectory" / "camera_trajectory.png"
    if trajectory_img.exists():
        return Image.open(trajectory_img)

    # Priority 2: ERP panorama (first view)
    erp_dir = room_dir / "erp"
    if erp_dir.exists():
        erp_images = sorted(erp_dir.glob("*_colors.png"))
        if erp_images:
            return Image.open(erp_images[0])

    # Priority 3: Cubic face (front view of first camera)
    cubic_dir = room_dir / "cubic_fov_120"
    if cubic_dir.exists():
        view_dirs = sorted([d for d in cubic_dir.iterdir() if d.is_dir()])
        if view_dirs:
            front_img = view_dirs[0] / "front.png"
            if front_img.exists():
                return Image.open(front_img)

    return None


def load_bbox(base_dir: str, house_id: str, room_name: str, suffix: str = "") -> dict:
    """Load bbox from ERP_3D_FRONT dataset NPZ file.

    ERP_3D_FRONT format: {house_id}/{room_name}/3d_bounding_box/{room_name}_scene_data.npz

    Args:
        base_dir: Base directory path
        house_id: House ID
        room_name: Room name
        suffix: Layout variant suffix (controls which components to include)

    Returns:
        dict with 'obbs', 'wall_obbs', 'floor_polygon', 'ceiling_polygon', etc.
    """
    result = {
        'bboxes': None, 'obbs': None, 'wall_obbs': None,
        'floor_polygon': None, 'ceiling_polygon': None
    }

    # ERP_3D_FRONT structure: {house_id}/{room_name}/3d_bounding_box/*.npz
    bbox_dir = Path(base_dir) / house_id / room_name / '3d_bounding_box'
    if not bbox_dir.exists():
        return result

    # Find NPZ file
    npz_files = list(bbox_dir.glob('*.npz'))
    if not npz_files:
        return result

    # Load the first NPZ file found
    npz_path = npz_files[0]
    data = np.load(npz_path, allow_pickle=True)

    # Asset OBBs (always included)
    if 'obbs' in data:
        result['obbs'] = data['obbs']

    # Wall OBBs - include if suffix contains 'wall' or 'ceiling'
    if 'wall_obbs' in data and ('wall' in suffix or 'ceiling' in suffix):
        result['wall_obbs'] = data['wall_obbs']

    # Floor polygon - include if suffix contains 'floor', 'wall', or 'ceiling'
    if 'floor_polygon' in data and ('floor' in suffix or 'wall' in suffix or 'ceiling' in suffix):
        result['floor_polygon'] = data['floor_polygon']
        result['floor_height'] = float(data['floor_height']) if 'floor_height' in data else 0.05
        result['floor_z'] = float(data['floor_z']) if 'floor_z' in data else 0.0

    # Ceiling polygon - include only if suffix contains 'ceiling'
    if 'ceiling_polygon' in data and 'ceiling' in suffix:
        result['ceiling_polygon'] = data['ceiling_polygon']
        result['ceiling_z'] = float(data['ceiling_z']) if 'ceiling_z' in data else 0.0
        result['ceiling_height'] = float(data['ceiling_height']) if 'ceiling_height' in data else 0.02

    # Normalization params
    if 'norm_center' in data:
        result['norm_center'] = data['norm_center']
    if 'norm_scale' in data:
        result['norm_scale'] = float(data['norm_scale'])

    # Load camera centers from camera_poses.json
    import json
    cam_poses_path = Path(base_dir) / house_id / room_name / 'camera_poses.json'
    if cam_poses_path.exists():
        with open(cam_poses_path) as f:
            cam_data = json.load(f)
        if 'views' in cam_data:
            world_centers = [v['location'] for v in cam_data['views']]
            # Normalize to O-Voxel space: (world - center) * scale
            norm_center = result.get('norm_center')
            norm_scale = result.get('norm_scale')
            if norm_center is not None and norm_scale is not None:
                centers = [(np.array(c) - norm_center) / norm_scale for c in world_centers]
                result['camera_centers'] = np.array(centers)

    return result


def create_bbox_mesh(bbox_data: dict, highlight_floor: bool = False, use_obb: bool = True,
                     show_camera_center: bool = False) -> Optional[str]:
    """Create GLB mesh visualization of bounding boxes.

    Args:
        bbox_data: dict with 'obbs', 'wall_obbs', 'floor_polygon', 'ceiling_polygon', etc.
        highlight_floor: Whether to highlight the floor/wall/ceiling
        use_obb: If True, use OBB (with rotation), else use AABB
        show_camera_center: If True, add red spheres at camera center positions
    """
    if not HAS_TRIMESH:
        return None

    obbs = bbox_data.get('obbs')
    wall_obbs = bbox_data.get('wall_obbs')
    bboxes = bbox_data.get('bboxes')
    floor_polygon = bbox_data.get('floor_polygon')
    floor_height = bbox_data.get('floor_height', 0.05)
    floor_z = bbox_data.get('floor_z', 0.0)
    ceiling_polygon = bbox_data.get('ceiling_polygon')
    ceiling_z = bbox_data.get('ceiling_z', 0.0)
    ceiling_height = bbox_data.get('ceiling_height', 0.02)
    camera_centers = bbox_data.get('camera_centers') if show_camera_center else None

    # Use the OBB when present, otherwise fall back to the AABB
    if use_obb and obbs is not None and len(obbs) > 0:
        return create_obb_mesh(
            obbs, wall_obbs, highlight_floor,
            floor_polygon, floor_height, floor_z,
            ceiling_polygon, ceiling_z, ceiling_height,
            camera_centers=camera_centers
        )
    elif bboxes is not None and len(bboxes) > 0:
        return create_aabb_mesh(bboxes, highlight_floor)
    else:
        return None


def _ear_clip_triangulate(polygon_2d: np.ndarray) -> List[List[int]]:
    """Triangulate a simple polygon using ear clipping.

    Works correctly for concave polygons unlike Delaunay + centroid filtering.

    Args:
        polygon_2d: (N, 2) polygon vertices in order.

    Returns:
        List of [i, j, k] index triples into polygon_2d.
    """
    n = len(polygon_2d)
    if n < 3:
        return []

    pts = polygon_2d

    # Compute signed area to determine winding order
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]

    # Build index list in CCW order
    indices = list(range(n))
    if area < 0:
        indices.reverse()

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _point_in_triangle(p, a, b, c):
        d1 = _cross(a, b, p)
        d2 = _cross(b, c, p)
        d3 = _cross(c, a, p)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    triangles = []
    while len(indices) > 2:
        ear_found = False
        m = len(indices)
        for i in range(m):
            prev_i = (i - 1) % m
            next_i = (i + 1) % m
            a = pts[indices[prev_i]]
            b = pts[indices[i]]
            c = pts[indices[next_i]]

            # Check convexity (positive cross product for CCW)
            if _cross(a, b, c) <= 0:
                continue

            # Check no other vertex inside this triangle
            is_ear = True
            for j in range(m):
                if j == prev_i or j == i or j == next_i:
                    continue
                if _point_in_triangle(pts[indices[j]], a, b, c):
                    is_ear = False
                    break

            if is_ear:
                triangles.append([indices[prev_i], indices[i], indices[next_i]])
                indices.pop(i)
                ear_found = True
                break

        if not ear_found:
            break

    return triangles


def create_polygon_extrusion(polygon_2d: np.ndarray, min_z: float, max_z: float, color) -> Optional[trimesh.Trimesh]:
    """Create extruded mesh from 2D polygon using ear clipping triangulation.

    Handles concave polygons correctly (L-shapes, corridors, etc.).
    """
    if polygon_2d is None or len(polygon_2d) < 3:
        return None

    height = max_z - min_z
    if height <= 0:
        return None

    try:
        n = len(polygon_2d)

        # Vertices: bottom ring + top ring
        bottom_verts = np.column_stack([polygon_2d, np.full(n, min_z)])
        top_verts = np.column_stack([polygon_2d, np.full(n, max_z)])
        vertices = np.vstack([bottom_verts, top_verts])  # [2n, 3]

        faces = []
        # Side faces (quads split into 2 triangles each)
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, j + n])
            faces.append([i, j + n, i + n])

        # Top/bottom faces using ear clipping (correct for concave polygons)
        tri_indices = _ear_clip_triangulate(polygon_2d)
        for tri in tri_indices:
            faces.append(tri)                                       # bottom face
            faces.append([tri[2] + n, tri[1] + n, tri[0] + n])     # top face (flipped normal)

        faces = np.array(faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.visual.face_colors = color
        return mesh

    except Exception as e:
        print(f"Polygon extrusion failed: {e}")
        return None


def create_obb_mesh(obbs: np.ndarray, wall_obbs: np.ndarray = None, highlight_layout: bool = False,
                    floor_polygon=None, floor_height=0.05, floor_z=0.0,
                    ceiling_polygon=None, ceiling_z=0.0, ceiling_height=0.02,
                    camera_centers=None) -> Optional[str]:
    """Create GLB mesh from OBB data with rotation applied.

    Args:
        obbs: (N, 7) Asset OBBs [center_x, center_y, center_z, size_x, size_y, size_z, yaw]
        wall_obbs: (M, 7) Wall OBBs (optional)
        highlight_layout: Whether floor/wall/ceiling are included
        floor_polygon: (K, 2) Floor boundary vertices
        floor_z: Starting Z position of floor (after normalization)
        floor_height: Thickness of floor
        ceiling_polygon: (K, 2) Ceiling boundary vertices
        ceiling_z: Starting Z position of ceiling (after normalization)
        ceiling_height: Thickness of ceiling
        camera_centers: (V, 3) Camera center positions in normalized coords (optional)
    """
    if obbs is None or len(obbs) == 0:
        return None

    meshes = []
    num_assets = len(obbs)

    # 1. Render asset OBBs (varied colors)
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
        translation = trimesh.transformations.translation_matrix(center)
        box.apply_transform(translation)
        box.visual.face_colors = color
        meshes.append(box)

    # 2. Render wall OBBs (gray tones)
    if wall_obbs is not None and len(wall_obbs) > 0:
        wall_color = [180, 180, 180, 120]  # Gray with transparency
        for obb in wall_obbs:
            center = obb[0:3]
            size = obb[3:6]
            yaw = obb[6]

            box = trimesh.creation.box(extents=size)
            rotation = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
            box.apply_transform(rotation)
            translation = trimesh.transformations.translation_matrix(center)
            box.apply_transform(translation)
            box.visual.face_colors = wall_color
            meshes.append(box)

    # Add the floor polygon mesh (actual floorplan shape)
    # Use floor_z to draw the floor at the correct Z in the normalized frame
    # Clamp to a minimum, since a very small floor_height would be invisible
    if floor_polygon is not None and len(floor_polygon) >= 3:
        floor_color = [100, 150, 255, 100]  # Blue
        min_floor_height = 0.02  # minimum thickness in the normalized frame
        effective_floor_height = max(floor_height, min_floor_height)
        floor_mesh = create_polygon_extrusion(floor_polygon, floor_z, floor_z + effective_floor_height, floor_color)
        if floor_mesh is not None:
            meshes.append(floor_mesh)
    
    # Add the ceiling polygon mesh
    # Clamp to a minimum, since a very small ceiling_height (thickness) would be invisible
    if ceiling_polygon is not None and len(ceiling_polygon) >= 3:
        ceiling_color = [255, 200, 100, 80]  # Orange
        min_thickness = 0.02  # minimum thickness in the normalized frame
        effective_thickness = max(ceiling_height, min_thickness)
        ceiling_mesh = create_polygon_extrusion(
            ceiling_polygon,
            ceiling_z,
            ceiling_z + effective_thickness,
            ceiling_color
        )
        if ceiling_mesh is not None:
            meshes.append(ceiling_mesh)
    
    # Camera center spheres (red)
    if camera_centers is not None and len(camera_centers) > 0:
        for center in camera_centers:
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.02)
            sphere.apply_translation(center)
            sphere.visual.face_colors = [255, 0, 0, 255]
            meshes.append(sphere)

    if not meshes:
        return None

    combined = trimesh.util.concatenate(meshes)

    scene = trimesh.Scene([combined])

    # Convert Z-up (bbox coords) to Y-up (Babylon.js convention)
    # so the viewer's vertical orbit covers full range properly
    zup_to_yup = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    scene.apply_transform(zup_to_yup)

    temp_path = tempfile.mktemp(suffix=".glb")
    scene.export(temp_path)
    return temp_path


def create_aabb_mesh(bboxes: np.ndarray, highlight_floor: bool = False) -> Optional[str]:
    """Create GLB mesh from AABB data (no rotation)."""
    if bboxes is None or len(bboxes) == 0:
        return None

    if bboxes.ndim == 2 and bboxes.shape == (2, 3):
        bboxes = np.expand_dims(bboxes, axis=0)

    meshes = []
    num_parts = len(bboxes)

    for i, bbox in enumerate(bboxes):
        min_corner = bbox[0]
        max_corner = bbox[1]

        size = max_corner - min_corner
        is_floor = highlight_floor and i == num_parts - 1 and size[2] < 0.05

        if is_floor:
            color = [100, 150, 255, 100]
        else:
            hue = i / max(num_parts - (1 if highlight_floor else 0), 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            color = [int(r*255), int(g*255), int(b*255), 150]

        box = trimesh.creation.box(
            extents=max_corner - min_corner,
            transform=trimesh.transformations.translation_matrix(
                (min_corner + max_corner) / 2
            )
        )
        box.visual.face_colors = color
        meshes.append(box)

    if not meshes:
        return None

    combined = trimesh.util.concatenate(meshes)

    scene = trimesh.Scene([combined])

    # Convert Z-up to Y-up (Babylon.js convention)
    zup_to_yup = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    scene.apply_transform(zup_to_yup)

    temp_path = tempfile.mktemp(suffix=".glb")
    scene.export(temp_path)
    return temp_path


def format_bbox_info(bbox_data: dict, room_id: str, include_layout: bool) -> str:
    """Format bbox information as text."""
    obbs = bbox_data.get('obbs')
    wall_obbs = bbox_data.get('wall_obbs')
    bboxes = bbox_data.get('bboxes')
    floor_polygon = bbox_data.get('floor_polygon')
    ceiling_polygon = bbox_data.get('ceiling_polygon')

    if obbs is None and bboxes is None:
        return "No bbox data"

    # Show OBB info when an OBB is present
    if obbs is not None and len(obbs) > 0:
        info = f"🏠 Room: {room_id}\n"
        info += f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        info += f"📦 Assets: {len(obbs)} OBBs\n"

        if wall_obbs is not None and len(wall_obbs) > 0:
            info += f"🧱 Walls: {len(wall_obbs)} OBBs\n"

        if floor_polygon is not None:
            floor_z = bbox_data.get('floor_z', 0.0)
            floor_h = bbox_data.get('floor_height', 0.0)
            info += f"🟦 Floor: {len(floor_polygon)} vertices (z={floor_z:.3f}, h={floor_h:.3f})\n"

        if ceiling_polygon is not None:
            ceiling_z = bbox_data.get('ceiling_z', 0.0)
            ceiling_h = bbox_data.get('ceiling_height', 0.0)
            info += f"🟧 Ceiling: {len(ceiling_polygon)} vertices (z={ceiling_z:.3f}, h={ceiling_h:.3f})\n"

        info += f"━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        # Asset details
        info += "📦 Asset Details:\n"
        for i, obb in enumerate(obbs):
            center = obb[0:3]
            size = obb[3:6]
            yaw = obb[6]
            yaw_deg = np.degrees(yaw)

            info += f"  [{i}] C:[{center[0]:+.2f},{center[1]:+.2f},{center[2]:+.2f}] "
            info += f"S:[{size[0]:.2f},{size[1]:.2f},{size[2]:.2f}] "
            info += f"Y:{yaw_deg:+.0f}°\n"

        return info
    
    # AABB-only case (legacy format)
    if bboxes is not None and len(bboxes) > 0:
        if bboxes.ndim == 2 and bboxes.shape == (2, 3):
            bboxes = np.expand_dims(bboxes, axis=0)

        info = f"🏠 Room: {room_id}\n"
        info += f"📦 Total parts: {len(bboxes)} (AABB - Legacy format)\n\n"

        for i, bbox in enumerate(bboxes):
            min_corner = bbox[0]
            max_corner = bbox[1]
            size = max_corner - min_corner
            center = (min_corner + max_corner) / 2

            info += f"━━━ Part {i} ━━━\n"
            info += f"  Min: [{min_corner[0]:+.3f}, {min_corner[1]:+.3f}, {min_corner[2]:+.3f}]\n"
            info += f"  Max: [{max_corner[0]:+.3f}, {max_corner[1]:+.3f}, {max_corner[2]:+.3f}]\n"
            info += f"  Size: [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]\n"
            info += f"  Center: [{center[0]:+.3f}, {center[1]:+.3f}, {center[2]:+.3f}]\n\n"

        return info
    
    return "No bbox data"


def create_viewer(base_dir: str):
    """Create Gradio viewer interface."""
    scenes = get_scene_folders(base_dir)

    if not scenes:
        print(f"No scenes found in {base_dir}")
        return None

    # Variant suffix mapping
    VARIANT_SUFFIXES = {
        "Assets Only": "",
        "Assets + Floor": "_with_floor",
        "Assets + Floor + Wall": "_with_floor_wall",
        "Assets + Floor + Wall + Ceiling": "_with_floor_wall_ceiling"
    }

    def on_scene_select(house_id: str, cam_center: bool = False):
        """Handle house selection - update room dropdown."""
        if not house_id:
            return gr.update(choices=[], value=None), None, "Select a house", None

        rooms = get_room_samples(base_dir, house_id)
        default_room = rooms[0] if rooms else None

        if default_room:
            return gr.update(choices=rooms, value=default_room), *load_room_data(house_id, default_room, "Assets + Floor", cam_center)
        else:
            return gr.update(choices=rooms, value=None), None, "No rooms found", None

    def load_room_data(house_id: str, room_name: str, variant: str, cam_center: bool = False):
        """Load and display room data."""
        if not house_id or not room_name:
            return None, "No room selected", None

        suffix = VARIANT_SUFFIXES.get(variant, "")
        include_layout = suffix != ""  # whether to include floor/wall/ceiling

        image = load_floorplan_image(base_dir, house_id, room_name)
        bbox_data = load_bbox(base_dir, house_id, room_name, suffix)

        bbox_info = format_bbox_info(bbox_data, room_name, include_layout)
        bbox_mesh = create_bbox_mesh(bbox_data, highlight_floor=include_layout, use_obb=True,
                                     show_camera_center=cam_center)

        return image, bbox_info, bbox_mesh

    def on_room_select(house_id: str, room_name: str, variant: str, cam_center: bool):
        """Handle room selection."""
        return load_room_data(house_id, room_name, variant, cam_center)

    def on_variant_change(house_id: str, room_name: str, variant: str, cam_center: bool):
        """Handle variant change."""
        return load_room_data(house_id, room_name, variant, cam_center)

    def on_camera_center_toggle(house_id: str, room_name: str, variant: str, cam_center: bool):
        """Handle camera center checkbox toggle."""
        return load_room_data(house_id, room_name, variant, cam_center)

    with gr.Blocks(title="ERP 3D-FRONT BBox Viewer") as demo:
        gr.Markdown("""
        # 🏠 ERP 3D-FRONT Bounding Box Viewer

        View floorplan images and 3D bounding boxes from ERP_3D_FRONT dataset.
        Select a house, then a room to visualize the furniture bounding boxes.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                scene_dropdown = gr.Dropdown(
                    choices=scenes,
                    label="🏛️ Select House",
                    value=scenes[0] if scenes else None,
                    interactive=True
                )

                room_dropdown = gr.Dropdown(
                    choices=[],
                    label="🚪 Select Room",
                    value=None,
                    interactive=True
                )
                
                variant_dropdown = gr.Dropdown(
                    choices=list(VARIANT_SUFFIXES.keys()),
                    label="📐 Layout Variant",
                    value="Assets + Floor",
                    interactive=True
                )

                show_camera_center = gr.Checkbox(
                    label="Show Camera Center",
                    value=False,
                    interactive=True
                )

                image_display = gr.Image(
                    label="2D Floorplan",
                    type="pil",
                    height=450
                )

            with gr.Column(scale=1):
                bbox_info_text = gr.Textbox(
                    label="📋 Bounding Box Info",
                    lines=20,
                    interactive=False
                )

        gr.Markdown("### 🎨 3D Bounding Box Visualization")
        with gr.Row():
            bbox_display = gr.Model3D(
                label="3D Bounding Boxes",
                height=500,
                camera_position=(0, 45, None),
                zoom_speed=2,
                pan_speed=2
            )

        # Event handlers
        scene_dropdown.change(
            on_scene_select,
            inputs=[scene_dropdown, show_camera_center],
            outputs=[room_dropdown, image_display, bbox_info_text, bbox_display]
        )

        room_dropdown.change(
            on_room_select,
            inputs=[scene_dropdown, room_dropdown, variant_dropdown, show_camera_center],
            outputs=[image_display, bbox_info_text, bbox_display]
        )

        variant_dropdown.change(
            on_variant_change,
            inputs=[scene_dropdown, room_dropdown, variant_dropdown, show_camera_center],
            outputs=[image_display, bbox_info_text, bbox_display]
        )

        show_camera_center.change(
            on_camera_center_toggle,
            inputs=[scene_dropdown, room_dropdown, variant_dropdown, show_camera_center],
            outputs=[image_display, bbox_info_text, bbox_display]
        )

        # Initialize with first scene
        if scenes:
            demo.load(
                on_scene_select,
                inputs=[scene_dropdown, show_camera_center],
                outputs=[room_dropdown, image_display, bbox_info_text, bbox_display]
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="3D-FRONT BBox Viewer")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=BASE_DIR,
        help="Path to floorplan data folder"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the Gradio server (default: auto-find available port)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link"
    )

    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory not found: {args.base_dir}")
        print("Please run read_one_3dfront_json_with_assets_251216.py first to generate data.")
        return

    scenes = get_scene_folders(args.base_dir)
    print(f"Found {len(scenes)} houses")
    for house in scenes[:5]:
        rooms = get_room_samples(args.base_dir, house)
        print(f"  - {house}: {len(rooms)} rooms")
    if len(scenes) > 5:
        print(f"  ... and {len(scenes) - 5} more")

    demo = create_viewer(args.base_dir)

    if demo is None:
        print("Failed to create viewer")
        return

    if args.port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            args.port = s.getsockname()[1]

    print(f"\n🚀 Starting viewer on http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

