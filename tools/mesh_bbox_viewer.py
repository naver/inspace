# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Gradio viewer for 3D-FRONT room meshes + bounding boxes.

Displays 4 panels in 2x2 grid:
  1. full_room_wo_ceiling.obj
  2. layout_wo_ceiling.obj
  3. assets.obj
  4. 3D Bounding Boxes

Usage:
    python mesh_bbox_viewer.py --port 7860
    python mesh_bbox_viewer.py --base-dir /path/to/dataset
"""

import os
import json
import tempfile
import colorsys
import argparse
import socket
from pathlib import Path
from typing import List, Optional

import gradio as gr
import numpy as np

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("trimesh not installed")

# BASE_DIR = "figure_sample"
BASE_DIR = "datasets/ERP_3D_FRONT_test"


# ── Data loading ──────────────────────────────────────────────

def get_scene_folders(base_dir: str) -> List[str]:
    base = Path(base_dir)
    if not base.exists():
        return []
    houses = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir():
            continue
        for room in folder.iterdir():
            if room.is_dir() and (room / 'mesh').exists():
                houses.append(folder.name)
                break
    return houses


def get_room_samples(base_dir: str, house_id: str) -> List[str]:
    house_dir = Path(base_dir) / house_id
    if not house_dir.exists():
        return []
    return [r.name for r in sorted(house_dir.iterdir())
            if r.is_dir() and (r / 'mesh').exists()]


def load_obj_as_meshes(obj_path: str) -> List[trimesh.Trimesh]:
    """Load OBJ and return list of trimesh geometries with original textures."""
    if not os.path.exists(obj_path):
        return []
    try:
        result = trimesh.load(obj_path, process=False, force='scene')
        if isinstance(result, trimesh.Trimesh):
            return [result]
        return [g for g in result.geometry.values() if isinstance(g, trimesh.Trimesh)]
    except Exception as e:
        print(f"Failed to load {obj_path}: {e}")
        return []


def load_bbox_data(base_dir: str, house_id: str, room_name: str) -> dict:
    result = {'obbs': None, 'wall_obbs': None,
              'floor_polygon': None, 'ceiling_polygon': None,
              'asset_names': [], 'norm_center': None, 'norm_scale': None}

    bbox_dir = Path(base_dir) / house_id / room_name / '3d_bounding_box'
    if not bbox_dir.exists():
        return result

    npz_files = list(bbox_dir.glob('*.npz'))
    if not npz_files:
        return result

    data = np.load(npz_files[0], allow_pickle=True)
    for key in ['obbs', 'wall_obbs', 'floor_polygon', 'ceiling_polygon', 'norm_center']:
        if key in data:
            result[key] = data[key]
    for key in ['norm_scale', 'floor_height', 'floor_z', 'ceiling_z', 'ceiling_height']:
        if key in data:
            result[key] = float(data[key])
    if 'asset_names' in data:
        result['asset_names'] = list(data['asset_names'])

    # Camera centers
    cam_path = Path(base_dir) / house_id / room_name / 'camera_poses.json'
    if cam_path.exists():
        with open(cam_path) as f:
            cam_data = json.load(f)
        if 'views' in cam_data:
            nc, ns = result.get('norm_center'), result.get('norm_scale')
            if nc is not None and ns is not None:
                centers = [(np.array(v['location']) - nc) / ns for v in cam_data['views']]
                result['camera_centers'] = np.array(centers)

    return result


# ── Mesh building helpers ─────────────────────────────────────

def _ear_clip_triangulate(polygon_2d: np.ndarray):
    n = len(polygon_2d)
    if n < 3:
        return []
    pts = polygon_2d
    area = sum(pts[i][0]*pts[(i+1)%n][1] - pts[(i+1)%n][0]*pts[i][1] for i in range(n))
    indices = list(range(n))
    if area < 0:
        indices.reverse()

    def _cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def _in_tri(p, a, b, c):
        d1, d2, d3 = _cross(a,b,p), _cross(b,c,p), _cross(c,a,p)
        return not ((d1<0 or d2<0 or d3<0) and (d1>0 or d2>0 or d3>0))

    triangles = []
    while len(indices) > 2:
        ear_found = False
        m = len(indices)
        for i in range(m):
            pi, ni = (i-1)%m, (i+1)%m
            a, b, c = pts[indices[pi]], pts[indices[i]], pts[indices[ni]]
            if _cross(a, b, c) <= 0:
                continue
            if all(not _in_tri(pts[indices[j]], a, b, c) for j in range(m) if j not in (pi, i, ni)):
                triangles.append([indices[pi], indices[i], indices[ni]])
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            break
    return triangles


def _create_polygon_extrusion(polygon_2d, min_z, max_z, color):
    if polygon_2d is None or len(polygon_2d) < 3 or max_z <= min_z:
        return None
    try:
        n = len(polygon_2d)
        bottom = np.column_stack([polygon_2d, np.full(n, min_z)])
        top = np.column_stack([polygon_2d, np.full(n, max_z)])
        verts = np.vstack([bottom, top])
        faces = []
        for i in range(n):
            j = (i+1) % n
            faces.extend([[i, j, j+n], [i, j+n, i+n]])
        for tri in _ear_clip_triangulate(polygon_2d):
            faces.append(tri)
            faces.append([tri[2]+n, tri[1]+n, tri[0]+n])
        mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces))
        mesh.visual.face_colors = color
        return mesh
    except Exception:
        return None


def meshes_to_glb(meshes: List[trimesh.Trimesh]) -> Optional[str]:
    """Combine meshes into GLB with Z-up to Y-up transform."""
    if not meshes:
        return None
    scene = trimesh.Scene(meshes)
    zup_to_yup = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    scene.apply_transform(zup_to_yup)
    path = tempfile.mktemp(suffix=".glb")
    scene.export(path)
    return path


def build_bbox_meshes(bbox_data: dict, show_walls=False, show_floor=True,
                      show_ceiling=False, show_camera=False) -> List[trimesh.Trimesh]:
    """Create bbox visualization meshes."""
    meshes = []
    obbs = bbox_data.get('obbs')

    # Asset OBBs
    if obbs is not None and len(obbs) > 0:
        n = len(obbs)
        for i, obb in enumerate(obbs):
            center, size, yaw = obb[0:3], obb[3:6], obb[6]
            hue = i / max(n, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            color = [int(r*255), int(g*255), int(b*255), 120]
            box = trimesh.creation.box(extents=size)
            box.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 0, 1]))
            box.apply_translation(center)
            box.visual.face_colors = color
            meshes.append(box)

    # Wall OBBs
    if show_walls and bbox_data.get('wall_obbs') is not None:
        for obb in bbox_data['wall_obbs']:
            center, size, yaw = obb[0:3], obb[3:6], obb[6]
            box = trimesh.creation.box(extents=size)
            box.apply_transform(trimesh.transformations.rotation_matrix(yaw, [0, 0, 1]))
            box.apply_translation(center)
            box.visual.face_colors = [180, 180, 180, 80]
            meshes.append(box)

    # Floor
    if show_floor and bbox_data.get('floor_polygon') is not None:
        fp = bbox_data['floor_polygon']
        fz = bbox_data.get('floor_z', 0.0)
        fh = max(bbox_data.get('floor_height', 0.05), 0.02)
        m = _create_polygon_extrusion(fp, fz, fz + fh, [100, 150, 255, 80])
        if m:
            meshes.append(m)

    # Ceiling
    if show_ceiling and bbox_data.get('ceiling_polygon') is not None:
        cp = bbox_data['ceiling_polygon']
        cz = bbox_data.get('ceiling_z', 0.0)
        ch = max(bbox_data.get('ceiling_height', 0.02), 0.02)
        m = _create_polygon_extrusion(cp, cz, cz + ch, [255, 200, 100, 60])
        if m:
            meshes.append(m)

    # Camera centers
    if show_camera and bbox_data.get('camera_centers') is not None:
        for c in bbox_data['camera_centers']:
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.02)
            sphere.apply_translation(c)
            sphere.visual.face_colors = [255, 0, 0, 255]
            meshes.append(sphere)

    return meshes


# ── GLB builders for each panel ───────────────────────────────

def build_full_room_glb(base_dir, house_id, room_name):
    if not HAS_TRIMESH or not house_id or not room_name:
        return None
    obj_path = Path(base_dir) / house_id / room_name / 'mesh' / 'full_room_wo_ceiling.obj'
    meshes = load_obj_as_meshes(str(obj_path))
    return meshes_to_glb(meshes)


def build_layout_glb(base_dir, house_id, room_name):
    if not HAS_TRIMESH or not house_id or not room_name:
        return None
    obj_path = Path(base_dir) / house_id / room_name / 'mesh' / 'layout_wo_ceiling.obj'
    meshes = load_obj_as_meshes(str(obj_path))
    return meshes_to_glb(meshes)


def build_assets_glb(base_dir, house_id, room_name):
    if not HAS_TRIMESH or not house_id or not room_name:
        return None
    obj_path = Path(base_dir) / house_id / room_name / 'mesh' / 'assets.obj'
    meshes = load_obj_as_meshes(str(obj_path))
    return meshes_to_glb(meshes)


def build_bbox_glb(base_dir, house_id, room_name, show_walls, show_floor, show_ceiling, show_camera):
    if not HAS_TRIMESH or not house_id or not room_name:
        return None
    bbox_data = load_bbox_data(base_dir, house_id, room_name)
    meshes = build_bbox_meshes(bbox_data, show_walls=show_walls, show_floor=show_floor,
                               show_ceiling=show_ceiling, show_camera=show_camera)
    return meshes_to_glb(meshes)


# ── Info text ─────────────────────────────────────────────────

def format_info(base_dir, house_id, room_name):
    if not house_id or not room_name:
        return "No room selected"

    mesh_dir = Path(base_dir) / house_id / room_name / 'mesh'
    bbox_data = load_bbox_data(base_dir, house_id, room_name)

    info = f"Room: {room_name}\nHouse: {house_id}\n{'='*40}\n\n"

    for name in ['full_room_wo_ceiling.obj', 'layout_wo_ceiling.obj', 'assets.obj']:
        exists = (mesh_dir / name).exists()
        info += f"  {'[v]' if exists else '[ ]'} {name}\n"
    info += "\n"

    obbs = bbox_data.get('obbs')
    if obbs is not None:
        info += f"Asset BBoxes: {len(obbs)}\n"
        names = bbox_data.get('asset_names', [])
        for i, obb in enumerate(obbs):
            c, s, yaw = obb[0:3], obb[3:6], obb[6]
            n = f" ({names[i]})" if i < len(names) else ""
            info += f"  [{i:2d}]{n}  C:[{c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}] S:[{s[0]:.2f},{s[1]:.2f},{s[2]:.2f}] Y:{np.degrees(yaw):+.0f}\n"

    if bbox_data.get('wall_obbs') is not None:
        info += f"Wall BBoxes: {len(bbox_data['wall_obbs'])}\n"
    if bbox_data.get('camera_centers') is not None:
        info += f"Camera views: {len(bbox_data['camera_centers'])}\n"

    return info


# ── Gradio UI ─────────────────────────────────────────────────

def create_viewer(base_dir: str):
    scenes = get_scene_folders(base_dir)
    if not scenes:
        print(f"No scenes found in {base_dir}")
        return None

    MODEL3D_KWARGS = dict(height=400, camera_position=(0, 45, None), zoom_speed=2, pan_speed=2)

    def update_all(house_id, room_name, show_walls, show_floor, show_ceiling, show_camera):
        """Update all 4 panels + info."""
        glb1 = build_full_room_glb(base_dir, house_id, room_name)
        glb2 = build_layout_glb(base_dir, house_id, room_name)
        glb3 = build_assets_glb(base_dir, house_id, room_name)
        glb4 = build_bbox_glb(base_dir, house_id, room_name, show_walls, show_floor, show_ceiling, show_camera)
        info = format_info(base_dir, house_id, room_name)
        return glb1, glb2, glb3, glb4, info

    def on_scene_select(house_id, show_walls, show_floor, show_ceiling, show_camera):
        if not house_id:
            return gr.update(choices=[], value=None), None, None, None, None, ""
        rooms = get_room_samples(base_dir, house_id)
        default = rooms[0] if rooms else None
        if default:
            glb1, glb2, glb3, glb4, info = update_all(
                house_id, default, show_walls, show_floor, show_ceiling, show_camera)
            return gr.update(choices=rooms, value=default), glb1, glb2, glb3, glb4, info
        return gr.update(choices=rooms, value=None), None, None, None, None, "No rooms"

    with gr.Blocks(title="Mesh + BBox Viewer") as demo:
        gr.Markdown("# Mesh + BBox 2x2 Viewer")

        # Controls row
        with gr.Row():
            scene_dd = gr.Dropdown(choices=scenes, label="House",
                                   value=scenes[0] if scenes else None, interactive=True)
            room_dd = gr.Dropdown(choices=[], label="Room", value=None, interactive=True)
            cb_walls = gr.Checkbox(label="Walls", value=False)
            cb_floor = gr.Checkbox(label="Floor", value=True)
            cb_ceiling = gr.Checkbox(label="Ceiling", value=False)
            cb_camera = gr.Checkbox(label="Camera", value=False)

        # 2x2 grid of Model3D viewers
        with gr.Row():
            view_full = gr.Model3D(label="full_room_wo_ceiling.obj", **MODEL3D_KWARGS)
            view_layout = gr.Model3D(label="layout_wo_ceiling.obj", **MODEL3D_KWARGS)
        with gr.Row():
            view_assets = gr.Model3D(label="assets.obj", **MODEL3D_KWARGS)
            view_bbox = gr.Model3D(label="3D Bounding Boxes", **MODEL3D_KWARGS)

        # Info
        info_text = gr.Textbox(label="Info", lines=12, interactive=False)

        # Outputs list
        all_outputs = [view_full, view_layout, view_assets, view_bbox, info_text]
        bbox_inputs = [cb_walls, cb_floor, cb_ceiling, cb_camera]

        # Scene select
        scene_dd.change(
            on_scene_select,
            inputs=[scene_dd] + bbox_inputs,
            outputs=[room_dd] + all_outputs
        )

        # Room select or bbox option change -> update all
        for component in [room_dd] + bbox_inputs:
            component.change(
                update_all,
                inputs=[scene_dd, room_dd] + bbox_inputs,
                outputs=all_outputs
            )

        # Init on load
        if scenes:
            demo.load(
                on_scene_select,
                inputs=[scene_dd] + bbox_inputs,
                outputs=[room_dd] + all_outputs
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Mesh + BBox 2x2 Viewer")
    parser.add_argument("--base-dir", type=str, default=BASE_DIR)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.base_dir):
        print(f"Error: {args.base_dir} not found")
        return

    scenes = get_scene_folders(args.base_dir)
    print(f"Found {len(scenes)} houses")

    demo = create_viewer(args.base_dir)
    if demo is None:
        return

    if args.port is None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            args.port = s.getsockname()[1]

    print(f"\nStarting viewer on http://localhost:{args.port}")
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
