# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
GT vs Pipeline Unified Viewer + Video Export

Scene-only comparison row (fast) with expandable detail (layout + assets) per mode.
Enlarge view with optional ceiling removal.
Shows input ERP panorama and cubemap.
**NEW**: Generate rotating turntable videos from GLB models.

Modes:
  1. GT Recon
  2-7. Pipeline variants (random/sdedit x gt/predicted bbox)

Usage:
    python eval/viewers/gt_vs_pipeline_unified_viewer_video.py --port 7863
"""

import os
import argparse
import tempfile
import shutil
import glob
import socket
import math

import gradio as gr
import trimesh
import numpy as np
from PIL import Image

try:
    import pyrender
    HAS_PYRENDER = True
except ImportError:
    HAS_PYRENDER = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


# ============================================================
# Config
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "datasets", "ERP_3D_FRONT_test")

MODES = [
    ("GT Recon",               "gt_recon"),
    ("Random + GT BBox",       "stage12_pipeline/random_gt"),
    ("Random + Pred BBox",     "stage12_pipeline/random_predicted"),
    ("SDEdit0.5 + GT BBox",    "stage12_pipeline/sdedit0.5_gt"),
    ("SDEdit0.5 + Pred BBox",  "stage12_pipeline/sdedit0.5_predicted"),
    ("SDEdit0.3 + Pred BBox",  "stage12_pipeline/sdedit0.3_predicted"),
    ("SDEdit0.7 + Pred BBox",  "stage12_pipeline/sdedit0.7_predicted"),
]

_tmp_dir = tempfile.mkdtemp(prefix="unified_viewer_video_")


# ============================================================
# Utilities (same as original)
# ============================================================

def discover_samples(base_dir):
    """Discover all (scene_id, room_id) from all mode directories."""
    all_pairs = set()
    scene_rooms = {}

    for _, mode_subdir in MODES:
        mode_dir = os.path.join(base_dir, mode_subdir)
        if not os.path.isdir(mode_dir):
            continue
        for scene_id in os.listdir(mode_dir):
            scene_path = os.path.join(mode_dir, scene_id)
            if not os.path.isdir(scene_path):
                continue
            for room_id in os.listdir(scene_path):
                mesh_dir = os.path.join(scene_path, room_id, "meshes")
                if os.path.isdir(mesh_dir):
                    all_pairs.add((scene_id, room_id))
                    scene_rooms.setdefault(scene_id, set()).add(room_id)

    scene_rooms = {k: sorted(v) for k, v in sorted(scene_rooms.items())}
    samples = sorted(all_pairs)
    scene_ids = sorted(scene_rooms.keys())
    return samples, scene_rooms, scene_ids


def _named_copy(src, scene_id, room_id, mode_subdir, suffix):
    """Copy file to temp dir with descriptive name for download."""
    if src is None or not os.path.exists(src):
        return None
    mode_short = mode_subdir.replace("stage12_pipeline/", "").replace("/", "_")
    filename = f"{scene_id}__{room_id}__{mode_short}__{suffix}.glb"
    dst = os.path.join(_tmp_dir, filename)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    return dst


def get_scene_glb(base_dir, mode_subdir, scene_id, room_id):
    """Return scene.glb copied to temp with descriptive name."""
    p = os.path.join(base_dir, mode_subdir, scene_id, room_id, "meshes", "scene.glb")
    return _named_copy(p, scene_id, room_id, mode_subdir, "scene")


def merge_asset_glbs(asset_dir, scene_id="", room_id="", mode_subdir=""):
    """Merge individual asset GLBs into one temp GLB with descriptive name."""
    if not os.path.isdir(asset_dir):
        return None
    glb_files = sorted(glob.glob(os.path.join(asset_dir, "*.glb")))
    if not glb_files:
        return None

    if scene_id and room_id:
        mode_short = mode_subdir.replace("stage12_pipeline/", "").replace("/", "_")
        dst = os.path.join(_tmp_dir, f"{scene_id}__{room_id}__{mode_short}__assets.glb")
    else:
        dst = os.path.join(_tmp_dir, f"assets_{id(asset_dir)}.glb")
    if os.path.exists(dst):
        return dst

    combined = trimesh.Scene()
    for i, glb_path in enumerate(glb_files):
        try:
            scene = trimesh.load(glb_path, force='scene')
            if isinstance(scene, trimesh.Trimesh):
                combined.add_geometry(scene, node_name=f"asset_{i:03d}")
            elif isinstance(scene, trimesh.Scene):
                for name, geom in scene.geometry.items():
                    transform = None
                    try:
                        transform = scene.graph.get(name)[0]
                    except Exception:
                        pass
                    node = f"asset_{i:03d}_{name}"
                    if transform is not None:
                        combined.add_geometry(geom, node_name=node, transform=transform)
                    else:
                        combined.add_geometry(geom, node_name=node)
        except Exception:
            continue

    if not combined.geometry:
        return None
    combined.export(dst, file_type="glb")
    return dst


def remove_ceiling_from_glb(glb_path, cut_ratio=0.15):
    """Remove top portion of mesh by height (Y-up for GLB)."""
    if glb_path is None or not os.path.exists(glb_path):
        return None

    scene = trimesh.load(glb_path, force='scene')
    if not scene.geometry:
        return None

    height_axis = 1
    all_bounds = []
    for name, geom in scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        try:
            transform = scene.graph.get(name)[0]
        except Exception:
            transform = np.eye(4)
        verts_world = trimesh.transformations.transform_points(geom.vertices, transform)
        all_bounds.append(verts_world)

    if not all_bounds:
        return None

    all_verts = np.concatenate(all_bounds, axis=0)
    y_min = all_verts[:, height_axis].min()
    y_max = all_verts[:, height_axis].max()
    y_cut = y_max - cut_ratio * (y_max - y_min)

    new_scene = trimesh.Scene()
    for name, geom in scene.geometry.items():
        try:
            transform = scene.graph.get(name)[0]
        except Exception:
            transform = np.eye(4) if isinstance(geom, trimesh.Trimesh) else None

        if not isinstance(geom, trimesh.Trimesh):
            new_scene.add_geometry(geom, node_name=name, transform=transform)
            continue

        verts_world = trimesh.transformations.transform_points(geom.vertices, transform)
        face_heights = verts_world[geom.faces, height_axis]
        keep_mask = face_heights.max(axis=1) < y_cut
        keep_indices = np.where(keep_mask)[0]

        if len(keep_indices) == 0:
            continue

        new_geom = geom.submesh([keep_indices], append=True)
        new_scene.add_geometry(new_geom, node_name=name, transform=transform)

    if not new_scene.geometry:
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False, dir=_tmp_dir)
    new_scene.export(tmp.name, file_type="glb")
    return tmp.name


def load_input_images(data_dir, scene_id, room_id):
    """Load ERP panorama and cubemap concat as temp PNGs with descriptive names."""
    room_dir = os.path.join(data_dir, scene_id, room_id)
    erp_path = None
    cubemap_path = None
    view_idx = None

    erp_dir = os.path.join(room_dir, "erp")
    if os.path.isdir(erp_dir):
        for f in sorted(os.listdir(erp_dir)):
            if f.endswith("_colors.png"):
                view_idx = f.split("_")[0]
                src = os.path.join(erp_dir, f)
                dst = os.path.join(_tmp_dir, f"{scene_id}__{room_id}__erp_{view_idx}.png")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                erp_path = dst
                break

    if view_idx:
        src = os.path.join(room_dir, "cubic_fov_120_concat", f"{view_idx}_concat.png")
        if os.path.exists(src):
            dst = os.path.join(_tmp_dir, f"{scene_id}__{room_id}__cubemap_{view_idx}.png")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            cubemap_path = dst

    return erp_path, cubemap_path


# ============================================================
# Video Generation
# ============================================================

def _compute_scene_center_and_scale(trimesh_scene):
    """Compute bounding box center and diagonal of a trimesh scene."""
    all_verts = []
    for name, geom in trimesh_scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        try:
            transform = trimesh_scene.graph.get(name)[0]
        except Exception:
            transform = np.eye(4)
        verts = trimesh.transformations.transform_points(geom.vertices, transform)
        all_verts.append(verts)

    if not all_verts:
        return np.zeros(3), 1.0

    all_verts = np.concatenate(all_verts, axis=0)
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    return center, diagonal


def _look_at(eye, target=(0, 0, 0), up=(0, 1, 0)):
    """Create a 4x4 camera-to-world matrix (OpenGL convention: -Z forward)."""
    eye, target, up = map(lambda v: np.asarray(v, dtype=np.float64), (eye, target, up))

    z = eye - target                          # backward (camera looks along -Z)
    z = z / (np.linalg.norm(z) + 1e-12)
    x = np.cross(up, z)                       # right
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)                        # true up

    mat = np.eye(4)
    mat[:3, 0] = x      # column 0 = right
    mat[:3, 1] = y      # column 1 = up
    mat[:3, 2] = z      # column 2 = backward (-forward)
    mat[:3, 3] = eye     # column 3 = position
    return mat


def render_turntable_video(
    glb_path,
    distance_factor=1.5,
    elevation_deg=30.0,
    n_frames=60,
    fps=30,
    width=1024,
    height=768,
    remove_ceiling=False,
    bg_color=None,
    output_format="gif",
):
    """Render a turntable video of a GLB file.

    Returns: path to mp4 file, or error message string.
    """
    if not HAS_PYRENDER:
        return "Error: pyrender not installed. Install with: pip install pyrender"
    if not HAS_IMAGEIO:
        return "Error: imageio not installed. Install with: pip install imageio[ffmpeg]"

    if glb_path is None or not os.path.exists(glb_path):
        return "Error: No GLB file selected."

    # Optionally remove ceiling
    render_path = glb_path
    if remove_ceiling:
        cut_path = remove_ceiling_from_glb(glb_path)
        if cut_path is not None:
            render_path = cut_path

    # Load scene
    trimesh_scene = trimesh.load(render_path, force='scene')
    if not trimesh_scene.geometry:
        return "Error: GLB has no geometry."

    center, diagonal = _compute_scene_center_and_scale(trimesh_scene)
    cam_distance = diagonal * distance_factor

    # Build pyrender scene
    if bg_color is None:
        bg_color = [1.0, 1.0, 1.0, 1.0]

    pr_scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.3, 0.3, 0.3])

    # Add meshes
    for name, geom in trimesh_scene.geometry.items():
        if not isinstance(geom, trimesh.Trimesh):
            continue
        try:
            transform = trimesh_scene.graph.get(name)[0]
        except Exception:
            transform = np.eye(4)

        # Create pyrender mesh with material
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.8, 0.8, 0.8, 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.6,
        )

        if geom.visual is not None:
            if hasattr(geom.visual, 'material'):
                try:
                    pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
                except Exception:
                    pr_mesh = pyrender.Mesh.from_trimesh(geom, material=material, smooth=False)
            elif hasattr(geom.visual, 'vertex_colors'):
                try:
                    pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
                except Exception:
                    pr_mesh = pyrender.Mesh.from_trimesh(geom, material=material, smooth=False)
            else:
                pr_mesh = pyrender.Mesh.from_trimesh(geom, material=material, smooth=False)
        else:
            pr_mesh = pyrender.Mesh.from_trimesh(geom, material=material, smooth=False)

        pr_scene.add(pr_mesh, pose=transform)

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=width / height)
    cam_node = pr_scene.add(camera, pose=np.eye(4))

    # Lights
    dir_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    pr_scene.add(dir_light, pose=np.eye(4))

    # point light above center
    point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = center + np.array([0, diagonal * 0.5, 0])
    pr_scene.add(point_light, pose=light_pose)

    # Renderer
    try:
        renderer = pyrender.OffscreenRenderer(width, height)
    except Exception as e:
        return f"Error creating renderer: {e}. Set PYOPENGL_PLATFORM=egl or osmesa."

    elevation_rad = math.radians(elevation_deg)
    frames = []

    try:
        for i in range(n_frames):
            angle = 2.0 * math.pi * i / n_frames
            # Camera position: orbit around center
            cx = center[0] + cam_distance * math.cos(elevation_rad) * math.sin(angle)
            cy = center[1] + cam_distance * math.sin(elevation_rad)
            cz = center[2] + cam_distance * math.cos(elevation_rad) * math.cos(angle)
            eye = np.array([cx, cy, cz])
            cam_pose = _look_at(eye, center)
            pr_scene.set_pose(cam_node, cam_pose)

            color, _ = renderer.render(pr_scene)
            frames.append(color)
    finally:
        renderer.delete()

    # Encode to output format
    basename = os.path.splitext(os.path.basename(glb_path))[0]

    if output_format == "mp4":
        video_filename = f"{basename}__turntable.mp4"
        video_path = os.path.join(_tmp_dir, video_filename)
        writer = imageio.get_writer(video_path, fps=fps, codec='libx264',
                                     output_params=['-pix_fmt', 'yuv420p'])
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    else:
        # GIF (default)
        video_filename = f"{basename}__turntable.gif"
        video_path = os.path.join(_tmp_dir, video_filename)
        imageio.mimsave(video_path, frames, fps=fps, loop=0)

    return video_path


def save_video_local(video_path, save_dir):
    """Copy video to a local directory. Returns the saved path or error."""
    if video_path is None or not os.path.exists(video_path):
        return "No video to save."
    if not save_dir or not save_dir.strip():
        return "Please specify a save directory."

    save_dir = save_dir.strip()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.basename(video_path)
    dst = os.path.join(save_dir, filename)
    shutil.copy2(video_path, dst)
    return f"Saved to: {dst}"


# ============================================================
# Gradio App
# ============================================================

def create_demo(base_dir, data_dir):
    samples, scene_rooms, scene_ids = discover_samples(base_dir)
    total_rooms = len(samples)
    flat_list = [f"{s}/{r}" for s, r in samples]

    # Filter MODES to only those that have at least one sample
    active_modes = []
    for mode_label, mode_subdir in MODES:
        mode_dir = os.path.join(base_dir, mode_subdir)
        if os.path.isdir(mode_dir) and os.listdir(mode_dir):
            active_modes.append((mode_label, mode_subdir))
    if not active_modes:
        active_modes = MODES

    def on_scene_change(scene_id):
        rooms = scene_rooms.get(scene_id, [])
        if rooms:
            return gr.update(choices=rooms, value=rooms[0])
        return gr.update(choices=[], value=None)

    def on_search(query):
        query = query.strip().lower()
        if not query:
            return (
                gr.update(choices=scene_ids, value=scene_ids[0] if scene_ids else None),
                gr.update(
                    choices=scene_rooms.get(scene_ids[0], []) if scene_ids else [],
                    value=scene_rooms.get(scene_ids[0], [None])[0] if scene_ids else None,
                ),
                f"All {len(scene_ids)} scenes ({total_rooms} rooms)",
            )

        matched_scenes = set()
        matched_rooms_map = {}
        for s, r in samples:
            if query in s.lower() or query in r.lower():
                matched_scenes.add(s)
                matched_rooms_map.setdefault(s, []).append(r)

        matched_list = sorted(matched_scenes)
        if not matched_list:
            return (
                gr.update(choices=scene_ids, value=scene_ids[0] if scene_ids else None),
                gr.update(
                    choices=scene_rooms.get(scene_ids[0], []) if scene_ids else [],
                    value=scene_rooms.get(scene_ids[0], [None])[0] if scene_ids else None,
                ),
                f"No results for '{query}'",
            )

        first = matched_list[0]
        rooms = matched_rooms_map.get(first, scene_rooms.get(first, []))
        n = sum(len(v) for v in matched_rooms_map.values())
        return (
            gr.update(choices=matched_list, value=first),
            gr.update(choices=rooms, value=rooms[0] if rooms else None),
            f"Found {n} rooms in {len(matched_list)} scenes",
        )

    def _check_availability(scene_id, room_id):
        parts = []
        for mode_label, mode_subdir in active_modes:
            p = os.path.join(base_dir, mode_subdir, scene_id, room_id, "meshes", "scene.glb")
            if os.path.exists(p):
                parts.append(f"**{mode_label}**")
            else:
                parts.append(f"~~{mode_label}~~")
        return " | ".join(parts)

    def _load_scenes(scene_id, room_id):
        outputs = []
        erp_img, cubemap_img = load_input_images(data_dir, scene_id, room_id)
        outputs.append(erp_img)
        outputs.append(cubemap_img)

        for _, mode_subdir in active_modes:
            glb = get_scene_glb(base_dir, mode_subdir, scene_id, room_id)
            outputs.append(glb)

        avail_text = _check_availability(scene_id, room_id)
        abs_path = os.path.abspath(os.path.join(base_dir, "gt_recon", scene_id, room_id))
        info = f"**{scene_id}** / **{room_id}**"
        outputs.append(avail_text)
        outputs.append(abs_path)
        outputs.append(info)
        return outputs

    def _load_detail(scene_id, room_id, mode_idx):
        _, mode_subdir = active_modes[mode_idx]
        mesh_dir = os.path.join(base_dir, mode_subdir, scene_id, room_id, "meshes")
        if not os.path.isdir(mesh_dir):
            return [None, None]
        layout_glb = os.path.join(mesh_dir, "layout.glb")
        layout_path = _named_copy(
            layout_glb if os.path.exists(layout_glb) else None,
            scene_id, room_id, mode_subdir, "layout",
        )
        assets_path = merge_asset_glbs(os.path.join(mesh_dir, "assets"), scene_id, room_id, mode_subdir)
        return [layout_path, assets_path]

    def on_load(scene_id, room_id):
        return _load_scenes(scene_id, room_id)

    def _get_index(scene_id, room_id):
        key = f"{scene_id}/{room_id}"
        return flat_list.index(key) if key in flat_list else 0

    def on_prev(scene_id, room_id):
        idx = (_get_index(scene_id, room_id) - 1) % len(flat_list)
        s, r = samples[idx]
        rooms = scene_rooms.get(s, [r])
        return [
            gr.update(choices=scene_ids, value=s),
            gr.update(choices=rooms, value=r),
        ] + _load_scenes(s, r)

    def on_next(scene_id, room_id):
        idx = (_get_index(scene_id, room_id) + 1) % len(flat_list)
        s, r = samples[idx]
        rooms = scene_rooms.get(s, [r])
        return [
            gr.update(choices=scene_ids, value=s),
            gr.update(choices=rooms, value=r),
        ] + _load_scenes(s, r)

    # ---- Build UI ----
    with gr.Blocks(title="GT vs Pipeline Unified Viewer + Video", fill_width=True) as demo:
        gr.Markdown(
            f"# GT vs Pipeline Comparison + Video Export\n"
            f"**{total_rooms}** rooms  |  {len(active_modes)} modes  |  "
            f"Scene overview + expandable layout/assets detail + turntable video"
        )

        # --- Controls ---
        with gr.Row():
            search_box = gr.Textbox(label="Search", placeholder="scene ID or room name", scale=2)
            search_status = gr.Textbox(
                value=f"All {len(scene_ids)} scenes ({total_rooms} rooms)",
                label="Results", interactive=False, max_lines=1, scale=2,
            )

        with gr.Row():
            scene_dd = gr.Dropdown(
                choices=scene_ids, value=scene_ids[0] if scene_ids else None,
                label="Scene", scale=3,
            )
            room_dd = gr.Dropdown(
                choices=scene_rooms.get(scene_ids[0], []) if scene_ids else [],
                value=scene_rooms.get(scene_ids[0], [None])[0] if scene_ids else None,
                label="Room", scale=2,
            )
            prev_btn = gr.Button("Prev", size="sm", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
            next_btn = gr.Button("Next", size="sm", scale=1)

        with gr.Row():
            path_box = gr.Textbox(label="Path", interactive=False, max_lines=1, value="", scale=5)
            copy_btn = gr.Button("Copy", size="sm", scale=1)
        info_box = gr.Markdown("")

        # --- Input images ---
        with gr.Accordion("Input Images", open=True):
            with gr.Row():
                erp_viewer = gr.Image(label="ERP Panorama", height=250)
                cubemap_viewer = gr.Image(label="Cubemap (FOV 120)", height=250)

        # --- Scene comparison row ---
        with gr.Row():
            gr.Markdown("### Scene Comparison")
            avail_box = gr.Markdown("", elem_id="avail_box")
        mode_scene_viewers = []
        mode_enlarge_btns = []
        mode_video_btns = []
        with gr.Row():
            for mode_label, _ in active_modes:
                with gr.Column(scale=1, min_width=160):
                    sv = gr.Model3D(label=mode_label, height=420)
                    with gr.Row():
                        eb = gr.Button("Enlarge", size="sm")
                        vb = gr.Button("Video", size="sm", variant="secondary")
                    mode_scene_viewers.append(sv)
                    mode_enlarge_btns.append(eb)
                    mode_video_btns.append(vb)

        # --- Enlarged view ---
        with gr.Accordion("Enlarged View", open=False) as enlarge_accordion:
            with gr.Row():
                enlarge_label = gr.Markdown("*Select a model to enlarge*")
                ceiling_checkbox = gr.Checkbox(label="Remove Ceiling (top 15%)", value=False)
            enlarge_viewer = gr.Model3D(label="Enlarged", height=700)
            enlarge_original_path = gr.State(value=None)

        # --- Video Generation ---
        with gr.Accordion("Video Generation", open=False) as video_accordion:
            gr.Markdown("### Turntable Video Generator")
            video_source_label = gr.Markdown("*Click 'Video' on any mode to select source GLB*")
            video_source_path = gr.State(value=None)

            with gr.Row():
                vid_distance = gr.Slider(
                    minimum=0.5, maximum=4.0, value=1.2, step=0.1,
                    label="Distance (x bbox diagonal)", scale=2,
                )
                vid_elevation = gr.Slider(
                    minimum=-60, maximum=89, value=45, step=1,
                    label="Elevation (degrees)", scale=2,
                )
            with gr.Row():
                vid_n_frames = gr.Slider(
                    minimum=12, maximum=360, value=120, step=1,
                    label="Number of Frames", scale=2,
                )
                vid_fps = gr.Slider(
                    minimum=10, maximum=60, value=30, step=1,
                    label="FPS", scale=1,
                )
            with gr.Row():
                vid_width = gr.Number(value=512, label="Width", precision=0, scale=1)
                vid_height = gr.Number(value=512, label="Height", precision=0, scale=1)
                vid_format = gr.Dropdown(choices=["gif", "mp4"], value="gif", label="Format", scale=1)
                vid_remove_ceiling = gr.Checkbox(label="Remove Ceiling", value=False, scale=1)
            with gr.Row():
                make_video_btn = gr.Button("Make Video", variant="primary", scale=2)
                video_status = gr.Textbox(label="Status", interactive=False, max_lines=1, scale=3)
            video_player = gr.Video(label="Preview (MP4)", height=500, visible=True)
            gif_player = gr.Image(label="Preview (GIF)", height=500, visible=False)
            video_download = gr.File(label="Download", visible=False)

            with gr.Row():
                save_dir_box = gr.Textbox(
                    label="Save Directory",
                    value=os.path.join(os.path.dirname(SCRIPT_DIR), "video_outputs"),
                    scale=4,
                )
                save_video_btn = gr.Button("Save Video", size="sm", scale=1)
                save_status = gr.Textbox(label="Save Status", interactive=False, max_lines=1, scale=2)

        # --- Per-mode detail accordions ---
        detail_viewers = {}
        detail_btns = {}
        for mode_idx, (mode_label, _) in enumerate(active_modes):
            with gr.Accordion(f"{mode_label} -- Layout & Assets", open=False):
                with gr.Row():
                    lv = gr.Model3D(label=f"{mode_label} Layout", height=380)
                    av = gr.Model3D(label=f"{mode_label} Assets", height=380)
                    detail_btn = gr.Button("Load Detail", size="sm", scale=0)
                detail_viewers[mode_idx] = (lv, av)
                detail_btns[mode_idx] = detail_btn

        # --- Wire outputs ---
        scene_outputs = [erp_viewer, cubemap_viewer] + mode_scene_viewers + [avail_box, path_box, info_box]
        nav_outputs = [scene_dd, room_dd] + scene_outputs

        # --- Events ---
        search_box.submit(on_search, [search_box], [scene_dd, room_dd, search_status])
        search_box.change(on_search, [search_box], [scene_dd, room_dd, search_status])
        scene_dd.change(on_scene_change, [scene_dd], [room_dd])

        load_btn.click(on_load, [scene_dd, room_dd], scene_outputs)
        prev_btn.click(on_prev, [scene_dd, room_dd], nav_outputs)
        next_btn.click(on_next, [scene_dd, room_dd], nav_outputs)

        copy_btn.click(
            fn=None, inputs=[path_box], outputs=None,
            js="(path) => { navigator.clipboard.writeText(path); }",
        )

        # --- Enlarge button events ---
        def _enlarge(glb_path, label_text, remove_ceiling):
            display_path = glb_path
            if remove_ceiling and glb_path is not None:
                cut_path = remove_ceiling_from_glb(glb_path)
                if cut_path is not None:
                    display_path = cut_path
            return [
                gr.update(open=True),
                f"**{label_text}**",
                display_path,
                glb_path,
            ]

        enlarge_outputs = [enlarge_accordion, enlarge_label, enlarge_viewer, enlarge_original_path]

        for mode_idx, (mode_label, _) in enumerate(active_modes):
            mode_enlarge_btns[mode_idx].click(
                lambda glb, rc, lbl=mode_label: _enlarge(glb, lbl, rc),
                [mode_scene_viewers[mode_idx], ceiling_checkbox],
                enlarge_outputs,
            )

        def _toggle_ceiling(remove_ceiling, original_path):
            if original_path is None:
                return None
            if remove_ceiling:
                cut_path = remove_ceiling_from_glb(original_path)
                return cut_path if cut_path is not None else original_path
            return original_path

        ceiling_checkbox.change(
            _toggle_ceiling,
            [ceiling_checkbox, enlarge_original_path],
            [enlarge_viewer],
        )

        # --- Video button events ---
        def _select_for_video(glb_path, mode_label):
            if glb_path is None:
                return [
                    gr.update(open=True),
                    f"**No GLB available for {mode_label}**",
                    None,
                ]
            return [
                gr.update(open=True),
                f"**Source: {mode_label}** — `{os.path.basename(glb_path)}`",
                glb_path,
            ]

        video_select_outputs = [video_accordion, video_source_label, video_source_path]

        for mode_idx, (mode_label, _) in enumerate(active_modes):
            mode_video_btns[mode_idx].click(
                lambda glb, lbl=mode_label: _select_for_video(glb, lbl),
                [mode_scene_viewers[mode_idx]],
                video_select_outputs,
            )

        def _make_video(source_path, distance, elevation, n_frames, fps, width, height, fmt, rm_ceiling):
            if source_path is None:
                return [None, "No source GLB selected. Click 'Video' on a mode first."]
            result = render_turntable_video(
                source_path,
                distance_factor=distance,
                elevation_deg=elevation,
                n_frames=int(n_frames),
                fps=int(fps),
                width=int(width),
                height=int(height),
                remove_ceiling=rm_ceiling,
                output_format=fmt,
            )
            if isinstance(result, str) and result.startswith("Error"):
                return [None, None, result, gr.update(visible=False, value=None)]
            if fmt == "gif":
                return [gr.update(value=None, visible=False),
                        gr.update(value=result, visible=True),
                        f"GIF generated: {int(n_frames)} frames at {int(fps)} FPS",
                        gr.update(visible=True, value=result)]
            else:
                return [gr.update(value=result, visible=True),
                        gr.update(value=None, visible=False),
                        f"MP4 generated: {int(n_frames)} frames at {int(fps)} FPS",
                        gr.update(visible=True, value=result)]

        make_video_btn.click(
            _make_video,
            [video_source_path, vid_distance, vid_elevation, vid_n_frames, vid_fps,
             vid_width, vid_height, vid_format, vid_remove_ceiling],
            [video_player, gif_player, video_status, video_download],
        )

        def _save_video(save_dir):
            # We need to get video path from video_player somehow
            # Gradio Video component value is the path
            return "Use the download button on the video player, or specify path below."

        def _save_video_fn(video_path, save_dir):
            return save_video_local(video_path, save_dir)

        save_video_btn.click(
            _save_video_fn,
            [video_player, save_dir_box],
            [save_status],
        )

        # --- Detail button events ---
        for mode_idx in range(len(active_modes)):
            lv, av = detail_viewers[mode_idx]
            btn = detail_btns[mode_idx]
            btn.click(
                lambda sn, vi, mi=mode_idx: _load_detail(sn, vi, mi),
                [scene_dd, room_dd],
                [lv, av],
            )

    return demo


if __name__ == "__main__":
    # Set pyrender backend for headless rendering
    if "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

    parser = argparse.ArgumentParser(description="GT vs Pipeline Unified Viewer + Video")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    demo = create_demo(args.base_dir, args.data_dir)

    port = args.port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", port)) != 0:
                break
            port += 1
    if port != args.port:
        print(f"Port {args.port} in use, using {port}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=args.share,
    )
