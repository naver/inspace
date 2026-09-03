# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
GT vs Pipeline Unified Viewer

Scene-only comparison row (fast) with expandable detail (layout + assets) per mode.
Enlarge view with optional ceiling removal.
Shows input ERP panorama and cubemap.

Modes:
  1. GT Recon
  2-7. Pipeline variants (random/sdedit × gt/predicted bbox)

Usage:
    python eval/viewers/gt_vs_pipeline_unified_viewer.py --port 7863
"""

import os
import argparse
import tempfile
import shutil
import glob
import socket

import gradio as gr
import trimesh
import numpy as np
from PIL import Image


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

_tmp_dir = tempfile.mkdtemp(prefix="unified_viewer_")


# ============================================================
# Utilities
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

    # Build descriptive filename
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

    # ERP: use first available view
    erp_dir = os.path.join(room_dir, "erp")
    if os.path.isdir(erp_dir):
        for f in sorted(os.listdir(erp_dir)):
            if f.endswith("_colors.png"):
                view_idx = f.split("_")[0]  # e.g. "0000"
                src = os.path.join(erp_dir, f)
                dst = os.path.join(_tmp_dir, f"{scene_id}__{room_id}__erp_{view_idx}.png")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                erp_path = dst
                break

    # Cubemap concat
    if view_idx:
        src = os.path.join(room_dir, "cubic_fov_120_concat", f"{view_idx}_concat.png")
        if os.path.exists(src):
            dst = os.path.join(_tmp_dir, f"{scene_id}__{room_id}__cubemap_{view_idx}.png")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            cubemap_path = dst

    return erp_path, cubemap_path


# ============================================================
# Gradio App
# ============================================================

def create_demo(base_dir, data_dir):
    samples, scene_rooms, scene_ids = discover_samples(base_dir)
    total_rooms = len(samples)
    flat_list = [f"{s}/{r}" for s, r in samples]
    n_modes = len(MODES)

    # Filter MODES to only those that have at least one sample
    active_modes = []
    for mode_label, mode_subdir in MODES:
        mode_dir = os.path.join(base_dir, mode_subdir)
        if os.path.isdir(mode_dir) and os.listdir(mode_dir):
            active_modes.append((mode_label, mode_subdir))
    # Fall back to all if discovery fails
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
        """Fast file-existence check. Returns markdown string with strikethrough for missing modes."""
        parts = []
        for mode_label, mode_subdir in active_modes:
            p = os.path.join(base_dir, mode_subdir, scene_id, room_id, "meshes", "scene.glb")
            if os.path.exists(p):
                parts.append(f"**{mode_label}**")
            else:
                parts.append(f"~~{mode_label}~~")
        return " | ".join(parts)

    def _load_scenes(scene_id, room_id):
        """Load scene GLBs + input images. Returns: [erp, cubemap, scene0, ..., sceneN, avail, path, info]."""
        outputs = []

        # Input images
        erp_img, cubemap_img = load_input_images(data_dir, scene_id, room_id)
        outputs.append(erp_img)
        outputs.append(cubemap_img)

        # Scene GLBs for all modes
        for _, mode_subdir in active_modes:
            glb = get_scene_glb(base_dir, mode_subdir, scene_id, room_id)
            outputs.append(glb)

        # Availability + path + info
        avail_text = _check_availability(scene_id, room_id)
        abs_path = os.path.abspath(os.path.join(base_dir, "gt_recon", scene_id, room_id))
        info = f"**{scene_id}** / **{room_id}**"
        outputs.append(avail_text)
        outputs.append(abs_path)
        outputs.append(info)
        return outputs

    def _load_detail(scene_id, room_id, mode_idx):
        """Load layout + assets for a specific mode."""
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
    with gr.Blocks(title="GT vs Pipeline Unified Viewer", fill_width=True) as demo:
        gr.Markdown(
            f"# GT vs Pipeline Comparison\n"
            f"**{total_rooms}** rooms  |  {len(active_modes)} modes  |  "
            f"Scene overview + expandable layout/assets detail"
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
        with gr.Row():
            for mode_label, _ in active_modes:
                with gr.Column(scale=1, min_width=160):
                    sv = gr.Model3D(label=mode_label, height=420)
                    eb = gr.Button("Enlarge", size="sm")
                    mode_scene_viewers.append(sv)
                    mode_enlarge_btns.append(eb)

        # --- Enlarged view ---
        with gr.Accordion("Enlarged View", open=False) as enlarge_accordion:
            with gr.Row():
                enlarge_label = gr.Markdown("*Select a model to enlarge*")
                ceiling_checkbox = gr.Checkbox(label="Remove Ceiling (top 15%)", value=False)
            enlarge_viewer = gr.Model3D(label="Enlarged", height=700)
            enlarge_original_path = gr.State(value=None)

        # --- Per-mode detail accordions ---
        detail_viewers = {}
        detail_btns = {}
        for mode_idx, (mode_label, _) in enumerate(active_modes):
            with gr.Accordion(f"{mode_label} — Layout & Assets", open=False):
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

        # Toggle ceiling on already-enlarged model
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
    parser = argparse.ArgumentParser(description="GT vs Pipeline Unified Viewer")
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
