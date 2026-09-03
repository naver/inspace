# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
GT vs Pipeline Comparison Viewer

Side-by-side comparison of GT reconstruction and 4 pipeline modes.
Each row shows scene.glb, layout.glb, and merged assets for one mode.
Missing GLBs (not yet generated) are silently skipped.

Rows:
  1. GT Recon
  2. random + gt_bbox
  3. random + predicted_bbox
  4. sdedit0.5 + gt_bbox
  5. sdedit0.5 + predicted_bbox

Usage:
    python eval/viewers/gt_vs_pipeline_viewer.py --port 7863
"""

import os
import argparse
import tempfile
import glob

import gradio as gr
import trimesh


# ============================================================
# Config
# ============================================================
BASE_DIR = "evals"

MODES = [
    ("GT Recon", "gt_recon"),
    ("Random + GT BBox", "stage12_pipeline/random_gt"),
    ("Random + Pred BBox", "stage12_pipeline/random_predicted"),
    ("SDEdit0.5 + GT BBox", "stage12_pipeline/sdedit0.5_gt"),
    ("SDEdit0.5 + Pred BBox", "stage12_pipeline/sdedit0.5_predicted"),
]


# ============================================================
# Utilities
# ============================================================

def discover_samples(base_dir):
    """Discover all (scene_id, room_id) from gt_recon (superset)."""
    samples = []
    scene_rooms = {}
    gt_dir = os.path.join(base_dir, "gt_recon")
    if not os.path.isdir(gt_dir):
        return samples, {}, []

    for scene_id in sorted(os.listdir(gt_dir)):
        scene_path = os.path.join(gt_dir, scene_id)
        if not os.path.isdir(scene_path):
            continue
        rooms = []
        for room_id in sorted(os.listdir(scene_path)):
            mesh_dir = os.path.join(scene_path, room_id, "meshes")
            if os.path.isdir(mesh_dir):
                samples.append((scene_id, room_id))
                rooms.append(room_id)
        if rooms:
            scene_rooms[scene_id] = rooms

    return samples, scene_rooms, sorted(scene_rooms.keys())


def merge_asset_glbs(asset_dir):
    """Merge individual asset GLBs into one temp GLB."""
    if not os.path.isdir(asset_dir):
        return None
    glb_files = sorted(glob.glob(os.path.join(asset_dir, "*.glb")))
    if not glb_files:
        return None

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
    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    combined.export(tmp.name, file_type="glb")
    return tmp.name


def load_mode_meshes(base_dir, mode_subdir, scene_id, room_id):
    """Load scene/layout/assets GLBs for one mode. Returns (scene, layout, assets) paths or None."""
    mesh_dir = os.path.join(base_dir, mode_subdir, scene_id, room_id, "meshes")
    if not os.path.isdir(mesh_dir):
        return None, None, None

    scene_glb = os.path.join(mesh_dir, "scene.glb")
    layout_glb = os.path.join(mesh_dir, "layout.glb")
    asset_dir = os.path.join(mesh_dir, "assets")

    scene_path = scene_glb if os.path.exists(scene_glb) else None
    layout_path = layout_glb if os.path.exists(layout_glb) else None
    assets_path = merge_asset_glbs(asset_dir)

    return scene_path, layout_path, assets_path


# ============================================================
# Gradio App
# ============================================================

def create_demo(base_dir):
    samples, scene_rooms, scene_ids = discover_samples(base_dir)
    total_rooms = len(samples)
    flat_list = [f"{s}/{r}" for s, r in samples]

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
                gr.update(choices=scene_rooms.get(scene_ids[0], []) if scene_ids else [],
                          value=scene_rooms.get(scene_ids[0], [None])[0] if scene_ids else None),
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
                gr.update(choices=scene_rooms.get(scene_ids[0], []) if scene_ids else [],
                          value=scene_rooms.get(scene_ids[0], [None])[0] if scene_ids else None),
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

    def _load_all(scene_id, room_id):
        """Load all 5 modes. Returns flat list of (scene, layout, assets) * 5 + path."""
        outputs = []
        for _, mode_subdir in MODES:
            s, l, a = load_mode_meshes(base_dir, mode_subdir, scene_id, room_id)
            outputs.extend([s, l, a])

        abs_path = os.path.abspath(os.path.join(base_dir, "gt_recon", scene_id, room_id)) if scene_id and room_id else ""
        info = f"**{scene_id}** / **{room_id}**" if scene_id and room_id else ""
        outputs.append(abs_path)
        outputs.append(info)
        return outputs

    def on_load(scene_id, room_id):
        return _load_all(scene_id, room_id)

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
        ] + _load_all(s, r)

    def on_next(scene_id, room_id):
        idx = (_get_index(scene_id, room_id) + 1) % len(flat_list)
        s, r = samples[idx]
        rooms = scene_rooms.get(s, [r])
        return [
            gr.update(choices=scene_ids, value=s),
            gr.update(choices=rooms, value=r),
        ] + _load_all(s, r)

    # ---- Build UI ----
    with gr.Blocks(title="GT vs Pipeline Viewer") as demo:
        gr.Markdown(f"# GT vs Pipeline Comparison\n**{total_rooms}** rooms  |  5 modes (GT Recon + 4 pipeline)")

        # --- Controls ---
        with gr.Row():
            search_box = gr.Textbox(label="Search", placeholder="scene ID or room name", scale=2)
            search_status = gr.Textbox(
                value=f"All {len(scene_ids)} scenes ({total_rooms} rooms)",
                label="Results", interactive=False, max_lines=1, scale=2,
            )

        with gr.Row():
            scene_dd = gr.Dropdown(choices=scene_ids, value=scene_ids[0] if scene_ids else None, label="Scene", scale=3)
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

        # --- 5 rows x 3 viewers ---
        all_viewers = []  # flat: [scene0, layout0, assets0, scene1, layout1, assets1, ...]
        for mode_label, _ in MODES:
            gr.Markdown(f"### {mode_label}")
            with gr.Row():
                sv = gr.Model3D(label="Scene", height=380)
                lv = gr.Model3D(label="Layout", height=380)
                av = gr.Model3D(label="Assets", height=380)
            all_viewers.extend([sv, lv, av])

        # --- Outputs ---
        load_outputs = all_viewers + [path_box, info_box]
        nav_outputs = [scene_dd, room_dd] + load_outputs

        # --- Events ---
        search_box.submit(on_search, [search_box], [scene_dd, room_dd, search_status])
        search_box.change(on_search, [search_box], [scene_dd, room_dd, search_status])
        scene_dd.change(on_scene_change, [scene_dd], [room_dd])

        load_btn.click(on_load, [scene_dd, room_dd], load_outputs)
        prev_btn.click(on_prev, [scene_dd, room_dd], nav_outputs)
        next_btn.click(on_next, [scene_dd, room_dd], nav_outputs)

        copy_btn.click(
            fn=None, inputs=[path_box], outputs=None,
            js="(path) => { navigator.clipboard.writeText(path); }",
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT vs Pipeline Viewer")
    parser.add_argument("--base_dir", type=str, default=BASE_DIR)
    parser.add_argument("--port", type=int, default=None, help="Port number (None for auto)")
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    demo = create_demo(args.base_dir)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
