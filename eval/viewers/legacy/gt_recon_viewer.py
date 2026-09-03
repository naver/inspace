# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
GT Reconstruction GLB Viewer

Simple Gradio app to browse and view GT reconstructed meshes:
  - scene.glb (combined scene)
  - layout.glb (floor + walls)
  - assets (all individual asset GLBs merged)

Usage:
    python eval/viewers/gt_recon_viewer.py --port 7861
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
DEFAULT_DATA_DIR = "evals/gt_recon"


# ============================================================
# Data Discovery
# ============================================================

def discover_samples(data_dir):
    """Discover all (scene_id, room_id) pairs with meshes."""
    samples = []
    scene_rooms = {}
    if not os.path.isdir(data_dir):
        return samples, {}, []

    for scene_id in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene_id)
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

    scene_ids = sorted(scene_rooms.keys())
    return samples, scene_rooms, scene_ids


def merge_asset_glbs(asset_dir):
    """Merge all individual asset GLBs into a single GLB file.
    Returns path to merged temp GLB, or None if no assets."""
    if not os.path.isdir(asset_dir):
        return None

    glb_files = sorted(glob.glob(os.path.join(asset_dir, "*.glb")))
    if not glb_files:
        return None

    scenes = []
    for glb_path in glb_files:
        try:
            scene = trimesh.load(glb_path, force='scene')
            if isinstance(scene, trimesh.Scene):
                scenes.append(scene)
            elif isinstance(scene, trimesh.Trimesh):
                s = trimesh.Scene()
                s.add_geometry(scene, node_name=os.path.basename(glb_path))
                scenes.append(s)
        except Exception:
            continue

    if not scenes:
        return None

    combined = trimesh.Scene()
    for i, scene in enumerate(scenes):
        for name, geom in scene.geometry.items():
            transform = None
            try:
                transform = scene.graph.get(name)[0]
            except Exception:
                pass
            node_name = f"asset_{i:03d}_{name}"
            if transform is not None:
                combined.add_geometry(geom, node_name=node_name, transform=transform)
            else:
                combined.add_geometry(geom, node_name=node_name)

    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    combined.export(tmp.name, file_type="glb")
    return tmp.name


def get_asset_names(asset_dir):
    """Get list of asset filenames."""
    if not os.path.isdir(asset_dir):
        return []
    glb_files = sorted(glob.glob(os.path.join(asset_dir, "*.glb")))
    return [os.path.basename(f).replace(".glb", "") for f in glb_files]


# ============================================================
# Gradio App
# ============================================================

def create_demo(data_dir):
    samples, scene_rooms, scene_ids = discover_samples(data_dir)
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
                f"Showing all {len(scene_ids)} scenes ({total_rooms} rooms)",
            )

        matched_scenes = set()
        matched_rooms_per_scene = {}
        for scene_id, room_id in samples:
            if query in scene_id.lower() or query in room_id.lower():
                matched_scenes.add(scene_id)
                if scene_id not in matched_rooms_per_scene:
                    matched_rooms_per_scene[scene_id] = []
                matched_rooms_per_scene[scene_id].append(room_id)

        matched_scene_list = sorted(matched_scenes)
        if not matched_scene_list:
            return (
                gr.update(choices=scene_ids, value=scene_ids[0] if scene_ids else None),
                gr.update(choices=scene_rooms.get(scene_ids[0], []) if scene_ids else [],
                          value=scene_rooms.get(scene_ids[0], [None])[0] if scene_ids else None),
                f"No results for '{query}'. Showing all.",
            )

        first = matched_scene_list[0]
        rooms = matched_rooms_per_scene.get(first, scene_rooms.get(first, []))
        n_matches = sum(len(v) for v in matched_rooms_per_scene.values())
        return (
            gr.update(choices=matched_scene_list, value=first),
            gr.update(choices=rooms, value=rooms[0] if rooms else None),
            f"Found {n_matches} rooms in {len(matched_scene_list)} scenes",
        )

    def _get_abs_path(scene_id, room_id):
        """Get absolute path for scene/room."""
        if not scene_id or not room_id:
            return ""
        return os.path.abspath(os.path.join(data_dir, scene_id, room_id))

    def _load_meshes(scene_id, room_id):
        """Core loading logic. Returns (scene_path, layout_path, assets_path, info_md, abs_path)."""
        if not scene_id or not room_id:
            return None, None, None, "Please select a scene and room.", ""

        mesh_dir = os.path.join(data_dir, scene_id, room_id, "meshes")
        abs_path = _get_abs_path(scene_id, room_id)
        if not os.path.isdir(mesh_dir):
            return None, None, None, "No meshes directory found.", abs_path

        scene_glb = os.path.join(mesh_dir, "scene.glb")
        layout_glb = os.path.join(mesh_dir, "layout.glb")
        asset_dir_path = os.path.join(mesh_dir, "assets")

        scene_path = scene_glb if os.path.exists(scene_glb) else None
        layout_path = layout_glb if os.path.exists(layout_glb) else None
        assets_path = merge_asset_glbs(asset_dir_path)

        asset_names = get_asset_names(asset_dir_path)
        n_assets = len(asset_names)

        info_lines = [f"### {scene_id}"]
        info_lines.append(f"**{room_id}**\n")
        info_lines.append(f"- scene.glb: {'found' if scene_path else 'missing'}")
        info_lines.append(f"- layout.glb: {'found' if layout_path else 'missing'}")
        info_lines.append(f"- **{n_assets} assets**")
        if asset_names:
            for name in asset_names:
                parts = name.split("_", 1)
                label = parts[1] if len(parts) > 1 else name
                info_lines.append(f"  - {label}")

        return scene_path, layout_path, assets_path, "\n".join(info_lines), abs_path

    def on_load(scene_id, room_id):
        """Load button click."""
        return _load_meshes(scene_id, room_id)

    def _get_index(scene_id, room_id):
        key = f"{scene_id}/{room_id}"
        if key in flat_list:
            return flat_list.index(key)
        return 0

    def on_prev(scene_id, room_id):
        """Navigate to previous sample."""
        idx = (_get_index(scene_id, room_id) - 1) % len(flat_list)
        s, r = samples[idx]
        scene_path, layout_path, assets_path, info, abs_path = _load_meshes(s, r)
        rooms = scene_rooms.get(s, [r])
        return (
            gr.update(choices=scene_ids, value=s),
            gr.update(choices=rooms, value=r),
            scene_path, layout_path, assets_path, info, abs_path,
        )

    def on_next(scene_id, room_id):
        """Navigate to next sample."""
        idx = (_get_index(scene_id, room_id) + 1) % len(flat_list)
        s, r = samples[idx]
        scene_path, layout_path, assets_path, info, abs_path = _load_meshes(s, r)
        rooms = scene_rooms.get(s, [r])
        return (
            gr.update(choices=scene_ids, value=s),
            gr.update(choices=rooms, value=r),
            scene_path, layout_path, assets_path, info, abs_path,
        )

    # ---- Build UI ----
    with gr.Blocks(title="GT Recon GLB Viewer") as demo:
        gr.Markdown(f"# GT Reconstruction Viewer\n**{total_rooms}** rooms across **{len(scene_ids)}** scenes")

        with gr.Row():
            # === Left panel: controls ===
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Browse")
                search_box = gr.Textbox(
                    label="Search (scene ID or room ID)",
                    placeholder="e.g. ea053 or Bedroom",
                )
                search_status = gr.Textbox(
                    label="Search Results",
                    interactive=False,
                    max_lines=1,
                    value=f"Showing all {len(scene_ids)} scenes ({total_rooms} rooms)",
                )
                scene_dd = gr.Dropdown(
                    choices=scene_ids,
                    value=scene_ids[0] if scene_ids else None,
                    label="Scene (House ID)",
                )
                room_dd = gr.Dropdown(
                    choices=scene_rooms.get(scene_ids[0], []) if scene_ids else [],
                    value=scene_rooms.get(scene_ids[0], [None])[0] if scene_ids else None,
                    label="Room",
                )
                with gr.Row():
                    prev_btn = gr.Button("Prev", size="sm")
                    load_btn = gr.Button("Load", variant="primary")
                    next_btn = gr.Button("Next", size="sm")

                gr.Markdown("---")
                with gr.Row():
                    path_box = gr.Textbox(
                        label="Path",
                        interactive=False,
                        max_lines=1,
                        value="",
                        scale=5,
                    )
                    copy_btn = gr.Button("Copy", size="sm", scale=1)
                info_box = gr.Markdown(value="Select a scene and room, then click **Load**.")

            # === Right panel: 3D viewers ===
            with gr.Column(scale=3):
                with gr.Row():
                    scene_viewer = gr.Model3D(label="Scene", height=550)
                    layout_viewer = gr.Model3D(label="Layout", height=550)
                    assets_viewer = gr.Model3D(label="Assets", height=550)

        # ---- Event bindings ----
        search_box.submit(on_search, [search_box], [scene_dd, room_dd, search_status])
        search_box.change(on_search, [search_box], [scene_dd, room_dd, search_status])
        scene_dd.change(on_scene_change, [scene_dd], [room_dd])

        load_btn.click(
            on_load, [scene_dd, room_dd],
            [scene_viewer, layout_viewer, assets_viewer, info_box, path_box],
        )

        nav_outputs = [scene_dd, room_dd, scene_viewer, layout_viewer, assets_viewer, info_box, path_box]
        prev_btn.click(on_prev, [scene_dd, room_dd], nav_outputs)
        next_btn.click(on_next, [scene_dd, room_dd], nav_outputs)

        copy_btn.click(
            fn=None, inputs=[path_box], outputs=None,
            js="(path) => { navigator.clipboard.writeText(path); }",
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT Recon GLB Viewer")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--port", type=int, default=7868)
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    demo = create_demo(args.data_dir)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
