# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
InSpace Batch Results Viewer

Compare InSpace results across different modes:
  - GT Recon (gt_recon)
  - Predicted BBox + Random (bbox_predicted_random)
  - Predicted BBox + SDEdit 0.3 (bbox_predicted_sdedit0.3)
  - Predicted BBox + SDEdit 0.7 (bbox_predicted_sdedit0.7)

Shows scene.glb by default (fast). Layout/assets loaded on demand via detail buttons.

Usage:
    python eval/viewers/inspace_batch_viewer.py
    python eval/viewers/inspace_batch_viewer.py --port 7860
"""

import os
import json
import argparse
import shutil
import tempfile

import gradio as gr

# ============================================================
# Config
# ============================================================
EVALS_DIR = os.path.dirname(os.path.abspath(__file__))

# (label, dir_name, is_nested)
# is_nested=True means gt_recon/{uuid}/{room_name}/meshes/ structure
MODE_DIRS = [
    ("GT Recon", "gt_recon", True),
    ("Pred BBox + Random", "output_InSpace_batch_bbox_predicted_random", False),
    ("Pred BBox + SDEdit 0.3", "output_InSpace_batch_bbox_predicted_sdedit0.3", False),
    ("Pred BBox + SDEdit 0.7", "output_InSpace_batch_bbox_predicted_sdedit0.7", False),
]

CUBEMAP_FACES = ["front", "right", "back", "left", "top", "bottom"]

DATA_DIR = "datasets/ERP_3D_FRONT_test"

_tmp_dir = tempfile.mkdtemp(prefix="inspace_viewer_")


# ============================================================
# Sample Discovery
# ============================================================

def discover_samples(evals_dir):
    """Discover samples from the perspective_eval_dataset_selected.json and available outputs."""
    json_path = os.path.join(evals_dir, "perspective_eval_dataset_selected.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        json_samples = data.get("samples", [])
    else:
        json_samples = []

    # Build folder_name -> sample_info mapping from JSON
    json_map = {}
    for s in json_samples:
        idx = s.get("idx", 0)
        room_name = s.get("room_name", "")
        view_idx = s.get("view_idx", 0)
        folder_name = f"{idx:04d}_{room_name}_v{view_idx}"
        json_map[folder_name] = s

    # Discover all folder names across flat mode dirs
    all_folders = set()
    for _, mode_dir, is_nested in MODE_DIRS:
        if is_nested:
            continue
        mode_path = os.path.join(evals_dir, mode_dir)
        if os.path.isdir(mode_path):
            for name in os.listdir(mode_path):
                if os.path.isdir(os.path.join(mode_path, name)):
                    all_folders.add(name)

    # Also add samples from JSON that may exist in nested dirs
    for folder_name, s in json_map.items():
        if folder_name not in all_folders:
            uuid = s.get("uuid", "")
            room_name = s.get("room_name", "")
            for _, mode_dir, is_nested in MODE_DIRS:
                if is_nested and uuid and room_name:
                    if os.path.isdir(os.path.join(evals_dir, mode_dir, uuid, room_name)):
                        all_folders.add(folder_name)
                        break

    samples = []
    for folder in sorted(all_folders):
        info = json_map.get(folder, {})
        samples.append({
            "folder_name": folder,
            "uuid": info.get("uuid", ""),
            "room_name": info.get("room_name", folder),
            "view_idx": info.get("view_idx", 0),
            "erp_image": info.get("erp_image", ""),
            "perspective_image": info.get("perspective_image", ""),
        })

    return samples


def get_cubemap_concat(sample_info):
    """Get the concat cubemap image path, copied with descriptive name."""
    uuid = sample_info.get("uuid", "")
    room_name = sample_info.get("room_name", "")
    view_idx = sample_info.get("view_idx", 0)
    if uuid and room_name:
        src = os.path.join(DATA_DIR, uuid, room_name, "cubic_fov_120_concat", f"{view_idx:04d}_concat.png")
        if os.path.exists(src):
            dst = os.path.join(_tmp_dir, f"{uuid}_{room_name}_cubemap_{view_idx:04d}.png")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            return dst
    return None


def _named_image_copy(src, uuid, room_name, view_idx, tag):
    """Copy image to temp with descriptive name. Always saves as .png."""
    if not src or not os.path.exists(src):
        return None
    dst = os.path.join(_tmp_dir, f"{uuid}_{room_name}_{tag}_{view_idx:04d}.png")
    if not os.path.exists(dst):
        # Convert to PNG if source is not PNG
        if src.lower().endswith(".png"):
            shutil.copy2(src, dst)
        else:
            from PIL import Image
            Image.open(src).save(dst, "PNG")
    return dst


def _resolve_glb_path(evals_dir, mode_dir, is_nested, folder_name, sample_info, mesh_type):
    """Resolve the actual GLB path depending on flat vs nested structure."""
    if mesh_type == "assets":
        fname = "assets.glb"
    elif mesh_type == "layout":
        fname = "layout.glb"
    else:
        fname = "scene.glb"

    if is_nested:
        uuid = sample_info.get("uuid", "")
        room_name = sample_info.get("room_name", "")
        if not uuid or not room_name:
            return None
        base = os.path.join(evals_dir, mode_dir, uuid, room_name, "meshes")
        if mesh_type == "assets":
            assets_dir = os.path.join(base, "assets")
            if not os.path.isdir(assets_dir):
                return None
            return ("merge", assets_dir)
        p = os.path.join(base, fname)
    else:
        p = os.path.join(evals_dir, mode_dir, folder_name, fname)

    return p if os.path.exists(p) else None


def _merge_asset_glbs(assets_dir):
    """Merge individual asset GLBs into one scene using trimesh."""
    import trimesh
    scene = trimesh.Scene()
    for glb_file in sorted(os.listdir(assets_dir)):
        if not glb_file.endswith(".glb"):
            continue
        path = os.path.join(assets_dir, glb_file)
        try:
            mesh = trimesh.load(path, force='scene')
            name = os.path.splitext(glb_file)[0]
            if isinstance(mesh, trimesh.Scene):
                for gname, geom in mesh.geometry.items():
                    scene.add_geometry(geom, node_name=f"{name}_{gname}")
            else:
                scene.add_geometry(mesh, node_name=name)
        except Exception:
            continue
    return scene


def get_glb(evals_dir, mode_dir, is_nested, folder_name, sample_info, mesh_type, mode_label):
    """Get a GLB path for a given mode/sample/mesh_type, copying to temp with descriptive name."""
    resolved = _resolve_glb_path(evals_dir, mode_dir, is_nested, folder_name, sample_info, mesh_type)
    if resolved is None:
        return None

    mode_short = mode_label.replace(" ", "_")
    dst_name = f"{folder_name}__{mode_short}__{mesh_type}.glb"
    dst = os.path.join(_tmp_dir, dst_name)

    if isinstance(resolved, tuple) and resolved[0] == "merge":
        if not os.path.exists(dst):
            assets_dir = resolved[1]
            scene = _merge_asset_glbs(assets_dir)
            if len(scene.geometry) == 0:
                return None
            scene.export(dst)
        return dst
    else:
        if not os.path.exists(dst):
            shutil.copy2(resolved, dst)
        return dst


def _check_scene_exists(evals_dir, mode_dir, is_nested, folder_name, sample_info):
    """Fast check if scene.glb exists for a given mode/sample."""
    resolved = _resolve_glb_path(evals_dir, mode_dir, is_nested, folder_name, sample_info, "scene")
    if resolved is None:
        return False
    if isinstance(resolved, tuple):
        return True
    return True  # _resolve_glb_path already checks os.path.exists


def load_metadata(evals_dir, mode_dir, folder_name):
    """Load metadata.json for a sample in a given mode."""
    p = os.path.join(evals_dir, mode_dir, folder_name, "metadata.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


# ============================================================
# Gradio App
# ============================================================

def create_demo(evals_dir):
    samples = discover_samples(evals_dir)
    n_samples = len(samples)
    folder_names = [s["folder_name"] for s in samples]
    sample_map = {s["folder_name"]: s for s in samples}

    # Check which modes actually have data
    available_modes = []
    for label, mode_dir, is_nested in MODE_DIRS:
        mode_path = os.path.join(evals_dir, mode_dir)
        if os.path.isdir(mode_path) and os.listdir(mode_path):
            available_modes.append((label, mode_dir, is_nested))

    def _check_availability(folder_name):
        """Fast file-existence check. Returns markdown with bold/strikethrough."""
        info = sample_map.get(folder_name, {})
        parts = []
        for label, mode_dir, is_nested in available_modes:
            exists = _check_scene_exists(evals_dir, mode_dir, is_nested, folder_name, info)
            if exists:
                parts.append(f"**{label}**")
            else:
                parts.append(f"~~{label}~~")
        return " | ".join(parts)

    def _load_scenes(folder_name):
        """Load scene GLBs + images only (fast). Returns: [erp, cubemap, persp, scene0..N, avail, info]."""
        info = sample_map.get(folder_name, {})
        uuid = info.get("uuid", "")
        room_name = info.get("room_name", "")
        view_idx = info.get("view_idx", 0)
        outputs = []

        # Images with descriptive names
        outputs.append(_named_image_copy(info.get("erp_image", ""), uuid, room_name, view_idx, "erp"))
        outputs.append(get_cubemap_concat(info))
        outputs.append(_named_image_copy(info.get("perspective_image", ""), uuid, room_name, view_idx, "perspective"))

        # Scene GLBs only
        for label, mode_dir, is_nested in available_modes:
            glb = get_glb(evals_dir, mode_dir, is_nested, folder_name, info, "scene", label)
            outputs.append(glb)

        # Availability text
        outputs.append(_check_availability(folder_name))

        # Info text
        meta_parts = []
        for label, mode_dir, is_nested in available_modes:
            meta = load_metadata(evals_dir, mode_dir, folder_name)
            if meta:
                bbox = meta.get("bbox_mode", "?")
                noise = meta.get("noise_mode", "?")
                alpha = meta.get("sdedit_alpha", None)
                n_txt = f"{noise}" + (f" (a={alpha})" if alpha is not None else "")
                meta_parts.append(f"**{label}**: bbox={bbox}, noise={n_txt}")

        idx_in_list = folder_names.index(folder_name) if folder_name in folder_names else 0
        info_md = f"### [{idx_in_list + 1}/{n_samples}] {folder_name}\n"
        if uuid:
            info_md += f"UUID: `{uuid}`  |  View: {view_idx}\n\n"
        if meta_parts:
            info_md += "\n".join(meta_parts)
        outputs.append(info_md)
        return outputs

    def _load_detail(folder_name, mode_idx):
        """Load layout + assets for a specific mode. Returns: [layout, assets]."""
        info = sample_map.get(folder_name, {})
        label, mode_dir, is_nested = available_modes[mode_idx]
        layout = get_glb(evals_dir, mode_dir, is_nested, folder_name, info, "layout", label)
        assets = get_glb(evals_dir, mode_dir, is_nested, folder_name, info, "assets", label)
        return [layout, assets]

    def on_select(folder_name):
        return _load_scenes(folder_name)

    def on_prev(folder_name):
        idx = folder_names.index(folder_name) if folder_name in folder_names else 0
        idx = (idx - 1) % n_samples
        new_folder = folder_names[idx]
        return [gr.update(value=new_folder)] + _load_scenes(new_folder)

    def on_next(folder_name):
        idx = folder_names.index(folder_name) if folder_name in folder_names else 0
        idx = (idx + 1) % n_samples
        new_folder = folder_names[idx]
        return [gr.update(value=new_folder)] + _load_scenes(new_folder)

    def on_search(query):
        query = query.strip().lower()
        if not query:
            return gr.update(choices=folder_names, value=folder_names[0] if folder_names else None)
        matched = [f for f in folder_names if query in f.lower()]
        if matched:
            return gr.update(choices=matched, value=matched[0])
        return gr.update(choices=folder_names, value=folder_names[0] if folder_names else None)

    # ---- Build UI ----
    title = f"InSpace Batch Viewer — {n_samples} samples, {len(available_modes)} modes"

    with gr.Blocks(title="InSpace Batch Viewer", fill_width=True) as demo:
        gr.Markdown(f"# {title}")

        # --- Navigation ---
        with gr.Row():
            search_box = gr.Textbox(label="Search", placeholder="room name or index...", scale=2)
            sample_dd = gr.Dropdown(
                choices=folder_names,
                value=folder_names[0] if folder_names else None,
                label="Sample",
                scale=4,
            )
            prev_btn = gr.Button("< Prev", size="sm", scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
            next_btn = gr.Button("Next >", size="sm", scale=1)

        info_box = gr.Markdown("")

        # --- Input Images ---
        with gr.Accordion("Input Images", open=True):
            with gr.Row():
                erp_img = gr.Image(label="ERP Panorama", height=250)
                cubemap_img = gr.Image(label="Cubemap (6 faces)", height=250)
                persp_img = gr.Image(label="Perspective", height=250)

        # --- Scene Comparison ---
        with gr.Row():
            gr.Markdown("### Scene Comparison")
            avail_box = gr.Markdown("")
        scene_viewers = []
        enlarge_btns = []
        with gr.Row():
            for label, _, _ in available_modes:
                with gr.Column(scale=1, min_width=160):
                    sv = gr.Model3D(label=label, height=480)
                    eb = gr.Button("Enlarge", size="sm")
                    scene_viewers.append(sv)
                    enlarge_btns.append(eb)

        # --- Enlarged View ---
        with gr.Accordion("Enlarged View", open=False) as enlarge_accordion:
            enlarge_label = gr.Markdown("*Select a model to enlarge*")
            enlarge_viewer = gr.Model3D(label="Enlarged", height=700)

        # --- Per-mode detail accordions (layout + assets, lazy load) ---
        detail_viewers = {}
        detail_btns = {}
        for mode_idx, (label, _, _) in enumerate(available_modes):
            with gr.Accordion(f"{label} — Layout & Assets", open=False):
                with gr.Row():
                    lv = gr.Model3D(label=f"{label} Layout", height=380)
                    av = gr.Model3D(label=f"{label} Assets", height=380)
                    detail_btn = gr.Button("Load Detail", size="sm", scale=0)
                detail_viewers[mode_idx] = (lv, av)
                detail_btns[mode_idx] = detail_btn

        # --- Wire outputs ---
        scene_outputs = [erp_img, cubemap_img, persp_img] + scene_viewers + [avail_box, info_box]
        nav_outputs = [sample_dd] + scene_outputs

        # --- Events ---
        search_box.submit(on_search, [search_box], [sample_dd])
        search_box.change(on_search, [search_box], [sample_dd])
        load_btn.click(on_select, [sample_dd], scene_outputs)
        sample_dd.change(on_select, [sample_dd], scene_outputs)
        prev_btn.click(on_prev, [sample_dd], nav_outputs)
        next_btn.click(on_next, [sample_dd], nav_outputs)

        # --- Enlarge events ---
        enlarge_outputs = [enlarge_accordion, enlarge_label, enlarge_viewer]
        for midx, (label, _, _) in enumerate(available_modes):
            enlarge_btns[midx].click(
                lambda glb, lbl=label: [
                    gr.update(open=True),
                    f"**{lbl}**",
                    glb,
                ],
                [scene_viewers[midx]],
                enlarge_outputs,
            )

        # --- Detail button events ---
        for mode_idx in range(len(available_modes)):
            lv, av = detail_viewers[mode_idx]
            btn = detail_btns[mode_idx]
            btn.click(
                lambda fn, mi=mode_idx: _load_detail(fn, mi),
                [sample_dd],
                [lv, av],
            )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InSpace Batch Results Viewer")
    parser.add_argument("--evals_dir", type=str, default=EVALS_DIR)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    demo = create_demo(args.evals_dir)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
