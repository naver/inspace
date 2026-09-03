# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
InSpace Gradio Interactive Demo

Interactive per-sample inference for the InSpace indoor scene generation pipeline.
Stages:
    1. Indoor Scene Layout Estimation (PSG from DA2 depth)
    2. Coarse Scene Geometry Generation (SS flow -> 64^3 voxel)
    3. 3D BBOX Estimator (GT or CenterPoint prediction)
    4. Layout and Asset-Aware Scene Generation (Shape + Texture -> Mesh)

Usage:
    python demo/app_inspace_erp_front.py --port 7860
"""

import os
import sys
import time
import argparse
import traceback

# Select the GPU before importing torch (must be set before the first torch import).
# Defaults to GPU 1; override from the shell, e.g.
#   CUDA_VISIBLE_DEVICES=0 python demo/app_inspace_erp_front.py
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
print(f"[DEBUG] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Set environment
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)  # Flush after every line

import gradio as gr
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from demo.app_inspace_utils import (
    DEFAULT_DATA_DIR,
    ensure_demo_samples,
    colorize_depth,
    discover_samples,
    load_psg_data,
    load_cubemap_images,
    load_camera_center,
    load_gt_voxel_64,
    run_stage1_single,
    run_bbox_gt_single,
    run_bbox_predicted_single,
    run_stage2_shape_single,
    run_stage2_texture_single,
    decode_meshes_single,
    save_all_results,
    model_manager,
    log,
)
from demo.app_inspace_viz import (
    create_psg_plotly_figure,
    create_voxel_glb,
    create_csg_comparison_glb,
    create_bbox_with_voxel_glb,
    create_scene_glb,
    create_layout_glb,
    create_exploded_glb,
    create_cubemap_grid_image,
)


# ============================================================
# Dataset Discovery
# ============================================================

SAMPLES = []  # List of (scene_id, room_id)
SCENE_IDS = []
SCENE_ROOMS = {}  # scene_id -> [room_ids]
VIEW_INDICES = {}  # (scene_id, room_id) -> [view_indices]

def init_dataset(data_dir):
    """Initialize dataset sample list."""
    global SAMPLES, SCENE_IDS, SCENE_ROOMS, VIEW_INDICES
    SAMPLES = discover_samples(data_dir)
    SCENE_ROOMS = {}
    for scene_id, room_id in SAMPLES:
        if scene_id not in SCENE_ROOMS:
            SCENE_ROOMS[scene_id] = []
        SCENE_ROOMS[scene_id].append(room_id)
    SCENE_IDS = sorted(SCENE_ROOMS.keys())

    # Discover view indices per sample
    for scene_id, room_id in SAMPLES:
        views = set()
        
        # Check cubic_fov_120 directory for views
        cubic_base = os.path.join(data_dir, scene_id, room_id, 'cubic_fov_120')
        if os.path.isdir(cubic_base):
            for v in sorted(os.listdir(cubic_base)):
                if os.path.isdir(os.path.join(cubic_base, v)):
                    views.add(v)
        
        # Check erp directory for views (from *_colors.png files)
        erp_dir = os.path.join(data_dir, scene_id, room_id, 'erp')
        if os.path.isdir(erp_dir):
            for f in os.listdir(erp_dir):
                if f.endswith('_colors.png'):
                    # Extract view index (e.g., "0002_colors.png" -> "0002")
                    view_idx = f.split('_')[0]
                    views.add(view_idx)
        
        VIEW_INDICES[(scene_id, room_id)] = sorted(list(views))


# ============================================================
# Event Handlers
# ============================================================

def on_scene_change(scene_id, data_dir):
    """Update room dropdown when scene changes."""
    rooms = SCENE_ROOMS.get(scene_id, [])
    if rooms:
        return gr.update(choices=rooms, value=rooms[0])
    return gr.update(choices=[], value=None)


def on_room_change(scene_id, room_id, data_dir):
    """Update view dropdown when room changes."""
    views = VIEW_INDICES.get((scene_id, room_id), ['0000'])
    return gr.update(choices=views, value=views[0])


def on_load_input(scene_id, room_id, view_idx_str, data_dir):
    """Load and display input ERP image and cubemap grid."""
    try:
        view_idx = int(view_idx_str)
        cubemap_dir = os.path.join(data_dir, scene_id, room_id, 'cubic_fov_120', f'{view_idx:04d}')
        cubemap_grid = create_cubemap_grid_image(cubemap_dir)

        # Load ERP panorama + depth visualization
        erp_dir = os.path.join(data_dir, scene_id, room_id, 'erp')
        erp_img = None
        depth_img = None
        if os.path.isdir(erp_dir):
            erp_path = os.path.join(erp_dir, f'{view_idx:04d}_colors.png')
            if os.path.exists(erp_path):
                erp_img = Image.open(erp_path).convert('RGB')
            # depth: raw npy (colormapped) if present, else pre-rendered vis png
            npy_path = os.path.join(erp_dir, f'{view_idx:04d}_depth_da2.npy')
            vis_path = os.path.join(erp_dir, f'{view_idx:04d}_depth_vis_da2.png')
            if os.path.exists(npy_path):
                depth_img = colorize_depth(np.load(npy_path))
            elif os.path.exists(vis_path):
                depth_img = Image.open(vis_path).convert('RGB')

        status = f"Loaded: {scene_id}/{room_id} (view {view_idx})"
        return erp_img, cubemap_grid, depth_img, status
    except Exception as e:
        return None, None, None, f"Error loading input: {e}"


def on_estimate_layout(scene_id, room_id, view_idx_str, data_dir,
                       show_camera, show_coords, state):
    """Stage 1: Indoor Scene Layout Estimation - Load PSG data."""
    try:
        view_idx = int(view_idx_str)
        psg_data = load_psg_data(data_dir, scene_id, room_id, view_idx)

        if psg_data is None:
            return None, state, "PSG data not available for this sample."

        fig = create_psg_plotly_figure(
            psg_data['points'],
            psg_data['colors'],
            psg_data['camera_center'],
            show_camera_center=show_camera,
            show_coordinates=show_coords,
        )

        state = dict(state) if state else {}
        state['scene_id'] = scene_id
        state['room_id'] = room_id
        state['view_idx'] = view_idx
        state['psg_points'] = psg_data['points']
        state['psg_colors'] = psg_data['colors']
        state['camera_center'] = psg_data['camera_center']
        state['psg_ss_latent'] = psg_data['psg_ss_latent']

        n_points = len(psg_data['points'])
        cc = psg_data['camera_center']
        status = f"PSG loaded: {n_points} points\nCamera center: [{cc[0]:.3f}, {cc[1]:.3f}, {cc[2]:.3f}]"
        return fig, state, status

    except Exception as e:
        traceback.print_exc()
        return None, state, f"Error in layout estimation: {e}"


def on_generate_csg(scene_id, room_id, view_idx_str, data_dir,
                    use_psg, alpha, seed, steps, cfg_strength,
                    show_camera, csg_color_mode, state):
    """Stage 2: Coarse Scene Geometry Generation. Returns (pred_glb, gt_glb, state, status)."""
    try:
        t_start = time.time()
        log(f"\n{'='*60}")
        log(f"=== Stage 2: Coarse Scene Geometry (CSG) ===")
        log(f"{'='*60}")
        view_idx = int(view_idx_str)
        state = dict(state) if state else {}

        psg_ss_latent = state.get('psg_ss_latent') if use_psg else None

        result = run_stage1_single(
            data_dir, scene_id, room_id, view_idx,
            use_psg=use_psg, alpha=alpha,
            psg_ss_latent=psg_ss_latent,
            steps=steps, cfg_strength=cfg_strength, seed=seed,
        )

        state['scene_id'] = scene_id
        state['room_id'] = room_id
        state['view_idx'] = view_idx
        state['ss_latent'] = result['ss_latent']
        state['voxel_64'] = result['voxel_64']
        state['encoded_cond'] = result['encoded_cond']
        state['camera_center'] = result['camera_center']
        state['show_camera'] = show_camera

        camera_center = result['camera_center'] if show_camera else None

        # Load GT voxel for comparison
        t1 = time.time()
        log("[CSG] Loading GT voxel for comparison...")
        gt_voxel_64 = load_gt_voxel_64(data_dir, scene_id, room_id)
        state['gt_voxel_64'] = gt_voxel_64
        log(f"[CSG] GT voxel loaded ({time.time()-t1:.1f}s)")

        # Create comparison GLBs
        t1 = time.time()
        log("[CSG] Creating visualization GLBs...")
        pred_glb, gt_glb = create_csg_comparison_glb(
            result['voxel_64'], gt_voxel_64,
            camera_center=camera_center,
            color_mode=csg_color_mode,
        )
        log(f"[CSG] GLBs created ({time.time()-t1:.1f}s)")

        n_active = int(result['voxel_64'].sum())
        status = f"CSG generated: {n_active} active voxels (64^3 grid)"
        if use_psg:
            status += f", SDEdit alpha={alpha}"
        if gt_voxel_64 is not None:
            n_gt = int(gt_voxel_64.sum())
            status += f"\nGT CSG: {n_gt} active voxels"

        log(f"[CSG] Total time: {time.time()-t_start:.1f}s")
        return pred_glb, gt_glb, state, status

    except Exception as e:
        traceback.print_exc()
        return None, None, state, f"Error in CSG generation: {e}"


def on_load_gt_bbox(data_dir, state):
    """Stage 3: Load GT 3D Bounding Boxes. Returns (gt_glb, pred_glb, state, status, bbox_source)."""
    try:
        state = dict(state) if state else {}
        scene_id = state.get('scene_id', '')
        room_id = state.get('room_id', '')
        view_idx = state.get('view_idx', 0)
        voxel_64 = state.get('voxel_64')
        camera_center = state.get('camera_center') if state.get('show_camera', True) else None

        if not scene_id or not room_id:
            return None, gr.skip(), state, "Please select a sample first (Browse tab).", "No BBox"

        log(f"[Stage 3] Loading GT BBox for {scene_id}/{room_id} view={view_idx}")

        bbox_result = run_bbox_gt_single(data_dir, scene_id, room_id, view_idx)
        if bbox_result is None:
            return None, gr.skip(), state, "No GT bounding boxes available for this sample.", "No BBox"

        state['obbs'] = bbox_result['obbs']
        state['asset_names'] = bbox_result['asset_names']
        state['asset_filenames'] = bbox_result['asset_filenames']
        state['bbox_source'] = 'gt'
        state['gt_obbs'] = bbox_result['obbs']  # Store GT separately for comparison

        # Create GT bbox GLB (left panel)
        gt_glb = create_bbox_with_voxel_glb(
            bbox_result['obbs'], voxel_64, camera_center,
            asset_names=bbox_result['asset_names'],
        )

        n_obbs = len(bbox_result['obbs'])
        status = f"GT BBox loaded: {n_obbs} objects (selected for Stage 4)"
        names = ', '.join(bbox_result['asset_names'][:5])
        if n_obbs > 5:
            names += f", ... (+{n_obbs-5} more)"
        status += f"\nAssets: {names}"

        bbox_source_text = f"GT BBox ({n_obbs} objects)"

        # Keep existing predicted bbox viewer unchanged
        return gt_glb, gr.skip(), state, status, bbox_source_text

    except Exception as e:
        traceback.print_exc()
        return None, gr.skip(), state, f"Error loading GT bbox: {e}", "Error"


def on_predict_bbox(data_dir, bbox_threshold, state):
    """Stage 3: Predict 3D Bounding Boxes. Returns (pred_glb, gt_glb, state, status, bbox_source)."""
    try:
        state = dict(state) if state else {}
        scene_id = state.get('scene_id', '')
        room_id = state.get('room_id', '')
        view_idx = state.get('view_idx', 0)
        voxel_64 = state.get('voxel_64')
        camera_center = state.get('camera_center') if state.get('show_camera', True) else None

        if not scene_id or not room_id:
            return gr.skip(), gr.skip(), state, "Please select a sample first (Browse tab).", "No BBox"

        if voxel_64 is None:
            return gr.skip(), gr.skip(), state, "Please generate CSG first (Stage 2).", "No BBox"

        log(f"[Stage 3] Predicting BBox for {scene_id}/{room_id} (threshold={bbox_threshold})")

        bbox_result = run_bbox_predicted_single(voxel_64, score_threshold=bbox_threshold)

        state['obbs'] = bbox_result['obbs']
        state['asset_names'] = bbox_result['asset_names']
        state['asset_filenames'] = bbox_result['asset_filenames']
        state['bbox_source'] = 'predicted'

        # Create predicted bbox GLB (right panel)
        pred_glb = create_bbox_with_voxel_glb(
            bbox_result['obbs'], voxel_64, camera_center,
        )

        # Also show GT for comparison (if available)
        gt_glb = None
        gt_obbs = state.get('gt_obbs')
        if gt_obbs is None:
            # Try loading GT bbox
            try:
                gt_result = run_bbox_gt_single(data_dir, scene_id, room_id, view_idx)
                if gt_result is not None:
                    state['gt_obbs'] = gt_result['obbs']
                    gt_glb = create_bbox_with_voxel_glb(
                        gt_result['obbs'], voxel_64, camera_center,
                        asset_names=gt_result['asset_names'],
                    )
            except Exception:
                pass
        else:
            gt_glb = create_bbox_with_voxel_glb(
                gt_obbs, voxel_64, camera_center,
            )

        n_obbs = len(bbox_result['obbs'])
        confs = bbox_result['confidences']
        status = f"Predicted BBox: {n_obbs} objects (selected for Stage 4)"
        if n_obbs > 0:
            status += f" (conf: {confs.min():.2f}-{confs.max():.2f})"

        bbox_source_text = f"Predicted BBox ({n_obbs} objects)"

        return pred_glb, gt_glb, state, status, bbox_source_text

    except Exception as e:
        traceback.print_exc()
        return gr.skip(), gr.skip(), state, f"Error predicting bbox: {e}", "Error"


def on_generate_scene(data_dir, seed, steps, cfg_strength, gen_texture, layout_mode, state):
    """Stage 4: Generate shape (+ optional texture) -> decode to mesh.

    Runs shape generation, then optionally texture generation, then decodes.
    Returns (combined_glb, exploded_glb, state, status).
    """
    try:
        log(f"\n{'='*60}")
        log(f"=== Stage 4: Scene Generation (Shape{' + Texture' if gen_texture else ''}, layout={layout_mode}) ===")
        log(f"{'='*60}")
        t_start = time.time()
        state = dict(state) if state else {}
        scene_id = state.get('scene_id', '')
        room_id = state.get('room_id', '')
        view_idx = state.get('view_idx', 0)
        voxel_64 = state.get('voxel_64')
        obbs = state.get('obbs')
        camera_center = state.get('camera_center')
        encoded_cond = state.get('encoded_cond')

        if voxel_64 is None or obbs is None:
            return None, None, state, "Please generate CSG and estimate bboxes first."

        # NOTE: 
        # === Step 1: Shape Generation ===
        log(f"\n--- Step 1/{'3' if gen_texture else '2'}: Shape Generation ---")
        t1 = time.time()
        shape_result = run_stage2_shape_single(
            data_dir, scene_id, room_id, view_idx,
            voxel_64, obbs, camera_center, encoded_cond,
            steps=steps, cfg_strength=cfg_strength, seed=seed,
            layout_mode=layout_mode,
        )

        if shape_result is None:
            return None, None, state, "Shape generation failed (no active voxels)."

        state['shape_coords'] = shape_result['shape_coords']
        state['shape_feats'] = shape_result['shape_feats']
        state['part_layouts'] = shape_result['part_layouts']
        state['has_layout'] = shape_result.get('has_layout', False)
        has_layout = state['has_layout']
        # Use filtered OBBs (only those with voxels) for texture stage
        obbs_for_texture = shape_result.get('obbs_filtered', obbs)

        n_total = shape_result['shape_coords'].shape[0]
        n_parts = len(shape_result['part_layouts'])
        if has_layout:
            n_assets = n_parts - 2
            parts_desc = f"1 overall + 1 layout + {n_assets} assets"
        else:
            n_assets = n_parts - 1
            parts_desc = f"1 overall + {n_assets} assets"
        log(f"[Shape] Done ({time.time()-t1:.1f}s): {n_total} voxels, {parts_desc}")

        # === Step 2 (optional): Texture Generation ===
        tex_feats = None
        if gen_texture:
            log(f"\n--- Step 2/3: Texture Generation ---")
            t2 = time.time()
            tex_result = run_stage2_texture_single(
                data_dir, scene_id, room_id, view_idx,
                shape_result['shape_coords'], shape_result['shape_feats'],
                shape_result['part_layouts'],
                obbs_for_texture, camera_center, has_layout=has_layout,
                steps=steps, cfg_strength=cfg_strength, seed=seed,
            )
            tex_feats = tex_result['tex_feats']
            state['tex_feats'] = tex_feats
            state['has_texture'] = True
            log(f"[Texture] Done ({time.time()-t2:.1f}s)")
        else:
            state['has_texture'] = False

        # === Step 3: Decode Meshes ===
        step_n = "3/3" if gen_texture else "2/2"
        log(f"\n--- Step {step_n}: Decode Meshes ---")
        t3 = time.time()
        asset_names = state.get('asset_names', [])
        meshes_list, trellis_rep_data = decode_meshes_single(
            shape_result['shape_coords'], shape_result['shape_feats'],
            shape_result['part_layouts'],
            tex_feats=tex_feats, asset_names=asset_names,
            has_layout=has_layout,
        )
        state['meshes_list'] = meshes_list
        state['trellis_rep_data'] = trellis_rep_data
        log(f"[Decode] Done ({time.time()-t3:.1f}s)")

        # === Create GLBs ===
        log("[Scene] Creating combined + layout + exploded GLBs...")
        combined_glb = create_scene_glb(meshes_list, asset_names)
        layout_glb = create_layout_glb(meshes_list, has_layout=has_layout)
        exploded_glb = create_exploded_glb(
            meshes_list, asset_names, explosion_scale=0.0, has_layout=has_layout)

        n_valid = sum(1 for m in meshes_list if m is not None)
        tex_label = "with texture" if gen_texture else "shape only"
        status = f"Scene generated: {n_total} voxels, {n_parts} parts ({parts_desc})"
        status += f"\nDecoded {n_valid}/{len(meshes_list)} meshes ({tex_label})"
        status += f"\nTotal time: {time.time()-t_start:.1f}s"
        log(f"[Scene] Complete: {n_valid}/{len(meshes_list)} meshes ({tex_label}, {time.time()-t_start:.1f}s)")

        return combined_glb, layout_glb, exploded_glb, state, status

    except Exception as e:
        traceback.print_exc()
        return None, None, None, state, f"Error in scene generation: {e}"


def on_update_explode(explosion_scale, state):
    """Update exploded view with new scale."""
    state = dict(state) if state else {}
    meshes_list = state.get('meshes_list')
    asset_names = state.get('asset_names', [])
    has_layout = state.get('has_layout', False)

    if meshes_list is None:
        return None

    return create_exploded_glb(meshes_list, asset_names, explosion_scale, has_layout=has_layout)


def on_save_all(data_dir, state):
    """Save all visualization results to vis/ and meshes/ folders."""
    try:
        state = dict(state) if state else {}
        scene_id = state.get('scene_id', '')
        room_id = state.get('room_id', '')
        meshes_list = state.get('meshes_list')

        if not scene_id or not room_id:
            return None, None, None, None, None, None, "Please select a sample first.", ""

        output_dir = os.path.join(
            PROJECT_ROOT, 'demo_outputs', 'erp_front', scene_id, room_id
        )

        saved = save_all_results(output_dir, state, meshes_list, data_dir)

        # Load preview images for the Save tab gallery
        def _load(key):
            return Image.open(saved[key]) if key in saved else None

        cubemap_img = _load('cubemap_input')
        bbox_img = _load('bbox_topdown')
        ss_ext_img = _load('ss_exterior')
        ss_int_img = _load('ss_interior')
        geom_ext_img = _load('geometry_exterior')
        geom_td_img = _load('geometry_topdown_cam')
        tex_ext_img = _load('texture_exterior')
        tex_int_img = _load('texture_interior')
        tex_td_img = _load('texture_topdown_cam')

        n_vis = sum(1 for k, v in saved.items() if isinstance(v, str) and v.endswith('.png'))
        status = f"Saved {n_vis} images to: {os.path.join(output_dir, 'vis/')}"
        if 'scene_obj' in saved:
            status += f"\nMeshes saved to: {os.path.join(output_dir, 'meshes/')}"

        return (cubemap_img, bbox_img, ss_ext_img, ss_int_img,
                geom_ext_img, geom_td_img, tex_ext_img, tex_int_img, tex_td_img,
                status, output_dir)

    except Exception as e:
        traceback.print_exc()
        return (None, None, None, None, None, None, None, None, None,
                f"Error saving: {e}", "")


# ============================================================
# Gradio UI
# ============================================================

# Theme + CSS (in gradio 6 these are passed to demo.launch(), not gr.Blocks())
INSPACE_THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
INSPACE_CSS = """
.gradio-container {max-width: 1500px !important; margin: auto !important;}
.inspace-header {text-align:center; padding: 14px 0 8px;}
.inspace-logo {height: 75px; margin-bottom: 6px;}
.inspace-title {font-size: 1.5rem; font-weight: 800; letter-spacing:-0.01em;}
.inspace-sub {font-weight: 500; font-size: 1.05rem; color: var(--body-text-color-subdued);}
.inspace-tag {color: var(--body-text-color-subdued); margin: 4px 0 8px;}
.inspace-badges img {display:inline; height: 20px; margin: 0 2px;}
.tabitem {border-radius: 10px;}
footer {display: none !important;}
"""


def create_demo(data_dir=DEFAULT_DATA_DIR):
    """Create the Gradio demo interface."""
    init_dataset(data_dir)

    # ---- branded header (base64 logo + title + badges) ----
    def _logo_data_uri():
        import base64
        p = os.path.join(PROJECT_ROOT, 'figures', 'inspace_logo.png')
        try:
            with open(p, 'rb') as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode()
        except Exception:
            return ""

    _BADGES = (
        '<a href="#"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a> '
        '<a href="https://kookie12.github.io/InSpace-Project-Page/"><img src="https://img.shields.io/badge/Project-Website-blue"></a> '
        '<a href="https://huggingface.co/GwanHyeong/InSpace"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow"></a> '
        '<a href="https://huggingface.co/datasets/GwanHyeong/ERP-FRONT-30K"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-orange"></a>'
    )
    HEADER_HTML = f"""
<div class="inspace-header">
  <img class="inspace-logo" src="{_logo_data_uri()}" alt="InSpace"/>
  <div class="inspace-title">InSpace<span class="inspace-sub"> · Structure-Aware 3D Indoor Scene Generation from a Single 360° Image</span></div>
  <div class="inspace-tag">Interactive ERP-FRONT demo &nbsp;·&nbsp; ECCV 2026</div>
  <div class="inspace-badges">{_BADGES}</div>
</div>
"""

    with gr.Blocks(title="InSpace ERP-FRONT Demo") as demo:
        gr.HTML(HEADER_HTML)

        state = gr.State({})

        with gr.Row():
            # ============ LEFT PANEL ============
            with gr.Column(scale=1):

                # --- Input ---
                gr.Markdown("### Input")
                with gr.Tab("Browse Dataset"):
                    scene_dd = gr.Dropdown(
                        choices=SCENE_IDS,
                        value=SCENE_IDS[0] if SCENE_IDS else None,
                        label="Scene (House ID)",
                    )
                    room_dd = gr.Dropdown(
                        choices=SCENE_ROOMS.get(SCENE_IDS[0], []) if SCENE_IDS else [],
                        value=SCENE_ROOMS.get(SCENE_IDS[0], [None])[0] if SCENE_IDS else None,
                        label="Room",
                    )
                    # Get initial views for the first scene/room
                    initial_scene = SCENE_IDS[0] if SCENE_IDS else None
                    initial_room = SCENE_ROOMS.get(initial_scene, [None])[0] if initial_scene else None
                    initial_views = VIEW_INDICES.get((initial_scene, initial_room), ['0000']) if initial_scene and initial_room else ['0000']
                    view_dd = gr.Dropdown(
                        choices=initial_views,
                        value=initial_views[0],
                        label="View Index",
                    )
                    load_btn = gr.Button("Load Input", variant="primary")

                # --- Stage 1: Indoor Scene Layout Estimation ---
                gr.Markdown("---")
                gr.Markdown("### Stage 1: Indoor Scene Layout Estimation")
                gr.Markdown("*Estimate Partial Scene Geometry (PSG) from DA2 depth*")
                with gr.Row():
                    psg_show_camera_cb = gr.Checkbox(label="Camera Center", value=True)
                    psg_show_coords_cb = gr.Checkbox(label="Coordinates", value=True)
                estimate_layout_btn = gr.Button("Estimate Layout (PSG)", variant="primary")

                # --- Stage 2: Coarse Scene Geometry Generation ---
                gr.Markdown("---")
                gr.Markdown("### Stage 2: Coarse Scene Geometry Generation")
                gr.Markdown("*SS flow model -> SS latent -> 64^3 voxel grid (CSG)*")
                use_psg_cb = gr.Checkbox(label="Use PSG (SDEdit)", value=True,
                                         info="Use PSG as initial noise via SDEdit")
                alpha_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Alpha (noise level)",
                                         info="0=clean PSG, 1=pure noise")
                csg_show_camera_cb = gr.Checkbox(label="Show Camera Center", value=True)
                csg_color_radio = gr.Radio(
                    choices=["ccm", "gray"],
                    value="ccm",
                    label="Voxel Color",
                    info="CCM: position-based RGB, Gray: uniform gray",
                )
                generate_csg_btn = gr.Button("Generate CSG", variant="primary")

                # --- Stage 3: 3D BBOX Estimator ---
                gr.Markdown("---")
                gr.Markdown("### Stage 3: 3D BBOX Estimator")
                gr.Markdown("*GT or CenterPoint-predicted 3D bounding boxes*")
                bbox_source_display = gr.Textbox(
                    label="Selected BBox Source (for Stage 4)",
                    value="None (select GT or Predicted)",
                    interactive=False,
                    max_lines=1,
                )
                with gr.Row():
                    gt_bbox_btn = gr.Button("Load GT BBox", variant="primary")
                    pred_bbox_btn = gr.Button("Predict BBox", variant="secondary")
                bbox_threshold = gr.Slider(0.1, 0.9, value=0.3, step=0.05,
                                            label="Score Threshold",
                                            info="For predicted bboxes")

                # --- Stage 4: Layout and Asset-Aware Scene Generation ---
                gr.Markdown("---")
                gr.Markdown("### Stage 4: Layout and Asset-Aware Scene Generation")
                gr.Markdown("*Shape + Texture generation -> Mesh decoding*")
                gen_texture_cb = gr.Checkbox(
                    label="Generate Texture", value=True,
                    info="Include texture generation (slower but produces colored mesh)")
                layout_mode_radio = gr.Radio(
                    choices=["floor_perimeter", "floor_perimeter_clean", "no_floor_assets"],
                    value="floor_perimeter",
                    label="Layout & Asset Mode")
                gr.Markdown(
                    "- **floor_perimeter**: Layout=floor+walls. Assets include floor voxels (matches training).\n"
                    "- **floor_perimeter_clean**: Same layout, but assets **exclude floor-layer** voxels. "
                    "Removes floor plane attached to tables/chairs.\n"
                    "- **no_floor_assets**: Alias for floor_perimeter_clean.")
                gen_scene_btn = gr.Button("Generate Scene", variant="primary")

                # --- Advanced ---
                with gr.Accordion("Advanced Settings", open=False):
                    seed_slider = gr.Slider(0, 99999, value=42, step=1, label="Seed")
                    steps_s1 = gr.Slider(4, 50, value=12, step=1, label="Stage 1 Steps (SS)")
                    steps_s2 = gr.Slider(4, 50, value=12, step=1, label="Stage 2 Steps (SLat)")
                    cfg_s1 = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Stage 1 CFG")
                    cfg_s2 = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Stage 2 CFG")

                # --- Status ---
                status_box = gr.Textbox(label="Status", lines=3, interactive=False)

            # ============ RIGHT PANEL ============
            with gr.Column(scale=2):
                with gr.Tabs() as tabs:

                    # Tab: Input
                    with gr.Tab("Input", id="tab_input"):
                        erp_image = gr.Image(label="ERP Panorama", height=200)
                        cubemap_image = gr.Image(label="Cubemap (2x3 grid)", height=400)
                        depth_image = gr.Image(label="ERP Depth Map", height=200)

                    # Tab: Layout (PSG)
                    with gr.Tab("Layout (PSG)", id="tab_layout"):
                        psg_plot = gr.Plot(label="Partial Scene Geometry (PSG)")

                    # Tab: CSG (Predicted vs GT side by side)
                    with gr.Tab("CSG", id="tab_csg"):
                        with gr.Row():
                            csg_pred_viewer = gr.Model3D(label="Predicted CSG (64^3 voxel)", height=500)
                            csg_gt_viewer = gr.Model3D(label="GT CSG (64^3 voxel)", height=500)

                    # Tab: BBox + CSG (Predicted vs GT side by side)
                    with gr.Tab("BBox + CSG", id="tab_bbox"):
                        with gr.Row():
                            bbox_pred_viewer = gr.Model3D(label="Predicted BBox + CSG", height=500)
                            bbox_gt_viewer = gr.Model3D(label="GT BBox + CSG", height=500)

                    # Tab: Mesh
                    with gr.Tab("Mesh", id="tab_mesh"):
                        with gr.Row():
                            combined_viewer = gr.Model3D(label="Overall Scene", height=350)
                            layout_viewer = gr.Model3D(label="Layout (floor+walls)", height=350)
                            exploded_viewer = gr.Model3D(label="Assets (exploded)", height=350)
                        explode_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05,
                                                     label="Explosion Scale")

                    # Tab: Save
                    with gr.Tab("Save", id="tab_save"):
                        gr.Markdown("### Save Visualization Results")
                        gr.Markdown("Saves vis/ folder (rendered images) + meshes/ folder (OBJ files)")
                        save_all_btn = gr.Button("Save All", variant="primary", size="lg")
                        save_dir_box = gr.Textbox(label="Output Directory", interactive=False)
                        with gr.Row():
                            save_cubemap_img = gr.Image(label="Cubemap Input", height=200)
                            save_bbox_img = gr.Image(label="BBox Top-Down", height=200)
                        with gr.Row():
                            save_ss_img = gr.Image(label="SS Exterior", height=200)
                            save_ss_int_img = gr.Image(label="SS Interior", height=200)
                        with gr.Row():
                            save_geom_img = gr.Image(label="Geometry Exterior", height=200)
                            save_geom_td_img = gr.Image(label="Geometry Top-Down", height=200)
                        with gr.Row():
                            save_tex_ext_img = gr.Image(label="Texture Exterior", height=200)
                            save_tex_int_img = gr.Image(label="Texture Interior", height=200)
                            save_tex_td_img = gr.Image(label="Texture Top-Down", height=200)

        # ============================================================
        # Event Bindings
        # ============================================================

        data_dir_state = gr.State(data_dir)

        # Dropdown cascading
        scene_dd.change(on_scene_change, [scene_dd, data_dir_state], [room_dd])
        room_dd.change(on_room_change, [scene_dd, room_dd, data_dir_state], [view_dd])

        # Load input
        load_btn.click(
            on_load_input,
            [scene_dd, room_dd, view_dd, data_dir_state],
            [erp_image, cubemap_image, depth_image, status_box],
        )

        # Stage 1: Estimate Layout (PSG)
        estimate_layout_btn.click(
            on_estimate_layout,
            [scene_dd, room_dd, view_dd, data_dir_state,
             psg_show_camera_cb, psg_show_coords_cb, state],
            [psg_plot, state, status_box],
        )

        # Stage 2: Generate CSG (outputs to both pred and GT viewers)
        generate_csg_btn.click(
            on_generate_csg,
            [scene_dd, room_dd, view_dd, data_dir_state,
             use_psg_cb, alpha_slider, seed_slider, steps_s1, cfg_s1,
             csg_show_camera_cb, csg_color_radio, state],
            [csg_pred_viewer, csg_gt_viewer, state, status_box],
        )

        # Stage 3: BBox (both viewers: GT left, Predicted right)
        gt_bbox_btn.click(
            on_load_gt_bbox,
            [data_dir_state, state],
            [bbox_gt_viewer, bbox_pred_viewer, state, status_box, bbox_source_display],
        )
        pred_bbox_btn.click(
            on_predict_bbox,
            [data_dir_state, bbox_threshold, state],
            [bbox_pred_viewer, bbox_gt_viewer, state, status_box, bbox_source_display],
        )

        # Stage 4: Generate scene (shape + optional texture) -> mesh viewers
        gen_scene_btn.click(
            on_generate_scene,
            [data_dir_state, seed_slider, steps_s2, cfg_s2, gen_texture_cb, layout_mode_radio, state],
            [combined_viewer, layout_viewer, exploded_viewer, state, status_box],
        )

        # Explode slider
        explode_slider.change(
            on_update_explode,
            [explode_slider, state],
            [exploded_viewer],
        )

        # Save
        save_all_btn.click(
            on_save_all,
            [data_dir_state, state],
            [save_cubemap_img, save_bbox_img, save_ss_img, save_ss_int_img,
             save_geom_img, save_geom_td_img, save_tex_ext_img, save_tex_int_img,
             save_tex_td_img, status_box, save_dir_box],
        )

    return demo


# ============================================================
# Main
# ============================================================
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'ERP_3D_FRONT_test_samples')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='InSpace Gradio Demo')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Data directory')
    parser.add_argument('--port', type=int, default=None,
                        help='Server port (default: auto-find free port)')
    parser.add_argument('--share', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    ensure_demo_samples(args.data_dir)
    demo = create_demo(args.data_dir)

    launch_kwargs = dict(server_name='0.0.0.0', share=args.share,
                         theme=INSPACE_THEME, css=INSPACE_CSS)
    if args.port is not None:
        launch_kwargs['server_port'] = args.port
    # When port is None, Gradio auto-finds a free port
    demo.launch(**launch_kwargs)


# Stage-2 input modes
# mode                  | Layout                    | Asset floor voxels
# floor_perimeter       | floor + walls (perimeter) | included (matches the training data)
# floor_perimeter_clean | floor + walls (perimeter) | excluded (floor Z layer removed)
# no_floor_assets       | floor + walls (perimeter) | excluded (alias)
#
# With floor_perimeter_clean, the floor plane under tables/chairs is dropped from the assets
# and kept only in the layout.
