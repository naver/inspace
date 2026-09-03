# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Stage 1+2 Pipeline Comparison Visualization

Creates comparison grids across all evaluation methods for paper figures.

Layout per visualization type:
    Rows: GT, Random(GT_SS), Random(Pred_SS), SDEdit0.5(GT_SS), SDEdit0.5(Pred_SS)
    Columns: Multiple samples (configurable batch size)

Visualization types (from existing images):
    SS:       exterior_ccm, exterior_height, topdown_ccm, topdown_height,
              topdown_cam_ccm, topdown_cam_height, interior_ccm, interior_height
    Geometry: exterior, topdown, topdown_cam, interior
    Texture:  exterior, topdown, topdown_cam, interior

Part-aware (rendered from saved .glb meshes, GPU required):
    geometry_parts_topdown, texture_parts_topdown

Output:
    evals/stage12_comparison/
    ├── ss_exterior_ccm/            # Batch grids
    │   ├── batch_000_005.png
    │   └── ...
    ├── parts_topdown/              # Part-aware per-sample
    │   └── {sample_name}/
    │       ├── geometry_parts_topdown.png
    │       └── texture_parts_topdown.png
    └── per_sample/                 # Combined per-sample
        └── {sample_name}/
            ├── ss_exterior_ccm.png
            └── ...

Usage:
    # Standard views only (CPU, fast)
    python eval/viewers/create_stage12_comparison.py --batch_only

    # Standard + part-aware (GPU needed for parts)
    python eval/viewers/create_stage12_comparison.py --render_parts

    # Specific stages/views
    python eval/viewers/create_stage12_comparison.py --stages texture --views exterior topdown

    # Limit samples
    python eval/viewers/create_stage12_comparison.py --max_samples 20 --batch_size 4
"""

import argparse
import os
import sys
import glob

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

METHODS = [
    ("random_gt",           "Random (GT BBox)"),
    ("random_predicted",    "Random (Pred BBox)"),
    ("sdedit0.5_gt",        "SDEdit (GT BBox)"),
    ("sdedit0.5_predicted", "SDEdit (Pred BBox)"),
]

GT_SOURCE_METHOD = "random_gt"

# gt_crop: how to extract GT from vis_concat
#   'left_half'    = side-by-side, GT is left half
#   'ss_interior'  = SS interior 3 content rows, GT is middle row
#   None           = no GT available
VIS_TYPES = {
    "ss_exterior_ccm":       {"stage": "ss",       "view": "exterior",    "gt_crop": "left_half"},
    "ss_exterior_height":    {"stage": "ss",       "view": "exterior",    "gt_crop": "left_half"},
    "ss_topdown_ccm":        {"stage": "ss",       "view": "topdown",     "gt_crop": "left_half"},
    "ss_topdown_height":     {"stage": "ss",       "view": "topdown",     "gt_crop": "left_half"},
    "ss_topdown_cam_ccm":    {"stage": "ss",       "view": "topdown_cam", "gt_crop": "left_half"},
    "ss_topdown_cam_height": {"stage": "ss",       "view": "topdown_cam", "gt_crop": "left_half"},
    "ss_interior_ccm":       {"stage": "ss",       "view": "interior",    "gt_crop": "ss_interior"},
    "ss_interior_height":    {"stage": "ss",       "view": "interior",    "gt_crop": "ss_interior"},
    "geometry_exterior":     {"stage": "geometry",  "view": "exterior",    "gt_crop": "left_half"},
    "geometry_topdown":      {"stage": "geometry",  "view": "topdown",     "gt_crop": "left_half"},
    "geometry_topdown_cam":  {"stage": "geometry",  "view": "topdown_cam", "gt_crop": "left_half"},
    "geometry_interior":     {"stage": "geometry",  "view": "interior",    "gt_crop": None},
    "texture_exterior":      {"stage": "texture",   "view": "exterior",    "gt_crop": "left_half"},
    "texture_topdown":       {"stage": "texture",   "view": "topdown",     "gt_crop": "left_half"},
    "texture_topdown_cam":   {"stage": "texture",   "view": "topdown_cam", "gt_crop": "left_half"},
    "texture_interior":      {"stage": "texture",   "view": "interior",    "gt_crop": None},
}

# vis_concat layout constants
LABEL_TOP_H = 24
LABEL_LEFT_W = 80
INTERIOR_INPUT_H = 512
INTERIOR_GT_H = 512


# ============================================================================
# Font helpers
# ============================================================================

_FONT_CACHE = {}

def get_font(size=14, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    if bold:
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for fp in paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, size)
            _FONT_CACHE[key] = font
            return font
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


# ============================================================================
# Sample discovery
# ============================================================================

def find_samples(pipeline_dir):
    """Find sample paths (uuid/room_name) common across all methods."""
    method_samples = []
    for method_name, _ in METHODS:
        method_dir = os.path.join(pipeline_dir, method_name)
        samples = set()
        if not os.path.isdir(method_dir):
            print(f"WARNING: method dir not found: {method_dir}")
            continue
        for uuid in os.listdir(method_dir):
            uuid_path = os.path.join(method_dir, uuid)
            if not os.path.isdir(uuid_path):
                continue
            for room in os.listdir(uuid_path):
                if os.path.isdir(os.path.join(uuid_path, room)):
                    samples.add(f"{uuid}/{room}")
        method_samples.append(samples)
    common = method_samples[0]
    for s in method_samples[1:]:
        common &= s
    return sorted(common)


def short_name(sample_path):
    parts = sample_path.split("/")
    return f"{parts[0][:8]}/{parts[1]}"


# ============================================================================
# Image loading (existing images)
# ============================================================================

def extract_gt_from_vis_concat(vis_concat_path, gt_crop_type):
    """Extract GT portion from a vis_concat image."""
    if not os.path.exists(vis_concat_path):
        return None
    img = Image.open(vis_concat_path)
    w, h = img.size

    if gt_crop_type == "left_half":
        # Layout: [label_top(24)] + [GT(left) | Pred(right)]
        return img.crop((0, LABEL_TOP_H, w // 2, h))

    elif gt_crop_type == "ss_interior":
        # Layout: cols [label_left(80)] + [content]
        #         rows [face_labels(24)] + [input(512)] + [GT(512)] + [pred(512)]
        gt_y0 = LABEL_TOP_H + INTERIOR_INPUT_H
        gt_y1 = gt_y0 + INTERIOR_GT_H
        return img.crop((LABEL_LEFT_W, gt_y0, w, gt_y1))

    return None


def load_vis_pred(pipeline_dir, method_name, sample, vis_type):
    path = os.path.join(pipeline_dir, method_name, sample, "vis_pred", f"{vis_type}.png")
    if os.path.exists(path):
        return Image.open(path)
    return None


def resize_to_cell(img, cell_w, cell_h):
    """Resize maintaining aspect ratio, center on white canvas."""
    if img is None:
        return create_placeholder(cell_w, cell_h, "Missing")
    orig_w, orig_h = img.size
    scale = min(cell_w / orig_w, cell_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (cell_w, cell_h), (255, 255, 255))
    canvas.paste(resized, ((cell_w - new_w) // 2, (cell_h - new_h) // 2))
    return canvas


def create_placeholder(w, h, text="N/A"):
    img = Image.new("RGB", (w, h), (180, 180, 180))
    draw = ImageDraw.Draw(img)
    font = get_font(18)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2), text, fill=(120, 120, 120), font=font)
    return img


# ============================================================================
# Batch grid creation (existing images)
# ============================================================================

def create_batch_grid(
    pipeline_dir, samples, vis_type, vis_info,
    cell_size=512, interior_cell_w=1536,
    label_col_w=160, header_h=28, sep=2,
):
    """Create comparison grid: rows=methods, columns=samples."""
    n_samples = len(samples)
    is_interior = vis_info["view"] == "interior"

    if is_interior:
        cell_w = interior_cell_w
        cell_h = int(cell_w * 512 / 3072)
    else:
        cell_w = cell_size
        cell_h = cell_size

    row_labels = ["GT"] + [lbl for _, lbl in METHODS]
    n_rows = len(row_labels)

    total_w = label_col_w + n_samples * (cell_w + sep) - sep
    total_h = header_h + n_rows * (cell_h + sep) - sep

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    hdr_font = get_font(11)
    lbl_font = get_font(13, bold=True)

    # Column headers (sample names)
    for ci, sample in enumerate(samples):
        name = short_name(sample)
        x_c = label_col_w + ci * (cell_w + sep) + cell_w // 2
        bb = draw.textbbox((0, 0), name, font=hdr_font)
        draw.text((x_c - (bb[2] - bb[0]) // 2, 4), name, fill=(0, 0, 0), font=hdr_font)

    # Row labels
    for ri, label in enumerate(row_labels):
        y_c = header_h + ri * (cell_h + sep) + cell_h // 2
        bb = draw.textbbox((0, 0), label, font=lbl_font)
        draw.text((8, y_c - (bb[3] - bb[1]) // 2), label, fill=(0, 0, 0), font=lbl_font)

    # Separator lines
    for ri in range(n_rows + 1):
        y = header_h + ri * (cell_h + sep) - sep
        if 0 <= y < total_h:
            draw.rectangle([label_col_w, y, total_w, y + sep - 1], fill=(220, 220, 220))
    for ci in range(n_samples + 1):
        x = label_col_w + ci * (cell_w + sep) - sep
        if 0 <= x < total_w:
            draw.rectangle([x, header_h, x + sep - 1, total_h], fill=(220, 220, 220))

    # Fill cells
    for ci, sample in enumerate(samples):
        x_off = label_col_w + ci * (cell_w + sep)

        # Row 0: GT
        gt_crop = vis_info["gt_crop"]
        if gt_crop is not None:
            vcp = os.path.join(
                pipeline_dir, GT_SOURCE_METHOD, sample,
                "vis_concat", f"{vis_type}.png",
            )
            gt_img = extract_gt_from_vis_concat(vcp, gt_crop)
            gt_cell = resize_to_cell(gt_img, cell_w, cell_h)
        else:
            gt_cell = create_placeholder(cell_w, cell_h, "N/A")
        canvas.paste(gt_cell, (x_off, header_h))

        # Rows 1-4: Methods
        for ri, (mname, _) in enumerate(METHODS, start=1):
            pred = load_vis_pred(pipeline_dir, mname, sample, vis_type)
            canvas.paste(
                resize_to_cell(pred, cell_w, cell_h),
                (x_off, header_h + ri * (cell_h + sep)),
            )

    return canvas


# ============================================================================
# Part-aware rendering from .glb meshes (GPU)
# ============================================================================

_RENDERER_INITIALIZED = False
_MESH_CLS = None
_GET_RENDERER = None
_CAM_FN = None


def _init_renderer():
    """Lazy-initialize trellis2 rendering (only when needed)."""
    global _RENDERER_INITIALIZED, _MESH_CLS, _GET_RENDERER, _CAM_FN
    if _RENDERER_INITIALIZED:
        return
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from trellis2.representations import Mesh
    from trellis2.utils.render_utils import (
        get_renderer,
        yaw_pitch_r_fov_to_extrinsics_intrinsics,
    )
    _MESH_CLS = Mesh
    _GET_RENDERER = get_renderer
    _CAM_FN = yaw_pitch_r_fov_to_extrinsics_intrinsics
    _RENDERER_INITIALIZED = True


def load_glb_as_mesh(glb_path, with_color=False):
    """Load a .glb file and return a trellis2 Mesh on GPU.

    Args:
        glb_path: path to .glb file
        with_color: if True, also extract vertex colors as vertex_attrs [N, 3] float in [0, 1]

    Returns:
        Mesh (or Mesh with vertex_attrs if with_color=True), or None on failure
    """
    import torch
    import trimesh as tm

    if not os.path.exists(glb_path):
        return None

    try:
        mesh = tm.load(glb_path, force="mesh")
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
        faces = torch.tensor(mesh.faces, dtype=torch.int64).cuda()

        vertex_attrs = None
        if with_color:
            try:
                rgba = mesh.visual.to_color().vertex_colors  # [N, 4] uint8
                vertex_attrs = torch.tensor(
                    rgba[:, :3].astype(np.float32) / 255.0, dtype=torch.float32
                ).cuda()
            except Exception:
                # Fallback: gray color
                vertex_attrs = torch.full(
                    (vertices.shape[0], 3), 0.7, dtype=torch.float32
                ).cuda()

        return _MESH_CLS(vertices=vertices, faces=faces, vertex_attrs=vertex_attrs)
    except Exception as e:
        print(f"  WARNING: failed to load {glb_path}: {e}")
        return None


def render_mesh_topdown(mesh, tile_size=256, mode="normal"):
    """Render a trellis2 Mesh from top-down view, return PIL Image.

    Args:
        mesh: trellis2 Mesh (with vertex_attrs if mode="color")
        tile_size: output image size
        mode: "normal" for normal map, "color" for vertex color rendering

    Returns:
        PIL.Image
    """
    import torch

    if mesh is None:
        return create_placeholder(tile_size, tile_size, "Empty")

    renderer = _GET_RENDERER(mesh, resolution=tile_size, ssaa=2)
    exts, ints = _CAM_FN([0], [np.pi / 2], [2], [30])

    if mode == "color" and mesh.vertex_attrs is not None:
        result = renderer.render(mesh, exts[0], ints[0], return_types=["mask", "attr"])
        attr = result["attr"]  # [3, H, W] vertex colors
        mask = result["mask"]  # [H, W]
        # White background where mask is 0
        bg = torch.ones_like(attr)
        mask_3 = mask.unsqueeze(0).expand_as(attr)
        img_t = attr * mask_3 + bg * (1 - mask_3)
        arr = np.clip(img_t.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    else:
        result = renderer.render(mesh, exts[0], ints[0])
        normal = result["normal"]  # [3, H, W]
        arr = np.clip(normal.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def load_parts_from_dir(meshes_dir, with_color=False):
    """Load scene, layout, and asset meshes from a meshes/ directory.

    Args:
        meshes_dir: path to meshes/ directory
        with_color: if True, also extract vertex colors for texture rendering

    Returns:
        list of (label, trellis2_Mesh or None)
    """
    parts = []

    # Scene (overall)
    scene_path = os.path.join(meshes_dir, "scene.glb")
    parts.append(("overall", load_glb_as_mesh(scene_path, with_color=with_color)))

    # Layout
    layout_path = os.path.join(meshes_dir, "layout.glb")
    if os.path.exists(layout_path):
        parts.append(("layout", load_glb_as_mesh(layout_path, with_color=with_color)))

    # Assets (sorted by index)
    assets_dir = os.path.join(meshes_dir, "assets")
    if os.path.isdir(assets_dir):
        asset_files = sorted(glob.glob(os.path.join(assets_dir, "*.glb")))
        for af in asset_files:
            fname = os.path.splitext(os.path.basename(af))[0]
            # Extract short label: e.g. "000_armoire" -> "armoire"
            # or "000_king-size_bed_king-size_bed_7bf721bf_inst001" -> "king-size_bed"
            label_parts = fname.split("_", 1)
            label = label_parts[1] if len(label_parts) > 1 else fname
            # Truncate long labels
            if len(label) > 20:
                label = label[:20]
            parts.append((label, load_glb_as_mesh(af, with_color=with_color)))

    return parts


def render_parts_row(parts_list, tile_size=256, label_height=24, mode="normal"):
    """Render a row of part meshes with labels on top.

    Args:
        parts_list: list of (label, trellis2_Mesh_or_None)
        tile_size: size of each tile
        label_height: height of label strip
        mode: "normal" for normal map, "color" for vertex color rendering

    Returns:
        PIL.Image of the row [label_h + tile_size, n_parts * tile_size]
    """
    n_parts = len(parts_list)
    if n_parts == 0:
        return create_placeholder(tile_size, tile_size + label_height, "No parts")

    row_w = n_parts * tile_size
    row_h = label_height + tile_size
    canvas = Image.new("RGB", (row_w, row_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = get_font(10)

    for i, (label, mesh) in enumerate(parts_list):
        x_off = i * tile_size

        # Label
        bb = draw.textbbox((0, 0), label, font=font)
        tw = bb[2] - bb[0]
        tx = x_off + max(0, (tile_size - tw) // 2)
        draw.text((tx, 2), label, fill=(0, 0, 0), font=font)

        # Rendered tile
        tile = render_mesh_topdown(mesh, tile_size, mode=mode)
        canvas.paste(tile, (x_off, label_height))

    return canvas


def _assemble_parts_grid(rendered_rows, sample, title_prefix, label_col_w=160, header_h=28, sep=2):
    """Assemble rendered part rows into a labeled comparison grid.

    Args:
        rendered_rows: list of PIL.Image (GT + 4 methods)
        sample: sample path string
        title_prefix: e.g. "Geometry Parts" or "Texture Parts"
        label_col_w, header_h, sep: layout params

    Returns:
        PIL.Image
    """
    row_labels = ["GT"] + [lbl for _, lbl in METHODS]
    n_rows = len(row_labels)

    max_row_w = max(r.width for r in rendered_rows)
    row_h = rendered_rows[0].height

    total_w = label_col_w + max_row_w
    total_h = header_h + n_rows * (row_h + sep) - sep

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    lbl_font = get_font(13, bold=True)
    title_font = get_font(14, bold=True)

    # Title
    sample_name = short_name(sample)
    draw.text((8, 4), f"{title_prefix}: {sample_name}", fill=(0, 0, 0), font=title_font)

    # Row labels and content
    for ri, (label, row_img) in enumerate(zip(row_labels, rendered_rows)):
        y_off = header_h + ri * (row_h + sep)

        # Label
        y_c = y_off + row_h // 2
        bb = draw.textbbox((0, 0), label, font=lbl_font)
        draw.text((8, y_c - (bb[3] - bb[1]) // 2), label, fill=(0, 0, 0), font=lbl_font)

        # Row image (pad to max width if needed)
        if row_img.width < max_row_w:
            padded = Image.new("RGB", (max_row_w, row_h), (255, 255, 255))
            padded.paste(row_img, (0, 0))
            row_img = padded
        canvas.paste(row_img, (label_col_w, y_off))

    # Separator lines
    for ri in range(n_rows + 1):
        y = header_h + ri * (row_h + sep) - sep
        if 0 <= y < total_h:
            draw.rectangle([label_col_w, y, total_w, y + sep - 1], fill=(220, 220, 220))

    return canvas


def create_parts_comparison(
    pipeline_dir, gt_recon_dir, sample,
    tile_size=256, label_col_w=160, header_h=28, sep=2, label_height=24,
    modes=None,
):
    """Create part-aware comparison images for one sample (geometry + texture).

    Shows GT + 4 methods, each as a row of per-part top-down renders.

    Args:
        modes: list of "geometry" and/or "texture". None = both.

    Returns:
        dict with keys "geometry" and/or "texture", values are PIL.Image
    """
    _init_renderer()

    all_modes = [
        ("normal", "Geometry Parts", False, "geometry"),
        ("color", "Texture Parts", True, "texture"),
    ]
    if modes is not None:
        all_modes = [m for m in all_modes if m[3] in modes]

    results = {}

    for render_mode, title_prefix, need_color, out_key in all_modes:
        rendered_rows = []

        # GT from gt_recon
        gt_meshes_dir = os.path.join(gt_recon_dir, sample, "meshes")
        if os.path.isdir(gt_meshes_dir):
            gt_parts = load_parts_from_dir(gt_meshes_dir, with_color=need_color)
            gt_row = render_parts_row(gt_parts, tile_size, label_height, mode=render_mode)
        else:
            gt_row = create_placeholder(tile_size * 3, tile_size + label_height, "GT not found")
        rendered_rows.append(gt_row)

        # Each method
        for method_name, _ in METHODS:
            meshes_dir = os.path.join(pipeline_dir, method_name, sample, "meshes")
            if os.path.isdir(meshes_dir):
                parts = load_parts_from_dir(meshes_dir, with_color=need_color)
                row = render_parts_row(parts, tile_size, label_height, mode=render_mode)
            else:
                row = create_placeholder(tile_size * 3, tile_size + label_height, "Missing")
            rendered_rows.append(row)

        grid = _assemble_parts_grid(
            rendered_rows, sample, title_prefix,
            label_col_w=label_col_w, header_h=header_h, sep=sep,
        )
        results[out_key] = grid

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 1+2 Pipeline Comparison")
    parser.add_argument("--pipeline_dir", default="evals/stage12_pipeline")
    parser.add_argument("--gt_recon_dir", default="evals/gt_recon")
    parser.add_argument("--output_dir", default="evals/stage12_comparison")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--cell_size", type=int, default=512)
    parser.add_argument("--interior_cell_w", type=int, default=1536)
    parser.add_argument("--parts_tile_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--per_sample_only", action="store_true",
                        help="Only per-sample comparisons (skip batch grids)")
    parser.add_argument("--batch_only", action="store_true",
                        help="Only batch grids (skip per-sample)")
    parser.add_argument("--render_parts", action="store_true",
                        help="Also render part-aware comparison from .glb meshes (GPU)")
    parser.add_argument("--parts_only", action="store_true",
                        help="Only render part-aware comparisons (GPU)")
    parser.add_argument("--max_parts_samples", type=int, default=None,
                        help="Limit part-aware rendering to first N samples")
    parser.add_argument("--parts_mode", nargs="+", default=None,
                        choices=["geometry", "texture"],
                        help="Which part renders to generate (default: both)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip files that already exist on disk")
    parser.add_argument("--rank", type=int, default=0,
                        help="Worker rank for distributed processing (0-indexed)")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Total number of workers for distributed processing")
    parser.add_argument("--stages", nargs="+", default=None,
                        choices=["ss", "geometry", "texture"])
    parser.add_argument("--views", nargs="+", default=None,
                        choices=["exterior", "topdown", "topdown_cam", "interior"])
    args = parser.parse_args()

    # Now you can run texture-only with:

    # python eval/viewers/create_stage12_comparison.py --parts_only --parts_mode texture
    # Or geometry-only:

    # python eval/viewers/create_stage12_comparison.py --parts_only --parts_mode geometry
    # Or both (default, same as before):

    # python eval/viewers/create_stage12_comparison.py --parts_only

    # Split the work across 4 GPUs/processes
    # CUDA_VISIBLE_DEVICES=1 python eval/viewers/create_stage12_comparison.py --parts_only --parts_mode texture --rank 0 --world_size 4 &
    # CUDA_VISIBLE_DEVICES=1 python eval/viewers/create_stage12_comparison.py --parts_only --parts_mode texture --rank 1 --world_size 4 &
    # CUDA_VISIBLE_DEVICES=2 python eval/viewers/create_stage12_comparison.py --parts_only --parts_mode texture --rank 2 --world_size 4 &
    # CUDA_VISIBLE_DEVICES=2 python eval/viewers/create_stage12_comparison.py --parts_only --parts_mode texture --rank 3 --world_size 4 &


    # Resolve paths
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    pipeline_dir = args.pipeline_dir
    gt_recon_dir = args.gt_recon_dir
    output_dir = args.output_dir
    if not os.path.isabs(pipeline_dir):
        pipeline_dir = os.path.join(base_dir, pipeline_dir)
    if not os.path.isabs(gt_recon_dir):
        gt_recon_dir = os.path.join(base_dir, gt_recon_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    pipeline_dir = os.path.abspath(pipeline_dir)
    gt_recon_dir = os.path.abspath(gt_recon_dir)
    output_dir = os.path.abspath(output_dir)

    print(f"Pipeline dir: {pipeline_dir}")
    print(f"GT recon dir: {gt_recon_dir}")
    print(f"Output dir:   {output_dir}")

    # Find samples
    samples = find_samples(pipeline_dir)
    print(f"Found {len(samples)} common samples")
    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Limited to {len(samples)} samples")

    # Distributed: split samples by rank
    if args.world_size > 1:
        samples = samples[args.rank::args.world_size]
        print(f"Rank {args.rank}/{args.world_size}: processing {len(samples)} samples")

    # Filter vis types
    vis_types_to_process = list(VIS_TYPES.keys())
    if args.stages:
        vis_types_to_process = [
            vt for vt in vis_types_to_process if VIS_TYPES[vt]["stage"] in args.stages
        ]
    if args.views:
        vis_types_to_process = [
            vt for vt in vis_types_to_process if VIS_TYPES[vt]["view"] in args.views
        ]
    print(f"Vis types ({len(vis_types_to_process)}): {vis_types_to_process}")

    # ==========================
    # Part 1: Standard views
    # ==========================
    if not args.parts_only:

        # ---- Batch grids ----
        if not args.per_sample_only:
            print("\n=== Creating batch comparison grids ===")
            for vt in vis_types_to_process:
                vis_info = VIS_TYPES[vt]
                vt_dir = os.path.join(output_dir, vt)
                os.makedirs(vt_dir, exist_ok=True)

                n_batches = (len(samples) + args.batch_size - 1) // args.batch_size
                skipped = 0
                for batch_idx in tqdm(range(n_batches), desc=vt):
                    start = batch_idx * args.batch_size
                    end = min(start + args.batch_size, len(samples))
                    out_path = os.path.join(vt_dir, f"batch_{start:04d}_{end-1:04d}.png")

                    if args.skip_existing and os.path.exists(out_path):
                        skipped += 1
                        continue

                    batch_samples = samples[start:end]
                    grid = create_batch_grid(
                        pipeline_dir, batch_samples, vt, vis_info,
                        cell_size=args.cell_size,
                        interior_cell_w=args.interior_cell_w,
                    )
                    grid.save(out_path, quality=95)
                if skipped:
                    print(f"  ({skipped}/{n_batches} skipped)")

        # ---- Per-sample comparisons ----
        if not args.batch_only:
            print("\n=== Creating per-sample comparisons ===")
            per_sample_dir = os.path.join(output_dir, "per_sample")

            skipped = 0
            for sample in tqdm(samples, desc="Per-sample"):
                sample_name = sample.replace("/", "_")
                sample_dir = os.path.join(per_sample_dir, sample_name)

                # Check if all vis types already exist for this sample
                if args.skip_existing:
                    all_exist = all(
                        os.path.exists(os.path.join(sample_dir, f"{vt}.png"))
                        for vt in vis_types_to_process
                    )
                    if all_exist:
                        skipped += 1
                        continue

                os.makedirs(sample_dir, exist_ok=True)
                for vt in vis_types_to_process:
                    out_path = os.path.join(sample_dir, f"{vt}.png")
                    if args.skip_existing and os.path.exists(out_path):
                        continue
                    grid = create_batch_grid(
                        pipeline_dir, [sample], vt, VIS_TYPES[vt],
                        cell_size=args.cell_size,
                        interior_cell_w=args.interior_cell_w,
                    )
                    grid.save(out_path, quality=95)
            if skipped:
                print(f"  ({skipped}/{len(samples)} samples skipped)")

    # ==========================
    # Part 2: Part-aware views
    # ==========================
    if args.render_parts or args.parts_only:
        print("\n=== Creating part-aware comparisons (GPU rendering) ===")
        parts_dir = os.path.join(output_dir, "parts_topdown")

        parts_samples = samples
        if args.max_parts_samples:
            parts_samples = parts_samples[:args.max_parts_samples]
            print(f"Part-aware limited to {len(parts_samples)} samples")

        parts_modes = args.parts_mode or ["geometry", "texture"]
        print(f"Parts modes: {parts_modes}")

        skipped = 0
        for sample in tqdm(parts_samples, desc="Parts"):
            sample_name = sample.replace("/", "_")
            sample_dir = os.path.join(parts_dir, sample_name)
            out_paths = {}
            if "geometry" in parts_modes:
                out_paths["geometry"] = os.path.join(sample_dir, "geometry_parts_topdown.png")
            if "texture" in parts_modes:
                out_paths["texture"] = os.path.join(sample_dir, "texture_parts_topdown.png")

            if args.skip_existing and all(os.path.exists(p) for p in out_paths.values()):
                skipped += 1
                continue

            os.makedirs(sample_dir, exist_ok=True)
            try:
                parts_dict = create_parts_comparison(
                    pipeline_dir, gt_recon_dir, sample,
                    tile_size=args.parts_tile_size,
                    modes=parts_modes,
                )
                if parts_dict is not None:
                    for key, path in out_paths.items():
                        if key in parts_dict and parts_dict[key] is not None:
                            parts_dict[key].save(path, quality=95)
            except Exception as e:
                print(f"\n  ERROR rendering parts for {sample}: {e}")
        if skipped:
            print(f"  ({skipped}/{len(parts_samples)} skipped)")

    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
