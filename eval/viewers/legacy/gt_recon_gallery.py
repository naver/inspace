# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
GT Reconstruction Gallery Viewer

Browse GT reconstructed scenes 6 at a time (scene.glb only).
Good for quickly scanning through samples to find good ones.

Usage:
    python eval/viewers/gt_recon_gallery.py --port 7862
"""

import os
import argparse

import gradio as gr

# ============================================================
# Config
# ============================================================
DEFAULT_DATA_DIR = "evals/gt_recon"
NUM_COLS = 3
NUM_ROWS = 2
PAGE_SIZE = NUM_COLS * NUM_ROWS  # 6 samples per page


# ============================================================
# Data Discovery
# ============================================================

def discover_samples(data_dir):
    """Discover all (scene_id, room_id) pairs with scene.glb."""
    samples = []
    if not os.path.isdir(data_dir):
        return samples

    for scene_id in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene_id)
        if not os.path.isdir(scene_path):
            continue
        for room_id in sorted(os.listdir(scene_path)):
            scene_glb = os.path.join(scene_path, room_id, "meshes", "scene.glb")
            if os.path.exists(scene_glb):
                samples.append((scene_id, room_id))

    return samples


# ============================================================
# Gradio App
# ============================================================

def create_demo(data_dir):
    samples = discover_samples(data_dir)
    total = len(samples)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    def _get_page_data(page_idx):
        """Get GLB paths and labels for a page. Returns list of (glb_path, label, abs_path)."""
        start = page_idx * PAGE_SIZE
        results = []
        for i in range(PAGE_SIZE):
            idx = start + i
            if idx < total:
                s, r = samples[idx]
                glb = os.path.join(data_dir, s, r, "meshes", "scene.glb")
                abs_path = os.path.abspath(os.path.join(data_dir, s, r))
                label = f"[{idx+1}/{total}] {r}"
                results.append((glb, label, abs_path))
            else:
                results.append((None, "", ""))
        return results

    def on_load_page(page_idx):
        """Load a page of 6 samples."""
        page_idx = int(page_idx)
        page_idx = max(0, min(page_idx, total_pages - 1))
        data = _get_page_data(page_idx)

        outputs = []
        for glb_path, label, abs_path in data:
            outputs.append(glb_path)       # Model3D
            outputs.append(label)          # label Markdown
            outputs.append(abs_path)       # path Textbox

        start = page_idx * PAGE_SIZE + 1
        end = min((page_idx + 1) * PAGE_SIZE, total)
        status = f"Page {page_idx + 1}/{total_pages}  |  Samples {start}-{end} of {total}"
        outputs.append(status)
        outputs.append(page_idx)  # update page slider
        return outputs

    def on_prev_page(page_idx):
        page_idx = max(0, int(page_idx) - 1)
        return on_load_page(page_idx)

    def on_next_page(page_idx):
        page_idx = min(total_pages - 1, int(page_idx) + 1)
        return on_load_page(page_idx)

    def on_search_and_go(query, page_idx):
        """Search for a sample and jump to its page."""
        query = query.strip().lower()
        if not query:
            return on_load_page(page_idx)

        for i, (s, r) in enumerate(samples):
            if query in s.lower() or query in r.lower():
                target_page = i // PAGE_SIZE
                return on_load_page(target_page)

        # Not found — stay on current page
        return on_load_page(page_idx)

    # ---- Build UI ----
    with gr.Blocks(title="GT Recon Gallery") as demo:
        gr.Markdown(f"# GT Reconstruction Gallery\n**{total}** rooms  |  **{total_pages}** pages  |  {PAGE_SIZE} per page")

        # --- Top controls ---
        with gr.Row():
            search_box = gr.Textbox(
                label="Search (scene ID or room ID)",
                placeholder="e.g. ea053 or Bedroom",
                scale=3,
            )
            search_btn = gr.Button("Go", variant="primary", scale=1)

        with gr.Row():
            prev_btn = gr.Button("Prev Page", size="sm")
            page_slider = gr.Slider(
                0, total_pages - 1, value=0, step=1,
                label="Page", scale=3,
            )
            next_btn = gr.Button("Next Page", size="sm")

        status_box = gr.Textbox(
            value=f"Page 1/{total_pages}  |  Samples 1-{min(PAGE_SIZE, total)} of {total}",
            label="Status", interactive=False, max_lines=1,
        )

        # --- 6 viewers in 2x3 grid ---
        viewers = []
        labels = []
        paths = []
        copy_btns = []

        for row in range(NUM_ROWS):
            with gr.Row():
                for col in range(NUM_COLS):
                    with gr.Column():
                        lbl = gr.Markdown(value="")
                        viewer = gr.Model3D(label=f"Slot {row*NUM_COLS+col+1}", height=380)
                        with gr.Row():
                            path = gr.Textbox(
                                label="Path", interactive=False,
                                max_lines=1, value="", scale=5,
                            )
                            cb = gr.Button("Copy", size="sm", scale=1)
                        viewers.append(viewer)
                        labels.append(lbl)
                        paths.append(path)
                        copy_btns.append(cb)

        # --- Collect all outputs ---
        all_outputs = []
        for i in range(PAGE_SIZE):
            all_outputs.append(viewers[i])
            all_outputs.append(labels[i])
            all_outputs.append(paths[i])
        all_outputs.append(status_box)
        all_outputs.append(page_slider)

        # --- Event bindings ---
        page_slider.release(on_load_page, [page_slider], all_outputs)
        prev_btn.click(on_prev_page, [page_slider], all_outputs)
        next_btn.click(on_next_page, [page_slider], all_outputs)
        search_btn.click(on_search_and_go, [search_box, page_slider], all_outputs)
        search_box.submit(on_search_and_go, [search_box, page_slider], all_outputs)

        # Copy buttons
        for i in range(PAGE_SIZE):
            copy_btns[i].click(
                fn=None, inputs=[paths[i]], outputs=None,
                js="(path) => { navigator.clipboard.writeText(path); }",
            )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT Recon Gallery")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true", default=False)
    args = parser.parse_args()

    demo = create_demo(args.data_dir)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
