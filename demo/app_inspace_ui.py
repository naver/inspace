# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Shared UI styling for the InSpace Gradio demos (branded header, theme, CSS).

In gradio 6, `theme` and `css` are passed to `demo.launch(...)`, not `gr.Blocks(...)`.
Usage:
    from demo.app_inspace_ui import header_html, INSPACE_THEME, INSPACE_CSS
    with gr.Blocks(title="InSpace ...") as demo:
        gr.HTML(header_html("Interactive ... demo"))
    ...
    demo.launch(..., theme=INSPACE_THEME, css=INSPACE_CSS)
"""

import os
import base64
import gradio as gr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INSPACE_THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

INSPACE_CSS = """
.gradio-container {max-width: 1500px !important; margin: auto !important;}
.inspace-header {text-align:center; padding: 14px 0 8px;}
.inspace-logo {height: 78px; margin-bottom: 6px;}
.inspace-title {font-size: 1.5rem; font-weight: 800; letter-spacing:-0.01em;}
.inspace-sub {font-weight: 500; font-size: 1.05rem; color: var(--body-text-color-subdued);}
.inspace-tag {color: var(--body-text-color-subdued); margin: 4px 0 8px;}
.inspace-badges img {display:inline; height: 20px; margin: 0 2px;}
.tabitem {border-radius: 10px;}
footer {display: none !important;}
"""

_BADGES = (
    '<a href="#"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a> '
    '<a href="https://kookie12.github.io/InSpace-Project-Page/"><img src="https://img.shields.io/badge/Project-Website-blue"></a> '
    '<a href="https://huggingface.co/GwanHyeong/InSpace"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow"></a> '
    '<a href="https://huggingface.co/datasets/GwanHyeong/ERP-FRONT-30K"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-orange"></a>'
)


def _logo_data_uri():
    p = os.path.join(PROJECT_ROOT, 'figures', 'inspace_logo.png')
    try:
        with open(p, 'rb') as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def header_html(tagline="Interactive demo · ECCV 2026"):
    """Branded header: logo + title + tagline + badge links."""
    return f"""
<div class="inspace-header">
  <img class="inspace-logo" src="{_logo_data_uri()}" alt="InSpace"/>
  <div class="inspace-title">InSpace<span class="inspace-sub"> · Structure-Aware 3D Indoor Scene Generation from a Single 360° Image</span></div>
  <div class="inspace-tag">{tagline}</div>
  <div class="inspace-badges">{_BADGES}</div>
</div>
"""
