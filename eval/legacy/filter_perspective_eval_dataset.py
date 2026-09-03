#!/usr/bin/env python3
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Filter perspective eval dataset by manually selected indices.

After reviewing examination grids, select the best views by index and
save a filtered JSON + copy selected perspective images.

Usage:
    python eval/filter_perspective_eval_dataset.py
"""

import os
import json
import shutil
from pathlib import Path


# ── Manually selected indices from examination grids ──
SELECTED_INDICES = [
    1, 2, 3, 6, 7, 11, 13, 15, 19, 20, 22, 24, 25, 26, 32, 34, 36, 38,
    40, 41, 44, 45, 47, 76, 80, 88, 94, 97, 103, 105, 118, 119, 121, 124,
    137, 138, 141, 149, 152, 168, 179, 186, 220, 244, 252, 268, 299, 304,
    320, 343, 349, 377, 476, 497, 501,
]

INPUT_JSON = "evals/perspective_eval_dataset.json"
OUTPUT_JSON = "evals/perspective_eval_dataset_selected.json"
OUTPUT_IMAGE_DIR = "evals/perspective_eval_images"


def main():
    with open(INPUT_JSON) as f:
        data = json.load(f)

    all_samples = data["samples"]
    print(f"Total samples in source JSON: {len(all_samples)}")
    print(f"Manually selected indices: {len(SELECTED_INDICES)}")

    # Validate indices
    max_idx = len(all_samples) - 1
    invalid = [i for i in SELECTED_INDICES if i > max_idx]
    if invalid:
        print(f"WARNING: indices out of range (max={max_idx}): {invalid}")

    # Filter samples
    selected = []
    for orig_idx in sorted(SELECTED_INDICES):
        if orig_idx > max_idx:
            continue
        sample = all_samples[orig_idx].copy()
        sample["original_idx"] = orig_idx
        selected.append(sample)

    # Re-index
    for i, s in enumerate(selected):
        s["idx"] = i

    print(f"Selected {len(selected)} samples")

    # Copy perspective images to output dir
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    for s in selected:
        src = s["perspective_image"]
        if not os.path.exists(src):
            print(f"  WARNING: missing {src}")
            continue
        dst_name = f"{s['idx']:04d}_{s['room_name']}_v{s['view_idx']}.png"
        dst = os.path.join(OUTPUT_IMAGE_DIR, dst_name)
        shutil.copy2(src, dst)

    # Build output JSON
    output = {
        "dataset_info": {
            **data["dataset_info"],
            "description": "Manually curated perspective images for evaluation",
            "manual_selection": {
                "source_json": INPUT_JSON,
                "n_source_samples": len(all_samples),
                "n_selected": len(selected),
                "selected_original_indices": sorted(SELECTED_INDICES),
            },
            "statistics": {
                **data["dataset_info"]["statistics"],
                "total_selected": len(selected),
                "selected_mean_score": round(
                    sum(s["score"] for s in selected) / len(selected), 2
                ) if selected else 0,
                "selected_visible_assets_distribution": {},
            },
        },
        "samples": selected,
    }

    # Visible assets distribution for selected
    if selected:
        vis_counts = [s["n_visible_assets"] for s in selected]
        for n in sorted(set(vis_counts)):
            output["dataset_info"]["statistics"]["selected_visible_assets_distribution"][str(n)] = vis_counts.count(n)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Filtered Perspective Eval Dataset")
    print(f"{'='*60}")
    print(f"  Selected: {len(selected)} / {len(all_samples)} samples")
    if selected:
        scores = [s["score"] for s in selected]
        vis = [s["n_visible_assets"] for s in selected]
        print(f"  Score range: {min(scores):.1f} - {max(scores):.1f} (mean {sum(scores)/len(scores):.1f})")
        print(f"  Visible assets range: {min(vis)} - {max(vis)} (mean {sum(vis)/len(vis):.1f})")
    print(f"\n  JSON saved to: {OUTPUT_JSON}")
    print(f"  Images copied to: {OUTPUT_IMAGE_DIR}/")


if __name__ == "__main__":
    main()
