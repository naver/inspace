# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Analyze the number of assets per scene in ERP_3D_FRONT dataset.
Usage:
    python analyze_dataset_assets.py --root datasets/ERP_3D_FRONT
"""

import argparse
import os
import glob
from collections import Counter
import numpy as np


def analyze(root):
    # Find all scenes: {uuid}/{room_name}
    scenes = []
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            if not os.path.isdir(room_path):
                continue
            scenes.append((uuid_dir, room_name, room_path))

    print(f"Total scenes: {len(scenes)}")

    asset_counts = []
    category_counter = Counter()
    scenes_with_no_assets = []
    scenes_missing_dir = []

    for uuid_dir, room_name, room_path in scenes:
        assets_dir = os.path.join(room_path, "mesh", "individual_assets")
        if not os.path.isdir(assets_dir):
            scenes_missing_dir.append(f"{uuid_dir}/{room_name}")
            continue

        glb_files = glob.glob(os.path.join(assets_dir, "*.glb"))
        n_assets = len(glb_files)
        asset_counts.append(n_assets)

        if n_assets == 0:
            scenes_with_no_assets.append(f"{uuid_dir}/{room_name}")

        # Extract category names (everything before the _inst{N}.glb)
        for f in glb_files:
            fname = os.path.basename(f)
            # Pattern: {category}_{category}_{hash}_inst{N}.glb
            # Extract top-level category (first word before first _)
            parts = fname.rsplit("_inst", 1)
            if len(parts) == 2:
                category_name = parts[0]
                # Get the leading category (first token)
                top_cat = category_name.split("_")[0]
                category_counter[top_cat] += 1

    asset_counts = np.array(asset_counts)

    print(f"\n{'='*60}")
    print(f"ASSET COUNT STATISTICS")
    print(f"{'='*60}")
    print(f"Scenes analyzed:    {len(asset_counts)}")
    print(f"Scenes missing dir: {len(scenes_missing_dir)}")
    print(f"Scenes with 0 assets: {len(scenes_with_no_assets)}")
    print(f"")
    print(f"Min assets:    {asset_counts.min()}")
    print(f"Max assets:    {asset_counts.max()}")
    print(f"Mean assets:   {asset_counts.mean():.2f}")
    print(f"Median assets: {np.median(asset_counts):.1f}")
    print(f"Std assets:    {asset_counts.std():.2f}")
    print(f"Total assets:  {asset_counts.sum()}")

    # Distribution histogram
    print(f"\n{'='*60}")
    print(f"DISTRIBUTION (asset count → number of scenes)")
    print(f"{'='*60}")
    count_dist = Counter(asset_counts.tolist())
    for k in sorted(count_dist.keys()):
        bar = "#" * min(count_dist[k], 80)
        print(f"  {k:3d} assets: {count_dist[k]:5d} scenes  {bar}")

    # Percentiles
    print(f"\n{'='*60}")
    print(f"PERCENTILES")
    print(f"{'='*60}")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:3d}th percentile: {np.percentile(asset_counts, p):.0f} assets")

    # Top categories
    print(f"\n{'='*60}")
    print(f"TOP 30 ASSET CATEGORIES")
    print(f"{'='*60}")
    for cat, cnt in category_counter.most_common(30):
        print(f"  {cat:45s}: {cnt:6d}")

    # Room type distribution
    print(f"\n{'='*60}")
    print(f"ROOM TYPE DISTRIBUTION")
    print(f"{'='*60}")
    room_type_counter = Counter()
    room_type_assets = {}
    for uuid_dir, room_name, room_path in scenes:
        # Room name pattern: RoomType-{number}
        room_type = room_name.rsplit("-", 1)[0] if "-" in room_name else room_name
        room_type_counter[room_type] += 1
        assets_dir = os.path.join(room_path, "mesh", "individual_assets")
        if os.path.isdir(assets_dir):
            n = len(glob.glob(os.path.join(assets_dir, "*.glb")))
            room_type_assets.setdefault(room_type, []).append(n)

    for rt, cnt in room_type_counter.most_common():
        arr = np.array(room_type_assets.get(rt, [0]))
        print(f"  {rt:30s}: {cnt:5d} scenes, assets: mean={arr.mean():.1f}, median={np.median(arr):.0f}, max={arr.max()}")

    # Print some example scenes with many assets
    if len(asset_counts) > 0:
        print(f"\n{'='*60}")
        print(f"SCENES WITH MOST ASSETS (top 10)")
        print(f"{'='*60}")
        indexed = [(c, i) for i, c in enumerate(asset_counts)]
        indexed.sort(reverse=True)
        valid_scenes = [(uuid_dir, room_name, room_path)
                        for uuid_dir, room_name, room_path in scenes
                        if os.path.isdir(os.path.join(room_path, "mesh", "individual_assets"))]
        for c, i in indexed[:30]:
            uuid_dir, room_name, _ = valid_scenes[i]
            print(f"  {c:3d} assets: {uuid_dir}/{room_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="datasets/ERP_3D_FRONT_test")
    args = parser.parse_args()
    analyze(args.root)
