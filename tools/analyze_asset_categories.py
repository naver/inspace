# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Scan ERP_3D_FRONT_test dataset and collect all asset names from mesh/individual_assets.
Extract categories and save summary to JSON.
"""
import os
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

def parse_asset_name(filename):
    """
    Parse asset filename like:
      'coffee_table_coffee_table_f4b1c97b_inst007.glb'
      'lounge_chair___cafe_chair___office_chair_lounge_chair___cafe_chair___office_chair_01687395_inst010.glb'

    Pattern: {category}_{category}_{hex_id}_inst{NNN}.glb
    where category may contain '___' separators for multi-label categories.

    The tricky part: big_category and small_category are often identical.
    The hex ID is always 8 chars [0-9a-f].
    """
    name = filename.replace('.glb', '')

    # Extract instance number
    inst_match = re.search(r'_inst(\d+)$', name)
    if not inst_match:
        return {'raw': filename, 'category': name, 'hex_id': None, 'inst': None}

    inst_num = int(inst_match.group(1))
    name = name[:inst_match.start()]

    # Extract hex ID (8 hex chars at the end)
    hex_match = re.search(r'_([0-9a-f]{8})$', name)
    if not hex_match:
        return {'raw': filename, 'category': name, 'hex_id': None, 'inst': inst_num}

    hex_id = hex_match.group(1)
    name = name[:hex_match.start()]

    # Now name = "{big_category}_{small_category}"
    # Often they are identical: "coffee_table_coffee_table"
    # Try to split by finding the repeated pattern
    # Strategy: try splitting name in half
    half = len(name) // 2
    # Check if name is "{X}_{X}" pattern (big == small)
    if len(name) % 2 == 1 and name[half] == '_':
        left = name[:half]
        right = name[half+1:]
        if left == right:
            category = left
        else:
            category = name  # Keep full name
    else:
        # Try finding the split point where big_cat == small_cat
        found = False
        for i in range(1, len(name)):
            if name[i] == '_':
                left = name[:i]
                right = name[i+1:]
                if left == right:
                    category = left
                    found = True
                    break
        if not found:
            category = name

    # Clean up category: replace '___' with ' / '
    category_clean = category.replace('___', ' / ')

    return {
        'raw': filename,
        'category': category_clean,
        'hex_id': hex_id,
        'inst': inst_num
    }


def main():
    root = 'datasets/ERP_3D_FRONT_test'
    output_path = os.path.join(root, 'asset_categories.json')

    # Collect all assets
    all_assets = []  # list of {room_path, room_name, asset_filename, parsed}
    per_room_assets = defaultdict(list)
    category_counter = Counter()
    unique_models = set()  # (category, hex_id) pairs

    uuids = sorted(os.listdir(root))
    for uuid in uuids:
        uuid_path = os.path.join(root, uuid)
        if not os.path.isdir(uuid_path):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            assets_dir = os.path.join(room_path, 'mesh', 'individual_assets')
            if not os.path.isdir(assets_dir):
                continue

            room_key = f"{uuid}/{room_name}"
            glb_files = sorted([f for f in os.listdir(assets_dir) if f.endswith('.glb')])

            for glb in glb_files:
                parsed = parse_asset_name(glb)
                all_assets.append({
                    'room': room_key,
                    'filename': glb,
                    'category': parsed['category'],
                    'hex_id': parsed['hex_id'],
                    'instance': parsed['inst']
                })
                per_room_assets[room_key].append(parsed['category'])
                category_counter[parsed['category']] += 1
                if parsed['hex_id']:
                    unique_models.add((parsed['category'], parsed['hex_id']))

    # Sort categories by count
    sorted_categories = sorted(category_counter.items(), key=lambda x: -x[1])

    # Build summary
    summary = {
        'total_rooms': len(per_room_assets),
        'total_asset_instances': len(all_assets),
        'total_unique_models': len(unique_models),
        'total_categories': len(category_counter),
        'categories_by_count': {cat: count for cat, count in sorted_categories},
        'assets_per_room_stats': {
            'min': min(len(v) for v in per_room_assets.values()),
            'max': max(len(v) for v in per_room_assets.values()),
            'avg': round(sum(len(v) for v in per_room_assets.values()) / len(per_room_assets), 1),
        },
        'all_assets': all_assets,
    }

    # Save
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_path}")
    print(f"\n=== Summary ===")
    print(f"Total rooms: {summary['total_rooms']}")
    print(f"Total asset instances: {summary['total_asset_instances']}")
    print(f"Total unique models (category+hex_id): {summary['total_unique_models']}")
    print(f"Total categories: {summary['total_categories']}")
    print(f"Assets per room: min={summary['assets_per_room_stats']['min']}, "
          f"max={summary['assets_per_room_stats']['max']}, "
          f"avg={summary['assets_per_room_stats']['avg']}")

    print(f"\n=== Top 30 Categories ===")
    for i, (cat, count) in enumerate(sorted_categories[:30]):
        print(f"  {i+1:3d}. {cat:<60s}  ({count} instances)")

    print(f"\n=== All Categories ({len(sorted_categories)} total) ===")
    for i, (cat, count) in enumerate(sorted_categories):
        print(f"  {i+1:3d}. {cat:<60s}  ({count})")


if __name__ == '__main__':
    main()
