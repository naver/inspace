# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Remove ceiling objects from layout.obj and create layout_wo_ceiling.obj + layout_wo_ceiling.mtl.

Ceiling objects are identified by object names starting with "Ceiling" or "CustomizedCeiling".

Usage:
    python data_toolkit/erp/remove_ceiling_from_layout.py --root datasets/ERP_3D_FRONT_test
    python data_toolkit/erp/remove_ceiling_from_layout.py --root datasets/ERP_3D_FRONT_test --rank 0 --world_size 4
"""

import argparse
import os
import re
from pathlib import Path
from tqdm import tqdm


CEILING_PREFIXES = (
    "Ceiling",                          # Ceiling_Ceiling_mesh, Ceiling.004_...
    "CustomizedCeiling",                # CustomizedCeiling_CustomizedCeiling_mesh, ...
    "ExtrusionCustomizedCeilingModel",  # ExtrusionCustomizedCeilingModel.001_...
    "SmartCustomizedCeiling",           # SmartCustomizedCeiling.001_...
)

CEILING_MATERIAL_PREFIXES = (
    "Ceiling_material",
    "CustomizedCeiling_material",
    "ExtrusionCustomizedCeilingModel_material",
    "SmartCustomizedCeiling_material",
)


def is_ceiling_object(obj_name: str) -> bool:
    """Check if an object name is a ceiling object."""
    return obj_name.startswith(CEILING_PREFIXES)


def is_ceiling_material(mtl_name: str) -> bool:
    """Check if a material name is a ceiling material."""
    return mtl_name.startswith(CEILING_MATERIAL_PREFIXES)


def remove_ceiling_from_obj(obj_path: str, out_obj_path: str, out_mtl_path: str) -> dict:
    """
    Parse layout.obj, remove ceiling object blocks, renumber vertex indices,
    and write layout_wo_ceiling.obj + layout_wo_ceiling.mtl.

    Returns dict with stats.
    """
    mtl_path = obj_path.replace(".obj", ".mtl")
    out_mtl_name = os.path.basename(out_mtl_path)

    # ---- Pass 1: Parse OBJ into object blocks ----
    blocks = []  # list of {name, lines, v_count, vt_count, vn_count, is_ceiling}
    header_lines = []  # lines before first 'o' (comments, mtllib)
    current_block = None

    with open(obj_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("o "):
                # Start new block
                if current_block is not None:
                    blocks.append(current_block)
                obj_name = stripped[2:].strip()
                current_block = {
                    'name': obj_name,
                    'lines': [line],
                    'v_count': 0,
                    'vt_count': 0,
                    'vn_count': 0,
                    'is_ceiling': is_ceiling_object(obj_name),
                }
            elif current_block is None:
                # Header lines (before first object)
                header_lines.append(line)
            else:
                current_block['lines'].append(line)
                if stripped.startswith("v "):
                    current_block['v_count'] += 1
                elif stripped.startswith("vt "):
                    current_block['vt_count'] += 1
                elif stripped.startswith("vn "):
                    current_block['vn_count'] += 1

    if current_block is not None:
        blocks.append(current_block)

    # ---- Build index remapping ----
    # OBJ uses 1-based global indices for v, vt, vn
    v_offset = 0  # how many v indices to subtract
    vt_offset = 0
    vn_offset = 0

    # For each block, compute the offset adjustments
    block_v_offsets = []
    block_vt_offsets = []
    block_vn_offsets = []

    for block in blocks:
        block_v_offsets.append(v_offset)
        block_vt_offsets.append(vt_offset)
        block_vn_offsets.append(vn_offset)
        if block['is_ceiling']:
            v_offset += block['v_count']
            vt_offset += block['vt_count']
            vn_offset += block['vn_count']

    ceiling_names = [b['name'] for b in blocks if b['is_ceiling']]
    kept_names = [b['name'] for b in blocks if not b['is_ceiling']]

    # ---- Write output OBJ ----
    face_pattern = re.compile(r'(\d+)(?:/(\d*)(?:/(\d+))?)?')

    with open(out_obj_path, 'w') as f:
        for line in header_lines:
            stripped = line.strip()
            if stripped.startswith("mtllib "):
                f.write(f"mtllib {out_mtl_name}\n")
            else:
                f.write(line)

        for i, block in enumerate(blocks):
            if block['is_ceiling']:
                continue

            v_adj = block_v_offsets[i]
            vt_adj = block_vt_offsets[i]
            vn_adj = block_vn_offsets[i]

            for line in block['lines']:
                stripped = line.strip()
                if stripped.startswith("f "):
                    # Renumber face indices
                    parts = stripped.split()
                    new_parts = [parts[0]]  # 'f'
                    for vert in parts[1:]:
                        m = face_pattern.match(vert)
                        if m:
                            vi = int(m.group(1)) - v_adj
                            vti = m.group(2)
                            vni = m.group(3)

                            if vti is not None and vti != '':
                                vti = int(vti) - vt_adj
                                if vni is not None:
                                    vni = int(vni) - vn_adj
                                    new_parts.append(f"{vi}/{vti}/{vni}")
                                else:
                                    new_parts.append(f"{vi}/{vti}")
                            elif vni is not None:
                                vni = int(vni) - vn_adj
                                new_parts.append(f"{vi}//{vni}")
                            else:
                                new_parts.append(str(vi))
                        else:
                            new_parts.append(vert)
                    f.write(' '.join(new_parts) + '\n')
                else:
                    f.write(line)

    # ---- Write output MTL (remove ceiling materials) ----
    if os.path.exists(mtl_path):
        with open(mtl_path, 'r') as f:
            mtl_content = f.read()

        # Parse MTL into material blocks
        mtl_blocks = []
        current_mtl = None
        header_mtl_lines = []

        for line in mtl_content.splitlines(keepends=True):
            stripped = line.strip()
            if stripped.startswith("newmtl "):
                if current_mtl is not None:
                    mtl_blocks.append(current_mtl)
                mtl_name = stripped[7:].strip()
                current_mtl = {
                    'name': mtl_name,
                    'lines': [line],
                    'is_ceiling': is_ceiling_material(mtl_name),
                }
            elif current_mtl is None:
                header_mtl_lines.append(line)
            else:
                current_mtl['lines'].append(line)

        if current_mtl is not None:
            mtl_blocks.append(current_mtl)

        kept_count = sum(1 for b in mtl_blocks if not b['is_ceiling'])

        with open(out_mtl_path, 'w') as f:
            for line in header_mtl_lines:
                if line.strip().startswith("# Material Count:"):
                    f.write(f"# Material Count: {kept_count}\n")
                else:
                    f.write(line)
            for block in mtl_blocks:
                if not block['is_ceiling']:
                    for line in block['lines']:
                        f.write(line)

    return {
        'ceiling_removed': ceiling_names,
        'kept': kept_names,
    }


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true", help="Print what would be done without writing")
    args = parser.parse_args()

    # args.root = "datasets/_ERP_3D_FRONT_before/ERP_3D_FRONT_test"
    args.root = "datasets/ERP_3D_FRONT"
    # args.root = "datasets/ERP_3D_FRONT_test"

    # Find all layout.obj files
    root = Path(args.root)
    layout_files = sorted(root.glob("*/*/mesh/layout.obj"))

    # Shard for distributed processing
    layout_files = layout_files[args.rank::args.world_size]

    print(f"Found {len(layout_files)} layout.obj files (rank {args.rank}/{args.world_size})")

    skipped = 0
    processed = 0
    errors = []

    for obj_path in tqdm(layout_files, desc="Removing ceilings"):
        obj_path = str(obj_path)
        mesh_dir = os.path.dirname(obj_path)
        out_obj = os.path.join(mesh_dir, "layout_wo_ceiling.obj")
        out_mtl = os.path.join(mesh_dir, "layout_wo_ceiling.mtl")

        if os.path.exists(out_obj) and os.path.exists(out_mtl):
            skipped += 1
            continue

        if args.dry_run:
            print(f"  Would process: {obj_path}")
            continue

        try:
            stats = remove_ceiling_from_obj(obj_path, out_obj, out_mtl)
            processed += 1
            if not stats['ceiling_removed']:
                print(f"  WARNING: No ceiling found in {obj_path}")
        except Exception as e:
            errors.append((obj_path, str(e)))
            print(f"  ERROR: {obj_path}: {e}")

    print(f"\nDone: {processed} processed, {skipped} skipped (already exist), {len(errors)} errors")
    if errors:
        for path, err in errors:
            print(f"  ERROR: {path}: {err}")


if __name__ == "__main__":
    main()
