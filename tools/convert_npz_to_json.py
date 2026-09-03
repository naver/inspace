#!/usr/bin/env python3
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Convert NPZ files in all 3d_bounding_box directories to JSON format (saved in the same directory).

Usage:
    python convert_npz_to_json.py --root datasets/ERP_3D_FRONT_test

Output:
    For each NPZ file at: {root}/*/*/3d_bounding_box/*.npz
    Creates JSON file at: {root}/*/*/3d_bounding_box/*.json (same directory)
"""

import numpy as np
import json
import argparse
import os
from pathlib import Path
from typing import Any


def numpy_to_json(obj: Any) -> Any:
    """
    Convert numpy arrays and scalars to JSON-serializable types.
    
    Args:
        obj: numpy array, scalar, or nested structure
        
    Returns:
        JSON-serializable object (list, float, int, str, etc.)
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_json(item) for item in obj]
    else:
        return obj


def convert_npz_to_json(npz_path: str, json_path: str) -> bool:
    """
    Convert a single NPZ file to JSON format.
    
    Args:
        npz_path: Path to input NPZ file
        json_path: Path to output JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load NPZ file
        data = np.load(npz_path, allow_pickle=True)
        
        # Convert to dictionary with JSON-serializable values
        json_data = {}
        for key in data.keys():
            value = data[key]
            json_data[key] = numpy_to_json(value)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        # Save as JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error converting {npz_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_all_3d_bbox_dirs(root_dir: str) -> list:
    """
    Find all 3d_bounding_box directories in the dataset.
    
    Args:
        root_dir: Root directory of ERP_3D_FRONT_test dataset
        
    Returns:
        List of 3d_bounding_box directory paths
    """
    root_path = Path(root_dir)
    bbox_dirs = []
    
    # Find all 3d_bounding_box directories
    for bbox_dir in root_path.rglob('3d_bounding_box'):
        if bbox_dir.is_dir():
            bbox_dirs.append(bbox_dir)
    
    return sorted(bbox_dirs)


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPZ files in all 3d_bounding_box directories to JSON format'
    )
    parser.add_argument(
        '--root',
        type=str,
        default="datasets/ERP_3D_FRONT_test",
        help='Root directory of ERP_3D_FRONT_test dataset'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only list files without converting'
    )
    
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.root)
    
    if not os.path.isdir(root_dir):
        print(f"Error: Directory does not exist: {root_dir}")
        return
    
    print(f"Root directory: {root_dir}")
    print(f"Dry run: {args.dry_run}\n")
    
    # Find all 3d_bounding_box directories
    print("Searching for 3d_bounding_box directories...")
    bbox_dirs = find_all_3d_bbox_dirs(root_dir)
    print(f"Found {len(bbox_dirs)} 3d_bounding_box directories\n")
    
    # Collect all NPZ files
    all_npz_files = []
    for bbox_dir in bbox_dirs:
        npz_files = sorted(bbox_dir.glob('*.npz'))
        for npz_path in npz_files:
            # Get relative path for display
            rel_path = npz_path.relative_to(root_dir)
            all_npz_files.append((npz_path, rel_path, bbox_dir))
    
    if len(all_npz_files) == 0:
        print("No NPZ files found in any 3d_bounding_box directories")
        return
    
    print(f"Found {len(all_npz_files)} NPZ files in {len(bbox_dirs)} directories\n")
    
    if args.dry_run:
        print("Files to convert:")
        for npz_path, rel_path, bbox_dir in all_npz_files:
            json_path = npz_path.with_suffix('.json')
            json_rel_path = json_path.relative_to(root_dir)
            print(f"  {rel_path} -> {json_rel_path}")
        return
    
    # Convert each NPZ file
    success_count = 0
    fail_count = 0
    
    for i, (npz_path, rel_path, bbox_dir) in enumerate(all_npz_files, 1):
        json_path = npz_path.with_suffix('.json')
        
        print(f"[{i}/{len(all_npz_files)}] Converting: {rel_path}")
        
        if convert_npz_to_json(str(npz_path), str(json_path)):
            print(f"  ✓ Saved: {json_path.name}")
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(all_npz_files)}")
    print(f"  Directories processed: {len(bbox_dirs)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
