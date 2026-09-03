#!/usr/bin/env python3
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Rebuild 3D Bounding Box from 3D-FRONT JSON

Reads the transform info (pos, rot, scale) from the 3D-FRONT JSON, and reads bbox_size from
raw_model.obj of the 3D-FUTURE model. Follows the same approach as rebuild_dataset_part_1.py,
and runs standalone without Blender.

Floor values are taken as-is from the existing 3D bounding box npz. The 3D bounding boxes are
computed from the center and scale in normalization_info.json produced by step1_dump_mesh_erp.py,
so building a dataset from scratch goes through this same path: full_room_wo_ceiling.obj is loaded
in Blender and its bounding box is used to normalize the 3D bounding boxes directly.

Usage:
    python rebuild_3d_bbox.py --dataset_root /path/to/ERP_3D_FRONT_Part_1_test
    python rebuild_3d_bbox.py --dataset_root /path/to/ERP_3D_FRONT_Part_1_test --scene_id xxx
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
import traceback
import time


# =============================================================================
# Utility Functions (from rebuild_dataset_part_1.py)
# =============================================================================

def get_category_from_title(title: str) -> tuple:
    """Extract category and name from a title (e.g., "bed/king-size bed")."""
    if '/' in title:
        parts = title.split('/')
        category = parts[0]
        name = parts[-1]
    else:
        category = title
        name = title
    return category, name


def get_furniture_bbox_from_obj(jid: str, model_base_path: str):
    """
    Read bounding box info from the OBJ file of a 3D-FUTURE model.

    Returns:
        dict with:
        - bbox_size: (width, height, depth)
        - min_coords: (min_x, min_y, min_z) - min point in the local frame
        - max_coords: (max_x, max_y, max_z) - max point in the local frame
    """
    obj_path = Path(model_base_path) / jid / "raw_model.obj"

    if not obj_path.exists():
        return None

    try:
        vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        vertices.append([x, y, z])

        if not vertices:
            return None

        vertices = np.array(vertices)
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        bbox_size = max_coords - min_coords

        return {
            'bbox_size': tuple(bbox_size),
            'min_coords': tuple(min_coords),
            'max_coords': tuple(max_coords),
        }
    except Exception:
        return None


# def is_ceiling_object(name: str, pivot_z: float = None, floor_threshold: float = 0.3) -> bool:
#     """
#     Decide whether the object is ceiling-mounted
#     """
#     name_lower = name.lower()

#     # Exclude objects explicitly known to be floor-standing
#     floor_keywords = ['floor lamp', 'floor light', 'floor-based', 'standing lamp', 'standing light']
#     if any(kw in name_lower for kw in floor_keywords):
#         return False

#     # A low pivot_z means a floor-standing object
#     if pivot_z is not None and pivot_z < floor_threshold:
#         return False

#     # Check ceiling keywords
#     ceiling_keywords = [
#         'pendant', 'chandelier', 'ceiling', 'downlight',
#         'spotlight', 'fan', 'fixture'
#     ]
#     return any(kw in name_lower for kw in ceiling_keywords)


def compute_rotated_bbox_2d(center_xy, half_width, half_depth, yaw_rad):
    """Compute the AABB of a rotated 2D bbox."""
    corners_local = np.array([
        [-half_width, -half_depth],
        [-half_width,  half_depth],
        [ half_width, -half_depth],
        [ half_width,  half_depth],
    ])

    cos_a = math.cos(-yaw_rad)
    sin_a = math.sin(-yaw_rad)
    rot_2d = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])

    corners_rotated = corners_local @ rot_2d.T
    corners_world = corners_rotated + center_xy

    min_xy = corners_world.min(axis=0)
    max_xy = corners_world.max(axis=0)

    return min_xy, max_xy


# =============================================================================
# JSON Loading Functions
# =============================================================================

def load_3dfront_json(front_folder: str, scene_json_name: str) -> dict:
    """Load the 3D-FRONT JSON."""
    json_path = Path(front_folder) / scene_json_name
    if not json_path.exists():
        return {}

    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def load_model_info(future_folder: str) -> dict:
    """Load the jid -> {category, super_category} mapping from model_info.json."""
    model_info_path = Path(future_folder) / 'model_info_revised.json'
    jid_to_model_info = {}

    if not model_info_path.exists():
        return jid_to_model_info

    try:
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)

        for item in model_info:
            model_id = item.get('model_id', '')
            category = item.get('category') or item.get('super-category') or 'Others'
            super_category = item.get('super-category', '')
            if model_id:
                jid_to_model_info[model_id] = {
                    'category': category,
                    'super_category': super_category,
                }
    except Exception:
        pass

    return jid_to_model_info


def build_jid_to_info(json_data: dict, model_info: dict = None) -> dict:
    """Build the jid -> {title, category, name} mapping."""
    jid_to_info = {}

    if model_info:
        for jid, info in model_info.items():
            category = info.get('category', '')
            if category:
                jid_to_info[jid] = {
                    'title': category,
                    'category': category,
                    'name': category
                }

    for furn in json_data.get('furniture', []):
        jid = furn.get('jid', '')
        title = furn.get('title', '')
        if jid and title:
            category, name = get_category_from_title(title)
            jid_to_info[jid] = {
                'title': title,
                'category': category,
                'name': name
            }

    return jid_to_info


def build_furniture_map(json_data: dict) -> dict:
    """Build the uid -> {jid, title, category} mapping."""
    furniture_map = {}
    for furn in json_data.get('furniture', []):
        uid = furn.get('uid', '')
        jid = furn.get('jid', '')
        title = furn.get('title', '')
        category = furn.get('category', '')
        if uid and jid:
            furniture_map[uid] = {
                'jid': jid,
                'title': title,
                'category': category,
            }
    return furniture_map


# =============================================================================
# Asset Info Extraction
# =============================================================================

def extract_assets_info_from_json(json_data: dict, room_instanceid: str,
                                   future_folder: str, jid_to_info: dict) -> list:
    """
    Extract asset info for a specific room from the 3D-FRONT JSON.

    bbox_size is read from raw_model.obj of the 3D-FUTURE model.
    """
    assets_info = []
    furniture_map = build_furniture_map(json_data)

    # Find the room
    target_room = None
    for room in json_data.get('scene', {}).get('room', []):
        instanceid = room.get('instanceid', '')
        room_type = room.get('type', '')
        full_id = f"{room_type}-{instanceid}"

        if instanceid == room_instanceid or full_id == room_instanceid:
            target_room = room
            break

    if target_room is None:
        return assets_info

    # Extract furniture from the room's children
    for child in target_room.get('children', []): # len(target_room.get('children', [])) = 90
        ref = child.get('ref', '')

        if ref not in furniture_map:
            continue

        furn_info = furniture_map[ref]
        jid = furn_info['jid']

        # NOTE: fetch bbox info (size + min/max coords)
        bbox_info = get_furniture_bbox_from_obj(jid, future_folder)

        if bbox_info is None:
            continue

        # Determine category/name
        if jid in jid_to_info:
            info = jid_to_info[jid]
            title = info['title']
            category = info['category']
            name = info['name']
        else:
            title = furn_info.get('title', '')
            if title:
                category, name = get_category_from_title(title)
            else:
                category = furn_info.get('category', 'unknown')
                name = category

        asset_data = {
            'pos': child.get('pos', [0, 0, 0]),
            'scale': child.get('scale', [1, 1, 1]),
            'rot': child.get('rot', [0, 0, 0, 1]),
            'bbox_size': bbox_info['bbox_size'],
            'local_min_coords': bbox_info['min_coords'],  # local frame min
            'local_max_coords': bbox_info['max_coords'],  # local frame max
            'name': name,
            'jid': jid,
            'uid': ref,
            'category': category,
            'title': title,
            'source_type': '3d_future',
        }
        assets_info.append(asset_data)

    return assets_info


# =============================================================================
# 3D Bounding Box Computation
# =============================================================================

def compute_3d_bboxes_with_alignment(assets_info, boundary_xy=None, normalize=True,
                                      include_floor=False, floor_height=0.05,
                                      external_norm_center=None, external_norm_scale=None):
    """
    Compute 3D OBBs for the assets and build the asset info array in one pass.

    Uses the local coordinates (local_min_coords, local_max_coords) directly to compute
    the exact Z position.

    Args:
        external_norm_center: externally provided normalization center (from normalization_info.json)
        external_norm_scale: externally provided normalization scale (from normalization_info.json)
                            Note: O-Voxel stores scale as 1/max_size
    """

    bboxes_list = []
    obbs_list = []
    asset_jids = []
    asset_uids = []
    asset_categories = []
    asset_filenames = []
    asset_names = []

    # Keep only 3D-FUTURE assets
    assets_info = [a for a in assets_info if a.get('source_type') == '3d_future']

    # Build per-instance file names
    jid_instance_count = {}

    for asset in assets_info:
        jid = asset.get('jid', '')
        category = asset.get('category', 'unknown')
        name = asset.get('name', 'unknown')

        category_safe = category.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('-', '_')
        name_safe = name.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('-', '_')
        short_jid = jid[:8] if jid else 'unknown'

        # Assign the instance index
        if jid not in jid_instance_count:
            jid_instance_count[jid] = 0
        inst_num = jid_instance_count[jid]
        jid_instance_count[jid] += 1

        # File name: category_name_jid_instXXX.glb
        filename = f"{category_safe}_{name_safe}_{short_jid}_inst{inst_num:03d}.glb"
        asset['instance_filename'] = filename

    # Process each asset
    for asset in assets_info:
        print(f"instance_filename: {asset.get('instance_filename', 'unknown')}")
        pos = asset['pos']  # [X, Z, Y] in 3D-FRONT coordinate
        scale = asset['scale']
        rot = asset['rot']
        bbox_size = asset.get('bbox_size')
        local_min = asset.get('local_min_coords')
        local_max = asset.get('local_max_coords')
        name = asset.get('name', '')

        if bbox_size is None or local_min is None or local_max is None:
            continue

        center_x = pos[0]
        center_y = pos[2]

        # Quaternion to Yaw
        yaw_rad = math.atan2(2 * (rot[3] * rot[1] + rot[0] * rot[2]),
                            1 - 2 * (rot[1]**2 + rot[2]**2))

        width = bbox_size[0] * scale[0]
        height = bbox_size[1] * scale[1]
        depth = bbox_size[2] * scale[2]

        half_width = width / 2
        half_depth = depth / 2

        center_xy = np.array([center_x, center_y])
        min_xy, max_xy = compute_rotated_bbox_2d(center_xy, half_width, half_depth, yaw_rad)

        # Convert directly from local to world coordinates
        # 3D-FRONT: pos[1] is the Z coordinate (height) of the model origin
        # raw_model.obj: local_min[1], local_max[1] are model-local Y coordinates (Y-up)
        # world Z = pos[1] + local_Y * scale[1]
        pivot_z = pos[1]
        min_z = pivot_z + local_min[1] * scale[1]
        max_z = pivot_z + local_max[1] * scale[1]
        center_z = (min_z + max_z) / 2
        height = max_z - min_z

        bbox = np.array([[min_xy[0], min_xy[1], min_z],
                         [max_xy[0], max_xy[1], max_z]])
        bboxes_list.append(bbox)

        obb = np.array([center_x, center_y, center_z, width, depth, height, -yaw_rad])
        obbs_list.append(obb)

        # Append asset info
        asset_jids.append(asset.get('jid', ''))
        asset_uids.append(asset.get('uid', ''))
        asset_categories.append(asset.get('category', ''))
        asset_filenames.append(asset.get('instance_filename', ''))
        asset_names.append(asset.get('name', ''))

    # Floor polygon
    floor_polygon = None
    floor_z = 0.0
    if include_floor and boundary_xy is not None and len(boundary_xy) >= 3:
        floor_polygon = boundary_xy.copy()

        floor_min_x = boundary_xy[:, 0].min()
        floor_max_x = boundary_xy[:, 0].max()
        floor_min_y = boundary_xy[:, 1].min()
        floor_max_y = boundary_xy[:, 1].max()
        floor_min_z = 0.0
        floor_max_z = floor_height

        floor_bbox = np.array([[floor_min_x, floor_min_y, floor_min_z],
                               [floor_max_x, floor_max_y, floor_max_z]])
        bboxes_list.append(floor_bbox)

    if not bboxes_list:
        return {
            'bboxes': np.array([]).reshape(0, 2, 3),
            'obbs': np.array([]).reshape(0, 7),
            'asset_jids': [],
            'asset_uids': [],
            'asset_categories': [],
            'asset_filenames': [],
            'asset_names': [],
            'floor_polygon': None,
            'floor_height': floor_height,
            'floor_z': 0.0,
            'norm_center': None,
            'norm_scale': None,
        }

    bboxes = np.array(bboxes_list)
    obbs = np.array(obbs_list) if obbs_list else np.array([]).reshape(0, 7)

    norm_center = None
    norm_scale = None

    if normalize and len(bboxes) > 0:
        # Use the external normalization params when provided (same frame as O-Voxel)
        if external_norm_center is not None and external_norm_scale is not None:
            center = np.array(external_norm_center)
            # O-Voxel scale is already 1/max_size, so use it as-is
            # O-Voxel: (coord - center) * scale
            scale = external_norm_scale
            norm_center = center
            norm_scale = 1.0 / scale  # for storage (stored as the reciprocal)

            # Apply the same normalization as O-Voxel: (coord - center) * scale
            bboxes = (bboxes - center) * scale

            if len(obbs) > 0:
                obbs[:, 0:3] = (obbs[:, 0:3] - center) * scale
                obbs[:, 3:6] = obbs[:, 3:6] * scale

            if floor_polygon is not None:
                floor_polygon = (floor_polygon - center[:2]) * scale

            floor_z = (0.0 - center[2]) * scale

            print(f"[INFO] Using external normalization: center={center}, scale={scale}")
        else:
            # Legacy path: compute center/scale locally
            all_mins = bboxes[:, 0, :].min(axis=0)
            all_maxs = bboxes[:, 1, :].max(axis=0)
            center = (all_mins + all_maxs) / 2
            size = all_maxs - all_mins
            max_size = size.max()

            if max_size > 0:
                norm_center = center
                norm_scale = max_size
                bboxes = (bboxes - center) / max_size

                if len(obbs) > 0:
                    obbs[:, 0:3] = (obbs[:, 0:3] - center) / max_size
                    obbs[:, 3:6] = obbs[:, 3:6] / max_size

                if floor_polygon is not None:
                    floor_polygon = (floor_polygon - center[:2]) / max_size

                floor_z = (0.0 - center[2]) / max_size

    # Normalize floor_height
    if external_norm_scale is not None:
        # With external normalization: multiply by scale
        normalized_floor_height = floor_height * external_norm_scale
    elif norm_scale is not None:
        # With local normalization: divide by max_size
        normalized_floor_height = floor_height / norm_scale
    else:
        normalized_floor_height = floor_height

    return {
        'bboxes': bboxes,
        'obbs': obbs,
        'asset_jids': asset_jids,
        'asset_uids': asset_uids,
        'asset_categories': asset_categories,
        'asset_filenames': asset_filenames,
        'asset_names': asset_names,
        'floor_polygon': floor_polygon,
        'floor_z': floor_z,
        'floor_height': normalized_floor_height,
        'norm_center': norm_center,
        'norm_scale': norm_scale,
    }


# =============================================================================
# Wall / Ceiling Generation
# =============================================================================

def create_wall_obbs_from_boundary(floor_polygon, wall_height, wall_thickness=0.1,
                                    norm_center=None, norm_scale=None):
    """Create a wall OBB along each edge of the floor boundary polygon."""
    if floor_polygon is None or len(floor_polygon) < 3:
        return []

    wall_obbs = []
    n_vertices = len(floor_polygon)

    for i in range(n_vertices):
        p1 = floor_polygon[i]
        p2 = floor_polygon[(i + 1) % n_vertices]

        center_x = (p1[0] + p2[0]) / 2
        center_y = (p1[1] + p2[1]) / 2
        center_z = wall_height / 2

        edge_vec = p2 - p1
        edge_length = np.linalg.norm(edge_vec)

        if edge_length < 0.01:
            continue

        yaw = math.atan2(edge_vec[1], edge_vec[0])

        size_x = edge_length
        size_y = wall_thickness
        size_z = wall_height

        if norm_center is not None and norm_scale is not None:
            center_x = (center_x - norm_center[0]) / norm_scale
            center_y = (center_y - norm_center[1]) / norm_scale
            center_z = (center_z - norm_center[2]) / norm_scale
            size_x /= norm_scale
            size_y /= norm_scale
            size_z /= norm_scale

        wall_obb = np.array([center_x, center_y, center_z,
                             size_x, size_y, size_z, yaw])
        wall_obbs.append(wall_obb)

    return wall_obbs


def create_ceiling_from_floor(floor_polygon, ceiling_top_height, ceiling_thickness=0.02,
                               norm_center=None, norm_scale=None):
    """Build ceiling info from the floor polygon."""
    if floor_polygon is None or len(floor_polygon) < 3:
        return None

    ceiling_polygon = floor_polygon.copy()
    ceiling_z = ceiling_top_height - ceiling_thickness

    if norm_center is not None and norm_scale is not None:
        ceiling_polygon = (ceiling_polygon - norm_center[:2]) / norm_scale
        ceiling_z = (ceiling_z - norm_center[2]) / norm_scale
        ceiling_thickness /= norm_scale

    return {
        'polygon': ceiling_polygon,
        'z': ceiling_z,
        'height': ceiling_thickness,
    }


# =============================================================================
# Room Processing
# =============================================================================

def get_ceiling_height_from_json_data(json_data: dict, room_instanceid: str) -> float:
    """Extract the ceiling height from the JSON."""
    for room in json_data.get('scene', {}).get('room', []):
        instanceid = room.get('instanceid', '')
        room_type = room.get('type', '')
        full_id = f"{room_type}-{instanceid}"

        if instanceid != room_instanceid and full_id != room_instanceid:
            continue

        for child in room.get('children', []):
            ref = child.get('ref', '')

            for mesh in json_data.get('mesh', []):
                if mesh.get('uid') == ref:
                    mesh_type = mesh.get('type', '').lower()

                    if 'ceiling' in mesh_type and 'xyz' in mesh and mesh['xyz']:
                        xyz = np.array(mesh['xyz'])
                        if xyz.ndim == 1 and len(xyz) % 3 == 0:
                            xyz = xyz.reshape(-1, 3)

                        if xyz.ndim == 2 and xyz.shape[1] == 3:
                            ceiling_z = xyz[:, 1].mean()
                            return ceiling_z
                    break
        break

    return None


def process_room(room_path: Path, json_data: dict, future_folder: str,
                 jid_to_info: dict, room_instanceid: str) -> dict:
    """
    Regenerate the 3D bounding boxes of a room.
    """
    result = {
        'room_path': str(room_path),
        'success': False,
        'message': '',
        'num_assets': 0,
        'num_obbs': 0,
    }

    try:
        # 1. Load room_geometry from config.json or room_info.json
        config_path = room_path / 'config.json'
        room_info_path = room_path / 'room_info.json'

        room_geometry = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            room_geometry = config.get('room_geometry', {})
        elif room_info_path.exists():
            with open(room_info_path, 'r') as f:
                room_info = json.load(f)
            room_geometry = room_info.get('room_geometry', {})

        # 2. Extract asset info
        assets_info = extract_assets_info_from_json(
            json_data, room_instanceid, future_folder, jid_to_info
        )
        result['num_assets'] = len(assets_info)

        if not assets_info:
            result['message'] = 'No assets found for this room'
            result['success'] = True
            return result

        # 3. Fetch the floor polygon
        min_corner = room_geometry.get('min_corner', [0, 0, 0])
        max_corner = room_geometry.get('max_corner', [0, 0, 0])

        # Try loading floor_polygon from the existing scene_data.npz
        bbox_folder = room_path / '3d_bounding_box'
        existing_npz = list(bbox_folder.glob('*_scene_data.npz')) if bbox_folder.exists() else []

        floor_polygon_raw = None
        if existing_npz:
            try:
                old_data = np.load(existing_npz[0], allow_pickle=True)
                old_norm_center = old_data.get('norm_center')
                old_norm_scale = old_data.get('norm_scale')
                if 'floor_polygon' in old_data and old_norm_center is not None and old_norm_scale is not None:
                    floor_polygon_raw = old_data['floor_polygon'] * float(old_norm_scale) + old_norm_center[:2]
            except:
                pass

        if floor_polygon_raw is None:
            floor_polygon_raw = np.array([
                [min_corner[0], min_corner[1]],
                [max_corner[0], min_corner[1]],
                [max_corner[0], max_corner[1]],
                [min_corner[0], max_corner[1]]
            ])

        # 4. Fetch the ceiling height (for wall/ceiling generation)
        ceiling_height = get_ceiling_height_from_json_data(json_data, room_instanceid)
        if ceiling_height is None:
            ceiling_height = 3.0

        # 4.5. Load normalization_info.json (same frame as O-Voxel)
        norm_info_path = room_path / 'mesh_dumps' / 'normalization_info.json'
        external_norm_center = None
        external_norm_scale = None

        if norm_info_path.exists():
            try:
                with open(norm_info_path, 'r') as f:
                    norm_info = json.load(f)
                external_norm_center = norm_info.get('center')
                external_norm_scale = norm_info.get('scale')
                print(f"[INFO] Loaded normalization_info.json: center={external_norm_center}, scale={external_norm_scale}")
            except Exception as e:
                print(f"[WARNING] Failed to load normalization_info.json: {e}")
        else:
            print(f"[WARNING] normalization_info.json not found at {norm_info_path}, using self-computed normalization")

        # 5. Compute 3D bounding boxes (Z taken directly from the local coordinates)
        bbox_result = compute_3d_bboxes_with_alignment(
            assets_info, floor_polygon_raw,
            normalize=True, include_floor=True, floor_height=0.05,
            external_norm_center=external_norm_center,
            external_norm_scale=external_norm_scale
        )

        if bbox_result['obbs'] is None or len(bbox_result['obbs']) == 0:
            result['message'] = 'No OBBs computed'
            return result

        result['num_obbs'] = len(bbox_result['obbs'])

        # 6. Generate walls / ceiling
        norm_center = bbox_result.get('norm_center')
        norm_scale = bbox_result.get('norm_scale')

        wall_obbs = create_wall_obbs_from_boundary(
            floor_polygon_raw, ceiling_height, wall_thickness=0.1,
            norm_center=norm_center, norm_scale=norm_scale
        )

        ceiling_result = create_ceiling_from_floor(
            floor_polygon_raw, ceiling_height, ceiling_thickness=0.02,
            norm_center=norm_center, norm_scale=norm_scale
        )

        # 7. Save
        bbox_folder = room_path / '3d_bounding_box'
        bbox_folder.mkdir(exist_ok=True)

        save_dict = {
            # Asset OBBs
            'obbs': bbox_result['obbs'],
            'asset_jids': np.array(bbox_result['asset_jids'], dtype=object),
            'asset_uids': np.array(bbox_result['asset_uids'], dtype=object),
            'asset_categories': np.array(bbox_result['asset_categories'], dtype=object),
            'asset_filenames': np.array(bbox_result['asset_filenames'], dtype=object),
            'asset_names': np.array(bbox_result['asset_names'], dtype=object),

            # Wall OBBs
            'wall_obbs': np.array(wall_obbs) if wall_obbs else np.array([]).reshape(0, 7),

            # Floor
            'floor_polygon': bbox_result.get('floor_polygon'),
            'floor_height': bbox_result.get('floor_height', 0.05),
            'floor_z': bbox_result.get('floor_z', 0.0),

            # Ceiling
            'ceiling_polygon': ceiling_result.get('polygon') if ceiling_result else None,
            'ceiling_z': ceiling_result.get('z', 0.0) if ceiling_result else 0.0,
            'ceiling_height': ceiling_result.get('height', 0.02) if ceiling_result else 0.02,

            # Normalization
            'norm_center': norm_center,
            'norm_scale': norm_scale,
        }

        # Drop None values
        save_dict = {k: v for k, v in save_dict.items() if v is not None}

        npz_path = bbox_folder / f"{room_instanceid}_scene_data.npz"
        np.savez_compressed(npz_path, **save_dict)

        result['success'] = True
        result['message'] = 'OK'
        return result

    except Exception as e:
        result['message'] = f'Error: {str(e)}\n{traceback.format_exc()}'
        return result


def process_scene(scene_path: Path, future_folder: str, front_folder: str) -> list:
    """Process every room in a scene."""
    results = []

    # Find room folders
    room_paths = [p for p in scene_path.iterdir() if p.is_dir()]
    room_paths = [p for p in room_paths if (p / 'config.json').exists() or (p / 'room_info.json').exists()]

    if not room_paths:
        return results

    # Take the scene JSON name from the first room
    first_room = room_paths[0]
    config_path = first_room / 'config.json'
    room_info_path = first_room / 'room_info.json'

    scene_json_name = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        scene_json_name = config.get('metadata', {}).get('scene_json', '')
    elif room_info_path.exists():
        with open(room_info_path, 'r') as f:
            room_info = json.load(f)
        scene_json_name = room_info.get('scene_id', '') + '.json'

    if not scene_json_name:
        print(f"  Cannot determine scene JSON for {scene_path}")
        return results

    # Load the 3D-FRONT JSON
    json_data = load_3dfront_json(front_folder, scene_json_name)
    if not json_data:
        print(f"  3D-FRONT JSON not found: {scene_json_name}")
        return results

    # Load model_info
    model_info = load_model_info(future_folder)
    jid_to_info = build_jid_to_info(json_data, model_info)

    # Process each room
    for room_path in room_paths:
        # Fetch the room instanceid
        config_path = room_path / 'config.json'
        room_info_path = room_path / 'room_info.json'

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            room_instanceid = config.get('metadata', {}).get('room_instanceid', '')
        elif room_info_path.exists():
            with open(room_info_path, 'r') as f:
                room_info = json.load(f)
            room_instanceid = room_info.get('room_instanceid', '')
        else:
            continue

        if not room_instanceid:
            continue

        result = process_room(room_path, json_data, future_folder, jid_to_info, room_instanceid)
        results.append(result)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Rebuild 3D Bounding Box from 3D-FRONT JSON")
    parser.add_argument("--dataset_root", type=str,
                        default="datasets/ERP_3D_FRONT_test",
                        help="Dataset root folder")
    parser.add_argument("--future_folder", type=str,
                        # default="/path/to/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model",
                        default="/path/to/3D-FUTURE/3D-FUTURE-model",
                        help="Path to 3D-FUTURE-model folder")
    parser.add_argument("--front_folder", type=str,
                        # default="/path/to/BlenderProc-3DFront/examples/datasets/front_3d_with_improved_mat/3D-FRONT",
                        default="/path/to/3D-FRONT",
                        help="Path to 3D-FRONT folder")
    parser.add_argument("--scene_id", type=str, default=None,
                        help="Process only specific scene ID (optional)")

    args = parser.parse_args()
    # args.dataset_root = "/path/to/ERP_3D_FRONT_Part_1_test"
    args.dataset_root = "datasets/ERP_3D_FRONT_test"
    args.front_folder = "/path/to/3D-FRONT"
    args.future_folder = "/path/to/3D-FUTURE/3D-FUTURE-model"# '00110bde-f580-40be-b8bb-88715b338a2a'
    
    # args.scene_id = '00110bde-f580-40be-b8bb-88715b338a2a'

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Error: Dataset root not found: {dataset_root}")
        return

    # Find scene folders
    if args.scene_id:
        scene_paths = [dataset_root / args.scene_id]
        if not scene_paths[0].exists():
            print(f"Error: Scene not found: {scene_paths[0]}")
            return
    else:
        scene_paths = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    print("=" * 60)
    print("Rebuild 3D Bounding Box from 3D-FRONT JSON")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"3D-FUTURE folder: {args.future_folder}")
    print(f"3D-FRONT folder: {args.front_folder}")
    print(f"Scenes to process: {len(scene_paths)}")
    print()

    batch_start_time = time.time()
    total_success = 0
    total_failed = 0
    total_rooms = 0
    total_obbs = 0

    for scene_path in tqdm(scene_paths, desc="Processing scenes"):
        results = process_scene(scene_path, args.future_folder, args.front_folder)

        for result in results:
            total_rooms += 1
            total_obbs += result.get('num_obbs', 0)

            if result['success']:
                total_success += 1
            else:
                total_failed += 1
                if result['message'] not in ['No assets found for this room']:
                    print(f"\n  Failed: {result['room_path']}: {result['message']}")

    # Summary
    batch_elapsed = time.time() - batch_start_time

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total scenes: {len(scene_paths)}")
    print(f"Total rooms: {total_rooms}")
    print(f"  Success: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"Total OBBs: {total_obbs}")
    print(f"Time: {batch_elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
