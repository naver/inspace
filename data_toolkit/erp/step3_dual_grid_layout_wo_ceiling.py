# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Step 3 (Layout): Convert layout_wo_ceiling mesh to O-Voxel (Flexible Dual Grid).

Simplified version of step3_dual_grid_erp.py that processes ONLY layout_wo_ceiling
(no individual assets).

IMPORTANT: step1_dump_mesh_layout_wo_ceiling.py normalizes layout mesh using
full_room_wo_ceiling's normalization_info.json (center, scale) so that layout
voxels share the same coordinate system as the full room.

Input structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        mesh_dumps/layout_wo_ceiling.pickle             # Normalized with full_room's center/scale

Output structure:
    datasets/ERP_3D_FRONT/{uuid}/{room_name}/
        dual_grid_{resolution}/layout_wo_ceiling.vxz

Logging:
    datasets/ERP_3D_FRONT_logs/step3_dual_grid_layout_wo_ceiling_{resolution}.json
    - Tracks processed rooms, success/failure status, timestamps
    - Enables resumable processing

Usage:
    python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512
    python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed
    python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --rank 0 --world_size 4
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
import o_voxel
from tqdm import tqdm
from datetime import datetime


class ProcessingLog:
    """Handles logging of processing progress to JSON file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.data = self._load()

    def _load(self) -> dict:
        """Load existing log or create new one."""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            'step': 'step3_dual_grid_layout_wo_ceiling',
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'summary': {
                'total_rooms': 0,
                'rooms_processed': 0,
                'rooms_failed': 0,
                'total_voxels': 0
            },
            'rooms': {}
        }

    def save(self):
        """Save log to file."""
        self.data['last_updated'] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_room_completed(self, room_key: str) -> bool:
        """Check if room has been successfully processed."""
        return room_key in self.data['rooms'] and self.data['rooms'][room_key].get('status') == 'completed'

    def log_room(self, room_key: str, result: dict):
        """Log processing result for a room."""
        self.data['rooms'][room_key] = {
            'status': 'completed' if result['room_processed'] else 'failed',
            'room_processed': result['room_processed'],
            'room_voxels': result['room_voxels'],
            'timestamp': datetime.now().isoformat()
        }

    def update_summary(self, total_rooms: int, rooms_processed: int, rooms_failed: int,
                       total_voxels: int):
        """Update summary statistics."""
        self.data['summary'] = {
            'total_rooms': total_rooms,
            'rooms_processed': rooms_processed,
            'rooms_failed': rooms_failed,
            'total_voxels': total_voxels
        }


def find_all_rooms(root: str) -> list:
    """Find all room directories in the dataset."""
    rooms = []
    for uuid_dir in sorted(os.listdir(root)):
        uuid_path = os.path.join(root, uuid_dir)
        if not os.path.isdir(uuid_path):
            continue
        for room_name in sorted(os.listdir(uuid_path)):
            room_path = os.path.join(uuid_path, room_name)
            if os.path.isdir(room_path) and os.path.exists(os.path.join(room_path, 'mesh_dumps')):
                rooms.append({
                    'uuid': uuid_dir,
                    'room_name': room_name,
                    'room_path': room_path
                })
    return rooms


def load_mesh_dump(pickle_path: str) -> dict:
    """Load mesh from pickle dump."""
    with open(pickle_path, 'rb') as f:
        dump = pickle.load(f)

    # Combine all objects
    start = 0
    vertices_list = []
    faces_list = []

    for obj in dump['objects']:
        if obj['vertices'].size == 0 or obj['faces'].size == 0:
            continue
        vertices_list.append(obj['vertices'])
        faces_list.append(obj['faces'] + start)
        start += len(obj['vertices'])

    if len(vertices_list) == 0:
        return None

    vertices = torch.from_numpy(np.concatenate(vertices_list, axis=0)).float()
    faces = torch.from_numpy(np.concatenate(faces_list, axis=0)).long()

    return {'vertices': vertices, 'faces': faces}


def mesh_to_dual_grid(vertices: torch.Tensor, faces: torch.Tensor, resolution: int) -> tuple:
    """
    Convert mesh to O-Voxel using Flexible Dual Grid.

    Returns:
        voxel_indices, dual_vertices, intersected
    """
    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices, faces,
        grid_size=resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
        timing=False
    )

    # Pack for storage
    dual_vertices = dual_vertices * resolution - voxel_indices
    assert torch.all(dual_vertices >= -1e-3) and torch.all(dual_vertices <= 1 + 1e-3), 'dual_vertices out of range'
    dual_vertices = torch.clamp(dual_vertices, 0, 1)
    dual_vertices = (dual_vertices * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)

    return voxel_indices, dual_vertices, intersected


def process_room(room_info: dict, resolution: int) -> dict:
    """
    Process a single room's layout_wo_ceiling mesh.

    Args:
        room_info: Dict with uuid, room_name, room_path
        resolution: O-Voxel resolution (e.g., 512)

    Returns:
        Dict with processing results
    """
    room_path = room_info['room_path']
    mesh_dumps_dir = os.path.join(room_path, 'mesh_dumps')
    output_dir = os.path.join(room_path, f'dual_grid_{resolution}')
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'uuid': room_info['uuid'],
        'room_name': room_info['room_name'],
        'room_processed': False,
        'room_voxels': 0
    }

    layout_pickle_path = os.path.join(mesh_dumps_dir, 'layout_wo_ceiling.pickle')
    layout_output_path = os.path.join(output_dir, 'layout_wo_ceiling.vxz')

    # Skip if output already exists
    if os.path.exists(layout_output_path):
        results['room_processed'] = True
        try:
            info = o_voxel.io.read_vxz_info(layout_output_path)
            results['room_voxels'] = info['num_voxel']
        except:
            pass
        return results

    # Check if input pickle exists
    if not os.path.exists(layout_pickle_path):
        print(f"  [Skip] {room_info['uuid']}/{room_info['room_name']}: layout_wo_ceiling.pickle not found")
        return results

    try:
        mesh_data = load_mesh_dump(layout_pickle_path)
        if mesh_data is None:
            print(f"  [Skip] {room_info['uuid']}/{room_info['room_name']}: empty mesh")
            return results

        # Vertices are already normalized in step1, use directly
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']

        # Convert to dual grid
        voxel_indices, dual_vertices, intersected = mesh_to_dual_grid(
            vertices, faces, resolution
        )

        # Save
        o_voxel.io.write_vxz(
            layout_output_path,
            voxel_indices,
            {'vertices': dual_vertices, 'intersected': intersected}
        )

        results['room_processed'] = True
        results['room_voxels'] = len(voxel_indices)

    except Exception as e:
        print(f"Error processing layout {room_info['uuid']}/{room_info['room_name']}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Convert layout_wo_ceiling mesh to O-Voxel for ERP_3D_FRONT dataset')
    parser.add_argument('--root', type=str, default='datasets/ERP_3D_FRONT',
                        help='Root directory of ERP_3D_FRONT dataset')
    parser.add_argument('--resolution', type=int, default=512,
                        help='O-Voxel resolution')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip rooms that are already logged as completed')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Save log every N rooms')
    args = parser.parse_args()

    args.root = 'figure_sample'
    args.resolution = 256

    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512
    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed
    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 0 --world_size 4
    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 1 --world_size 4
    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 2 --world_size 4
    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT --resolution 512 --skip_completed --rank 3 --world_size 4

    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 0 --world_size 4
    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 1 --world_size 4
    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 2 --world_size 4
    # python data_toolkit/erp/step3_dual_grid_layout_wo_ceiling.py --root datasets/ERP_3D_FRONT_test --resolution 512 --skip_completed --rank 3 --world_size 4


    # Initialize logging (outside dataset folder)
    log_suffix = f"_rank{args.rank}" if args.world_size > 1 else ""
    log_dir = args.root.rstrip('/') + '_logs'
    log_path = os.path.join(log_dir, f'step3_dual_grid_layout_wo_ceiling_{args.resolution}{log_suffix}.json')
    log = ProcessingLog(log_path)
    print(f"Logging to: {log_path}")

    # Find all rooms
    print("Finding rooms...")
    rooms = find_all_rooms(args.root)
    rooms.sort(key=lambda x: (x['uuid'], x['room_name']))

    total_rooms = len(rooms)
    print(f"Found {total_rooms} rooms")

    # Distribute across ranks
    start = len(rooms) * args.rank // args.world_size
    end = len(rooms) * (args.rank + 1) // args.world_size
    rooms = rooms[start:end]
    print(f"Processing {len(rooms)} rooms (rank {args.rank}/{args.world_size})")

    # Filter already completed rooms if requested
    if args.skip_completed:
        original_count = len(rooms)
        rooms = [r for r in rooms if not log.is_room_completed(f"{r['uuid']}/{r['room_name']}")]
        skipped = original_count - len(rooms)
        if skipped > 0:
            print(f"Skipping {skipped} already completed rooms")

    # Process rooms
    total_room_processed = 0
    total_room_failed = 0
    total_room_voxels = 0
    resolution = args.resolution

    for i, room_info in enumerate(tqdm(rooms, desc="Processing layout_wo_ceiling")):
        room_key = f"{room_info['uuid']}/{room_info['room_name']}"
        result = process_room(room_info, resolution)

        if result['room_processed']:
            total_room_processed += 1
            total_room_voxels += result['room_voxels']
        else:
            total_room_failed += 1

    #     log.log_room(room_key, result)

    #     if (i + 1) % args.log_interval == 0:
    #         log.update_summary(total_rooms, total_room_processed, total_room_failed,
    #                          total_room_voxels)
    #         log.save()

    # # Final log save
    # log.update_summary(total_rooms, total_room_processed, total_room_failed,
    #                   total_room_voxels)
    # log.save()

    print(f"\nSummary:")
    print(f"  Rooms processed: {total_room_processed}")
    print(f"  Rooms failed: {total_room_failed}")
    print(f"  Total voxels: {total_room_voxels}")
    print(f"  Avg voxels per room: {total_room_voxels / max(1, total_room_processed):.0f}")
    print(f"\nLog saved to: {log_path}")


if __name__ == '__main__':
    main()
