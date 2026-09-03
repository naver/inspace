# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Dataset for 3D bounding box estimation from decoded voxel grids.

Each sample consists of:
- Voxel grid [1, 64, 64, 64] binary occupancy from voxels_64/full_room_wo_ceiling.ply
- GT bounding boxes [N, 7] from 3d_bounding_box/{room_name}_scene_data.npz
  Format: [cx, cy, cz, sx, sy, sz, rotation_yaw] in O-Voxel normalized space
"""

import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset


def load_ply_voxels(ply_path):
    """
    Load voxel coordinates from a binary little-endian PLY file.

    Returns:
        coords: [N, 3] float32 array of voxel centers in [-0.5, 0.5]
    """
    with open(ply_path, 'rb') as f:
        # Parse header
        num_vertices = 0
        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line == 'end_header':
                break
        # Read binary vertex data: N x (float32 x, float32 y, float32 z)
        data = f.read(num_vertices * 12)
        coords = np.frombuffer(data, dtype=np.float32).reshape(num_vertices, 3)
    return coords


def voxel_coords_to_grid(coords, resolution=64):
    """
    Convert voxel center coordinates to a dense binary occupancy grid.

    Args:
        coords: [N, 3] in [-0.5, 0.5] normalized space
        resolution: Grid resolution (default: 64)

    Returns:
        grid: [1, R, R, R] float32 tensor (1.0 = occupied, 0.0 = empty)
    """
    # Convert [-0.5, 0.5] -> [0, resolution-1] grid indices
    indices = ((coords + 0.5) * resolution).astype(np.int64)
    indices = np.clip(indices, 0, resolution - 1)

    grid = np.zeros((1, resolution, resolution, resolution), dtype=np.float32)
    grid[0, indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    return grid


class ERPBBoxDataset(Dataset):
    """
    Dataset for training a 3D bounding box estimator on indoor scenes.

    Loads decoded voxel grid (binary occupancy) and GT oriented bounding boxes.

    Args:
        data_root: Root directory (e.g., datasets/ERP_3D_FRONT)
        voxel_resolution: Voxel grid resolution (default: 64)
        max_objects: Maximum number of objects per scene for padding (default: 50)
        min_bbox_size: Minimum bbox extent to filter tiny objects (default: 0.005)
    """
    def __init__(
        self,
        data_root,
        voxel_resolution=64,
        max_objects=50,
        min_bbox_size=0.005,
        **kwargs,
    ):
        self.data_root = data_root
        self.voxel_resolution = voxel_resolution
        self.max_objects = max_objects
        self.min_bbox_size = min_bbox_size
        self.value_range = (0, 1)

        self.samples = self._find_samples()
        print(f'ERPBBoxDataset: Found {len(self.samples)} samples in {data_root}')

    def _find_samples(self):
        """Find all rooms that have both voxel_64 and 3D bounding box data."""
        samples = []
        voxel_dir_name = f'voxels_{self.voxel_resolution}'

        for uuid_dir in sorted(os.listdir(self.data_root)):
            uuid_path = os.path.join(self.data_root, uuid_dir)
            if not os.path.isdir(uuid_path):
                continue
            for room_name in sorted(os.listdir(uuid_path)):
                room_path = os.path.join(uuid_path, room_name)
                if not os.path.isdir(room_path):
                    continue

                voxel_path = os.path.join(
                    room_path, voxel_dir_name, 'full_room_wo_ceiling.ply'
                )
                bbox_dir = os.path.join(room_path, '3d_bounding_box')

                if not os.path.exists(voxel_path):
                    continue
                if not os.path.isdir(bbox_dir):
                    continue

                # Find bbox npz file
                bbox_path = os.path.join(bbox_dir, f'{room_name}_scene_data.npz')
                if not os.path.exists(bbox_path):
                    npz_files = [f for f in os.listdir(bbox_dir) if f.endswith('_scene_data.npz')]
                    if len(npz_files) == 0:
                        continue
                    bbox_path = os.path.join(bbox_dir, npz_files[0])

                samples.append({
                    'voxel_path': voxel_path,
                    'bbox_path': bbox_path,
                    'room_path': room_path,
                    'room_name': room_name,
                    'sample_id': f'{uuid_dir}/{room_name}',
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return (
            f'ERPBBoxDataset(root={self.data_root}, '
            f'samples={len(self.samples)}, '
            f'resolution={self.voxel_resolution}, '
            f'max_objects={self.max_objects})'
        )

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load voxel grid from PLY
        coords = load_ply_voxels(sample['voxel_path'])
        voxel_grid = voxel_coords_to_grid(coords, self.voxel_resolution)
        voxel_grid = torch.from_numpy(voxel_grid)  # [1, 64, 64, 64]

        # Load GT bounding boxes
        bbox_data = np.load(sample['bbox_path'], allow_pickle=True)
        obbs = bbox_data['obbs'].astype(np.float32)  # [N, 7]
        n_assets = int(bbox_data['n_assets']) if 'n_assets' in bbox_data else obbs.shape[0]

        # Filter tiny bboxes
        valid_mask = np.ones(n_assets, dtype=bool)
        for i in range(n_assets):
            sx, sy, sz = obbs[i, 3], obbs[i, 4], obbs[i, 5]
            if sx < self.min_bbox_size and sy < self.min_bbox_size and sz < self.min_bbox_size:
                valid_mask[i] = False

        obbs = obbs[valid_mask]
        n_valid = len(obbs)

        # Pad to max_objects
        gt_bboxes = torch.zeros(self.max_objects, 7, dtype=torch.float32)
        gt_mask = torch.zeros(self.max_objects, dtype=torch.bool)

        n_use = min(n_valid, self.max_objects)
        if n_use > 0:
            gt_bboxes[:n_use] = torch.from_numpy(obbs[:n_use])
            gt_mask[:n_use] = True

        return {
            'voxel_grid': voxel_grid,               # [1, 64, 64, 64]
            'gt_bboxes': gt_bboxes,                  # [max_objects, 7]
            'gt_mask': gt_mask,                      # [max_objects] bool
            'num_objects': n_use,
            'sample_id': sample['sample_id'],
        }

    @staticmethod
    def collate_fn(batch):
        """Custom collate that handles both tensors and non-tensor data."""
        return {
            'voxel_grid': torch.stack([b['voxel_grid'] for b in batch]),
            'gt_bboxes': torch.stack([b['gt_bboxes'] for b in batch]),
            'gt_mask': torch.stack([b['gt_mask'] for b in batch]),
            'num_objects': [b['num_objects'] for b in batch],
            'sample_id': [b['sample_id'] for b in batch],
        }
