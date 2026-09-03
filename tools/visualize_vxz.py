# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

"""
Convert VXZ (Dual Grid) files to PLY mesh for visualization.

Usage:
    python visualize_vxz.py <input.vxz> <output.ply>
    python visualize_vxz.py datasets/Toys4k/dual_grid_256/abc123.vxz output.ply
"""

import sys
import argparse
import numpy as np
import o_voxel
from o_voxel.convert import flexible_dual_grid_to_mesh


def vxz_to_ply(vxz_path, ply_path, grid_size=256):
    """
    Convert VXZ dual grid file to PLY mesh.

    Args:
        vxz_path: Path to input .vxz file
        ply_path: Path to output .ply file
        grid_size: Grid resolution (default: 256)
    """
    print(f"Reading VXZ file: {vxz_path}")

    # Read VXZ file
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)

    print(f"Loaded {coords.shape[0]} dual grid vertices")
    print(f"Attributes: {list(attr.keys())}")

    # Extract vertex positions and intersected flags
    vertices = attr['vertices'].cpu().numpy().astype(np.float32) / 255.0  # Normalize to [0, 1]
    intersected = attr['intersected'].cpu().numpy()

    # Convert to boolean intersected flags (3 bits)
    intersected_bool = np.stack([
        intersected % 2,
        intersected // 2 % 2,
        intersected // 4 % 2,
    ], axis=-1).astype(bool)

    print(f"Converting to mesh (grid_size={grid_size})...")

    # Convert dual grid to mesh
    # Use default quad_lerp=0.5 for all vertices
    quad_lerp = np.full((vertices.shape[0], 1), 0.5, dtype=np.float32)

    mesh_vertices, mesh_faces = flexible_dual_grid_to_mesh(
        coords,
        vertices,
        intersected_bool,
        quad_lerp,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        grid_size=grid_size,
        train=False
    )

    print(f"Generated mesh: {mesh_vertices.shape[0]} vertices, {mesh_faces.shape[0]} faces")

    # Save as PLY
    print(f"Saving to: {ply_path}")
    save_ply(ply_path, mesh_vertices, mesh_faces)

    print("Done!")


def save_ply(filepath, vertices, faces):
    """
    Save mesh as PLY file.

    Args:
        filepath: Output PLY file path
        vertices: Vertex positions [N, 3]
        faces: Face indices [F, 3]
    """
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {faces.shape[0]}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        # Vertices
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")

        # Faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert VXZ dual grid to PLY mesh')
    parser.add_argument('input', type=str, help='Input .vxz file path')
    parser.add_argument('output', type=str, help='Output .ply file path')
    parser.add_argument('--grid_size', type=int, default=256,
                        help='Grid resolution (default: 256)')

    args = parser.parse_args()

    vxz_to_ply(args.input, args.output, args.grid_size)
