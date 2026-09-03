# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

# read vxz file and convert to PLY point cloud
import o_voxel
import argparse
import torch
import utils3d
import os
import glob

'''
Info:
Change vxz file to Point Cloud:
- dual_grid_{resolution}_ply/full_room_wo_ceiling_pointcloud.ply: full room without ceiling
- dual_grid_{resolution}_ply/layout_wo_ceiling_pointcloud.ply: layout (floor + walls) without ceiling
- dual_grid_{resolution}_ply/individual_assets_room_coord/asset_name_pointcloud.ply: individual assets in room coordinate

Usage
python data_dual_grid_to_ply.py --vxz_dir /path/to/datasets/ERP_3D_FRONT_test --resolution 256
python data_dual_grid_to_ply.py --vxz_dir /path/to/room/dual_grid_512 --resolution 512

Output:
- dual_grid_{resolution}_ply/full_room_wo_ceiling_pointcloud.ply
- dual_grid_{resolution}_ply/layout_wo_ceiling_pointcloud.ply
- dual_grid_{resolution}_ply/individual_assets_room_coord/asset_name_pointcloud.ply
'''

parser = argparse.ArgumentParser()
parser.add_argument('--vxz_dir', type=str, default=None, help='Specific dual_grid_{resolution} directory path, or root directory to search all')
parser.add_argument('--resolution', type=int, default=256, help='Voxel grid resolution (e.g., 256, 512)')
args = parser.parse_args()

args.vxz_dir = "figure_sample"
args.resolution = 256

# Default to root directory if not provided
if args.vxz_dir is None:
    args.vxz_dir = "datasets/_ERP_3D_FRONT_before/ERP_3D_FRONT_test"

resolution = args.resolution
dual_grid_name = f'dual_grid_{resolution}'
dual_grid_ply_name = f'dual_grid_{resolution}_ply'

def vxz_to_ply(vxz_path, output_path, resolution=256):
    """Convert vxz file to PLY point cloud."""
    # Read vxz file
    coords, attr = o_voxel.io.read_vxz(vxz_path, num_threads=4)
    
    # Convert voxel grid coordinates to actual 3D positions [-0.5, 0.5]
    positions = (coords.float() / resolution) - 0.5  # [N, 3] in [-0.5, 0.5]
    positions_np = positions.cpu().numpy()
    
    # Use position-based colors (RGB from normalized XYZ coordinates)
    # Map [-0.5, 0.5] to [0, 1] then to [0, 255]
    colors_np = ((positions_np + 0.5) * 255).astype('uint8')
    
    # Save PLY file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils3d.io.write_ply(
        output_path,
        positions_np,
        vertex_colors=colors_np,
    )
    print(f"Saved: {output_path} ({len(positions_np)} points)")

def process_dual_grid_dir(dual_grid_dir):
    """Process a single dual_grid_256 directory."""
    print(f"\nProcessing: {dual_grid_dir}")
    
    # Create output directory (same level as dual_grid_{resolution})
    output_base_dir = os.path.join(os.path.dirname(dual_grid_dir), dual_grid_ply_name)
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process full_room_wo_ceiling.vxz
    room_vxz_path = os.path.join(dual_grid_dir, 'full_room_wo_ceiling.vxz')
    if os.path.exists(room_vxz_path):
        output_name = 'full_room_wo_ceiling_pointcloud.ply'
        ply_path = os.path.join(output_base_dir, output_name)
        vxz_to_ply(room_vxz_path, ply_path, resolution)

    # Process layout_wo_ceiling.vxz
    layout_vxz_path = os.path.join(dual_grid_dir, 'layout_wo_ceiling.vxz')
    if os.path.exists(layout_vxz_path):
        output_name = 'layout_wo_ceiling_pointcloud.ply'
        ply_path = os.path.join(output_base_dir, output_name)
        vxz_to_ply(layout_vxz_path, ply_path, resolution)

    # Process individual_assets_room_coord folder
    assets_dir = os.path.join(dual_grid_dir, 'individual_assets_room_coord')
    if os.path.exists(assets_dir):
        assets_output_dir = os.path.join(output_base_dir, 'individual_assets_room_coord')
        os.makedirs(assets_output_dir, exist_ok=True)
        
        vxz_files = glob.glob(os.path.join(assets_dir, '*.vxz'))
        if len(vxz_files) > 0:
            print(f"  Processing {len(vxz_files)} asset files...")
            
            for vxz_path in vxz_files:
                asset_name = os.path.splitext(os.path.basename(vxz_path))[0]
                ply_path = os.path.join(assets_output_dir, f'{asset_name}_pointcloud.ply')
                vxz_to_ply(vxz_path, ply_path, resolution)
    
    print(f"  ✓ Completed: {output_base_dir}")

# Check if args.vxz_dir is a specific dual_grid_{resolution} directory or root directory
if os.path.basename(args.vxz_dir) == dual_grid_name and os.path.exists(args.vxz_dir):
    # Process single directory
    process_dual_grid_dir(args.vxz_dir)
    print(f"\nProcessed single directory: {args.vxz_dir}")
else:
    # Find all dual_grid_{resolution} directories
    dual_grid_dirs = []
    for root, dirs, files in os.walk(args.vxz_dir):
        if dual_grid_name in dirs:
            dual_grid_path = os.path.join(root, dual_grid_name)
            # Check if it contains vxz files
            if os.path.exists(os.path.join(dual_grid_path, 'full_room_wo_ceiling.vxz')):
                dual_grid_dirs.append(dual_grid_path)

    print(f"Found {len(dual_grid_dirs)} {dual_grid_name} directories to process")

    # Process each directory
    for dual_grid_dir in dual_grid_dirs:
        try:
            process_dual_grid_dir(dual_grid_dir)
        except Exception as e:
            print(f"  Error processing {dual_grid_dir}: {e}")
            continue

    print(f"\nCompleted processing {len(dual_grid_dirs)} directories")


# print("Attributes:", list(attr.keys()))
# print(f"coords shape: {coords.shape}, dtype: {coords.dtype}")
# print(f"coords range: [{coords.min()}, {coords.max()}]")

# # Inspect each attribute
# for key, value in attr.items():
#     print(f"{key} shape: {value.shape}, dtype: {value.dtype}")
#     if key == 'vertices':
#         print(f"  vertices range: [{value.min()}, {value.max()}] (0-255, normalize to [0,1])")
#     elif key == 'intersected':
#         print(f"  intersected range: [{value.min()}, {value.max()}] (0-7)")

# coords
# tensor([[ 68,   0,  75],
#         [ 68,   1,  75],
#         [ 69,   0,  75],
#         ...,
#         [188, 253, 180],
#         [188, 254, 180],
#         [188, 255, 180]], dtype=torch.int32)
# coords.shape
# torch.Size([196206, 3])
# coords.min()
# tensor(0, dtype=torch.int32)
# coords.max()
# tensor(255, dtype=torch.int32)
