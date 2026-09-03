# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

# read npz file
import numpy as np
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--npz_path', type=str, required=True)
args = parser.parse_args()
# args.npz_path = "datasets/ERP_3D_FRONT_test/00ad8345-45e0-45b3-867d-4a3c88c2517a/MasterBedroom-46277/shape_latents/shape_enc_next_dc_f16c32_fp16_256/full_room_wo_ceiling.npz" # 0-15
# args.npz_path = "datasets/ERP_3D_FRONT_test/00ad8345-45e0-45b3-867d-4a3c88c2517a/MasterBedroom-46277/shape_latents/shape_enc_next_dc_f16c32_fp16_256/individual_assets_room_coord/bed_king-size_bed_12c0a7d0_inst000.npz"

# args.npz_path = "datasets/ERP_3D_FRONT_test/ea3a121c-0c63-4c90-b675-69cf500dd635/LivingDiningRoom-1127/shape_latents/shape_enc_next_dc_f16c32_fp16_512/full_room_wo_ceiling.npz"
# npz['coords'].shape = (1933, 3), npz['feats'].shape = (1933, 32)

args.npz_path = "datasets/ERP_3D_FRONT_test/ea3a121c-0c63-4c90-b675-69cf500dd635/LivingDiningRoom-1127/shape_latents/shape_enc_next_dc_f16c32_fp16_256/full_room_wo_ceiling.npz"
# npz['coords'].shape = (458, 3), npz['feats'].shape = (458, 32)


# args.npz_path = "datasets/ERP_3D_FRONT_test/ea5eb904-675c-4ca8-a2e4-dd042ce31d04/DiningRoom-217315/ss_latents/ss_enc_conv3d_16l8_fp16_64/full_room_wo_ceiling.npz"
# npz['z'].shape = (8, 16, 16, 16)

# 3D bounding box
# args.npz_path = "datasets/ERP_3D_FRONT_test/00ad8345-45e0-45b3-867d-4a3c88c2517a/MasterBedroom-46277/3d_bounding_box/MasterBedroom-46277_scene_data.npz"

npz = np.load(args.npz_path, allow_pickle=True)
print("Keys:", npz.keys())


# list(npz.keys())
# ['obbs', 'asset_jids', 'asset_uids', 'asset_categories', 'asset_filenames', 'asset_names', 'wall_obbs', 'floor_polygon', 'floor_height', 'floor_z', 'ceiling_polygon', 'ceiling_z', 'ceiling_height', 'norm_center', 'norm_scale']







# Option 1: direct key access
# feats = npz['feats']
# coords = npz['coords']
# print(f"feats shape: {feats.shape}, dtype: {feats.dtype}")
# print(f"coords shape: {coords.shape}, dtype: {coords.dtype}")

# Option 2: use get() (safe when the key is missing)
# feats = npz.get('feats')
# coords = npz.get('coords')

# Option 3: dict-style access
# feats = npz['feats']
# coords = npz['coords']

# Close the file (optional; a with statement closes it automatically)
# npz.close()

 
# npz['feats'].shape -> (500, 32) 
# npz['coords'].shape -> (500, 3) # 0-15 # It seems like 16x16x16 voxel grid