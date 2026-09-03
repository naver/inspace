#!/bin/bash
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

# Reconstruct GT meshes from latent NPZ files (run once, shared across all eval configs)
# Output: evals/gt_recon/{scene_id}/{room_id}/meshes/{scene,layout,assets/*.glb}

CUDA_VISIBLE_DEVICES=5 python eval/pipeline/recon_gt.py \
    --data_dir datasets/ERP_3D_FRONT_test \
    --output_dir evals/gt_recon \
    --enable_texture \
    --skip_existing \
    --max_meshes 500 \
    --gpu_id 0
