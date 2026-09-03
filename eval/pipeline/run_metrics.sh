#!/bin/bash
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

# Compute metrics for all 4 eval configs
# Voxel IoU: instant (no GPU decoder needed)
# Chamfer/F1: needs GPU for shape_dec (~1s/sample)
# 2D metrics: from vis_concat images

METRICS="voxel_iou chamfer f1"  # add "psnr ssim lpips" after vis is generated

for config in random_gt sdedit0.3_predicted sdedit0.5_predicted sdedit0.7_predicted; do
    pred_dir="evals/stage12_pipeline/$config"
    if [ ! -d "$pred_dir" ]; then
        echo "Skipping $config (not found)"
        continue
    fi
    echo "=========================================="
    echo "Computing metrics for: $config"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0 python eval/pipeline/compute_metrics.py \
        --pred_dir "$pred_dir" \
        --data_dir datasets/ERP_3D_FRONT_test \
        --metrics $METRICS \
        --gpu_id 0
done
