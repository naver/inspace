#!/bin/bash
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

# Stage 2-2 Texture: SDEdit + Predicted bbox (first 200 samples only)
# Run AFTER shape generation (shape_latent.npz must exist)
# Uses --skip_existing to reuse existing shape results
CUDA_VISIBLE_DEVICES=6 python eval/pipeline/eval_pipeline.py \
    --noise_mode sdedit \
    --sdedit_alpha 0.7 \
    --bbox_mode predicted \
    --enable_texture \
    --output_dir evals/stage12_pipeline/sdedit0.7_predicted \
    --max_samples 200 \
    --num_vis 200 \
    --save_concat \
    --max_meshes 200 \
    --skip_existing \
    --gpu_id 0
