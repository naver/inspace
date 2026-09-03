#!/bin/bash
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

# Random noise + GT 3D bbox
CUDA_VISIBLE_DEVICES=1 python eval/pipeline/eval_pipeline.py \
    --noise_mode random \
    --bbox_mode gt \
    --enable_texture \
    --output_dir evals/stage12_pipeline/random_gt \
    --max_samples 200 \
    --num_vis 200 \
    --save_concat \
    --max_meshes 200 \
    --skip_existing \
    --gpu_id 0
