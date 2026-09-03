#!/bin/bash
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license

# SDEdit inversion (alpha=0.5) + Predicted 3D bbox (CenterPoint v2)
CUDA_VISIBLE_DEVICES=7 python eval/pipeline/eval_pipeline.py \
    --noise_mode sdedit \
    --sdedit_alpha 0.5 \
    --bbox_mode predicted \
    --output_dir evals/stage12_pipeline/sdedit0.5_predicted \
    --max_samples -1 \
    --num_vis -1 \
    --save_concat \
    --gpu_id 0
