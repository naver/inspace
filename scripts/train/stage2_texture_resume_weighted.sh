#!/bin/bash
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license


# Training script for Stage 2-2: Texture Generation (Asset-Aware)
# Resume from existing checkpoint
# Usage:
#   bash scripts/train/stage2_texture_resume_weighted.sh              # uniform sampling (default)
#   bash scripts/train/stage2_texture_resume_weighted.sh weighted      # weighted sampling (oversample large rooms)

# NCCL settings for stability
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes
export NCCL_DEBUG=WARN    # Show NCCL warnings

# Configuration paths
CONFIG="configs/gen/erp_slat_flow_imgshape2tex_asset_aware_bf16.json"
OUTPUT_DIR="results/erp_slat_flow_imgshape2tex_asset_aware_bf16_weight_sampling"
DATA_DIR="datasets/ERP_3D_FRONT"
EVAL_DATA_DIR="datasets/ERP_3D_FRONT_test"

# Resume settings
LOAD_DIR="results/erp_slat_flow_imgshape2tex_asset_aware_bf16_weight_sampling"
CKPT="latest"

# Sampling strategy: "uniform" (default) or "weighted" (oversample large rooms)
# SAMPLER=${1:-"uniform"}
SAMPLER="weighted"
# trellis2/datasets/erp_structured_latent.py -> adjust weight

# Multi-node settings (if needed)
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"12346"}

# GPU settings
# GPU_IDS='3,5,6,7'
# NUM_GPUS=4
GPU_IDS='4,5'
NUM_GPUS=2


# Print configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "  Stage: 2-2 (Texture Generation) — RESUME"
echo "  CONFIG: $CONFIG"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  LOAD_DIR: $LOAD_DIR"
echo "  CKPT: $CKPT"
echo "  DATA_DIR: $DATA_DIR"
echo "  EVAL_DATA_DIR: $EVAL_DATA_DIR"
echo "  SAMPLER: $SAMPLER"
echo "  GPU_IDS: $GPU_IDS"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "=========================================="
echo ""

# Run training command
CUDA_VISIBLE_DEVICES=$GPU_IDS python train.py \
  --config $CONFIG \
  --output_dir $OUTPUT_DIR \
  --load_dir $LOAD_DIR \
  --ckpt $CKPT \
  --data_dir $DATA_DIR \
  --eval_data_dir $EVAL_DATA_DIR \
  --num_gpus $NUM_GPUS \
  --gpu_ids $GPU_IDS \
  --master_port $MASTER_PORT \
  --sampler $SAMPLER
