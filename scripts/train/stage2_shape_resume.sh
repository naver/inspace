#!/bin/bash
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license


# Training script for Stage 2 shape generation - CONTINUE from checkpoint
# Changes from original:
#   1. Hybrid self-attention: whole-scale self-attn + grouped cross-attn
#   2. Resumes from checkpoint automatically (--load_dir)
#   3. Optional weighted sampling (--sampler weighted) to oversample large rooms
#
# Usage:
#   bash scripts/train/stage2_shape_resume.sh              # uniform sampling (default)
#   bash scripts/train/stage2_shape_resume.sh weighted      # weighted sampling (oversample large rooms)

# NCCL settings for stability
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes
export NCCL_DEBUG=WARN    # Show NCCL warnings

# Configuration paths
CONFIG="configs/gen/erp_slat_flow_img2shape_asset_aware_bf16.json"
OUTPUT_DIR="results/erp_slat_flow_img2shape_asset_aware_bf16"
DATA_DIR="datasets/ERP_3D_FRONT"
EVAL_DATA_DIR="datasets/ERP_3D_FRONT_test"

# Resume from existing checkpoint (auto-detects latest)
LOAD_DIR="results/erp_slat_flow_img2shape_asset_aware_bf16"

# Sampling strategy: "uniform" (default) or "weighted" (oversample large rooms)
# SAMPLER=${1:-"uniform"}
SAMPLER="weighted"

# Multi-node settings (if needed)
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"12345"}

# GPU settings
GPU_IDS='1,2,3,5,6,7'  # Change this to your desired GPU numbers
NUM_GPUS=6

# Print configuration
echo "=========================================="
echo "Training Configuration (CONTINUE from ckpt):"
echo "=========================================="
echo "  CONFIG: $CONFIG"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  DATA_DIR: $DATA_DIR"
echo "  EVAL_DATA_DIR: $EVAL_DATA_DIR"
echo "  LOAD_DIR: $LOAD_DIR (auto-detect latest ckpt)"
echo "  SAMPLER: $SAMPLER"
echo "  GPU_IDS: $GPU_IDS"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "=========================================="
echo "  Changes: hybrid self-attention (whole-scale self + grouped cross)"
echo "=========================================="
echo ""

# Run training command
CUDA_VISIBLE_DEVICES=$GPU_IDS python train.py \
  --config $CONFIG \
  --output_dir $OUTPUT_DIR \
  --data_dir $DATA_DIR \
  --eval_data_dir $EVAL_DATA_DIR \
  --load_dir $LOAD_DIR \
  --num_gpus $NUM_GPUS \
  --gpu_ids $GPU_IDS \
  --master_port $MASTER_PORT \
  --sampler $SAMPLER
