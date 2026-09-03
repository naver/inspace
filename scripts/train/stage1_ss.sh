#!/bin/bash
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license


# Training script for train.py
# Usage: bash scripts/train/stage1_ss.sh

# Set environment variables
# export CUDA_VISIBLE_DEVICES=4,5
# export ATTN_BACKEND=sdpa
# export CUDA_LAUNCH_BLOCKING=1  # Disable for better performance (only enable for debugging)

# NCCL settings for stability
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes
export NCCL_DEBUG=WARN    # Show NCCL warnings

# Configuration paths
CONFIG="configs/gen/erp_ss_flow_img_dit_L_16l8_bf16_spatial.json"
OUTPUT_DIR="results/erp_ss_flow_img_dit_L_16l8_bf16_spatial"
DATA_DIR="datasets/ERP_3D_FRONT"
EVAL_DATA_DIR="datasets/ERP_3D_FRONT_test"

# LOAD_DIR=""  # Uncomment and set if resuming from checkpoint
# CKPT=""      # Uncomment and set if resuming from checkpoint

# Multi-node settings (if needed)
MASTER_ADDR=${MASTER_ADDR:-"localhost"}  # Default to localhost if not set
MASTER_PORT=${MASTER_PORT:-"12345"}       # Default to 12345 if not set

# GPU settings
GPU_IDS='2,3,5,6,7'  # Change this to your desired GPU numbers (excluding GPU 4 which has issues)
NUM_GPUS=5  # Number of GPUs per node

# Print configuration
echo "=========================================="
echo "Training Configuration:"
echo "=========================================="
echo "  CONFIG: $CONFIG"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  DATA_DIR: $DATA_DIR"
echo "  GPU_IDS: $GPU_IDS"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "=========================================="
echo ""

# Run training command
CUDA_VISIBLE_DEVICES=$GPU_IDS python train.py \
  --config $CONFIG \
  --output_dir $OUTPUT_DIR \
  --data_dir $DATA_DIR \
  --eval_data_dir $EVAL_DATA_DIR \
  --num_gpus $NUM_GPUS \
  --gpu_ids $GPU_IDS \
  --master_port $MASTER_PORT

