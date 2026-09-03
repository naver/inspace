#!/bin/bash
# InSpace
# Copyright (c) 2026 NAVER Corp.
# MIT license


# Training script for CenterPoint-style 3D Bounding Box Estimator v2
# v2 improvements:
#   - Multi-bin rotation (12 bins + residual regression)
#   - IoU-based NMS post-processing
#   - Dense rotation supervision (within Gaussian radius)
#   - Larger NMS kernel (7), broader Gaussians (min_sigma=2.0, max_sigma=4.0)
#
# Usage: bash scripts/train/bbox.sh

# NCCL settings for stability
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN

# Configuration paths
CONFIG="configs/bbox/erp_bbox_centerpoint_v2.json"
OUTPUT_DIR="results/bbox_centerpoint_v2"
DATA_DIR="datasets/ERP_3D_FRONT"
EVAL_DATA_DIR="datasets/ERP_3D_FRONT_test"

# Multi-node settings
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"12348"}

# GPU settings - CenterPoint is lightweight, 1 GPU is sufficient
GPU_IDS='2'
NUM_GPUS=1

# Print configuration
echo "=========================================="
echo "CenterPoint 3D BBox Estimator v2 Training"
echo "=========================================="
echo "  CONFIG: $CONFIG"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  DATA_DIR: $DATA_DIR"
echo "  EVAL_DATA_DIR: $EVAL_DATA_DIR"
echo "  GPU_IDS: $GPU_IDS"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "=========================================="
echo "  v2 improvements:"
echo "    - Multi-bin rotation (12 bins)"
echo "    - IoU NMS (threshold=0.3)"
echo "    - Dense rotation supervision"
echo "    - NMS kernel=7, sigma=[2.0, 4.0]"
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
