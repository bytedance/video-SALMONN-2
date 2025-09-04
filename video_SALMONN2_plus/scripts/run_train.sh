#!/bin/bash

# poetry shell
# Set working directory to the parent of scripts
cd "$(cd $(dirname $0); pwd)/.."

# Set environment variables for better logging
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/eporat/txt2img:$PYTHONPATH
export WANDB_API_KEY=${WAND_API_KEY:-$WANDB_API_KEY}
export WANDB_ENTITY="lightricks"
export WANDB_PROJECT="txt2img"

# Run training with full loggingx output
echo "Starting training with full logging..."
echo "Output will be saved to training.log"
echo "Press Ctrl+C to stop training"
echo "----------------------------------------"

poetry run bash scripts/train.sh 2>&1 | tee training.log