#!/bin/bash

# SALMONN Captioner Inference Script
# Usage: bash my_inference.sh

# Set your GPU (change if needed)
export CUDA_VISIBLE_DEVICES=0

# Set local distributed training environment variables for torchrun
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST="localhost"

# Model paths (will download automatically from Hugging Face)
# Choose your model version:
MODEL="tsinghua-ee/video-SALMONN-2_plus_7B"    # Newest 7B model (recommended, ~16GB VRAM)
# MODEL="tsinghua-ee/video-SALMONN-2_plus_72B"   # Newest 72B model (best quality, ~80GB VRAM)
# MODEL="tsinghua-ee/video-SALMONN-2"            # Original 7B model

MODEL_BASE="lmms-lab/llava-onevision-qwen2-7b-ov"

# Configuration file with your media paths  
CONFIG_FILE="scripts/inference/my_inference.yaml"

echo "🎬 Starting SALMONN Captioner Inference..."
echo "📁 Using config: $CONFIG_FILE"
echo "🤖 Model: $MODEL"
echo "🔧 Base model: $MODEL_BASE"
echo "🖥️  Single GPU local inference mode"

# Navigate to video-SALMONN-2 root directory
cd ../../

# Add current directory to Python path so llava module can be found
export PYTHONPATH=$PWD:$PYTHONPATH

# Set cache directories to local writable paths
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export HF_HOME="$HOME/.cache/huggingface"

# Disable torch_xla which is causing compatibility issues
export USE_TORCH_XLA=False

# YAML file is now hardcoded in the demo mode
echo "🔄 Starting demo mode with hardcoded YAML config..."
echo "🐞 Debug: Passing --model $MODEL to override default checkpoint path..."
bash scripts/run.sh \
    --do_demo \
    --max_time 110 \
    --fps 1 \
    --model "$MODEL" \
    --model_base "$MODEL_BASE" \
    --add_time_token \
    --mm_pooling_position after \
    --audio_visual \
    --winqf_second 0.5

echo "✅ Inference completed!"
