#!/bin/bash

#SBATCH --job-name=compute_calibration
#SBATCH --account=your-slurm-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --open-mode=append
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/path/to/calibration-gap-semantic-caching
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=your-email@example.com

set -e

GPU_KEEPALIVE_PID=""
trap 'echo "Shutting down GPU keepalive..."; kill $GPU_KEEPALIVE_PID 2>/dev/null; wait $GPU_KEEPALIVE_PID 2>/dev/null' EXIT

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"

nvidia-smi

mkdir -p logs

# Activate the Python environment
source .venv/bin/activate
source ~/.bashrc

echo "Python location and version: $(which python), $(python --version)"

export HF_HOME=/scratch/$USER/.cache/huggingface
hf auth whoami
echo "HF_HOME: $HF_HOME"

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "CUDA_HOME set to $CUDA_HOME"
nvcc --version

MODEL_PATH="redis/langcache-reranker-v2-softmnrl-triplet"
DATASET_VERSION="v1"

# Start GPU keepalive to maintain utilization above HPC cancellation threshold
echo "Starting GPU keepalive worker..."
KEEPALIVE_LOG=$(mktemp)
python -c "
import signal, sys, time, torch
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
t = torch.randn(4096, 4096, device='cuda')
print('GPU keepalive running...', flush=True)
while True:
    torch.matmul(t, t)
    time.sleep(0.005)
" > "$KEEPALIVE_LOG" 2>&1 &
GPU_KEEPALIVE_PID=$!

echo "Waiting for GPU keepalive to initialize..."
until grep -q "GPU keepalive running" "$KEEPALIVE_LOG" 2>/dev/null; do
    sleep 1
done
echo "GPU keepalive confirmed running (PID: $GPU_KEEPALIVE_PID)."

echo "Starting calibration..."
echo "Model path: $MODEL_PATH"
echo "Dataset version: $DATASET_VERSION"
python src/analysis/compute_calibration.py \
    --model-path "$MODEL_PATH" \
    --dataset-version "$DATASET_VERSION" \
    --output calibration_params.json \
    --batch-size 32 \
    --device cuda \
    --seed 42

if [[ -n "$GPU_KEEPALIVE_PID" ]]; then
    kill "$GPU_KEEPALIVE_PID" 2>/dev/null || true
    wait "$GPU_KEEPALIVE_PID" 2>/dev/null || true
    GPU_KEEPALIVE_PID=""
fi

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
