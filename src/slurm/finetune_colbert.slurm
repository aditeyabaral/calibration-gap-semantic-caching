#!/bin/bash

#SBATCH --job-name=finetune_colbert
#SBATCH --account=your-slurm-account
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:4
#SBATCH --open-mode=append
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/path/to/calibration-gap-semantic-caching
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=your-email@example.com

set -e

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"

nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the Python environment
source .venv/bin/activate

# Activate bashrc to get any additional environment variables
source ~/.bashrc

# Print environment info for debugging
echo "Python location and version: $(which python), $(python --version)"
echo "HF_HOME: $HF_HOME"
echo "CUDA_HOME: $CUDA_HOME"

# Point Hugging Face cache to a directory in scratch
export HF_HOME=/scratch/$USER/.cache/huggingface
hf auth whoami
echo "HF_HOME: $HF_HOME"

# Point CUDA toolkit
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "CUDA_HOME set to $CUDA_HOME"
nvcc --version

COMBINE_TRAIN_AND_VAL=true
WANDB_RUN_NAME=""

GPU_KEEPALIVE_PIDS=()
trap 'echo "Shutting down GPU keepalives..."; for pid in "${GPU_KEEPALIVE_PIDS[@]}"; do kill $pid 2>/dev/null; done; wait "${GPU_KEEPALIVE_PIDS[@]}" 2>/dev/null' EXIT

# Start GPU keepalive on all GPUs to maintain utilization above HPC cancellation threshold
echo "Starting GPU keepalive workers..."
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    KEEPALIVE_LOG=$(mktemp)
    python -c "
import signal, sys, time, torch
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
t = torch.randn(4096, 4096, device='cuda:${gpu_idx}')
print('GPU keepalive running on cuda:${gpu_idx}...', flush=True)
while True:
    torch.matmul(t, t)
    time.sleep(0.005)
" > "$KEEPALIVE_LOG" 2>&1 &
    GPU_KEEPALIVE_PIDS+=($!)
    echo "Waiting for GPU keepalive on cuda:${gpu_idx} to initialize..."
    until grep -q "GPU keepalive running" "$KEEPALIVE_LOG" 2>/dev/null; do
        sleep 1
    done
    echo "GPU keepalive confirmed running on cuda:${gpu_idx} (PID: ${GPU_KEEPALIVE_PIDS[-1]})."
done

# Run fine-tuning via accelerate (uses all available GPUs by default)
echo "Starting ColBERT fine-tuning..."
EXTRA_ARGS=""
if [ "$COMBINE_TRAIN_AND_VAL" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --combine-train-and-val"
fi
if [ -n "$WANDB_RUN_NAME" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --wandb-run-name $WANDB_RUN_NAME"
fi

accelerate launch src/reranker/finetune_colbert.py \
    --pretrained-model-path "lightonai/GTE-ModernColBERT-v1" \
    --finetuned-model-path "langcache-colbert-v2" \
    --query-length 512 \
    --document-length 512 \
    --dataset-version "v3" \
    --train-dataset-subsets "all" \
    --eval-split test \
    --batch-size 48 \
    --learning-rate 2e-4 \
    --epochs 5 \
    --eval-steps 50000 \
    --save-steps 50000 \
    --logging-steps 1000 \
    --save-total-limit 5 \
    --num-negatives 1 \
    --temperature 0.02 \
    --warmup-ratio 0.10 \
    --weight-decay 0.001 \
    --lr-scheduler-type linear \
    --optim adamw_torch \
    --output-dir /scratch/$USER/langcache-colbert-models \
    --device cuda \
    --seed 42 \
    $EXTRA_ARGS

if [[ ${#GPU_KEEPALIVE_PIDS[@]} -gt 0 ]]; then
    for pid in "${GPU_KEEPALIVE_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait "${GPU_KEEPALIVE_PIDS[@]}" 2>/dev/null || true
    GPU_KEEPALIVE_PIDS=()
fi

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
