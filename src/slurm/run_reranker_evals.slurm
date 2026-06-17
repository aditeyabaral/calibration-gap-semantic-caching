#!/bin/bash

#SBATCH --job-name=langcache_evals
#SBATCH --account=your-slurm-account
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:1
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

# Create Redis data, logs, tmp and lib directories if they don't exist
mkdir -p $SCRATCH/redis-data $SCRATCH/redis-logs $SCRATCH/redis-tmp $SCRATCH/redis-lib

# Trap to ensure Redis is stopped when the script exits
GPU_KEEPALIVE_PID=""
trap 'echo "Shutting down Redis and GPU keepalive..."; kill $REDIS_PID 2>/dev/null; kill $GPU_KEEPALIVE_PID 2>/dev/null; wait $REDIS_PID 2>/dev/null; wait $GPU_KEEPALIVE_PID 2>/dev/null' EXIT

# Download the Redis Stack image if it doesn't exist
if [ ! -f redis-stack.sif ]; then
    echo "Downloading Redis Stack Singularity image..."
    singularity pull redis-stack.sif docker://redis/redis-stack:latest
fi

# Start Redis server
singularity exec \
  --bind $SCRATCH/redis-data:/data \
  --bind $SCRATCH/redis-logs:/var/log/redis \
  --bind $SCRATCH/redis-tmp:/tmp \
  --bind $SCRATCH/redis-lib:/var/lib/redis-stack \
  redis-stack.sif \
  /opt/redis-stack/bin/redis-stack-server \
  --port 6379 \
  --dir /data \
  --logfile /var/log/redis/redis.log \
  --daemonize no &

REDIS_PID=$!

# Wait until Redis is ready
echo "Waiting for Redis to be ready..."
REDIS_READY=0
for i in $(seq 1 600); do
  if singularity exec redis-stack.sif redis-cli -p 6379 ping 2>/dev/null | grep -q PONG; then
    echo "Redis is ready."
    REDIS_READY=1
    break
  fi
  sleep 1
done

if [ $REDIS_READY -eq 0 ]; then
  echo "ERROR: Redis failed to start after 10 minutes. Check $SCRATCH/redis-logs/redis.log"
  exit 1
fi

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

# Run evaluations
echo "Starting reranker evaluations..."
bash src/shell/run_reranker_evals.sh

if [[ -n "$GPU_KEEPALIVE_PID" ]]; then
    kill "$GPU_KEEPALIVE_PID" 2>/dev/null || true
    wait "$GPU_KEEPALIVE_PID" 2>/dev/null || true
    GPU_KEEPALIVE_PID=""
fi

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
