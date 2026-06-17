#!/bin/bash

#SBATCH --job-name=langcache_latency
#SBATCH --account=your-slurm-account
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --open-mode=append
#SBATCH --error=logs/%x_%j.err
#SBATCH --output=logs/%x_%j.out
#SBATCH --chdir=/path/to/calibration-gap-semantic-caching
#SBATCH --mail-type=FAIL,END,TIME_LIMIT
#SBATCH --mail-user=your-email@example.com

set -eo pipefail

echo "############### Run Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
echo "Current working directory: $(pwd)"

# Activate the Python environment
source .venv/bin/activate
source ~/.bashrc

# Configuration
RESULTS_DIR="results/k50/raw"
PLOTS_DIR="results/k50/latency"
WORKERS=16
OUTPUT="results/k50/latency/latency_metrics.json"

mkdir -p "$PLOTS_DIR"

echo "--- analyze_latency.py ---"
uv run python -m src.analysis.analyze_latency \
    --results-dir "$RESULTS_DIR" \
    --plots-dir   "$PLOTS_DIR" \
    --output      "$OUTPUT" \
    --workers     "$WORKERS" \
    2>&1 | tee "${OUTPUT%.json}.log"

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
