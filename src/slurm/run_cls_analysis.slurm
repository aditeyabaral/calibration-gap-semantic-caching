#!/bin/bash

#SBATCH --job-name=langcache_cls
#SBATCH --account=your-slurm-account
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=3:00:00
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
WORKERS=16

# Calibration: set CALIBRATION to empty string to disable, or a path to enable.
# CALIBRATION_METHOD: "temperature" or "platt" (only used when CALIBRATION is set).
CALIBRATION=""
CALIBRATION_METHOD=""

# Derive output subdir from calibration setting
if [ -n "$CALIBRATION" ]; then
    CALIB_SUBDIR="$CALIBRATION_METHOD"
    calib_args=(--calibration "$CALIBRATION" --calibration-method "$CALIBRATION_METHOD")
else
    CALIB_SUBDIR="no-calibration"
    calib_args=()
fi

PLOTS_DIR="results/k50/cls/$CALIB_SUBDIR"
OUTPUT_METRICS="results/k50/cls/$CALIB_SUBDIR/cls_metrics.json"

mkdir -p "$PLOTS_DIR"

echo "--- analyze_cls.py ---"
uv run python -m src.analysis.analyze_cls \
    --results-dir "$RESULTS_DIR" \
    --plots-dir "$PLOTS_DIR" \
    "${calib_args[@]}" \
    --output "$OUTPUT_METRICS" \
    --workers "$WORKERS" \
    2>&1 | tee "${OUTPUT_METRICS%.json}.log"

echo "############### End Log: $(date +%Y-%m-%d_%H:%M:%S) ###############"
