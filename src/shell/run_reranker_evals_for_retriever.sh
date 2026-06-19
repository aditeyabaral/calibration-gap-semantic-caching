#!/bin/bash

# Usage: run_reranker_evals_for_retriever.sh <retriever> <redis_port> [top_k]
RETRIEVER="$1"
REDIS_PORT="$2"
TOP_K="${3:-50}"

if [ -z "$RETRIEVER" ] || [ -z "$REDIS_PORT" ]; then
    echo "Usage: $0 <retriever> <redis_port> [top_k]"
    exit 1
fi

# Configuration
SEED=42
DEVICE="cuda"
DATASET_VERSION="v3"
REDIS_HOST="localhost"

# Cross-encoder rerankers
CROSSENCODER_RERANKERS=(
    # v1 rerankers
    "redis/langcache-reranker-v1"
    "redis/langcache-reranker-v1-softmnrl-triplet"
    # v2 rerankers
    "redis/langcache-reranker-v2-modernbert-bce-eps0.5"
    "redis/langcache-reranker-v2-softmnrl-triplet"
    # MS MARCO reranker
    "cross-encoder/ms-marco-MiniLM-L12-v2"
    # baseline reranker
    "Alibaba-NLP/gte-reranker-modernbert-base"
)

# ColBERT rerankers
COLBERT_RERANKERS=(
    "lightonai/ColBERT-Zero"
    "lightonai/GTE-ModernColBERT-v1"
    "lightonai/Reason-ModernColBERT"
    "colbert-ir/colbertv2.0"
)

# Flush cache on the first run so embeddings are re-populated correctly.
# Subsequent rerankers on the same biencoder reuse the cached embeddings.
FLUSH_FLAG="--flush-cache"

# Cross-encoder runs
for RERANKER in "${CROSSENCODER_RERANKERS[@]}"; do
    echo "retriever   : $RETRIEVER"
    echo "reranker    : $RERANKER (crossencoder)"

    uv run src/eval/eval_reranker.py \
        --biencoder-model-path "$RETRIEVER" \
        --reranker-model-path "$RERANKER" \
        --reranker-type "crossencoder" \
        --dataset-version "$DATASET_VERSION" \
        --top-k "$TOP_K" \
        --redis-host "$REDIS_HOST" \
        --redis-port "$REDIS_PORT" \
        --device "$DEVICE" \
        --seed "$SEED" \
        $FLUSH_FLAG
    echo "Completed evaluation for retriever $RETRIEVER with cross-encoder reranker $RERANKER"
    echo "---------------------------------------------"

    FLUSH_FLAG=""  # only flush on the first run per biencoder
done

# ColBERT runs
for RERANKER in "${COLBERT_RERANKERS[@]}"; do
    echo "retriever   : $RETRIEVER"
    echo "reranker    : $RERANKER (colbert)"

    uv run src/eval/eval_reranker.py \
        --biencoder-model-path "$RETRIEVER" \
        --reranker-model-path "$RERANKER" \
        --reranker-type "colbert" \
        --dataset-version "$DATASET_VERSION" \
        --top-k "$TOP_K" \
        --redis-host "$REDIS_HOST" \
        --redis-port "$REDIS_PORT" \
        --device "$DEVICE" \
        --seed "$SEED" \
        $FLUSH_FLAG
    echo "Completed evaluation for retriever $RETRIEVER with colbert reranker $RERANKER"
    echo "---------------------------------------------"
done

echo "Completed all evaluations for retriever $RETRIEVER"
echo "=========================================================================================="
