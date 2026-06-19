#!/bin/bash

# Configuration
SEED=42
DEVICE="cuda"
TOP_K=50
DATASET_VERSION="v3"
REDIS_HOST="localhost"
REDIS_PORT=6379

# Retrievers (bi-encoder models)
RETRIEVERS=(
    # langcache retrievers
    "redis/langcache-embed-v1"
    "redis/langcache-embed-v2"
    "redis/langcache-embed-v3-small"
    # snowflake retrievers
    "Snowflake/snowflake-arctic-embed-m-v2.0"
    # BAAI retrievers
    "BAAI/bge-base-en-v1.5"
    # intfloat retrievers
    "intfloat/e5-base-v2"
    # nomic retrievers
    "nomic-ai/nomic-embed-text-v1.5"
    # jina retrievers
    "jinaai/jina-embeddings-v2-base-en"
    # baseline retriever
    "Alibaba-NLP/gte-modernbert-base"
)

# Cross-encoder rerankers
CROSSENCODER_RERANKERS=(
    # v1 rerankers
    "redis/langcache-reranker-v1-bce"
    "redis/langcache-reranker-v1-mnrl"
    # v2 rerankers
    "redis/langcache-reranker-v2-bce"
    "redis/langcache-reranker-v2-mnrl"
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

for RETRIEVER in "${RETRIEVERS[@]}"; do
    # Flush cache once per biencoder so embeddings are re-populated correctly.
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
done

echo "All evaluations complete."
