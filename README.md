# Semantic Cache Re-ranking Evaluation

Official code for the paper **"Closing the Calibration Gap in Semantic Caching."**

If you use this code, the models, or the datasets, please cite:

```bibtex
@article{baral2026calibration,
  title   = {Closing the Calibration Gap in Semantic Caching},
  author  = {Baral, Aditeya and Ralev, Radoslav and Zhechev, Iliya Sotirov and Rajamohan, Srijith and Agarwal, Jen},
  year    = {2026},
  eprint  = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
<!-- TODO: replace the arXiv eprint ID once available. -->

Semantic caching cuts LLM inference costs by serving a cached response when a new query is *semantically* similar to a previously seen one. Whether a cache should fire is decided by a **score threshold**, yet semantic-cache retrievers and re-rankers are almost always selected by **PR-AUC** — a threshold-agnostic metric that says nothing about whether those scores are *usable at a fixed threshold*. This repository contains everything needed to reproduce our study of that mismatch: dataset curation, re-ranker fine-tuning, end-to-end retrieval + re-ranking evaluation against a live [Redis](https://redis.io/) semantic cache, and the cache-aware analysis that quantifies the **calibration gap**.

> **Authors:** Aditeya Baral, Radoslav Ralev, Iliya Sotirov Zhechev, Srijith Rajamohan, Jen Agarwal &nbsp;·&nbsp; New York University · Redis
>
> 📄 **Paper:** *Closing the Calibration Gap in Semantic Caching* — [arXiv](https://arxiv.org/abs/XXXX.XXXXX) <!-- TODO: replace XXXX.XXXXX with the arXiv ID -->
> 🤗 **Models & datasets:** [`redis` on Hugging Face](https://huggingface.co/redis)

## Table of Contents

- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Models and Datasets](#models-and-datasets)
- [Reproducing the Paper](#reproducing-the-paper)
- [Detailed Usage](#detailed-usage)
  - [1. Dataset Curation](#1-dataset-curation)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Analysis](#4-analysis)

## Overview

Standard practice evaluates semantic-cache retrievers and re-rankers with **PR-AUC**, a threshold-agnostic metric (agnostic to score magnitude) that ignores whether scores are usable at a fixed operating threshold. We show this leads to systematically poor deployment choices — **the models with the highest PR-AUC are often the worst in operation.**

The paper introduces two cache-aware tools and a decomposition:

- **Precision–Cache Hit Ratio (P-CHR) AUC** — precision measured across the full range of cache *utilization* levels, rather than recall.
- **Calibration Retention Rate (CRR)** — how much of a model's offline ranking quality actually survives at deployment.
- A decomposition of the offline-to-deployed quality gap into a **recoverable calibration component** and an **irreducible structural component** fixed by the dataset's positive rate.

Our experiments show the calibration gap is governed by the **training objective** (binary cross-entropy vs. multiple-negatives ranking loss) rather than data scale, and that post-hoc calibration only partially closes it. The central takeaway: **model selection for semantic caching is a calibration problem, not a ranking one.**

This repository lets you reproduce that result end-to-end and apply the same analysis to your own retrievers and re-rankers.

## Key Concepts

These definitions make the repository self-contained; see the paper for full treatment.

| Term | Definition |
|------|------------|
| **Semantic cache** | A store of past query→response pairs. A new query is embedded, the nearest cached entries are retrieved, and a re-ranker scores them; if the top score clears a threshold, the cached response is served (a **cache hit**). |
| **Retriever (bi-encoder)** | Embedding model that fetches candidate matches from the cache by vector similarity. |
| **Re-ranker** | Cross-encoder or ColBERT model that re-scores the retrieved candidates more precisely. |
| **Cache Hit Ratio (CHR)** | Fraction of queries for which the cache fires (top score ≥ threshold), regardless of correctness. |
| **Valid Cache Hit Ratio (VCHR)** | Fraction of queries where the cache fires **and** returns the correct match (= precision × CHR). |
| **PR-AUC** | Area under the precision–recall curve, swept over the decision threshold and reported for both retrievers and re-rankers. It is agnostic to score magnitude, so a high PR-AUC need not translate into good precision at a fixed deployment threshold. |
| **P-CHR AUC** | Area under the precision-vs-CHR curve — precision across operating points defined by *how much* of the cache is used. The paper's recommended selection metric. |
| **Calibration Retention Rate (CRR)** | P-CHR AUC / PR-AUC — how much offline ranking quality is retained at a deployable fixed threshold. |
| **Calibration gap** | The recoverable part of the offline→deployed quality drop, attributable to mis-calibrated scores. Partially closed by temperature / Platt scaling. |
| **Structural gap** | The irreducible part, fixed by the dataset's positive rate. |

## Repository Structure

```
.
├── src/
│   ├── analysis/
│   │   ├── analyze_cls.py              # PR-AUC and P-CHR-AUC metric analysis and plots
│   │   ├── analyze_distribution.py    # Score distribution (KDE) analysis and plots
│   │   ├── analyze_latency.py         # Retrieval and reranking latency analysis and plots
│   │   ├── compute_calibration.py     # Temperature and Platt scaling calibration
│   │   └── util.py                     # Shared utilities (library module)
│   ├── eval/
│   │   ├── eval_reranker.py            # End-to-end retrieval + re-ranking evaluation
│   │   └── retrieve_rerank_evaluator.py  # Evaluator class (library module)
│   ├── reranker/
│   │   ├── cache_evaluator.py          # Cache-aware SentenceEvaluator (library module)
│   │   ├── finetune_colbert.py         # ColBERT re-ranker fine-tuning script
│   │   ├── finetune_crossencoder.py    # Cross-encoder re-ranker fine-tuning script
│   │   └── util.py                     # Dataset loading and InfoNCE utilities (library module)
│   ├── sentencepairs/
│   │   ├── create_sentencepairs_v1.py  # Dataset curation: v1 (1M train pairs)
│   │   ├── create_sentencepairs_v2.py  # Dataset curation: v2 (8M train pairs)
│   │   └── create_sentencepairs_v3.py  # Dataset curation: v3 (40M train pairs)
│   └── shell/
│       ├── run_reranker_evals.sh               # Run all retriever–reranker eval combinations
│       └── run_reranker_evals_for_retriever.sh # Run all rerankers for one retriever
├── results/                            # Evaluation result JSON files (created by eval_reranker.py)
├── plots/                              # Analysis plots (created by the analysis scripts)
├── pyproject.toml
└── uv.lock
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and targets **Python 3.12**.

```bash
# Clone the repository
git clone https://github.com/aditeyabaral/calibration-gap-semantic-caching.git
cd calibration-gap-semantic-caching

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies into a local virtual environment
uv sync

# Activate the environment
source .venv/bin/activate
```

All scripts are run from the **repository root** (each script adds the root to `sys.path`, so imports resolve as `src.<module>`).

### Redis

Evaluation runs against a live semantic cache backed by **Redis Stack**. The easiest way to start one locally is with Docker:

```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```

By default the scripts connect to `localhost:6379` (configurable via `--redis-host` / `--redis-port`).

### Hugging Face

A Hugging Face account and access token are required to download the models/datasets and (for training/curation) to push artifacts to the Hub. Authenticate once:

```bash
huggingface-cli login
```

### Hardware

- **Evaluation & analysis:** a single GPU is recommended; CPU works but is slow. The analysis scripts are CPU-parallel.
- **Training:** a CUDA GPU is required. All training scripts support single- and multi-GPU execution via `accelerate launch` (run `accelerate config` once to set up), and use **FlashAttention-2** with `bfloat16`.

## Models and Datasets

All models and datasets introduced in the paper are published under the [`redis`](https://huggingface.co/redis) organization on the Hugging Face Hub. The evaluation suite also includes a set of third-party baselines, which are downloaded automatically.

### Our artifacts

| Type | Hugging Face ID |
|------|-----------------|
| Retriever (bi-encoder) | [`redis/langcache-embed-v1`](https://huggingface.co/redis/langcache-embed-v1) |
| Retriever (bi-encoder) | [`redis/langcache-embed-v2`](https://huggingface.co/redis/langcache-embed-v2) |
| Retriever (bi-encoder) | [`redis/langcache-embed-v3-small`](https://huggingface.co/redis/langcache-embed-v3-small) |
| Cross-encoder re-ranker | [`redis/langcache-reranker-v1`](https://huggingface.co/redis/langcache-reranker-v1) |
| Cross-encoder re-ranker | [`redis/langcache-reranker-v1-softmnrl-triplet`](https://huggingface.co/redis/langcache-reranker-v1-softmnrl-triplet) |
| Cross-encoder re-ranker (BCE) | [`redis/langcache-reranker-v2-modernbert-bce-eps0.5`](https://huggingface.co/redis/langcache-reranker-v2-modernbert-bce-eps0.5) |
| Cross-encoder re-ranker (MNRL) | [`redis/langcache-reranker-v2-softmnrl-triplet`](https://huggingface.co/redis/langcache-reranker-v2-softmnrl-triplet) |
| Sentence-pair datasets | [`redis/langcache-sentencepairs-v1`](https://huggingface.co/datasets/redis/langcache-sentencepairs-v1) · [`v2`](https://huggingface.co/datasets/redis/langcache-sentencepairs-v2) · [`v3`](https://huggingface.co/datasets/redis/langcache-sentencepairs-v3) |
| LLM paraphrase source | [`redis/llm-paraphrases`](https://huggingface.co/datasets/redis/llm-paraphrases) |

### Baselines evaluated

- **Retrievers:** `Snowflake/snowflake-arctic-embed-m-v1.5`, `Snowflake/snowflake-arctic-embed-m-v2.0`, `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2`, `nomic-ai/nomic-embed-text-v1.5`, `Alibaba-NLP/gte-modernbert-base`
- **Cross-encoder re-rankers:** `cross-encoder/ms-marco-MiniLM-L12-v2`, `Alibaba-NLP/gte-reranker-modernbert-base`
- **ColBERT re-rankers:** `lightonai/ColBERT-Zero`, `lightonai/GTE-ModernColBERT-v1`, `lightonai/Reason-ModernColBERT`, `colbert-ir/colbertv2.0`

### SentencePairs dataset versions

Three cumulative versions of the sentence-pair dataset are provided, each building on the previous:

| Version | Train | Val | Test | Sources |
|---------|------:|----:|-----:|---------|
| [v1](https://huggingface.co/datasets/redis/langcache-sentencepairs-v1) | 1M | 8.4K | 62K | APT, PAWS, QQP, SICK, STS-B, MRPC, PARADE, PIT-2015 |
| [v2](https://huggingface.co/datasets/redis/langcache-sentencepairs-v2) | 8M | 8.4K | 74K | v1 + LLM-generated paraphrases |
| [v3](https://huggingface.co/datasets/redis/langcache-sentencepairs-v3) | 40M | 10.8K | 74K | v2 + OpusParcus, TTIC-31190, TaPaCo, Paraphrase Collections, ChatGPT Paraphrases, ParaNMT-5M, Task275-WSC, ParaBank2 |

All three versions are pre-built and published on the Hub (linked above), so you do **not** need to rebuild them to reproduce results — they are loaded automatically by the training and evaluation scripts. The paper uses **v3**.

## Reproducing the Paper

The full pipeline is **Datasets → Training → Evaluation → Analysis**. Because the datasets and models are already on the Hub, most users can skip straight to **Evaluation**.

**Quickstart (single combination):**

```bash
# 1. Start Redis and authenticate with Hugging Face
docker run -d -p 6379:6379 redis/redis-stack:latest
huggingface-cli login

# 2. Evaluate one retriever + re-ranker pair at k=50 (the paper's setting)
python src/eval/eval_reranker.py \
  --biencoder-model-path redis/langcache-embed-v3-small \
  --reranker-model-path redis/langcache-reranker-v2-modernbert-bce-eps0.5 \
  --reranker-type crossencoder \
  --dataset-version v3 \
  --top-k 50 \
  --flush-cache \
  --output results/example.json

# 3. Fit post-hoc calibration (temperature / Platt) for the re-ranker
python src/analysis/compute_calibration.py \
  --model-path redis/langcache-reranker-v2-modernbert-bce-eps0.5 \
  --dataset-version v3 \
  --output calibration_params.json

# 4. Analyze: PR-AUC and P-CHR-AUC, with calibration applied
python src/analysis/analyze_cls.py \
  --results-dir results/ \
  --plots-dir plots/classification/ \
  --output plots/classification/cls_metrics.json \
  --calibration calibration_params.json \
  --calibration-method temperature
```

**Full reproduction (all combinations):**

```bash
# Evaluate every retriever × re-ranker combination from the paper (k=5 by default;
# edit TOP_K inside the script for k=50). Requires Redis running.
bash src/shell/run_reranker_evals.sh

# Compute calibration for each re-ranker (BCE and MNRL; repeat per model), then run all analyses
python src/analysis/compute_calibration.py --model-path <reranker> --dataset-version v3 --output calibration_params.json
python src/analysis/analyze_cls.py          --results-dir results/ --plots-dir plots/classification/ --output plots/classification/cls_metrics.json --calibration calibration_params.json
python src/analysis/analyze_distribution.py --results-dir results/ --plots-dir plots/distribution/   --output plots/distribution/dist_metrics.json   --calibration calibration_params.json
python src/analysis/analyze_latency.py      --results-dir results/ --plots-dir plots/latency/        --output plots/latency/latency_metrics.json
```

> **Cache reuse.** Populating the cache for a retriever is the expensive step. Pass `--flush-cache` on the **first** evaluation for each new retriever, then omit it for subsequent re-rankers that share the same retriever so the cached embeddings are reused. The shell scripts handle this automatically.

## Detailed Usage

### 1. Dataset Curation

> Optional — only needed to rebuild or extend the datasets from scratch; the pre-built versions are on the Hub.

Each script curates one dataset version, pushes each source split individually, then pushes a merged `all` config to the Hub under your configured account.

```bash
python src/sentencepairs/create_sentencepairs_v1.py
python src/sentencepairs/create_sentencepairs_v2.py
python src/sentencepairs/create_sentencepairs_v3.py
```

Most sources download automatically from the Hub. A few must be placed under a `data/` directory at the repo root beforehand:

**Required for v1, v2 and v3:**

| Source | Expected path | Files |
|--------|--------------|-------|
| PARADE | `data/parade/` | `PARADE_train.txt`, `PARADE_validation.txt`, `PARADE_test.txt` |
| PIT-2015 | `data/pit2015/` | `train.data`, `dev.data`, `test.data` |
| APT | `data/apt/` | `train.tsv`, `test.tsv` |
| SICK | `data/sick/` | `SICK.txt` |

**Additional sources for v3:**

| Source | Expected path | Files |
|--------|--------------|-------|
| TTIC-31190 | `data/ttic31190/` | `train.tsv`, `dev.tsv`, `devtest.tsv` |
| OpusParcus | `data/opusparcus/` | `train_en.70.jsonl`, `validation.jsonl`, `test.jsonl` |
| ParaNMT-5M | `data/paranmt/para-nmt-5m-processed/` | `para-nmt-5m-processed.txt` |
| ParaBank2 | `data/parabank2/` | `parabank2.tsv` |

All other sources (PAWS, MRPC, QQP, STS-B, TaPaCo, Paraphrase Collections, ChatGPT Paraphrases, Task275-WSC, and the LLM paraphrases) are pulled from the Hub automatically.

### 2. Training

> Optional — the trained re-rankers are on the Hub. Training requires a CUDA GPU and supports multi-GPU via `accelerate`.

#### Cross-Encoder Fine-tuning

```bash
accelerate launch src/reranker/finetune_crossencoder.py \
  --pretrained-model-path Alibaba-NLP/gte-reranker-modernbert-base \
  --finetuned-model-path <your-hf-username>/my-reranker \
  --dataset-version v3 \
  --loss-function bce \
  --epsilon 0.5 \
  --batch-size 48 \
  --learning-rate 2e-4 \
  --epochs 5 \
  --output-dir /path/to/checkpoints
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained-model-path` | `Alibaba-NLP/gte-reranker-modernbert-base` | Base cross-encoder to fine-tune |
| `--finetuned-model-path` | `redis/langcache-reranker-v2` | Output model name / Hub ID (also the push target — override with your own namespace) |
| `--dataset-version` | `v3` | SentencePairs version (`v1`, `v2`, `v3`) |
| `--train-dataset-subsets` | `["all"]` | Subset names within the dataset version |
| `--loss-function` | *(required)* | `bce`, `mnrl-sampled`, `mnrl-positive`, or `mse` |
| `--epsilon` | `0` | Label-smoothing coefficient for BCE |
| `--num-negatives` | `1` | Negatives per anchor (MNRL) |
| `--scale` | `20.0` | Scale for MNRL |
| `--batch-size` | `48` | Per-device train/eval batch size |
| `--learning-rate` | `2e-4` | Peak learning rate |
| `--epochs` | `5` | Training epochs |
| `--warmup-ratio` | `0.10` | LR warmup fraction |
| `--weight-decay` | `0.001` | AdamW weight decay |
| `--eval-split` | `val` | Split used for checkpoint selection |
| `--combine-train-and-val` | `False` | Merge train+val into the training set |
| `--eval-steps` / `--save-steps` / `--logging-steps` | `1000` / `10000` / `1000` | Step intervals |
| `--save-total-limit` | `5` | Max checkpoints to keep |
| `--output-dir` | `/opt/dlami/nvme/langcache-reranker-models` | Local checkpoint directory |
| `--wandb-run-name` | `None` | Optional Weights & Biases run name |
| `--device` / `--seed` | `cuda` / `42` | Device and random seed |

The best checkpoint (by validation F1) is pushed to `--finetuned-model-path` at the end of training.

#### ColBERT Fine-tuning

```bash
accelerate launch src/reranker/finetune_colbert.py \
  --pretrained-model-path lightonai/GTE-ModernColBERT-v1 \
  --finetuned-model-path <your-hf-username>/my-colbert \
  --dataset-version v3 \
  --num-negatives 1 \
  --temperature 0.02 \
  --batch-size 48 \
  --learning-rate 2e-4 \
  --epochs 5 \
  --output-dir /path/to/checkpoints
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained-model-path` | `lightonai/GTE-ModernColBERT-v1` | Base ColBERT model to fine-tune |
| `--finetuned-model-path` | `redis/langcache-colbert-v2` | Output model name / Hub ID (push target — override with your own namespace) |
| `--query-length` / `--document-length` | `512` / `512` | Max query / document token lengths |
| `--dataset-version` | `v3` | SentencePairs version (`v1`, `v2`, `v3`) |
| `--train-dataset-subsets` | `["all"]` | Subset names within the dataset version |
| `--num-negatives` | `1` | Negatives per anchor (contrastive / InfoNCE) |
| `--temperature` | `0.02` | InfoNCE temperature |
| `--batch-size` | `48` | Per-device train/eval batch size |
| `--learning-rate` | `2e-4` | Peak learning rate |
| `--epochs` | `5` | Training epochs |
| `--warmup-ratio` | `0.10` | LR warmup fraction |
| `--weight-decay` | `0.001` | AdamW weight decay |
| `--eval-split` | `val` | Split used for checkpoint selection |
| `--combine-train-and-val` | `False` | Merge train+val into the training set |
| `--eval-steps` / `--save-steps` / `--logging-steps` | `1000` / `10000` / `1000` | Step intervals |
| `--save-total-limit` | `5` | Max checkpoints to keep |
| `--output-dir` | `/opt/dlami/nvme/langcache-colbert-models` | Local checkpoint directory |
| `--wandb-run-name` | `None` | Optional Weights & Biases run name |
| `--device` / `--seed` | `cuda` / `42` | Device and random seed |

### 3. Evaluation

`eval_reranker.py` runs the full two-stage pipeline against a live Redis semantic cache for **one** retriever–reranker pair and writes a result JSON to `results/`. The cache is populated with the test set's unique candidates, then every test query is retrieved (`top-k`) and re-ranked. Raw re-ranker scores are stored as-is; activation and calibration are applied later, at analysis time.

```bash
python src/eval/eval_reranker.py \
  --biencoder-model-path redis/langcache-embed-v3-small \
  --reranker-model-path redis/langcache-reranker-v2-modernbert-bce-eps0.5 \
  --reranker-type crossencoder \
  --dataset-version v3 \
  --top-k 50 \
  --flush-cache \
  --output results/example.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--biencoder-model-path` | *(required)* | Retriever (bi-encoder) Hub ID or local path |
| `--reranker-model-path` | *(required)* | Re-ranker Hub ID or local path |
| `--reranker-type` | `crossencoder` | `crossencoder` or `colbert` |
| `--dataset-version` | `v3` | SentencePairs version used as the test set |
| `--top-k` / `-k` | `5` | Candidates retrieved per query (use `50` for the paper) |
| `--redis-host` / `--redis-port` | `localhost` / `6379` | Redis connection |
| `--flush-cache` | `False` | Flush Redis before populating the cache |
| `--output` | auto-generated | Result JSON path |
| `--device` / `--seed` | `cuda` / `42` | Device and random seed |

If `--output` is omitted, the file is named automatically:

```
eval_results_[biencoder_model_path=...]_[reranker_model_path=...]_[reranker_type=...]_[top_k=...].json
```

To sweep all combinations, use the shell helpers:

```bash
bash src/shell/run_reranker_evals.sh                                  # every retriever × re-ranker
bash src/shell/run_reranker_evals_for_retriever.sh <retriever> <redis_port> [top_k]   # one retriever, all re-rankers
```

Each result JSON records, per query: the retrieved candidates and scores, the re-ranked candidates and scores, the ground-truth label, and retrieval/reranking/total latencies — everything the analysis scripts need.

### 4. Analysis

Once `results/` is populated, the analysis scripts compute the paper's metrics and figures. All of them accept `--workers` (default `-1` = all CPUs).

#### Score Calibration

Fit **temperature** and **Platt** scaling parameters for a re-ranker on the train+val split. The output JSON is consumed by the classification and distribution analyses via `--calibration`. The paper applies and compares post-hoc calibration on both **BCE-** and **MNRL-trained** re-rankers; ColBERT-family models need no calibration, since their gap is entirely structural.

```bash
python src/analysis/compute_calibration.py \
  --model-path redis/langcache-reranker-v2-modernbert-bce-eps0.5 \
  --dataset-version v3 \
  --output calibration_params.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | *(required)* | Hub ID or local path of the re-ranker |
| `--dataset-version` | *(required)* | Version the model was trained on (`v1`/`v2`/`v3`) |
| `--output` | *(required)* | Calibration JSON to write/merge into |
| `--batch-size` | `64` | Inference batch size |
| `--device` / `--seed` | `cuda` / `42` | Device and random seed |

#### PR-AUC and P-CHR-AUC (classification)

Computes PR-AUC, Precision–CHR AUC and Precision–VCHR AUC for every combination across `k = 1..K`, plus precision/recall at the F1-optimal threshold, and renders the paper's curves.

```bash
python src/analysis/analyze_cls.py \
  --results-dir results/ \
  --plots-dir plots/classification/ \
  --output plots/classification/cls_metrics.json \
  --calibration calibration_params.json \
  --calibration-method temperature
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | *(required)* | Directory of evaluation result JSONs |
| `--plots-dir` | *(required)* | Output directory for plots |
| `--output` | *(required)* | JSON summary of computed metrics |
| `--thresholds` | `0.0–1.0` step `0.01` | Threshold values to sweep |
| `--calibration` | `None` | `calibration_params.json` from `compute_calibration.py` |
| `--calibration-method` | `temperature` | `temperature` or `platt` |
| `--workers` | `-1` | Parallel workers (`-1` = all CPUs) |

**Outputs:** `cls_metrics.json`, summary `*_vs_k.png` plots, and per-`k` `combined_pr_curves.png` / `combined_precision_chr_curves.png` / `combined_precision_vchr_curves.png`. Retriever-only baselines are drawn as dotted lines, reranker-augmented systems as solid lines, sorted by AUC.

#### Score Distribution Analysis

KDE plots of positive vs. negative ground-truth scores per retriever and per retriever–reranker pair, with ROC-AUC, KS statistic and KDE overlap.

```bash
python src/analysis/analyze_distribution.py \
  --results-dir results/ \
  --plots-dir plots/distribution/ \
  --output plots/distribution/dist_metrics.json \
  --calibration calibration_params.json
```

Arguments mirror `analyze_cls.py` (`--results-dir`, `--plots-dir`, `--output`, `--calibration`, `--calibration-method`, `--workers`). Outputs one KDE plot per unique retriever and per retriever–reranker pair, plus a metrics JSON.

#### Latency Analysis

Per-query retrieval / reranking / total latency (mean, std, p95) and reranking overhead, across all combinations.

```bash
python src/analysis/analyze_latency.py \
  --results-dir results/ \
  --plots-dir plots/latency/ \
  --output plots/latency/latency_metrics.json
```

Arguments: `--results-dir`, `--plots-dir`, `--output` (all required), `--workers`. Outputs `latency_breakdown.png`, `latency_per_reranker.png`, `latency_per_retriever.png` and a metrics JSON.
