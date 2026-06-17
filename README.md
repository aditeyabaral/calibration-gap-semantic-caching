# Semantic Cache Re-ranking Evaluation

This repository contains the code accompanying our paper. It covers: sentence-pair dataset curation, cross-encoder re-ranker fine-tuning, end-to-end retrieval and re-ranking evaluation (cross-encoder and ColBERT re-rankers) against a live semantic cache, and analysis tools for computing and visualising PR-AUC, Precision–Cache Hit Ratio (P-CHR) AUC, score distributions, and latency metrics.

To reproduce the results in the paper, follow the steps in order: **Setup → Datasets → Training → Evaluation → Analysis**.

---

## Repository Structure

```
.
├── src/
│   ├── analysis/
│   │   ├── analyze_cls.py              # PR-AUC and P-CHR-AUC metric analysis and plots
│   │   ├── analyze_distribution.py    # Score distribution (KDE) analysis and plots
│   │   ├── analyze_latency.py         # Retrieval and reranking latency analysis and plots
│   │   ├── compute_calibration.py     # Temperature and Platt scaling calibration for BCE models
│   │   └── util.py                     # Shared utilities (library module)
│   ├── eval/
│   │   ├── eval_reranker.py            # End-to-end retrieval + re-ranking evaluation
│   │   └── retrieve_rerank_evaluator.py  # Evaluator class (library module)
│   ├── reranker/
│   │   ├── cache_evaluator.py          # Cache-aware SentenceEvaluator (library module)
│   │   ├── finetune_crossencoder.py    # Cross-encoder fine-tuning script
│   │   └── util.py                     # Dataset loading and InfoNCE utilities (library module)
│   ├── sentencepairs/
│   │   ├── create_sentencepairs_v1.py  # Dataset curation: v1 (~1M train pairs)
│   │   ├── create_sentencepairs_v2.py  # Dataset curation: v2 (~8M train pairs)
│   │   └── create_sentencepairs_v3.py  # Dataset curation: v3 (~40M train pairs)
│   └── shell/
│       ├── run_reranker_evals.sh               # Run all retriever–reranker eval combinations
│       └── run_reranker_evals_for_retriever.sh # Run all rerankers for one retriever
├── results/                            # Evaluation result JSON files (populated by eval_reranker.py)
├── plots/
│   ├── classification/                 # PR / P-CHR curve plots (populated by analyze_cls.py)
│   ├── distribution/                   # KDE score distribution plots (populated by analyze_distribution.py)
│   └── latency/                        # Latency plots (populated by analyze_latency.py)
├── pyproject.toml
└── uv.lock
```

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/aditeyabaral/calibration-gap-semantic-caching.git
cd calibration-gap-semantic-caching

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Redis

A running Redis Stack instance is required for evaluation. The easiest way to start one is with Docker:

```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```

### HuggingFace

A HuggingFace account and access token are required to push datasets and models to the Hub (dataset curation and training) and to load models during evaluation. Authenticate once with:

```bash
huggingface-cli login
```

---

## Datasets

### SentencePairs Dataset Versions

Three versions of the sentence-pair dataset are provided, each building on the previous:

| Version | Train | Val | Test | Sources |
|---------|------:|----:|-----:|---------|
| v1 | ~1M | ~8.4K | ~62K | APT, PAWS, QQP, SICK, STS-B, MRPC, PARADE, PIT-2015 |
| v2 | ~8M | ~8.4K | ~74K | v1 + LLM-generated paraphrases |
| v3 | ~40M | ~10.8K | ~74K | v2 + OpusParcus, TTIC-31190, TaPaCo, Paraphrase Collections, ChatGPT Paraphrases, ParaNMT-5M, Task275-WSC, ParaBank2 |

Each script curates the dataset, pushes each source split individually, and then pushes a merged `all` config to the HuggingFace Hub under your configured account. Running the curation scripts is only necessary to reproduce or extend the dataset from scratch; the pre-built v3 dataset used in the paper is available at [`redis/langcache-sentencepairs-v3`](https://huggingface.co/datasets/redis/langcache-sentencepairs-v3).

### Local Data Requirements

Some sources cannot be downloaded automatically and must be placed under a `data/` directory at the repo root before running the curation scripts.

**Required for v1, v2, and v3:**

| Source | Expected path | Files |
|--------|--------------|-------|
| PARADE | `data/parade/` | `PARADE_train.txt`, `PARADE_validation.txt`, `PARADE_test.txt` |
| PIT-2015 | `data/pit2015/` | `train.data`, `dev.data`, `test.data` |
| APT | `data/apt/` | `train.tsv`, `test.tsv` |
| SICK | `data/sick/` | `SICK.txt` |

**Additional requirements for v3:**

| Source | Expected path | Files |
|--------|--------------|-------|
| TTIC-31190 | `data/ttic31190/` | `train.tsv`, `dev.tsv`, `devtest.tsv` |
| OpusParcus | `data/opusparcus/` | `train_en.70.jsonl`, `validation.jsonl`, `test.jsonl` |
| ParaNMT-5M | `data/paranmt/para-nmt-5m-processed/` | `para-nmt-5m-processed.txt` |
| ParaBank2 | `data/parabank2/` | `parabank2.tsv` |

All other sources (PAWS, MRPC, QQP, STS-B, TaPaCo, Paraphrase Collections, ChatGPT Paraphrases, Task275-WSC, and the LLM paraphrase dataset) are downloaded automatically from HuggingFace Datasets.

### Running Dataset Curation

```bash
python src/sentencepairs/create_sentencepairs_v1.py
python src/sentencepairs/create_sentencepairs_v2.py
python src/sentencepairs/create_sentencepairs_v3.py
```

---

## Training

### Cross-Encoder Fine-tuning

Fine-tune a cross-encoder re-ranker on the SentencePairs dataset:

```bash
python src/reranker/finetune_crossencoder.py \
  --pretrained-model-path Alibaba-NLP/gte-reranker-modernbert-base \
  --finetuned-model-path <output-model-name> \
  --dataset-version v3 \
  --loss-function mnrl-sampled \
  --batch-size 48 \
  --learning-rate 2e-4 \
  --epochs 5 \
  --output-dir /path/to/checkpoints
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--pretrained-model-path` | `Alibaba-NLP/gte-reranker-modernbert-base` | Base cross-encoder model to fine-tune |
| `--finetuned-model-path` | `redis/langcache-reranker-v2` | Output model name / HuggingFace Hub ID |
| `--dataset-version` | `v3` | SentencePairs version to train on (`v1`, `v2`, `v3`) |
| `--train-dataset-subsets` | `["all"]` | Subset names within the dataset version |
| `--loss-function` | — | `bce`, `mnrl-sampled`, `mnrl-positive`, or `mse` (**required**) |
| `--epsilon` | `0` | Label smoothing coefficient for BCE loss |
| `--num-negatives` | `1` | Negatives per anchor for MNRL |
| `--batch-size` | `48` | Per-device train and eval batch size |
| `--learning-rate` | `2e-4` | Peak learning rate |
| `--epochs` | `5` | Number of training epochs |
| `--warmup-ratio` | `0.10` | Fraction of steps used for learning rate warmup |
| `--weight-decay` | `0.001` | AdamW weight decay |
| `--eval-split` | `val` | Split used for checkpoint selection during training |
| `--combine-train-and-val` | `False` | Merge train and val splits into the training set |
| `--eval-steps` | `1000` | Evaluate every N steps |
| `--save-steps` | `10000` | Save checkpoint every N steps |
| `--save-total-limit` | `5` | Maximum number of checkpoints to keep |
| `--output-dir` | — | Directory for checkpoints and logs |
| `--wandb-run-name` | — | Optional Weights & Biases run name |
| `--device` | `cuda` | Device to train on (`cuda`, `mps`, `cpu`) |
| `--seed` | `42` | Random seed |

The best checkpoint is selected by validation F1 score and pushed to the HuggingFace Hub under `--finetuned-model-path` at the end of training.

---

## Evaluation

Run the full two-stage retrieval and re-ranking pipeline against a live Redis semantic cache. Each invocation evaluates one retriever–reranker pair and saves a result JSON to `results/`. Repeat for each combination to be analysed.

```bash
python src/eval/eval_reranker.py \
  --biencoder-model-path redis/langcache-embed-v3-small \
  --reranker-model-path redis/langcache-reranker-v2-modernbert-bce-eps0.5 \
  --dataset-version v3 \
  --top-k 50 \
  --activation-fn sigmoid \
  --flush-cache \
  --output results/my_eval_result.json
```

Use `--flush-cache` on the first run for each new retriever, and omit it when reusing the same populated cache across different re-rankers with the same retriever.

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--biencoder-model-path` | — | Retriever (bi-encoder) model path or HF Hub ID (**required**) |
| `--reranker-model-path` | — | Re-ranker model path or HF Hub ID (**required**) |
| `--reranker-type` | `crossencoder` | `crossencoder` or `colbert` |
| `--dataset-version` | `v3` | SentencePairs version to use as test set |
| `--sampling-ratio` | `1.0` | Fraction of unique candidates to populate in the cache |
| `--top-k` | `5` | Number of candidates retrieved per query |
| `--activation-fn` | `sigmoid` | Score activation: `sigmoid` for BCE-trained models, `identity` for MNRL-trained models |
| `--redis-host` | `localhost` | Redis host |
| `--redis-port` | `6379` | Redis port |
| `--flush-cache` | `False` | Flush Redis before populating the cache |
| `--output` | auto-generated | Path to save the result JSON |
| `--device` | `cuda` | Device |
| `--seed` | `42` | Random seed |

If `--output` is not specified, the result file is named automatically:

```
eval_results_[sampling_ratio=...]_[biencoder_model_path=...]_[reranker_model_path=...]_[activation_fn=...]_[top_k=...].json
```

To run all retriever–reranker combinations in one go, use the shell script:

```bash
bash src/shell/run_reranker_evals.sh
```

---

## Analysis

Once all evaluation result JSONs are in `results/`, run the analysis scripts.

### Score Calibration

Compute temperature or Platt scaling calibration parameters for a BCE-trained reranker. The output JSON can then be passed to the classification and distribution analysis scripts via `--calibration`.

```bash
python src/analysis/compute_calibration.py \
  --model-path redis/langcache-reranker-v2-modernbert-bce-eps0.5 \
  --dataset-version v3 \
  --output calibration_params.json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | — | HF Hub ID or local path of the BCE reranker (**required**) |
| `--dataset-version` | — | Dataset version the model was trained on: `v1`, `v2`, or `v3` (**required**) |
| `--output` | — | Path to write (or merge into) the calibration JSON (**required**) |
| `--batch-size` | `64` | Inference batch size |
| `--device` | `cuda` | Device |
| `--seed` | `42` | Random seed |

### PR-AUC and P-CHR-AUC

Compute Precision-Recall AUC and Precision–Cache Hit Ratio AUC across all retriever–reranker combinations and generate plots:

```bash
python src/analysis/analyze_cls.py \
  --results-dir results/ \
  --plots-dir plots/classification/ \
  --output plots/classification/cls_metrics.json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | — | Directory containing evaluation JSON files (**required**) |
| `--plots-dir` | — | Directory to save output plots (**required**) |
| `--output` | — | Path to save a JSON summary of computed metrics (**required**) |
| `--thresholds` | `0.0–1.0, step 0.01` | Custom threshold values to sweep |
| `--calibration` | — | Path to `calibration_params.json` from `compute_calibration.py` |
| `--calibration-method` | `temperature` | `temperature` or `platt` (used when `--calibration` is set) |
| `--workers` | `-1` | Parallel worker processes (`-1` = all CPUs) |

**Outputs:**
- `cls_metrics.json` — JSON summary of PR-AUC and P-CHR-AUC for every combination
- `combined_pr_curves.png` — Precision-Recall curves for all combinations
- `combined_precision_chr_curves.png` — Precision vs Cache Hit Ratio curves for all combinations

Retriever-only baselines are shown as dotted lines; reranker-augmented systems as solid lines. Curves are sorted by AUC descending in the legend.

### Score Distribution Analysis

Plot KDE score distributions for ground-truth candidates under each retriever and reranker:

```bash
python src/analysis/analyze_distribution.py \
  --results-dir results/ \
  --plots-dir plots/distribution/ \
  --output plots/distribution/dist_metrics.json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | — | Directory containing evaluation JSON files (**required**) |
| `--plots-dir` | — | Directory to save output plots (**required**) |
| `--output` | — | Path to save a JSON summary of computed metrics (**required**) |
| `--calibration` | — | Path to `calibration_params.json` from `compute_calibration.py` |
| `--calibration-method` | `temperature` | `temperature` or `platt` (used when `--calibration` is set) |
| `--workers` | `-1` | Parallel worker processes (`-1` = all CPUs) |

**Outputs (one file per unique model or pair):**
- `<retriever>_retriever_kde.png` — Positive vs negative score distributions for each retriever
- `<retriever>__<reranker>_reranker_kde.png` — Positive vs negative score distributions for each retriever–reranker pair

Each plot shows the KDE for positive and negative ground-truth scores, the overlap region, and summary statistics (μ, σ, overlap area). ROC-AUC and KS statistics are also printed to stdout.

### Latency Analysis

Analyse per-query retrieval and reranking latency across all model combinations:

```bash
python src/analysis/analyze_latency.py \
  --results-dir results/ \
  --plots-dir plots/latency/ \
  --output plots/latency/latency_metrics.json
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--results-dir` | — | Directory containing evaluation JSON files (**required**) |
| `--plots-dir` | — | Directory to save output plots (**required**) |
| `--output` | — | Path to save a JSON latency summary (**required**) |
| `--workers` | `-1` | Parallel worker processes (`-1` = all CPUs) |
