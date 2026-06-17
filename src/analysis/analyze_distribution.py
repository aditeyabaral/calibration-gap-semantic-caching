import argparse
import os
import json
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, gaussian_kde
from multiprocessing import Pool
from rich.console import Console
from rich.table import Table

from src.analysis.util import (
    extract_models_from_filename,
    get_calibration_params,
    load_calibration,
    nan_to_none,
    normalize_reranker_scores,
)


def extract_gt_scores_labels(
    results,
    reranker_type=None,
    calib_params: dict = None,
    calibration_method: str = "temperature",
):
    """Extract GT retriever scores and normalized reranker scores with labels."""
    retriever_gt_scores = []
    reranker_gt_scores = []
    labels = []

    for example in results:
        gt = example["ground_truth"]
        label = example.get("label", 0)

        try:
            gt_idx = example["retrieved_candidates"].index(gt)
            retriever_score = example["retrieved_scores"][gt_idx]
        except ValueError:
            retriever_score = 0.0

        reranker_score_norm = 0.0

        ranked_scores = example.get("ranked_scores", [])
        ranked_candidates = example.get("ranked_candidates", [])

        try:
            gt_idx_rer = ranked_candidates.index(gt)
            normalized = normalize_reranker_scores(
                ranked_scores,
                reranker_type=reranker_type,
                calib_params=calib_params,
                calibration_method=calibration_method,
            )
            reranker_score_norm = float(normalized[gt_idx_rer])
        except Exception:
            reranker_score_norm = 0.0

        retriever_gt_scores.append(retriever_score)
        reranker_gt_scores.append(reranker_score_norm)
        labels.append(label)

    return retriever_gt_scores, reranker_gt_scores, labels


def ks_score(scores, labels):
    pos_scores = [s for s, lbl in zip(scores, labels) if lbl == 1]
    neg_scores = [s for s, lbl in zip(scores, labels) if lbl == 0]
    if not pos_scores or not neg_scores:
        return float("nan")
    stat, _ = ks_2samp(pos_scores, neg_scores)
    return stat


def area_of_overlap_kde(scores, labels, n_points=1000):
    pos_scores = np.array([s for s, lbl in zip(scores, labels) if lbl == 1])
    neg_scores = np.array([s for s, lbl in zip(scores, labels) if lbl == 0])
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    try:
        kde_pos = gaussian_kde(pos_scores)
        kde_neg = gaussian_kde(neg_scores)
    except np.linalg.LinAlgError:
        return float("nan")

    x_grid = np.linspace(0.0, 1.0, n_points)
    f_pos = kde_pos(x_grid)
    f_neg = kde_neg(x_grid)
    return float(np.trapezoid(np.minimum(f_pos, f_neg), x_grid))


def plot_distributions(scores, labels, title, output_path):
    pos_scores = np.array([s for s, lbl in zip(scores, labels) if lbl == 1])
    neg_scores = np.array([s for s, lbl in zip(scores, labels) if lbl == 0])

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        print(f"Cannot plot {title}: missing positive or negative samples")
        return

    x_grid = np.linspace(0.0, 1.0, 1000)
    try:
        kde_pos = gaussian_kde(pos_scores)
        kde_neg = gaussian_kde(neg_scores)
    except np.linalg.LinAlgError:
        print(
            f"Cannot plot {title}: degenerate score distribution (all scores identical)"
        )
        return

    f_pos = kde_pos(x_grid)
    f_neg = kde_neg(x_grid)
    overlap_density = np.minimum(f_pos, f_neg)
    overlap = np.trapezoid(overlap_density, x_grid)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        x_grid,
        f_pos,
        color="green",
        linewidth=2,
        label=f"Positive (n={len(pos_scores)})",
        alpha=0.8,
    )
    ax.plot(
        x_grid,
        f_neg,
        color="red",
        linewidth=2,
        label=f"Negative (n={len(neg_scores)})",
        alpha=0.8,
    )
    ax.fill_between(
        x_grid, overlap_density, alpha=0.3, color="purple", label="Overlap Region"
    )

    stats_text = (
        f"Positive: μ={pos_scores.mean():.3f}, σ={pos_scores.std():.3f}\n"
        f"Negative: μ={neg_scores.mean():.3f}, σ={neg_scores.std():.3f}\n"
        f"Overlap: {overlap:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    plt.close()


def process_file(args_tuple):
    filename, results_dir, calibration_data, calibration_method = args_tuple
    path = os.path.join(results_dir, filename)
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {path}: {e}")
        return None
    results = data.get("results", [])
    if not results:
        return None

    retriever_name, reranker_name, reranker_type = extract_models_from_filename(
        filename
    )
    calib_params = get_calibration_params(reranker_name, calibration_data)
    retriever_scores, reranker_scores, labels = extract_gt_scores_labels(
        results,
        reranker_type=reranker_type,
        calib_params=calib_params,
        calibration_method=calibration_method,
    )

    try:
        retr_auc = roc_auc_score(labels, retriever_scores)
    except ValueError:
        retr_auc = float("nan")
    try:
        rerank_auc = roc_auc_score(labels, reranker_scores)
    except ValueError:
        rerank_auc = float("nan")

    retr_ks = ks_score(retriever_scores, labels)
    rer_ks = ks_score(reranker_scores, labels)
    retr_overlap_kde = area_of_overlap_kde(retriever_scores, labels)
    rer_overlap_kde = area_of_overlap_kde(reranker_scores, labels)

    return {
        "filename": filename,
        "retriever_name": retriever_name,
        "reranker_name": reranker_name,
        "retriever_scores": retriever_scores,
        "reranker_scores": reranker_scores,
        "labels": labels,
        "retr_auc": retr_auc,
        "rerank_auc": rerank_auc,
        "retr_ks": retr_ks,
        "rer_ks": rer_ks,
        "retr_overlap_kde": retr_overlap_kde,
        "rer_overlap_kde": rer_overlap_kde,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot score distributions for results."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--plots-dir", type=str, required=True, help="Directory to save output plots"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration_params.json produced by compute_calibration.py. Applied to any model whose key is found in the file.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["temperature", "platt"],
        default="temperature",
        help="Calibration method to apply when --calibration is provided (default: temperature).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save a JSON summary of computed metrics.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="Number of parallel worker processes (-1 uses all available CPUs, default: -1)",
    )
    args = parser.parse_args()

    calibration_data = load_calibration(args.calibration)

    Path(args.plots_dir).mkdir(parents=True, exist_ok=True)

    files = [
        f
        for f in os.listdir(args.results_dir)
        if f.startswith("eval_results_") and f.endswith(".json")
    ]
    num_workers = (
        int(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) * 0.75)
        if args.workers == -1
        else args.workers
    )

    print("Configuration:")
    print(f"  Input directory:  {args.results_dir}")
    print(f"  Output directory: {args.plots_dir}")
    print(f"  Output JSON:      {args.output}")
    print(f"  Calibration:      {args.calibration or 'none'}")
    if args.calibration:
        print(f"  Calib method:     {args.calibration_method}")
    print(f"  Workers:          {num_workers}")

    worker_args = [
        (
            f,
            args.results_dir,
            calibration_data,
            args.calibration_method,
        )
        for f in files
    ]

    print(f"\nProcessing {len(files)} files with {num_workers} worker(s)...")
    with Pool(processes=num_workers) as pool:
        file_results = [
            r
            for r in tqdm(
                pool.imap(process_file, worker_args),
                total=len(worker_args),
                desc="Files completed",
                dynamic_ncols=True,
            )
            if r is not None
        ]

    # Plotting and deduplication are sequential (matplotlib is not parallel-safe)
    saved_retrievers = set()
    saved_pairs = set()
    retriever_metrics = []  # one entry per unique retriever
    reranker_metrics = []  # one entry per retriever+reranker pair

    for result in file_results:
        filename = result["filename"]
        retriever_name = result["retriever_name"]
        reranker_name = result["reranker_name"]
        retriever_scores = result["retriever_scores"]
        reranker_scores = result["reranker_scores"]
        labels = result["labels"]

        base_name = filename.replace(".json", "")

        # Retriever plot: one KDE per unique retriever
        if retriever_name is None:
            plot_distributions(
                retriever_scores,
                labels,
                title=f"{base_name} Retriever Score Distribution",
                output_path=os.path.join(
                    args.plots_dir, f"{base_name}_retriever_kde.png"
                ),
            )
            retriever_metrics.append(
                {
                    "label": base_name,
                    "retr_auc": result["retr_auc"],
                    "retr_ks": result["retr_ks"],
                    "retr_overlap_kde": result["retr_overlap_kde"],
                }
            )
        elif retriever_name not in saved_retrievers:
            plot_distributions(
                retriever_scores,
                labels,
                title=f"{retriever_name.replace('--', '/')} Retriever Score Distribution",
                output_path=os.path.join(
                    args.plots_dir, f"{retriever_name}_retriever_kde.png"
                ),
            )
            retriever_metrics.append(
                {
                    "label": retriever_name,
                    "retr_auc": result["retr_auc"],
                    "retr_ks": result["retr_ks"],
                    "retr_overlap_kde": result["retr_overlap_kde"],
                }
            )
            saved_retrievers.add(retriever_name)

        # Reranker plot: one KDE per unique retriever+reranker pair
        if retriever_name is None or reranker_name is None:
            plot_distributions(
                reranker_scores,
                labels,
                title=f"{base_name} Reranker Score Distribution",
                output_path=os.path.join(
                    args.plots_dir, f"{base_name}_reranker_kde.png"
                ),
            )
            reranker_metrics.append(
                {
                    "label": base_name,
                    "rerank_auc": result["rerank_auc"],
                    "rer_ks": result["rer_ks"],
                    "rer_overlap_kde": result["rer_overlap_kde"],
                }
            )
        else:
            pair = (retriever_name, reranker_name)
            if pair not in saved_pairs:
                plot_distributions(
                    reranker_scores,
                    labels,
                    title=f"{retriever_name.replace('--', '/')} + {reranker_name.replace('--', '/')} Reranker Score Distribution",
                    output_path=os.path.join(
                        args.plots_dir,
                        f"{retriever_name}__{reranker_name}_reranker_kde.png",
                    ),
                )
                reranker_metrics.append(
                    {
                        "label": f"{retriever_name}+{reranker_name}",
                        "rerank_auc": result["rerank_auc"],
                        "rer_ks": result["rer_ks"],
                        "rer_overlap_kde": result["rer_overlap_kde"],
                    }
                )
                saved_pairs.add(pair)

    console = Console(width=300)

    ret_table = Table(
        title="Retriever Score Distribution Metrics",
        show_header=True,
        header_style="bold cyan",
    )
    ret_table.add_column("Retriever", style="dim", no_wrap=True)
    ret_table.add_column("ROC AUC", justify="right", min_width=8)
    ret_table.add_column("KS Stat", justify="right", min_width=7)
    ret_table.add_column("KDE Overlap", justify="right", min_width=11)
    for entry in retriever_metrics:
        ret_table.add_row(
            entry["label"],
            f"{entry['retr_auc']:.4f}",
            f"{entry['retr_ks']:.4f}",
            f"{entry['retr_overlap_kde']:.4f}",
        )
    console.print(ret_table)

    rer_table = Table(
        title="Reranker Score Distribution Metrics",
        show_header=True,
        header_style="bold cyan",
    )
    rer_table.add_column("Setup", style="dim", no_wrap=True)
    rer_table.add_column("ROC AUC", justify="right", min_width=8)
    rer_table.add_column("KS Stat", justify="right", min_width=7)
    rer_table.add_column("KDE Overlap", justify="right", min_width=11)
    for entry in reranker_metrics:
        rer_table.add_row(
            entry["label"],
            f"{entry['rerank_auc']:.4f}",
            f"{entry['rer_ks']:.4f}",
            f"{entry['rer_overlap_kde']:.4f}",
        )
    console.print(rer_table)

    output_data = {
        "calibration": args.calibration,
        "calibration_method": args.calibration_method if args.calibration else None,
        "retriever_results": [
            {
                "label": entry["label"],
                "retr_auc": nan_to_none(entry["retr_auc"]),
                "retr_ks": nan_to_none(entry["retr_ks"]),
                "retr_overlap_kde": nan_to_none(entry["retr_overlap_kde"]),
            }
            for entry in retriever_metrics
        ],
        "reranker_results": [
            {
                "label": entry["label"],
                "rerank_auc": nan_to_none(entry["rerank_auc"]),
                "rer_ks": nan_to_none(entry["rer_ks"]),
                "rer_overlap_kde": nan_to_none(entry["rer_overlap_kde"]),
            }
            for entry in reranker_metrics
        ],
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved metrics to {args.output}")
