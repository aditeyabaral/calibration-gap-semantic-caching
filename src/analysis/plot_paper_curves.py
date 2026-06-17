"""Generate combined PR and P-CHR paper figures for a single fixed retriever at one k."""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics import precision_recall_curve
from tqdm.auto import tqdm

from src.analysis.analyze_cls import (
    load_results,
    slice_datapoint_for_k,
    compute_metrics_across_thresholds,
    compute_auc_metrics,
)
from src.analysis.util import load_calibration, get_calibration_params


def _to_display(name: str) -> str:
    """Convert sanitized org--model name to HF-style org/model."""
    return name.replace("--", "/")


def _process_one(args):
    result, thresholds, calibration_data, calibration_method, k = args
    try:
        with open(result["file_path"]) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {result['file_path']}: {e}")
        return None

    params = result.get("params") or {}
    biencoder_raw = params.get("biencoder_model_path", "unknown")
    reranker_raw = params.get("reranker_model_path", "unknown")
    reranker_type = params.get("reranker_type")

    calib_params = get_calibration_params(reranker_raw, calibration_data)
    data_points = data.get("results", [])
    sliced = [slice_datapoint_for_k(dp, k) for dp in data_points]

    ret_metrics, ret_y_true, ret_y_scores = compute_metrics_across_thresholds(
        sliced, thresholds, setup="retriever"
    )
    ret_auc = compute_auc_metrics(ret_metrics, ret_y_true, ret_y_scores)

    rer_metrics, rer_y_true, rer_y_scores = compute_metrics_across_thresholds(
        sliced, thresholds, setup="retriever+reranker",
        normalize_scores=True, reranker_type=reranker_type,
        calib_params=calib_params, calibration_method=calibration_method,
    )
    rer_auc = compute_auc_metrics(rer_metrics, rer_y_true, rer_y_scores)

    precs_ret, recs_ret, _ = precision_recall_curve(ret_y_true, ret_y_scores)
    precs_rer, recs_rer, _ = precision_recall_curve(rer_y_true, rer_y_scores)

    sort_ret = np.argsort([m["cache_hit_ratio"] for m in ret_metrics])
    sort_rer = np.argsort([m["cache_hit_ratio"] for m in rer_metrics])

    return {
        "biencoder": _to_display(biencoder_raw),
        "reranker": _to_display(reranker_raw),
        "ret_pr_auc": ret_auc["pr_auc"],
        "rer_pr_auc": rer_auc["pr_auc"],
        "ret_chr_auc": ret_auc["precision_chr_auc"],
        "rer_chr_auc": rer_auc["precision_chr_auc"],
        "precs_ret": precs_ret,
        "recs_ret": recs_ret,
        "precs_rer": precs_rer,
        "recs_rer": recs_rer,
        "chrs_ret": np.array([m["cache_hit_ratio"] for m in ret_metrics])[sort_ret],
        "precs_chr_ret": np.array([m["precision"] for m in ret_metrics])[sort_ret],
        "chrs_rer": np.array([m["cache_hit_ratio"] for m in rer_metrics])[sort_rer],
        "precs_chr_rer": np.array([m["precision"] for m in rer_metrics])[sort_rer],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined PR and P-CHR curve plots for a single retriever."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--plots-dir", required=True)
    parser.add_argument(
        "--retriever",
        default="redis--langcache-embed-v3-small",
        help="Sanitized retriever model path to filter on (default: redis--langcache-embed-v3-small)",
    )
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--calibration", default=None)
    parser.add_argument(
        "--calibration-method", choices=["temperature", "platt"], default="temperature"
    )
    parser.add_argument("--workers", type=int, default=-1)
    args = parser.parse_args()

    calibration_data = load_calibration(args.calibration)
    thresholds = np.arange(0.0, 1.01, 0.01).tolist()

    all_results = load_results(args.results_dir)
    filtered = [
        r for r in all_results
        if (r.get("params") or {}).get("biencoder_model_path") == args.retriever
    ]
    print(f"Found {len(filtered)} files for retriever '{args.retriever}'")
    if not filtered:
        print("No matching files — check --retriever value.")
        return

    num_workers = (
        int(os.cpu_count() * 0.75) if args.workers == -1 else args.workers
    )

    worker_args = [
        (r, thresholds, calibration_data, args.calibration_method, args.k)
        for r in filtered
    ]
    with Pool(processes=num_workers) as pool:
        processed = [
            r for r in tqdm(
                pool.imap(_process_one, worker_args),
                total=len(worker_args),
                desc="Processing",
            )
            if r is not None
        ]

    os.makedirs(args.plots_dir, exist_ok=True)
    num_colors = max(len(processed) * 2, 16)
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))

    def _build_entries(processed, x_key, y_key, x_key_ret, y_key_ret, auc_ret_key, auc_rer_key):
        entries = []
        seen_retriever = False
        for idx, r in enumerate(processed):
            if not seen_retriever:
                entries.append({
                    "auc": r[auc_ret_key],
                    "xs": r[x_key_ret], "ys": r[y_key_ret],
                    "label": f"{r['biencoder']} (AUC={r[auc_ret_key]:.3f})",
                    "lw": 1.5, "ls": ":", "alpha": 0.7, "color": colors[idx * 2],
                })
                seen_retriever = True
            entries.append({
                "auc": r[auc_rer_key],
                "xs": r[x_key], "ys": r[y_key],
                "label": f"{r['reranker']} (AUC={r[auc_rer_key]:.3f})",
                "lw": 2, "ls": "-", "alpha": 1.0, "color": colors[idx * 2 + 1],
            })
        entries.sort(key=lambda x: x["auc"], reverse=True)
        return entries

    # --- PR curves ---
    pr_entries = _build_entries(
        processed,
        x_key="recs_rer", y_key="precs_rer",
        x_key_ret="recs_ret", y_key_ret="precs_ret",
        auc_ret_key="ret_pr_auc", auc_rer_key="rer_pr_auc",
    )
    fig, ax = plt.subplots(figsize=(14, 10))
    for e in pr_entries:
        ax.plot(e["xs"], e["ys"], label=e["label"],
                linewidth=e["lw"], linestyle=e["ls"], alpha=e["alpha"], color=e["color"])
    ax.set_xlabel("Recall", fontsize=19)
    ax.set_ylabel("Precision", fontsize=19)
    ax.set_title(f"Precision vs Recall (k={args.k})", fontsize=20, fontweight="bold")
    ax.legend(fontsize=15, loc="best")
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    out = os.path.join(args.plots_dir, f"combined_pr_curves_{args.retriever}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()

    # --- P-CHR curves ---
    chr_entries = _build_entries(
        processed,
        x_key="chrs_rer", y_key="precs_chr_rer",
        x_key_ret="chrs_ret", y_key_ret="precs_chr_ret",
        auc_ret_key="ret_chr_auc", auc_rer_key="rer_chr_auc",
    )
    fig, ax = plt.subplots(figsize=(14, 10))
    for e in chr_entries:
        ax.plot(e["xs"], e["ys"], label=e["label"],
                linewidth=e["lw"], linestyle=e["ls"], alpha=e["alpha"], color=e["color"])
    ax.set_xlabel("Cache Hit Ratio", fontsize=19)
    ax.set_ylabel("Precision", fontsize=19)
    ax.set_title(
        f"Precision vs Cache Hit Ratio (k={args.k})", fontsize=20, fontweight="bold"
    )
    ax.legend(fontsize=15, loc="upper right")
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    out = os.path.join(args.plots_dir, f"combined_precision_chr_curves_{args.retriever}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
