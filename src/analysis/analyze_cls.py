import argparse
import os
import json
import numpy as np
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from multiprocessing import Pool
from rich.console import Console
from rich.table import Table

from src.analysis.util import (
    get_calibration_params,
    load_calibration,
    normalize_reranker_scores,
    parse_filename_params,
)


def parse_eval_filename(filename):
    name_without_ext = filename.replace(".json", "")
    if not name_without_ext.startswith("eval_results_"):
        return None
    return parse_filename_params(filename)


def load_results(dir: str):
    results = []
    for file in os.listdir(dir):
        if file.endswith(".json"):
            params = parse_eval_filename(file)
            if params is None:
                continue
            file_path = os.path.join(dir, file)
            results.append({"filename": file, "params": params, "file_path": file_path})
    return results


def slice_datapoint_for_k(dp: Dict[str, Any], k: int) -> Dict[str, Any]:
    """Simulate top-k retrieval by slicing candidates and looking up reranker scores."""
    top_k_candidates = dp["retrieved_candidates"][:k]
    top_k_set = set(top_k_candidates)
    score_map = {c: s for c, s in zip(dp["ranked_candidates"], dp["ranked_scores"])}
    ranked_k = [
        (c, score_map[c])
        for c in dp["ranked_candidates"]
        if c in top_k_set and c in score_map
    ]
    if ranked_k:
        ranked_cands, ranked_scores = zip(*ranked_k)
    else:
        ranked_cands, ranked_scores = [], []
    return {
        **dp,
        "retrieved_candidates": top_k_candidates,
        "retrieved_scores": dp["retrieved_scores"][:k],
        "ranked_candidates": list(ranked_cands),
        "ranked_scores": list(ranked_scores),
    }


def _precompute_scores(
    data_points: List[Dict[str, Any]],
    setup: str,
    normalize_scores: bool = False,
    reranker_type: str = None,
    calib_params: Optional[dict] = None,
    calibration_method: str = "temperature",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute threshold-independent per-datapoint scores for a fast threshold sweep.

    Returns (labels, gt_scores, top_scores, is_correct, has_candidates):
      labels[i]        = dp["label"]: 1 if c* is a genuine match, 0 otherwise.
      gt_scores[i]     = s(q, c*): score of the labeled candidate if c* was retrieved,
                         else 0.0. Used as the ranking signal for PR-AUC via
                         average_precision_score.
      top_scores[i]    = max_k s(q, c_k): score of the highest-ranked candidate.
                         Determines whether the cache fires (CHR).
      is_correct[i]    = (top_candidate == c*). True when the system would return c*.
                         Invariant: is_correct=True implies has_candidates=True and
                         top_scores[i] == s(q, c*).
      has_candidates[i]= False only when the candidate pool is empty after filtering
                         out the query itself.

    Metric definitions at threshold tau:
      CHR (cache hit):       has_candidates[i] AND top_scores[i] >= tau
                             — cache fires regardless of correctness or label.
      VCHR (valid cache hit): labels[i]==1 AND is_correct[i] AND cache_hit
                             — cache fires, returns c*, and c* is a genuine match.
                             Equivalently: VCHR = TP / total.
      TP:                    labels[i]==1 AND is_correct[i] AND cache_hit
      FP:                    cache_hit AND NOT TP  (wrong candidate OR label==0)
      FN:                    labels[i]==1 AND NOT cache_hit
      TN:                    labels[i]==0 AND NOT cache_hit
      Precision = TP / (TP+FP) = TP / total_cache_hits = VCHR / CHR
      Recall    = TP / (TP+FN)
    """
    labels, gt_scores, top_scores, is_correct_arr, has_candidates = [], [], [], [], []
    for dp in data_points:
        query = dp["query"]
        ground_truth = dp["ground_truth"]

        if setup == "retriever":
            pairs = [
                (c, s)
                for c, s in zip(dp["retrieved_candidates"], dp["retrieved_scores"])
                if c != query
            ]
        elif setup == "retriever+reranker":
            all_scores = list(dp["ranked_scores"])
            if all_scores and normalize_scores:
                all_scores = normalize_reranker_scores(
                    all_scores,
                    reranker_type=reranker_type,
                    calib_params=calib_params,
                    calibration_method=calibration_method,
                )
            pairs = [
                (c, s)
                for c, s in zip(dp["ranked_candidates"], all_scores)
                if c != query
            ]
        else:
            raise ValueError(f"Unknown setup: {setup}")

        labels.append(dp["label"])
        if not pairs:
            gt_scores.append(0.0)
            top_scores.append(0.0)
            is_correct_arr.append(False)
            has_candidates.append(False)
            continue

        candidates, scores = zip(*pairs)
        scores_arr = np.array(scores, dtype=np.float64)
        top_idx = int(np.argmax(scores_arr))
        top_candidate = candidates[top_idx]
        gt_score = (
            float(scores_arr[list(candidates).index(ground_truth)])
            if ground_truth in candidates
            else 0.0
        )
        gt_scores.append(gt_score)
        top_scores.append(float(scores_arr[top_idx]))
        is_correct_arr.append(top_candidate == ground_truth)
        has_candidates.append(True)

    return (
        np.array(labels, dtype=np.int32),
        np.array(gt_scores, dtype=np.float64),
        np.array(top_scores, dtype=np.float64),
        np.array(is_correct_arr, dtype=bool),
        np.array(has_candidates, dtype=bool),
    )


def compute_metrics_across_thresholds(
    data_points: List[Dict[str, Any]],
    thresholds: List[float],
    setup: str,
    normalize_scores: bool = False,
    reranker_type: str = None,
    calib_params: Optional[dict] = None,
    calibration_method: str = "temperature",
    pbar=None,
) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
    y_true, y_scores, top_scores, is_correct, has_candidates = _precompute_scores(
        data_points,
        setup,
        normalize_scores,
        reranker_type,
        calib_params,
        calibration_method,
    )

    metrics_per_threshold = []
    for threshold in thresholds:
        cache_decisions = has_candidates & (top_scores >= threshold)

        total = len(y_true)
        tp = int(np.sum((y_true == 1) & is_correct & cache_decisions))
        total_cache_hits = int(np.sum(cache_decisions))
        fp = total_cache_hits - tp
        fn = int(np.sum((y_true == 1) & ~cache_decisions))
        tn = int(np.sum((y_true == 0) & ~cache_decisions))

        precision = tp / total_cache_hits if total_cache_hits > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cache_hit_ratio = total_cache_hits / total if total > 0 else 0.0
        valid_cache_hit_ratio = (
            tp / total if total > 0 else 0.0
        )  # VCHR = precision × CHR

        metrics_per_threshold.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "cache_hit_ratio": cache_hit_ratio,
                "valid_cache_hit_ratio": valid_cache_hit_ratio,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "total": total,
            }
        )
        if pbar is not None:
            pbar.update(1)

    return metrics_per_threshold, y_true, y_scores


def _deduplicate_curve(xs, ys):
    """For duplicate x values keep the maximum y. Assumes xs is sorted."""
    unique_xs, unique_ys = [], []
    i = 0
    while i < len(xs):
        current_x = xs[i]
        max_y = ys[i]
        j = i + 1
        while j < len(xs) and xs[j] == current_x:
            max_y = max(max_y, ys[j])
            j += 1
        unique_xs.append(current_x)
        unique_ys.append(max_y)
        i = j
    return unique_xs, unique_ys


def compute_auc_metrics(
    metrics: List[Dict[str, float]],
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    metrics_sorted = sorted(metrics, key=lambda x: x["threshold"])
    precisions = np.array([m["precision"] for m in metrics_sorted])
    cache_hit_ratios = np.array([m["cache_hit_ratio"] for m in metrics_sorted])
    valid_cache_hit_ratios = np.array(
        [m["valid_cache_hit_ratio"] for m in metrics_sorted]
    )

    # PR AUC
    if len(np.unique(y_true)) > 1:
        pr_auc = average_precision_score(y_true, y_scores)
    else:
        print(
            "Warning: PR-AUC is undefined when all labels are identical. Returning 0.0."
        )
        pr_auc = 0.0

    # Precision-CHR AUC
    chr_sort_idx = np.argsort(cache_hit_ratios)
    unique_chr, unique_prec_chr = _deduplicate_curve(
        cache_hit_ratios[chr_sort_idx], precisions[chr_sort_idx]
    )
    precision_chr_auc = auc(unique_chr, unique_prec_chr) if len(unique_chr) > 1 else 0.0

    # Precision-VCHR AUC
    vchr_sort_idx = np.argsort(valid_cache_hit_ratios)
    unique_vchr, unique_prec_vchr = _deduplicate_curve(
        valid_cache_hit_ratios[vchr_sort_idx], precisions[vchr_sort_idx]
    )
    precision_vchr_auc = (
        auc(unique_vchr, unique_prec_vchr) if len(unique_vchr) > 1 else 0.0
    )

    return {
        "pr_auc": pr_auc,
        "precision_chr_auc": precision_chr_auc,
        "precision_vchr_auc": precision_vchr_auc,
    }


def _f1_optimal_metrics(metrics: List[Dict[str, float]]) -> Tuple[float, float, float]:
    """Return (precision, recall, valid_cache_hit_ratio) at the threshold that maximizes F1."""
    best_f1, best_p, best_r, best_vchr = -1.0, 0.0, 0.0, 0.0
    for m in metrics:
        p, r = m["precision"], m["recall"]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_p, best_r, best_vchr = f1, p, r, m["valid_cache_hit_ratio"]
    return best_p, best_r, best_vchr


def _build_pr_curves_for_k(
    all_results: List[Dict[str, Any]], k: int, colors: np.ndarray
) -> List[Dict[str, Any]]:
    pr_curves = []
    plotted_retrievers = set()
    for idx, result in enumerate(all_results):
        label = result["label"]
        retriever_label = label.split("+")[0] if "+" in label else label
        kr = result["k_results"][k]

        if retriever_label not in plotted_retrievers:
            precs, recs, _ = precision_recall_curve(
                kr["retriever_y_true"], kr["retriever_y_scores"]
            )
            ret_auc = kr["retriever_auc"]["pr_auc"]
            pr_curves.append(
                {
                    "auc": ret_auc,
                    "recalls": recs,
                    "precisions": precs,
                    "label": f"{retriever_label} (AUC={ret_auc:.3f})",
                    "linewidth": 1.5,
                    "linestyle": ":",
                    "alpha": 0.7,
                    "color": colors[idx * 2],
                }
            )
            plotted_retrievers.add(retriever_label)

        precs_rer, recs_rer, _ = precision_recall_curve(
            kr["reranker_y_true"], kr["reranker_y_scores"]
        )
        rer_auc = kr["reranker_auc"]["pr_auc"]
        pr_curves.append(
            {
                "auc": rer_auc,
                "recalls": recs_rer,
                "precisions": precs_rer,
                "label": f"{label} (AUC={rer_auc:.3f})",
                "linewidth": 2,
                "linestyle": "-",
                "alpha": 1.0,
                "color": colors[idx * 2 + 1],
            }
        )
    return pr_curves


def _build_chr_curves_for_k(
    all_results: List[Dict[str, Any]], k: int, colors: np.ndarray
) -> List[Dict[str, Any]]:
    chr_curves = []
    plotted_retrievers = set()
    for idx, result in enumerate(all_results):
        label = result["label"]
        retriever_label = label.split("+")[0] if "+" in label else label
        kr = result["k_results"][k]

        if retriever_label not in plotted_retrievers:
            ret_chrs = np.array([m["cache_hit_ratio"] for m in kr["retriever_metrics"]])
            ret_precs = np.array([m["precision"] for m in kr["retriever_metrics"]])
            sort_idx = np.argsort(ret_chrs)
            ret_chr_auc = kr["retriever_auc"]["precision_chr_auc"]
            chr_curves.append(
                {
                    "auc": ret_chr_auc,
                    "chrs": ret_chrs[sort_idx],
                    "precisions": ret_precs[sort_idx],
                    "label": f"{retriever_label} (AUC={ret_chr_auc:.3f})",
                    "linewidth": 1.5,
                    "linestyle": ":",
                    "alpha": 0.7,
                    "color": colors[idx * 2],
                }
            )
            plotted_retrievers.add(retriever_label)

        rer_chrs = np.array([m["cache_hit_ratio"] for m in kr["reranker_metrics"]])
        rer_precs = np.array([m["precision"] for m in kr["reranker_metrics"]])
        sort_idx = np.argsort(rer_chrs)
        rer_chr_auc = kr["reranker_auc"]["precision_chr_auc"]
        chr_curves.append(
            {
                "auc": rer_chr_auc,
                "chrs": rer_chrs[sort_idx],
                "precisions": rer_precs[sort_idx],
                "label": f"{label} (AUC={rer_chr_auc:.3f})",
                "linewidth": 2,
                "linestyle": "-",
                "alpha": 1.0,
                "color": colors[idx * 2 + 1],
            }
        )
    return chr_curves


def _build_vchr_curves_for_k(
    all_results: List[Dict[str, Any]], k: int, colors: np.ndarray
) -> List[Dict[str, Any]]:
    vchr_curves = []
    plotted_retrievers = set()
    for idx, result in enumerate(all_results):
        label = result["label"]
        retriever_label = label.split("+")[0] if "+" in label else label
        kr = result["k_results"][k]

        if retriever_label not in plotted_retrievers:
            ret_vchrs = np.array(
                [m["valid_cache_hit_ratio"] for m in kr["retriever_metrics"]]
            )
            ret_precs = np.array([m["precision"] for m in kr["retriever_metrics"]])
            sort_idx = np.argsort(ret_vchrs)
            ret_vchr_auc = kr["retriever_auc"]["precision_vchr_auc"]
            vchr_curves.append(
                {
                    "auc": ret_vchr_auc,
                    "vchrs": ret_vchrs[sort_idx],
                    "precisions": ret_precs[sort_idx],
                    "label": f"{retriever_label} (AUC={ret_vchr_auc:.3f})",
                    "linewidth": 1.5,
                    "linestyle": ":",
                    "alpha": 0.7,
                    "color": colors[idx * 2],
                }
            )
            plotted_retrievers.add(retriever_label)

        rer_vchrs = np.array(
            [m["valid_cache_hit_ratio"] for m in kr["reranker_metrics"]]
        )
        rer_precs = np.array([m["precision"] for m in kr["reranker_metrics"]])
        sort_idx = np.argsort(rer_vchrs)
        rer_vchr_auc = kr["reranker_auc"]["precision_vchr_auc"]
        vchr_curves.append(
            {
                "auc": rer_vchr_auc,
                "vchrs": rer_vchrs[sort_idx],
                "precisions": rer_precs[sort_idx],
                "label": f"{label} (AUC={rer_vchr_auc:.3f})",
                "linewidth": 2,
                "linestyle": "-",
                "alpha": 1.0,
                "color": colors[idx * 2 + 1],
            }
        )
    return vchr_curves


def plot_auc_vs_k(
    all_processed: List[Dict[str, Any]],
    output_dir: str,
):
    if not all_processed:
        return
    os.makedirs(output_dir, exist_ok=True)
    num_colors = max(len(all_processed) * 2, 16)
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
    k_values = sorted(all_processed[0]["k_results"].keys())

    for metric_key, metric_name, filename in [
        ("pr_auc", "PR-AUC", "pr_auc_vs_k.png"),
        ("precision_chr_auc", "P-CHR-AUC", "pchr_auc_vs_k.png"),
        ("precision_vchr_auc", "P-VCHR-AUC", "pvchr_auc_vs_k.png"),
    ]:
        print(f"\nCreating {metric_name} vs k plot...")
        fig, ax = plt.subplots(figsize=(14, 10))

        curves = []
        plotted_retrievers = set()

        for idx, result in enumerate(all_processed):
            label = result["label"]
            retriever_label = label.split("+")[0] if "+" in label else label
            K = result["K"]
            ks = sorted(result["k_results"].keys())

            if retriever_label not in plotted_retrievers:
                ret_aucs = [
                    result["k_results"][k]["retriever_auc"][metric_key] for k in ks
                ]
                curves.append(
                    {
                        "auc_at_K": ret_aucs[-1],
                        "ks": ks,
                        "aucs": ret_aucs,
                        "label": f"{retriever_label} (k={K}: {ret_aucs[-1]:.3f})",
                        "linewidth": 1.5,
                        "linestyle": ":",
                        "alpha": 0.7,
                        "color": colors[idx * 2],
                    }
                )
                plotted_retrievers.add(retriever_label)

            rer_aucs = [result["k_results"][k]["reranker_auc"][metric_key] for k in ks]
            curves.append(
                {
                    "auc_at_K": rer_aucs[-1],
                    "ks": ks,
                    "aucs": rer_aucs,
                    "label": f"{label} (k={K}: {rer_aucs[-1]:.3f})",
                    "linewidth": 2,
                    "linestyle": "-",
                    "alpha": 1.0,
                    "color": colors[idx * 2 + 1],
                }
            )

        curves.sort(key=lambda x: x["auc_at_K"], reverse=True)
        for curve in curves:
            ax.plot(
                curve["ks"],
                curve["aucs"],
                label=curve["label"],
                linewidth=curve["linewidth"],
                linestyle=curve["linestyle"],
                alpha=curve["alpha"],
                color=curve["color"],
            )

        ax.set_xlabel("k (number of retrieved candidates)", fontsize=13)
        ax.set_ylabel(metric_name, fontsize=13)
        ax.set_title(f"{metric_name} vs k", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(k_values), max(k_values)])
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close()

    # Precision and Recall at F1-optimal threshold vs k
    for metric_name, filename, metric_idx in [
        ("Precision (at F1-optimal threshold)", "precision_vs_k.png", 0),
        ("Recall (at F1-optimal threshold)", "recall_vs_k.png", 1),
    ]:
        print(f"\nCreating {metric_name} vs k plot...")
        fig, ax = plt.subplots(figsize=(14, 10))

        curves = []
        plotted_retrievers = set()

        for idx, result in enumerate(all_processed):
            label = result["label"]
            retriever_label = label.split("+")[0] if "+" in label else label
            K = result["K"]
            ks = sorted(result["k_results"].keys())

            if retriever_label not in plotted_retrievers:
                ret_vals = [
                    _f1_optimal_metrics(result["k_results"][k]["retriever_metrics"])[
                        metric_idx
                    ]
                    for k in ks
                ]
                curves.append(
                    {
                        "val_at_K": ret_vals[-1],
                        "ks": ks,
                        "vals": ret_vals,
                        "label": f"{retriever_label} (k={K}: {ret_vals[-1]:.3f})",
                        "linewidth": 1.5,
                        "linestyle": ":",
                        "alpha": 0.7,
                        "color": colors[idx * 2],
                    }
                )
                plotted_retrievers.add(retriever_label)

            rer_vals = [
                _f1_optimal_metrics(result["k_results"][k]["reranker_metrics"])[
                    metric_idx
                ]
                for k in ks
            ]
            curves.append(
                {
                    "val_at_K": rer_vals[-1],
                    "ks": ks,
                    "vals": rer_vals,
                    "label": f"{label} (k={K}: {rer_vals[-1]:.3f})",
                    "linewidth": 2,
                    "linestyle": "-",
                    "alpha": 1.0,
                    "color": colors[idx * 2 + 1],
                }
            )

        curves.sort(key=lambda x: x["val_at_K"], reverse=True)
        for curve in curves:
            ax.plot(
                curve["ks"],
                curve["vals"],
                label=curve["label"],
                linewidth=curve["linewidth"],
                linestyle=curve["linestyle"],
                alpha=curve["alpha"],
                color=curve["color"],
            )

        ax.set_xlabel("k (number of retrieved candidates)", fontsize=13)
        ax.set_ylabel(metric_name, fontsize=13)
        ax.set_title(f"{metric_name} vs k", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(k_values), max(k_values)])
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close()


def plot_per_k_curves(
    all_processed: List[Dict[str, Any]],
    output_dir: str,
):
    if not all_processed:
        return
    K = all_processed[0]["K"]
    k_values = sorted(all_processed[0]["k_results"].keys())
    num_colors = max(len(all_processed) * 2, 16)
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))

    print(
        f"\nGenerating per-k threshold-sweep plots (k=1..{K}) → {output_dir}/k{{01..{K:02d}}}/"
    )
    for k in k_values:
        k_dir = os.path.join(output_dir, f"k{k:02d}")
        os.makedirs(k_dir, exist_ok=True)

        # PR curves
        fig, ax = plt.subplots(figsize=(14, 10))
        pr_curves = _build_pr_curves_for_k(all_processed, k, colors)
        pr_curves.sort(key=lambda x: x["auc"], reverse=True)
        for curve in pr_curves:
            ax.plot(
                curve["recalls"],
                curve["precisions"],
                label=curve["label"],
                linewidth=curve["linewidth"],
                linestyle=curve["linestyle"],
                alpha=curve["alpha"],
                color=curve["color"],
            )
        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title(f"Precision vs Recall (k={k})", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(
            os.path.join(k_dir, "combined_pr_curves.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        # CHR curves
        fig, ax = plt.subplots(figsize=(14, 10))
        chr_curves = _build_chr_curves_for_k(all_processed, k, colors)
        chr_curves.sort(key=lambda x: x["auc"], reverse=True)
        for curve in chr_curves:
            ax.plot(
                curve["chrs"],
                curve["precisions"],
                label=curve["label"],
                linewidth=curve["linewidth"],
                linestyle=curve["linestyle"],
                alpha=curve["alpha"],
                color=curve["color"],
            )
        ax.set_xlabel("Cache Hit Ratio", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title(
            f"Precision vs Cache Hit Ratio (k={k})", fontsize=14, fontweight="bold"
        )
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(
            os.path.join(k_dir, "combined_precision_chr_curves.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # VCHR curves
        fig, ax = plt.subplots(figsize=(14, 10))
        vchr_curves = _build_vchr_curves_for_k(all_processed, k, colors)
        vchr_curves.sort(key=lambda x: x["auc"], reverse=True)
        for curve in vchr_curves:
            ax.plot(
                curve["vchrs"],
                curve["precisions"],
                label=curve["label"],
                linewidth=curve["linewidth"],
                linestyle=curve["linestyle"],
                alpha=curve["alpha"],
                color=curve["color"],
            )
        ax.set_xlabel("Valid Cache Hit Ratio", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title(
            f"Precision vs Valid Cache Hit Ratio (k={k})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(
            os.path.join(k_dir, "combined_precision_vchr_curves.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(
        f"  Done: {3 * K} plots saved under {output_dir}/k01/ .. k{K:02d}/ "
        f"(combined_pr_curves.png, combined_precision_chr_curves.png, combined_precision_vchr_curves.png)"
    )


def process_result(args_tuple):
    (
        result,
        thresholds,
        calibration_data,
        calibration_method,
    ) = args_tuple

    try:
        with open(result["file_path"], "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {result['file_path']}: {e}")
        return None

    reranker_type = None
    reranker_model = None
    if result["params"]:
        biencoder = (
            result["params"].get("biencoder_model_path", "unknown").split("--")[-1]
        )
        reranker_model = result["params"].get("reranker_model_path", "unknown")
        reranker_type = result["params"].get("reranker_type")
        label = f"{biencoder}+{reranker_model.split('--')[-1]}"
    else:
        label = result["filename"].replace(".json", "")

    calib_params = get_calibration_params(reranker_model, calibration_data)
    data_points = data["results"]
    K = len(data_points[0]["retrieved_candidates"]) if data_points else 1

    k_results = {}
    for k in range(1, K + 1):
        sliced = [slice_datapoint_for_k(dp, k) for dp in data_points]

        retriever_metrics, ret_y_true, ret_y_scores = compute_metrics_across_thresholds(
            sliced, thresholds, setup="retriever"
        )
        retriever_auc = compute_auc_metrics(retriever_metrics, ret_y_true, ret_y_scores)

        reranker_metrics, rer_y_true, rer_y_scores = compute_metrics_across_thresholds(
            sliced,
            thresholds,
            setup="retriever+reranker",
            normalize_scores=True,
            reranker_type=reranker_type,
            calib_params=calib_params,
            calibration_method=calibration_method,
        )
        reranker_auc = compute_auc_metrics(reranker_metrics, rer_y_true, rer_y_scores)

        k_results[k] = {
            "retriever_metrics": retriever_metrics,
            "retriever_auc": retriever_auc,
            "retriever_y_true": ret_y_true,
            "retriever_y_scores": ret_y_scores,
            "reranker_metrics": reranker_metrics,
            "reranker_auc": reranker_auc,
            "reranker_y_true": rer_y_true,
            "reranker_y_scores": rer_y_scores,
        }

    max_kr = k_results[K]
    print(
        f"  [{label}] k={K}:"
        f"  Retriever PR-AUC={max_kr['retriever_auc']['pr_auc']:.4f}"
        f"  P-CHR-AUC={max_kr['retriever_auc']['precision_chr_auc']:.4f}"
        f"  P-VCHR-AUC={max_kr['retriever_auc']['precision_vchr_auc']:.4f}"
        f"  |  Reranker PR-AUC={max_kr['reranker_auc']['pr_auc']:.4f}"
        f"  P-CHR-AUC={max_kr['reranker_auc']['precision_chr_auc']:.4f}"
        f"  P-VCHR-AUC={max_kr['reranker_auc']['precision_vchr_auc']:.4f}"
    )

    return {
        "label": label,
        "K": K,
        "k_results": k_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analyze results (PR/CHR metrics across k)")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing evaluation result files",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Thresholds to sweep (default: 0.0–1.0 step 0.01)",
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

    if args.thresholds is None:
        args.thresholds = np.arange(0.0, 1.01, 0.01).tolist()

    num_workers = (
        int(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)) * 0.75)
        if args.workers == -1
        else args.workers
    )

    print("Configuration:")
    print(f"  Input directory:  {args.results_dir}")
    print(f"  Output directory: {args.plots_dir}")
    print(f"  Output JSON:      {args.output}")
    print(
        f"  Thresholds:       {len(args.thresholds)} values "
        f"[{min(args.thresholds):.3f}, {max(args.thresholds):.3f}]"
    )
    print(f"  Calibration:      {args.calibration or 'none'}")
    if args.calibration:
        print(f"  Calib method:     {args.calibration_method}")
    print(f"  Workers:          {num_workers}")

    results = load_results(args.results_dir)
    print(f"\nFound {len(results)} result files (data loaded per-worker)")
    os.makedirs(args.plots_dir, exist_ok=True)

    worker_args = [
        (
            r,
            args.thresholds,
            calibration_data,
            args.calibration_method,
        )
        for r in results
    ]

    print(f"\nProcessing {len(results)} files with {num_workers} worker(s)...")
    with Pool(processes=num_workers) as pool:
        all_processed = [
            r
            for r in tqdm(
                pool.imap(process_result, worker_args),
                total=len(worker_args),
                desc="Files completed",
                leave=True,
                dynamic_ncols=True,
            )
            if r is not None
        ]

    # Per-model-combo tables: k as rows, all metrics as columns
    console = Console(width=300)
    for result in all_processed:
        label = result["label"]
        k_values = sorted(result["k_results"].keys())

        table = Table(
            title=f"Metrics — {label}",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("k", justify="right", min_width=4)
        table.add_column("PR-AUC (Ret)", justify="right", min_width=13)
        table.add_column("P-CHR (Ret)", justify="right", min_width=11)
        table.add_column("P-VCHR (Ret)", justify="right", min_width=12)
        table.add_column("Prec (Ret)", justify="right", min_width=10)
        table.add_column("Rec (Ret)", justify="right", min_width=9)
        table.add_column("VCHR (Ret)", justify="right", min_width=10)
        table.add_column("PR-AUC (Rer)", justify="right", min_width=13)
        table.add_column("P-CHR (Rer)", justify="right", min_width=11)
        table.add_column("P-VCHR (Rer)", justify="right", min_width=12)
        table.add_column("Prec (Rer)", justify="right", min_width=10)
        table.add_column("Rec (Rer)", justify="right", min_width=9)
        table.add_column("VCHR (Rer)", justify="right", min_width=10)

        for k in k_values:
            kr = result["k_results"][k]
            ret_prec, ret_rec, ret_vchr = _f1_optimal_metrics(kr["retriever_metrics"])
            rer_prec, rer_rec, rer_vchr = _f1_optimal_metrics(kr["reranker_metrics"])
            table.add_row(
                str(k),
                f"{kr['retriever_auc']['pr_auc']:.4f}",
                f"{kr['retriever_auc']['precision_chr_auc']:.4f}",
                f"{kr['retriever_auc']['precision_vchr_auc']:.4f}",
                f"{ret_prec:.4f}",
                f"{ret_rec:.4f}",
                f"{ret_vchr:.4f}",
                f"{kr['reranker_auc']['pr_auc']:.4f}",
                f"{kr['reranker_auc']['precision_chr_auc']:.4f}",
                f"{kr['reranker_auc']['precision_vchr_auc']:.4f}",
                f"{rer_prec:.4f}",
                f"{rer_rec:.4f}",
                f"{rer_vchr:.4f}",
            )
        console.print(table)

    results_json = []
    for r in all_processed:
        k_results_json = {}
        for k in sorted(r["k_results"].keys()):
            kr = r["k_results"][k]
            ret_p, ret_r, ret_vchr = _f1_optimal_metrics(kr["retriever_metrics"])
            rer_p, rer_r, rer_vchr = _f1_optimal_metrics(kr["reranker_metrics"])
            k_results_json[str(k)] = {
                "retriever_pr_auc": kr["retriever_auc"]["pr_auc"],
                "retriever_precision_chr_auc": kr["retriever_auc"]["precision_chr_auc"],
                "retriever_precision_vchr_auc": kr["retriever_auc"][
                    "precision_vchr_auc"
                ],
                "retriever_precision": ret_p,
                "retriever_recall": ret_r,
                "retriever_valid_cache_hit_ratio": ret_vchr,
                "reranker_pr_auc": kr["reranker_auc"]["pr_auc"],
                "reranker_precision_chr_auc": kr["reranker_auc"]["precision_chr_auc"],
                "reranker_precision_vchr_auc": kr["reranker_auc"]["precision_vchr_auc"],
                "reranker_precision": rer_p,
                "reranker_recall": rer_r,
                "reranker_valid_cache_hit_ratio": rer_vchr,
            }
        results_json.append(
            {"label": r["label"], "K": r["K"], "k_results": k_results_json}
        )

    output_data = {
        "calibration": args.calibration,
        "calibration_method": args.calibration_method if args.calibration else None,
        "results": results_json,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Saved metrics to {args.output}")

    # Plots
    print(f"\n{'=' * 80}")
    print("GENERATING PLOTS")
    print(f"{'=' * 80}")
    if not all_processed:
        print("No results to plot.")
    else:
        plot_auc_vs_k(all_processed, args.plots_dir)
        plot_per_k_curves(all_processed, args.plots_dir)
        K = all_processed[0]["K"]
        print(f"\n✓ All plots saved to: {args.plots_dir}")
        print("  Summary plots (5 files):")
        print(f"    {os.path.join(args.plots_dir, 'pr_auc_vs_k.png')}")
        print(f"    {os.path.join(args.plots_dir, 'pchr_auc_vs_k.png')}")
        print(f"    {os.path.join(args.plots_dir, 'pvchr_auc_vs_k.png')}")
        print(f"    {os.path.join(args.plots_dir, 'precision_vs_k.png')}")
        print(f"    {os.path.join(args.plots_dir, 'recall_vs_k.png')}")
        print(f"  Per-k threshold-sweep plots ({3 * K} files):")
        print(
            f"    {args.plots_dir}/k01/ .. k{K:02d}/ × {{combined_pr_curves.png, combined_precision_chr_curves.png, combined_precision_vchr_curves.png}}"
        )
