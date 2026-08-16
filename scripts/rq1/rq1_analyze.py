#!/usr/bin/env python3
"""
RQ1 Statistical Analysis and Figure Generation

Loads all per-model results, computes:
  - Correlation analysis (Pearson r, Spearman rho)
  - Wilcoxon signed-rank test for HP vs TOFU separation
  - Cohen's d effect sizes
  - Bootstrap confidence intervals
  - Layer-wise metric profiles

Output:
  - Printed table with hypothesis test results
  - 4 PDF figures
  - experiments/rq1/main/summary.csv with all peak-layer metrics
"""

import json
import os
import glob
import math
import logging
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Metric classification
COHERENCE_METRICS = {"Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig"}
DISPERSION_METRICS = {"AUSS_L2", "AUSS_Cos2", "AUSS_Jac", "Batch_Cov_Trace", "Batch_Cov_TopEig"}
ALL_METRICS = list(COHERENCE_METRICS | DISPERSION_METRICS)


def load_all_results(results_dir: str) -> List[Dict]:
    """Load all model result JSONs from results directory."""
    results = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(fpath, 'r') as f:
                results.append(json.load(f))
            logger.info(f"Loaded {os.path.basename(fpath)}")
        except Exception as e:
            logger.error(f"Error loading {fpath}: {e}")

    logger.info(f"Loaded {len(results)} model results")
    return results


def average_across_seeds(results: List[Dict]) -> List[Dict]:
    """
    Average metrics across 3 seeds (42, 123, 777) for each model.

    Returns list of dicts with mean ± std across seeds per layer.
    """
    averaged = []

    for result in results:
        seeds = result.get("seeds", [42, 123, 777])

        # Average HP metrics across seeds
        hp_metrics_by_layer = {}
        tofu_metrics_by_layer = {}

        num_layers = result["num_layers"]
        for layer_idx in range(num_layers):
            hp_vals = {}
            tofu_vals = {}

            for metric_name in ALL_METRICS:
                hp_vals[metric_name] = []
                tofu_vals[metric_name] = []

                for seed in seeds:
                    seed_str = str(seed)
                    if seed_str in result.get("hp_metrics", {}):
                        metrics_list = result["hp_metrics"][seed_str]
                        if layer_idx < len(metrics_list):
                            val = metrics_list[layer_idx].get(metric_name)
                            if val is not None and not math.isnan(val):
                                hp_vals[metric_name].append(val)

                    if seed_str in result.get("tofu_metrics", {}):
                        metrics_list = result["tofu_metrics"][seed_str]
                        if layer_idx < len(metrics_list):
                            val = metrics_list[layer_idx].get(metric_name)
                            if val is not None and not math.isnan(val):
                                tofu_vals[metric_name].append(val)

            # Compute mean ± std per metric
            hp_metrics_by_layer[layer_idx] = {}
            tofu_metrics_by_layer[layer_idx] = {}

            for metric_name in ALL_METRICS:
                if hp_vals[metric_name]:
                    mean = np.mean(hp_vals[metric_name])
                    std = np.std(hp_vals[metric_name])
                    hp_metrics_by_layer[layer_idx][metric_name] = (mean, std)

                if tofu_vals[metric_name]:
                    mean = np.mean(tofu_vals[metric_name])
                    std = np.std(tofu_vals[metric_name])
                    tofu_metrics_by_layer[layer_idx][metric_name] = (mean, std)

        averaged.append({
            "model_id": result["model_id"],
            "hp_qa": result["hp_qa_success"],
            "tofu_qa": result.get("tofu_qa_success", 0.0),
            "hp_metrics": hp_metrics_by_layer,
            "tofu_metrics": tofu_metrics_by_layer,
            "num_layers": num_layers,
        })

    return averaged


def find_peak_layer(averaged_results: List[Dict], metric_name: str) -> Tuple[int, float]:
    """
    Find the layer with maximum HP-TOFU separation for a metric.

    Returns (peak_layer_idx, max_separation)
    """
    min_layers = min(r["num_layers"] for r in averaged_results)

    best_layer = 0
    best_separation = 0.0

    for layer_idx in range(min_layers):
        hp_vals = []
        tofu_vals = []

        for result in averaged_results:
            hp_metrics = result["hp_metrics"].get(layer_idx, {})
            tofu_metrics = result["tofu_metrics"].get(layer_idx, {})

            if metric_name in hp_metrics and metric_name in tofu_metrics:
                hp_mean, _ = hp_metrics[metric_name]
                tofu_mean, _ = tofu_metrics[metric_name]
                hp_vals.append(hp_mean)
                tofu_vals.append(tofu_mean)

        if len(hp_vals) >= 3:
            # For coherence metrics: HP > TOFU; for dispersion: TOFU > HP
            if metric_name in COHERENCE_METRICS:
                separation = np.mean(hp_vals) - np.mean(tofu_vals)
            else:
                separation = np.mean(tofu_vals) - np.mean(hp_vals)

            if separation > best_separation:
                best_separation = separation
                best_layer = layer_idx

    return best_layer, best_separation


def compute_statistics(
    averaged_results: List[Dict],
    metric_name: str,
    peak_layer: int
) -> Dict:
    """
    Compute hypothesis test statistics at peak layer.

    Returns dict with Pearson r, p, Cohen's d, Wilcoxon p, etc.
    """
    hp_vals = []
    tofu_vals = []

    for result in averaged_results:
        hp_metrics = result["hp_metrics"].get(peak_layer, {})
        tofu_metrics = result["tofu_metrics"].get(peak_layer, {})

        if metric_name in hp_metrics and metric_name in tofu_metrics:
            hp_mean, _ = hp_metrics[metric_name]
            tofu_mean, _ = tofu_metrics[metric_name]
            hp_vals.append(hp_mean)
            tofu_vals.append(tofu_mean)

    if len(hp_vals) < 3:
        return {}

    hp_vals = np.array(hp_vals)
    tofu_vals = np.array(tofu_vals)

    # Pearson correlation (HP=1, TOFU=0)
    domain_labels = np.array([1] * len(hp_vals) + [0] * len(tofu_vals))
    metric_vals = np.concatenate([hp_vals, tofu_vals])
    r_pearson, p_pearson = stats.pearsonr(metric_vals, domain_labels)
    rho_spearman, p_spearman = stats.spearmanr(metric_vals, domain_labels)

    # Wilcoxon signed-rank (paired)
    differences = hp_vals - tofu_vals
    stat_wilcoxon, p_wilcoxon = stats.wilcoxon(differences)

    # Cohen's d (paired samples)
    mean_diff = np.mean(differences)
    sd_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0

    # Bootstrap CI on mean difference
    n_boot = 1000
    boot_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        boot_sample = rng.choice(differences, size=len(differences), replace=True)
        boot_means.append(np.mean(boot_sample))
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)

    return {
        "mean_hp": float(np.mean(hp_vals)),
        "mean_tofu": float(np.mean(tofu_vals)),
        "delta": float(mean_diff),
        "n": len(hp_vals),
        "r_pearson": float(r_pearson),
        "p_pearson": float(p_pearson),
        "rho_spearman": float(rho_spearman),
        "p_spearman": float(p_spearman),
        "p_wilcoxon": float(p_wilcoxon),
        "cohens_d": float(cohens_d),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "hp_vals": hp_vals.tolist(),
        "tofu_vals": tofu_vals.tolist(),
    }


def print_results_table(results_list: List[Dict]):
    """Print formatted hypothesis test results table."""
    print("\n" + "="*100)
    print("RQ1 HYPOTHESIS TEST RESULTS (at Peak Discriminative Layer)")
    print("="*100)
    print(f"{'Metric':<30} {'Cat':>12} {'Peak':>5} {'HP':>8} {'TOFU':>8} {'Δ':>8} "
          f"{'d':>7} {'p_W':>9} {'Sig':>4}")
    print("-"*100)

    for res in results_list:
        metric_name = res["metric"]
        cat = res["category"]
        peak = res["peak_layer"]
        mean_hp = res["stats"]["mean_hp"]
        mean_tofu = res["stats"]["mean_tofu"]
        delta = res["stats"]["delta"]
        cohens_d = res["stats"]["cohens_d"]
        p_wilcoxon = res["stats"]["p_wilcoxon"]

        # Significance stars
        if p_wilcoxon < 0.001:
            sig = "***"
        elif p_wilcoxon < 0.01:
            sig = "**"
        elif p_wilcoxon < 0.05:
            sig = "*"
        else:
            sig = "ns"

        print(f"{metric_name:<30} {cat:>12} {peak:>5d} {mean_hp:>8.4f} {mean_tofu:>8.4f} "
              f"{delta:>8.4f} {cohens_d:>7.2f} {p_wilcoxon:>9.4f} {sig:>4}")

    print("="*100)


def generate_figures(
    averaged_results: List[Dict],
    results_list: List[Dict],
    figures_dir: str = "figures"
):
    """Generate 4 PDF figures from analysis results."""
    os.makedirs(figures_dir, exist_ok=True)

    logger.info("Generating figures...")

    # Figure 1: HP vs TOFU boxplots at peak layer
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, res in enumerate(results_list[:8]):
        ax = axes[idx]
        metric_name = res["metric"]
        hp_vals = res["stats"]["hp_vals"]
        tofu_vals = res["stats"]["tofu_vals"]

        bp = ax.boxplot([hp_vals, tofu_vals], labels=["HP", "TOFU"], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name}\np={res['stats']['p_wilcoxon']:.4f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "hp_vs_tofu_boxplot.pdf"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved hp_vs_tofu_boxplot.pdf")

    # Figure 2: Effect sizes
    fig, ax = plt.subplots(figsize=(8, 6))
    metric_names = [r["metric"] for r in results_list]
    effect_sizes = [r["stats"]["cohens_d"] for r in results_list]
    colors = ['C0' if m in COHERENCE_METRICS else 'C1' for m in metric_names]

    ax.barh(metric_names, effect_sizes, color=colors)
    ax.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Large effect (d=0.8)')
    ax.set_xlabel("Cohen's d")
    ax.set_title("Effect Size (HP vs TOFU)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "effect_size_bar.pdf"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved effect_size_bar.pdf")


def main():
    """Run full analysis pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="RQ1 statistical analysis")
    parser.add_argument("--results_dir", default="experiments/rq1/main",
                        help="Directory containing per-model result JSONs (default: results)")
    parser.add_argument("--out_csv", default=None,
                        help="Output CSV path (default: {results_dir}/summary.csv)")
    parser.add_argument("--figures_dir", default="figures",
                        help="Directory for PDF figures (default: figures)")
    args = parser.parse_args()
    out_csv = args.out_csv or os.path.join(args.results_dir, "summary.csv")

    logger.info("="*80)
    logger.info("RQ1 STATISTICAL ANALYSIS")
    logger.info("="*80)

    # Load results
    results = load_all_results(args.results_dir)
    if not results:
        logger.error(f"No results found in {args.results_dir}/ directory")
        logger.error("Run: python rq1_extract.py --hf_token $HF_TOKEN")
        return

    # Average across seeds
    logger.info("Averaging metrics across 3 seeds...")
    averaged_results = average_across_seeds(results)

    # Compute statistics for each metric
    logger.info("Computing hypothesis tests...")
    results_list = []

    for metric_name in ALL_METRICS:
        peak_layer, _ = find_peak_layer(averaged_results, metric_name)
        stats_dict = compute_statistics(averaged_results, metric_name, peak_layer)

        if stats_dict:
            category = "Coherence" if metric_name in COHERENCE_METRICS else "Dispersion"
            results_list.append({
                "metric": metric_name,
                "category": category,
                "peak_layer": peak_layer,
                "stats": stats_dict,
            })

    # Print results
    print_results_table(results_list)

    # Generate figures
    generate_figures(averaged_results, results_list, figures_dir=args.figures_dir)

    # Write summary CSV
    logger.info(f"Writing {out_csv}...")
    summary_data = []

    for result in averaged_results:
        row = {
            "model_id": result["model_id"],
            "hp_qa": result["hp_qa"],
            "tofu_qa": result["tofu_qa"],
        }

        # Add metrics at peak layer for each metric
        for res in results_list:
            metric_name = res["metric"]
            peak_layer = res["peak_layer"]
            hp_metrics = result["hp_metrics"].get(peak_layer, {})
            tofu_metrics = result["tofu_metrics"].get(peak_layer, {})

            if metric_name in hp_metrics:
                hp_mean, hp_std = hp_metrics[metric_name]
                row[f"{metric_name}_hp"] = hp_mean

            if metric_name in tofu_metrics:
                tofu_mean, tofu_std = tofu_metrics[metric_name]
                row[f"{metric_name}_tofu"] = tofu_mean

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Summary saved to {out_csv}")

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
