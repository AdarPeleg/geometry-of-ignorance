#!/usr/bin/env python3
"""
RQ2 Analysis: AUSS Metrics vs. Attack Recoverability

Loads all __metrics.json and __attacks.json files from results_rq2/,
builds Table 1 (main results) and Table 2 (Spearman correlations),
generates 3 figures, and saves rq2_summary.csv.

Usage:
    python rq2_analyze.py
    python rq2_analyze.py --min_runs 5  # run even with partial data
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results_rq2")
FIGURES_DIR = Path("figures_rq2")

MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
METHODS = ["Base", "GradAscent", "DPO", "NPO", "NPO+KL", "RMU", "WHP"]
CONCEPTS = ["harry_potter", "star_wars", "william_shakespeare"]

AUSS_METRICS = [
    "Centroid_Norm",
    "Global_Coherence",
    "Batch_SecondMoment_TopEig",
    "AUSS_L2",
    "AUSS_Cos2",
    "AUSS_Jac",
    "Batch_Cov_Trace",
    "Batch_Cov_TopEig",
]

ENTROPY_METRICS = [
    "Reg_S_alpha_2",
    "Reg_MeanCos",
    "Reg_MeanL2",
    "AnonDiff_S_alpha_2",
    "AnonDiff_MeanL2",
    "AnonDiff_MeanCos",
]

ATTACK_NAMES = ["Steering", "ICL", "GCG", "MIA"]

ATTACK_METRIC_NAMES = {
    "Steering":         "Steering AUC",
    "Steering_ASR":     "Steering ASR",       # mean_steered_wf — absolute word-freq score
    "Steering_retain_wf": "Steer retain-wf",  # concept keyword freq on retain questions
    "ICL":              "ICL Acc",
    "GCG":              "GCG Acc",
    "MIA":              "MIA AUC",
}

# Verification thresholds (from plan)
FORGET_ROUGE_THRESHOLD = 0.10
RETAIN_ROUGE_RATIO = 0.80  # post retain >= 0.80 * base retain

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def run_id(model_id: str, method: str, concept: str) -> str:
    model_safe = model_id.replace("/", "__")
    return f"{model_safe}__{method}__{concept}"


def load_all_results() -> List[Dict]:
    """Load all __metrics.json and __attacks.json pairs and merge into rows."""
    rows = []

    for model_id in MODELS:
        for method in METHODS:
            for concept in CONCEPTS:
                rid = run_id(model_id, method, concept)
                m_path = RESULTS_DIR / f"{rid}__metrics.json"
                a_path = RESULTS_DIR / f"{rid}__attacks.json"

                row = {
                    "model_id": model_id,
                    "model_short": model_id.split("/")[-1],
                    "method": method,
                    "concept": concept,
                }

                # Load metrics
                if m_path.exists():
                    with open(m_path) as f:
                        m = json.load(f)
                    def _f(v): return float(v) if v is not None else float("nan")
                    row["forget_rouge"] = _f(m.get("post_verification", {}).get("forget_rouge_l"))
                    row["retain_rouge"] = _f(m.get("post_verification", {}).get("retain_rouge_l"))
                    row["base_retain_rouge"] = _f(m.get("base_verification", {}).get("retain_rouge_l"))
                    row["qa_accuracy"] = m.get("qa_accuracy", float("nan"))
                    # AUSS peak metrics: use layer with max AUSS_L2 (most discriminative per RQ1)
                    all_layers = m.get("auss_metrics", {}).get("all_layers", [])
                    if all_layers:
                        peak_idx = max(range(len(all_layers)), key=lambda i: all_layers[i].get("AUSS_L2", 0.0))
                        peak = all_layers[peak_idx]
                    else:
                        peak = m.get("auss_metrics", {}).get("peak_metrics", {})
                    for metric in AUSS_METRICS:
                        row[metric] = peak.get(metric, float("nan"))
                    row["has_metrics"] = True
                    # Entropy: check metrics JSON first (new pipeline), then sidecar file
                    inline_entropy = m.get("auss_metrics", {}).get("entropy_metrics", {})
                    if inline_entropy:
                        entropy = inline_entropy
                    else:
                        e_path = RESULTS_DIR / f"{rid}__entropy.json"
                        if e_path.exists():
                            with open(e_path) as f:
                                entropy = json.load(f).get("entropy_metrics", {})
                        else:
                            entropy = {}
                    for metric in ENTROPY_METRICS:
                        v = entropy.get(metric)
                        row[metric] = float(v) if v is not None else float("nan")
                else:
                    row["forget_rouge"] = float("nan")
                    row["retain_rouge"] = float("nan")
                    row["base_retain_rouge"] = float("nan")
                    row["qa_accuracy"] = float("nan")
                    for metric in AUSS_METRICS:
                        row[metric] = float("nan")
                    for metric in ENTROPY_METRICS:
                        row[metric] = float("nan")
                    row["has_metrics"] = False

                # Load attacks
                if a_path.exists():
                    with open(a_path) as f:
                        a = json.load(f)
                    for attack in ATTACK_NAMES:
                        score = a.get("attacks", {}).get(attack, {}).get("score", float("nan"))
                        row[f"attack_{attack}"] = float(score) if score is not None else float("nan")
                    # Steering ASR and retain-wf: V2 runs only (hook == "decode_step")
                    steer_d = a.get("attacks", {}).get("Steering", {})
                    is_v2 = steer_d.get("hook") == "decode_step"
                    msw = steer_d.get("mean_steered_wf") if is_v2 else None
                    mrw = steer_d.get("mean_retain_wf") if is_v2 else None
                    row["attack_Steering_ASR"] = float(msw) if msw is not None else float("nan")
                    row["attack_Steering_retain_wf"] = float(mrw) if mrw is not None else float("nan")
                    row["has_attacks"] = True
                else:
                    for attack in ATTACK_NAMES:
                        row[f"attack_{attack}"] = float("nan")
                    row["attack_Steering_ASR"] = float("nan")
                    row["attack_Steering_retain_wf"] = float("nan")
                    row["has_attacks"] = False

                # Verification flag: Base rows always fail; unlearn rows checked
                if method == "Base":
                    row["verified"] = False
                else:
                    forget_ok = row["forget_rouge"] < FORGET_ROUGE_THRESHOLD
                    base_ret = row["base_retain_rouge"] if not np.isnan(row["base_retain_rouge"]) else 1.0
                    ret_rouge = row["retain_rouge"]
                    retain_ok = np.isnan(ret_rouge) or ret_rouge >= RETAIN_ROUGE_RATIO * base_ret
                    row["verified"] = bool(forget_ok and retain_ok and row["has_metrics"])

                rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Table 1: Main results
# ---------------------------------------------------------------------------

def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model_short", "method", "concept",
        "forget_rouge", "retain_rouge", "verified",
        "Centroid_Norm", "AUSS_L2",
        "attack_Steering", "attack_Steering_ASR", "attack_ICL", "attack_GCG", "attack_MIA",
    ]
    t1 = df[cols].copy()
    t1 = t1.rename(columns={
        "model_short": "Model",
        "method": "Method",
        "concept": "Concept",
        "forget_rouge": "Forget ROUGE↓",
        "retain_rouge": "Retain ROUGE↑",
        "verified": "✓",
        "Centroid_Norm": "Centroid_Norm↓",
        "AUSS_L2": "AUSS_L2↑",
        "attack_Steering": "Steering AUC↓",
        "attack_Steering_ASR": "Steering ASR↓",
        "attack_ICL": "ICL Acc↓",
        "attack_GCG": "GCG Acc↓",
        "attack_MIA": "MIA AUC↓",
    })
    # Sort: model → concept → method order
    method_order = {m: i for i, m in enumerate(METHODS)}
    t1["_method_ord"] = t1["Method"].map(method_order).fillna(99)
    t1 = t1.sort_values(["Model", "Concept", "_method_ord"]).drop(columns=["_method_ord"])
    return t1


# ---------------------------------------------------------------------------
# Table 2: Spearman correlations
# ---------------------------------------------------------------------------

def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman correlation between each AUSS metric and each attack score.
    Only uses rows where verified=True (unlearning passed verification).
    """
    verified = df[df["verified"] == True].copy()
    logger.info(f"Verified rows for correlation: {len(verified)}")

    if len(verified) < 5:
        logger.warning("Too few verified rows for meaningful correlations. Using all non-base rows.")
        verified = df[df["method"] != "Base"].copy()

    attack_cols = [
        ("Steering",           "attack_Steering"),
        ("Steering_ASR",       "attack_Steering_ASR"),
        ("Steering_retain_wf", "attack_Steering_retain_wf"),
        ("ICL",                "attack_ICL"),
        ("GCG",                "attack_GCG"),
        ("MIA",                "attack_MIA"),
    ]
    results = []
    for metric in AUSS_METRICS:
        row = {"AUSS Metric": metric}
        for atk_key, col in attack_cols:
            x = verified[metric].values
            y = verified[col].values
            # Filter NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            label = ATTACK_METRIC_NAMES[atk_key]
            if mask.sum() < 4:
                row[label] = "n.a."
                row[f"{label}_p"] = float("nan")
                continue
            rho, pval = stats.spearmanr(x[mask], y[mask])
            sig = ""
            if pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            row[label] = f"ρ={rho:.2f}{sig}"
            row[f"{label}_p"] = pval
        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_metric_vs_recovery_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 1: Scatter plots of AUSS_L2 + Centroid_Norm vs. each attack score."""
    verified = df[df["method"] != "Base"].copy()
    metrics_to_plot = ["AUSS_L2", "Centroid_Norm"]
    attacks_to_plot = ATTACK_NAMES

    fig, axes = plt.subplots(
        len(metrics_to_plot),
        len(attacks_to_plot),
        figsize=(4 * len(attacks_to_plot), 3 * len(metrics_to_plot)),
        squeeze=False,
    )

    method_colors = {
        "GradAscent": "#e6194b",
        "DPO":        "#3cb44b",
        "NPO":        "#4363d8",
        "NPO+KL":     "#f58231",
        "RMU":        "#911eb4",
        "WHP":        "#a65628",
    }

    for row_i, metric in enumerate(metrics_to_plot):
        for col_j, attack in enumerate(attacks_to_plot):
            ax = axes[row_i][col_j]
            attack_col = f"attack_{attack}"

            for method, grp in verified.groupby("method"):
                x = grp[metric].values
                y = grp[attack_col].values
                mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() == 0:
                    continue
                ax.scatter(x[mask], y[mask],
                           label=method,
                           color=method_colors.get(method, "gray"),
                           alpha=0.7, s=50, edgecolors="white", linewidth=0.5)

            # Correlation line
            x_all = verified[metric].values
            y_all = verified[attack_col].values
            mask_all = ~(np.isnan(x_all) | np.isnan(y_all))
            if mask_all.sum() >= 4:
                rho, pval = stats.spearmanr(x_all[mask_all], y_all[mask_all])
                sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
                ax.set_title(f"ρ={rho:.2f}{sig}", fontsize=10)
            ax.set_xlabel(metric, fontsize=9)
            ax.set_ylabel(ATTACK_METRIC_NAMES[attack], fontsize=9)
            ax.tick_params(labelsize=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5,
                   fontsize=8, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("AUSS Metrics vs. Attack Recoverability (RQ2)", fontsize=12, y=1.01)
    fig.tight_layout()
    path = out_dir / "metric_vs_recovery_scatter.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def fig_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 2: Heatmap of Spearman ρ between all 8 AUSS metrics and 4 attacks."""
    verified = df[df["verified"] == True].copy()
    if len(verified) < 5:
        verified = df[df["method"] != "Base"].copy()

    rho_matrix = np.zeros((len(AUSS_METRICS), len(ATTACK_NAMES)))
    pval_matrix = np.ones((len(AUSS_METRICS), len(ATTACK_NAMES)))

    for i, metric in enumerate(AUSS_METRICS):
        for j, attack in enumerate(ATTACK_NAMES):
            col = f"attack_{attack}"
            x = verified[metric].values
            y = verified[col].values
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() >= 4:
                rho, pval = stats.spearmanr(x[mask], y[mask])
                rho_matrix[i, j] = rho
                pval_matrix[i, j] = pval

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(len(ATTACK_NAMES)))
    ax.set_xticklabels([ATTACK_METRIC_NAMES[a] for a in ATTACK_NAMES], fontsize=10)
    ax.set_yticks(range(len(AUSS_METRICS)))
    ax.set_yticklabels(AUSS_METRICS, fontsize=9)

    for i in range(len(AUSS_METRICS)):
        for j in range(len(ATTACK_NAMES)):
            rho = rho_matrix[i, j]
            pval = pval_matrix[i, j]
            sig = "**" if pval < 0.01 else ("*" if pval < 0.05 else "")
            color = "white" if abs(rho) > 0.5 else "black"
            ax.text(j, i, f"{rho:.2f}{sig}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("Spearman ρ: AUSS Metrics × Attack Recoverability", fontsize=11)
    fig.tight_layout()
    path = out_dir / "correlation_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


def fig_method_comparison_bar(df: pd.DataFrame, out_dir: Path) -> None:
    """Figure 3: Bar chart comparing unlearning methods on AUSS_L2 and attack recoverability."""
    non_base = df[df["method"] != "Base"]

    metrics_to_plot = ["AUSS_L2", "attack_Steering", "attack_ICL", "attack_MIA"]
    labels = ["AUSS_L2↑\n(fragmentation)", "Steering AUC↓", "ICL Acc↓", "MIA AUC↓"]

    unlearn_methods = [m for m in METHODS if m != "Base"]
    x = np.arange(len(metrics_to_plot))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#a65628"]

    for k, method in enumerate(unlearn_methods):
        grp = non_base[non_base["method"] == method]
        means = []
        sems = []
        for m in metrics_to_plot:
            vals = grp[m].dropna().values
            means.append(vals.mean() if len(vals) > 0 else 0.0)
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

        offset = (k - len(unlearn_methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=method,
                      color=colors[k], alpha=0.8, yerr=sems, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Unlearning Method Comparison: Fragmentation vs. Recoverability", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="AUC=0.5 (chance)")
    fig.tight_layout()
    path = out_dir / "method_comparison_bar.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, t2: pd.DataFrame) -> None:
    total = len(df[df["method"] != "Base"])
    completed = df[(df["method"] != "Base") & df["has_metrics"]].shape[0]
    verified_n = df["verified"].sum()

    print(f"\n{'='*70}")
    print(f"RQ2 Analysis Summary")
    print(f"{'='*70}")
    print(f"Total runs (excluding base):  {total}")
    print(f"Completed (has metrics JSON): {completed}")
    print(f"Verified (forget↓ + retain↑): {verified_n}")
    print(f"\nTable 2 — Spearman Correlations (verified rows only):")
    print(t2.to_string(index=False))
    print(f"\nKey correlations to confirm RQ2:")
    print(f"  AUSS_L2 × Steering AUC: look for ρ > 0.5, p < 0.05")
    print(f"  AUSS_L2 × ICL Acc:      look for ρ > 0.5, p < 0.05")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ2 Analysis")
    parser.add_argument("--min_runs", type=int, default=1, help="Min completed runs to proceed")
    parser.add_argument("--no_figures", action="store_true")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_all_results()
    df = pd.DataFrame(rows)

    completed = df["has_metrics"].sum()
    logger.info(f"Loaded {completed} completed runs out of {len(df)} total")

    if completed < args.min_runs:
        logger.error(f"Only {completed} completed runs — need at least {args.min_runs}. Exiting.")
        sys.exit(1)

    # Table 1
    t1 = build_table1(df)
    t1.to_csv(RESULTS_DIR / "rq2_table1.csv", index=False, float_format="%.3f")
    logger.info(f"Table 1 saved: results_rq2/rq2_table1.csv ({len(t1)} rows)")

    # Table 2
    t2 = build_table2(df)
    t2.to_csv(RESULTS_DIR / "rq2_table2_correlations.csv", index=False)
    logger.info(f"Table 2 saved: results_rq2/rq2_table2_correlations.csv")

    # Summary CSV (committed to git)
    df.to_csv(RESULTS_DIR / "rq2_summary.csv", index=False, float_format="%.4f")
    logger.info(f"Summary saved: results_rq2/rq2_summary.csv ({len(df)} rows)")

    # Figures
    if not args.no_figures:
        try:
            fig_metric_vs_recovery_scatter(df, FIGURES_DIR)
        except Exception as e:
            logger.warning(f"Scatter figure failed: {e}")
        try:
            fig_correlation_heatmap(df, FIGURES_DIR)
        except Exception as e:
            logger.warning(f"Heatmap figure failed: {e}")
        try:
            fig_method_comparison_bar(df, FIGURES_DIR)
        except Exception as e:
            logger.warning(f"Bar figure failed: {e}")

    print_summary(df, t2)

    logger.info("rq2_analyze.py complete.")


if __name__ == "__main__":
    main()
