#!/usr/bin/env python3
"""
Per-layer metric × ASR Spearman correlation analysis.

For each (layer index, metric, attack) triple, compute Spearman ρ across all
verified runs, revealing which layer best predicts attack recoverability.

Covers two metric families:
  - AUSS / coherence metrics: loaded from *__metrics.json → auss_metrics.all_layers
  - Entropy / AnonDiff metrics: loaded from *__entropy_layers.json (produced by
    rq2_entropy_layer_backfill.py). Rows without entropy data show NaN for those
    columns and are excluded from entropy-metric correlations.

Usage (from project root):
    python scripts/rq2/rq2_layer_analysis.py
    python scripts/rq2/rq2_layer_analysis.py --all_rows  # include non-verified

Output:
    results_rq2/rq2_layer_correlations.csv  — long-format (layer, metric, attack, rho, pval, n)
    figures_rq2/layer_correlation_heatmap.pdf  — heatmap over AUSS metrics
    figures_rq2/layer_correlation_lines.pdf    — line plot over AUSS metrics
    figures_rq2/layer_entropy_heatmap.pdf      — heatmap over entropy/AnonDiff metrics
    figures_rq2/layer_entropy_lines.pdf        — line plot over entropy/AnonDiff metrics
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

# Entropy / AnonDiff metrics from *__entropy_layers.json
ENTROPY_METRICS = [
    "Reg_S_alpha_2",           # Reg Rényi-2 entropy  (↑ = fragmented)
    "Reg_MeanCos",             # Reg mean pairwise cosine (↑ = coherent)
    "Reg_MeanL2",              # Reg mean L2 norm
    "AnonDiff_S_alpha_2",      # AnonDiff Rényi-2 entropy (↑ = fragmented)
    "AnonDiff_MeanCos",        # AnonDiff mean pairwise cosine (↑ = coherent)
    "AnonDiff_MeanL2",         # AnonDiff mean L2 norm (↑ = stronger signal)
    "NormAnonDiff_S_alpha_2",  # Normalised AnonDiff entropy
    "NormAnonDiff_MeanCos",    # Normalised AnonDiff mean cosine
]

ALL_METRICS = AUSS_METRICS + ENTROPY_METRICS

ATTACKS = ["Steering", "ICL", "GCG", "MIA"]

FORGET_THRESH = 0.10
RETAIN_RATIO  = 0.80


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _run_id(model_id: str, method: str, concept: str) -> str:
    return f"{model_id.replace('/', '__')}__{method}__{concept}"


def _is_verified(m: dict, method: str) -> bool:
    if method == "Base":
        return False
    pv = m.get("post_verification", {})
    bv = m.get("base_verification", {})
    fr = pv.get("forget_rouge_l", 1.0) or 1.0
    rr = pv.get("retain_rouge_l")
    br = bv.get("retain_rouge_l")
    if fr >= FORGET_THRESH:
        return False
    if not br:
        return True
    return (rr or 0.0) >= RETAIN_RATIO * br


def load_long_df() -> pd.DataFrame:
    """Build a long-format DataFrame: one row per (run, layer).

    AUSS metrics come from *__metrics.json → auss_metrics.all_layers.
    Entropy metrics come from *__entropy_layers.json (optional; NaN if absent).
    """
    rows = []
    n_runs = 0
    n_with_entropy = 0

    for model_id in MODELS:
        for method in METHODS:
            for concept in CONCEPTS:
                rid     = _run_id(model_id, method, concept)
                m_path  = RESULTS_DIR / f"{rid}__metrics.json"
                a_path  = RESULTS_DIR / f"{rid}__attacks.json"
                e_path  = RESULTS_DIR / f"{rid}__entropy_layers.json"

                if not m_path.exists():
                    continue
                m = json.loads(m_path.read_text())

                all_layers = m.get("auss_metrics", {}).get("all_layers", [])
                if not all_layers:
                    logger.warning(f"No all_layers in {rid}")
                    continue

                # Load entropy layers (optional)
                entropy_layers: list[dict] = []
                if e_path.exists():
                    e = json.loads(e_path.read_text())
                    entropy_layers = e.get("all_layers", [])
                    if len(entropy_layers) != len(all_layers):
                        logger.warning(
                            f"Entropy layer count mismatch {rid}: "
                            f"{len(entropy_layers)} vs {len(all_layers)} — ignoring"
                        )
                        entropy_layers = []
                    else:
                        n_with_entropy += 1

                verified   = _is_verified(m, method)
                peak_layer = m.get("auss_metrics", {}).get("peak_layer")

                # Attack scores
                atk_scores: dict[str, float] = {}
                if a_path.exists():
                    a = json.loads(a_path.read_text())
                    atk = a.get("attacks", {})
                    for attack in ATTACKS:
                        s = atk.get(attack, {}).get("score")
                        atk_scores[attack] = float(s) if s is not None else float("nan")
                else:
                    for attack in ATTACKS:
                        atk_scores[attack] = float("nan")

                for i, lm in enumerate(all_layers):
                    row: dict = {
                        "run_id":    rid,
                        "model":     model_id.split("/")[-1],
                        "method":    method,
                        "concept":   concept,
                        "layer":     i,
                        "is_peak":   (i == peak_layer),
                        "verified":  verified,
                        "has_entropy": len(entropy_layers) > 0,
                    }
                    for k in AUSS_METRICS:
                        v = lm.get(k)
                        row[k] = float(v) if v is not None else float("nan")
                    # Entropy metrics: NaN when entropy_layers not available
                    em = entropy_layers[i] if entropy_layers else {}
                    for k in ENTROPY_METRICS:
                        v = em.get(k)
                        row[k] = float(v) if (v is not None and v == v) else float("nan")
                    for attack in ATTACKS:
                        row[f"attack_{attack}"] = atk_scores[attack]
                    rows.append(row)
                n_runs += 1

    logger.info(
        f"Loaded {n_runs} runs → {len(rows)} (run, layer) rows  "
        f"({n_with_entropy} runs have entropy data)"
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_correlations(df: pd.DataFrame, use_verified: bool = True) -> pd.DataFrame:
    """Compute Spearman ρ for every (layer, metric, attack) triple.

    AUSS metrics use all verified runs.
    Entropy metrics additionally filter to rows where has_entropy=True.
    """
    if use_verified:
        base_data = df[df["verified"]].copy()
        if len(base_data["run_id"].unique()) < 5:
            logger.warning("Fewer than 5 verified runs; falling back to all non-Base rows.")
            base_data = df[df["method"] != "Base"].copy()
    else:
        base_data = df[df["method"] != "Base"].copy()

    n_runs_used = len(base_data["run_id"].unique())
    logger.info(f"Correlation based on {n_runs_used} unique runs (AUSS); "
                f"{len(base_data[base_data['has_entropy']]['run_id'].unique())} with entropy")

    n_layers = int(df["layer"].max()) + 1
    records  = []

    for layer in range(n_layers):
        ld_all    = base_data[base_data["layer"] == layer]
        ld_entropy = ld_all[ld_all["has_entropy"]]

        for metric in ALL_METRICS:
            # Use entropy-filtered subset for entropy metrics
            ld = ld_entropy if metric in ENTROPY_METRICS else ld_all
            for attack in ATTACKS:
                x    = ld[metric].values
                y    = ld[f"attack_{attack}"].values
                mask = ~(np.isnan(x) | np.isnan(y))
                n    = int(mask.sum())
                if n < 4:
                    records.append({
                        "layer": layer, "metric": metric, "attack": attack,
                        "rho": float("nan"), "pval": float("nan"), "n": n,
                        "family": "entropy" if metric in ENTROPY_METRICS else "auss",
                    })
                    continue
                rho, pval = spearmanr(x[mask], y[mask])
                records.append({
                    "layer": layer, "metric": metric, "attack": attack,
                    "rho": float(rho), "pval": float(pval), "n": n,
                    "family": "entropy" if metric in ENTROPY_METRICS else "auss",
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

METRIC_SHORT = {
    # AUSS / coherence
    "Centroid_Norm":              "CN",
    "Global_Coherence":           "GC",
    "Batch_SecondMoment_TopEig":  "SM-Eig",
    "AUSS_L2":                    "L2",
    "AUSS_Cos2":                  "Cos²",
    "AUSS_Jac":                   "Jac",
    "Batch_Cov_Trace":            "Cov-Tr",
    "Batch_Cov_TopEig":           "Cov-Eig",
    # Entropy / AnonDiff
    "Reg_S_alpha_2":              "Reg-S₂",
    "Reg_MeanCos":                "Reg-cos",
    "Reg_MeanL2":                 "Reg-L2",
    "AnonDiff_S_alpha_2":         "AD-S₂",
    "AnonDiff_MeanCos":           "AD-cos",
    "AnonDiff_MeanL2":            "AD-L2",
    "NormAnonDiff_S_alpha_2":     "NAD-S₂",
    "NormAnonDiff_MeanCos":       "NAD-cos",
}


def make_heatmap_figure(
    corr_df: pd.DataFrame, out_path: Path,
    metrics_list: list | None = None,
) -> None:
    """4-panel heatmap: rows=layer, cols=metric, color=ρ, one subplot per attack."""
    if metrics_list is None:
        metrics_list = AUSS_METRICS
    n_layers = int(corr_df["layer"].max()) + 1
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flat

    for ax, attack in zip(axes, ATTACKS):
        sub = corr_df[corr_df["attack"] == attack]
        pivot = sub.pivot(index="layer", columns="metric", values="rho")
        # reorder columns to match metrics_list order
        pivot = pivot.reindex(columns=[m for m in metrics_list if m in pivot.columns])
        z = pivot.values

        im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{attack}", fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([METRIC_SHORT.get(c, c) for c in pivot.columns],
                            rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 8)))
        ax.set_yticklabels([f"L{i}" for i in range(0, n_layers, max(1, n_layers // 8))],
                            fontsize=8)
        ax.set_xlabel("AUSS Metric", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        plt.colorbar(im, ax=ax, label="ρ", shrink=0.85)

        # Annotate best cell
        if not np.all(np.isnan(z)):
            idx = np.nanargmax(np.abs(z))
            row_i, col_i = np.unravel_index(idx, z.shape)
            ax.add_patch(plt.Rectangle((col_i - 0.5, row_i - 0.5), 1, 1,
                                        fill=False, edgecolor="black", linewidth=2))

    fig.suptitle("Per-Layer AUSS × ASR Spearman ρ (black box = |ρ|_max)", fontsize=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def make_line_figure(
    corr_df: pd.DataFrame, long_df: pd.DataFrame, out_path: Path,
    metrics_list: list | None = None,
) -> None:
    """4-panel line plot: x=layer, y=ρ, one line per metric, one subplot per attack."""
    if metrics_list is None:
        metrics_list = AUSS_METRICS
    n_layers = int(corr_df["layer"].max()) + 1

    # Modal peak layer across runs
    peaks = long_df[long_df["is_peak"]]["layer"]
    modal_peak = int(peaks.mode()[0]) if len(peaks) else None

    colors = cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, sharex=True)
    axes = axes.flat

    for ax, attack in zip(axes, ATTACKS):
        sub = corr_df[corr_df["attack"] == attack]
        for ci, metric in enumerate(metrics_list):
            msub = sub[sub["metric"] == metric].sort_values("layer")
            ax.plot(msub["layer"], msub["rho"],
                    label=METRIC_SHORT.get(metric, metric),
                    color=colors[ci % len(colors)], linewidth=1.5)
        ax.axhline(0, color="#999", linewidth=0.8, linestyle="--")
        if modal_peak is not None:
            ax.axvline(modal_peak, color="#333", linewidth=1, linestyle=":",
                       label=f"peak (L{modal_peak})")
        ax.set_title(f"{attack}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Spearman ρ", fontsize=9)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, n_layers - 1)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.set_xlabel("Layer index", fontsize=9)

    fig.suptitle("Spearman ρ by Layer (dashed line = modal peak-AUSS layer)", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all_rows", action="store_true",
                        help="Include non-verified runs (default: verified only)")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(exist_ok=True)

    long_df = load_long_df()
    if long_df.empty:
        logger.error("No data found — check that results_rq2/ contains __metrics.json files.")
        sys.exit(1)

    has_entropy = long_df["has_entropy"].any()
    if not has_entropy:
        logger.warning(
            "No __entropy_layers.json files found — entropy metrics will be NaN.\n"
            "Run: python scripts/rq2/rq2_entropy_layer_backfill.py --hf_token $HF_TOKEN"
        )

    corr_df = compute_correlations(long_df, use_verified=not args.all_rows)

    out_csv = RESULTS_DIR / "rq2_layer_correlations.csv"
    corr_df.to_csv(out_csv, index=False)
    logger.info(f"Saved {out_csv}  ({len(corr_df)} rows)")

    # Summary: best |ρ| per attack, separately for each family
    for family, metrics in [("AUSS", AUSS_METRICS), ("Entropy", ENTROPY_METRICS)]:
        sub = corr_df[(corr_df["family"] == family.lower()) & corr_df["rho"].notna()]
        if sub.empty:
            continue
        logger.info(f"\n=== Best |ρ| per attack — {family} metrics ===")
        for attack in ATTACKS:
            asub = sub[sub["attack"] == attack]
            if asub.empty:
                continue
            best = asub.loc[asub["rho"].abs().idxmax()]
            logger.info(
                f"  {attack:10s}  L{int(best.layer):02d}  "
                f"{best.metric:35s}  ρ={best.rho:+.3f}  p={best.pval:.4f}  n={int(best.n)}"
            )

    # AUSS figures (always generated)
    auss_corr = corr_df[corr_df["family"] == "auss"]
    make_heatmap_figure(auss_corr, FIGURES_DIR / "layer_correlation_heatmap.pdf")
    make_line_figure(auss_corr, long_df, FIGURES_DIR / "layer_correlation_lines.pdf")

    # Entropy figures (only when data is available)
    if has_entropy:
        ent_corr = corr_df[corr_df["family"] == "entropy"]
        make_heatmap_figure(
            ent_corr, FIGURES_DIR / "layer_entropy_heatmap.pdf",
            metrics_list=ENTROPY_METRICS,
        )
        make_line_figure(
            ent_corr, long_df, FIGURES_DIR / "layer_entropy_lines.pdf",
            metrics_list=ENTROPY_METRICS,
        )
        logger.info("Entropy figures saved.")
    else:
        logger.info("Skipping entropy figures (no data yet).")


if __name__ == "__main__":
    main()
