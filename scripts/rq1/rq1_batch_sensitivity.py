#!/usr/bin/env python3
"""
Batch-size sensitivity for AUSS metrics (B=10 justification).

Quantifies how sensitive AUSS discriminability is to the batch-count
hyperparameter B, and how it degrades toward the small-sample regime.

Recomputes HP-vs-TOFU discriminability at B in {1, 2, 5, 10, 20, 50} directly
from the raw per-example hidden states in experiments/rq1/main/*__vectors.npz
(produced by rq1_extract.py) -- no model re-extraction needed. For each B, repeats the random
batch-assignment R times (different seeds) to quantify variance from the
shuffle itself, then runs the same paired HP-vs-TOFU test as rq1_analyze.py
(two-sided Wilcoxon, Cohen's d) across the 9 available models.

Input:  experiments/rq1/main/*__vectors.npz
Output: experiments/rq1/batch_sensitivity/table_batch_sensitivity.csv

Usage:
    python scripts/rq1/rq1_batch_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.vectors import compute_per_batch_directions
from src.metrics import compute_all_metrics

RESULTS_DIR = ROOT / "experiments" / "rq1" / "main"
OUT_DIR = ROOT / "experiments/rq1/batch_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZES = [1, 2, 5, 10, 20, 50]
N_SEEDS = 5
LAYER = 18  # peak layer for 6/8 metrics in the fresh 9-model re-analysis (Centroid_Norm, AUSS_L2, etc.)

METRICS = [
    "Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig",
    "AUSS_L2", "AUSS_Cos2", "AUSS_Jac", "Batch_Cov_Trace", "Batch_Cov_TopEig",
]
COHERENCE = {"Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig"}


def model_id_from_filename(fname: str) -> str:
    # e.g. "google_gemma-2b-it__vectors.npz" or "Qwen__Qwen-14B-Chat__vectors.npz"
    stem = fname.replace("__vectors.npz", "")
    if "__" in stem:
        org, name = stem.split("__", 1)
        return f"{org}/{name}"
    return stem.replace("_", "/", 1)


def metrics_at_batch_size(reg: torch.Tensor, anon: torch.Tensor, n_batches: int, seed: int):
    dirs = compute_per_batch_directions(reg, anon, n_batches=n_batches, seed=seed)
    return compute_all_metrics(dirs), (dirs.shape[0] if dirs is not None else 0)


def main():
    npz_files = sorted(RESULTS_DIR.glob("*__vectors.npz"))
    if not npz_files:
        sys.exit("ERROR: no experiments/rq1/main/*__vectors.npz found.")
    print(f"Found {len(npz_files)} models with raw vectors.")

    # per_model[model_id][B][metric] = {"hp": [...seeds...], "tofu": [...seeds...]}
    rows = []

    for f in npz_files:
        model_id = model_id_from_filename(f.name)
        d = np.load(f)
        hp_reg = torch.from_numpy(d["hp_reg"][LAYER].astype(np.float32))
        hp_anon = torch.from_numpy(d["hp_anon"][LAYER].astype(np.float32))
        tofu_reg = torch.from_numpy(d["tofu_reg"][LAYER].astype(np.float32))
        tofu_anon = torch.from_numpy(d["tofu_anon"][LAYER].astype(np.float32))
        n_layers = d["hp_reg"].shape[0]
        layer = LAYER if LAYER < n_layers else n_layers - 1

        print(f"  {model_id}: layer {layer}/{n_layers}, N_hp={hp_reg.shape[0]}, N_tofu={tofu_reg.shape[0]}")

        for B in BATCH_SIZES:
            for seed in range(N_SEEDS):
                hp_m, hp_b = metrics_at_batch_size(hp_reg, hp_anon, B, seed=100 + seed)
                tofu_m, tofu_b = metrics_at_batch_size(tofu_reg, tofu_anon, B, seed=100 + seed)
                for metric in METRICS:
                    rows.append({
                        "model_id": model_id, "B": B, "seed": seed, "metric": metric,
                        "hp_val": hp_m[metric], "tofu_val": tofu_m[metric],
                        "hp_n_batches_actual": hp_b, "tofu_n_batches_actual": tofu_b,
                    })

    df = pd.DataFrame(rows)
    raw_out = OUT_DIR / "table_batch_sensitivity_raw.csv"
    df.to_csv(raw_out, index=False, float_format="%.5f")
    print(f"\nSaved raw per-(model,B,seed,metric) values: {raw_out} ({len(df)} rows)")

    # Average across seeds per (model, B, metric) first (reduces shuffle noise),
    # then run the paired HP-vs-TOFU test across models, same as rq1_analyze.py.
    agg = df.groupby(["model_id", "B", "metric"], as_index=False)[["hp_val", "tofu_val"]].mean()

    summary_rows = []
    for B in BATCH_SIZES:
        for metric in METRICS:
            sub = agg[(agg["B"] == B) & (agg["metric"] == metric)]
            hp_vals = sub["hp_val"].values
            tofu_vals = sub["tofu_val"].values
            valid = ~(np.isnan(hp_vals) | np.isnan(tofu_vals))
            n_valid = int(valid.sum())
            if n_valid < 3:
                summary_rows.append({
                    "B": B, "metric": metric, "n_models": n_valid,
                    "mean_hp": float("nan"), "mean_tofu": float("nan"),
                    "cohens_d": float("nan"), "p_wilcoxon": float("nan"),
                    "correct_direction": None,
                })
                continue
            hv, tv = hp_vals[valid], tofu_vals[valid]
            diff = hv - tv
            d_eff = diff.mean() / (diff.std(ddof=1) + 1e-12)
            try:
                _, p = stats.wilcoxon(hv, tv, alternative="two-sided")
            except Exception:
                p = float("nan")
            # "correct direction" per the fresh (bug-fixed) re-analysis: HP higher for
            # coherence metrics, HP lower for dispersion metrics.
            if metric in COHERENCE:
                correct_dir = bool(hv.mean() > tv.mean())
            else:
                correct_dir = bool(hv.mean() < tv.mean())
            summary_rows.append({
                "B": B, "metric": metric, "n_models": n_valid,
                "mean_hp": float(hv.mean()), "mean_tofu": float(tv.mean()),
                "cohens_d": float(d_eff), "p_wilcoxon": float(p),
                "correct_direction": correct_dir,
            })

    summary = pd.DataFrame(summary_rows)
    out_csv = OUT_DIR / "table_batch_sensitivity.csv"
    summary.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Saved: {out_csv}")

    print(f"\n{'B':>4} {'metric':30s} {'n':>3} {'d':>7} {'p':>8} {'dir_ok':>7}")
    print("-" * 65)
    for _, r in summary.iterrows():
        d_str = f"{r['cohens_d']:.2f}" if not np.isnan(r["cohens_d"]) else "nan"
        p_str = f"{r['p_wilcoxon']:.4f}" if not np.isnan(r["p_wilcoxon"]) else "nan"
        dir_str = "" if r["correct_direction"] is None else ("Y" if r["correct_direction"] else "N")
        print(f"{r['B']:>4} {r['metric']:30s} {r['n_models']:>3} {d_str:>7} {p_str:>8} {dir_str:>7}")


if __name__ == "__main__":
    main()
