#!/usr/bin/env python3
"""
RQ1 — HP vs TOFU discriminability on the direction-verified 9-model extraction,
with BH/FDR correction across all 8 metrics.

summary_fresh.csv (9 models) reflects a verified sign convention for all 8
metrics. This script runs the paper's discriminability test (paired two-sided
Wilcoxon + Cohen's d) on that data and applies BH/FDR correction across the
8 metrics, producing Table 1's statistics.

Input:  experiments/rq1/main/summary_fresh.csv  (9 models, direction-verified)
Output: experiments/rq1/main/table_fresh_hp_tofu_discriminability.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

FRESH_CSV = Path("experiments/rq1/main/summary_fresh.csv")
OUT_DIR = ROOT / "experiments/rq1/main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig",
    "AUSS_L2", "AUSS_Cos2", "AUSS_Jac", "Batch_Cov_Trace", "Batch_Cov_TopEig",
]
COHERENCE = {"Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig"}


def bh_correct(pvals):
    pvals = np.array(pvals, dtype=float)
    if HAS_STATSMODELS:
        return multipletests(pvals, method="fdr_bh")[1]
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    q = pvals * n / ranks
    q_adj = np.minimum.accumulate(q[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = q_adj
    return np.minimum(result, 1.0)


def bootstrap_ci(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    gaps = np.mean(np.abs(a[idx] - b[idx]), axis=1)
    return tuple(np.percentile(gaps, [2.5, 97.5]))


def main():
    if not FRESH_CSV.exists():
        sys.exit(f"ERROR: {FRESH_CSV} not found.")
    df = pd.read_csv(FRESH_CSV)
    n_models = len(df)
    print(f"Loaded fresh re-extraction: {n_models} models")

    rows = []
    for metric in METRICS:
        hp = df[f"{metric}_hp"].values.astype(float)
        tofu = df[f"{metric}_tofu"].values.astype(float)
        diff = hp - tofu
        d_eff = diff.mean() / (diff.std(ddof=1) + 1e-12)
        try:
            _, p = stats.wilcoxon(hp, tofu, alternative="two-sided")
        except Exception:
            p = float("nan")
        ci_lo, ci_hi = bootstrap_ci(hp, tofu)
        correct_dir = bool(hp.mean() > tofu.mean()) if metric in COHERENCE else bool(hp.mean() < tofu.mean())
        rows.append({
            "metric": metric,
            "mean_hp": hp.mean(), "mean_tofu": tofu.mean(),
            "gap": float(np.mean(np.abs(diff))),
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "cohens_d": d_eff, "p_raw": p, "n": n_models,
            "correct_direction": correct_dir,
        })

    result = pd.DataFrame(rows)
    result["q_adj"] = bh_correct(result["p_raw"].values)
    result["sig"] = result["q_adj"].apply(lambda q: "*" if q < 0.05 else "")

    out_csv = OUT_DIR / "table_fresh_hp_tofu_discriminability.csv"
    result.to_csv(out_csv, index=False, float_format="%.6g")
    print(f"\nSaved: {out_csv}\n")

    print(f"{'metric':30s} {'mean_hp':>9} {'mean_tofu':>9} {'d':>7} {'p_raw':>9} {'q_adj':>9} {'dir_ok':>6}")
    print("-" * 90)
    for _, r in result.iterrows():
        dir_str = "Y" if r["correct_direction"] else "N"
        print(f"{r['metric']:30s} {r['mean_hp']:>9.4f} {r['mean_tofu']:>9.4f} "
              f"{r['cohens_d']:>7.2f} {r['p_raw']:>9.4g} {r['q_adj']:>9.4g} {dir_str:>6}{r['sig']}")

    n_correct_dir = result["correct_direction"].sum()
    n_sig = (result["q_adj"] < 0.05).sum()
    print(f"\n{n_correct_dir}/8 metrics in the hypothesized direction "
          f"(vs. the unverified experiments/rq1/main/summary.csv).")
    print(f"{n_sig}/8 metrics survive BH/FDR correction at q<0.05.")


if __name__ == "__main__":
    main()
