#!/usr/bin/env python3
"""
RQ1 — Full four-way comparison (HP-forget / HP-retain / SW / TOFU).

Combines the direction-verified HP-forget/TOFU extraction
(experiments/rq1/main/summary_fresh.csv, 9 models) with the Star Wars and
HP-retain extractions (experiments/rq1/main/summary_sw.csv,
experiments/rq1/main/*_hp_retain.json) into a single four-way comparison.

This script is a drop-in variant of rq1_full_comparison.py that swaps only the
HP-forget/TOFU source and writes to experiments/rq1/multidomain/ (never overwrites the
original committed results/full_comparison.csv or results/full_discriminability.csv).

Loads:
  - experiments/rq1/main/summary_fresh.csv           → HP-forget and TOFU
  - results/summary_sw.csv                         → SW (unchanged, already correct)
  - experiments/rq1/main/*_hp_retain.json            → HP-retain

Produces:
  - experiments/rq1/multidomain/table_full_comparison_fresh.csv
  - experiments/rq1/multidomain/table_full_discriminability_fresh.csv
  - figures_rq1/rq1_full_comparison_fresh.pdf
"""

import json
import glob
import sys
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    "Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig",
    "AUSS_L2", "AUSS_Cos2", "AUSS_Jac", "Batch_Cov_Trace", "Batch_Cov_TopEig",
]
COHERENCE = {"Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig"}
OUT_DIR = ROOT / "experiments/rq1/multidomain"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def peak_from_layer_metrics(metrics_by_seed, metric):
    vals = []
    for seed_data in metrics_by_seed.values():
        layer_vals = [l[metric] for l in seed_data if metric in l and not np.isnan(l[metric])]
        if layer_vals:
            vals.append(max(layer_vals) if metric in COHERENCE else min(layer_vals))
    return float(np.mean(vals)) if vals else float("nan")


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


print("Loading fresh (fixed-direction) HP-forget + TOFU...")
summary = pd.read_csv(ROOT / "experiments" / "rq1" / "main" / "summary_fresh.csv")
hp_rows, tofu_rows = [], []
for _, r in summary.iterrows():
    rec_hp = {"model_id": r["model_id"]}
    rec_tofu = {"model_id": r["model_id"]}
    for m in METRICS:
        rec_hp[m] = r.get(f"{m}_hp", float("nan"))
        rec_tofu[m] = r.get(f"{m}_tofu", float("nan"))
    hp_rows.append(rec_hp)
    tofu_rows.append(rec_tofu)

df_hp = pd.DataFrame(hp_rows).set_index("model_id")
df_tofu = pd.DataFrame(tofu_rows).set_index("model_id")
print(f"  HP-forget: {len(df_hp)} models, TOFU: {len(df_tofu)} models")

print("Loading SW from summary_sw.csv (unchanged, already correct direction)...")
sw_summary = pd.read_csv(ROOT / "experiments" / "rq1" / "main" / "summary_sw.csv")
sw_rows = []
for _, r in sw_summary.iterrows():
    rec = {"model_id": r["model_id"]}
    for m in METRICS:
        rec[m] = r.get(f"{m}_sw", float("nan"))
    sw_rows.append(rec)
df_sw = pd.DataFrame(sw_rows).set_index("model_id")
print(f"  SW: {len(df_sw)} models")

print("Loading HP-retain from *_hp_retain.json (unchanged, already correct direction)...")
retain_files = sorted(glob.glob(str(ROOT / "experiments" / "rq1" / "main" / "*_hp_retain.json")))
retain_rows = []
for f in retain_files:
    d = json.load(open(f))
    rec = {"model_id": d["model_id"]}
    for m in METRICS:
        rec[m] = peak_from_layer_metrics(d["metrics"], m) if d.get("metrics") else float("nan")
    retain_rows.append(rec)
df_retain = pd.DataFrame(retain_rows).set_index("model_id")
print(f"  HP-retain: {len(df_retain)} models")

common = df_hp.index.intersection(df_tofu.index).intersection(df_sw.index).intersection(df_retain.index)
print(f"\nCommon models across all 4 conditions: {len(common)}")
for m in sorted(common):
    print(f"  {m}")

df_hp = df_hp.loc[common]
df_tofu = df_tofu.loc[common]
df_sw = df_sw.loc[common]
df_retain = df_retain.loc[common]
N = len(common)

rows = []
for model in common:
    rec = {"model_id": model}
    for m in METRICS:
        rec[f"{m}_hp_forget"] = df_hp.loc[model, m]
        rec[f"{m}_hp_retain"] = df_retain.loc[model, m]
        rec[f"{m}_sw"]        = df_sw.loc[model, m]
        rec[f"{m}_tofu"]      = df_tofu.loc[model, m]
    rows.append(rec)
df_full = pd.DataFrame(rows)
df_full.to_csv(OUT_DIR / "table_full_comparison_fresh.csv", index=False)
print(f"\nSaved {OUT_DIR / 'table_full_comparison_fresh.csv'}")

print(f"\n{'Metric':35s} {'HP-forget':>10} {'HP-retain':>10} {'SW':>10} {'TOFU':>10}")
print("-" * 75)
for m in METRICS:
    means = [df_hp[m].mean(), df_retain[m].mean(), df_sw[m].mean(), df_tofu[m].mean()]
    print(f"{m:35s} " + "  ".join(f"{v:8.4f}" for v in means))

PAIRS = [
    ("HP-forget", df_hp,     "TOFU", df_tofu,  "hp_forget_vs_tofu"),
    ("HP-retain", df_retain, "TOFU", df_tofu,  "hp_retain_vs_tofu"),
    ("SW",        df_sw,     "TOFU", df_tofu,  "sw_vs_tofu"),
    ("HP-forget", df_hp,     "SW",   df_sw,    "hp_forget_vs_sw"),
    ("HP-retain", df_retain, "SW",   df_sw,    "hp_retain_vs_sw"),
    ("HP-forget", df_hp,  "HP-retain", df_retain, "hp_forget_vs_retain"),
]

all_rows = []
all_pvals = []
for (label_a, dfa, label_b, dfb, pair_key) in PAIRS:
    for m in METRICS:
        a = dfa[m].values.astype(float)
        b = dfb[m].values.astype(float)
        valid = ~(np.isnan(a) | np.isnan(b))
        av, bv = a[valid], b[valid]
        if len(av) < 3:
            p = 1.0; d = float("nan"); gap = float("nan"); lo = hi = float("nan")
        else:
            try:
                _, p = stats.wilcoxon(av, bv, alternative="two-sided")
            except Exception:
                p = 1.0
            diff = av - bv
            d = diff.mean() / (diff.std(ddof=1) + 1e-12)
            gap = np.mean(np.abs(av - bv))
            lo, hi = bootstrap_ci(av, bv)
        all_rows.append({
            "pair": pair_key, "label_a": label_a, "label_b": label_b,
            "metric": m, "mean_a": float(np.nanmean(a)), "mean_b": float(np.nanmean(b)),
            "gap": gap, "ci_lo": lo, "ci_hi": hi, "d": d, "p_raw": p, "n": int(valid.sum()),
        })
        all_pvals.append(p)

qs = bh_correct(all_pvals)
for row, q in zip(all_rows, qs):
    row["q_adj"] = q
    row["sig"] = "**" if q < 0.01 else ("*" if q < 0.05 else "")

disc_df = pd.DataFrame(all_rows)
disc_df.to_csv(OUT_DIR / "table_full_discriminability_fresh.csv", index=False)
print(f"Saved {OUT_DIR / 'table_full_discriminability_fresh.csv'}")

print(f"\n\n{'Pair':35s} {'Sig (q<.05)':>12} {'Correct dir':>12}")
print("-" * 62)
for (label_a, dfa, label_b, dfb, pair_key) in PAIRS:
    sub = disc_df[disc_df["pair"] == pair_key]
    n_sig = (sub["q_adj"] < 0.05).sum()
    n_correct = 0
    for _, r in sub.iterrows():
        if r["metric"] in COHERENCE:
            n_correct += int(r["mean_a"] > r["mean_b"])
        else:
            n_correct += int(r["mean_a"] < r["mean_b"])
    print(f"{label_a+' vs '+label_b:35s} {n_sig:>4}/{len(sub)} metrics  {n_correct:>4}/{len(sub)} correct dir")

fig_dir = ROOT / "figures_rq1"
fig_dir.mkdir(exist_ok=True)
conditions = ["HP-forget", "HP-retain", "SW", "TOFU"]
dfs = [df_hp, df_retain, df_sw, df_tofu]
colors = ["#1565C0", "#42A5F5", "#FF9800", "#E53935"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for ax, m in zip(axes, METRICS):
    means = [df[m].mean() for df in dfs]
    sems  = [df[m].std() / np.sqrt(len(df)) for df in dfs]
    x = np.arange(len(conditions))
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.85, capsize=4, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=7, rotation=20, ha="right")
    ax.set_title(m.replace("_", " "), fontsize=8, fontweight="bold")
fig.suptitle(f"RQ1: Four-condition comparison, fresh HP/TOFU (n={N} models)", fontsize=10)
plt.tight_layout()
fig_path = fig_dir / "rq1_full_comparison_fresh.pdf"
plt.savefig(fig_path, bbox_inches="tight")
print(f"\nSaved {fig_path}")

if __name__ == "__main__":
    pass
