#!/usr/bin/env python3
"""
RQ1 — Trivial-baseline comparison.

Compares AUSS against the simplest possible baseline one could construct
from the same raw data, to confirm the anon-vector construction is doing
real work rather than any raw-similarity probe being sufficient on its own.

Baseline: mean pairwise cosine similarity of the RAW regular ("reg", i.e.
non-anonymized) hidden states at the same peak layer used for AUSS -- no
anon-vector subtraction, no batching, just "how similar are this domain's
per-example representations to each other." This is the simplest possible
coherence probe one could construct from the same raw data.

Compares, on the fresh 9-model direction-fixed re-extraction:
  1. Discriminability (paired two-sided Wilcoxon + Cohen's d, BH n/a since
     it's a single test) of the baseline metric, HP vs TOFU.
  2. Leave-one-model-out classifier AUC using ONLY the 1-dim baseline feature,
     compared directly against the existing 8-dim AUSS classifier
     (experiments/rq1/main/table_classifier_discriminability.csv, AUC=0.92).

Input:  experiments/rq1/main/*__vectors.npz (9 models)
Output: experiments/rq1/main/table_baseline_comparison.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "rq1" / "main"
OUT_DIR = ROOT / "experiments/rq1/main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER = 18  # peak layer used throughout the direction-verified re-analysis


def model_id_from_filename(fname: str) -> str:
    stem = fname.replace("__vectors.npz", "")
    if "__" in stem:
        org, name = stem.split("__", 1)
        return f"{org}/{name}"
    return stem.replace("_", "/", 1)


def mean_pairwise_cosine(x: torch.Tensor) -> float:
    """Mean pairwise cosine similarity across all example pairs (excl. diagonal)."""
    xn = torch.nn.functional.normalize(x, dim=-1)
    sim = xn @ xn.T
    n = sim.shape[0]
    off_diag_sum = sim.sum() - torch.diagonal(sim).sum()
    return float(off_diag_sum / (n * (n - 1)))


def main():
    npz_files = sorted(RESULTS_DIR.glob("*__vectors.npz"))
    if not npz_files:
        sys.exit("ERROR: no experiments/rq1/main/*__vectors.npz found.")

    rows = []
    for f in npz_files:
        model_id = model_id_from_filename(f.name)
        d = np.load(f)
        n_layers = d["hp_reg"].shape[0]
        layer = LAYER if LAYER < n_layers else n_layers - 1
        hp_reg = torch.from_numpy(d["hp_reg"][layer].astype(np.float32))
        tofu_reg = torch.from_numpy(d["tofu_reg"][layer].astype(np.float32))
        hp_base = mean_pairwise_cosine(hp_reg)
        tofu_base = mean_pairwise_cosine(tofu_reg)
        rows.append({"model_id": model_id, "baseline_hp": hp_base, "baseline_tofu": tofu_base})
        print(f"  {model_id}: baseline_hp={hp_base:.4f} baseline_tofu={tofu_base:.4f}")

    df = pd.DataFrame(rows)

    hp = df["baseline_hp"].values
    tofu = df["baseline_tofu"].values
    diff = hp - tofu
    d_eff = diff.mean() / (diff.std(ddof=1) + 1e-12)
    _, p = stats.wilcoxon(hp, tofu, alternative="two-sided")
    n_correct_dir = int((hp > tofu).sum())  # coherent domain (HP) expected higher raw cosine

    print(f"\nBaseline discriminability (raw-cosine, no anon-vector subtraction):")
    print(f"  mean_hp={hp.mean():.4f}  mean_tofu={tofu.mean():.4f}  d={d_eff:.2f}  p={p:.4g}  "
          f"{n_correct_dir}/{len(df)} models in coherent-direction")

    # Leave-one-model-out classifier AUC using only this 1-dim feature
    X, y, groups = [], [], []
    for _, r in df.iterrows():
        X.append([r["baseline_hp"]]); y.append(1); groups.append(r["model_id"])
        X.append([r["baseline_tofu"]]); y.append(0); groups.append(r["model_id"])
    X, y, groups = np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups)

    logo = LeaveOneGroupOut()
    fold_rows = []
    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xtr, y[train_idx])
        pred = clf.predict(Xte)
        score = clf.predict_proba(Xte)[:, 1]
        for i in range(len(test_idx)):
            fold_rows.append({"y_true": int(y[test_idx][i]), "y_pred": int(pred[i]), "y_score": float(score[i])})
    fold_df = pd.DataFrame(fold_rows)
    acc = accuracy_score(fold_df["y_true"], fold_df["y_pred"])
    try:
        auc = roc_auc_score(fold_df["y_true"], fold_df["y_score"])
    except ValueError:
        auc = float("nan")

    print(f"\nBaseline (1-dim) LOMO-CV classifier: accuracy={acc:.4f}  AUC={auc:.4f}")
    print("Compare to AUSS (8-dim) LOMO-CV classifier: accuracy=0.95  AUC=0.92 "
          "(experiments/rq1/main/table_classifier_discriminability.csv)")

    summary = pd.DataFrame([{
        "n_models": len(df),
        "mean_hp": hp.mean(), "mean_tofu": tofu.mean(),
        "cohens_d": d_eff, "p_wilcoxon": p,
        "n_correct_direction": n_correct_dir,
        "classifier_accuracy": acc, "classifier_auc": auc,
        "auss_classifier_accuracy_ref": 0.95, "auss_classifier_auc_ref": 0.92,
    }])
    out_csv = OUT_DIR / "table_baseline_comparison.csv"
    summary.to_csv(out_csv, index=False, float_format="%.4f")
    df.to_csv(OUT_DIR / "table_baseline_comparison_per_model.csv", index=False, float_format="%.6f")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
