#!/usr/bin/env python3
"""
Classifier-based discriminability for the RQ1 AUSS metrics (HP vs. TOFU).

Effect-size gaps (Cohen's d) depend on the scale of the underlying metric, so
comparing confidence intervals across metrics isn't a scale-invariant measure
of discriminability on its own. This directly evaluates whether a simple
classifier can distinguish known and unknown concepts, independent of any
metric's raw scale.

This trains a logistic regression and a small MLP on the 8-dim AUSS peak-layer
feature vector per model, one sample per (model, domain) pair, under
leave-one-model-out CV so both domains of a held-out model are scored by a
classifier that never saw that model. This is scale-invariant (features are
standardized) and doesn't require agreeing on which direction "coherent"
points for each metric -- the classifier just finds whatever hyperplane
separates the classes.

Input:  experiments/rq1/main/summary_fresh.csv (direction-verified,
        9 models x 8 metrics x {hp, tofu})
Output: experiments/rq1/classifier/table_classifier_discriminability.csv

Usage:
    python scripts/rq1/rq1_classifier_discriminability.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments/rq1/classifier"
IN_CSV = ROOT / "experiments" / "rq1" / "main" / "summary_fresh.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig",
    "AUSS_L2", "AUSS_Cos2", "AUSS_Jac", "Batch_Cov_Trace", "Batch_Cov_TopEig",
]


def build_dataset(df: pd.DataFrame):
    """One row per (model, domain): 8-dim feature vector + label (1=HP, 0=TOFU) + group=model."""
    X, y, groups, model_ids = [], [], [], []
    for _, r in df.iterrows():
        hp_feat = [r[f"{m}_hp"] for m in METRICS]
        tofu_feat = [r[f"{m}_tofu"] for m in METRICS]
        if any(pd.isna(v) for v in hp_feat) or any(pd.isna(v) for v in tofu_feat):
            continue
        X.append(hp_feat); y.append(1); groups.append(r["model_id"]); model_ids.append(r["model_id"])
        X.append(tofu_feat); y.append(0); groups.append(r["model_id"]); model_ids.append(r["model_id"])
    return np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups)


def logo_cv(X, y, groups, make_model, seed=0):
    """Leave-one-model-out CV. Returns per-fold (group, y_true, y_pred, y_score)."""
    logo = LeaveOneGroupOut()
    rows = []
    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        ytr, yte = y[train_idx], y[test_idx]
        clf = make_model(seed)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        try:
            score = clf.predict_proba(Xte)[:, 1]
        except Exception:
            score = pred.astype(float)
        held_out_group = groups[test_idx][0]
        for i in range(len(test_idx)):
            rows.append({
                "held_out_model": held_out_group,
                "y_true": int(yte[i]), "y_pred": int(pred[i]), "y_score": float(score[i]),
            })
    return pd.DataFrame(rows)


def summarize(fold_df: pd.DataFrame, name: str) -> dict:
    acc = accuracy_score(fold_df["y_true"], fold_df["y_pred"])
    try:
        auc = roc_auc_score(fold_df["y_true"], fold_df["y_score"])
    except ValueError:
        auc = float("nan")
    n_correct = int((fold_df["y_true"] == fold_df["y_pred"]).sum())
    n = len(fold_df)
    n_models_perfect = fold_df.groupby("held_out_model")[["y_true", "y_pred"]].apply(
        lambda g: (g["y_true"] == g["y_pred"]).all()
    ).sum()
    n_models = fold_df["held_out_model"].nunique()
    return {
        "classifier": name, "n_samples": n, "accuracy": acc, "auc": auc,
        "n_correct": n_correct, "n_models_both_correct": int(n_models_perfect),
        "n_models": int(n_models),
    }


def main():
    if not IN_CSV.exists():
        sys.exit(f"ERROR: {IN_CSV} not found.")
    df = pd.read_csv(IN_CSV)
    X, y, groups = build_dataset(df)
    n_models = len(set(groups))
    print(f"Dataset: {len(X)} samples ({n_models} models x 2 domains), {X.shape[1]} features.")

    results = []
    fold_dfs = {}

    logreg_folds = logo_cv(X, y, groups, lambda seed: LogisticRegression(max_iter=2000, C=1.0))
    fold_dfs["LogisticRegression"] = logreg_folds
    results.append(summarize(logreg_folds, "LogisticRegression"))

    mlp_folds = logo_cv(
        X, y, groups,
        lambda seed: MLPClassifier(hidden_layer_sizes=(8,), max_iter=3000, random_state=seed, alpha=1e-2),
    )
    fold_dfs["MLP(8)"] = mlp_folds
    results.append(summarize(mlp_folds, "MLP(8)"))

    res_df = pd.DataFrame(results)
    out_csv = OUT_DIR / "table_classifier_discriminability.csv"
    res_df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")
    print(res_df.to_string(index=False))

    # Per-model breakdown for the logistic regression (simplest, least overfitting risk given n=20)
    print("\nPer-held-out-model results (LogisticRegression, leave-one-model-out):")
    lr = fold_dfs["LogisticRegression"]
    per_model = lr.groupby("held_out_model")[["y_true", "y_pred"]].apply(
        lambda g: pd.Series({
            "n_correct": int((g["y_true"] == g["y_pred"]).sum()),
            "n": len(g),
        })
    )
    print(per_model.to_string())
    out_per_model = OUT_DIR / "table_classifier_per_model.csv"
    per_model.to_csv(out_per_model)
    print(f"Saved: {out_per_model}")


if __name__ == "__main__":
    main()
