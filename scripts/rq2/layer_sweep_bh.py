#!/usr/bin/env python3
"""
BH/FDR-correct the full layer-resolved AUSS/entropy x ASR Spearman sweep.

The layer-resolved sweep tests ~700 (layer, metric) hypotheses at only n=10-12
per test; without correction, extreme correlations are expected by chance, so
this applies BH/FDR correction at three scopes.

Input:  experiments/rq2/main/rq2_layer_correlations.csv (produced by
        scripts/rq2/rq2_layer_analysis.py, one row per (layer, metric, attack)
        triple, raw Spearman rho/pval, NO correction)
Output: experiments/rq2/layer_sweep/table6_layer_sweep_bh.csv -- same rows + q-values under
        three BH scopes:
          - q_per_attack   : BH within each attack column (n_tests = n_layers * n_metrics)
          - q_per_family   : BH within each (attack, family) block (auss vs entropy separately)
          - q_global       : BH across every row with a valid p-value (most conservative)

Usage:
    python scripts/rq2/layer_sweep_bh.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "experiments" / "rq2" / "main" / "rq2_layer_correlations.csv"
OUT_DIR = ROOT / "experiments" / "rq2" / "layer_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns q-values (adjusted p-values)."""
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    q = pvals * n / ranks
    q_adj = np.minimum.accumulate(q[order][::-1])[::-1]
    result = np.empty(n)
    result[order] = q_adj
    return np.minimum(result, 1.0)


def bh_within_groups(df: pd.DataFrame, group_cols: list, out_col: str) -> pd.Series:
    """Apply BH correction independently within each group defined by group_cols."""
    q = pd.Series(np.nan, index=df.index)
    for _, idx in df.groupby(group_cols).groups.items():
        sub = df.loc[idx, "pval"]
        valid = sub.notna()
        if valid.sum() == 0:
            continue
        q.loc[sub[valid].index] = bh_correct(sub[valid].values)
    return q


def sig_label(q: float) -> str:
    if pd.isna(q):
        return ""
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def main():
    if not IN_CSV.exists():
        sys.exit(f"ERROR: {IN_CSV} not found. Run scripts/rq2/rq2_layer_analysis.py first.")

    df = pd.read_csv(IN_CSV)
    n_total = len(df)
    n_valid = df["pval"].notna().sum()
    print(f"Loaded {n_total} (layer, metric, attack) rows ({n_valid} with a valid p-value).")

    n_sig_raw = (df["pval"] < 0.05).sum()
    print(f"Raw p<0.05: {n_sig_raw}/{n_valid} ({100*n_sig_raw/n_valid:.1f}%)")

    # Scope 1: BH within each attack column (n_tests = n_layers * n_metrics per attack)
    df["q_per_attack"] = bh_within_groups(df, ["attack"], "q_per_attack")

    # Scope 2: BH within each (attack, family) block -- auss and entropy metrics scored separately
    df["q_per_family"] = bh_within_groups(df, ["attack", "family"], "q_per_family")

    # Scope 3: global BH across every valid test in the whole sweep (most conservative)
    valid = df["pval"].notna()
    q_global = np.full(n_total, np.nan)
    q_global[valid.values] = bh_correct(df.loc[valid, "pval"].values)
    df["q_global"] = q_global

    for col in ["q_per_attack", "q_per_family", "q_global"]:
        df[f"sig_{col.replace('q_', '')}"] = df[col].apply(sig_label)

    out_csv = OUT_DIR / "table6_layer_sweep_bh.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved: {out_csv}")

    print(f"\n{'Scope':30s} {'n_tests':>10} {'sig q<0.05':>12} {'sig q<0.01':>12}")
    print("-" * 66)
    for label, col in [
        ("raw p<0.05 (uncorrected)", None),
        ("BH per-attack column", "q_per_attack"),
        ("BH per (attack, family)", "q_per_family"),
        ("BH global (all valid tests)", "q_global"),
    ]:
        if col is None:
            n_sig05 = (df["pval"] < 0.05).sum()
            n_sig01 = (df["pval"] < 0.01).sum()
            n_tests = n_valid
        else:
            n_sig05 = (df[col] < 0.05).sum()
            n_sig01 = (df[col] < 0.01).sum()
            n_tests = n_valid
        print(f"{label:30s} {n_tests:>10d} {n_sig05:>12d} {n_sig01:>12d}")

    # Best surviving triple per attack, under the global (most conservative) scope
    print("\nBest surviving (layer, metric) per attack after global BH correction:")
    for attack in sorted(df["attack"].unique()):
        sub = df[(df["attack"] == attack) & df["q_global"].notna()]
        sig = sub[sub["q_global"] < 0.05]
        if sig.empty:
            best = sub.loc[sub["pval"].idxmin()] if len(sub) else None
            if best is not None:
                print(f"  {attack:10s}  NONE survive q<0.05  "
                      f"(closest: L{int(best.layer):02d} {best.metric:30s} "
                      f"rho={best.rho:+.3f} p={best.pval:.4f} q_global={best.q_global:.4f})")
            else:
                print(f"  {attack:10s}  no valid tests")
        else:
            best = sig.loc[sig["rho"].abs().idxmax()]
            print(f"  {attack:10s}  L{int(best.layer):02d}  {best.metric:30s}  "
                  f"rho={best.rho:+.3f}  q_global={best.q_global:.4f}")


if __name__ == "__main__":
    main()
