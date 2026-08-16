#!/usr/bin/env python3
"""
Camera-ready refresh of Table 2 (main text) and Table 5 (appendix): Spearman
correlation between each of the paper's 9 curated metrics (AUSS dispersion:
AUSS_L2, AUSS_Cos2, Centroid_Norm; AnonDiff: MeanCos, S_alpha_2, MeanL2; Reg:
S_alpha_2, MeanCos, MeanL2) and each of the 4 original attacks (Steering,
ICL, GCG, MIA), on the complete RQ2 dataset.

Matches the paper's own stated methodology exactly (RQ2 Setup: correlation
analysis excluding Base and WHP runs, non-base runs with complete attack
data). This is distinct from the two correlation tables produced directly by
rq2_analyze.py:
  - experiments/rq2/main/rq2_table2_correlations.csv       (verified_only=True)
  - experiments/rq2/main/rq2_table2_correlations_all.csv   (all non-Base, including WHP)
Both use a different run population than the paper's declared protocol, and
neither uses the paper's specific 9-metric set (they use the pipeline's
generic 8-metric AUSS_METRICS list instead).

BH correction is applied within each attack column (9 metrics per column),
matching Table 5's caption ("BH-adjusted p<0.01/0.05/0.10 within each attack
column").

Input:  experiments/rq2/main/rq2_summary.csv
Output: experiments/rq2/correlations/table2_5.csv
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

IN_CSV = ROOT / "experiments" / "rq2" / "main" / "rq2_summary.csv"
OUT_DIR = ROOT / "experiments" / "rq2" / "correlations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    ("AUSS_L2", "AUSS dispersion"),
    ("AUSS_Cos2", "AUSS dispersion"),
    ("Centroid_Norm", "AUSS dispersion"),
    ("AnonDiff_MeanCos", "AnonDiff"),
    ("AnonDiff_S_alpha_2", "AnonDiff"),
    ("AnonDiff_MeanL2", "AnonDiff"),
    ("Reg_S_alpha_2", "Reg"),
    ("Reg_MeanCos", "Reg"),
    ("Reg_MeanL2", "Reg"),
]
ATTACKS = [
    ("Steering", "attack_Steering"),
    ("ICL", "attack_ICL"),
    ("GCG", "attack_GCG"),
    ("MIA", "attack_MIA"),
]


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


def main():
    df = pd.read_csv(IN_CSV)
    pop = df[~df["method"].isin(["Base", "WHP"])].copy()
    print(f"Population: non-Base, non-WHP rows = {len(pop)}")

    rows = []
    for attack_label, col in ATTACKS:
        pvals = []
        cell_rows = []
        for metric, group in METRICS:
            x = pop[metric].values
            y = pop[col].values
            mask = ~(np.isnan(x) | np.isnan(y))
            n = int(mask.sum())
            if n < 4:
                cell_rows.append({"Metric": metric, "Group": group, "Attack": attack_label, "rho": float("nan"), "p": float("nan"), "n": n})
                pvals.append(1.0)
                continue
            rho, pval = stats.spearmanr(x[mask], y[mask])
            cell_rows.append({"Metric": metric, "Group": group, "Attack": attack_label, "rho": rho, "p": pval, "n": n})
            pvals.append(pval)
        qvals = bh_correct(pvals)
        for cr, q in zip(cell_rows, qvals):
            cr["q_bh_per_attack"] = q
            cr["sig"] = "**" if q < 0.01 else ("*" if q < 0.05 else ("†" if q < 0.10 else ""))
        rows.extend(cell_rows)

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "table2_5_camera_ready.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    print("\n=== Camera-ready Table 2/5 (n = {} non-Base non-WHP rows) ===".format(len(pop)))
    for attack_label, _ in ATTACKS:
        sub = out[out["Attack"] == attack_label].sort_values("q_bh_per_attack")
        n_sig = (sub["q_bh_per_attack"] < 0.05).sum()
        print(f"\n{attack_label} (n_sig BH<0.05: {n_sig}/9):")
        for _, r in sub.iterrows():
            print(f"  {r['Metric']:22s} rho={r['rho']:+.3f}  p={r['p']:.4f}  q={r['q_bh_per_attack']:.4f}  {r['sig']}  (n={r['n']})")


if __name__ == "__main__":
    main()
