#!/usr/bin/env python3
"""
Generate Table 7 (full per-run RQ2 results) as 3 per-model LaTeX table*
environments, splitting a single 54-row table (an overflow risk at \tiny)
into 3 tables of 21 rows each at \small.

Input:  experiments/rq2/main/rq2_summary.csv
Output: experiments/rq2/main/table7_latex_llama2.tex
        experiments/rq2/main/table7_latex_llama3.tex
        experiments/rq2/main/table7_latex_qwen.tex
(paste-ready \begin{table*}...\end{table*} blocks)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "experiments" / "rq2" / "main" / "rq2_summary.csv"
OUT_DIR = ROOT / "experiments" / "rq2" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("meta-llama/Llama-2-7b-chat-hf", "llama2", "LLaMA-2-7B-chat", "tab:rq2_main_llama2"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "llama3", "Meta-LLaMA-3-8B-Instruct", "tab:rq2_main_llama3"),
    ("Qwen/Qwen2.5-7B-Instruct", "qwen", "Qwen2.5-7B-Instruct", "tab:rq2_main_qwen"),
]
METHODS = ["Base", "GradAscent", "DPO", "NPO", "NPO+KL", "RMU", "WHP"]
METHOD_LABEL = {"GradAscent": "GA"}
CONCEPTS = [("harry_potter", "HP"), ("star_wars", "SW"), ("william_shakespeare", "WS")]


def fmt(v, decimals=3, signed=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if signed:
        sign = r"$-$" if v < 0 else "+"
        return f"{sign}{abs(v):.{decimals}f}"
    return f"{v:.{decimals}f}"


def main():
    df = pd.read_csv(IN_CSV)

    for model_id, slug, label, tab_label in MODELS:
        sub = df[df["model_id"] == model_id]
        lines = []
        lines.append(r"\begin{table*}[h]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\setlength{\tabcolsep}{4pt}")
        lines.append(r"\caption{")
        lines.append(rf"    \textbf{{RQ2: Full post-unlearning results, {label}}}.")
        lines.append(r"    ROUGE-L$\downarrow$: forget-set ROUGE-L after unlearning.")
        lines.append(r"    $\checkmark$: verified (ROUGE-L~$<0.10$ \& retain~$\geq 0.80\times$ base).")
        lines.append(r"    $\|\bar{\Delta\hat{v}}\|{\uparrow}$: centroid norm; AUSS$_\text{L2}{\downarrow}$, AUSS$_{\cos^2}{\downarrow}$: dispersion (per-method peak layer $\ell^*$).")
        lines.append(r"    Steer-AUC$\downarrow$: steering AUC; Steer-WF$\downarrow$: mean steered word-frequency score.")
        lines.append(r"    ICL~$\Delta{\uparrow}$: ICL accuracy minus baseline.")
        lines.append(r"    MIA$\downarrow$: membership inference AUC; GCG$\downarrow$: GCG ASR.")
        lines.append(r"    Blank cells: checkpoint unavailable.")
        lines.append(r"}")
        lines.append(rf"\label{{{tab_label}}}")
        lines.append(r"\begin{tabular}{|ll|c||c||ccc||ccccc|}")
        lines.append(r"\hline")
        lines.append(r" & & & Forget & \multicolumn{3}{c||}{Geometry $\ell^*$} & \multicolumn{5}{c|}{Attack Success $\downarrow$} \\")
        lines.append(r"\cline{4-12}")
        lines.append(r"Method & Cpt & $\checkmark$ & ROUGE-L$\downarrow$ & $\|\bar{\Delta\hat{v}}\|{\uparrow}$ & AUSS$_\text{L2}{\downarrow}$ & AUSS$_{\cos^2}{\downarrow}$ & Steer-AUC & Steer-WF & ICL~$\Delta$ & MIA & GCG \\")
        lines.append(r"\hline\hline")

        for method in METHODS:
            mlabel = METHOD_LABEL.get(method, method)
            for i, (concept, clabel) in enumerate(CONCEPTS):
                row = sub[(sub["method"] == method) & (sub["concept"] == concept)]
                prefix = f" {mlabel}" if i == 0 else "        "
                if row.empty or not bool(row.iloc[0]["has_attacks"]):
                    lines.append(rf"{prefix} & {clabel} & \multicolumn{{10}}{{c|}}{{---}} \\")
                    continue
                r = row.iloc[0]
                check = r"\checkmark" if bool(r["verified"]) else ""
                cells = [
                    fmt(r["forget_rouge"]),
                    fmt(r["Centroid_Norm"]),
                    fmt(r["AUSS_L2"]),
                    fmt(r["AUSS_Cos2"]),
                    fmt(r["attack_Steering"]),
                    fmt(r["attack_Steering_ASR"]),
                    fmt(r["attack_ICL"], signed=True),
                    fmt(r["attack_MIA"]),
                    fmt(r["attack_GCG"]),
                ]
                if any(c is None for c in cells[:6] + cells[7:]):
                    # geometry or core attack data missing -> treat as unavailable row
                    lines.append(rf"{prefix} & {clabel} & \multicolumn{{10}}{{c|}}{{---}} \\")
                    continue
                icl_cell = cells[6] if cells[6] is not None else "---"
                lines.append(
                    rf"{prefix} & {clabel} & {check} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} & "
                    rf"{cells[4]} & {cells[5]} & {icl_cell} & {cells[7]} & {cells[8]} \\"
                )
            lines.append(r"  \hline")

        lines.append(r"\end{tabular}")
        lines.append(r"\end{table*}")

        out_path = OUT_DIR / f"table7_latex_{slug}.tex"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"Wrote {out_path} ({len(sub)} rows in source data)")


if __name__ == "__main__":
    main()
