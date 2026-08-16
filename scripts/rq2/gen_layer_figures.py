#!/usr/bin/env python3
"""
Camera-ready regeneration of the paper's two custom layer-sweep figures
(Figures/layer_rho_gcg.pdf, main text; Figures/layer_rho_all_attacks.pdf,
appendix), which have no generating script in this repo (they were
hand-built for the original submission) and are stale relative to the
the layer-sweep BH-correction data.

Input:  experiments/rq2/layer_sweep/table6_layer_sweep_bh.csv
Output: figures_rq2/layer_rho_gcg_camera_ready.pdf
        figures_rq2/layer_rho_all_attacks_camera_ready.pdf
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "experiments" / "rq2" / "layer_sweep" / "table6_layer_sweep_bh.csv"
OUT_DIR = ROOT / "figures_rq2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUSS_METRICS = ["AUSS_L2", "AUSS_Cos2", "AUSS_Jac"]
ENTROPY_METRICS = ["Reg_S_alpha_2", "AnonDiff_MeanCos", "NormAnonDiff_S_alpha_2"]
METRIC_LABELS = {
    "AUSS_L2": r"AUSS$_\mathrm{L2}$",
    "AUSS_Cos2": r"AUSS$_{\cos^2}$",
    "AUSS_Jac": r"AUSS$_\mathrm{Jac}$",
    "Reg_S_alpha_2": r"Reg $S_2$",
    "AnonDiff_MeanCos": r"AnonDiff $\overline{\cos}$",
    "NormAnonDiff_S_alpha_2": r"NormAnonDiff $S_2$",
}
AUSS_COLORS = ["#1b9e77", "#d95f02", "#7570b3"]
ENTROPY_COLORS = ["#1b9e77", "#d95f02", "#7570b3"]


def best_point(df, attack, metrics):
    sub = df[(df["attack"] == attack) & (df["metric"].isin(metrics))]
    row = sub.loc[sub["rho"].abs().idxmax()]
    return row


def fig_gcg(df):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sub = df[df["attack"] == "GCG"]
    for metric, color in zip(AUSS_METRICS, AUSS_COLORS):
        s = sub[sub["metric"] == metric].sort_values("layer")
        ax.plot(s["layer"], s["rho"], label=METRIC_LABELS[metric], color=color, linewidth=1.6)
    peak = best_point(df, "GCG", AUSS_METRICS)
    ax.scatter([peak["layer"]], [peak["rho"]], color="black", zorder=5, s=30)
    ax.annotate(
        f"L{int(peak['layer']):02d} ({METRIC_LABELS[peak['metric']]}, "
        f"$\\rho$={peak['rho']:.3f})",
        xy=(peak["layer"], peak["rho"]),
        xytext=(0.5, 0.08), textcoords="axes fraction",
        fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ent_peak = best_point(df, "GCG", ENTROPY_METRICS)
    ax.set_title(
        f"Strongest single predictor overall: {METRIC_LABELS[ent_peak['metric']]} "
        f"at L{int(ent_peak['layer']):02d} ($\\rho$={ent_peak['rho']:.3f}, entropy family)",
        fontsize=7.5,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Spearman $\rho$")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = OUT_DIR / "layer_rho_gcg_camera_ready.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}  (peak: L{int(peak['layer'])} {peak['metric']} rho={peak['rho']:.4f}; "
          f"entropy best: L{int(ent_peak['layer'])} {ent_peak['metric']} rho={ent_peak['rho']:.4f})")


def fig_all_attacks(df):
    attacks = ["GCG", "ICL", "MIA", "Steering"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True)
    peaks = {}
    for ax, attack in zip(axes.flat, attacks):
        sub = df[df["attack"] == attack]
        for metric, color in zip(AUSS_METRICS, AUSS_COLORS):
            s = sub[sub["metric"] == metric].sort_values("layer")
            ax.plot(s["layer"], s["rho"], color=color, linewidth=1.4, linestyle="-",
                     label=METRIC_LABELS[metric])
        for metric, color in zip(ENTROPY_METRICS, ENTROPY_COLORS):
            s = sub[sub["metric"] == metric].sort_values("layer")
            ax.plot(s["layer"], s["rho"], color=color, linewidth=1.2, linestyle="--",
                     label=METRIC_LABELS[metric])
        peak = best_point(df, attack, AUSS_METRICS + ENTROPY_METRICS)
        peaks[attack] = peak
        ax.axvline(peak["layer"], color="red", linewidth=1.0, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(f"{attack} (peak L{int(peak['layer']):02d}, $\\rho$={peak['rho']:.3f})", fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"Spearman $\rho$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT_DIR / "layer_rho_all_attacks_camera_ready.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    for atk, p in peaks.items():
        print(f"  {atk}: L{int(p['layer']):02d} {p['metric']} rho={p['rho']:.4f} p={p['pval']:.2e}")


def main():
    df = pd.read_csv(IN_CSV)
    fig_gcg(df)
    fig_all_attacks(df)


if __name__ == "__main__":
    main()
