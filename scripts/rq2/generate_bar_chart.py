#!/usr/bin/env python3
"""
Generate rq2_bar_chart.pdf — 2×2 panel of attack scores by method and model.
No figure title (added by LaTeX caption). Legend at bottom.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_CSV = Path("experiments/rq2/main/rq2_summary.csv")
OUT_PATHS = [
    Path("figures/rq2_bar_chart.pdf"),
]

METHOD_ORDER   = ["GradAscent", "DPO", "NPO", "NPO+KL", "RMU", "WHP"]
METHOD_LABELS  = {"GradAscent": "GA", "DPO": "DPO", "NPO": "NPO",
                  "NPO+KL": "NPO+KL", "RMU": "RMU", "WHP": "WHP"}

MODEL_ORDER    = ["Llama-2-7b-chat-hf", "Meta-Llama-3-8B-Instruct", "Qwen2.5-7B-Instruct"]
MODEL_LABELS   = {"Llama-2-7b-chat-hf":       "LLaMA-2-7B",
                  "Meta-Llama-3-8B-Instruct":  "LLaMA-3-8B",
                  "Qwen2.5-7B-Instruct":       "Qwen2.5-7B"}
MODEL_COLORS   = {"Llama-2-7b-chat-hf":       "#2166ac",   # blue
                  "Meta-Llama-3-8B-Instruct":  "#4dac26",   # green
                  "Qwen2.5-7B-Instruct":       "#d6604d"}   # orange-red

ATTACKS = [
    ("attack_Steering", "Steering AUC",  0.5, True,  False),
    ("attack_ICL",      "ICL Δ",         0.0, False, True),
    ("attack_GCG",      "GCG Acc",       None, False, False),
    ("attack_MIA",      "MIA AUC",       0.5, True,  False),
]

BAR_WIDTH  = 0.22
BAR_GAP    = 0.04   # gap between model-bars within a method group
GROUP_GAP  = 0.55   # x-spacing per method group


def load_data():
    df = pd.read_csv(RESULTS_CSV)
    return df[df["method"] != "Base"].copy()


def compute_stats(df, atk_col):
    """Return {method: {model: (mean, std, n)}} over concepts."""
    result = {}
    for method in METHOD_ORDER:
        result[method] = {}
        for model in MODEL_ORDER:
            vals = df.loc[
                (df["method"] == method) & (df["model_short"] == model),
                atk_col
            ].dropna().values
            if len(vals) > 0:
                result[method][model] = (float(vals.mean()), float(vals.std(ddof=0) if len(vals)>1 else 0.0), len(vals))
            else:
                result[method][model] = None
    return result


def compute_bar_extent(stats):
    """Min/max of (mean - std, mean + std) across every bar in a panel."""
    los, his = [], []
    for method_stats in stats.values():
        for stat in method_stats.values():
            if stat is None:
                continue
            mean, std, _ = stat
            los.append(mean - std)
            his.append(mean + std)
    return min(los), max(his)


def draw_panel(ax, stats, ylabel, chance, chance_is_dashed, ymin=None, ymax=None):
    n_methods = len(METHOD_ORDER)
    n_models  = len(MODEL_ORDER)
    group_positions = np.arange(n_methods) * GROUP_GAP

    for mi, model in enumerate(MODEL_ORDER):
        color = MODEL_COLORS[model]
        offset = (mi - n_models / 2 + 0.5) * (BAR_WIDTH + BAR_GAP)
        xs, ys, errs = [], [], []
        for gi, method in enumerate(METHOD_ORDER):
            stat = stats[method].get(model)
            if stat is None:
                continue
            mean, std, n = stat
            xs.append(group_positions[gi] + offset)
            ys.append(mean)
            errs.append(std)

        ax.bar(xs, ys, width=BAR_WIDTH,
               color=color, alpha=0.88,
               linewidth=0.5, edgecolor="white",
               zorder=3)
        ax.errorbar(xs, ys, yerr=errs,
                    fmt="none", ecolor="#444",
                    elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=4)

    # Chance line
    group_half_width = (n_models * BAR_WIDTH + (n_models - 1) * BAR_GAP) / 2
    margin = group_half_width * 1.15  # clear the outermost bars plus a little padding
    x_lo = group_positions[0] - margin
    x_hi = group_positions[-1] + margin
    if chance is not None:
        ls = "--" if chance_is_dashed else ":"
        ax.axhline(chance, color="#888", linewidth=0.9, linestyle=ls, zorder=2, label="_nolegend_")

    # Axes
    ax.set_xticks(group_positions)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=7.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=4)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.tick_params(axis="x", length=0)

    # y-axis range
    if ymin is not None or ymax is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(ymin if ymin is not None else lo,
                    ymax if ymax is not None else hi)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.grid(axis="y", linewidth=0.4, linestyle=":", color="#ccc", zorder=1)


def main():
    df = load_data()

    plt.rcParams.update({
        "font.family":     "serif",
        "font.serif":      ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize":  8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi":      300,
        "pdf.fonttype":    42,
    })

    fig, axes = plt.subplots(
        2, 2, figsize=(7.0, 4.2),
        gridspec_kw={"hspace": 0.55, "wspace": 0.38}
    )
    axes = axes.flatten()

    for i, (col, label, chance, dashed, auto_scale) in enumerate(ATTACKS):
        stats = compute_stats(df, col)
        if auto_scale:
            lo, hi = compute_bar_extent(stats)
            if chance is not None:
                lo, hi = min(lo, chance), max(hi, chance)
            pad = (hi - lo) * 0.18
            ymin, ymax = lo - pad, hi + pad
        else:
            ymin, ymax = 0.0, 1.12
        draw_panel(axes[i], stats, label, chance, dashed, ymin=ymin, ymax=ymax)

        # Chance-line annotation (small text inside panel)
        if chance is not None:
            axes[i].text(
                0.01, chance + 0.03, "chance",
                transform=axes[i].get_yaxis_transform(),
                fontsize=6.5, color="#888", va="bottom"
            )

    # Shared legend — one patch per model family, placed below the figure
    legend_handles = [
        mpatches.Patch(facecolor=MODEL_COLORS[m], edgecolor="white",
                       linewidth=0.5, label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=1.2,
        handleheight=0.9,
        columnspacing=1.2,
    )

    fig.subplots_adjust(bottom=0.16)

    for out in OUT_PATHS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
