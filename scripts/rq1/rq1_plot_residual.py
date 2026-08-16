"""
rq1_plot_residual.py — Academic-quality PCA / t-SNE / UMAP figures for Residual Structure.

Reads:  results/*__peak_dirs.npz  (raw direction vectors, produced by rq1_extract_vectors.py)
        results/pca_coords.json   (PCA 2D coords, produced by rq1_pca_vectors.py)
Writes: figures/residual_pca_grid.pdf/png    — 2×5 PCA grid
        figures/residual_pca_combined.pdf/png
        figures/residual_tsne_grid.pdf/png   — 2×5 t-SNE grid  (perplexity=10)
        figures/residual_umap_grid.pdf/png   — 2×5 UMAP grid   (n_neighbors=8)

Usage:
    conda run -n kg-research python rq1_plot_residual.py
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})

HP_COLOR   = "#2563eb"   # blue
TOFU_COLOR = "#dc2626"   # red
HP_MARKER  = "o"
TOFU_MARKER = "^"
ALPHA      = 0.75
MSIZE      = 28          # scatter marker size (points²)

MODEL_ORDER = [
    "Qwen-7B-Chat",
    "Qwen-14B-Chat",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "gemma-2b-it",
    "gemma-7b-it",
    "Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf",
    "Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
]

# Short display names for panel titles
DISPLAY = {
    "Qwen-7B-Chat":              "Qwen-7B",
    "Qwen-14B-Chat":             "Qwen-14B",
    "Qwen2.5-7B-Instruct":       "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct":      "Qwen2.5-14B",
    "gemma-2b-it":               "Gemma-2B",
    "gemma-7b-it":               "Gemma-7B",
    "Llama-2-7b-chat-hf":        "Llama-2-7B",
    "Llama-2-13b-chat-hf":       "Llama-2-13B",
    "Meta-Llama-3-8B-Instruct":  "Llama-3-8B",
    "Meta-Llama-3.1-8B-Instruct":"Llama-3.1-8B",
}


def load_coords(path="results/pca_coords.json"):
    with open(path) as f:
        pts = json.load(f)
    by_model = defaultdict(lambda: {"HP": [], "TOFU": []})
    for p in pts:
        by_model[p["model"]][p["domain"]].append((p["x"], p["y"], p["peak_layer"]))
    return by_model


def plot_panel(ax, model, hp_pts, tofu_pts, show_legend=False):
    """Draw one PCA scatter panel."""
    if hp_pts:
        hx, hy = zip(*[(x, y) for x, y, _ in hp_pts])
        ax.scatter(hx, hy, c=HP_COLOR,   marker=HP_MARKER,   s=MSIZE, alpha=ALPHA,
                   linewidths=0.3, edgecolors="white", label="HP (known)", zorder=3)
    if tofu_pts:
        tx, ty = zip(*[(x, y) for x, y, _ in tofu_pts])
        ax.scatter(tx, ty, c=TOFU_COLOR, marker=TOFU_MARKER, s=MSIZE, alpha=ALPHA,
                   linewidths=0.3, edgecolors="white", label="TOFU (unknown)", zorder=3)

    peak = hp_pts[0][2] if hp_pts else "?"
    ax.set_title(f"{DISPLAY.get(model, model)}\n(layer {peak})", pad=4)
    ax.set_xlabel("PC 1", labelpad=2)
    ax.set_ylabel("PC 2", labelpad=2)
    ax.tick_params(length=3)

    # Remove tick numbers — PCA units are not interpretable
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_legend:
        ax.legend(loc="best", framealpha=0.9, handlelength=1.2)


# ── Figure 1: 2×5 grid ────────────────────────────────────────────────────

def make_grid_figure(by_model):
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5.2),
                             constrained_layout=True)
    axes = axes.flatten()

    models_present = [m for m in MODEL_ORDER if m in by_model]

    for i, model in enumerate(models_present):
        hp_pts   = by_model[model]["HP"]
        tofu_pts = by_model[model]["TOFU"]
        plot_panel(axes[i], model, hp_pts, tofu_pts, show_legend=(i == 0))

    # Hide unused panels
    for j in range(len(models_present), len(axes)):
        axes[j].set_visible(False)

    # Shared legend below the grid
    hp_patch   = mpatches.Patch(color=HP_COLOR,   label="HP — Harry Potter (known domain)")
    tofu_patch = mpatches.Patch(color=TOFU_COLOR, label="TOFU — Fictitious authors (unknown domain)")
    fig.legend(handles=[hp_patch, tofu_patch],
               loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04),
               frameon=True, framealpha=0.9,
               fontsize=8.5)

    fig.suptitle(
        "PCA of Residual Direction Vectors: Known (HP) vs Unknown (TOFU) Knowledge\n"
        r"$\mathbf{d}_b = \mathrm{mean}_{\mathrm{batch}}(\mathbf{h}_{\mathrm{reg}} - "
        r"\mathbf{h}_{\mathrm{anon}}) \,/\, \|\cdot\|$  at peak discriminative layer",
        fontsize=10, y=1.02
    )
    return fig


# ── Figure 2: combined single-panel with per-model offsets ─────────────────

def make_combined_figure(by_model):
    """All models overlaid in one panel; each model's PCA cloud offset to a grid cell."""
    models_present = [m for m in MODEL_ORDER if m in by_model]
    ncols = 5
    nrows = (len(models_present) + ncols - 1) // ncols

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_aspect("equal")
    ax.axis("off")

    cell_w, cell_h = 2.2, 2.2

    for idx, model in enumerate(models_present):
        row = idx // ncols
        col = idx % ncols
        ox  = col * cell_w
        oy  = -row * cell_h

        hp_pts   = by_model[model]["HP"]
        tofu_pts = by_model[model]["TOFU"]

        # Normalize within this model to fit in unit square
        all_x = [p[0] for p in hp_pts + tofu_pts]
        all_y = [p[1] for p in hp_pts + tofu_pts]
        rng_x = max(all_x) - min(all_x) or 1
        rng_y = max(all_y) - min(all_y) or 1
        scale = 1.6 / max(rng_x, rng_y)

        def norm(pts):
            return [((p[0] - min(all_x)) * scale + ox,
                     (p[1] - min(all_y)) * scale + oy) for p in pts]

        nhp   = norm(hp_pts)
        ntofu = norm(tofu_pts)

        if nhp:
            ax.scatter(*zip(*nhp),   c=HP_COLOR,   marker=HP_MARKER,   s=18, alpha=ALPHA,
                       linewidths=0, zorder=3)
        if ntofu:
            ax.scatter(*zip(*ntofu), c=TOFU_COLOR, marker=TOFU_MARKER, s=18, alpha=ALPHA,
                       linewidths=0, zorder=3)

        # Panel border
        rect = plt.Rectangle((ox - 0.05, oy - 0.05), 1.7, 1.7,
                              fill=False, edgecolor="#d1d5db", linewidth=0.8, zorder=1)
        ax.add_patch(rect)

        # Panel label
        peak = hp_pts[0][2] if hp_pts else "?"
        ax.text(ox + 0.82, oy + 1.75,
                f"{DISPLAY.get(model, model)} (L{peak})",
                ha="center", va="bottom", fontsize=7, color="#374151")

    # Legend
    hp_h   = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=HP_COLOR,
                        markersize=7, label="HP (known)")
    tofu_h = plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=TOFU_COLOR,
                        markersize=7, label="TOFU (unknown)")
    ax.legend(handles=[hp_h, tofu_h], loc="lower right",
              bbox_to_anchor=(1.0, 0.0), framealpha=0.9, fontsize=8)

    fig.suptitle(
        "Residual Direction Vectors — PCA Projections Across 10 LLMs\n"
        "HP vectors cluster (coherent); TOFU vectors scatter (fragmented)",
        fontsize=10
    )
    fig.tight_layout()
    return fig


# ── Load raw vectors and project with any method ───────────────────────────

SEEDS_LIST  = [42, 123, 777]
RESULTS_DIR = Path("experiments/rq1/main")

def load_raw_vectors():
    """Load peak_dirs.npz for all models. Returns dict: model → {hp, tofu, peak_layer}."""
    out = {}
    for mid in [
        "Qwen/Qwen-7B-Chat", "Qwen/Qwen-14B-Chat",
        "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
        "google/gemma-2b-it", "google/gemma-7b-it",
        "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ]:
        p = RESULTS_DIR / (mid.replace("/", "__") + "__peak_dirs.npz")
        if not p.exists():
            continue
        d = np.load(p)
        short = mid.split("/")[1]
        out[short] = {
            "hp":         d["hp_dirs"].astype(np.float32),    # [30, dim]
            "tofu":       d["tofu_dirs"].astype(np.float32),  # [30, dim]
            "peak_layer": int(d["peak_layer"][0]),
        }
    return out


def project(hp, tofu, method="pca"):
    """Project [30, d] HP and [30, d] TOFU to 2D jointly. Returns (hp_2d, tofu_2d)."""
    X = np.vstack([hp, tofu])   # [60, d]
    n = len(hp)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    elif method == "tsne":
        # perplexity must be < n_samples/3; with 60 points, 10 is safe
        reducer = TSNE(n_components=2, perplexity=10, max_iter=1000,
                       random_state=42, init="pca", learning_rate="auto")
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=8, min_dist=0.3,
                            random_state=42, verbose=False)
    else:
        raise ValueError(method)

    X_2d = reducer.fit_transform(X)
    return X_2d[:n], X_2d[n:]   # hp_2d, tofu_2d


def make_projection_grid(raw, method="tsne"):
    """Generic 2×5 grid figure for any projection method."""
    METHOD_LABEL = {
        "pca":  ("PCA", "PC 1", "PC 2"),
        "tsne": ("t-SNE (perplexity=10)", "t-SNE 1", "t-SNE 2"),
        "umap": ("UMAP (n_neighbors=8)", "UMAP 1", "UMAP 2"),
    }
    title_method, xlabel, ylabel = METHOD_LABEL[method]

    models_present = [m for m in MODEL_ORDER if m in raw]
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5.2), constrained_layout=True)
    axes = axes.flatten()

    for i, model in enumerate(models_present):
        d = raw[model]
        print(f"  [{method}] {model} ...", flush=True)
        hp_2d, tofu_2d = project(d["hp"], d["tofu"], method=method)

        ax = axes[i]
        ax.scatter(hp_2d[:, 0],   hp_2d[:, 1],   c=HP_COLOR,   marker=HP_MARKER,
                   s=MSIZE, alpha=ALPHA, linewidths=0.3, edgecolors="white",
                   label="HP (known)", zorder=3)
        ax.scatter(tofu_2d[:, 0], tofu_2d[:, 1], c=TOFU_COLOR, marker=TOFU_MARKER,
                   s=MSIZE, alpha=ALPHA, linewidths=0.3, edgecolors="white",
                   label="TOFU (unknown)", zorder=3)

        ax.set_title(f"{DISPLAY.get(model, model)}\n(layer {d['peak_layer']})", pad=4)
        ax.set_xlabel(xlabel, labelpad=2)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(length=3)

    for j in range(len(models_present), len(axes)):
        axes[j].set_visible(False)

    hp_patch   = mpatches.Patch(color=HP_COLOR,   label="HP — Harry Potter (known domain)")
    tofu_patch = mpatches.Patch(color=TOFU_COLOR, label="TOFU — Fictitious authors (unknown domain)")
    fig.legend(handles=[hp_patch, tofu_patch], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), frameon=True, framealpha=0.9, fontsize=8.5)

    fig.suptitle(
        f"{title_method} of Residual Direction Vectors: Known (HP) vs Unknown (TOFU) Knowledge\n"
        r"$\mathbf{d}_b = \mathrm{mean}_{\mathrm{batch}}(\mathbf{h}_{\mathrm{reg}} - "
        r"\mathbf{h}_{\mathrm{anon}}) \,/\, \|\cdot\|$  at peak discriminative layer",
        fontsize=10, y=1.02
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    coords_path = Path("experiments/rq1/main/pca_coords.json")
    if not coords_path.exists():
        print("results/pca_coords.json not found. Run rq1_pca_vectors.py first.")
        return

    by_model = load_coords(coords_path)
    raw      = load_raw_vectors()
    out_dir  = Path("figures")
    out_dir.mkdir(exist_ok=True)

    print("Generating PCA grid figure (2×5 panels)...")
    fig1 = make_grid_figure(by_model)
    fig1.savefig(out_dir / "residual_pca_grid.pdf", bbox_inches="tight")
    fig1.savefig(out_dir / "residual_pca_grid.png", bbox_inches="tight", dpi=300)
    plt.close(fig1)
    print("  → figures/residual_pca_grid.pdf + .png")

    print("Generating PCA combined overview figure...")
    fig2 = make_combined_figure(by_model)
    fig2.savefig(out_dir / "residual_pca_combined.pdf", bbox_inches="tight")
    fig2.savefig(out_dir / "residual_pca_combined.png", bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print("  → figures/residual_pca_combined.pdf + .png")

    if raw:
        print("\nGenerating t-SNE grid figure (2×5 panels, perplexity=10)...")
        fig3 = make_projection_grid(raw, method="tsne")
        fig3.savefig(out_dir / "residual_tsne_grid.pdf", bbox_inches="tight")
        fig3.savefig(out_dir / "residual_tsne_grid.png", bbox_inches="tight", dpi=300)
        plt.close(fig3)
        print("  → figures/residual_tsne_grid.pdf + .png")

        print("\nGenerating UMAP grid figure (2×5 panels, n_neighbors=8)...")
        fig4 = make_projection_grid(raw, method="umap")
        fig4.savefig(out_dir / "residual_umap_grid.pdf", bbox_inches="tight")
        fig4.savefig(out_dir / "residual_umap_grid.png", bbox_inches="tight", dpi=300)
        plt.close(fig4)
        print("  → figures/residual_umap_grid.pdf + .png")
    else:
        print("\nSkipping t-SNE/UMAP — no *__peak_dirs.npz files found.")
        print("Run rq1_extract_vectors.py first.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
