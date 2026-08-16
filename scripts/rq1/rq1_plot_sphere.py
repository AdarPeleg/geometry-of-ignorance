"""
rq1_plot_sphere.py — Project peak-layer direction vectors to S² and save static figures.

Pipeline per model:
  1. Load results/*__peak_dirs.npz → hp_dirs [30,d], tofu_dirs [30,d]
  2. Joint PCA(n_components=3) on [60,d] → [60,3]
  3. L2-normalize each row → coordinates on S²
  4. Plot on unit sphere wireframe

Geometric justification:
  The vectors d_b = mean_batch(h_reg − h_anon)/||·|| are unit vectors on S^(d-1).
  AUSS metrics measure angular dispersion of this spherical distribution.
  PCA(3) → L2-normalize is the correct projection to S²: it finds the 3 highest-variance
  directions, then renormalizes so all points remain on the sphere surface.
  HP vectors (coherent) cluster at one pole; TOFU (fragmented) scatter uniformly.

Output:
  figures/residual_sphere_grid.pdf/png  — 2×5 grid, one sphere per model

Usage:
    conda run -n kg-research python rq1_plot_sphere.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     9,
    "axes.titlesize": 8,
    "figure.dpi":    150,
})

HP_COLOR   = "#2563eb"
TOFU_COLOR = "#dc2626"

MODEL_IDS = [
    "Qwen/Qwen-7B-Chat",
    "Qwen/Qwen-14B-Chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "google/gemma-2b-it",
    "google/gemma-7b-it",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

DISPLAY = {
    "Qwen-7B-Chat":               "Qwen-7B",
    "Qwen-14B-Chat":              "Qwen-14B",
    "Qwen2.5-7B-Instruct":        "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct":       "Qwen2.5-14B",
    "gemma-2b-it":                "Gemma-2B",
    "gemma-7b-it":                "Gemma-7B",
    "Llama-2-7b-chat-hf":         "Llama-2-7B",
    "Llama-2-13b-chat-hf":        "Llama-2-13B",
    "Meta-Llama-3-8B-Instruct":   "Llama-3-8B",
    "Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}

RESULTS_DIR = Path("experiments/rq1/main")


def project_to_sphere(hp: np.ndarray, tofu: np.ndarray):
    """Joint PCA(3) on [hp; tofu] then L2-normalize each row → S²."""
    X  = np.vstack([hp, tofu])
    X3 = PCA(n_components=3, random_state=42).fit_transform(X)
    norms = np.linalg.norm(X3, axis=1, keepdims=True)
    X3 /= np.maximum(norms, 1e-8)
    n = len(hp)
    return X3[:n], X3[n:]   # hp_s2 [30,3], tofu_s2 [30,3]


def add_wireframe(ax, n_lat=7, n_lon=12, alpha=0.10):
    u = np.linspace(0, 2 * np.pi, 100)
    for lat in np.linspace(-np.pi / 2, np.pi / 2, n_lat):
        ax.plot(np.cos(lat) * np.cos(u),
                np.cos(lat) * np.sin(u),
                np.sin(lat) * np.ones_like(u),
                color="#9ca3af", alpha=alpha, lw=0.4)
    v = np.linspace(-np.pi / 2, np.pi / 2, 100)
    for lon in np.linspace(0, 2 * np.pi, n_lon, endpoint=False):
        ax.plot(np.cos(lon) * np.cos(v),
                np.sin(lon) * np.cos(v),
                np.sin(v),
                color="#9ca3af", alpha=alpha, lw=0.4)


def sphere_panel(ax, hp_s2, tofu_s2, title):
    add_wireframe(ax)
    ax.scatter(*hp_s2.T, c=HP_COLOR, s=24, alpha=0.85,
               linewidths=0.3, edgecolors="white", marker="o", zorder=3)
    ax.scatter(*tofu_s2.T, c=TOFU_COLOR, s=24, alpha=0.85,
               linewidths=0.3, edgecolors="white", marker="^", zorder=3)
    ax.set_title(title, pad=2)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    # Make panes transparent (keep structure so bbox is non-empty)
    ax.xaxis.pane.fill = False; ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.fill = False; ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.fill = False; ax.zaxis.pane.set_edgecolor("none")
    ax.grid(False)


def main():
    raw = {}
    for mid in MODEL_IDS:
        p = RESULTS_DIR / (mid.replace("/", "__") + "__peak_dirs.npz")
        if p.exists():
            d     = np.load(p)
            short = mid.split("/")[1]
            raw[short] = {
                "hp":         d["hp_dirs"].astype(np.float32),
                "tofu":       d["tofu_dirs"].astype(np.float32),
                "peak_layer": int(d["peak_layer"][0]),
            }

    if not raw:
        print("No *__peak_dirs.npz files found — run rq1_extract_vectors.py first.")
        sys.exit(1)

    ordered = [mid.split("/")[1] for mid in MODEL_IDS if mid.split("/")[1] in raw]
    n       = len(ordered)
    ncols   = 5
    nrows   = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(14, 5.5 * nrows))

    for i, model in enumerate(ordered):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        d  = raw[model]
        hp_s2, tofu_s2 = project_to_sphere(d["hp"], d["tofu"])
        sphere_panel(ax, hp_s2, tofu_s2,
                     f"{DISPLAY.get(model, model)}\n(layer {d['peak_layer']})")
        print(f"  {model}")

    hp_p   = mpatches.Patch(color=HP_COLOR,
                             label="HP — Harry Potter (known domain)")
    tofu_p = mpatches.Patch(color=TOFU_COLOR,
                             label="TOFU — Fictitious authors (unknown domain)")
    fig.legend(handles=[hp_p, tofu_p], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=9.5, framealpha=0.9)

    fig.suptitle(
        r"Unit-Normalized Direction Vectors on $S^2$: Known (HP) vs Unknown (TOFU)"
        "\n"
        r"$\mathbf{d}_b = \mathrm{mean}_\mathrm{batch}(\mathbf{h}_\mathrm{reg}"
        r" - \mathbf{h}_\mathrm{anon}) / \|\cdot\|$"
        r"  projected via PCA(3D) $\to$ L2-normalize."
        "\nHP clusters at a pole (coherent); TOFU scatters uniformly (fragmented).",
        fontsize=10, y=1.01,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.06,
                        wspace=0.0, hspace=0.05)

    out = Path("figures")
    out.mkdir(exist_ok=True)
    fig.savefig(out / "residual_sphere_grid.pdf", dpi=300)
    fig.savefig(out / "residual_sphere_grid.png", dpi=300)
    plt.close(fig)
    print("→ figures/residual_sphere_grid.pdf + .png")
    print("All done.")


if __name__ == "__main__":
    main()
