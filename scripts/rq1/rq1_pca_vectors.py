"""
rq1_pca_vectors.py — Project peak-layer direction vectors to 2D via PCA.

Reads:  results/{model}__peak_dirs.npz  (produced by rq1_extract_vectors.py)
Writes: results/pca_coords.json         (600 points: 10 models × 60 vectors each)

Each point in pca_coords.json:
  model      — short model name (e.g. "Llama-2-7b-chat-hf")
  domain     — "HP" or "TOFU"
  x, y       — PCA 2D coordinates
  peak_layer — which layer the vectors came from
  seed       — 42, 123, or 777
  batch      — batch index within that seed (0–9)

PCA is fit jointly on HP + TOFU vectors per model so both domains share the same axes.
Skips models whose .npz is missing and warns.

Usage:
    conda run -n kg-research python rq1_pca_vectors.py
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RQ1_MODELS = [
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

RESULTS_DIR = Path("experiments/rq1/main")
SEEDS = [42, 123, 777]


def main():
    out = []
    n_done = 0

    for model_id in RQ1_MODELS:
        npz_path = RESULTS_DIR / (model_id.replace("/", "__") + "__peak_dirs.npz")
        if not npz_path.exists():
            logger.warning(f"Missing: {npz_path.name} — skipping (run rq1_extract_vectors.py first)")
            continue

        data = np.load(npz_path)
        hp   = data["hp_dirs"].astype(np.float32)    # [30, d]
        tofu = data["tofu_dirs"].astype(np.float32)  # [30, d]
        peak = int(data["peak_layer"][0])
        short = model_id.split("/")[1]
        n_hp, n_tofu = len(hp), len(tofu)
        d = hp.shape[1]
        logger.info(f"{short}: peak_layer={peak}, d={d}, hp={n_hp}, tofu={n_tofu}")

        # Fit PCA jointly on HP + TOFU so axes are shared
        X = np.vstack([hp, tofu])                     # [60, d]
        X_2d = PCA(n_components=2, random_state=42).fit_transform(X)  # [60, 2]

        # Build seed and batch labels
        # 30 vectors: 10 batches per seed × 3 seeds, stacked in order [seed42, seed123, seed777]
        seed_labels  = [s for s in SEEDS for _ in range(10)][:n_hp]
        batch_labels = [b for _ in SEEDS for b in range(10)][:n_hp]

        for i in range(n_hp):
            out.append({
                "model":      short,
                "domain":     "HP",
                "x":          float(X_2d[i, 0]),
                "y":          float(X_2d[i, 1]),
                "peak_layer": peak,
                "seed":       seed_labels[i],
                "batch":      batch_labels[i],
            })
        for i in range(n_tofu):
            out.append({
                "model":      short,
                "domain":     "TOFU",
                "x":          float(X_2d[n_hp + i, 0]),
                "y":          float(X_2d[n_hp + i, 1]),
                "peak_layer": peak,
                "seed":       seed_labels[i],
                "batch":      batch_labels[i],
            })
        n_done += 1

    out_path = RESULTS_DIR / "pca_coords.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Written {len(out)} points for {n_done} models → {out_path}")
    if n_done < len(RQ1_MODELS):
        missing = len(RQ1_MODELS) - n_done
        logger.warning(f"{missing} model(s) skipped — run rq1_extract_vectors.py to complete them.")


if __name__ == "__main__":
    main()
