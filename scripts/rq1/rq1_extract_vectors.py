"""
rq1_extract_vectors.py — Extract peak-layer batch direction vectors for all RQ1 models.

Saves results/{model}__peak_dirs.npz per model:
  hp_dirs   — float32 [30, d]   (10 batches × 3 seeds, HP domain)
  tofu_dirs — float32 [30, d]   (10 batches × 3 seeds, TOFU domain)
  peak_layer — int, the layer index used (argmax Centroid_Norm separation)
  hidden_dim — int, d

Crash-safe: skips any model whose .npz already exists.
Reuses existing src/vectors.py and src/model_utils.py without modification.

Usage:
    conda activate kg-research
    export HF_TOKEN="hf_..."
    python rq1_extract_vectors.py --hf_token $HF_TOKEN
    python rq1_extract_vectors.py --hf_token $HF_TOKEN --models gemma-2b  # single model
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_utils import load_model_and_tokenizer, free_model
from src.vectors import compute_vectors_all_layers, compute_per_batch_directions

# ---------------------------------------------------------------------------
# Config (mirrors dashboard.py constants)
# ---------------------------------------------------------------------------

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

RESULTS_DIR  = Path("experiments/rq1/main")
DATA_DIR     = Path("data")
SEEDS        = [42, 123, 777]
N_BATCHES    = 10
PEAK_METRIC  = "Centroid_Norm"    # use the most discriminative metric to pick layer
RQ1_COHERENCE = {"Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig"}

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rq1_extract_vectors.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pairs(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def find_peak_layer(model_id: str) -> int:
    """Read existing results JSON and return the peak Centroid_Norm layer (seed 42)."""
    p = RESULTS_DIR / (model_id.replace("/", "__") + ".json")
    if not p.exists():
        raise FileNotFoundError(
            f"No existing results for {model_id} at {p}. "
            "Run rq1_extract.py first to compute AUSS metrics."
        )
    with open(p) as f:
        data = json.load(f)

    hp_layers   = data.get("hp_metrics",   {}).get("42", [])
    tofu_layers = data.get("tofu_metrics", {}).get("42", [])

    best_layer, best_sep = 0, -1e9
    coherent = PEAK_METRIC in RQ1_COHERENCE
    for i, (h, t) in enumerate(zip(hp_layers, tofu_layers)):
        hv, tv = h.get(PEAK_METRIC), t.get(PEAK_METRIC)
        if hv is None or tv is None:
            continue
        sep = (hv - tv) if coherent else (tv - hv)
        if sep > best_sep:
            best_sep, best_layer = sep, i

    logger.info(f"  Peak layer for {model_id.split('/')[-1]}: {best_layer} "
                f"(Centroid_Norm sep={best_sep:.4f})")
    return best_layer


def extract_peak_dirs(model_id: str, hp_pairs: list, tofu_pairs: list,
                      hf_token: str, peak_layer: int) -> dict:
    """Load model, extract direction vectors at peak_layer, free model. Returns dict of arrays."""
    logger.info(f"Loading: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, hf_token=hf_token)

    logger.info(f"  Embedding {len(hp_pairs)} HP pairs ...")
    hp_reg_by_layer, hp_anon_by_layer = compute_vectors_all_layers(
        model, tokenizer, hp_pairs, pooling="mean"
    )
    logger.info(f"  Embedding {len(tofu_pairs)} TOFU pairs ...")
    tofu_reg_by_layer, tofu_anon_by_layer = compute_vectors_all_layers(
        model, tokenizer, tofu_pairs, pooling="mean"
    )

    # Free GPU before CPU-heavy direction computation
    free_model(model, tokenizer)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    logger.info("  GPU freed.")

    # Slice peak layer: [N, d]
    hp_reg   = hp_reg_by_layer[peak_layer]
    hp_anon  = hp_anon_by_layer[peak_layer]
    tfu_reg  = tofu_reg_by_layer[peak_layer]
    tfu_anon = tofu_anon_by_layer[peak_layer]
    d = hp_reg.shape[1]

    # Compute direction vectors for each seed: [10, d] per seed → stack to [30, d]
    hp_dirs_all, tofu_dirs_all = [], []
    for seed in SEEDS:
        hp_d = compute_per_batch_directions(hp_reg, hp_anon, n_batches=N_BATCHES, seed=seed)
        td   = compute_per_batch_directions(tfu_reg, tfu_anon, n_batches=N_BATCHES, seed=seed)
        if hp_d is None or td is None:
            logger.warning(f"  Degenerate batch directions for seed={seed}, skipping.")
            continue
        hp_dirs_all.append(hp_d.float().numpy())    # [10, d]
        tofu_dirs_all.append(td.float().numpy())    # [10, d]

    if not hp_dirs_all:
        raise RuntimeError(f"All seeds produced degenerate directions for {model_id}")

    return {
        "hp_dirs":   np.vstack(hp_dirs_all).astype(np.float32),    # [30, d]
        "tofu_dirs": np.vstack(tofu_dirs_all).astype(np.float32),  # [30, d]
        "peak_layer": np.array([peak_layer], dtype=np.int32),
        "hidden_dim": np.array([d], dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--models",   nargs="*", default=[],
                        help="Filter by model name substring (e.g. gemma-2b Llama-2)")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    hp_pairs   = load_pairs(DATA_DIR / "hp_pairs.json")
    tofu_pairs = load_pairs(DATA_DIR / "tofu_pairs.json")
    logger.info(f"Loaded {len(hp_pairs)} HP pairs, {len(tofu_pairs)} TOFU pairs.")

    models = [m for m in RQ1_MODELS
              if not args.models or any(f in m for f in args.models)]
    logger.info(f"Processing {len(models)} model(s).")

    for model_id in models:
        out_path = RESULTS_DIR / (model_id.replace("/", "__") + "__peak_dirs.npz")
        if out_path.exists():
            logger.info(f"[SKIP] {model_id} — {out_path.name} already exists.")
            continue

        logger.info(f"\n{'='*60}\n[START] {model_id}\n{'='*60}")
        try:
            peak_layer = find_peak_layer(model_id)
            arrays = extract_peak_dirs(
                model_id, hp_pairs, tofu_pairs, args.hf_token, peak_layer
            )
            np.savez_compressed(out_path, **arrays)
            logger.info(f"[SAVED] {out_path.name}  "
                        f"hp={arrays['hp_dirs'].shape}  "
                        f"tofu={arrays['tofu_dirs'].shape}")
        except Exception as e:
            logger.error(f"[FAIL] {model_id}: {e}", exc_info=True)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)

    logger.info("\nAll done.")


if __name__ == "__main__":
    main()
