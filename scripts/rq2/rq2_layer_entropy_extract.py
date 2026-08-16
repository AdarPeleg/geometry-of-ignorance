#!/usr/bin/env python3
"""
rq2_layer_entropy_extract.py

Compute entropy and AnonDiff metrics at EVERY transformer layer for all RQ2 runs.

For each *__metrics.json in experiments/rq2/main/ (produced by rq2_pipeline.py):
  - Load the matching checkpoint (or base model for Base runs)
  - Call compute_vectors_all_layers() on forget_pairs (~50-80 pairs)
  - Call compute_entropy_metrics(reg[l], anon[l]) for every layer l
  - Save experiments/rq2/main/{run_id}__entropy_layers.json

Output schema per file:
  {
    "model_id": ...,
    "method": ...,
    "concept": ...,
    "n_layers": 33,
    "all_layers": [
      {
        "layer": 0,
        "Reg_S_alpha_2": ...,   "Reg_S_alpha_1": ...,   "Reg_EffRank": ...,
        "Reg_MeanL2": ...,      "Reg_MeanCos": ...,
        "AnonDiff_S_alpha_2":..,"AnonDiff_S_alpha_1":...,"AnonDiff_EffRank":...,
        "AnonDiff_MeanL2":...,  "AnonDiff_MeanCos":...,
        "NormAnonDiff_S_alpha_2":...,"NormAnonDiff_MeanCos":...
      },
      ...
    ]
  }

Crash-safe: skips any run whose __entropy_layers.json already exists.

Usage (from project root):
    python scripts/rq2/rq2_layer_entropy_extract.py --hf_token $HF_TOKEN
"""

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_utils import load_model_and_tokenizer, free_model
from src.vectors import compute_vectors_all_layers
from src.entropy import compute_entropy_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("experiments/rq2/main")
MODELS_DIR  = RESULTS_DIR / "models"
DATA_DIR    = Path("data/concepts")

MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
METHODS  = ["Base", "GradAscent", "DPO", "NPO", "NPO+KL", "RMU", "WHP"]
CONCEPTS = ["harry_potter", "star_wars", "william_shakespeare"]

# Entropy keys to keep in each layer record (drop NormAnonDiff_MeanL2: always 1.0)
KEEP_KEYS = [
    "Reg_S_alpha_2", "Reg_S_alpha_1", "Reg_EffRank",
    "Reg_MeanL2",    "Reg_MeanCos",
    "AnonDiff_S_alpha_2", "AnonDiff_S_alpha_1", "AnonDiff_EffRank",
    "AnonDiff_MeanL2",    "AnonDiff_MeanCos",
    "NormAnonDiff_S_alpha_2", "NormAnonDiff_S_alpha_1", "NormAnonDiff_EffRank",
    "NormAnonDiff_MeanCos",
]


def safe_id(model_id: str, method: str, concept: str) -> str:
    return f"{model_id.replace('/', '__')}__{method}__{concept}"


def atomic_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def process_one(model_id: str, method: str, concept: str, hf_token: str) -> str:
    rid    = safe_id(model_id, method, concept)
    m_path = RESULTS_DIR / f"{rid}__metrics.json"
    o_path = RESULTS_DIR / f"{rid}__entropy_layers.json"

    if not m_path.exists():
        logger.info(f"  SKIP — no metrics JSON: {rid}")
        return "skip"
    if o_path.exists():
        logger.info(f"  SKIP — already done: {rid}")
        return "skip"

    # Decide where to load the model from
    ckpt = MODELS_DIR / rid
    if method == "Base" or not (ckpt.exists() and any(ckpt.iterdir())):
        if method != "Base":
            logger.warning(f"  No checkpoint — loading base model instead")
        load_from = model_id
        token = hf_token
    else:
        load_from = str(ckpt)
        token = None

    logger.info(f"  Loading: {load_from}")
    model, tokenizer = load_model_and_tokenizer(load_from, hf_token=token)

    try:
        concept_path = DATA_DIR / f"{concept}.json"
        if not concept_path.exists():
            logger.warning(f"  No concept data: {concept_path}")
            return "skip"
        with open(concept_path) as f:
            concept_data = json.load(f)
        pairs = concept_data.get("forget_pairs") or concept_data.get("qa_eval_pairs", [])
        if not pairs:
            logger.warning(f"  No pairs for {concept}")
            return "skip"

        logger.info(f"  Extracting all-layer vectors for {len(pairs)} pairs…")
        reg_by_layer, anon_by_layer = compute_vectors_all_layers(model, tokenizer, pairs)
        n_layers = len(reg_by_layer)
        logger.info(f"  Computing entropy at {n_layers} layers…")

        all_layers_out = []
        for layer_idx in range(n_layers):
            em = compute_entropy_metrics(reg_by_layer[layer_idx], anon_by_layer[layer_idx])
            row = {"layer": layer_idx}
            for k in KEEP_KEYS:
                v = em.get(k)
                row[k] = float(v) if (v is not None and v == v) else None
            all_layers_out.append(row)

        # Sanity log at the middle layer
        mid = n_layers // 2
        em_mid = all_layers_out[mid]
        logger.info(
            f"  L{mid}: Reg_S2={em_mid.get('Reg_S_alpha_2') or float('nan'):.4f}  "
            f"AD_S2={em_mid.get('AnonDiff_S_alpha_2') or float('nan'):.4f}  "
            f"Reg_cos={em_mid.get('Reg_MeanCos') or float('nan'):.4f}  "
            f"AD_cos={em_mid.get('AnonDiff_MeanCos') or float('nan'):.4f}"
        )

        result = {
            "model_id":   model_id,
            "method":     method,
            "concept":    concept,
            "n_layers":   n_layers,
            "all_layers": all_layers_out,
            "timestamp":  datetime.now().isoformat(),
        }
        atomic_write(o_path, result)
        logger.info(f"  Saved → {o_path.name}")
        return "done"

    finally:
        free_model(model, tokenizer)
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument(
        "--subset",
        default=None,
        help="model_id:method:concept — run only this triple (for testing)",
    )
    args = parser.parse_args()

    if args.subset:
        parts = args.subset.split(":")
        if len(parts) != 3:
            logger.error("--subset must be model_id:method:concept")
            sys.exit(1)
        runs = [tuple(parts)]
    else:
        runs = [(m, meth, c) for m in MODELS for meth in METHODS for c in CONCEPTS]

    logger.info(f"Entropy layer extraction: {len(runs)} candidate runs")
    done = skipped = failed = 0

    for model_id, method, concept in runs:
        logger.info(f"\n{'='*60}")
        logger.info(f"RUN: {model_id} | {method} | {concept}")
        try:
            result = process_one(model_id, method, concept, args.hf_token)
            if result == "done":
                done += 1
            else:
                skipped += 1
        except Exception:
            logger.error(f"FAILED:\n{traceback.format_exc()}")
            failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Extraction complete: {done} done | {skipped} skipped | {failed} failed")
    logger.info("Next step: python scripts/rq2/rq2_layer_analysis.py")


if __name__ == "__main__":
    main()
