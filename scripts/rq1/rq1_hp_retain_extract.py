#!/usr/bin/env python3
"""
RQ1 — HP retain_qa fragmentation extraction.

Runs the same AUSS pipeline as rq1_sw_qa_filtered.py but on HP retain_qa pairs
(94 pairs from MUSE-Books retain_qa split, same HP domain as hp_pairs.json).

This gives us a second HP subset (retain vs forget) to check:
  - Are both forget and retain HP questions coherent? (they should be)
  - Does coherence hold across the full HP domain, not just the forget split?

Output: results/{model_id}_hp_retain.json per model.
Resume-safe: skips models whose output already exists.
"""

import json
import os
import re
import argparse
import logging
from datetime import datetime

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_utils import MODEL_REGISTRY, load_model_and_tokenizer, free_model
from src.vectors import compute_vectors_all_layers, compute_per_batch_directions
from src.metrics import compute_all_metrics

N_BATCHES = 10   # 94 pairs / ~9-10 per batch — matches HP/TOFU baseline
BATCH_SIZE = 5   # must divide evenly into n_batches
SEEDS = [42, 123, 777]
MIN_PAIRS = 10


def setup_logging(log_file="logs/rq1_hp_retain.log"):
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def safe_id(model_id):
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", model_id)


def out_path(results_dir, model_id):
    return os.path.join(results_dir, f"{safe_id(model_id)}_hp_retain.json")


def compute_metrics(reg_by_layer, anon_by_layer, indices, n_batches):
    n = len(indices)
    if n < MIN_PAIRS:
        return None
    idx_t = torch.tensor(indices, dtype=torch.long)
    num_layers = len(reg_by_layer)
    metrics_by_seed = {str(s): [] for s in SEEDS}
    for layer_idx in range(num_layers):
        reg_sub = reg_by_layer[layer_idx][idx_t]
        anon_sub = anon_by_layer[layer_idx][idx_t]
        for seed in SEEDS:
            dirs = compute_per_batch_directions(reg_sub, anon_sub, n_batches=n_batches, seed=seed)
            m = compute_all_metrics(dirs)
            m["layer_idx"] = layer_idx
            metrics_by_seed[str(seed)].append(m)
    return metrics_by_seed


def process_model(model_id, pairs, hf_token, results_dir):
    path = out_path(results_dir, model_id)
    if os.path.exists(path):
        logger.info(f"[SKIP] {model_id}")
        return

    try:
        start = datetime.utcnow()
        logger.info(f"[START] {model_id}")
        model, tokenizer = load_model_and_tokenizer(model_id, hf_token=hf_token)

        logger.info(f"[VECS] Extracting hidden states for {len(pairs)} HP retain pairs...")
        reg_by_layer, anon_by_layer = compute_vectors_all_layers(model, tokenizer, pairs, pooling="mean")
        num_layers = len(reg_by_layer)
        logger.info(f"[VECS] {num_layers} layers")

        free_model(model, tokenizer)

        n = len(pairs)
        n_batches = max(2, n // BATCH_SIZE)
        logger.info(f"[METRICS] n={n}, n_batches={n_batches}")
        metrics = compute_metrics(reg_by_layer, anon_by_layer, list(range(n)), n_batches)

        result = {
            "model_id": model_id,
            "num_layers": num_layers,
            "n_pairs": n,
            "n_batches": n_batches,
            "metrics": metrics,
            "start_time": start.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - start).total_seconds(),
        }

        os.makedirs(results_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"[DONE] {model_id} → {path}")

    except Exception as e:
        logger.error(f"[FAIL] {model_id}: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--models_subset", nargs="*", default=None)
    parser.add_argument("--results_dir", default="experiments/rq1/main")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    pairs_path = os.path.join(args.data_dir, "hp_retain_pairs.json")
    with open(pairs_path) as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} HP retain pairs")

    models = list(MODEL_REGISTRY.keys())
    if args.models_subset:
        models = [m for m in models if m in args.models_subset]
    logger.info(f"Models: {len(models)}")

    for i, model_id in enumerate(models):
        logger.info(f"\n[{i+1}/{len(models)}] {model_id}")
        process_model(model_id, pairs, args.hf_token, args.results_dir)

    logger.info("All done.")


if __name__ == "__main__":
    main()
