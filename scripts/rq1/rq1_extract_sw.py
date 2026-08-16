#!/usr/bin/env python3
"""
RQ1 Star Wars extraction — third-dataset robustness experiment.

Extracts AUSS metrics for Star Wars QA pairs (25 topically-coherent questions
about one well-known fictional universe) across all 10 base models.

Expected result: COHERENT (high Centroid_Norm, low AUSS_L2), same as HP,
because the model is well-trained on Star Wars content.

Outputs: results/{model_id}_sw.json  (gitignored)
         results/summary_sw.csv       (committed)

Crash-safe: skips models whose _sw.json already exists.

Usage:
    conda activate kg-research
    python scripts/rq1/rq1_extract_sw.py --hf_token $HF_TOKEN

    # Smoke test (single model):
    python scripts/rq1/rq1_extract_sw.py --hf_token $HF_TOKEN --models_subset google/gemma-2b-it
"""

import json
import os
import re
import argparse
import logging
from datetime import datetime

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_utils import MODEL_REGISTRY, load_model_and_tokenizer, free_model
from src.vectors import compute_vectors_all_layers, compute_per_batch_directions
from src.metrics import compute_all_metrics
from src.qa_eval import run_qa_eval


def setup_logging(log_file: str = "logs/rq1_sw_extract.log"):
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def safe_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", model_id)


def sw_result_path(results_dir: str, model_id: str) -> str:
    return os.path.join(results_dir, f"{safe_model_id(model_id)}_sw.json")


def process_model(
    model_id: str,
    sw_pairs: list,
    hf_token: str,
    results_dir: str,
    n_batches: int = 5,
) -> dict | None:
    """
    Extract AUSS metrics for SW pairs for one model.
    Returns result dict on success, None on failure.
    n_batches=5 (not 10) because we only have 25 pairs (5 per batch).
    """
    out_path = sw_result_path(results_dir, model_id)
    if os.path.exists(out_path):
        logger.info(f"[SKIP] {model_id} — {out_path} exists")
        with open(out_path) as f:
            return json.load(f)

    try:
        start = datetime.utcnow()
        logger.info(f"[START] {model_id}")

        model, tokenizer = load_model_and_tokenizer(model_id, hf_token=hf_token)

        logger.info("[QA] SW knowledge eval...")
        sw_qa = run_qa_eval(model, tokenizer, sw_pairs)
        logger.info(f"[QA] SW QA success: {sw_qa:.3f}")

        logger.info("[VECS] Extracting SW hidden states...")
        sw_reg, sw_anon = compute_vectors_all_layers(model, tokenizer, sw_pairs, pooling="mean")
        num_layers = len(sw_reg)
        logger.info(f"[VECS] {num_layers} layers extracted")

        free_model(model, tokenizer)

        seeds = [42, 123, 777]
        sw_metrics = {str(s): [] for s in seeds}

        for layer_idx in range(num_layers):
            for seed in seeds:
                dirs = compute_per_batch_directions(
                    sw_reg[layer_idx], sw_anon[layer_idx],
                    n_batches=n_batches, seed=seed,
                )
                m = compute_all_metrics(dirs)
                m["layer_idx"] = layer_idx
                sw_metrics[str(seed)].append(m)

            if (layer_idx + 1) % max(1, num_layers // 5) == 0:
                logger.info(f"[METRICS] layer {layer_idx+1}/{num_layers}")

        result = {
            "model_id": model_id,
            "num_layers": num_layers,
            "n_sw_pairs": len(sw_pairs),
            "n_batches": n_batches,
            "seeds": seeds,
            "sw_qa_success": sw_qa,
            "sw_metrics": sw_metrics,
            "start_time": start.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - start).total_seconds(),
        }

        os.makedirs(results_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"[DONE] {model_id} saved → {out_path}")

        return result

    except Exception as e:
        logger.error(f"[FAIL] {model_id}: {e}", exc_info=True)
        return None


def peak_layer_metrics(metrics_by_seed: dict, metric: str) -> float:
    """Average metric at its peak layer across 3 seeds."""
    all_vals = []
    for seed_data in metrics_by_seed.values():
        vals = [layer[metric] for layer in seed_data if metric in layer]
        if vals:
            all_vals.append(max(vals) if metric in {"Centroid_Norm", "Global_Coherence",
                                                      "Batch_SecondMoment_TopEig"} else min(vals))
    return float(np.mean(all_vals)) if all_vals else float("nan")


METRICS = [
    "Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig",
    "AUSS_L2", "AUSS_Cos2", "AUSS_Jac", "Batch_Cov_Trace", "Batch_Cov_TopEig",
]


def build_summary(results: list, results_dir: str):
    import csv
    rows = []
    for r in results:
        if r is None:
            continue
        row = {"model_id": r["model_id"], "sw_qa": r["sw_qa_success"]}
        for m in METRICS:
            row[f"{m}_sw"] = peak_layer_metrics(r["sw_metrics"], m)
        rows.append(row)

    if not rows:
        logger.warning("No results to summarize")
        return

    out = os.path.join(results_dir, "summary_sw.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"[SUMMARY] Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--models_subset", nargs="*", default=None)
    parser.add_argument("--results_dir", default="experiments/rq1/main")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--n_batches", type=int, default=5,
                        help="Batches for direction computation (default 5; 25 pairs / 5 = 5 per batch)")
    args = parser.parse_args()

    sw_path = os.path.join(args.data_dir, "sw_pairs.json")
    if not os.path.exists(sw_path):
        logger.error(f"sw_pairs.json not found at {sw_path}. Run: python data/build_sw_pairs.py")
        return

    with open(sw_path) as f:
        sw_pairs = json.load(f)
    logger.info(f"Loaded {len(sw_pairs)} SW pairs")

    models = list(MODEL_REGISTRY.keys())
    if args.models_subset:
        models = [m for m in models if m in args.models_subset]
    logger.info(f"Models to process: {len(models)}")

    results = []
    for i, model_id in enumerate(models):
        logger.info(f"\n[{i+1}/{len(models)}] {model_id}")
        r = process_model(model_id, sw_pairs, args.hf_token, args.results_dir, args.n_batches)
        results.append(r)

    build_summary([r for r in results if r], args.results_dir)
    logger.info("All done.")


if __name__ == "__main__":
    main()
