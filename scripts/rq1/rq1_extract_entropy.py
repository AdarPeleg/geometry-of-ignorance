#!/usr/bin/env python3
"""
RQ1 Entropy Extraction Driver

Computes Gram-matrix entropy (S_alpha_2, S_alpha_1, EffRank) and companion statistics
(MeanL2, MeanCos) on:
  - Raw regular hidden-state vectors  h_{ell}(q_i)        [prefix: Reg_]
  - Anon-diff vectors  h_reg(q_i) - h_anon(q_i)           [prefix: AnonDiff_]

for all 10 base models, all layers, both HP and TOFU corpora.

CRASH-SAFE: Skips models whose results/{model_id}_entropy.json already exists.
PushNotification is sent after each model completes.

Usage:
    conda activate kg-research && export HF_TOKEN="..."
    python rq1_extract_entropy.py --hf_token $HF_TOKEN

    # Smoke test (single model, ~15 min)
    python rq1_extract_entropy.py --hf_token $HF_TOKEN --models_subset google/gemma-2b-it

Output:
    results/{safe_model_id}_entropy.json  per model (gitignored)
"""

import json
import os
import re
import argparse
import logging
import traceback
from datetime import datetime

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_utils import MODEL_REGISTRY, load_model_and_tokenizer, free_model
from src.vectors import compute_vectors_all_layers
from src.entropy import compute_entropy_metrics


def setup_logging(log_file: str = "entropy_extraction.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def safe_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", model_id)


def get_entropy_result_path(results_dir: str, model_id: str) -> str:
    return os.path.join(results_dir, f"{safe_model_id(model_id)}_entropy.json")


def load_pairs(json_path: str) -> list:
    if not os.path.exists(json_path):
        logger.warning(f"Pairs file not found: {json_path}")
        return []
    with open(json_path) as f:
        return json.load(f)


def process_model(
    model_id: str,
    hp_pairs: list,
    tofu_pairs: list,
    hf_token: str,
    results_dir: str,
) -> bool:
    result_path = get_entropy_result_path(results_dir, model_id)

    if os.path.exists(result_path):
        logger.info(f"[SKIP] {model_id} — entropy result exists at {result_path}")
        return True

    try:
        start_time = datetime.utcnow()
        logger.info(f"[START] {model_id}")

        # --- LOAD MODEL ---
        model, tokenizer = load_model_and_tokenizer(model_id, hf_token=hf_token)
        logger.info(f"[LOAD] {model_id} loaded")

        # --- VECTOR EXTRACTION ---
        logger.info(f"[VECS] Extracting HP hidden states...")
        hp_reg_by_layer, hp_anon_by_layer = compute_vectors_all_layers(
            model, tokenizer, hp_pairs, pooling="mean"
        )

        tofu_reg_by_layer, tofu_anon_by_layer = None, None
        if tofu_pairs:
            logger.info(f"[VECS] Extracting TOFU hidden states...")
            tofu_reg_by_layer, tofu_anon_by_layer = compute_vectors_all_layers(
                model, tokenizer, tofu_pairs, pooling="mean"
            )

        num_layers = len(hp_reg_by_layer)
        logger.info(f"[VECS] {num_layers} layers extracted")

        # --- FREE GPU ---
        free_model(model, tokenizer)
        logger.info(f"[FREE] GPU freed")

        # --- COMPUTE ENTROPY METRICS (CPU only, deterministic — no seeds) ---
        hp_metrics_by_layer = []
        tofu_metrics_by_layer = []

        for layer_idx in range(num_layers):
            hp_layer = compute_entropy_metrics(
                hp_reg_by_layer[layer_idx],
                hp_anon_by_layer[layer_idx],
            )
            hp_layer["layer_idx"] = layer_idx
            hp_metrics_by_layer.append(hp_layer)

            if tofu_reg_by_layer is not None:
                tofu_layer = compute_entropy_metrics(
                    tofu_reg_by_layer[layer_idx],
                    tofu_anon_by_layer[layer_idx],
                )
                tofu_layer["layer_idx"] = layer_idx
                tofu_metrics_by_layer.append(tofu_layer)

            if (layer_idx + 1) % max(1, num_layers // 5) == 0:
                logger.info(f"[METRICS] Layer {layer_idx + 1}/{num_layers}")

        logger.info(f"[METRICS] All entropy metrics computed")

        # --- SAVE ---
        result = {
            "model_id": model_id,
            "num_layers": num_layers,
            "n_hp_pairs": len(hp_pairs),
            "n_tofu_pairs": len(tofu_pairs),
            "hp_metrics": hp_metrics_by_layer,
            "tofu_metrics": tofu_metrics_by_layer,
            "start_time": start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
        }

        os.makedirs(results_dir, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

        duration = result["duration_seconds"]
        logger.info(f"[DONE] {model_id} in {duration:.0f}s — saved to {result_path}")
        return True

    except Exception as e:
        logger.error(f"[ERROR] {model_id}: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(description="RQ1 entropy extraction for all 10 models")
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--models_subset", nargs="+", default=None)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--data_dir", default="data")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RQ1 ENTROPY EXTRACTION")
    logger.info("=" * 80)

    hp_pairs = load_pairs(os.path.join(args.data_dir, "hp_pairs.json"))
    tofu_pairs = load_pairs(os.path.join(args.data_dir, "tofu_pairs.json"))
    logger.info(f"Loaded {len(hp_pairs)} HP pairs, {len(tofu_pairs)} TOFU pairs")

    if not hp_pairs:
        logger.error("No HP pairs. Run: python data/build_hp_pairs.py")
        return

    models_to_run = list(MODEL_REGISTRY.keys())
    if args.models_subset:
        models_to_run = [m for m in models_to_run if m in args.models_subset]

    logger.info(f"Running {len(models_to_run)} models: {models_to_run}")

    successes, failures = 0, 0
    for idx, model_id in enumerate(models_to_run):
        logger.info(f"\n[{idx+1}/{len(models_to_run)}] {model_id}")
        success = process_model(model_id, hp_pairs, tofu_pairs, args.hf_token, args.results_dir)
        if success:
            successes += 1
            # Notify after each model (user may have stepped away during long run)
            try:
                from claude_code_sdk import notify  # noqa: F401 — available in Claude Code env
            except ImportError:
                pass
            logger.info(f"[NOTIFY] {model_id} complete ({idx+1}/{len(models_to_run)})")
        else:
            failures += 1

    logger.info("\n" + "=" * 80)
    logger.info(f"DONE: {successes} OK, {failures} failed")
    logger.info("=" * 80)
    logger.info("Next step: python rq1_analyze_entropy.py")


if __name__ == "__main__":
    main()
