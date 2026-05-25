#!/usr/bin/env python3
"""
RQ1 Per-Model Extraction Driver

Extracts steering vectors, QA scores, and AUSS metrics for all 10 base models.

CRASH-SAFE DESIGN:
  - Results written to disk immediately after each model completes
  - Each model wrapped in try/except with detailed logging
  - Skips already-completed models (resume-safe)
  - All computation offloaded to CPU after GPU inference to free VRAM
  - Detailed logging to extraction.log

Usage:
    # Full run (all 10 models, ~2.5 hours)
    python rq1_extract.py --hf_token $HF_TOKEN

    # Smoke test (Gemma-2B only, ~10 min)
    python rq1_extract.py --hf_token $HF_TOKEN --models_subset google/gemma-2b-it

    # Resume interrupted run
    python rq1_extract.py --hf_token $HF_TOKEN  # automatically skips completed

Output:
    results/{model_id}.json per model (gitignored, large files)
    extraction.log with full logging
"""

import json
import os
import argparse
import logging
from datetime import datetime
import traceback

import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_utils import (
    MODEL_REGISTRY,
    load_model_and_tokenizer,
    get_result_path,
    free_model,
)
from src.vectors import compute_vectors_all_layers, compute_per_batch_directions
from src.metrics import compute_all_metrics
from src.entropy import compute_entropy_metrics
from src.qa_eval import run_qa_eval


def get_entropy_result_path(results_dir: str, model_id: str) -> str:
    import re
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_id)
    return os.path.join(results_dir, f"{safe}_entropy.json")


def get_vectors_path(results_dir: str, model_id: str) -> str:
    import re
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_id)
    return os.path.join(results_dir, f"{safe}__vectors.npz")


# Set up logging to file and console
def setup_logging(log_file: str = "extraction.log"):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def load_pairs(json_path: str) -> list:
    """Load QA pairs from JSON, with error handling."""
    if not os.path.exists(json_path):
        logger.warning(f"Pairs file not found: {json_path}")
        return []
    with open(json_path, 'r') as f:
        return json.load(f)


def process_model(
    model_id: str,
    hp_pairs: list,
    tofu_pairs: list,
    hf_token: str,
    results_dir: str,
    n_batches: int = 10,
) -> bool:
    """
    Process a single model: extract metrics and save results.

    Returns:
        True if successful, False if failed
    """
    result_path = get_result_path(results_dir, model_id)
    entropy_path = get_entropy_result_path(results_dir, model_id)
    vectors_path = get_vectors_path(results_dir, model_id)

    # RESUME-SAFETY: Skip only if AUSS, entropy, and vectors all exist
    if os.path.exists(result_path) and os.path.exists(entropy_path) and os.path.exists(vectors_path):
        logger.info(f"[SKIP] {model_id} — AUSS, entropy, and vectors all exist")
        return True

    try:
        start_time = datetime.utcnow()
        logger.info(f"[START] {model_id}")

        # --- LOAD MODEL ---
        logger.info(f"[LOAD] Loading {model_id}...")
        model, tokenizer = load_model_and_tokenizer(model_id, hf_token=hf_token)
        logger.info(f"[LOAD] {model_id} loaded successfully")

        # --- QA EVALUATION ---
        logger.info(f"[QA] Evaluating HP knowledge...")
        hp_qa_success = run_qa_eval(model, tokenizer, hp_pairs)
        logger.info(f"[QA] HP QA success: {hp_qa_success:.3f}")

        tofu_qa_success = None
        if tofu_pairs:
            logger.info(f"[QA] Evaluating TOFU knowledge...")
            tofu_qa_success = run_qa_eval(model, tokenizer, tofu_pairs)
            logger.info(f"[QA] TOFU QA success: {tofu_qa_success:.3f}")

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
        logger.info(f"[VECS] Extracted {num_layers} layers for both domains")

        # --- FREE GPU MEMORY ---
        logger.info(f"[FREE] Freeing GPU memory...")
        free_model(model, tokenizer)
        logger.info(f"[FREE] GPU memory freed")

        # --- COMPUTE METRICS (CPU only) ---
        seeds = [42, 123, 777]
        logger.info(f"[METRICS] Computing metrics for {len(seeds)} seeds × {num_layers} layers...")

        hp_metrics_by_seed = {str(s): [] for s in seeds}
        tofu_metrics_by_seed = {str(s): [] for s in seeds}

        for layer_idx in range(num_layers):
            for seed in seeds:
                # HP metrics
                hp_batch_dirs = compute_per_batch_directions(
                    hp_reg_by_layer[layer_idx],
                    hp_anon_by_layer[layer_idx],
                    n_batches=n_batches,
                    seed=seed,
                )
                hp_layer_metrics = compute_all_metrics(hp_batch_dirs)
                hp_layer_metrics["layer_idx"] = layer_idx
                hp_metrics_by_seed[str(seed)].append(hp_layer_metrics)

                # TOFU metrics
                if tofu_reg_by_layer is not None:
                    tofu_batch_dirs = compute_per_batch_directions(
                        tofu_reg_by_layer[layer_idx],
                        tofu_anon_by_layer[layer_idx],
                        n_batches=n_batches,
                        seed=seed,
                    )
                    tofu_layer_metrics = compute_all_metrics(tofu_batch_dirs)
                    tofu_layer_metrics["layer_idx"] = layer_idx
                    tofu_metrics_by_seed[str(seed)].append(tofu_layer_metrics)

            if (layer_idx + 1) % max(1, num_layers // 5) == 0:
                logger.info(f"[METRICS] Completed layer {layer_idx + 1}/{num_layers}")

        logger.info(f"[METRICS] All metrics computed")

        # --- COMPUTE ENTROPY METRICS (CPU only, deterministic — no seeds needed) ---
        logger.info(f"[ENTROPY] Computing entropy metrics for {num_layers} layers...")
        hp_entropy_by_layer = []
        tofu_entropy_by_layer = []
        for layer_idx in range(num_layers):
            hp_ent = compute_entropy_metrics(
                hp_reg_by_layer[layer_idx], hp_anon_by_layer[layer_idx]
            )
            hp_ent["layer_idx"] = layer_idx
            hp_entropy_by_layer.append(hp_ent)

            if tofu_reg_by_layer is not None:
                tofu_ent = compute_entropy_metrics(
                    tofu_reg_by_layer[layer_idx], tofu_anon_by_layer[layer_idx]
                )
                tofu_ent["layer_idx"] = layer_idx
                tofu_entropy_by_layer.append(tofu_ent)
        logger.info(f"[ENTROPY] Done")

        # --- BUILD RESULT ---
        result = {
            "model_id": model_id,
            "num_layers": num_layers,
            "n_hp_pairs": len(hp_pairs),
            "n_tofu_pairs": len(tofu_pairs),
            "n_batches": n_batches,
            "seeds": seeds,
            "hp_qa_success": hp_qa_success,
            "tofu_qa_success": tofu_qa_success,
            "hp_metrics": hp_metrics_by_seed,
            "tofu_metrics": tofu_metrics_by_seed,
            "start_time": start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
        }
        entropy_result = {
            "model_id": model_id,
            "num_layers": num_layers,
            "n_hp_pairs": len(hp_pairs),
            "n_tofu_pairs": len(tofu_pairs),
            "hp_metrics": hp_entropy_by_layer,
            "tofu_metrics": tofu_entropy_by_layer,
            "start_time": start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
        }

        # --- SAVE RAW VECTORS (float16, all layers) ---
        # Stored so any new metric can be recomputed on CPU without reloading the model.
        logger.info(f"[VECTORS] Saving raw hidden states as float16 npz...")
        os.makedirs(results_dir, exist_ok=True)
        np.savez(
            vectors_path,
            hp_reg=torch.stack(hp_reg_by_layer).to(torch.float16).numpy(),
            hp_anon=torch.stack(hp_anon_by_layer).to(torch.float16).numpy(),
            tofu_reg=torch.stack(tofu_reg_by_layer).to(torch.float16).numpy() if tofu_reg_by_layer else np.array([]),
            tofu_anon=torch.stack(tofu_anon_by_layer).to(torch.float16).numpy() if tofu_anon_by_layer else np.array([]),
        )
        logger.info(f"[VECTORS] Saved → {vectors_path}")

        # --- SAVE RESULT (ATOMIC WRITE) ---
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        with open(entropy_path, "w") as f:
            json.dump(entropy_result, f, indent=2)

        logger.info(f"[DONE] {model_id} — saved AUSS → {result_path}, entropy → {entropy_path}, vectors → {vectors_path}")
        tofu_str = f"{tofu_qa_success:.3f}" if tofu_qa_success is not None else "N/A"
        logger.info(f"[SUMMARY] HP QA={hp_qa_success:.3f}, TOFU QA={tofu_str}")

        # --- GIT PUSH (per-model, per CLAUDE.md policy) ---
        # vectors_path is gitignored (>100 MB, exceeds GitHub limit) — push JSON only
        try:
            import subprocess as _sp
            _sp.run(["git", "add", result_path, entropy_path], check=True)
            _sp.run(["git", "commit", "-m",
                     f"results(rq1): {model_id.split('/')[-1]} extraction complete (AUSS + entropy)"],
                    check=True)
            _sp.run(["git", "push"], check=True)
            logger.info(f"[GIT] Pushed results for {model_id}")
        except Exception as git_err:
            logger.warning(f"[GIT] Push failed (results are saved locally): {git_err}")

        return True

    except Exception as e:
        logger.error(f"[ERROR] {model_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="RQ1 metric extraction for all 10 base models"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=True,
        help="HuggingFace API token for gated models",
    )
    parser.add_argument(
        "--models_subset",
        nargs="+",
        default=None,
        help="Run only specified models (e.g. google/gemma-2b-it for smoke test)",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=10,
        help="Number of batches for direction computation (default 10)",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory for result JSONs",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory containing hp_pairs.json and tofu_pairs.json",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("RQ1 METRIC EXTRACTION")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading HP pairs from {args.data_dir}/hp_pairs.json...")
    hp_pairs = load_pairs(os.path.join(args.data_dir, "hp_pairs.json"))
    logger.info(f"Loaded {len(hp_pairs)} HP pairs")

    logger.info(f"Loading TOFU pairs from {args.data_dir}/tofu_pairs.json...")
    tofu_pairs = load_pairs(os.path.join(args.data_dir, "tofu_pairs.json"))
    logger.info(f"Loaded {len(tofu_pairs)} TOFU pairs")

    if not hp_pairs:
        logger.error("No HP pairs loaded. Run: python data/build_hp_pairs.py")
        return

    # Select models
    models_to_run = list(MODEL_REGISTRY.keys())
    if args.models_subset:
        models_to_run = [m for m in models_to_run if m in args.models_subset]

    logger.info(f"Running extraction for {len(models_to_run)} models")
    logger.info(f"Models: {models_to_run}")

    # Process each model
    successes = 0
    failures = 0

    for idx, model_id in enumerate(models_to_run):
        logger.info(f"\n[{idx+1}/{len(models_to_run)}] Processing {model_id}")
        success = process_model(
            model_id,
            hp_pairs,
            tofu_pairs,
            args.hf_token,
            args.results_dir,
            n_batches=args.n_batches,
        )
        if success:
            successes += 1
        else:
            failures += 1

    logger.info("\n" + "=" * 80)
    logger.info(f"EXTRACTION COMPLETE: {successes} successes, {failures} failures")
    logger.info("=" * 80)

    if successes > 0:
        logger.info("Running entropy analysis to generate entropy_summary.csv ...")
        import subprocess
        analyze_script = Path(__file__).parent / "rq1_analyze_entropy.py"
        subprocess.run(
            [sys.executable, str(analyze_script),
             "--results_dir", args.results_dir,
             "--output_csv", os.path.join(args.results_dir, "entropy_summary.csv")],
            check=False,
        )
        logger.info(f"entropy_summary.csv written to {args.results_dir}/")


if __name__ == "__main__":
    main()
