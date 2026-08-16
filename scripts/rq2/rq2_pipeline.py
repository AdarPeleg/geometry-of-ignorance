#!/usr/bin/env python3
"""
RQ2 Unified Pipeline — unlearn → attack, per run.

Order of execution:
  1. For every run that already has a checkpoint/metrics but NO attacks JSON:
     load the saved checkpoint and run all 4 attacks.
  2. For every run not yet started:
     load base model → unlearn → save checkpoint → verify → AUSS metrics
     → (model still in memory) → run 4 attacks → save both JSONs → free model.

This means each model is loaded ONCE and produces both metrics + attack results
before moving to the next run.

Crash-safe: skip-if-exists for both metrics and attacks JSONs. Re-running the same
command resumes from wherever it left off.

Usage:
    python rq2_pipeline.py --hf_token $HF_TOKEN
    python rq2_pipeline.py --hf_token $HF_TOKEN --attacks_only   # attack all completed, no new unlearn
    python rq2_pipeline.py --hf_token $HF_TOKEN --models Llama-2-7b
    python rq2_pipeline.py --hf_token $HF_TOKEN --epochs 2 --methods GradAscent  # smoke test
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Reduce CUDA memory fragmentation across back-to-back large model runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_utils import load_model_and_tokenizer, free_model
from src.unlearn import apply_unlearning, UnlearnConfig
from src.attacks import run_attacks as _run_attacks, ATTACK_REGISTRY
from src.vectors import compute_vectors_all_layers, compute_per_batch_directions
from src.metrics import compute_all_metrics
from src.entropy import compute_entropy_metrics
from src.qa_eval import run_qa_eval

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
METHODS  = ["GradAscent", "DPO", "NPO", "NPO+KL", "RMU", "WHP"]
CONCEPTS = ["harry_potter", "star_wars", "william_shakespeare"]

RESULTS_DIR      = Path("experiments/rq2/main")
MODELS_DIR       = RESULTS_DIR / "models"
MIA_HIDDEN_DIR   = RESULTS_DIR / "mia_hiddens"   # per-concept base-model hidden state references
DATA_DIR         = Path("data/concepts")

FORGET_THRESHOLD = 0.10
RETAIN_RATIO     = 0.80
DEFAULT_LAYER_ID = 15   # for activation steering attack


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file="rq2_pipeline.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def safe_id(model_id, method, concept):
    return f"{model_id.replace('/', '__')}__{method}__{concept}"

def metrics_path(model_id, method, concept):
    return RESULTS_DIR / f"{safe_id(model_id, method, concept)}__metrics.json"

def attacks_path(model_id, method, concept):
    return RESULTS_DIR / f"{safe_id(model_id, method, concept)}__attacks.json"

def checkpoint_dir(model_id, method, concept):
    return MODELS_DIR / safe_id(model_id, method, concept)

def checkpoint_exists(model_id, method, concept):
    d = checkpoint_dir(model_id, method, concept)
    return d.exists() and any(d.iterdir())

def atomic_write(path, data):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_concept(concept):
    with open(DATA_DIR / f"{concept}.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Shared post-unlearn computations (verification + AUSS)
# ---------------------------------------------------------------------------

def compute_rouge_l(hyp, ref):
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r:
        return 0.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if r[i-1] == h[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / n if n else 0
    r_ = lcs / m if m else 0
    return 2 * p * r_ / (p + r_) if (p + r_) else 0.0


def greedy_answer(model, tokenizer, question, max_new_tokens=60):
    device = next(model.parameters()).device
    enc = tokenizer(f"{question}\nAnswer:", return_tensors="pt", truncation=True, max_length=300)
    enc = {k: v.to(device) for k, v in enc.items()}
    plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=1.0,
                             pad_token_id=tokenizer.eos_token_id, use_cache=False)
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True)


def compute_verification(model, tokenizer, forget_pairs, retain_pairs, qa_pairs=None, n=20):
    # Use qa_pairs for forget ROUGE if available (proper Q&A; forget_pairs are Wikipedia completions)
    f_source = qa_pairs[:n] if qa_pairs else forget_pairs[:n]
    f_scores = [compute_rouge_l(greedy_answer(model, tokenizer, p["question"]), p["answer"])
                for p in f_source if p.get("answer", "")]

    # Retain: only score pairs with non-empty answers
    r_source = [p for p in retain_pairs[:n] if p.get("answer", "")]
    r_scores = [compute_rouge_l(greedy_answer(model, tokenizer, p["question"]), p["answer"])
                for p in r_source]

    return {
        "forget_rouge_l": float(sum(f_scores) / len(f_scores)) if f_scores else 0.0,
        "retain_rouge_l": float(sum(r_scores) / len(r_scores)) if r_scores else None,
    }


def compute_auss(model, tokenizer, forget_pairs):
    logger.info("  Computing AUSS metrics…")
    reg_by_layer, anon_by_layer = compute_vectors_all_layers(model, tokenizer, forget_pairs)
    n_layers = len(reg_by_layer)
    all_metrics = []
    all_entropy = []
    for i in range(n_layers):
        bd = compute_per_batch_directions(reg_by_layer[i], anon_by_layer[i])
        all_metrics.append({k: float(v) for k, v in compute_all_metrics(bd).items()})
        ent = compute_entropy_metrics(reg_by_layer[i], anon_by_layer[i])
        all_entropy.append({k: (float(v) if v == v else None) for k, v in ent.items()})
    # Peak = layer with maximum AUSS_L2 (most fragmented — most discriminative per RQ1)
    peak_layer = max(range(n_layers), key=lambda i: all_metrics[i].get("AUSS_L2", 0.0))
    peak = all_metrics[peak_layer]
    # Entropy metrics at peak layer (kept for backward compat) + per-layer array
    entropy = all_entropy[peak_layer]
    return {"peak_layer": peak_layer, "peak_metrics": peak, "all_layers": all_metrics,
            "entropy_all_layers": all_entropy,
            "n_layers": n_layers, "entropy_metrics": entropy}


# ---------------------------------------------------------------------------
# Progress log
# ---------------------------------------------------------------------------

EVENTS_FILE = Path("rq2_events.jsonl")

def write_event(event_type, model_id, method, concept, extra=None):
    """Append a structured event line, useful for tailing progress on a long run."""
    entry = {"ts": datetime.now().isoformat(), "type": event_type,
             "model": model_id.split("/")[-1], "method": method, "concept": concept}
    if extra:
        entry.update(extra)
    with open(EVENTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Reference data helpers for calibrated MIA variants
# ---------------------------------------------------------------------------

def _load_base_mia_ref_data(model_id: str, concept: str):
    """Load per-pair NLL lists from the base model's MIA attack JSON."""
    path = attacks_path(model_id, "Base", concept)
    if not path.exists():
        return None, None
    with open(path) as f:
        d = json.load(f)
    mia = d.get("attacks", {}).get("MIA", {})
    return mia.get("forget_nlls"), mia.get("retain_nlls")


def _load_base_mia_hidden_data(model_id: str, concept: str):
    """Load per-pair hidden-state arrays from the base model's .npz file."""
    safe_model = model_id.replace("/", "__")
    npz_path = MIA_HIDDEN_DIR / f"{safe_model}__{concept}.npz"
    if not npz_path.exists():
        return None, None
    ref = np.load(npz_path)
    return ref["forget_hiddens"], ref["retain_hiddens"]


def _save_base_hidden_states(model, tokenizer, forget_pairs, retain_pairs, qa_pairs,
                              model_id: str, concept: str, n_forget: int = 50, n_retain: int = 50,
                              seed: int = 42):
    """Extract and save last-layer hidden states from base model for RepSimilarity reference."""
    from src.attacks import _extract_last_hidden  # local import to avoid circular at module level
    safe_model = model_id.replace("/", "__")
    MIA_HIDDEN_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = MIA_HIDDEN_DIR / f"{safe_model}__{concept}.npz"
    if npz_path.exists():
        logger.info(f"  Base hidden states already exist: {npz_path.name}")
        return

    from tqdm import tqdm as _tqdm
    forget_source = qa_pairs if qa_pairs else forget_pairs
    forget_sample = random.Random(seed).sample(forget_source, min(n_forget, len(forget_source)))
    retain_sample = random.Random(seed + 1).sample(retain_pairs, min(n_retain, len(retain_pairs)))

    forget_hiddens = []
    for p in _tqdm(forget_sample, desc="Base hiddens (forget)", leave=False):
        h = _extract_last_hidden(model, tokenizer, p.get("question", ""), p.get("answer", ""))
        if h is not None:
            forget_hiddens.append(h)

    retain_hiddens = []
    for p in _tqdm(retain_sample, desc="Base hiddens (retain)", leave=False):
        q, a = p.get("question", ""), p.get("answer", "")
        h = _extract_last_hidden(model, tokenizer, "", q) if not a else \
            _extract_last_hidden(model, tokenizer, q, a)
        if h is not None:
            retain_hiddens.append(h)

    np.savez_compressed(npz_path,
                        forget_hiddens=np.array(forget_hiddens, dtype=np.float32),
                        retain_hiddens=np.array(retain_hiddens, dtype=np.float32))
    logger.info(f"  Base hidden states saved → {npz_path.name}")


# ---------------------------------------------------------------------------
# Base model evaluation: attacks on the unmodified model (no unlearning)
# ---------------------------------------------------------------------------

def run_base(model_id, concept, hf_token, layer_id, force_steering=False, attacks_filter=None, force_attacks=False):
    """
    Run attacks on the unmodified base model.
    Provides the high-score anchor for RQ2 correlations:
    - Base model should have high attack scores (knows the concept fully)
    - Unlearned models have lower scores (knowledge suppressed or erased)
    """
    m_path = metrics_path(model_id, "Base", concept)
    a_path = attacks_path(model_id, "Base", concept)

    if m_path.exists() and a_path.exists():
        if not force_steering and not force_attacks:
            logger.info(f"SKIP (base complete): {model_id.split('/')[-1]}/{concept}")
            return
        # Check if Steering already has corrected AUC
        existing_data = json.load(open(a_path))
        steer = existing_data.get("attacks", {}).get("Steering", {})
        if steer.get("mean_retain_wf") is not None:
            logger.info(f"SKIP (steering already corrected): {model_id.split('/')[-1]}/{concept}")
            return

    logger.info("=" * 65)
    logger.info(f"BASE EVAL: {model_id.split('/')[-1]} | {concept}")
    logger.info("=" * 65)

    concept_data = load_concept(concept)
    forget_pairs = concept_data["forget_pairs"]
    retain_pairs = concept_data["retain_pairs"]
    qa_pairs     = concept_data.get("qa_eval_pairs", forget_pairs[:10])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = None, None
    try:
        logger.info(f"  Loading base model: {model_id}")
        model, tokenizer = load_model_and_tokenizer(model_id, hf_token=hf_token)

        if not m_path.exists():
            logger.info("  Base verification…")
            verif   = compute_verification(model, tokenizer, forget_pairs, retain_pairs, qa_pairs)
            qa_acc  = run_qa_eval(model, tokenizer, qa_pairs[:20])
            auss    = compute_auss(model, tokenizer, forget_pairs)
            logger.info(f"    forget={verif['forget_rouge_l']:.3f}  qa_acc={qa_acc:.3f}")
            atomic_write(m_path, {
                "model_id": model_id, "method": "Base", "concept": concept,
                "base_verification": verif, "post_verification": verif,
                "verified": False,  # base always fails — hasn't been unlearned
                "qa_accuracy": qa_acc, "auss_metrics": auss,
                "timestamp": datetime.now().isoformat(),
            })

        if not a_path.exists() or force_steering or force_attacks:
            if force_steering and not force_attacks and a_path.exists():
                existing_data = json.load(open(a_path))
                attack_results = existing_data.get("attacks", {})
                attacks_to_run = ["Steering"]
                logger.info("  Re-running Steering only (corrected AUC)…")
            elif force_attacks and a_path.exists():
                existing_data = json.load(open(a_path))
                attack_results = existing_data.get("attacks", {})  # preserve existing
                _all_known = list(ATTACK_REGISTRY.keys())
                attacks_to_run = (
                    [a for a in _all_known if a in attacks_filter]
                    if attacks_filter else list(ATTACK_REGISTRY.keys())
                )
                logger.info(f"  Force re-running attacks ({' / '.join(attacks_to_run)})…")
            else:
                # Default base attacks; new MIA variants only if explicitly requested
                _default_base = ["Steering", "ICL", "MIA", "GCG"]
                _all_known = list(ATTACK_REGISTRY.keys())
                attacks_to_run = (
                    [a for a in _all_known if a in attacks_filter]
                    if attacks_filter else _default_base
                )
                attack_results = {}
                logger.info(f"  Running base attacks ({' / '.join(attacks_to_run)})…")
            for aname in attacks_to_run:
                logger.info(f"    → {aname}")
                fn = ATTACK_REGISTRY[aname]
                if aname == "GCG":
                    # n_eval controls how many forget-set pairs GCG evaluates per run;
                    # default 3 keeps runtime manageable across the full grid, override
                    # via GCG_N_EVAL to evaluate more pairs per run at increased cost.
                    kw = {"seed": 42, "n_eval": int(os.environ.get("GCG_N_EVAL", 3)), "gcg_steps": 50}
                elif aname in ("MIA_Ref", "RepSimilarity"):
                    # Reference attacks on base model have no base reference — skip gracefully
                    attack_results[aname] = {"score": 1.0, "auc": 1.0,
                                             "note": "base model — reference is self"}
                    continue
                else:
                    kw = {"seed": 42, "qa_pairs": qa_pairs}
                try:
                    attack_results[aname] = fn(model, tokenizer, forget_pairs, retain_pairs, **kw)
                    score_val = attack_results[aname].get("score", float("nan"))
                    logger.info(f"      score={score_val:.3f}" if score_val == score_val else "      score=nan")
                except Exception as e:
                    logger.error(f"      {aname} failed: {e}")
                    attack_results[aname] = {"score": float("nan"), "error": str(e)}
            atomic_write(a_path, {
                "model_id": model_id, "method": "Base", "concept": concept,
                "layer_id": layer_id, "attacks": attack_results,
                "timestamp": datetime.now().isoformat(),
            })
            logger.info(f"  Base attacks saved → {a_path.name}")

        # Collect hidden states for RepSimilarity reference (after MIA attack JSON is saved)
        if attacks_filter and "RepSimilarity" in attacks_filter:
            _save_base_hidden_states(model, tokenizer, forget_pairs, retain_pairs, qa_pairs,
                                     model_id, concept)

        write_event("done", model_id, "Base", concept, {"completed": 0, "total": 0})

    finally:
        if model is not None:
            free_model(model, tokenizer)


# ---------------------------------------------------------------------------
# Single run: unlearn (or load ckpt) + verify + AUSS + attacks + push
# ---------------------------------------------------------------------------

def run_one(model_id, method, concept, hf_token, cfg, layer_id,
            force_attacks=False, steering_only=False, gcg_only=False, attacks_filter=None):
    m_path = metrics_path(model_id, method, concept)
    a_path = attacks_path(model_id, method, concept)
    ckpt   = checkpoint_dir(model_id, method, concept)
    ckpt_ok = checkpoint_exists(model_id, method, concept)

    # Both done — skip unless force_attacks requested
    if m_path.exists() and a_path.exists() and not force_attacks:
        logger.info(f"SKIP (complete): {safe_id(model_id, method, concept)}")
        return

    logger.info("=" * 65)
    logger.info(f"RUN: {model_id.split('/')[-1]} | {method} | {concept}")
    if ckpt_ok and not m_path.exists():
        logger.info("  ↳ checkpoint exists — loading from disk (no training)")
    elif ckpt_ok and m_path.exists():
        logger.info("  ↳ metrics done, checkpoint exists — attacks only")
    logger.info("=" * 65)

    concept_data = load_concept(concept)
    forget_pairs = concept_data["forget_pairs"]
    retain_pairs = concept_data["retain_pairs"]
    qa_pairs     = concept_data.get("qa_eval_pairs", forget_pairs[:10])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = None, None
    base_verif = {}

    try:
        # ── Phase 1: get a model in memory ──────────────────────────────────
        if m_path.exists():
            # metrics already done — load checkpoint for attacks only
            logger.info(f"  Loading checkpoint for attacks: {ckpt}")
            model, tokenizer = load_model_and_tokenizer(str(ckpt), hf_token=None)
        elif ckpt_ok:
            # checkpoint exists but metrics not written — load ckpt, skip training
            logger.info(f"  Loading checkpoint (skip training): {ckpt}")
            model, tokenizer = load_model_and_tokenizer(str(ckpt), hf_token=None)
            base_verif = {"note": "loaded from checkpoint — base metrics unavailable"}
        else:
            # fresh run: load base model, train, save checkpoint
            logger.info(f"  Loading base model: {model_id}")
            model, tokenizer = load_model_and_tokenizer(model_id, hf_token=hf_token)

            logger.info("  Base verification…")
            base_verif = compute_verification(model, tokenizer, forget_pairs, retain_pairs, qa_pairs)
            ret_str = f"{base_verif['retain_rouge_l']:.3f}" if base_verif['retain_rouge_l'] is not None else "N/A"
            logger.info(f"    forget={base_verif['forget_rouge_l']:.3f}  retain={ret_str}")

            logger.info(f"  Unlearning: {method}")
            apply_unlearning(method, model, tokenizer, forget_pairs, retain_pairs, cfg, str(ckpt))
            logger.info(f"  Checkpoint saved → {ckpt}")

        # ── Phase 2: post-unlearn metrics (if not already done) ────────────
        if not m_path.exists():
            logger.info("  Post-unlearn verification…")
            post_verif = compute_verification(model, tokenizer, forget_pairs, retain_pairs, qa_pairs)
            ret_str = f"{post_verif['retain_rouge_l']:.3f}" if post_verif['retain_rouge_l'] is not None else "N/A"
            logger.info(f"    forget={post_verif['forget_rouge_l']:.3f}  retain={ret_str}")

            logger.info("  QA accuracy…")
            qa_acc = run_qa_eval(model, tokenizer, qa_pairs[:20])
            logger.info(f"    qa_acc={qa_acc:.3f}")

            auss = compute_auss(model, tokenizer, forget_pairs)

            fr   = post_verif["forget_rouge_l"]
            rr   = post_verif.get("retain_rouge_l")
            br   = base_verif.get("retain_rouge_l") or None
            # If retain answers are all empty, skip retain check; only require forget drop
            if rr is None:
                verified = fr < FORGET_THRESHOLD
            else:
                verified = fr < FORGET_THRESHOLD and rr >= RETAIN_RATIO * (br or 1.0)

            metrics_result = {
                "model_id": model_id,
                "method":   method,
                "concept":  concept,
                "config":   {"n_epochs": cfg.n_epochs, "lr": cfg.lr,
                             "batch_size": cfg.batch_size, "rmu_layer_id": cfg.rmu_layer_id},
                "base_verification": base_verif,
                "post_verification": post_verif,
                "verified":    verified,
                "qa_accuracy": qa_acc,
                "auss_metrics": auss,
                "checkpoint_dir": str(ckpt),
                "timestamp": datetime.now().isoformat(),
            }
            atomic_write(m_path, metrics_result)
            logger.info(f"  Metrics saved → {m_path.name}  verified={verified}")
        else:
            with open(m_path) as f:
                d = json.load(f)
            pv  = d.get("post_verification", {})
            bv  = d.get("base_verification", {})
            fr  = pv.get("forget_rouge_l", 1.0)
            rr  = pv.get("retain_rouge_l")
            br  = bv.get("retain_rouge_l") or None
            if rr is None:
                verified = fr < FORGET_THRESHOLD
            else:
                verified = fr < FORGET_THRESHOLD and rr >= RETAIN_RATIO * (br or 1.0)

        # ── Phase 3: attacks ────────────────────────────────────────────────
        need_attacks = not a_path.exists() or force_attacks
        if need_attacks:
            # In merge mode (force + existing file): load prior results so non-targeted
            # attacks are preserved; only re-run the requested subset.
            if a_path.exists() and force_attacks:
                with open(a_path) as f:
                    prior = json.load(f)
                attack_results = prior.get("attacks", {})
            else:
                attack_results = {}

            _default_attacks = ["Steering", "ICL", "MIA", "GCG"]
            if steering_only:
                attack_names = ["Steering"]
            elif gcg_only:
                attack_names = ["GCG"]
            elif attacks_filter:
                attack_names = [a for a in list(ATTACK_REGISTRY.keys()) if a in attacks_filter]
            else:
                attack_names = _default_attacks
            logger.info(f"  Running attacks ({' / '.join(attack_names)})…")

            # Load calibrated MIA reference data once (base model JSON + hidden state npz)
            base_f_nlls, base_r_nlls = _load_base_mia_ref_data(model_id, concept)
            base_f_hids, base_r_hids = _load_base_mia_hidden_data(model_id, concept)
            if any(a in attack_names for a in ("MIA_Ref", "RepSimilarity")):
                if base_f_nlls is None:
                    logger.warning("  MIA_Ref/RepSimilarity: no base NLLs found — run base model MIA first")
                if base_f_hids is None:
                    logger.warning("  RepSimilarity: no base hidden states found — run base model with --attacks RepSimilarity")

            for aname in attack_names:
                logger.info(f"    → {aname}")
                fn = ATTACK_REGISTRY[aname]
                try:
                    if aname == "GCG":
                        # n_eval controls how many forget-set pairs GCG evaluates per run;
                        # default 3 keeps runtime manageable across the full grid, override
                        # via GCG_N_EVAL to evaluate more pairs per run at increased cost.
                        kw = {"seed": 42, "n_eval": int(os.environ.get("GCG_N_EVAL", 3)), "gcg_steps": 50}
                    elif aname == "MIA_Ref":
                        kw = {"seed": 42, "qa_pairs": qa_pairs,
                              "base_forget_nlls": base_f_nlls, "base_retain_nlls": base_r_nlls}
                    elif aname == "RepSimilarity":
                        kw = {"seed": 42, "qa_pairs": qa_pairs,
                              "base_forget_hiddens": base_f_hids, "base_retain_hiddens": base_r_hids}
                    else:
                        kw = {"seed": 42, "qa_pairs": qa_pairs if qa_pairs else None}
                    attack_results[aname] = fn(model, tokenizer, forget_pairs, retain_pairs, **kw)
                    score_val = attack_results[aname].get('score', float('nan'))
                    logger.info(f"      score={score_val:.3f}" if score_val == score_val else "      score=nan")
                except Exception as e:
                    logger.error(f"      {aname} failed: {e}")
                    attack_results[aname] = {"score": float("nan"), "error": str(e)}

            atomic_write(a_path, {
                "model_id": model_id, "method": method, "concept": concept,
                "layer_id": layer_id, "attacks": attack_results,
                "timestamp": datetime.now().isoformat(),
            })
            logger.info(f"  Attacks saved → {a_path.name}")
        else:
            logger.info(f"  Attacks already done: {a_path.name}")

    finally:
        if model is not None:
            free_model(model, tokenizer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RQ2 Unified Pipeline")
    parser.add_argument("--hf_token",    default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--models",      nargs="+", default=None)
    parser.add_argument("--methods",     nargs="+", default=None)
    parser.add_argument("--concepts",    nargs="+", default=None)
    parser.add_argument("--attacks_only", action="store_true",
                        help="Only run attacks on already-completed unlearn runs")
    parser.add_argument("--base_attacks", action="store_true",
                        help="Run attacks on unmodified base models (anchor for correlations)")
    parser.add_argument("--epochs",   type=int,   default=10)
    parser.add_argument("--lr",       type=float, default=1e-5)
    parser.add_argument("--batch",    type=int,   default=4)
    parser.add_argument("--rmu_layer",type=int,   default=7)
    parser.add_argument("--layer_id", type=int,   default=DEFAULT_LAYER_ID,
                        help="Layer for activation steering attack")
    parser.add_argument("--force",         action="store_true")
    parser.add_argument("--force_attacks", action="store_true",
                        help="Re-run attacks even on runs that already have attack files")
    parser.add_argument("--steering_only", action="store_true",
                        help="With --force_attacks: re-run only Steering; preserve ICL/GCG/MIA results")
    parser.add_argument("--gcg_only", action="store_true",
                        help="With --force_attacks: re-run only GCG; preserve Steering/ICL/MIA results")
    parser.add_argument("--attacks", nargs="+", default=None,
                        metavar="ATTACK",
                        help="Explicit list of attacks to run, e.g. --attacks ICL MIA Steering")
    args = parser.parse_args()

    setup_logging()
    logger.info("RQ2 Pipeline started")

    models   = [m for m in MODELS   if not args.models   or any(f in m for f in args.models)]
    methods  = [m for m in METHODS  if not args.methods  or m in args.methods]
    concepts = [c for c in CONCEPTS if not args.concepts or c in args.concepts]

    cfg = UnlearnConfig(
        n_epochs=args.epochs, lr=args.lr, batch_size=args.batch, rmu_layer_id=args.rmu_layer
    )

    logger.info(f"Grid: {len(models)}M × {len(methods)}Mth × {len(concepts)}C = "
                f"{len(models)*len(methods)*len(concepts)} runs")

    # Build ordered run list:
    # Priority 1 — runs with metrics done but attacks missing (evaluate what we have first)
    # Priority 2 — runs with checkpoint but no metrics (finish partial runs)
    # Priority 3 — runs not started at all
    force_attacks = getattr(args, "force_attacks", False)
    def priority(model_id, method, concept):
        m_done = metrics_path(model_id, method, concept).exists()
        a_done = attacks_path(model_id, method, concept).exists()
        ck     = checkpoint_exists(model_id, method, concept)
        if m_done and a_done and not force_attacks: return 99  # complete, skip
        if m_done and (not a_done or force_attacks): return 0  # attacks missing or forced
        if ck and not m_done:    return 1  # checkpoint exists, no metrics
        return 2                            # not started

    runs = []
    for mid in models:
        for mth in methods:
            for con in concepts:
                p = priority(mid, mth, con)
                if p < 99:
                    runs.append((p, mid, mth, con))

    runs.sort(key=lambda x: x[0])

    logger.info(f"Runs to process: {len(runs)}")
    for p, mid, mth, con in runs[:5]:
        logger.info(f"  [{p}] {mid.split('/')[-1]} / {mth} / {con}")
    if len(runs) > 5:
        logger.info(f"  … and {len(runs)-5} more")

    # ── Optional: base model attacks (high-score anchor for correlations) ──────
    if args.base_attacks:
        logger.info("Running base model attacks…")
        for mid in models:
            for con in concepts:
                try:
                    run_base(mid, con, args.hf_token, args.layer_id,
                             attacks_filter=getattr(args, "attacks", None),
                             force_attacks=force_attacks)
                except Exception as e:
                    logger.error(f"Base eval FAILED {mid}/{con}: {e}")
                    logger.error(traceback.format_exc())
        logger.info("Base model attacks complete.")
        if not runs:
            return  # --base_attacks only mode

    errors, completed = [], 0
    for p, model_id, method, concept in runs:
        if args.attacks_only and not metrics_path(model_id, method, concept).exists():
            continue
        try:
            run_one(model_id, method, concept,
                    args.hf_token, cfg, args.layer_id,
                    force_attacks=force_attacks,
                    steering_only=getattr(args, "steering_only", False),
                    gcg_only=getattr(args, "gcg_only", False),
                    attacks_filter=getattr(args, "attacks", None))
            completed += 1
            logger.info(f"Progress: {completed}/{len(runs)}")
            write_event("done", model_id, method, concept, {"completed": completed, "total": len(runs)})
        except Exception as e:
            msg = f"FAILED {model_id}/{method}/{concept}: {e}"
            logger.error(msg)
            logger.error(traceback.format_exc())
            errors.append(msg)
            write_event("error", model_id, method, concept, {"error": str(e)[:200]})
        finally:
            # Aggressively reclaim VRAM between runs to avoid OOM on back-to-back large models
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            time.sleep(3)  # let OS reclaim pinned memory

    logger.info("=" * 65)
    logger.info(f"Pipeline complete: {completed}/{len(runs)} succeeded, "
                f"{len(errors)} failures")
    for e in errors:
        logger.error(f"  {e}")


if __name__ == "__main__":
    main()
