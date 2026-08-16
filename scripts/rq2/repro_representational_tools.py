#!/usr/bin/env python3
"""
Head-to-head benchmark vs. Xu et al. (2025) "Unlearning Isn't Deletion".

This reproduces the AUTHORS' OWN released code (github.com/XiaoyuXU1/
Representational_Analysis_Tools, cloned unmodified into
external/Representational_Analysis_Tools/, one directory above this repo --
see README.md "Reproducing RQ2" for setup) and runs it on our own RQ2
base-vs-unlearned checkpoint pairs, using the exact same forget-set probe
queries (qa_eval_pairs) our own attacks use -- a genuine, same-settings
head-to-head, not a re-implementation from the paper text.

For each (model, method, concept) with both a base model and a saved
unlearned checkpoint on disk:
  1. Calls their UNMODIFIED run_feature_analysis(feature="pca_sim"/"cka") to
     produce their official PDF plots (proof of faithful reproduction).
  2. Additionally extracts the same underlying per-layer values (their
     functions only plot, they don't return numbers) using the identical
     computation logic copied verbatim from their source, to build a
     quantitative summary table.
  3. Compares the resulting pre/post similarity summary against the run's
     attack success scores (Steering/ICL/GCG/MIA), alongside our own AUSS
     metrics for the same runs, in experiments/rq2/xu_comparison/table_repro_comparison.csv.

Usage:
    python scripts/rq2/repro_representational_tools.py
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "external" / "Representational_Analysis_Tools"
                        / "representational_analysis" / "src"))

# Their code (unmodified on disk) tokenizes with padding=True/"max_length" but
# never sets a pad token, which crashes on Llama-2-family tokenizers (no
# default pad token). Rather than editing their source, transparently patch
# AutoTokenizer.from_pretrained to fill in pad_token=eos_token when missing --
# an environment compatibility fix, not a change to their algorithm.
_orig_from_pretrained = transformers.AutoTokenizer.from_pretrained.__func__


def _patched_from_pretrained(cls, *args, **kwargs):
    tok = _orig_from_pretrained(cls, *args, **kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


transformers.AutoTokenizer.from_pretrained = classmethod(_patched_from_pretrained)

from representational_toolkit.analysis import run_feature_analysis  # noqa: E402

RESULTS_DIR = ROOT / "experiments" / "rq2" / "main"
DATA_DIR = ROOT / "data" / "concepts"
OUT_DIR = ROOT / "experiments" / "rq2" / "xu_comparison"
PLOT_DIR = OUT_DIR / "repro_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# (model_id, method, concept) for every base+unlearned checkpoint pair currently
# on disk. Extend this list as more checkpoint pairs become available; the
# per-run metrics/attack results remain valid in experiments/rq2/main/ even
# after a checkpoint itself is cleaned up, only this comparison needs the
# checkpoint pair present locally.
RUNS = [
    ("meta-llama/Llama-2-7b-chat-hf", "RMU", "star_wars"),
]


def load_queries(concept: str, n: int = 20) -> list:
    d = json.loads((DATA_DIR / f"{concept}.json").read_text())
    pairs = d["qa_eval_pairs"][:n]
    return [f"{p['question']}\nAnswer: {p['answer']}" for p in pairs]


def _load_model(path, device="cuda"):
    return (AutoModelForCausalLM
            .from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True)
            .to(device).eval())


def extract_mean_hidden(model, tokenizer, query, layer_idx, device, max_length=128):
    enc = tokenizer(query, return_tensors="pt", padding=True, truncation=True,
                     max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states[layer_idx].float().cpu().numpy()
    return hs.mean(axis=1)


def numeric_pca_similarity(model_ref, model_upd, tokenizer, query, device):
    """Same math as their pca_sim_analysis.run_pca_similarity, but returns values."""
    cfg = model_ref.config
    n = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    layers = list(range(n + 1))
    sims = []
    for L in layers:
        feats_ref = extract_mean_hidden(model_ref, tokenizer, query, L, device)
        feats_upd = extract_mean_hidden(model_upd, tokenizer, query, L, device)
        comp_ref = PCA(n_components=1).fit(feats_ref).components_[0]
        comp_upd = PCA(n_components=1).fit(feats_upd).components_[0]
        cos = (comp_ref @ comp_upd) / (np.linalg.norm(comp_ref) * np.linalg.norm(comp_upd) + 1e-12)
        sims.append(float(cos))
    return sims


def center_gram(K):
    n = K.shape[0]
    u = np.ones((n, n)) / n
    return K - u @ K - K @ u + u @ K @ u


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    Kx, Ky = Xc @ Xc.T, Yc @ Yc.T
    hsic = np.trace(center_gram(Kx) @ center_gram(Ky))
    denom = np.sqrt(np.trace(center_gram(Kx) @ center_gram(Kx)) *
                     np.trace(center_gram(Ky) @ center_gram(Ky)) + 1e-12)
    return float(hsic / denom)


def numeric_cka(model_ref, model_upd, tokenizer, query, device, max_length=128):
    """Same math as their cka_analysis.run_cka_analysis, but returns values.

    Note: an earlier version of this extraction took only token position 0
    (`t[:, 0, :]`) with right-padded, max_length-padded batches -- position 0
    is the same leading/BOS token for every query in the batch, so it has
    near-zero variance across the 20 queries and the CKA Gram matrices
    degenerate (observed: cka=0.000 on every combo where both methods
    otherwise produced valid output). Replaced with an attention-mask-aware
    mean over the real (non-pad) tokens per sequence, matching the
    masked-mean convention used elsewhere for representation comparison.
    """
    enc = tokenizer(query, return_tensors="pt", truncation=True,
                     padding="max_length", max_length=max_length).to(device)
    mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
    denom = mask.sum(dim=1).clamp_min(1e-6)              # [B, 1]

    def extract_acts(model):
        acts = {}
        num_layers = len(model.model.layers)
        for L in range(num_layers):
            buf = []

            def hook(m, i, o, buf=buf):
                t = o[0] if isinstance(o, tuple) else o
                t = t.float()
                pooled = (t * mask).sum(dim=1) / denom     # [B, D] masked mean
                buf.append(pooled.detach().cpu().numpy())

            h = dict(model.named_modules())[f"model.layers.{L}"].register_forward_hook(hook)
            with torch.no_grad():
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            h.remove()
            acts[L] = np.concatenate(buf, axis=0)
        return acts

    ref_acts = extract_acts(model_ref)
    upd_acts = extract_acts(model_upd)
    return {L: linear_cka(ref_acts[L], upd_acts[L]) for L in ref_acts}


def load_attack_scores(model_id, method, concept):
    rid = f"{model_id.replace('/', '__')}__{method}__{concept}"
    a_path = RESULTS_DIR / f"{rid}__attacks.json"
    m_path = RESULTS_DIR / f"{rid}__metrics.json"
    scores = {}
    if a_path.exists():
        a = json.loads(a_path.read_text())["attacks"]
        for k in ("Steering", "ICL", "GCG", "MIA"):
            scores[f"attack_{k}"] = a.get(k, {}).get("score")
    if m_path.exists():
        m = json.loads(m_path.read_text())
        scores["auss_peak_metrics"] = m.get("auss_metrics", {}).get("peak_layer")
    return scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for model_id, method, concept in RUNS:
        rid = f"{model_id.replace('/', '__')}__{method}__{concept}"
        ckpt_dir = RESULTS_DIR / "models" / rid
        if not ckpt_dir.exists():
            print(f"[SKIP] {rid}: no checkpoint on disk")
            continue

        print(f"\n=== {rid} ===")
        query = load_queries(concept)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 1) Faithful reproduction: call their UNMODIFIED plotting functions.
        # Each call loads its own pair of models internally; free GPU memory
        # between calls (gc + empty_cache) since these functions never do so
        # themselves and two 7B models' worth of leftover tensors is enough
        # to OOM the next call on a single 48GB GPU.
        print("  Running their official run_feature_analysis (pca_sim)...")
        try:
            run_feature_analysis(
                feature="pca_sim", model_reference_path=model_id, model_path=str(ckpt_dir),
                query=query, output_path=str(PLOT_DIR / f"{rid}__pca_sim.pdf"), device=device,
            )
        except Exception as e:
            print(f"  [WARN] official pca_sim plot failed: {e}")
        gc.collect()
        torch.cuda.empty_cache()

        print("  Running their official run_feature_analysis (cka)...")
        try:
            run_feature_analysis(
                feature="cka", model_reference_path=model_id, model_path=str(ckpt_dir),
                query=query, output_path=str(PLOT_DIR / f"{rid}__cka.pdf"), device=device,
                batch_size=4, num_batches=5,
            )
        except Exception as e:
            print(f"  [WARN] official cka plot failed: {e}")
        gc.collect()
        torch.cuda.empty_cache()

        # 2) Numeric extraction (identical math, for quantitative comparison).
        # Load models fresh here, after the official calls' models are freed.
        print("  Loading base + unlearned models for numeric extraction...")
        model_ref = _load_model(model_id, device)
        model_upd = _load_model(str(ckpt_dir), device)

        print("  Extracting numeric PCA-similarity...")
        pca_sims = numeric_pca_similarity(model_ref, model_upd, tokenizer, query, device)
        print("  Extracting numeric CKA...")
        cka_vals = numeric_cka(model_ref, model_upd, tokenizer, query, device)

        del model_ref, model_upd
        gc.collect()
        torch.cuda.empty_cache()

        scores = load_attack_scores(model_id, method, concept)
        row = {
            "run_id": rid, "model_id": model_id, "method": method, "concept": concept,
            "pca_sim_mean": float(np.mean(pca_sims)),
            "pca_sim_min": float(np.min(pca_sims)),
            "cka_mean": float(np.mean(list(cka_vals.values()))),
            "cka_min": float(np.min(list(cka_vals.values()))),
            **scores,
        }
        rows.append(row)
        print(f"  pca_sim_mean={row['pca_sim_mean']:.4f}  cka_mean={row['cka_mean']:.4f}  "
              f"attacks={ {k: v for k, v in scores.items() if k.startswith('attack_')} }")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "table_repro_comparison.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nSaved: {out_csv}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
