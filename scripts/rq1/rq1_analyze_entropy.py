#!/usr/bin/env python3
"""
RQ1 Entropy Analysis

Aggregates per-model entropy JSONs (produced by rq1_extract_entropy.py) into
experiments/rq1/main/entropy_summary.csv.

For each metric, finds the peak layer = layer that maximises HP-TOFU separation
(TOFU_entropy - HP_entropy, since lower entropy = model knows the concept).

Also merges with experiments/rq1/main/summary.csv to print a unified view.

Usage:
    python rq1_analyze_entropy.py
    python rq1_analyze_entropy.py --models_subset google/gemma-2b-it   # smoke test
    python rq1_analyze_entropy.py --results_dir results --output_csv results/entropy_summary.csv
"""

import csv
import json
import os
import re
import argparse
import math
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_utils import MODEL_REGISTRY


ENTROPY_METRICS = [
    "Reg_S_alpha_2",
    "Reg_S_alpha_1",
    "Reg_EffRank",
    "Reg_MeanL2",
    "Reg_MeanCos",
    "AnonDiff_S_alpha_2",
    "AnonDiff_S_alpha_1",
    "AnonDiff_EffRank",
    "AnonDiff_MeanL2",
    "AnonDiff_MeanCos",
    "NormAnonDiff_S_alpha_2",
    "NormAnonDiff_S_alpha_1",
    "NormAnonDiff_EffRank",
    "NormAnonDiff_MeanL2",
    "NormAnonDiff_MeanCos",
]

# For separation: HP should be LOWER than TOFU for entropy metrics (lower = more compressed)
# HP should have HIGHER MeanL2 and MeanCos for anon-diff (stronger, more consistent shift)
# sep = TOFU - HP for entropy/EffRank (dispersion — higher TOFU = better signal)
# sep = HP - TOFU for MeanL2/MeanCos on AnonDiff (coherence — higher HP = better signal)
COHERENCE_METRICS = {"Reg_MeanCos", "AnonDiff_MeanL2", "AnonDiff_MeanCos",
                     "NormAnonDiff_MeanCos", "NormAnonDiff_MeanL2"}


def safe_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", model_id)


def load_entropy_result(results_dir: str, model_id: str) -> Optional[dict]:
    path = os.path.join(results_dir, f"{safe_model_id(model_id)}_entropy.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def nan_mean(values: List[float]) -> float:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def find_peak_layer(
    model_results: List[dict],  # list of per-model entropy JSONs
    metric_name: str,
    domain: str,  # "hp" or "tofu"
) -> Tuple[int, List[float]]:
    """
    For each layer, collect metric values across all models, return
    (best_peak_layer, per_layer_mean_values).
    """
    # First pass: determine max layers available
    num_layers = min(len(r[f"{domain}_metrics"]) for r in model_results if r[f"{domain}_metrics"])
    if num_layers == 0:
        return 0, []

    per_layer = []
    for layer_idx in range(num_layers):
        vals = []
        for res in model_results:
            layer_data = res[f"{domain}_metrics"]
            if layer_idx < len(layer_data):
                v = layer_data[layer_idx].get(metric_name, float("nan"))
                if v is not None and math.isfinite(v):
                    vals.append(v)
        per_layer.append(nan_mean(vals))

    return per_layer


def compute_separation(hp_per_layer: List[float], tofu_per_layer: List[float], metric_name: str) -> List[float]:
    n = min(len(hp_per_layer), len(tofu_per_layer))
    seps = []
    for i in range(n):
        hp_v, tofu_v = hp_per_layer[i], tofu_per_layer[i]
        if not (math.isfinite(hp_v) and math.isfinite(tofu_v)):
            seps.append(float("-inf"))
            continue
        if metric_name in COHERENCE_METRICS:
            # HP should be higher
            seps.append(hp_v - tofu_v)
        else:
            # Entropy/EffRank: TOFU should be higher
            seps.append(tofu_v - hp_v)
    return seps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="experiments/rq1/main")
    parser.add_argument("--output_csv", default="experiments/rq1/main/entropy_summary.csv")
    parser.add_argument("--models_subset", nargs="+", default=None)
    args = parser.parse_args()

    all_models = list(MODEL_REGISTRY.keys())
    if args.models_subset:
        all_models = [m for m in all_models if m in args.models_subset]

    # Load all entropy results
    model_results = []
    skipped = []
    for model_id in all_models:
        res = load_entropy_result(args.results_dir, model_id)
        if res is None:
            skipped.append(model_id)
            print(f"[SKIP] No entropy result for {model_id}")
        else:
            model_results.append(res)
            print(f"[OK]   Loaded {model_id} ({len(res['hp_metrics'])} layers)")

    if not model_results:
        print("No entropy results found. Run rq1_extract_entropy.py first.")
        return

    if skipped:
        print(f"\nWarning: {len(skipped)} models missing: {skipped}")

    # For each metric, find peak layer (max separation across models)
    print("\n--- Peak Layer Analysis ---")
    peak_layers = {}
    for metric in ENTROPY_METRICS:
        hp_per_layer = compute_separation_layers(model_results, metric, "hp")
        tofu_per_layer = compute_separation_layers(model_results, metric, "tofu")
        seps = compute_separation(hp_per_layer, tofu_per_layer, metric)
        if not seps or all(s == float("-inf") for s in seps):
            peak_layers[metric] = 0
            print(f"  {metric:<30}  peak=N/A (all NaN)")
            continue
        best_layer = int(max(range(len(seps)), key=lambda i: seps[i]))
        peak_layers[metric] = best_layer
        print(f"  {metric:<30}  peak_layer={best_layer}  sep={seps[best_layer]:.4f}")

    # Build summary rows: one per model, peak-layer values for each metric
    print("\n--- Building entropy_summary.csv ---")
    rows = []
    for res in model_results:
        model_id = res["model_id"]
        row = {"model_id": model_id}

        for metric in ENTROPY_METRICS:
            for domain in ("hp", "tofu"):
                peak_layer = peak_layers.get(metric, 0)
                domain_key = f"{domain}_metrics"
                layer_data = res.get(domain_key, [])
                val = float("nan")
                if peak_layer < len(layer_data):
                    val = layer_data[peak_layer].get(metric, float("nan"))
                    if val is None:
                        val = float("nan")
                row[f"{domain}_{metric}"] = round(val, 4) if math.isfinite(val) else "nan"

        rows.append(row)

    # Write CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    fieldnames = ["model_id"] + [
        f"{domain}_{metric}"
        for metric in ENTROPY_METRICS
        for domain in ("hp", "tofu")
    ]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.output_csv}")

    # Print human-readable summary table
    print("\n=== Entropy Summary (peak-layer values) ===")
    KEY_METRICS = ["Reg_S_alpha_2", "AnonDiff_S_alpha_2", "Reg_MeanCos", "AnonDiff_MeanCos"]
    header = f"{'Model':<30}" + "".join(
        f"  HP_{m:<20}  TOFU_{m:<20}" for m in KEY_METRICS
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['model_id']:<30}"
        for m in KEY_METRICS:
            hp_v = row.get(f"hp_{m}", "nan")
            tofu_v = row.get(f"tofu_{m}", "nan")
            line += f"  {str(hp_v):<22}  {str(tofu_v):<22}"
        print(line)

    # Merge with existing summary.csv if present
    existing = os.path.join(args.results_dir, "summary.csv")
    if os.path.exists(existing):
        print(f"\nTip: merge with {existing} for full combined table.")

    print(f"\nWrote {os.path.join(args.results_dir, 'entropy_summary.csv')}")


def compute_separation_layers(model_results, metric, domain):
    """Return per-layer mean metric value across all models."""
    domain_key = f"{domain}_metrics"
    num_layers = min(
        len(r[domain_key]) for r in model_results if r.get(domain_key)
    )
    per_layer = []
    for layer_idx in range(num_layers):
        vals = []
        for res in model_results:
            layer_data = res.get(domain_key, [])
            if layer_idx < len(layer_data):
                v = layer_data[layer_idx].get(metric, float("nan"))
                if v is not None and math.isfinite(v):
                    vals.append(v)
        per_layer.append(nan_mean(vals))
    return per_layer


if __name__ == "__main__":
    main()
