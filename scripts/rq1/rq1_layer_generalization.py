#!/usr/bin/env python3
"""
RQ1 — Fixed/non-oracle layer generalization check.

Peak-layer selection (searching all layers per model for the one maximizing
HP-vs-TOFU separation) isn't practical at deployment time, since a genuinely
unknown domain has no contrastive labels to search against. This checks two non-oracle alternatives against the oracle (per-model peak
search) baseline, using the fresh 9-model re-extraction (all layers):

  1. Fixed relative-depth rule: pick ONE relative depth r = layer/n_layers
     (shared across all models, which differ in total depth: 19-41 layers)
     and evaluate discriminability at that r for every model -- no per-model
     search at all.
  2. Leave-one-model-out transfer: for each held-out model, take the median
     oracle peak relative-depth from the OTHER 8 models, map it to the held-out
     model's own depth, and evaluate discriminability there -- i.e. "if you'd
     never seen this model's contrastive labels, how well would the layer
     choice learned from other models transfer?"

Input:  experiments/rq1/main/*__vectors.npz (9 models)
Output: experiments/rq1/layer_selection/table_layer_generalization.csv (fixed-r sweep)
        experiments/rq1/layer_selection/table_layer_lomo_transfer.csv (LOMO transfer)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.vectors import compute_per_batch_directions
from src.metrics import compute_all_metrics

RESULTS_DIR = ROOT / "experiments" / "rq1" / "main"
OUT_DIR = ROOT / "experiments/rq1/layer_selection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig",
    "AUSS_L2", "AUSS_Cos2", "AUSS_Jac", "Batch_Cov_Trace", "Batch_Cov_TopEig",
]
COHERENCE = {"Centroid_Norm", "Global_Coherence", "Batch_SecondMoment_TopEig"}
RELATIVE_DEPTHS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def model_id_from_filename(fname: str) -> str:
    stem = fname.replace("__vectors.npz", "")
    if "__" in stem:
        org, name = stem.split("__", 1)
        return f"{org}/{name}"
    return stem.replace("_", "/", 1)


def metrics_at_layer(reg_all, anon_all, layer, seed=42):
    reg = torch.from_numpy(reg_all[layer].astype(np.float32))
    anon = torch.from_numpy(anon_all[layer].astype(np.float32))
    dirs = compute_per_batch_directions(reg, anon, n_batches=10, seed=seed)
    return compute_all_metrics(dirs)


def main():
    npz_files = sorted(RESULTS_DIR.glob("*__vectors.npz"))
    if not npz_files:
        sys.exit("ERROR: no experiments/rq1/main/*__vectors.npz found.")

    models = {}
    for f in npz_files:
        model_id = model_id_from_filename(f.name)
        d = np.load(f)
        models[model_id] = {
            "hp_reg": d["hp_reg"], "hp_anon": d["hp_anon"],
            "tofu_reg": d["tofu_reg"], "tofu_anon": d["tofu_anon"],
            "n_layers": d["hp_reg"].shape[0],
        }
    print(f"Loaded {len(models)} models.")

    # ── 1. Oracle peak layer per model (per-metric, direction-aware) ────────────
    oracle_peak_layer = {}
    oracle_summary = {}
    for model_id, v in models.items():
        n_layers = v["n_layers"]
        per_layer = {}
        for layer in range(n_layers):
            hp_m = metrics_at_layer(v["hp_reg"], v["hp_anon"], layer)
            tofu_m = metrics_at_layer(v["tofu_reg"], v["tofu_anon"], layer)
            per_layer[layer] = (hp_m, tofu_m)
        # peak layer for Centroid_Norm (primary metric), direction-aware
        best_layer, best_sep = 0, -1e9
        for layer, (hp_m, tofu_m) in per_layer.items():
            sep = hp_m["Centroid_Norm"] - tofu_m["Centroid_Norm"]
            if not np.isnan(sep) and sep > best_sep:
                best_sep, best_layer = sep, layer
        oracle_peak_layer[model_id] = best_layer
        oracle_summary[model_id] = per_layer
        print(f"  {model_id}: n_layers={n_layers}, oracle peak={best_layer} "
              f"(rel_depth={best_layer/n_layers:.2f})")

    # ── 2. Fixed relative-depth sweep (no per-model search at all) ─────────────
    fixed_rows = []
    for r in RELATIVE_DEPTHS:
        per_metric_hp = {m: [] for m in METRICS}
        per_metric_tofu = {m: [] for m in METRICS}
        for model_id, v in models.items():
            n_layers = v["n_layers"]
            layer = min(int(round(r * (n_layers - 1))), n_layers - 1)
            hp_m, tofu_m = oracle_summary[model_id][layer]
            for m in METRICS:
                per_metric_hp[m].append(hp_m[m])
                per_metric_tofu[m].append(tofu_m[m])
        for m in METRICS:
            hp_v = np.array(per_metric_hp[m])
            tofu_v = np.array(per_metric_tofu[m])
            valid = ~(np.isnan(hp_v) | np.isnan(tofu_v))
            if valid.sum() < 3:
                continue
            hv, tv = hp_v[valid], tofu_v[valid]
            diff = hv - tv
            d_eff = diff.mean() / (diff.std(ddof=1) + 1e-12)
            try:
                _, p = stats.wilcoxon(hv, tv, alternative="two-sided")
            except Exception:
                p = float("nan")
            correct_dir = bool(hv.mean() > tv.mean()) if m in COHERENCE else bool(hv.mean() < tv.mean())
            fixed_rows.append({
                "relative_depth": r, "metric": m, "n": int(valid.sum()),
                "mean_hp": hv.mean(), "mean_tofu": tv.mean(),
                "cohens_d": d_eff, "p_wilcoxon": p, "correct_direction": correct_dir,
            })

    fixed_df = pd.DataFrame(fixed_rows)
    fixed_df.to_csv(OUT_DIR / "table_layer_generalization.csv", index=False, float_format="%.4f")
    print(f"\nSaved: {OUT_DIR / 'table_layer_generalization.csv'}")
    print(f"\n{'rel_depth':>10} {'n_correct_dir/8':>16} {'mean_|d|':>10}")
    for r in RELATIVE_DEPTHS:
        sub = fixed_df[fixed_df["relative_depth"] == r]
        n_correct = sub["correct_direction"].sum()
        mean_abs_d = sub["cohens_d"].abs().mean()
        print(f"{r:>10.1f} {n_correct:>13}/{len(sub)} {mean_abs_d:>10.2f}")

    # ── 3. Leave-one-model-out transfer of the oracle relative depth ───────────
    oracle_rel_depths = {mid: oracle_peak_layer[mid] / models[mid]["n_layers"] for mid in models}
    lomo_rows = []
    for held_out in models:
        others_rel_depth = np.median([oracle_rel_depths[m] for m in models if m != held_out])
        n_layers = models[held_out]["n_layers"]
        transfer_layer = min(int(round(others_rel_depth * (n_layers - 1))), n_layers - 1)
        hp_m, tofu_m = oracle_summary[held_out][transfer_layer]
        oracle_layer = oracle_peak_layer[held_out]
        hp_oracle, tofu_oracle = oracle_summary[held_out][oracle_layer]
        row = {
            "held_out_model": held_out, "n_layers": n_layers,
            "transfer_layer": transfer_layer, "transfer_rel_depth": others_rel_depth,
            "oracle_layer": oracle_layer, "oracle_rel_depth": oracle_layer / n_layers,
        }
        for m in METRICS:
            correct_transfer = bool(hp_m[m] > tofu_m[m]) if m in COHERENCE else bool(hp_m[m] < tofu_m[m])
            correct_oracle = bool(hp_oracle[m] > tofu_oracle[m]) if m in COHERENCE else bool(hp_oracle[m] < tofu_oracle[m])
            row[f"{m}_correct_transfer"] = correct_transfer
            row[f"{m}_correct_oracle"] = correct_oracle
        lomo_rows.append(row)

    lomo_df = pd.DataFrame(lomo_rows)
    lomo_df.to_csv(OUT_DIR / "table_layer_lomo_transfer.csv", index=False, float_format="%.4f")
    print(f"\nSaved: {OUT_DIR / 'table_layer_lomo_transfer.csv'}")

    n_transfer_correct = sum(lomo_df[f"{m}_correct_transfer"].sum() for m in METRICS)
    n_oracle_correct = sum(lomo_df[f"{m}_correct_oracle"].sum() for m in METRICS)
    total = len(models) * len(METRICS)
    print(f"\nLOMO transfer (median other-models' relative depth): "
          f"{n_transfer_correct}/{total} (model, metric) pairs correct direction")
    print(f"Oracle (per-model peak search): {n_oracle_correct}/{total} (model, metric) pairs correct direction")
    print(f"\nOracle peak relative depths across models: "
          f"{sorted(round(v, 2) for v in oracle_rel_depths.values())}")


if __name__ == "__main__":
    main()
