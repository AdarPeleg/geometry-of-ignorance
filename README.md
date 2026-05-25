# The Geometry of Ignorance

Reproducibility code for the EMNLP submission "The Geometry of Ignorance."

This repository contains all code, data, and precomputed results needed to reproduce
the experiments in the paper. The paper studies whether geometric metrics derived from
hidden-state representations of language models can predict how thoroughly a concept
has been unlearned — and how easily that unlearned knowledge can be recovered by an
adversary.

---

## Hardware Requirements

- **GPUs:** 2 × GPU with ≥ 40 GB VRAM each (e.g., 2× L40S, 2× A100-40GB)
- **RAM:** ≥ 64 GB system RAM
- **Disk:** ≥ 100 GB free (model checkpoints during RQ2 are ~7–14 GB each)

The extraction scripts use `device_map="auto"` and distribute large models (13–14B)
across both GPUs automatically.

---

## Software Requirements

- conda (Miniconda or Anaconda)
- Python 3.10

```bash
conda create -n kg-research python=3.10 -y
conda activate kg-research
pip install -r requirements.txt
pip install nanogcg scikit-learn   # required for RQ2 attack phase
```

---

## HuggingFace Token

Six of the ten models (Gemma, Llama-2, Llama-3 families) require accepting a license
on HuggingFace before downloading. After accepting, set your token:

```bash
export HF_TOKEN="<your_hf_token>"
```

Models that require a license agreement:
- `google/gemma-2b-it`, `google/gemma-7b-it`
- `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`
- `meta-llama/Meta-Llama-3-8B-Instruct`, `meta-llama/Meta-Llama-3.1-8B-Instruct`

---

## Repository Structure

```
data/                   QA pair datasets and build scripts
  hp_pairs.json         Harry Potter forget/retain pairs (from MUSE-Books)
  tofu_pairs.json       TOFU fictitious author pairs
  concepts/             ConceptVectors-derived Q&A pairs (3 concepts)
  anonamized_dataset/   Anonymized Q&A variants for Activation Steering attack
  build_*.py            Scripts to rebuild the datasets from scratch
  generate_anon_qa.py   Generates anonymized question variants

src/                    Core library
  model_utils.py        Model loading, GPU memory management
  vectors.py            Hidden-state extraction across all layers
  metrics.py            AUSS geometric metrics (Centroid Norm, AUSS-L2, etc.)
  entropy.py            Token-level entropy metrics
  attacks.py            Four attack implementations (Steering, ICL, GCG, MIA)
  unlearn.py            Unlearning methods (GradAscent, DPO, NPO, RMU, WHP)
  qa_eval.py            ROUGE-L QA evaluation utilities
  AttackDetails.md      Full technical reference for all four attacks

scripts/
  rq1/                  RQ1 extraction and analysis
  rq2/                  RQ2 pipeline and analysis
  shell/                Shell wrappers

results/summary.csv           RQ1 main results (one row per model)
results_v2/entropy_summary.csv  RQ1 entropy results
results_rq2/rq2_summary.csv   RQ2 full results grid
results_rq2/rq2_table1.csv    Paper Table 1 (verification + attacks)
results_rq2/rq2_table2_correlations.csv  Paper Table 2 (Spearman correlations)

figures/                RQ1 analysis plots (PDF + PNG)
figures_rq2/            RQ2 analysis plots (PDF)
```

---

## Reproducing RQ1

**Research Question 1:** Do AUSS geometric metrics correctly distinguish known-domain
concepts (Harry Potter) from unknown-domain concepts (TOFU fictitious authors)?

### Step 1 — Build datasets (skip if using provided data)

```bash
# Requires HuggingFace access; downloads from muse-bench/MUSE-Books and locuslab/TOFU
python data/build_hp_pairs.py       # produces data/hp_pairs.json
python data/build_tofu_pairs.py     # produces data/tofu_pairs.json
```

### Step 2 — Extract hidden-state metrics (≈ 2.5 hours, 2× GPU)

```bash
python scripts/rq1/rq1_extract.py --hf_token $HF_TOKEN
```

This script is **crash-safe**: it skips any model whose output file already exists
in `results_v2/`. Re-run without flags to resume after an interruption.

To also extract per-layer entropy metrics:
```bash
python scripts/rq1/rq1_extract_entropy.py --hf_token $HF_TOKEN
```

### Step 3 — Statistical analysis and figures

```bash
python scripts/rq1/rq1_analyze.py
python scripts/rq1/rq1_analyze_entropy.py
```

### Expected outputs

| File | Description |
|------|-------------|
| `results/summary.csv` | Per-model AUSS metrics + Mann-Whitney U stats |
| `results_v2/entropy_summary.csv` | Per-model entropy metrics |
| `figures/hp_vs_tofu_boxplot.pdf` | Figure 1: HP vs TOFU metric distributions |
| `figures/effect_size_bar.pdf` | Figure 2: Cohen's d effect sizes |
| `figures/residual_pca_combined.pdf` | Figure 3: PCA of residual directions |

---

## Reproducing RQ2

**Research Question 2:** Do AUSS geometric metrics predict how easily an attacker can
recover information from an unlearned model?

### Grid

- **Models:** Llama-2-7b-chat-hf, Meta-Llama-3-8B-Instruct, Qwen2.5-7B-Instruct
- **Unlearning methods:** GradAscent, DPO, NPO, NPO+KL, RMU
- **Concepts:** harry_potter, star_wars, william_shakespeare
- **Attacks:** Activation Steering (AUC), ICL (accuracy), GCG (ASR), MIA (AUC)

### Step 1 — Build concept pairs (skip if using provided data)

```bash
python data/build_concept_pairs.py --all
```

This downloads concept Q&A pairs from `YihuaiHong/ConceptVectors` on HuggingFace.

### Step 2 — Run unified pipeline (unlearn + attack, ≈ 12–24 hours, 2× GPU)

```bash
python scripts/rq2/rq2_pipeline.py --hf_token $HF_TOKEN
```

The pipeline is **crash-safe**: each completed run saves results to
`results_rq2/<model>__<method>__<concept>__metrics.json` and
`results_rq2/<model>__<method>__<concept>__attacks.json`. Re-run to resume.

For a quick smoke test (2 epochs, GradAscent only, Harry Potter only, ≈ 15 min):
```bash
python scripts/rq2/rq2_pipeline.py --hf_token $HF_TOKEN \
    --epochs 2 --methods GradAscent --concepts harry_potter
```

### Step 3 — Analysis and figures

```bash
python scripts/rq2/rq2_analyze.py
python scripts/rq2/rq2_layer_analysis.py
python scripts/rq2/generate_bar_chart.py
```

### Expected outputs

| File | Description |
|------|-------------|
| `results_rq2/rq2_summary.csv` | All runs: metrics, attack scores, verification |
| `results_rq2/rq2_table1.csv` | Table 1: verified runs with attack scores |
| `results_rq2/rq2_table2_correlations.csv` | Table 2: Spearman correlations (metric vs attack) |
| `results_rq2/rq2_layer_correlations.csv` | Layer-by-layer correlation profiles |
| `figures_rq2/method_comparison_bar.pdf` | Figure: method comparison bar chart |
| `figures_rq2/metric_vs_recovery_scatter.pdf` | Figure: metric vs recovery scatter |
| `figures_rq2/layer_correlation_heatmap.pdf` | Figure: layer-correlation heatmap |

---

## Precomputed Results

All results tables and figures are included in the repository. The `results_rq2/`
directory contains only CSV files (raw JSON result files are excluded due to size).
Figures are provided as PDFs in `figures/` and `figures_rq2/`.

---

## Known Compatibility Notes

- **Qwen-7B-Chat / Qwen-14B-Chat** require additional packages: `pip install tiktoken einops`
  and a compatibility stub for `transformers_stream_generator`. Both are applied
  automatically by the pipeline.
- **Gated models** require accepting the relevant HuggingFace license before the
  token will grant download access.
- The pipeline sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically
  to prevent CUDA memory fragmentation across back-to-back model runs.

---

## References

See `REFERENCES.md` for full citations of all datasets, models, and prior work
used in this paper.
