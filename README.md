# The Geometry of Ignorance

Reproducibility code for **"The Geometry of Ignorance: Detecting Suppressed Knowledge in
LLMs through Activation Fragmentation."**

This repository contains all code, data, and precomputed results needed to reproduce
the experiments in the paper. The paper studies whether geometric metrics derived from
hidden-state representations of language models can predict how thoroughly a concept
has been unlearned, and how easily that unlearned knowledge can be recovered by an
adversary, all without generating a single token.

---

## Citation

TODO: fill in venue, year, and paper URL/DOI once public.

```bibtex
@inproceedings{geometry_of_ignorance,
  title     = {The Geometry of Ignorance: Detecting Suppressed Knowledge in LLMs
               through Activation Fragmentation},
  author    = {Peleg, Adar and Alsheich, Dvir and Ashuach, Tomer and Mendelson, Avi},
  year      = {TODO},
  booktitle = {TODO},
  note      = {Adar Peleg and Dvir Alsheich contributed equally.}
}
```

---

## Hardware Requirements

- **GPUs:** 2 x GPU with >= 40 GB VRAM each (e.g., 2x L40S, 2x A100-40GB)
- **RAM:** >= 64 GB system RAM
- **Disk:** >= 100 GB free (model checkpoints during RQ2 are ~7-14 GB each)

The extraction scripts use `device_map="auto"` and distribute large models (13-14B)
across both GPUs automatically.

---

## Software Requirements

- conda (Miniconda or Anaconda)
- Python 3.10

```bash
conda create -n kg-research python=3.10 -y
conda activate kg-research
pip install -r requirements.txt
pip install nanogcg scikit-learn   # required for RQ2 attack phase and classifier validation
```

---

## HuggingFace Token

Six of the ten RQ1 models (Gemma, Llama-2, Llama-3 families) require accepting a license
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
data/                          QA pair datasets and build scripts
  hp_pairs.json                 Harry Potter forget/retain pairs (from MUSE-Books)
  hp_retain_pairs.json          Harry Potter retain-split pairs (4-way domain comparison)
  tofu_pairs.json               TOFU fictitious-author pairs
  sw_pairs.json                 Star Wars pairs (4-way domain comparison)
  concepts/                     ConceptVectors-derived Q&A pairs (3 concepts, RQ2)
  anonamized_dataset/           Anonymized Q&A variants for the Activation Steering attack
  build_*.py                    Scripts to rebuild the datasets from scratch
  generate_anon_qa.py           Generates anonymized question variants

src/                            Core library
  model_utils.py                 Model loading, GPU memory management
  vectors.py                     Hidden-state extraction across all layers
  metrics.py                     AUSS geometric metrics (Centroid Norm, AUSS-L2, etc.)
  entropy.py                     Gram-matrix entropy metrics
  attacks.py                     Four attack implementations (Steering, ICL, GCG, MIA)
  unlearn.py                     Unlearning methods (GA, DPO, NPO, NPO+KL, RMU, WHP)
  qa_eval.py                     ROUGE-L QA evaluation utilities
  AttackDetails.md               Full technical reference for all four attacks

scripts/
  rq1/                           RQ1 extraction, analysis, and robustness checks
  rq2/                           RQ2 pipeline, analysis, and robustness checks
  shell/                         Unattended shell wrappers for both pipelines

experiments/                    All result tables, organized to mirror the paper
  rq1/
    main/                        HP vs TOFU discriminability (Table 1)
    multidomain/                 4-way domain comparison (Appendix)
    classifier/                  LOMO-CV classifier validation (Appendix)
    layer_selection/             Layer-generalization + 0.8x-depth heuristic (Appendix)
    batch_sensitivity/           Batch-size sensitivity sweep (Appendix)
  rq2/
    main/                        RQ2 grid: per-run metrics, Table 7
    correlations/                Full geometry-vs-attack correlation grid (Table 2)
    layer_sweep/                 Layer-resolved sweep + BH correction (Appendix)
    xu_comparison/               Head-to-head benchmark vs. Xu et al. (2025) (Appendix)

figures/                        RQ1 plots (main-result plots + supplementary
                                 residual-geometry visualizations)
figures_rq2/                    RQ2 plots
```

---

## Reproducing RQ1

**Research Question 1:** Do AUSS geometric metrics correctly distinguish known-domain
concepts (Harry Potter) from unknown-domain concepts (TOFU fictitious authors)?

### Main result: HP vs TOFU discriminability (Table 1)

```bash
# Build datasets (skip if using the provided data/*.json)
python data/build_hp_pairs.py        # data/hp_pairs.json
python data/build_tofu_pairs.py      # data/tofu_pairs.json

# Extract hidden-state metrics (~2.5 hours, 2x GPU, crash-safe/resumable)
python scripts/rq1/rq1_extract.py --hf_token $HF_TOKEN
python scripts/rq1/rq1_extract_entropy.py --hf_token $HF_TOKEN

# Statistical analysis and figures
python scripts/rq1/rq1_analyze.py
python scripts/rq1/rq1_analyze_entropy.py
```

Outputs: `experiments/rq1/main/summary.csv`, `experiments/rq1/main/entropy_summary.csv`,
`figures/hp_vs_tofu_boxplot.pdf`, `figures/effect_size_bar.pdf`.

A trivial raw-cosine baseline (no anon-vector subtraction) used for comparison in Table 1
is produced by:

```bash
python scripts/rq1/rq1_baseline_comparison.py
```

The BH-corrected, direction-verified version of Table 1 (`experiments/rq1/main/
table_fresh_hp_tofu_discriminability.csv`) is produced from a fresh 9-model re-extraction
(`experiments/rq1/main/summary_fresh.csv`, provided pre-computed and matching the paper's
numbers; reproducible by re-running the extraction pipeline above):

```bash
python scripts/rq1/rq1_fresh_discriminability.py
```

### Robustness & appendix analyses

#### 4-way domain comparison (Harry Potter forget/retain, Star Wars, TOFU)

```bash
python scripts/rq1/rq1_hp_retain_extract.py --hf_token $HF_TOKEN
python scripts/rq1/rq1_extract_sw.py --hf_token $HF_TOKEN
python scripts/rq1/rq1_full_comparison_fresh.py
```

Output: `experiments/rq1/multidomain/table_full_comparison_fresh.csv`,
`experiments/rq1/multidomain/table_full_discriminability_fresh.csv`.

#### Classifier validation (leave-one-model-out)

```bash
python scripts/rq1/rq1_classifier_discriminability.py
```

Output: `experiments/rq1/classifier/table_classifier_discriminability.csv` (Logistic
Regression / MLP accuracy per model).

#### Layer-selection heuristic

```bash
python scripts/rq1/rq1_layer_generalization.py
```

Output: `experiments/rq1/layer_selection/table_layer_generalization.csv` (fixed relative-depth
sweep) and `table_layer_lomo_transfer.csv` (leave-one-model-out transfer).

#### Batch-size sensitivity

```bash
python scripts/rq1/rq1_batch_sensitivity.py
```

Output: `experiments/rq1/batch_sensitivity/table_batch_sensitivity.csv`.

### Supplementary: residual-geometry visualizations

Not tied to a specific numbered paper figure, but useful for inspecting the underlying
geometry directly (PCA / sphere / t-SNE / UMAP projections of the anon-difference vectors).

```bash
python scripts/rq1/rq1_extract_vectors.py --hf_token $HF_TOKEN
python scripts/rq1/rq1_pca_vectors.py
python scripts/rq1/rq1_plot_residual.py
python scripts/rq1/rq1_plot_sphere.py
```

Output: `figures/residual_pca_*.pdf`, `figures/residual_sphere_grid.pdf`,
`figures/residual_tsne_grid.pdf`, `figures/residual_umap_grid.pdf`.

---

## Reproducing RQ2

**Research Question 2:** Do AUSS geometric metrics predict how easily an attacker can
recover information from an unlearned model?

### Grid

- **Models:** Llama-2-7b-chat-hf, Meta-Llama-3-8B-Instruct, Qwen2.5-7B-Instruct
- **Unlearning methods:** GradAscent, DPO, NPO, NPO+KL, RMU (plus Base and WHP controls)
- **Concepts:** harry_potter, star_wars, william_shakespeare
- **Attacks:** Activation Steering (AUC), ICL (Delta), GCG (accuracy), MIA (AUC)

### Main grid

```bash
# Build concept pairs (skip if using the provided data/concepts/*.json)
python data/build_concept_pairs.py --all

# Unified unlearn + attack pipeline (crash-safe/resumable, ~12-24 hours, 2x GPU)
python scripts/rq2/rq2_pipeline.py --hf_token $HF_TOKEN

# Smoke test (2 epochs, GradAscent only, Harry Potter only, ~15 min)
python scripts/rq2/rq2_pipeline.py --hf_token $HF_TOKEN \
    --epochs 2 --methods GradAscent --concepts harry_potter

# Analysis and figures
python scripts/rq2/rq2_analyze.py
python scripts/rq2/generate_bar_chart.py
python scripts/rq2/gen_table7_latex.py
```

Outputs: `experiments/rq2/main/rq2_summary.csv` (all runs), `rq2_table1.csv`,
`rq2_table2_correlations.csv`, `table7_latex_{llama2,llama3,qwen}.tex` (per-model
per-run results table), `figures_rq2/method_comparison_bar.pdf`,
`figures_rq2/metric_vs_recovery_scatter.pdf`, `figures/rq2_bar_chart.pdf`.

### Robustness & appendix analyses

#### Full correlation grid (Table 2)

BH-corrected Spearman correlation between every fragmentation metric and every attack,
across the full run grid.

```bash
python scripts/rq2/table2_5_camera_ready.py
```

Output: `experiments/rq2/correlations/table2_5.csv`.

#### Layer-resolved sweep

Correlation between fragmentation and attack success at every transformer layer
independently, BH-corrected at both per-attack and global scope.

```bash
python scripts/rq2/rq2_layer_entropy_extract.py --hf_token $HF_TOKEN
python scripts/rq2/rq2_layer_analysis.py
python scripts/rq2/layer_sweep_bh.py
python scripts/rq2/gen_layer_figures.py
```

Output: `experiments/rq2/main/rq2_layer_correlations.csv`,
`experiments/rq2/layer_sweep/table6_layer_sweep_bh.csv`,
`figures_rq2/layer_correlation_heatmap.pdf`, `figures_rq2/layer_rho_gcg.pdf`,
`figures_rq2/layer_rho_all_attacks.pdf`.

#### Head-to-head benchmark vs. Xu et al. (2025)

Reproduces Xu et al.'s own PCA-similarity/shift and CKA metrics on the same RQ2
configurations, alongside AUSS. Requires their released toolkit, cloned unmodified
as a sibling directory to this repo:

```bash
cd ..
git clone https://github.com/XiaoyuXU1/Representational_Analysis_Tools.git external/Representational_Analysis_Tools
cd geometry-of-ignorance

python scripts/rq2/repro_representational_tools.py
```

Output: `experiments/rq2/xu_comparison/table_repro_comparison.csv`.

---

## Precomputed Results

All result tables are included in the repository as CSVs (or `.tex` for the per-run
results tables), organized under `experiments/`. Raw per-run JSON files and model
checkpoints are excluded from the repository due to size (see `.gitignore`); re-running
the scripts above regenerates them locally.

---

## Known Compatibility Notes

- **Qwen-7B-Chat / Qwen-14B-Chat** require additional packages: `pip install tiktoken einops`
  and a compatibility stub for `transformers_stream_generator`. Both are applied
  automatically by `scripts/shell/run_all.sh` and `scripts/shell/run_rq2.sh`.
- **Gated models** require accepting the relevant HuggingFace license before the
  token will grant download access.
- `scripts/rq2/rq2_pipeline.py` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  automatically to prevent CUDA memory fragmentation across back-to-back model runs.

---

## References

See `REFERENCES.md` for full citations of all datasets, models, and prior work
used in this paper.
