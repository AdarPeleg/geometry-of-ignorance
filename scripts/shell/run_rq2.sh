#!/usr/bin/env bash
# RQ2 unattended pipeline
# Runs: install deps → build pairs (if missing) → unlearn → attack → analyze → push
#
# Usage:
#   bash run_rq2.sh                           # full run
#   bash run_rq2.sh --smoke                   # smoke test: 2 epochs, GradAscent only, HP only
#   bash run_rq2.sh --skip_unlearn            # skip unlearning (attack + analyze only)
#   bash run_rq2.sh --skip_attack             # unlearn only
#
# Env:
#   HF_TOKEN  — set in environment or sourced from .env

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SMOKE=0
SKIP_UNLEARN=0
SKIP_ATTACK=0
SKIP_PUSH=0
EPOCHS=10
METHODS_ARG=""
CONCEPTS_ARG=""
MODELS_ARG=""

for arg in "$@"; do
    case $arg in
        --smoke)        SMOKE=1; EPOCHS=2; METHODS_ARG="--methods GradAscent"; CONCEPTS_ARG="--concepts harry_potter" ;;
        --skip_unlearn) SKIP_UNLEARN=1 ;;
        --skip_attack)  SKIP_ATTACK=1 ;;
        --skip_push)    SKIP_PUSH=1 ;;
        --epochs=*)     EPOCHS="${arg#*=}" ;;
        --methods=*)    METHODS_ARG="--methods ${arg#*=}" ;;
        --concepts=*)   CONCEPTS_ARG="--concepts ${arg#*=}" ;;
        --models=*)     MODELS_ARG="--models ${arg#*=}" ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
echo "=== RQ2 Pipeline ==="
date

# Activate conda environment
if ! command -v conda &>/dev/null; then
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
fi
conda activate kg-research

# HF token
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"
echo "HF_TOKEN: ${HF_TOKEN:0:10}..."

# ---------------------------------------------------------------------------
# Step 0: Install extra dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 0: Installing dependencies ---"
pip install nanogcg scikit-learn -q

# Apply transformers_stream_generator stub (safe for models that need it)
pip install tiktoken einops -q
python - <<'PYEOF'
import site, pathlib
stub_dir = pathlib.Path(site.getsitepackages()[0]) / "transformers_stream_generator"
stub_dir.mkdir(exist_ok=True)
(stub_dir / "main.py").write_text(
    "from transformers import PreTrainedModel\n"
    "class StreamGenerationConfig: pass\n"
    "class NewGenerationMixin(PreTrainedModel): pass\n"
    "def init_stream_support(): pass\n"
)
(stub_dir / "__init__.py").write_text(
    "from .main import init_stream_support, NewGenerationMixin, StreamGenerationConfig\n"
)
print("transformers_stream_generator stub: OK")
PYEOF

# ---------------------------------------------------------------------------
# Step 1: Build concept pairs (idempotent — skips if files already exist)
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Building concept pairs ---"
mkdir -p data/concepts results_rq2 figures_rq2

for concept in harry_potter star_wars william_shakespeare; do
    if [ ! -f "data/concepts/${concept}.json" ]; then
        echo "Building: ${concept}"
        python data/build_concept_pairs.py --concept "${concept}"
    else
        echo "Skip (exists): data/concepts/${concept}.json"
    fi
done

# ---------------------------------------------------------------------------
# Step 2: Unlearning grid
# ---------------------------------------------------------------------------
if [ "$SKIP_UNLEARN" -eq 0 ]; then
    echo ""
    echo "--- Step 2: Unlearning grid (epochs=${EPOCHS}) ---"
    # shellcheck disable=SC2086
    python scripts/rq2/rq2_unlearn.py \
        --hf_token "$HF_TOKEN" \
        --epochs "$EPOCHS" \
        $METHODS_ARG \
        $CONCEPTS_ARG \
        $MODELS_ARG \
        2>&1 | tee -a rq2_unlearn.log
    echo "Unlearning phase complete."
else
    echo "--- Step 2: SKIPPED (--skip_unlearn) ---"
fi

# ---------------------------------------------------------------------------
# Step 3: Attack phase
# ---------------------------------------------------------------------------
if [ "$SKIP_ATTACK" -eq 0 ]; then
    echo ""
    echo "--- Step 3: Attack phase ---"
    GCG_STEPS=500
    if [ "$SMOKE" -eq 1 ]; then
        GCG_STEPS=50
    fi
    # shellcheck disable=SC2086
    python scripts/rq2/rq2_attack.py \
        --hf_token "$HF_TOKEN" \
        --gcg_steps "$GCG_STEPS" \
        $METHODS_ARG \
        $CONCEPTS_ARG \
        $MODELS_ARG \
        2>&1 | tee -a rq2_attack.log
    echo "Attack phase complete."
else
    echo "--- Step 3: SKIPPED (--skip_attack) ---"
fi

# ---------------------------------------------------------------------------
# Step 4: Analysis
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Analysis ---"
python scripts/rq2/rq2_analyze.py --min_runs 1 2>&1 | tee rq2_analyze.log
echo "Analysis complete. Results in results_rq2/"

# ---------------------------------------------------------------------------
# Step 5: Git commit and push
# ---------------------------------------------------------------------------
if [ "$SKIP_PUSH" -eq 0 ]; then
    echo ""
    echo "--- Step 5: Git commit and push ---"
    git add \
        results_rq2/rq2_summary.csv \
        results_rq2/rq2_table1.csv \
        results_rq2/rq2_table2_correlations.csv \
        figures_rq2/*.pdf \
        2>/dev/null || true

    N_DONE=$(python -c "
import pathlib, json
done = 0
for p in pathlib.Path('results_rq2').glob('*__metrics.json'):
    done += 1
print(done)
")
    git commit -m "results: RQ2 — ${N_DONE} runs complete, rq2_summary.csv + figures" \
        -m "" \
        2>/dev/null || echo "Nothing new to commit."

    git push origin main
    echo "Pushed to GitHub."
fi

echo ""
echo "=== RQ2 Pipeline Complete ==="
date
