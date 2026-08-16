#!/usr/bin/env bash
# RQ2 unattended pipeline
# Runs: install deps -> build concept pairs (if missing) -> unlearn + attack -> analyze
#
# Usage:
#   bash run_rq2.sh                           # full run
#   bash run_rq2.sh --smoke                   # smoke test: 2 epochs, GradAscent only, HP only
#
# Env:
#   HF_TOKEN  — set in environment or sourced from .env

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DIR"

SMOKE_ARGS=""
for arg in "$@"; do
    case $arg in
        --smoke) SMOKE_ARGS="--epochs 2 --methods GradAscent --concepts harry_potter" ;;
    esac
done

echo "=== RQ2 Pipeline ==="
date

# Activate conda environment
if ! command -v conda &>/dev/null; then
    eval "$(~/miniconda3/bin/conda shell.bash hook)"
fi
conda activate kg-research

export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"
echo "HF_TOKEN: ${HF_TOKEN:0:10}..."

# ---------------------------------------------------------------------------
# Step 0: Install extra dependencies
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 0: Installing dependencies ---"
pip install nanogcg scikit-learn -q
pip install tiktoken einops -q

# Apply transformers_stream_generator stub (needed for Qwen-7B/14B-Chat)
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
mkdir -p data/concepts experiments/rq2/main figures_rq2

for concept in harry_potter star_wars william_shakespeare; do
    if [ ! -f "data/concepts/${concept}.json" ]; then
        echo "Building: ${concept}"
        python data/build_concept_pairs.py --concept "${concept}"
    else
        echo "Skip (exists): data/concepts/${concept}.json"
    fi
done

# ---------------------------------------------------------------------------
# Step 2: Unified unlearn + attack pipeline
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Unlearn + attack pipeline ---"
# shellcheck disable=SC2086
python scripts/rq2/rq2_pipeline.py --hf_token "$HF_TOKEN" $SMOKE_ARGS \
    2>&1 | tee -a rq2_pipeline.log
echo "Pipeline phase complete."

# ---------------------------------------------------------------------------
# Step 3: Analysis
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Analysis ---"
python scripts/rq2/rq2_analyze.py --min_runs 1 2>&1 | tee rq2_analyze.log
echo "Analysis complete. Results in experiments/rq2/main/"

echo ""
echo "=== RQ2 Pipeline Complete ==="
date
