#!/bin/bash
# Full unattended pipeline: extraction → retry failed → analysis → GitHub commit
# Safe to run at any time — skips models that already have results.
# Usage: bash run_all.sh
set -euo pipefail

export PATH="$HOME/miniconda3/envs/kg-research/bin:$PATH"
export HF_TOKEN="YOUR_HF_TOKEN"
export GITHUB_TOKEN="YOUR_GITHUB_TOKEN"
DIR="$HOME/AtlasResearch/KnowlageGeometryResearch"
LOG="$DIR/extraction_full.log"

ALL_MODELS=(
  "Qwen__Qwen-7B-Chat"
  "Qwen__Qwen-14B-Chat"
  "Qwen__Qwen2.5-7B-Instruct"
  "Qwen__Qwen2.5-14B-Instruct"
  "google__gemma-2b-it"
  "google__gemma-7b-it"
  "meta-llama__Llama-2-7b-chat-hf"
  "meta-llama__Llama-2-13b-chat-hf"
  "meta-llama__Meta-Llama-3-8B-Instruct"
  "meta-llama__Meta-Llama-3.1-8B-Instruct"
)

log() { echo "[run_all $(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cd "$DIR"

# ── Fix known package issues ──────────────────────────────────────────────────
log "Ensuring tiktoken is installed (required for Qwen-7B/14B-Chat)..."
pip install tiktoken -q
log "Ensuring einops is installed (required for Qwen-7B/14B-Chat model loading)..."
pip install einops -q

log "Patching transformers_stream_generator stub (incompatible with transformers>=4.44)..."
STUB_DIR="$(python -c 'import site; print(site.getsitepackages()[0])')/transformers_stream_generator"
mkdir -p "$STUB_DIR"
python - <<'PYEOF'
import site, os, pathlib
stub_dir = pathlib.Path(site.getsitepackages()[0]) / "transformers_stream_generator"
stub_dir.mkdir(exist_ok=True)

main_py = stub_dir / "main.py"
main_py.write_text(
    "from transformers import PreTrainedModel\n"
    "class StreamGenerationConfig: pass\n"
    "class NewGenerationMixin(PreTrainedModel): pass\n"
    "def init_stream_support(): pass\n"
)

init_py = stub_dir / "__init__.py"
init_py.write_text(
    "from .main import init_stream_support, NewGenerationMixin, StreamGenerationConfig\n"
)
print("transformers_stream_generator stub OK")
PYEOF

# ── Phase 1: main extraction pass (skips completed models) ───────────────────
log "=== PHASE 1: Main extraction pass ==="
python scripts/rq1/rq1_extract.py --hf_token "$HF_TOKEN" 2>&1 | tee -a "$LOG"

# ── Phase 2: retry any still-missing models (handles failures) ───────────────
log "=== PHASE 2: Retry missing models ==="
MISSING=()
for m in "${ALL_MODELS[@]}"; do
  [ ! -f "$DIR/results/${m}.json" ] && MISSING+=("${m//__//}")
done

if [ ${#MISSING[@]} -gt 0 ]; then
  log "Retrying ${#MISSING[@]} model(s): ${MISSING[*]}"
  python scripts/rq1/rq1_extract.py --hf_token "$HF_TOKEN" \
      --models_subset "${MISSING[@]}" 2>&1 | tee -a "$LOG"
else
  log "No missing models — skipping retry phase."
fi

# ── Phase 3: final count check ───────────────────────────────────────────────
DONE=$(ls "$DIR/results/"*.json 2>/dev/null | grep -v gitkeep | wc -l)
log "=== Results: $DONE / 10 models complete ==="

if [ "$DONE" -lt 8 ]; then
  log "WARNING: Only $DONE models completed. Analysis may have low statistical power (need ≥8)."
fi

# ── Phase 4: statistical analysis ────────────────────────────────────────────
log "=== PHASE 3: Running rq1_analyze.py ==="
python scripts/rq1/rq1_analyze.py 2>&1 | tee -a "$LOG"

# ── Phase 5: commit results to GitHub ────────────────────────────────────────
log "=== PHASE 4: Committing results to GitHub ==="
git add results/summary.csv figures/*.pdf 2>/dev/null || true
git diff --cached --quiet && log "Nothing new to commit." || \
  git commit -m "results: RQ1 complete — $DONE/10 models, analysis + figures added" && \
  git push origin main && \
  log "Pushed to GitHub successfully."

log "=== ALL DONE ==="
log "Summary: results/summary.csv | Figures: figures/*.pdf | Log: $LOG"
