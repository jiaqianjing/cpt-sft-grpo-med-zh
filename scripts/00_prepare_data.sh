#!/usr/bin/env bash
# Prepare CPT / SFT / GRPO data. Slim recipe — all logic lives in med_pipeline.data.*
#   bash scripts/00_prepare_data.sh                 # full prep (distillation stays DRY-RUN)
#   SMOKE=--smoke bash scripts/00_prepare_data.sh   # tiny end-to-end validation
#   DISTILL=1 bash scripts/00_prepare_data.sh       # call an already-running SGLang teacher
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$PWD"
SMOKE="${SMOKE:-}"
DISTILL_ARGS="--dry-run"; [ "${DISTILL:-0}" = "1" ] && DISTILL_ARGS=""

echo "== CPT data =="
python -m med_pipeline.data.cpt_fetch $SMOKE
python -m med_pipeline.data.cpt_dedup --near
python -m med_pipeline.data.cpt_pack

echo "== SFT data =="
python -m med_pipeline.data.sft_build $SMOKE
python -m med_pipeline.data.sft_distill $DISTILL_ARGS $SMOKE
python -m med_pipeline.data.sft_build --merge

echo "== GRPO data =="
python -m med_pipeline.data.grpo_build $SMOKE
echo ">> data prep complete."
