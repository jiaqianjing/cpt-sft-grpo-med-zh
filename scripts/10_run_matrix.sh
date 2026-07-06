#!/usr/bin/env bash
# Run the ablation matrix R0..R4 and build the gain waterfall. Slim recipe (LOD Pattern 5).
# Prereq: bash setup/install.sh --full  &&  bash scripts/00_prepare_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$PWD"
E="python -m med_pipeline.eval"; T="python -m med_pipeline.train"

echo "== R0: base baseline =="
$E.run_eval      --run-id R0_base --base
$E.generate_eval --run-id R0_base --base
$E.perplexity    --run-id R0_base --base

echo "== CPT -> checkpoints/cpt (this checkpoint is also run R2) =="
$T.run_cpt
$E.perplexity    --run-id R2_cpt --model-path checkpoints/cpt   # guaranteed CPT signal
$E.run_eval      --run-id R2_cpt --model-path checkpoints/cpt

echo "== R1: SFT-only (from base) =="
$T.run_sft --run-id R1_sft --init-model base
$E.run_eval      --run-id R1_sft --model-path checkpoints/R1_sft
$E.generate_eval --run-id R1_sft --model-path checkpoints/R1_sft

echo "== R3: CPT -> SFT =="
$T.run_sft --run-id R3_cpt_sft --init-model cpt
$E.run_eval      --run-id R3_cpt_sft --model-path checkpoints/R3_cpt_sft
$E.generate_eval --run-id R3_cpt_sft --model-path checkpoints/R3_cpt_sft

echo "== R4: CPT -> SFT -> GRPO =="
$T.run_grpo --run-id R4_cpt_sft_grpo --init-model R3_cpt_sft
$E.run_eval      --run-id R4_cpt_sft_grpo --model-path checkpoints/R4_cpt_sft_grpo
$E.generate_eval --run-id R4_cpt_sft_grpo --model-path checkpoints/R4_cpt_sft_grpo

echo "== Gain attribution waterfall -> W&B =="
python -m med_pipeline.report.waterfall
echo ">> matrix complete. See W&B project for the gain waterfall."
