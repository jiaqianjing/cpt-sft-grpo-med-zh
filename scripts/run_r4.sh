#!/usr/bin/env bash
# R4 GRPO (from R3) -> R4 eval -> gain waterfall. Run after R1/R3 SFT + their evals exist.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

source .venv-grpo/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
log "R4 GRPO (from R3_cpt_sft, 12k prompts, 8xGPU colocate)"
accelerate launch --num_processes 8 --multi_gpu -m med_pipeline.train.run_grpo \
    --run-id R4_cpt_sft_grpo --init-model R3_cpt_sft --limit 12000 || { log "R4 GRPO FAILED"; exit 1; }

source .venv/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1 VLLM_LOGGING_LEVEL=WARNING WANDB_SILENT=true
log "R4 eval (knowledge + generative)"
CUDA_VISIBLE_DEVICES=0 python -m med_pipeline.eval.run_eval      --run-id R4_cpt_sft_grpo --model-path checkpoints/R4_cpt_sft_grpo > logs/ev_R4_know.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m med_pipeline.eval.generate_eval --run-id R4_cpt_sft_grpo --model-path checkpoints/R4_cpt_sft_grpo --tp 1 > logs/ev_R4_gen.log 2>&1 &
wait
log "gain waterfall -> W&B"
python -m med_pipeline.report.waterfall
log "R4 PIPELINE COMPLETE"
