#!/usr/bin/env bash
# R4b GRPO tail (local-Qwen matrix): resume after R1b/R3b SFT + their evals already exist.
# R4b GRPO (from R3b_cpt_sft) -> R4b eval -> local-Qwen gain waterfall. Mirrors run_r4.sh
# with the *b run-ids. Requires the run_grpo.py robust-tokenizer fix (train/GRPO tf split).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

[ -d checkpoints/R3b_cpt_sft ] || { log "FATAL: checkpoints/R3b_cpt_sft missing (run R3b SFT first)"; exit 1; }

source .venv-grpo/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
log "R4b GRPO (from R3b_cpt_sft, 12k prompts, 8xGPU colocate)"
accelerate launch --num_processes 8 --multi_gpu -m med_pipeline.train.run_grpo \
    --run-id R4b_cpt_sft_grpo --init-model R3b_cpt_sft --limit 12000 || { log "R4b GRPO FAILED"; exit 1; }

source .venv/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1 \
       VLLM_LOGGING_LEVEL=WARNING WANDB_SILENT=true
E="python -m med_pipeline.eval"
log "R4b eval (knowledge + generative)"
CUDA_VISIBLE_DEVICES=0 $E.run_eval      --run-id R4b_cpt_sft_grpo --model-path checkpoints/R4b_cpt_sft_grpo > logs/ev_R4b_know.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $E.generate_eval --run-id R4b_cpt_sft_grpo --model-path checkpoints/R4b_cpt_sft_grpo --tp 1 > logs/ev_R4b_gen.log 2>&1 &
wait
log "gain waterfall (local-Qwen matrix) -> W&B"
python -m med_pipeline.report.waterfall \
    --runs R0_base,R2_cpt,R1b_sft,R3b_cpt_sft,R4b_cpt_sft_grpo \
    --name gain-waterfall-qwen
log "R4b PIPELINE COMPLETE"
