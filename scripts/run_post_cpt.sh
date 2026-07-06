#!/usr/bin/env bash
# Autonomous post-CPT pipeline: R1 SFT -> R3 SFT -> evals(R1/R2/R3) -> R4 GRPO -> R4 eval -> waterfall.
# Assumes checkpoints/cpt exists (CPT done). Uses the 3-env split:
#   training (.venv-train), GRPO (.venv-grpo via accelerate), eval/report (.venv).
# Exits on a TRAINING failure; continues past individual eval failures.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---------- R1 + R3 SFT (.venv-train, 8 GPU) ----------
source .venv-train/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1
log "R1 SFT (from base)"
python -m med_pipeline.train.run_sft --run-id R1_sft     --init-model base || { log "R1 SFT FAILED"; exit 1; }
log "R3 SFT (from cpt)"
python -m med_pipeline.train.run_sft --run-id R3_cpt_sft --init-model cpt  || { log "R3 SFT FAILED"; exit 1; }

# ---------- Evals for R1 / R2(cpt) / R3 (.venv, parallel across GPUs) ----------
log "evals R1/R2/R3 (parallel)"
bash scripts/eval_all.sh || log "some R1/R2/R3 evals failed (continuing)"

# ---------- R4 GRPO (.venv-grpo, accelerate DDP on 8 GPU, colocate vLLM) ----------
source .venv-grpo/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
log "R4 GRPO (from R3, 12k prompts)"
accelerate launch --num_processes 8 --multi_gpu -m med_pipeline.train.run_grpo \
    --run-id R4_cpt_sft_grpo --init-model R3_cpt_sft --limit 12000 || { log "R4 GRPO FAILED"; exit 1; }

# ---------- R4 eval + waterfall (.venv) ----------
source .venv/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1 VLLM_LOGGING_LEVEL=WARNING WANDB_SILENT=true
log "R4 eval (knowledge + generative)"
CUDA_VISIBLE_DEVICES=0 python -m med_pipeline.eval.run_eval      --run-id R4_cpt_sft_grpo --model-path checkpoints/R4_cpt_sft_grpo > logs/ev_R4_know.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m med_pipeline.eval.generate_eval --run-id R4_cpt_sft_grpo --model-path checkpoints/R4_cpt_sft_grpo --tp 1 > logs/ev_R4_gen.log 2>&1 &
wait
log "gain waterfall -> W&B"
python -m med_pipeline.report.waterfall
log "PIPELINE COMPLETE"
