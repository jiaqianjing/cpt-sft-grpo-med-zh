#!/usr/bin/env bash
# Local-Qwen-teacher post-CPT matrix (A/B vs the Gemini baseline).
# Trains fresh *b run-ids from the re-merged SFT set (open + local-Qwen distilled) WITHOUT
# touching the published Gemini checkpoints/evals (R1_sft/R3_cpt_sft/R4_cpt_sft_grpo).
# R0_base / R2_cpt are teacher-independent, so their existing evals are reused as-is.
#   R1b SFT(base) -> R3b SFT(cpt) -> evals(R1b/R3b) -> R4b GRPO(R3b) -> eval(R4b) -> waterfall
# Assumes checkpoints/cpt exists and data/sft/sft_train.jsonl is the re-merged local-Qwen set.
# Exits on a TRAINING failure; continues past individual eval failures.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

[ -d checkpoints/cpt ]            || { log "FATAL: checkpoints/cpt missing (CPT not done)"; exit 1; }
[ -f data/sft/sft_train.jsonl ]  || { log "FATAL: data/sft/sft_train.jsonl missing (run sft_build --merge)"; exit 1; }

# ---------- R1b + R3b SFT (.venv-train, 8 GPU) ----------
source .venv-train/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1
log "R1b SFT (from base, local-Qwen SFT data)"
python -m med_pipeline.train.run_sft --run-id R1b_sft     --init-model base || { log "R1b SFT FAILED"; exit 1; }
log "R3b SFT (from cpt, local-Qwen SFT data)"
python -m med_pipeline.train.run_sft --run-id R3b_cpt_sft --init-model cpt  || { log "R3b SFT FAILED"; exit 1; }

# ---------- Evals for R1b / R3b (.venv, parallel across GPUs) ----------
source .venv/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1 \
       VLLM_LOGGING_LEVEL=WARNING WANDB_SILENT=true
E="python -m med_pipeline.eval"
log "evals R1b/R3b (knowledge + generative, parallel)"
CUDA_VISIBLE_DEVICES=0 $E.run_eval      --run-id R1b_sft     --model-path checkpoints/R1b_sft         > logs/ev_R1b_know.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $E.run_eval      --run-id R3b_cpt_sft --model-path checkpoints/R3b_cpt_sft     > logs/ev_R3b_know.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $E.generate_eval --run-id R1b_sft     --model-path checkpoints/R1b_sft --tp 1  > logs/ev_R1b_gen.log  2>&1 &
CUDA_VISIBLE_DEVICES=3 $E.generate_eval --run-id R3b_cpt_sft --model-path checkpoints/R3b_cpt_sft --tp 1 > logs/ev_R3b_gen.log 2>&1 &
wait
log "R1b/R3b evals done (failures, if any, are non-fatal)"

# ---------- R4b GRPO (.venv-grpo, accelerate DDP on 8 GPU, colocate vLLM) ----------
source .venv-grpo/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
log "R4b GRPO (from R3b_cpt_sft, 12k prompts)"
accelerate launch --num_processes 8 --multi_gpu -m med_pipeline.train.run_grpo \
    --run-id R4b_cpt_sft_grpo --init-model R3b_cpt_sft --limit 12000 || { log "R4b GRPO FAILED"; exit 1; }

# ---------- R4b eval + A/B waterfall (.venv) ----------
source .venv/bin/activate
export PYTHONPATH="$ROOT" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1 \
       VLLM_LOGGING_LEVEL=WARNING WANDB_SILENT=true
log "R4b eval (knowledge + generative)"
CUDA_VISIBLE_DEVICES=0 $E.run_eval      --run-id R4b_cpt_sft_grpo --model-path checkpoints/R4b_cpt_sft_grpo > logs/ev_R4b_know.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $E.generate_eval --run-id R4b_cpt_sft_grpo --model-path checkpoints/R4b_cpt_sft_grpo --tp 1 > logs/ev_R4b_gen.log 2>&1 &
wait
log "gain waterfall (local-Qwen matrix) -> W&B"
python -m med_pipeline.report.waterfall \
    --runs R0_base,R2_cpt,R1b_sft,R3b_cpt_sft,R4b_cpt_sft_grpo \
    --name gain-waterfall-qwen
log "POST-CPT (local-Qwen) PIPELINE COMPLETE"
