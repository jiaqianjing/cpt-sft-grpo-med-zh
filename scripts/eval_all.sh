#!/usr/bin/env bash
# Evaluate all trained checkpoints (R1, R2/cpt, R3) in PARALLEL, one per GPU, using the eval
# env (.venv). R0 was already evaluated for knowledge+generative; here we add R0/R2 perplexity
# and the R1/R2/R3 knowledge + R1/R3 generative measures. R2 (CPT-only) generative is skipped
# (a non-instruction model can't follow the boxed format; not needed for the waterfall).
set -uo pipefail   # NOT -e: let every eval attempt even if one fails
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH="$PWD" HF_HUB_DISABLE_PROGRESS_BARS=1 HF_DATASETS_DISABLE_PROGRESS_BARS=1 \
       VLLM_LOGGING_LEVEL=WARNING WANDB_SILENT=true
E="python -m med_pipeline.eval"

echo "== launching parallel evals @ $(date) =="
CUDA_VISIBLE_DEVICES=0 $E.perplexity    --run-id R0_base    --base                                  > logs/ev_R0_ppl.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 $E.perplexity    --run-id R2_cpt     --model-path checkpoints/cpt            > logs/ev_R2_ppl.log  2>&1 &
CUDA_VISIBLE_DEVICES=2 $E.run_eval      --run-id R1_sft     --model-path checkpoints/R1_sft         > logs/ev_R1_know.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $E.run_eval      --run-id R2_cpt     --model-path checkpoints/cpt            > logs/ev_R2_know.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $E.run_eval      --run-id R3_cpt_sft --model-path checkpoints/R3_cpt_sft     > logs/ev_R3_know.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $E.generate_eval --run-id R1_sft     --model-path checkpoints/R1_sft --tp 1  > logs/ev_R1_gen.log  2>&1 &
CUDA_VISIBLE_DEVICES=6 $E.generate_eval --run-id R3_cpt_sft --model-path checkpoints/R3_cpt_sft --tp 1 > logs/ev_R3_gen.log 2>&1 &
wait
echo "== all evals finished @ $(date) =="
ls -1 data/eval/*.json | sed 's#data/eval/##'
