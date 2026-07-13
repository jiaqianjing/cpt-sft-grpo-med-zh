#!/usr/bin/env bash
# Serve Qwen3.6-27B with 8-way tensor parallelism, distill traces, then merge SFT data.
# Inputs: prepared MCQ sources. Output: data/sft/{distilled,sft_train}.jsonl and a server log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export PATH="$PWD/.venv-sglang/bin:$PATH"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv-sglang/bin/python -m sglang.launch_server \
    --model-path Qwen/Qwen3.6-27B \
    --tp-size 8 \
    --context-length 4096 \
    --disable-cuda-graph \
    --reasoning-parser qwen3 \
    --host 127.0.0.1 \
    --port 30000 \
    >logs/sglang-qwen3.6-27b.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

for _ in {1..180}; do
  if curl -fsS http://127.0.0.1:30000/health >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "SGLang server exited; inspect logs/sglang-qwen3.6-27b.log" >&2
    exit 1
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:30000/health >/dev/null

PYTHONPATH="$PWD" .venv/bin/python -m med_pipeline.data.sft_distill
PYTHONPATH="$PWD" .venv/bin/python -m med_pipeline.data.sft_build --merge
