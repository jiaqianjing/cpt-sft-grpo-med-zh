#!/usr/bin/env bash
# Staged environment setup. Data env installs now (cheap, needed to smoke-test data
# pipelines); the training stack installs only with --full (at the approval gate, before
# GPU runs). Uses `uv` for fast, reproducible installs into ./.venv.
#
#   bash setup/install.sh          # data + eval-client env only (no torch/vllm)
#   bash setup/install.sh --full   # + training stack (torch, LLaMA-Factory, TRL, vLLM, lm-eval)
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=".venv"
if [ ! -d "$VENV" ]; then
  echo ">> creating venv at $VENV (python 3.12)"
  uv venv "$VENV" --python 3.12
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">> installing DATA env (datasets, distillation client, dedup, tracking)"
uv pip install \
  "pydantic>=2.6" "PyYAML>=6.0" "python-dotenv>=1.0" "orjson>=3.9" "tqdm>=4.66" \
  "datasets>=2.20,<3.0" "huggingface_hub>=0.23" "transformers>=4.51,<5" "tokenizers>=0.19" \
  "pandas>=2.2" "datasketch>=1.6" "regex>=2024.4" \
  "google-genai>=0.3" "wandb>=0.17" "tenacity>=8.2"

if [ "${1:-}" == "--full" ]; then
  echo ">> installing TRAINING stack (heavy — only run at the approval gate)"
  # torch first (cu12 wheels; driver 580 / CUDA 12.9 supports these)
  uv pip install "torch>=2.4" --index-url https://download.pytorch.org/whl/cu124 || \
    uv pip install "torch>=2.4"
  uv pip install \
    "accelerate>=0.33" "peft>=0.12" "deepspeed>=0.14" "trl>=0.9" \
    "vllm>=0.5" "lm-eval>=0.4.3"
  # LLaMA-Factory from source (CPT `pt` + SFT `sft` stages)
  if [ ! -d "third_party/LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory third_party/LLaMA-Factory
  fi
  uv pip install -e "third_party/LLaMA-Factory[metrics]"
  # flash-attn last (compiles; can be skipped -> attn_impl=sdpa)
  uv pip install "flash-attn>=2.5" --no-build-isolation || \
    echo "!! flash-attn build failed — set model.attn_impl=sdpa in config"
  # re-pin datasets<3.0 in case the training stack bumped it (script-based CN datasets need it)
  uv pip install "datasets>=2.20,<3.0"
fi

echo ">> done. Activate with: source $VENV/bin/activate"
