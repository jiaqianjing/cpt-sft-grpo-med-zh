#!/usr/bin/env bash
# Install the isolated SGLang serving environment used by local teacher distillation.
# Input: network access to package indexes. Output: .venv-sglang. No project data changes.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv-sglang ]; then
  uv venv .venv-sglang --python 3.12
fi
# 0.5.11+ pins CUDA 13 / NCCL 2.28.9, which hangs on this host. 0.5.10 is the
# first Qwen3.6-compatible release and stays on the validated CUDA 12 NCCL stack.
uv pip install --python .venv-sglang/bin/python --prerelease=allow "sglang[all]==0.5.10"
