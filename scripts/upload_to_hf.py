"""Upload the 4 stage checkpoints to the user's HF account as separate public repos.

Only inference-necessary files are uploaded (no checkpoint-*/ optimizer states). R4 (GRPO) is
saved fp32 by TRL, so it is re-cast to bf16 before upload. Each repo gets an honest model card.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")
from configs.loader import load_secrets

load_secrets()  # export HF_TOKEN from .env

import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

USER = "jiaqianjing"
BASE = "Qwen/Qwen3-1.7B-Base"
api = HfApi()

ALLOW = [
    "config.json", "generation_config.json",
    "model.safetensors", "model-*-of-*.safetensors", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
    "README.md",
]

# (src dir, repo, stage label, one-line desc, metrics line)
# Weights are the local-Qwen-teacher *b matrix (SFT CoT distilled by Qwen3.6-27B); see the A/B
# in the project README. CPT is teacher-independent (same checkpoint as before; only its card refreshes).
MODELS = [
    ("checkpoints/cpt", f"{USER}/qwen3-1.7b-med-zh-cpt", "CPT",
     "Continued pre-training on ~0.9B tokens of Chinese medical text (TCM-heavy).",
     "held-out perplexity 7.04 → **5.79** (−1.25 vs base); teacher-independent."),
    ("checkpoints/R1b_sft", f"{USER}/qwen3-1.7b-med-zh-sft", "SFT",
     "Supervised fine-tuning on 60k Chinese medical CoT (local Qwen3.6-27B distillation, from base).",
     "generative MCQ acc 0.589 → **0.646** (+0.057); knowledge 0.661 → 0.638."),
    ("checkpoints/R3b_cpt_sft", f"{USER}/qwen3-1.7b-med-zh-cpt-sft", "CPT→SFT",
     "CPT checkpoint then SFT on the same 60k medical CoT (local-Qwen teacher).",
     "generative 0.627; knowledge 0.626 (CPT perplexity gain retained)."),
    ("checkpoints/R4b_cpt_sft_grpo", f"{USER}/qwen3-1.7b-med-zh-grpo", "CPT→SFT→GRPO",
     "Full pipeline: GRPO with verifiable MCQ reward (exact-match + format) on top of CPT→SFT.",
     "generative **0.654**; knowledge 0.635; GRPO train reward 0.58 → 0.81."),
]

CARD = """---
license: apache-2.0
base_model: {base}
language: [zh]
tags: [medical, chinese, qwen3, {tag}]
---

# Qwen3-1.7B 中文医学 · {stage}

{desc}

本模型是 **CPT → SFT → GRPO 增益归因实验**中的 `{stage}` 阶段权重（基座 [{base}](https://huggingface.co/{base})）。
SFT 的推理链由本地 `Qwen/Qwen3.6-27B` 蒸馏（简洁、thinking-off、`\\boxed{{}}` CoT、质量门 + 拒绝采样）。

## 本阶段评测
- {metrics}

## 实验总览（同一固定评测，Qwen3-1.7B · 本地 Qwen3.6-27B 蒸馏教师）
| 阶段 | 生成准确率 | 知识准确率 | 困惑度 |
|---|---|---|---|
| base | 0.589 | 0.661 | 7.04 |
| CPT | — | 0.636 | 5.79 |
| SFT | 0.646 | 0.638 | — |
| CPT+SFT | 0.627 | 0.626 | — |
| +GRPO | 0.654 | 0.635 | — |

**结论**：把 SFT 蒸馏教师从 Gemini flash-lite 换成本地 `Qwen/Qwen3.6-27B` 后，SFT 生成增益由
**−0.045 翻正为 +0.057**，全链路生成准确率 0.589 → **0.654**——证实瓶颈是**蒸馏数据质量**而非
管线设计。CPT 困惑度增益（−1.25）与教师无关。完整 A/B 对比与报告见
[GitHub 项目](https://github.com/jiaqianjing/cpt-sft-grpo-med-zh)（`README.md` / `REPORT.md`）。

## 用法
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("{repo}", torch_dtype="bfloat16", device_map="auto")
t = AutoTokenizer.from_pretrained("{repo}")
```
"""


def main() -> int:
    for src, repo, stage, desc, metrics in MODELS:
        print(f"\n=== {repo} ({stage}) ===")
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        upload_dir = src
        if repo.endswith("-grpo"):  # R4 is fp32 -> re-cast to bf16
            upload_dir = "checkpoints/_stage_grpo_bf16"
            print("  re-casting fp32 -> bf16 ...")
            m = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)
            m.save_pretrained(upload_dir, safe_serialization=True)
            AutoTokenizer.from_pretrained(src).save_pretrained(upload_dir)
            del m
        tag = repo.rsplit("-", 1)[-1]
        with open(f"{upload_dir}/README.md", "w") as f:
            f.write(CARD.format(base=BASE, tag=tag, stage=stage, desc=desc, metrics=metrics, repo=repo))
        api.upload_folder(folder_path=upload_dir, repo_id=repo, allow_patterns=ALLOW)
        print(f"  DONE -> https://huggingface.co/{repo}")
    print("\nALL UPLOADS COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
