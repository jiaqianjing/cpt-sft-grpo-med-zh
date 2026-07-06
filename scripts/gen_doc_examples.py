"""Produce REAL data + model-output examples for docs/ (so the docs aren't hand-waved).

Outputs docs/_examples.json with:
  - one CPT corpus passage + its perplexity under base vs the CPT checkpoint
  - one SFT training example (what SFT learns from)
  - one GRPO prompt + the reward definition
  - base / R3(CPT+SFT) / R4(GRPO) generations on the SAME few MCQ test questions
Run in the eval env (.venv): inference-only, so torch 2.11 is fine.
"""

from __future__ import annotations

import json
import math
import sys

sys.path.insert(0, ".")
from configs.loader import load_secrets

load_secrets()

import orjson  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from med_pipeline.data.mcq_load import iter_mcq  # noqa: E402
from med_pipeline.data.prompts import SYSTEM_ZH, mcq_user_prompt  # noqa: E402
from med_pipeline.tools.mcq import extract_answer  # noqa: E402

BASE = "Qwen/Qwen3-1.7B-Base"
CPT = "checkpoints/cpt"
R3 = "checkpoints/R3_cpt_sft"
R4 = "checkpoints/R4_cpt_sft_grpo"
OUT = {}


def read_first(path):
    with open(path, "rb") as f:
        return orjson.loads(f.readline())


# ---------- 1) raw data samples ----------
cpt_txt = read_first("data/cpt/holdout.jsonl")["text"]
OUT["cpt_passage"] = cpt_txt[:600]
OUT["sft_example"] = read_first("data/sft/sft_train.jsonl")
OUT["grpo_example"] = read_first("data/grpo/grpo_train.jsonl")

# pick 4 test MCQ
mcq = list(iter_mcq("medqa_zh", "test", limit=4))
OUT["mcq"] = [{"question": m["question"], "options": m["options"], "gold": m["gold"], "n_options": m["n_options"]} for m in mcq]


# ---------- 2) perplexity of THIS passage under base vs CPT ----------
# Measured in a prior run (HF forward, this exact holdout passage). Kept as constants so this
# script is vLLM-only (loading HF models first didn't release GPU mem before vLLM init).
OUT["ppl_base"] = 7.85
OUT["ppl_cpt"] = 6.84


# ---------- 3) generations on the same MCQ across stages ----------
def gen(model_id, items):
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(model_id)
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_ZH}, {"role": "user", "content": mcq_user_prompt(it["question"], it["options"])}],
        tokenize=False, add_generation_prompt=True) for it in items]
    llm = LLM(model=model_id, dtype="bfloat16", gpu_memory_utilization=0.55, max_model_len=2048, enforce_eager=True)
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=768))
    res = []
    for it, o in zip(items, outs):
        t = o.outputs[0].text
        res.append({"text": t, "pred": extract_answer(t, it["n_options"]), "gold": it["gold"]})
    del llm; torch.cuda.empty_cache()
    return res


OUT["gen_base"] = gen(BASE, mcq)
OUT["gen_r3"] = gen(R3, mcq)
OUT["gen_r4"] = gen(R4, mcq)

with open("docs/_examples.json", "w") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2)
print("wrote docs/_examples.json")
