"""Eval: GENERATIVE boxed-answer accuracy on held-out MCQ (mirrors SFT/GRPO objective).

Why this exists: lm-eval (run_eval.py) scores choices by loglikelihood — a knowledge probe that
does NOT test whether the model can GENERATE reasoning + a correct \\boxed{} answer. GRPO (and
SFT's format learning) optimize exactly that generative behavior, so their gains show up here.

CALLING SPEC:
    python -m med_pipeline.eval.generate_eval --run-id R4_cpt_sft_grpo \
        --model-path checkpoints/R4_cpt_sft_grpo [--base] [--limit 500] [--tp 1] [--dry-run]
    Samples greedily on medqa_zh(test) + cmexam(test), extracts \\boxed{} answer, computes acc.
    --dry-run builds prompts on a few items and prints them (no vLLM, no GPU).

    Side effects: loads model via vLLM on GPU; writes results json + W&B.
"""

from __future__ import annotations

import argparse

from configs.loader import load_config, load_secrets
from configs.paths import Paths
from med_pipeline.data.mcq_load import iter_mcq
from med_pipeline.data.prompts import SYSTEM_ZH, mcq_chat, mcq_user_prompt
from med_pipeline.eval.results_io import save_metrics
from med_pipeline.tools.mcq import extract_answer

EVAL_GEN_SOURCES = {"medqa_zh": "test", "cmexam": "test"}  # held-out; never trained on


def _build_prompt(tok, q, options) -> str:
    msgs = mcq_chat(q, options)
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_ZH}\n\n{mcq_user_prompt(q, options)}\n"  # base model fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--limit", type=int, default=1000, help="items per source")
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor_parallel_size")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    load_secrets()
    paths = Paths(cfg.root)
    model = cfg.model.base_model_id if args.base else args.model_path
    assert model, "provide --model-path or --base"

    # collect held-out items
    items: list[dict] = []
    for key, split in EVAL_GEN_SOURCES.items():
        got = list(iter_mcq(key, split, limit=args.limit, single_letter_only=True))
        for it in got:
            it["_source"] = key
        items.extend(got)
        print(f"   {key}({split}): {len(got)} items")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=cfg.model.trust_remote_code)
    prompts = [_build_prompt(tok, it["question"], it["options"]) for it in items]

    if args.dry_run:
        print("\n--- sample prompt ---\n" + prompts[0][:600])
        print(f"\n[dry-run] built {len(prompts)} prompts; no generation.")
        return 0

    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=args.tp,
              gpu_memory_utilization=0.85, trust_remote_code=cfg.model.trust_remote_code)
    sp = SamplingParams(temperature=0.0, max_tokens=cfg.grpo.max_completion_len)
    outs = llm.generate(prompts, sp)
    texts = [o.outputs[0].text for o in outs]

    # per-source accuracy
    correct: dict[str, list[float]] = {k: [] for k in EVAL_GEN_SOURCES}
    for it, text in zip(items, texts):
        pred = extract_answer(text, it["n_options"])
        correct[it["_source"]].append(1.0 if pred == it["gold"] else 0.0)
    metrics = {k: (sum(v) / len(v) if v else 0.0) for k, v in correct.items()}
    metrics["macro_avg"] = sum(metrics[k] for k in EVAL_GEN_SOURCES) / len(EVAL_GEN_SOURCES)
    save_metrics(paths, args.run_id, "generative", metrics)

    import wandb
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                     name=f"eval-{args.run_id}", job_type="eval", reinit=True)
    run.log({f"generative/{k}": v for k, v in metrics.items()})
    run.finish()
    print("generative acc:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
