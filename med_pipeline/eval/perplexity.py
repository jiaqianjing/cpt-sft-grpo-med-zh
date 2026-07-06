"""Eval: held-out perplexity on the medical corpus — the GUARANTEED CPT signal.

CALLING SPEC:
    python -m med_pipeline.eval.perplexity --run-id R2_cpt --model-path checkpoints/cpt [--base]
    Computes token-level perplexity over paths.cpt_holdout (docs truncated to max_seq_len).
    CPT is guaranteed to lower this vs the base model, so it isolates the CPT effect even when
    downstream accuracy barely moves. Saves + logs to W&B.

    Side effects: loads the model on GPU; reads holdout jsonl; writes results json + W&B.
"""

from __future__ import annotations

import argparse
import math

from configs.loader import load_config, load_secrets
from configs.paths import Paths
from med_pipeline.data.io_jsonl import read_jsonl
from med_pipeline.eval.results_io import save_metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--max-docs", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    load_secrets()
    paths = Paths(cfg.root)
    model_id = cfg.model.base_model_id if args.base else args.model_path
    assert model_id, "provide --model-path or --base"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=cfg.model.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=cfg.model.trust_remote_code,
    ).eval()

    total_nll, total_tok = 0.0, 0
    seq_len = cfg.model.max_seq_len
    for i, rec in enumerate(read_jsonl(paths.cpt_holdout)):
        if i >= args.max_docs:
            break
        ids = tok(rec.get("text", ""), return_tensors="pt", truncation=True,
                  max_length=seq_len)["input_ids"].to(model.device)
        if ids.shape[1] < 2:
            continue
        with torch.no_grad():
            out = model(ids, labels=ids)
        n = ids.shape[1] - 1
        total_nll += out.loss.item() * n  # HF loss is mean NLL over the n shifted tokens
        total_tok += n

    ppl = math.exp(total_nll / total_tok) if total_tok else float("nan")
    metrics = {"perplexity": ppl, "tokens": total_tok}
    save_metrics(paths, args.run_id, "perplexity", metrics)

    import wandb
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                     name=f"eval-{args.run_id}", job_type="eval", reinit=True)
    run.log({"perplexity/holdout": ppl})
    run.finish()
    print(f"holdout perplexity = {ppl:.3f} over {total_tok} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
