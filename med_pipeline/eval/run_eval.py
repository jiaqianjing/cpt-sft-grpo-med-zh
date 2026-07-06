"""Eval: knowledge probe via lm-evaluation-harness (loglikelihood MCQ).

CALLING SPEC:
    python -m med_pipeline.eval.run_eval --run-id R3_cpt_sft --model-path checkpoints/R3_cpt_sft
    python -m med_pipeline.eval.run_eval --run-id R0_base --base
    Runs cfg.eval.tasks (cmmlu_medical, ceval_medical, cmexam, medqa_zh) with a UNIFORM
    num_fewshot across all runs (so gains are comparable). Custom tasks are loaded from
    med_pipeline/eval/tasks. Saves per-task acc + logs to W&B.

    Side effects: loads the model on GPU, downloads eval datasets, writes results json + W&B.
"""

from __future__ import annotations

import argparse

from configs.loader import load_config, load_secrets
from configs.paths import Paths
from med_pipeline.eval.results_io import save_metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model-path", default=None, help="checkpoint dir; omit with --base")
    ap.add_argument("--base", action="store_true", help="evaluate the base model")
    ap.add_argument("--limit", type=int, default=None, help="override cfg for a smoke run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    load_secrets()
    paths = Paths(cfg.root)
    model = cfg.model.base_model_id if args.base else args.model_path
    assert model, "provide --model-path or --base"

    import wandb
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    tm = TaskManager(include_path=str((paths.root / "med_pipeline/eval/tasks")))
    trc = ",trust_remote_code=True" if cfg.model.trust_remote_code else ""
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model},dtype=bfloat16{trc}",
        tasks=cfg.eval.tasks,
        num_fewshot=cfg.eval.num_fewshot,
        batch_size=cfg.eval.batch_size,
        limit=args.limit if args.limit is not None else cfg.eval.limit,
        task_manager=tm,
    )
    # extract acc per task/group
    metrics = {}
    for task, res in results["results"].items():
        acc = res.get("acc,none", res.get("acc", None))
        if acc is not None:
            metrics[task] = float(acc)
    save_metrics(paths, args.run_id, "knowledge", metrics)

    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                     name=f"eval-{args.run_id}", job_type="eval", reinit=True)
    run.log({f"knowledge/{k}": v for k, v in metrics.items()})
    run.finish()
    print("knowledge metrics:", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
