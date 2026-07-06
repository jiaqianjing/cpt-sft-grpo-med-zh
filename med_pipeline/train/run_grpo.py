"""Train stage: GRPO on verifiable Chinese medical MCQ via TRL GRPOTrainer (+ vLLM rollouts).

CALLING SPEC:
    python -m med_pipeline.train.run_grpo --run-id R4_cpt_sft_grpo --init-model R3_cpt_sft [--dry-run]
      --init-model: checkpoints/ subdir to start from (R3_cpt_sft for the full pipeline;
                    R1_sft for the optional no-CPT control R5).
    Reads paths.grpo_jsonl ({"prompt":[msgs], "answer":letter, "n_options":int}). Reward =
    correctness (exact-match) + format, weighted by cfg.grpo.*. Logs reward/KL/len to W&B.

    Side effects: multi-GPU RL training, vLLM generation, writes checkpoints/<run-id>.
    NOTE: exact vLLM flags vary by installed TRL version — validate at the approval gate.
"""

from __future__ import annotations

import argparse

from configs.loader import load_config, load_secrets
from configs.paths import Paths
from med_pipeline.tools.reward import build_grpo_rewards


def _patch_trl_optional_dep_bug() -> None:
    """Work around a TRL 0.24 bug: `_is_package_available(name, return_version=True)` returns a
    tuple `(False, None)` for missing optional deps, and `is_<pkg>_available()` returns that tuple
    unchanged — a non-empty tuple is truthy, so guards like `if is_vllm_ascend_available():` fire
    and import packages that aren't installed (vllm_ascend, mergekit, ...). Coerce the module-level
    `_<pkg>_available` tuples to plain bools BEFORE trl.trainer.grpo_trainer is imported.
    """
    import trl.import_utils as u
    for name in dir(u):
        if name.startswith("_") and name.endswith("_available"):
            val = getattr(u, name)
            if isinstance(val, tuple):
                setattr(u, name, bool(val[0]))


def _anomaly_callback():
    """LOD Pattern 9: warn on sustained zero-reward (reward collapse / format breakdown)."""
    from transformers import TrainerCallback

    class RewardWatch(TrainerCallback):
        def __init__(self):
            self.zero_streak = 0

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            r = logs.get("reward", logs.get("rewards/correctness", None))
            if r is None:
                return
            self.zero_streak = self.zero_streak + 1 if r <= 1e-6 else 0
            if self.zero_streak >= 20:
                print(f"!! reward ~0 for {self.zero_streak} logs (step {state.global_step}) — "
                      f"check prompt/format; base may not emit \\boxed{{}} yet.")

    return RewardWatch()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--init-model", required=True, help="checkpoints/ subdir to start from")
    ap.add_argument("--no-vllm", action="store_true", help="use HF generate instead of vLLM rollouts")
    ap.add_argument("--limit", type=int, default=None, help="subset GRPO prompts (for a shorter run)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    load_secrets()
    paths = Paths(cfg.root)
    init = cfg.model.base_model_id if args.init_model == "base" else str(paths.ckpt(args.init_model))
    out = str(paths.ckpt(args.run_id))
    print(f">> GRPO {args.run_id}: init={init} -> {out}  data={paths.grpo_jsonl}")

    if args.dry_run:
        print("[dry-run] validated GRPO config; not launching training.")
        return 0

    import os
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)

    _patch_trl_optional_dep_bug()  # must run before importing the GRPO trainer
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    use_vllm = not args.no_vllm

    ds = load_dataset("json", data_files=str(paths.grpo_jsonl), split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"   GRPO dataset: {len(ds)} prompts")
    rewards = build_grpo_rewards(cfg.grpo.reward_correct_weight, cfg.grpo.reward_format_weight)

    grpo_args = GRPOConfig(
        output_dir=out,
        run_name=args.run_id,
        learning_rate=cfg.grpo.learning_rate,
        num_train_epochs=cfg.grpo.num_train_epochs,
        num_generations=cfg.grpo.num_generations,
        per_device_train_batch_size=cfg.grpo.per_device_batch,
        gradient_accumulation_steps=cfg.grpo.grad_accum,
        max_prompt_length=cfg.grpo.max_prompt_len,
        max_completion_length=cfg.grpo.max_completion_len,
        temperature=cfg.grpo.temperature,
        beta=cfg.grpo.beta_kl,
        bf16=True,
        gradient_checkpointing=True,  # cut activation memory (GRPO backward over G generations)
        use_vllm=use_vllm,
        vllm_mode="colocate",  # run vLLM in-process on training GPUs (default "server" hangs w/o a server)
        vllm_gpu_memory_utilization=cfg.grpo.vllm_gpu_frac,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
    )
    trainer = GRPOTrainer(
        model=init,
        reward_funcs=rewards,
        args=grpo_args,
        train_dataset=ds,
        callbacks=[_anomaly_callback()],
    )
    trainer.train()
    trainer.save_model(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
