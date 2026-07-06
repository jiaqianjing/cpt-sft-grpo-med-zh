"""Train stage: SFT on Chinese medical CoT via LLaMA-Factory.

CALLING SPEC:
    python -m med_pipeline.train.run_sft --run-id R1_sft      --init-model base   [--dry-run]
    python -m med_pipeline.train.run_sft --run-id R3_cpt_sft  --init-model cpt    [--dry-run]
      --init-model: "base" (cfg.model.base_model_id) or a checkpoint dir name under checkpoints/
                    (e.g. "cpt"). This is what separates R1 (SFT-only) from R3 (CPT->SFT).

    Side effects: registers datasets, launches multi-GPU training, writes checkpoints/<run-id>.
"""

from __future__ import annotations

import argparse
import os

from configs.loader import load_config, load_secrets
from configs.paths import Paths
from med_pipeline.train.lf_common import render_and_launch, write_dataset_info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", required=True, help="e.g. R1_sft or R3_cpt_sft")
    ap.add_argument("--init-model", default="base", help='"base" or a checkpoints/ subdir name')
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    load_secrets()
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)
    paths = Paths(cfg.root)
    write_dataset_info(paths)

    init = cfg.model.base_model_id if args.init_model == "base" else str(paths.ckpt(args.init_model))
    flash = "fa2" if cfg.model.attn_impl == "flash_attention_2" else cfg.model.attn_impl
    overrides = {
        "model_name_or_path": init,
        "trust_remote_code": cfg.model.trust_remote_code,
        "dataset": "sft_zh_medical",
        "output_dir": str(paths.ckpt(args.run_id)),
        "run_name": args.run_id,
        "cutoff_len": cfg.sft.cutoff_len,
        "per_device_train_batch_size": cfg.sft.per_device_batch,
        "gradient_accumulation_steps": cfg.sft.grad_accum,
        "learning_rate": cfg.sft.learning_rate,
        "num_train_epochs": cfg.sft.epochs,
        "warmup_ratio": cfg.sft.warmup_ratio,
        "flash_attn": flash,
    }
    render_and_launch("configs/sft.yaml", overrides, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
