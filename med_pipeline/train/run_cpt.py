"""Train stage: CPT (continued pre-training) on Chinese medical text via LLaMA-Factory.

CALLING SPEC:
    python -m med_pipeline.train.run_cpt [--config ...] [--dry-run]
    Produces the shared CPT checkpoint at checkpoints/cpt (reused by R3/R4; is itself run R2).
    Hyperparameters come from cfg.cpt; W&B run_name="cpt".

    Side effects: registers datasets, launches multi-GPU training, writes checkpoints/cpt.
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    load_secrets()
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)
    paths = Paths(cfg.root)
    write_dataset_info(paths)

    flash = "fa2" if cfg.model.attn_impl == "flash_attention_2" else cfg.model.attn_impl
    overrides = {
        "model_name_or_path": cfg.model.base_model_id,
        "trust_remote_code": cfg.model.trust_remote_code,
        "dataset": "cpt_zh_medical",
        "output_dir": str(paths.ckpt("cpt")),
        "run_name": "cpt",
        "cutoff_len": cfg.cpt.seq_len,
        "packing": cfg.cpt.packing,
        "per_device_train_batch_size": cfg.cpt.per_device_batch,
        "gradient_accumulation_steps": cfg.cpt.grad_accum,
        "learning_rate": cfg.cpt.learning_rate,
        "num_train_epochs": cfg.cpt.epochs,
        "warmup_ratio": cfg.cpt.warmup_ratio,
        "weight_decay": cfg.cpt.weight_decay,
        "save_steps": cfg.cpt.save_steps,
        "flash_attn": flash,
    }
    render_and_launch("configs/cpt.yaml", overrides, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
