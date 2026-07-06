"""LLaMA-Factory glue — register our jsonl datasets + render/launch a train config.

CALLING SPEC:
    from med_pipeline.train.lf_common import write_dataset_info, render_and_launch
    write_dataset_info(paths)                       # -> data/llamafactory/dataset_info.json (+ links)
    render_and_launch("configs/sft.yaml", overrides, dry_run=False)  # -> runs llamafactory-cli

    Side effects: writes dataset_info.json + symlinks under data/llamafactory/; render writes a
    resolved YAML to logs/; launch shells out to `llamafactory-cli train` via torchrun (all GPUs).
    dry_run=True stops after writing the YAML (no training) — used for scaffold validation.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from configs.paths import Paths


def write_dataset_info(paths: Paths) -> Path:
    """Point LLaMA-Factory at our produced jsonl files (CPT text + SFT sharegpt messages)."""
    lf_dir = paths.data / "llamafactory"
    lf_dir.mkdir(parents=True, exist_ok=True)

    def _link(target: Path, name: str) -> None:
        link = lf_dir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        if target.exists():
            link.symlink_to(target.resolve())

    _link(paths.cpt_packed / "train.jsonl", "cpt_train.jsonl")
    _link(paths.sft_jsonl, "sft_train.jsonl")

    info = {
        "cpt_zh_medical": {
            "file_name": "cpt_train.jsonl",
            "columns": {"prompt": "text"},   # pt stage consumes raw text
        },
        "sft_zh_medical": {
            "file_name": "sft_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role", "content_tag": "content",
                "user_tag": "user", "assistant_tag": "assistant", "system_tag": "system",
            },
        },
    }
    info_path = lf_dir / "dataset_info.json"
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info_path


def render_and_launch(template_path: str, overrides: dict, dry_run: bool = False) -> Path:
    """Overlay `overrides` onto a LLaMA-Factory YAML template, write it, and launch training."""
    base = yaml.safe_load(Path(template_path).read_text())
    base.update({k: v for k, v in overrides.items() if v is not None})

    run_name = base.get("run_name", "run")
    out_yaml = Path("logs") / f"lf_{run_name}.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True))
    print(f">> rendered {out_yaml}  (model={base['model_name_or_path']} -> {base['output_dir']})")

    if dry_run:
        print("[dry-run] not launching training.")
        return out_yaml

    env = {**os.environ, "FORCE_TORCHRUN": "1"}  # LLaMA-Factory launches torchrun on all GPUs
    subprocess.run(["llamafactory-cli", "train", str(out_yaml)], check=True, env=env)
    return out_yaml
