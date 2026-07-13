"""Repair a checkpoint's tokenizer files across the train/GRPO transformers split.

TOOL CONTRACT:
    sync_compatible_tokenizer(ckpt_dir, base_id) -> bool

    Input:  a checkpoint dir and the base model id whose tokenizer it derives from.
    Output: True if the checkpoint tokenizer was repaired, False if it already loaded.
    Side effects: may overwrite the tokenizer_* files in ckpt_dir (weights untouched).

WHY: the training env (transformers 5.x) saves SFT/CPT checkpoints whose tokenizer_config
uses `extra_special_tokens` as a list, which the GRPO env (transformers 4.57) cannot parse
('list' object has no attribute 'keys') — breaking BOTH AutoTokenizer and the vLLM rollout
engine that loads its own tokenizer from the model dir. SFT/CPT add no tokens, so the base
tokenizer is authoritative: re-save it (written by the running transformers, hence loadable)
while grafting the checkpoint's saved chat template so the prompt format is unchanged.
Idempotent and a no-op when the checkpoint tokenizer already loads.
"""

from __future__ import annotations

from pathlib import Path


def sync_compatible_tokenizer(ckpt_dir: str | Path, base_id: str) -> bool:
    from transformers import AutoTokenizer

    ckpt = Path(ckpt_dir)
    try:
        AutoTokenizer.from_pretrained(ckpt)
        return False  # already loadable in this env; leave it untouched
    except Exception:  # noqa: BLE001 - version-skew tokenizer_config; repair below
        pass

    tok = AutoTokenizer.from_pretrained(base_id)
    chat_template = ckpt / "chat_template.jinja"
    if chat_template.exists():
        tok.chat_template = chat_template.read_text()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(ckpt)
    return True
