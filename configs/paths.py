"""Central path registry — derive every filesystem path from one root.

CALLING SPEC:
    from configs.paths import Paths
    p = Paths("/mnt/nvme4/ken/workspace/llm")
    p.cpt_packed          # -> Path to packed CPT dataset dir
    p.sft_jsonl           # -> Path to assembled SFT jsonl
    p.grpo_jsonl          # -> Path to GRPO prompt jsonl
    p.ckpt("R3_cpt_sft")  # -> Path to a run's checkpoint dir
    p.ensure()            # mkdir -p all base dirs (idempotent)

TOOL CONTRACT: pure path construction + one idempotent mkdir helper. No other side effects.
Keeping paths in one place means no script hardcodes a directory string (LOD: explicit,
single-responsibility). Nothing here reads or writes data files.
"""

from __future__ import annotations

from pathlib import Path


class Paths:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    # --- base dirs ---
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # --- CPT ---
    @property
    def cpt_raw(self) -> Path:
        return self.data / "cpt" / "raw.jsonl"          # cleaned, pre-dedup

    @property
    def cpt_dedup(self) -> Path:
        return self.data / "cpt" / "dedup.jsonl"         # after dedup

    @property
    def cpt_packed(self) -> Path:
        return self.data / "cpt" / "packed"              # tokenized/packed train dataset

    @property
    def cpt_holdout(self) -> Path:
        return self.data / "cpt" / "holdout.jsonl"       # perplexity eval shard

    # --- SFT ---
    @property
    def sft_open(self) -> Path:
        return self.data / "sft" / "open.jsonl"          # assembled open datasets

    @property
    def sft_distilled(self) -> Path:
        return self.data / "sft" / "distilled.jsonl"     # Gemini CoT, rejection-sampled

    @property
    def sft_jsonl(self) -> Path:
        return self.data / "sft" / "sft_train.jsonl"     # final merged SFT set

    # --- GRPO ---
    @property
    def grpo_jsonl(self) -> Path:
        return self.data / "grpo" / "grpo_train.jsonl"   # verifiable MCQ prompts

    # --- eval ---
    @property
    def eval_results(self) -> Path:
        return self.data / "eval"                        # per-run lm-eval json dumps

    # --- per-run checkpoints ---
    def ckpt(self, run_id: str) -> Path:
        return self.checkpoints / run_id

    def ensure(self) -> None:
        for d in (
            self.data / "cpt",
            self.data / "sft",
            self.data / "grpo",
            self.eval_results,
            self.checkpoints,
            self.logs,
        ):
            d.mkdir(parents=True, exist_ok=True)
