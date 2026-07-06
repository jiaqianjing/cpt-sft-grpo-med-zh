"""GRPO reward functions for verifiable Chinese medical MCQ (LOD Pattern 3: registry).

TOOL CONTRACT:
    Reward fns are TRL-GRPOTrainer compatible:  fn(completions, **kwargs) -> list[float]
    where `completions` is a list of generated strings and dataset columns arrive as kwargs
    (we read `answer` = gold option letters, optional `n_options`). Also independently
    testable with plain lists.

    reward_answer_correct  -> 1.0 if extracted letter == gold else 0.0   (the true signal)
    reward_format          -> 1.0 if output contains exactly one \boxed{letter} else 0.0
    build_grpo_rewards(correct_w, format_w) -> [weighted fns] for GRPOTrainer(reward_funcs=...)

    Deterministic given inputs. No side effects. Pure functions over strings.
"""

from __future__ import annotations

import re
from typing import Callable

from med_pipeline.tools.mcq import extract_answer, normalize_letter

_ONE_BOXED = re.compile(r"\\boxed\{\s*([A-Za-zＡ-Ｚ])\s*\}")


def _coerce_text(c) -> str:
    """TRL passes completions as plain strings OR as [{'role','content'}] for conversational
    datasets. Normalize either to the assistant text."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[-1], dict):
        return str(c[-1].get("content", ""))
    return str(c or "")


def _golds(kwargs: dict) -> list[str | None]:
    raw = kwargs.get("answer") or kwargs.get("gold") or []
    return [normalize_letter(str(g)) if g is not None else None for g in raw]


def _n_options(kwargs: dict, i: int, default: int = 4) -> int:
    n = kwargs.get("n_options")
    if isinstance(n, list):
        return int(n[i]) if i < len(n) and n[i] else default
    return int(n) if n else default


def reward_answer_correct(completions: list, **kwargs) -> list[float]:
    golds = _golds(kwargs)
    out: list[float] = []
    for i, c in enumerate(completions):
        gold = golds[i] if i < len(golds) else None
        pred = extract_answer(_coerce_text(c), _n_options(kwargs, i))
        out.append(1.0 if (gold is not None and pred == gold) else 0.0)
    return out


def reward_format(completions: list, **kwargs) -> list[float]:
    return [1.0 if len(_ONE_BOXED.findall(_coerce_text(c))) == 1 else 0.0 for c in completions]


def build_grpo_rewards(correct_w: float, format_w: float) -> list[Callable]:
    """Return weighted reward fns (named, so TRL logs each separately in W&B)."""

    def correctness(completions, **kwargs):
        return [correct_w * r for r in reward_answer_correct(completions, **kwargs)]

    def formatting(completions, **kwargs):
        return [format_w * r for r in reward_format(completions, **kwargs)]

    correctness.__name__ = "correctness"
    formatting.__name__ = "formatting"
    return [correctness, formatting]


REGISTRY = {
    "answer_correct": reward_answer_correct,
    "format": reward_format,
}
