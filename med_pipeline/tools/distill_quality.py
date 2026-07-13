"""Deterministic quality gate for distilled medical reasoning traces.

TOOL CONTRACT:
    passes_distill_quality(text, max_chars=3000) -> bool

    Input: normalized assistant target containing reasoning and a final boxed answer.
    Output: whether it is concise, predominantly Chinese, and free of leaked meta-reasoning.
    Side effects: none. Deterministic and independently testable.
"""

from __future__ import annotations

from med_pipeline.tools.text_clean import cjk_ratio

_LEAK_MARKERS = (
    "<think>",
    "</think>",
    "here's a thinking process",
    "analyze user input",
    "medical knowledge retrieval",
    "constraints:",
)


def passes_distill_quality(text: str, max_chars: int = 3000) -> bool:
    if not text or len(text) > max_chars:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _LEAK_MARKERS):
        return False
    if text.count("\\boxed{") != 1:
        return False
    return cjk_ratio(text) >= 0.25
