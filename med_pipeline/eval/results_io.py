"""Eval-results persistence — one flat json per (run, kind) under data/eval/.

TOOL CONTRACT:
    save_metrics(paths, run_id, kind, metrics)   # kind in {"knowledge","generative","perplexity"}
    load_all(paths) -> {run_id: {kind: metrics, ...}}
    Side effects: reads/writes json. Deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

from configs.paths import Paths

KINDS = ("knowledge", "generative", "perplexity")


def save_metrics(paths: Paths, run_id: str, kind: str, metrics: dict) -> Path:
    assert kind in KINDS, kind
    paths.eval_results.mkdir(parents=True, exist_ok=True)
    p = paths.eval_results / f"{run_id}__{kind}.json"
    p.write_text(json.dumps({"run_id": run_id, "kind": kind, "metrics": metrics},
                            ensure_ascii=False, indent=2))
    print(f"   saved {kind} metrics -> {p}")
    return p


def load_all(paths: Paths) -> dict:
    out: dict = {}
    for p in sorted(Path(paths.eval_results).glob("*__*.json")):
        d = json.loads(p.read_text())
        out.setdefault(d["run_id"], {})[d["kind"]] = d["metrics"]
    return out
