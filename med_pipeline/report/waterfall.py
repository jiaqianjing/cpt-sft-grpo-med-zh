"""Report: build the CPT/SFT/GRPO gain-attribution waterfall from saved eval results.

CALLING SPEC:
    python -m med_pipeline.report.waterfall [--config ...]
    Reads data/eval/*.json (written by run_eval / generate_eval / perplexity), computes:
        SFT gain  = R1 - R0     CPT gain = R3 - R1     GRPO gain = R4 - R3   (accuracy)
        CPT gain (perplexity)   = PPL(R0) - PPL(R2)                          (reduction)
    on both generative accuracy (primary) and knowledge accuracy, logs a W&B Table + bar charts,
    and prints a text summary. Missing runs are skipped gracefully.

    Side effects: reads json; writes to W&B. Deterministic given inputs.
"""

from __future__ import annotations

import argparse

from configs.loader import load_config, load_secrets
from configs.paths import Paths
from med_pipeline.eval.results_io import load_all

# canonical stage order for the waterfall — these are LOGICAL stage keys used for gain
# formulas and labels. Actual run-ids are looked up via run_map (identity by default; the
# --runs flag remaps stages to alternate run-ids, e.g. the local-Qwen *b matrix for A/B).
ORDER = ["R0_base", "R2_cpt", "R1_sft", "R3_cpt_sft", "R4_cpt_sft_grpo"]


def _gen(run: dict) -> float | None:
    m = run.get("generative")
    return m.get("macro_avg") if m else None


_HEADLINE = ("cmmlu_medical", "ceval_medical", "cmexam", "medqa_zh")


def _know(run: dict) -> float | None:
    m = run.get("knowledge")
    if not m:
        return None
    # average only the 4 headline tasks (the group aggregates + the 2 custom tasks) to avoid
    # double-counting the per-subject cmmlu_*/ceval-valid_* entries that also live in the dict.
    vals = [m[k] for k in _HEADLINE if isinstance(m.get(k), (int, float))]
    if not vals:  # fallback: average whatever is present
        vals = [v for v in m.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def _ppl(run: dict) -> float | None:
    m = run.get("perplexity")
    return m.get("perplexity") if m else None


def _delta(a, b):
    return None if (a is None or b is None) else round(b - a, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--runs", default=None,
                    help="comma-separated actual run-ids for the 5 stages, in ORDER: "
                         "R0_base,R2_cpt,R1_sft,R3_cpt_sft,R4_cpt_sft_grpo. "
                         "Defaults to identity; use to render an alternate matrix (e.g. *b).")
    ap.add_argument("--name", default="gain-waterfall", help="W&B report run name")
    args = ap.parse_args()
    cfg = load_config(args.config)
    load_secrets()
    paths = Paths(cfg.root)
    runs = load_all(paths)

    # map each logical stage to an actual run-id (identity unless --runs overrides)
    if args.runs:
        override = [r.strip() for r in args.runs.split(",")]
        if len(override) != len(ORDER):
            raise SystemExit(f"--runs needs exactly {len(ORDER)} ids in ORDER {ORDER}")
        run_map = dict(zip(ORDER, override))
    else:
        run_map = {stage: stage for stage in ORDER}

    gen = {r: _gen(runs.get(run_map[r], {})) for r in ORDER}
    know = {r: _know(runs.get(run_map[r], {})) for r in ORDER}
    ppl = {r: _ppl(runs.get(run_map[r], {})) for r in ORDER}

    gains = {
        "gen/SFT_gain(R1-R0)": _delta(gen["R0_base"], gen["R1_sft"]),
        "gen/CPT_gain(R3-R1)": _delta(gen["R1_sft"], gen["R3_cpt_sft"]),
        "gen/GRPO_gain(R4-R3)": _delta(gen["R3_cpt_sft"], gen["R4_cpt_sft_grpo"]),
        "know/SFT_gain(R1-R0)": _delta(know["R0_base"], know["R1_sft"]),
        "know/CPT_gain(R3-R1)": _delta(know["R1_sft"], know["R3_cpt_sft"]),
        "know/GRPO_gain(R4-R3)": _delta(know["R3_cpt_sft"], know["R4_cpt_sft_grpo"]),
        "ppl/CPT_reduction(R0-R2)": (
            None if (ppl["R0_base"] is None or ppl["R2_cpt"] is None)
            else round(ppl["R0_base"] - ppl["R2_cpt"], 4)
        ),
    }

    # text summary
    print(f"\n=== Eval by run ({args.name}) ===")
    print(f"{'stage':<16}{'run_id':<20}{'gen_acc':>10}{'know_acc':>10}{'ppl':>10}")
    for r in ORDER:
        g = f"{gen[r]:.4f}" if gen[r] is not None else "-"
        k = f"{know[r]:.4f}" if know[r] is not None else "-"
        p = f"{ppl[r]:.2f}" if ppl[r] is not None else "-"
        print(f"{r:<16}{run_map[r]:<20}{g:>10}{k:>10}{p:>10}")
    print("\n=== Gain attribution ===")
    for k, v in gains.items():
        print(f"  {k:<28} {v if v is not None else '(missing run)'}")

    import wandb
    run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                     name=args.name, job_type="report", reinit=True)
    table = wandb.Table(columns=["run", "gen_acc", "know_acc", "perplexity"])
    for r in ORDER:
        table.add_data(r, gen[r], know[r], ppl[r])
    run.log({"eval_table": table})
    run.log({k: v for k, v in gains.items() if v is not None})
    # bar chart of the three generative-accuracy gains
    bar = wandb.Table(columns=["stage", "gen_acc_gain"], data=[
        ["SFT", gains["gen/SFT_gain(R1-R0)"]],
        ["CPT", gains["gen/CPT_gain(R3-R1)"]],
        ["GRPO", gains["gen/GRPO_gain(R4-R3)"]],
    ])
    run.log({"gain_waterfall": wandb.plot.bar(bar, "stage", "gen_acc_gain",
                                              title="Per-stage generative-accuracy gain")})
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
