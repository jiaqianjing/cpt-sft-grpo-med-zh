# med-4b-cpt-sft-grpo

Measure and **cleanly separate** the gains of **CPT → SFT → GRPO** on a ~4B base model
(`Qwen3-4B-Base`) in the **Chinese medical** domain, tracked in Weights & Biases.

See **[DESIGN.md](./DESIGN.md)** for the full design, the gain-attribution ablation matrix,
data plan, and frameworks.

## TL;DR — how gains are separated

Five runs on one fixed eval suite:

| Run | CPT | SFT | GRPO |
|-----|:---:|:---:|:----:|
| R0  |  —  |  —  |  —   |
| R1  |  —  |  ✓  |  —   |
| R2  |  ✓  |  —  |  —   |
| R3  |  ✓  |  ✓  |  —   |
| R4  |  ✓  |  ✓  |  ✓   |

```
SFT gain  = R1 - R0      CPT gain = R3 - R1      GRPO gain = R4 - R3
CPT gain (perplexity) = PPL(R2) - PPL(R0)
```

## Layout

```
configs/       project schema (Pydantic) + framework YAMLs
med_pipeline/  data/ tools/ train/ eval/ report/
setup/         install.sh · check_env.py
scripts/       prepare-data + run-matrix orchestrators
```

## Quickstart (once approved)

```bash
bash setup/install.sh           # staged: data env now, training stack at approval
python setup/check_env.py       # verify GPUs, versions, .env keys
bash scripts/00_prepare_data.sh # CPT/SFT/GRPO data (uses GEMINI_API_KEY for distill)
bash scripts/10_run_matrix.sh   # R0..R4 + evals, logged to W&B
```

## Secrets

`.env` (already present) provides `WANDB_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`.
Never commit real data, checkpoints, or `.env`.
