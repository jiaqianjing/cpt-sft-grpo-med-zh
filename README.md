# cpt-sft-grpo-med-zh

[![🤗 Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-jiaqianjing-yellow)](https://huggingface.co/jiaqianjing)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-med--1b7--cpt--sft--grpo-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/jiaqianjing/med-1b7-cpt-sft-grpo)
[![Base](https://img.shields.io/badge/base-Qwen3--1.7B-blue)](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://www.apache.org/licenses/LICENSE-2.0)

Measure and **cleanly separate** the gains of **CPT → SFT → GRPO** on a small base model
(`Qwen3-1.7B-Base`) in the **Chinese medical** domain, tracked in Weights & Biases.
Full write-up: **[REPORT.md](./REPORT.md)** · design: **[DESIGN.md](./DESIGN.md)**.

> Started on Qwen3-4B-Base but it was near-ceiling on Chinese medical MCQ (naive SFT only
> degraded it), so the study moved to the weaker **1.7B** base for measurable headroom.

## 🤗 Model weights (Hugging Face)

| Stage | Model | Link |
|-------|-------|------|
| CPT | continued pre-training (0.9B tok, TCM-heavy) | [jiaqianjing/qwen3-1.7b-med-zh-cpt](https://huggingface.co/jiaqianjing/qwen3-1.7b-med-zh-cpt) |
| SFT | SFT-only (60k medical CoT) | [jiaqianjing/qwen3-1.7b-med-zh-sft](https://huggingface.co/jiaqianjing/qwen3-1.7b-med-zh-sft) |
| CPT→SFT | CPT then SFT | [jiaqianjing/qwen3-1.7b-med-zh-cpt-sft](https://huggingface.co/jiaqianjing/qwen3-1.7b-med-zh-cpt-sft) |
| CPT→SFT→GRPO | full pipeline | [jiaqianjing/qwen3-1.7b-med-zh-grpo](https://huggingface.co/jiaqianjing/qwen3-1.7b-med-zh-grpo) |

📊 All training curves, eval tables, and the gain-attribution **waterfall** →
[**W&B: med-1b7-cpt-sft-grpo**](https://wandb.ai/jiaqianjing/med-1b7-cpt-sft-grpo)

## TL;DR — how gains are separated

Ablation matrix on one fixed eval suite (three axes: knowledge / generative / perplexity):

| Run | CPT | SFT | GRPO |
|-----|:---:|:---:|:----:|
| R0  |  —  |  —  |  —   |
| R1  |  —  |  ✓  |  —   |
| R2  |  ✓  |  —  |  —   |
| R3  |  ✓  |  ✓  |  —   |
| R4  |  ✓  |  ✓  |  ✓   |

```
SFT = R1 - R0      CPT = R3 - R1      GRPO = R4 - R3      CPT-ppl = PPL(R0) - PPL(R2)
```

## Results (Qwen3-1.7B) — see [REPORT.md](./REPORT.md)

| Run | generative | knowledge | perplexity |
|-----|:---:|:---:|:---:|
| R0 base | 0.589 | 0.661 | 7.04 |
| R2 CPT | — | 0.636 | **5.79** |
| R1 SFT | 0.544 | 0.645 | — |
| R3 CPT+SFT | 0.537 | 0.635 | — |
| R4 +GRPO | 0.552 | 0.646 | — |

**Honest finding:** CPT gives a clear perplexity gain (−1.25); SFT/GRPO downstream accuracy is
limited by a weak distillation teacher (Gemini flash-lite) — data quality dominated stage design.
GRPO trained correctly (reward 0.40→0.625) but transferred little to test. See REPORT §5–7.

## Layout

```
configs/       project schema (Pydantic) + framework YAMLs
med_pipeline/  data/ tools/ train/ eval/ report/
setup/         install.sh · check_env.py
scripts/       prepare-data · run-matrix · post-CPT · HF upload
```

## Quickstart

```bash
bash setup/install.sh --full     # 3 envs: eval / train (torch2.8) / grpo (trl+vllm)
python setup/check_env.py --full # verify GPUs, versions, .env keys
bash scripts/00_prepare_data.sh  # CPT/SFT/GRPO data (DISTILL=1 uses GEMINI_API_KEY)
python -m med_pipeline.train.run_cpt   # then:
bash scripts/run_post_cpt.sh     # R1→R3→evals→R4→gain waterfall to W&B
```

## Secrets

`.env` (gitignored) provides `WANDB_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`.
Never commit real data, checkpoints, or `.env`.
