# CPT → SFT → GRPO Gain-Attribution Study (Chinese Medical, 4B)

**Goal:** On a single ~4B base model, quantify and *cleanly separate* the gain contributed
by each post-training stage — Continued Pre-Training (CPT), Supervised Fine-Tuning (SFT),
and Group Relative Policy Optimization (GRPO) — in a Chinese-language **medical** domain,
with every result tracked in Weights & Biases.

- **Base model:** `Qwen/Qwen3-4B-Base` (base, not instruct — so SFT gain is visible)
- **Domain / language:** Medicine / Chinese (Simplified)
- **Hardware:** 8× H100 80GB (CUDA 12.9), 192 CPU, 2 TB RAM, 3.1 TB free on `/mnt/nvme4`
- **Tracking:** one W&B project, `med-4b-cpt-sft-grpo`

---

## 1. Why medicine + Chinese

- **GRPO needs a verifiable reward.** Medical licensing-exam multiple-choice (MedQA-Mainland,
  CMExam, CMB) gives a *programmatically checkable* gold answer → reward = exact-match on the
  chosen option + a format reward. This is the cleanest possible GRPO signal.
- **CPT has abundant Chinese medical raw text** (guidelines, textbooks, encyclopedias, TCM).
- **Standard Chinese eval suites exist** (CMExam, CMB, CMMLU-medical, C-Eval-medical, MedQA-zh).

**Risk:** Qwen3-4B-Base already knows a lot of Chinese medicine, so the CPT *downstream* gain
may be small. Mitigations: (a) bias the CPT corpus toward content the base is weaker on
(clinical guidelines, TCM, specialty texts); (b) always report **held-out perplexity**, which
CPT is guaranteed to reduce, as a second, independent CPT-gain axis.

---

## 2. The core design: separate gains by an ablation matrix

Running the pipeline once and eval-ing after each stage does **not** cleanly attribute gains,
because CPT alone barely moves downstream QA accuracy (a base model can't follow the answer
format until SFT). We therefore run a small ablation matrix on **one fixed eval suite**:

| Run | CPT | SFT | GRPO | Purpose / what it isolates                     |
|-----|:---:|:---:|:----:|------------------------------------------------|
| R0  |  —  |  —  |  —   | Base model, few-shot baseline                  |
| R1  |  —  |  ✓  |  —   | SFT from the cold base                          |
| R2  |  ✓  |  —  |  —   | CPT-only (perplexity ↓; few-shot accuracy)      |
| R3  |  ✓  |  ✓  |  —   | CPT then SFT                                     |
| R4  |  ✓  |  ✓  |  ✓   | Full pipeline                                   |

**Gain attribution — all on the same held-out test sets:**

```
SFT gain   = R1 - R0            # instruction-following unlocked (base uses few-shot)
CPT gain   = R3 - R1            # SFT held constant, add ONLY CPT  ← the clean isolation
GRPO gain  = R4 - R3            # RL on verifiable reward, on top of CPT+SFT
CPT gain (perplexity) = PPL(R2) - PPL(R0)   # guaranteed, independent second axis
```

Optional **R5 = SFT+GRPO (no CPT)** double-checks that the GRPO gain is not CPT-dependent
(`GRPO gain (no CPT) = R5 - R1`). Kept optional to save GPU time.

All five runs + a **gain-waterfall** chart live in one W&B project so the story reads at a glance:
`base → +CPT → +SFT → +GRPO`.

---

## 3. Data plan (Chinese medical)

> Exact dataset IDs/configs are being verified by a research pass and pinned in
> `med_pipeline/data/sources.py` before any download. Below is the strategy.

### 3.1 CPT — raw Chinese medical text, ~1–2B tokens
- Chinese medical corpus (`shibing624/medical` pretrain split) + clinical guidelines +
  textbook/encyclopedia text + TCM corpora.
- Pipeline: `cpt_fetch` (download+clean) → `cpt_dedup` (dedup) → `cpt_pack` (tokenize, pack to
  fixed length, carve a held-out shard for perplexity).
- ~4–8 GB of text → trivial on 3.1 TB disk. Full-parameter CPT.

### 3.2 SFT — Chinese medical instruction / chain-of-thought, ~30–50k
- Open sets: `FreedomIntelligence/medical-o1-reasoning-SFT` (Chinese config) + others.
- **Gemini distillation with rejection sampling** (uses your `GEMINI_API_KEY`): feed
  MedQA-zh / CMExam *train* questions to a cheap Gemini model, keep only samples whose final
  answer matches gold. ~20M output tokens ≈ **< ~$15**. Produces high-quality Chinese CoT.
- Formatted as chat pairs (system + user + assistant CoT with a boxed final answer).

### 3.3 GRPO — verifiable Chinese medical MCQ, ~20–40k prompts
- MedQA-Mainland (zh) + CMExam *train* questions.
- Reward = `answer_correct` (exact-match on option) + `format_ok` (answer inside the required
  tag / `\boxed{}`), via a pure-function reward registry (`tools/reward.py`).
- Test sets are held out for eval — never trained on.

### 3.4 Eval — fixed suite, run identically after every stage
- **lm-evaluation-harness**: CMMLU-medical subsets, C-Eval-medical subsets (built-in), plus
  **custom YAML tasks** for CMExam / CMB / MedQA-zh where no built-in exists.
- **Perplexity** on the held-out medical shard (CPT signal).
- Same suite, same few-shot config, every checkpoint → apples-to-apples.

---

## 4. Frameworks (battle-tested, cost-effective)

| Concern       | Tool                                   | Why |
|---------------|----------------------------------------|-----|
| CPT + SFT     | **LLaMA-Factory** (`pt` + `sft`)       | one framework, full-param FT, DeepSpeed ZeRO-3, native W&B |
| GRPO          | **TRL `GRPOTrainer`** + vLLM rollouts  | simple, easy custom verifiable reward, native W&B (verl = scale-up alt) |
| Eval          | **lm-evaluation-harness**              | standard Chinese medical tasks + custom YAML |
| Tracking      | **W&B**                                | one project, waterfall chart, curves, eval tables |

The repo wraps these behind slim, LOD-style orchestrators — we do **not** re-implement training
loops. Project-level knobs live in a Pydantic schema; framework specifics live in their YAMLs.

---

## 5. Compute & rough timeline (8× H100, full-param throughout, 4B fits easily)

| Stage        | Work                              | Est. wall-clock |
|--------------|-----------------------------------|-----------------|
| CPT          | ~1–2B tokens                      | 7–15 h          |
| SFT          | ~30–50k × 2–3 epochs              | 1–3 h           |
| GRPO         | ~1–2k steps w/ vLLM rollouts      | 8–20 h          |
| Eval × runs  | full suite per checkpoint         | minutes–1 h each|
| **Total**    | R0–R4 + evals                     | **~2–4 days**   |

---

## 6. Repo layout (LOD-aligned: slim orchestrators, phases, pure-function tools)

```
configs/          schema.py (Pydantic, bounds) · paths.py · cpt/sft/grpo/eval YAMLs
med_pipeline/
  data/           cpt_fetch · cpt_dedup · cpt_pack · sft_build · sft_distill · grpo_build · sources.py
  tools/          gemini_client · mcq · reward · text_clean   (pure, tested, spec'd)
  train/          run_cpt · run_sft (LLaMA-Factory) · run_grpo (TRL)
  eval/           run_eval (lm-eval driver) · perplexity · tasks/*.yaml (custom CN tasks)
  report/         waterfall (build W&B gain-attribution chart from eval table)
setup/            install.sh (staged) · check_env.py
scripts/          00_prepare_data.sh · 10_run_matrix.sh (R0..R4)
data/ checkpoints/ logs/   (gitignored)
```

---

## 7. Eval measures (three complementary axes)

| Axis | How | Shows the gain of |
|------|-----|-------------------|
| **Knowledge** | lm-eval loglikelihood MCQ (`cmmlu_medical`, `ceval_medical`, `cmexam`, `medqa_zh`), uniform few-shot | CPT + SFT (works on the base too) |
| **Generative** | vLLM sample CoT on held-out MedQA/CMExam test → extract `\boxed{}` → acc | SFT (format) + **GRPO** (the RL objective) |
| **Perplexity** | held-out medical shard NLL | **CPT** (guaranteed to drop) |

`report/waterfall.py` merges these and logs the per-stage gains + a bar chart to W&B.

## 8. Environment pin that matters

Every Chinese medical **benchmark** here is a loading-script dataset (`bigbio/med_qa`,
`shibing624/medical`, `haonan-li/cmmlu`, `ceval/ceval-exam`). `datasets>=4.0` **removed script
support**, so `setup/install.sh` pins **`datasets>=2.20,<3.0`** everywhere (data env *and* the
`--full` training stack re-pin). Do NOT mix datasets versions — a dataset cached under 4.x cannot
be re-read by 2.x (`TypeError` in `Features.from_dict`); clear `~/.cache/huggingface/datasets/<id>`
if that happens. `transformers` is pinned `>=4.51,<5` (Qwen3 support, LLaMA-Factory/TRL-compatible).

## 9. Status & open items

- [x] Environment verified (8×H100, disk, internet, `.env` keys present)
- [x] Chinese dataset/eval IDs verified against live HF/GitHub → pinned in `data/sources.py`
- [x] Repo scaffold + data pipelines — **validated end-to-end** in `SMOKE` mode (tiny samples,
      no GPU/API): CPT fetch→dedup→pack, SFT build+merge, Gemini distill (dry-run cost est.),
      GRPO build, and reward wiring (correct `\boxed{}` → 1.0, wrong → 0.0) all pass.
- [ ] **Approval gate** → `install.sh --full`, full data prep (incl. real Gemini distill),
      run R0–R4, publish the waterfall. Validate custom eval YAMLs load at this point.
