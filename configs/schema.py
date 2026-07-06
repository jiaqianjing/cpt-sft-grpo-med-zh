"""Project configuration schema — the single source of truth for run-level knobs.

CALLING SPEC:
    from configs.schema import ProjectConfig
    cfg = ProjectConfig()                      # all defaults, validated
    cfg = ProjectConfig(**yaml_dict)           # from a loaded YAML dict (see configs/loader.py)
    cfg.model.base_model_id                     # -> "Qwen/Qwen3-4B-Base"
    cfg.cpt.target_tokens                        # -> int, bounded

This file is SCHEMA ONLY (LOD Pattern 6): Pydantic models with explicit bounds and
`extra='forbid'`. No file I/O, no training logic. Loading/derivation lives elsewhere
(configs/loader.py, configs/paths.py). Framework-specific hyperparameters that LLaMA-Factory
or TRL own live in their own YAMLs; this schema captures only the project-level orchestration
knobs that our own data/eval/report code reads.

Zero-hallucination contract (LOD Pattern 7): every numeric field has ge/le bounds and unknown
fields are rejected, so an invalid config fails immediately with a ValidationError.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_Strict = ConfigDict(extra="forbid")  # reject hallucinated field names everywhere


class ModelCfg(BaseModel):
    model_config = _Strict
    base_model_id: str = Field(default="Qwen/Qwen3-1.7B-Base")  # weaker base → clearer stage gains
    trust_remote_code: bool = Field(default=False)  # Qwen3 loads on stock transformers>=4.51
    max_seq_len: int = Field(default=4096, ge=512, le=32768)  # base card documents 32768 native
    attn_impl: Literal["flash_attention_2", "sdpa", "eager"] = "sdpa"  # flash-attn wheel absent on cu130/torch2.11


class WandbCfg(BaseModel):
    model_config = _Strict
    project: str = Field(default="med-1b7-cpt-sft-grpo")
    entity: str | None = Field(default=None)
    mode: Literal["online", "offline", "disabled"] = "online"


class CPTCfg(BaseModel):
    """Continued pre-training on raw Chinese medical text (LLaMA-Factory stage=pt)."""
    model_config = _Strict
    target_tokens: int = Field(default=1_500_000_000, ge=10_000_000, le=20_000_000_000)
    holdout_tokens: int = Field(default=5_000_000, ge=100_000, le=200_000_000)
    epochs: float = Field(default=2.0, gt=0.0, le=10.0)  # 2 epochs over ~0.9B tok to strengthen CPT signal
    learning_rate: float = Field(default=2e-5, gt=0.0, le=1e-3)
    per_device_batch: int = Field(default=4, ge=1, le=64)
    grad_accum: int = Field(default=8, ge=1, le=256)
    seq_len: int = Field(default=4096, ge=512, le=32768)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    save_steps: int = Field(default=500, ge=10, le=100_000)
    deepspeed_stage: Literal[0, 1, 2, 3] = 3
    packing: bool = Field(default=True)


class SFTCfg(BaseModel):
    """Supervised fine-tuning on Chinese medical CoT (LLaMA-Factory stage=sft)."""
    model_config = _Strict
    max_examples: int = Field(default=60_000, ge=100, le=2_000_000)  # CoT-only pool (~45k+15k)
    epochs: float = Field(default=2.0, gt=0.0, le=10.0)  # fewer epochs: avoid overfitting strong base
    learning_rate: float = Field(default=1e-5, gt=0.0, le=1e-3)
    per_device_batch: int = Field(default=8, ge=1, le=64)
    grad_accum: int = Field(default=2, ge=1, le=256)
    cutoff_len: int = Field(default=4096, ge=512, le=32768)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    deepspeed_stage: Literal[0, 1, 2, 3] = 3


class GRPOCfg(BaseModel):
    """GRPO on verifiable Chinese medical MCQ (TRL GRPOTrainer + vLLM)."""
    model_config = _Strict
    max_prompts: int = Field(default=30_000, ge=100, le=1_000_000)
    num_train_epochs: float = Field(default=1.0, gt=0.0, le=10.0)
    learning_rate: float = Field(default=1e-6, gt=0.0, le=1e-4)
    num_generations: int = Field(default=8, ge=2, le=64)      # group size G
    per_device_batch: int = Field(default=2, ge=1, le=128)    # per-GPU prompts; *G seqs/backward (memory)
    grad_accum: int = Field(default=4, ge=1, le=256)
    max_prompt_len: int = Field(default=1024, ge=128, le=8192)
    max_completion_len: int = Field(default=768, ge=64, le=8192)  # medical CoT is short; faster rollouts
    temperature: float = Field(default=1.0, gt=0.0, le=2.0)
    beta_kl: float = Field(default=0.04, ge=0.0, le=1.0)       # KL penalty coefficient
    reward_correct_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    reward_format_weight: float = Field(default=0.2, ge=0.0, le=10.0)
    vllm_gpu_frac: float = Field(default=0.3, gt=0.0, le=0.95)  # colocate: leave room for training


class DistillCfg(BaseModel):
    """Gemini CoT distillation with rejection sampling (uses GEMINI_API_KEY)."""
    model_config = _Strict
    enabled: bool = Field(default=True)
    teacher_model: str = Field(default="gemini-2.5-flash-lite")  # cheapest GA: $0.10/$0.40 per 1M
    max_questions: int = Field(default=20_000, ge=0, le=500_000)
    samples_per_question: int = Field(default=1, ge=1, le=8)
    max_new_tokens: int = Field(default=2048, ge=128, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    concurrency: int = Field(default=16, ge=1, le=256)
    keep_only_correct: bool = Field(default=True)  # rejection sampling on gold match


class EvalCfg(BaseModel):
    model_config = _Strict
    # lm-eval-harness task names: two custom group YAMLs aggregating built-in CMMLU/C-Eval
    # medical subjects, plus custom YAMLs for CMExam and Chinese MedQA (verified: no built-in).
    tasks: list[str] = Field(
        default_factory=lambda: ["cmmlu_medical", "ceval_medical", "cmexam", "medqa_zh"]
    )
    num_fewshot: int = Field(default=0, ge=0, le=32)
    batch_size: int = Field(default=16, ge=1, le=512)
    limit: int | None = Field(default=None, ge=1)  # None = full test set; set for smoke tests


class RunMatrixCfg(BaseModel):
    """Which ablation runs to execute (LOD: dict-dispatch friendly)."""
    model_config = _Strict
    R0_base: bool = True        # base few-shot baseline
    R1_sft: bool = True         # SFT only
    R2_cpt: bool = True         # CPT only (perplexity)
    R3_cpt_sft: bool = True     # CPT -> SFT
    R4_cpt_sft_grpo: bool = True  # full
    R5_sft_grpo: bool = False   # optional GRPO-without-CPT control


class ProjectConfig(BaseModel):
    model_config = _Strict
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    domain: Literal["medical"] = "medical"
    language: Literal["zh", "en"] = "zh"
    root: str = Field(default="/mnt/nvme4/ken/workspace/llm")

    model: ModelCfg = Field(default_factory=ModelCfg)
    wandb: WandbCfg = Field(default_factory=WandbCfg)
    cpt: CPTCfg = Field(default_factory=CPTCfg)
    sft: SFTCfg = Field(default_factory=SFTCfg)
    grpo: GRPOCfg = Field(default_factory=GRPOCfg)
    distill: DistillCfg = Field(default_factory=DistillCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    runs: RunMatrixCfg = Field(default_factory=RunMatrixCfg)
