"""Teacher-client registry for SFT trace distillation.

CALLING SPEC:
    client = create_teacher_client(cfg.distill)
    texts = client.generate_batch(prompts, concurrency=..., max_new_tokens=..., temperature=...)

    Input: validated DistillCfg with provider-specific settings.
    Output: a GeminiClient or SGLangClient sharing the generate_batch interface.
    Side effects: initializes an API client; model serving remains outside this process.
"""

from __future__ import annotations

from typing import Callable


def _create_gemini(cfg):
    from med_pipeline.tools.gemini_client import GeminiClient

    return GeminiClient(model=cfg.teacher_model)


def _create_sglang(cfg):
    from med_pipeline.tools.sglang_client import SGLangClient

    return SGLangClient(
        model=cfg.teacher_model,
        base_url=cfg.server_url,
    )


TEACHER_CLIENTS: dict[str, Callable] = {
    "gemini": _create_gemini,
    "qwen_sglang": _create_sglang,
}


def create_teacher_client(cfg):
    """Create the configured teacher; config validation guarantees a known provider."""
    return TEACHER_CLIENTS[cfg.provider](cfg)
