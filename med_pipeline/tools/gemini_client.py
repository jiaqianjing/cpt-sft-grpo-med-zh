"""Thin Gemini client for CoT distillation (uses GEMINI_API_KEY).

CALLING SPEC:
    from med_pipeline.tools.gemini_client import GeminiClient
    gc = GeminiClient(model="gemini-2.5-flash")
    text = gc.generate("提示词", max_new_tokens=2048, temperature=0.7)   # -> str ("" on failure)
    texts = gc.generate_batch(prompts, concurrency=16, ...)             # -> list[str], order-preserving

    Side effects: NETWORK calls to the Gemini API; reads GEMINI_API_KEY from env.
    NOT deterministic (sampling). Retries transient errors with backoff (tenacity).
    This is an I/O tool, isolated here so pure data logic stays testable without the network.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from tenacity import retry, stop_after_attempt, wait_exponential


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        from google import genai  # lazy import so the package isn't required at import time

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set (load .env via configs.loader.load_secrets)")
        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.model = model

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        cfg = self._genai.types.GenerateContentConfig(
            max_output_tokens=max_new_tokens, temperature=temperature
        )
        resp = self.client.models.generate_content(
            model=self.model, contents=prompt, config=cfg
        )
        return resp.text or ""

    def generate(self, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Single generation. Returns "" if all retries fail (caller filters empties)."""
        try:
            return self._call(prompt, max_new_tokens, temperature)
        except Exception:  # noqa: BLE001 — distillation tolerates per-item failures
            return ""

    def generate_batch(
        self,
        prompts: list[str],
        concurrency: int = 16,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> list[str]:
        """Concurrent generation; results are returned in the SAME order as `prompts`."""
        results: list[str] = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {
                pool.submit(self.generate, p, max_new_tokens, temperature): i
                for i, p in enumerate(prompts)
            }
            for fut in futs:
                results[futs[fut]] = fut.result()
        return results
