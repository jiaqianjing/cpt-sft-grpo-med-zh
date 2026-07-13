"""OpenAI-compatible client for a local SGLang teacher server.

CALLING SPEC:
    client = SGLangClient(model="Qwen/Qwen3.6-27B", base_url="http://127.0.0.1:30000/v1")
    texts = client.generate_batch(prompts, concurrency=16, max_new_tokens=2048, temperature=1.0)

    Input: text prompts and sampling settings.
    Output: one final response per prompt, in input order; failures return "".
    Internal ``reasoning_content`` is intentionally excluded from SFT targets.
    Side effects: HTTP requests to an already-running local SGLang server.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


class SGLangClient:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.endpoint = f"{base_url.rstrip('/')}/chat/completions"

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.8,
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode()
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                message = json.load(response)["choices"][0]["message"]
        except Exception:  # noqa: BLE001 - failed samples are rejected by the caller
            return ""
        return (message.get("content") or "").strip()

    def generate_batch(
        self,
        prompts: list[str],
        concurrency: int = 16,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> list[str]:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(
                pool.map(
                    lambda prompt: self.generate(prompt, max_new_tokens, temperature),
                    prompts,
                )
            )
