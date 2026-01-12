"""Inference client for vLLM OpenAI-compatible API."""

import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI, OpenAI

from minimax_deploy.config import Settings, get_settings


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    time_to_first_token_ms: float | None = None
    tokens_per_second: float = field(init=False)

    def __post_init__(self):
        if self.latency_ms > 0 and self.completion_tokens > 0:
            self.tokens_per_second = (self.completion_tokens / self.latency_ms) * 1000
        else:
            self.tokens_per_second = 0.0


class InferenceClient:
    """Client for vLLM inference with OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        settings: Settings | None = None,
    ):
        self._settings = settings or get_settings()
        self.base_url = base_url or self._settings.vllm_base_url
        self.api_key = api_key or self._settings.vllm_api_key
        self.model = model or self._settings.model_name

        self._sync_client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self._async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> InferenceResult:
        """Synchronous text generation."""
        start_time = time.perf_counter()

        response = self._sync_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        usage = response.usage

        return InferenceResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    async def agenerate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> InferenceResult:
        """Asynchronous text generation."""
        start_time = time.perf_counter()

        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        usage = response.usage

        return InferenceResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_ms=latency_ms,
        )

    async def agenerate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Async streaming text generation."""
        stream = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            # Try to hit the models endpoint
            url = self.base_url.rstrip("/v1") + "/health"
            response = httpx.get(url, timeout=5.0)
            return response.status_code == 200
        except Exception:
            # Fallback: try a simple completion
            try:
                self.generate("test", max_tokens=1)
                return True
            except Exception:
                return False

    def close(self):
        """Close underlying clients."""
        self._sync_client.close()

    async def aclose(self):
        """Async close underlying clients."""
        await self._async_client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
