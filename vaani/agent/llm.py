"""Thin wrapper over the Gemini chat API.

Isolating the SDK here keeps :class:`~vaani.agent.agent.VaaniAgent` free of
vendor details (and trivially fakeable in tests).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from google.genai import errors, types

from vaani.config import get_settings
from vaani.gemini import get_client

logger = logging.getLogger(__name__)

_RETRYABLE = {429, 500, 502, 503, 504}


def _is_daily_quota(exc: errors.APIError) -> bool:
    return exc.code == 429 and "per day" in (exc.message or "").lower()


class LLMError(RuntimeError):
    """A non-recoverable LLM failure (bad key, invalid request, ...)."""


class LLMUnavailableError(LLMError):
    """The LLM is rate-limited or down even after retrying."""


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class LLMTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: types.Content | None = None  # raw model content, echoed back verbatim in the next round


class LLM(Protocol):
    def generate(self, system: str, contents: list[types.Content], tools: list[dict]) -> LLMTurn: ...


def generate_with_retry(model: str, contents, config, retries: int):
    """generate_content with exponential backoff on transient (429/5xx) errors."""
    for attempt in range(retries + 1):
        try:
            return get_client().models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as exc:
            if exc.code not in _RETRYABLE:
                raise LLMError(f"Gemini request failed: {exc}") from exc
            # A daily quota cannot recover within seconds: fail fast so the fallback model is tried.
            if attempt == retries or _is_daily_quota(exc):
                raise LLMUnavailableError(f"{model} unavailable ({exc.code})") from exc
            delay = 2**attempt
            logger.warning("%s returned %s (attempt %d); retrying in %ss", model, exc.code, attempt + 1, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


class GeminiLLM:
    def __init__(
        self,
        model: str | None = None,
        max_retries: int | None = None,
        fallback_models: Sequence[str] | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.chat_model
        self.max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        fallbacks = settings.fallback_models if fallback_models is None else fallback_models
        self.fallback_models = [m for m in fallbacks if m and m != self.model]

    def generate(self, system: str, contents: list[types.Content], tools: list[dict]) -> LLMTurn:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=tools)] if tools else None,
            # We run the tool loop ourselves so every call can be logged and returned to the UI.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.3,
        )
        response = None
        for position, model in enumerate([self.model, *self.fallback_models]):
            try:
                # Fallbacks get a single retry: the primary already absorbed the transient errors.
                response = generate_with_retry(model, contents, config, self.max_retries if position == 0 else 1)
                break
            except LLMUnavailableError:
                logger.warning("%s unavailable; trying next model", model)
        if response is None:
            raise LLMUnavailableError("All configured Gemini models are unavailable")
        return self._to_turn(response)

    @staticmethod
    def _to_turn(response) -> LLMTurn:
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        calls = [ToolCall(fc.name, dict(fc.args or {})) for fc in (response.function_calls or [])]
        text = ""
        if content and content.parts:
            text = "".join(p.text for p in content.parts if p.text and not getattr(p, "thought", False))
        return LLMTurn(text=text.strip(), tool_calls=calls, content=content)
