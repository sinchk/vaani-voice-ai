"""Thin wrapper over the Gemini chat API.

Isolating the SDK here keeps :class:`~vaani.agent.agent.VaaniAgent` free of
vendor details (and trivially fakeable in tests).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from google.genai import errors, types

from vaani.config import get_settings
from vaani.gemini import get_client

logger = logging.getLogger(__name__)

_RETRYABLE = {429, 500, 502, 503, 504}


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


class GeminiLLM:
    def __init__(
        self, model: str | None = None, max_retries: int | None = None, fallback_model: str | None = None
    ) -> None:
        settings = get_settings()
        self.model = model or settings.chat_model
        self.max_retries = max_retries if max_retries is not None else settings.llm_max_retries
        fallback = settings.fallback_model if fallback_model is None else fallback_model
        self.fallback_model = fallback if fallback and fallback != self.model else None

    def generate(self, system: str, contents: list[types.Content], tools: list[dict]) -> LLMTurn:
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=tools)] if tools else None,
            # We run the tool loop ourselves so every call can be logged and returned to the UI.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.3,
        )
        try:
            response = self._call(self.model, contents, config, self.max_retries)
        except LLMUnavailableError:
            if not self.fallback_model:
                raise
            logger.warning("%s unavailable; falling back to %s", self.model, self.fallback_model)
            response = self._call(self.fallback_model, contents, config, retries=1)
        return self._to_turn(response)

    @staticmethod
    def _call(model: str, contents, config, retries: int):
        """generate_content with exponential backoff on transient (429/5xx) errors."""
        for attempt in range(retries + 1):
            try:
                return get_client().models.generate_content(model=model, contents=contents, config=config)
            except errors.APIError as exc:
                if exc.code not in _RETRYABLE:
                    raise LLMError(f"Gemini request failed: {exc}") from exc
                if attempt == retries:
                    raise LLMUnavailableError(f"{model} unavailable ({exc.code})") from exc
                delay = 2**attempt
                logger.warning("%s returned %s (attempt %d); retrying in %ss", model, exc.code, attempt + 1, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _to_turn(response) -> LLMTurn:
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        calls = [ToolCall(fc.name, dict(fc.args or {})) for fc in (response.function_calls or [])]
        text = ""
        if content and content.parts:
            text = "".join(p.text for p in content.parts if p.text and not getattr(p, "thought", False))
        return LLMTurn(text=text.strip(), tool_calls=calls, content=content)
