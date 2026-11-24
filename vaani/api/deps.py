"""Lazily-built shared singletons, injected into routes (and overridable in tests)."""

from __future__ import annotations

from functools import lru_cache

from vaani.agent.agent import VaaniAgent
from vaani.speech.asr import ASRProcessor
from vaani.speech.tts import TTSProcessor
from vaani.tools import ToolRegistry


@lru_cache(maxsize=1)
def get_agent() -> VaaniAgent:
    return VaaniAgent(tools=get_tools())


@lru_cache(maxsize=1)
def get_tools() -> ToolRegistry:
    return ToolRegistry.default()


@lru_cache(maxsize=1)
def get_asr() -> ASRProcessor:
    return ASRProcessor()


@lru_cache(maxsize=1)
def get_tts() -> TTSProcessor:
    return TTSProcessor()
