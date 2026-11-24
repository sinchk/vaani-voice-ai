"""Shared Gemini client factory."""

from __future__ import annotations

from functools import lru_cache

from google import genai

from vaani.config import get_settings


class MissingAPIKeyError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    key = get_settings().gemini_api_key
    if not key:
        raise MissingAPIKeyError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return genai.Client(api_key=key)
