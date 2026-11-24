"""Gemini embeddings (multilingual, so Kannada queries match English documents)."""

from __future__ import annotations

from google.genai import types

from vaani.config import get_settings
from vaani.gemini import get_client

_BATCH = 50


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or get_settings().embedding_model

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        client = get_client()
        config = types.EmbedContentConfig(task_type=task_type)
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            result = client.models.embed_content(model=self.model_name, contents=texts[i : i + _BATCH], config=config)
            vectors.extend(list(e.values) for e in result.embeddings)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
