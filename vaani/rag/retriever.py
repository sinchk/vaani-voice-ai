"""Semantic retrieval over the knowledge base."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from vaani.config import get_settings
from vaani.rag.ingest import get_collection, ingest_knowledge_base

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    content: str
    source: str
    confidence: float  # cosine similarity in [0, 1]

    def to_dict(self) -> dict:
        return asdict(self)


class KBRetriever:
    def __init__(
        self,
        embedder=None,
        min_score: float | None = None,
        top_k: int | None = None,
        margin: float | None = None,
    ) -> None:
        settings = get_settings()
        self.min_score = settings.rag_min_score if min_score is None else min_score
        self.margin = settings.rag_margin if margin is None else margin
        self.top_k = top_k or settings.rag_top_k
        if embedder is None:
            from vaani.rag.embeddings import EmbeddingModel

            embedder = EmbeddingModel()
        self.embedder = embedder
        self.collection = get_collection()
        if self.collection.count() == 0:
            logger.info("Knowledge base is empty - ingesting")
            ingest_knowledge_base(embedder=self.embedder)
            self.collection = get_collection()

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Up to ``top_k`` chunks scoring >= ``min_score`` and within ``margin`` of the best, best first."""
        result = self.collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True):
            score = round(1.0 - dist, 3)  # cosine distance -> similarity
            if score >= self.min_score:
                chunks.append(RetrievedChunk(content=doc, source=meta["source"], confidence=score))
        if chunks:
            best = chunks[0].confidence
            chunks = [c for c in chunks if best - c.confidence <= self.margin]
        return chunks
