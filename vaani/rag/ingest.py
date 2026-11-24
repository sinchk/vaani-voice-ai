"""Knowledge-base ingestion: read policy files -> chunk -> embed -> store in Chroma.

Run ``python -m vaani.rag.ingest`` (add ``--reset`` to rebuild from scratch).
Ingestion is idempotent: chunk IDs derive from file name + position, so running
it twice never duplicates content.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import re
from pathlib import Path

import chromadb

from vaani.config import get_settings
from vaani.logging_setup import configure_logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "kb_collection"
# Cosine space makes `1 - distance` a true similarity score.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """Split on blank lines, then pack sentences into chunks of at most ``max_chars``.

    Sentence boundaries are '.', '!' and '?' followed by whitespace.
    """
    chunks: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            if current and len(current) + len(sentence) + 1 > max_chars:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current)
    return chunks


def title_of(path: Path) -> str:
    """'return_policy.txt' -> 'Return policy'. Prepended when embedding so chunks carry their topic."""
    return path.stem.replace("_", " ").capitalize()


def load_chunks(kb_dir: Path) -> list[dict]:
    items = []
    for path in sorted(Path(kb_dir).glob("*.txt")):
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            items.append({"id": f"{path.name}::{i}", "text": chunk, "source": path.name, "title": title_of(path)})
    return items


def get_collection(client: chromadb.ClientAPI | None = None):
    client = client or chromadb.PersistentClient(path=str(get_settings().chroma_dir))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata=COLLECTION_METADATA)


def ingest_knowledge_base(kb_dir: Path | None = None, reset: bool = False, embedder=None) -> int:
    """Embed and upsert the knowledge base. Returns the number of chunks stored."""
    settings = get_settings()
    kb_dir = kb_dir or settings.knowledge_base_dir
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if reset:
        with contextlib.suppress(Exception):  # the collection may not exist yet
            client.delete_collection(COLLECTION_NAME)
    collection = get_collection(client)

    items = load_chunks(kb_dir)
    if not items:
        logger.warning("No .txt files found in %s", kb_dir)
        return 0

    if embedder is None:
        from vaani.rag.embeddings import EmbeddingModel

        embedder = EmbeddingModel()
    collection.upsert(
        ids=[i["id"] for i in items],
        documents=[i["text"] for i in items],
        metadatas=[{"source": i["source"]} for i in items],
        embeddings=embedder.embed_documents([f"{i['title']}: {i['text']}" for i in items]),
    )
    logger.info("Ingested %d chunks from %s", len(items), kb_dir)
    return len(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Vaani knowledge-base vector store.")
    parser.add_argument("--reset", action="store_true", help="drop the existing collection first")
    args = parser.parse_args()
    configure_logging()
    print(f"Ingested {ingest_knowledge_base(reset=args.reset)} chunks.")
