"""Pure functions used by the weather embedding pipeline."""

from __future__ import annotations

import hashlib


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text with a deterministic character-based sliding window."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(normalized), step):
        chunk = normalized[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(normalized):
            break
    return chunks


def stable_embedding_id(document_id: str, model_name: str, chunk_index: int) -> str:
    value = f"{document_id}|{model_name}|{chunk_index}".encode("utf-8")
    return f"weather-emb-{hashlib.sha256(value).hexdigest()[:32]}"


def vector_literal(values: list[float]) -> str:
    """Serialize numeric values into pgvector's text input representation."""

    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"

