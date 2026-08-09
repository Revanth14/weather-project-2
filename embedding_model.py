"""One cached sentence-transformers model shared by ingestion and search."""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Sequence


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
_ENCODE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    model_name = os.environ.get("WEATHER_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    model = SentenceTransformer(model_name)
    actual_dimension = model.get_sentence_embedding_dimension()
    if actual_dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"{model_name} produces {actual_dimension}-dimensional vectors; "
            f"the schema requires {EMBEDDING_DIMENSION}"
        )
    return model


def embed_texts(texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []
    # SentenceTransformer uses shared model state. Serializing encode calls
    # keeps concurrent Flask requests predictable in a threaded WSGI worker.
    with _ENCODE_LOCK:
        vectors = get_embedding_model().encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return vectors.tolist()
