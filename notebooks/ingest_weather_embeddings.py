"""Batch weather chunking and embedding job using psycopg2 only.

This file can run as a normal Python script or as a Databricks Python task.
It intentionally does not use Spark for Lakebase reads or writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import weather_repository  # noqa: E402
from embedding_model import DEFAULT_MODEL_NAME, embed_texts  # noqa: E402
from embedding_utils import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_text,
    stable_embedding_id,
)


def ingest_embeddings(
    *,
    model_name: str,
    fetch_batch_size: int,
    embedding_batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    max_documents: int | None,
) -> dict[str, int | str]:
    os.environ["WEATHER_EMBEDDING_MODEL"] = model_name
    weather_repository.ensure_schema()

    documents_processed = 0
    chunks_written = 0
    batches = 0

    while max_documents is None or documents_processed < max_documents:
        remaining = (
            fetch_batch_size
            if max_documents is None
            else min(fetch_batch_size, max_documents - documents_processed)
        )
        if remaining <= 0:
            break
        documents = weather_repository.documents_needing_embeddings(model_name, remaining)
        if not documents:
            break

        chunk_records: list[dict] = []
        chunk_strings: list[str] = []
        for document in documents:
            chunks = chunk_text(document["narrative_text"], chunk_size, chunk_overlap)
            for chunk_index, text in enumerate(chunks):
                chunk_records.append(
                    {
                        "id": stable_embedding_id(document["id"], model_name, chunk_index),
                        "document_id": document["id"],
                        "chunk_index": chunk_index,
                        "chunk_text": text,
                        "content_hash": document["content_hash"],
                    }
                )
                chunk_strings.append(text)

        vectors = embed_texts(chunk_strings, batch_size=embedding_batch_size)
        for record, vector in zip(chunk_records, vectors, strict=True):
            record["embedding"] = vector

        chunks_written += weather_repository.replace_embeddings(
            [document["id"] for document in documents],
            model_name,
            chunk_records,
        )
        documents_processed += len(documents)
        batches += 1
        print(
            json.dumps(
                {
                    "event": "embedding_batch_complete",
                    "batch": batches,
                    "documents": len(documents),
                    "chunks": len(chunk_records),
                }
            ),
            flush=True,
        )

    return {
        "model_name": model_name,
        "documents_processed": documents_processed,
        "chunks_written": chunks_written,
        "batches": batches,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed unprocessed Lakebase weather documents")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--fetch-batch-size", type=int, default=64)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--max-documents", type=int)
    args = parser.parse_args()
    if args.fetch_batch_size < 1 or args.embedding_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.chunk_size < 1 or not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("chunk overlap must be non-negative and smaller than chunk size")
    if args.max_documents is not None and args.max_documents < 1:
        parser.error("max-documents must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    summary = ingest_embeddings(
        model_name=arguments.model_name,
        fetch_batch_size=arguments.fetch_batch_size,
        embedding_batch_size=arguments.embedding_batch_size,
        chunk_size=arguments.chunk_size,
        chunk_overlap=arguments.chunk_overlap,
        max_documents=arguments.max_documents,
    )
    print(json.dumps({"event": "embedding_run_complete", **summary}, indent=2))
