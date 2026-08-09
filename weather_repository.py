"""Persistence operations for weather documents and pgvector embeddings."""

from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

from psycopg2.extras import Json, execute_values

import lakebase
from embedding_model import DEFAULT_MODEL_NAME, EMBEDDING_DIMENSION
from embedding_utils import vector_literal


DOCUMENTS_TABLE = "weather_documents"
EMBEDDINGS_TABLE = "weather_embeddings"


DDL_STATEMENTS = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    f"""
    CREATE TABLE IF NOT EXISTS {DOCUMENTS_TABLE} (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        location TEXT NOT NULL,
        latitude DOUBLE PRECISION NOT NULL,
        longitude DOUBLE PRECISION NOT NULL,
        source_type TEXT NOT NULL CHECK (source_type IN ('alert', 'forecast')),
        headline TEXT NOT NULL,
        narrative_text TEXT NOT NULL,
        issued_at TIMESTAMPTZ,
        effective_at TIMESTAMPTZ,
        expires_at TIMESTAMPTZ,
        content_hash TEXT NOT NULL,
        payload JSONB NOT NULL,
        synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_weather_documents_location ON {DOCUMENTS_TABLE} (location)",
    f"CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type ON {DOCUMENTS_TABLE} (source_type)",
    f"CREATE INDEX IF NOT EXISTS idx_weather_documents_effective_at ON {DOCUMENTS_TABLE} (effective_at DESC)",
    f"""
    CREATE TABLE IF NOT EXISTS {EMBEDDINGS_TABLE} (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES {DOCUMENTS_TABLE}(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
        chunk_text TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
        model_name TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (document_id, chunk_index, model_name)
    )
    """,
    f"CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id ON {EMBEDDINGS_TABLE} (document_id)",
    f"""
    CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
    ON {EMBEDDINGS_TABLE} USING hnsw (embedding vector_cosine_ops)
    """,
)


def ensure_schema() -> None:
    """Create the pgvector extension, tables, and supporting indexes."""

    with lakebase.get_connection() as conn, conn.cursor() as cursor:
        for statement in DDL_STATEMENTS:
            cursor.execute(statement)
        conn.commit()


def upsert_documents(documents: Sequence[dict[str, Any]]) -> int:
    if not documents:
        return 0

    values = [
        (
            document["id"],
            document["source_id"],
            document["location"],
            document["latitude"],
            document["longitude"],
            document["source_type"],
            document["headline"],
            document["narrative_text"],
            document.get("issued_at"),
            document.get("effective_at"),
            document.get("expires_at"),
            document["content_hash"],
            Json(document["payload"]),
        )
        for document in documents
    ]
    sql = f"""
        INSERT INTO {DOCUMENTS_TABLE} (
            id, source_id, location, latitude, longitude, source_type,
            headline, narrative_text, issued_at, effective_at, expires_at,
            content_hash, payload, synced_at
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            source_id = EXCLUDED.source_id,
            location = EXCLUDED.location,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            issued_at = EXCLUDED.issued_at,
            effective_at = EXCLUDED.effective_at,
            expires_at = EXCLUDED.expires_at,
            content_hash = EXCLUDED.content_hash,
            payload = EXCLUDED.payload,
            synced_at = now()
    """
    with lakebase.get_connection() as conn, conn.cursor() as cursor:
        execute_values(cursor, sql, values, page_size=200)
        conn.commit()
    return len(documents)


def documents_needing_embeddings(model_name: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return new or changed documents that lack current-model vectors."""

    return lakebase.run_query(
        f"""
        SELECT d.id, d.narrative_text, d.content_hash
        FROM {DOCUMENTS_TABLE} d
        WHERE NOT EXISTS (
            SELECT 1
            FROM {EMBEDDINGS_TABLE} e
            WHERE e.document_id = d.id
              AND e.model_name = %s
              AND e.content_hash = d.content_hash
        )
        ORDER BY d.synced_at, d.id
        LIMIT %s
        """,
        (model_name, limit),
    )


def replace_embeddings(
    document_ids: Sequence[str],
    model_name: str,
    embeddings: Sequence[dict[str, Any]],
) -> int:
    """Atomically replace a model's chunks for the supplied documents."""

    if not document_ids:
        return 0
    rows = [
        (
            item["id"],
            item["document_id"],
            item["chunk_index"],
            item["chunk_text"],
            item["content_hash"],
            vector_literal(item["embedding"]),
            model_name,
        )
        for item in embeddings
    ]
    with lakebase.get_connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            f"DELETE FROM {EMBEDDINGS_TABLE} WHERE document_id = ANY(%s) AND model_name = %s",
            (list(document_ids), model_name),
        )
        if rows:
            execute_values(
                cursor,
                f"""
                INSERT INTO {EMBEDDINGS_TABLE} (
                    id, document_id, chunk_index, chunk_text, content_hash,
                    embedding, model_name, created_at
                ) VALUES %s
                ON CONFLICT (document_id, chunk_index, model_name) DO UPDATE SET
                    id = EXCLUDED.id,
                    chunk_text = EXCLUDED.chunk_text,
                    content_hash = EXCLUDED.content_hash,
                    embedding = EXCLUDED.embedding,
                    created_at = now()
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s::vector, %s, now())",
                page_size=200,
            )
        conn.commit()
    return len(rows)


def semantic_search(
    query_embedding: Sequence[float],
    top_k: int,
    source_type: str | None = None,
    location: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise ValueError(f"Expected a {EMBEDDING_DIMENSION}-dimensional query embedding")

    query_vector = vector_literal(list(query_embedding))
    selected_model = model_name or os.environ.get("WEATHER_EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
    rows = lakebase.run_query(
        f"""
        SELECT
            d.id,
            d.location,
            d.latitude,
            d.longitude,
            d.source_type,
            d.headline,
            d.narrative_text,
            d.issued_at,
            d.effective_at,
            d.expires_at,
            e.chunk_index,
            e.chunk_text,
            e.model_name,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {EMBEDDINGS_TABLE} e
        JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
        WHERE e.model_name = %s
          AND e.content_hash = d.content_hash
          AND (%s IS NULL OR d.source_type = %s)
          AND (%s IS NULL OR d.location ILIKE '%%' || %s || '%%')
        ORDER BY e.embedding <=> %s::vector, d.id, e.chunk_index
        LIMIT %s
        """,
        (
            query_vector,
            selected_model,
            source_type,
            source_type,
            location,
            location,
            query_vector,
            top_k,
        ),
    )
    for row in rows:
        row["similarity"] = float(row["similarity"])
    return rows


def status_counts() -> dict[str, int]:
    rows = lakebase.run_query(
        f"""
        SELECT
            (SELECT COUNT(*) FROM {DOCUMENTS_TABLE}) AS documents,
            (
                SELECT COUNT(*)
                FROM {EMBEDDINGS_TABLE} e
                JOIN {DOCUMENTS_TABLE} d ON d.id = e.document_id
                WHERE e.content_hash = d.content_hash
            ) AS embeddings,
            (SELECT COUNT(DISTINCT location) FROM {DOCUMENTS_TABLE}) AS locations,
            (SELECT COUNT(*) FROM {DOCUMENTS_TABLE} WHERE source_type = 'alert') AS alerts,
            (SELECT COUNT(*) FROM {DOCUMENTS_TABLE} WHERE source_type = 'forecast') AS forecasts
        """
    )
    if not rows:
        return {"documents": 0, "embeddings": 0, "locations": 0, "alerts": 0, "forecasts": 0}
    return {key: int(value) for key, value in rows[0].items()}


def explain_search(query_embedding: Iterable[float], top_k: int = 5) -> list[str]:
    """Return EXPLAIN ANALYZE output for the optional HNSW benchmark."""

    vector = vector_literal(list(query_embedding))
    with lakebase.get_connection() as conn, conn.cursor() as cursor:
        cursor.execute(
            f"""
            EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
            SELECT document_id
            FROM {EMBEDDINGS_TABLE}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector, top_k),
        )
        return [str(next(iter(row.values()))) for row in cursor.fetchall()]
