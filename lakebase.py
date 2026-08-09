"""Lakebase connection helpers shared by the API and embedding job."""

from __future__ import annotations

import base64
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

import psycopg2
from databricks.sdk import WorkspaceClient
from psycopg2.extensions import connection as PsycopgConnection
from psycopg2.extras import RealDictCursor


@lru_cache(maxsize=1)
def lakebase_url() -> str:
    """Return a Postgres URL from local env or a Databricks secret.

    ``LAKEBASE_URL`` makes local development straightforward. In Databricks,
    the URL is read from the configured secret scope and never logged.
    """

    direct_url = os.environ.get("LAKEBASE_URL", "").strip()
    if direct_url:
        return direct_url

    scope = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
    key = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")
    secret = WorkspaceClient().secrets.get_secret(scope=scope, key=key)
    if not secret.value:
        raise RuntimeError(f"Databricks secret {scope}/{key} is empty")
    return base64.b64decode(secret.value).decode("utf-8")


@contextmanager
def get_connection() -> Iterator[PsycopgConnection]:
    """Yield a psycopg2 connection whose rows behave like dictionaries."""

    conn = psycopg2.connect(lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def run_query(sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]


def run_write(sql: str, params: tuple[Any, ...] | None = None) -> int:
    with get_connection() as conn, conn.cursor() as cursor:
        cursor.execute(sql, params)
        affected = cursor.rowcount
        conn.commit()
        return affected

