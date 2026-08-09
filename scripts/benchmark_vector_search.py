"""Print an EXPLAIN ANALYZE plan for the pgvector HNSW search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import weather_repository  # noqa: E402
from embedding_model import embed_texts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="flash flood risk this weekend")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    weather_repository.ensure_schema()
    query_vector = embed_texts([args.query])[0]
    print("\n".join(weather_repository.explain_search(query_vector, args.top_k)))


if __name__ == "__main__":
    main()

