"""Scheduled NWS-to-Lakebase sync for the Databricks refresh workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import weather_repository  # noqa: E402
from weather_client import WeatherClient, WeatherClientError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync NWS narratives into Lakebase")
    parser.add_argument(
        "--locations",
        nargs="+",
        default=["Chicago, IL", "Austin, TX", "Miami, FL"],
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error("limit must be between 1 and 100")

    weather_repository.ensure_schema()
    client = WeatherClient()
    synced = 0
    errors: list[dict[str, str]] = []
    for requested_location in args.locations:
        try:
            resolved, documents = client.fetch_documents(requested_location, args.limit)
            count = weather_repository.upsert_documents(documents)
            synced += count
            print(
                json.dumps(
                    {
                        "event": "location_sync_complete",
                        "requested": requested_location,
                        "resolved": resolved.label,
                        "documents": count,
                    }
                ),
                flush=True,
            )
        except WeatherClientError as exc:
            errors.append({"location": requested_location, "error": str(exc)})

    print(json.dumps({"event": "weather_sync_complete", "synced": synced, "errors": errors}))
    if errors and synced == 0:
        raise SystemExit("All configured locations failed to sync")


if __name__ == "__main__":
    main()

