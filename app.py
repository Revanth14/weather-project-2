"""Weather Intelligence Flask API and live demo UI."""

from __future__ import annotations

import logging
import os
from typing import Any

from flask import Flask, jsonify, render_template, request

import weather_repository
from embedding_model import DEFAULT_MODEL_NAME, embed_texts
from weather_client import WeatherClient, WeatherClientError


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("weather-intelligence")

app = Flask(__name__)

MAX_LOCATIONS_PER_SYNC = 10
MAX_DOCUMENTS_PER_LOCATION = 100
MAX_TOP_K = 20


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "service": "weather-intelligence"})


@app.get("/weather/status")
def weather_status():
    weather_repository.ensure_schema()
    return jsonify(weather_repository.status_counts())


@app.post("/weather/sync")
def sync_weather():
    body = _json_body()
    locations = body.get("locations")
    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "locations must be a non-empty list"}), 400
    if len(locations) > MAX_LOCATIONS_PER_SYNC:
        return jsonify({"error": f"At most {MAX_LOCATIONS_PER_SYNC} locations are allowed"}), 400

    limit = _bounded_integer(
        body.get("limit", 50),
        field="limit",
        minimum=1,
        maximum=MAX_DOCUMENTS_PER_LOCATION,
    )
    if isinstance(limit, tuple):
        return limit

    weather_repository.ensure_schema()
    client = WeatherClient()
    synced = 0
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for location_input in locations:
        display_location = _display_location(location_input)
        try:
            resolved, documents = client.fetch_documents(location_input, limit=limit)
            count = weather_repository.upsert_documents(documents)
            synced += count
            results.append(
                {
                    "requested": display_location,
                    "resolved": resolved.label,
                    "latitude": resolved.latitude,
                    "longitude": resolved.longitude,
                    "synced": count,
                }
            )
        except WeatherClientError as exc:
            logger.warning("Weather sync failed for %s: %s", display_location, exc)
            errors.append({"location": display_location, "error": str(exc)})

    payload = {
        "synced": synced,
        "locations": results,
        "errors": errors,
        "next_step": "Run notebooks/ingest_weather_embeddings.py before searching.",
    }
    if errors and not results:
        return jsonify(payload), 502
    return jsonify(payload)


@app.post("/weather/search")
def search_weather():
    body = _json_body()
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        return jsonify({"error": "query must be a non-empty string"}), 400
    if len(query) > 1_000:
        return jsonify({"error": "query must contain at most 1000 characters"}), 400

    top_k = _bounded_integer(body.get("top_k", 5), "top_k", 1, MAX_TOP_K)
    if isinstance(top_k, tuple):
        return top_k

    source_type = body.get("source_type") or None
    if source_type not in (None, "alert", "forecast"):
        return jsonify({"error": "source_type must be alert or forecast"}), 400
    location = body.get("location") or None
    if location is not None and (not isinstance(location, str) or len(location) > 200):
        return jsonify({"error": "location must be a string of at most 200 characters"}), 400

    weather_repository.ensure_schema()
    query_embedding = embed_texts([query.strip()])[0]
    matches = weather_repository.semantic_search(
        query_embedding,
        top_k=top_k,
        source_type=source_type,
        location=location.strip() if isinstance(location, str) else None,
    )
    return jsonify(
        {
            "query": query.strip(),
            "top_k": top_k,
            "filters": {"source_type": source_type, "location": location},
            "model": os.environ.get("WEATHER_EMBEDDING_MODEL", DEFAULT_MODEL_NAME),
            "count": len(matches),
            "matches": matches,
            "message": None if matches else "No embeddings found. Sync and run the embedding job first.",
        }
    )


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    logger.exception("Unhandled request error")
    status_code = getattr(error, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(error)}), status_code


def _json_body() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _bounded_integer(
    value: Any,
    field: str,
    minimum: int,
    maximum: int,
) -> int | tuple[Any, int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return jsonify({"error": f"{field} must be an integer"}), 400
    return min(max(parsed, minimum), maximum)


def _display_location(location: Any) -> str:
    if isinstance(location, str):
        return location
    if isinstance(location, dict):
        return str(location.get("name") or location.get("location") or location)
    return str(location)


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_RUN_HOST", "0.0.0.0"),
        port=int(os.environ.get("DATABRICKS_APP_PORT", os.environ.get("FLASK_RUN_PORT", "8000"))),
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )
