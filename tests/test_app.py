from dataclasses import dataclass

import pytest

import app as app_module


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(app_module.weather_repository, "ensure_schema", lambda: None)
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def test_health_check(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_dashboard_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Find the forecast by" in response.data
    assert b"POST /weather/sync" in response.data


def test_sync_requires_locations(client):
    response = client.post("/weather/sync", json={})
    assert response.status_code == 400
    assert "locations" in response.get_json()["error"]


def test_sync_reports_success_and_upserts(client, monkeypatch):
    @dataclass
    class Location:
        label: str = "Chicago, IL"
        latitude: float = 41.8781
        longitude: float = -87.6298

    class FakeWeatherClient:
        def fetch_documents(self, location, limit):
            assert location == "Chicago, IL"
            assert limit == 100  # clamped from the request
            return Location(), [{"id": "one"}, {"id": "two"}]

    monkeypatch.setattr(app_module, "WeatherClient", FakeWeatherClient)
    monkeypatch.setattr(app_module.weather_repository, "upsert_documents", lambda docs: len(docs))
    response = client.post(
        "/weather/sync",
        json={"locations": ["Chicago, IL"], "limit": 999},
    )
    assert response.status_code == 200
    assert response.get_json()["synced"] == 2


def test_search_validates_source_type(client):
    response = client.post(
        "/weather/search",
        json={"query": "snow", "source_type": "observation"},
    )
    assert response.status_code == 400


def test_search_clamps_top_k_and_returns_ranked_matches(client, monkeypatch):
    monkeypatch.setattr(app_module, "embed_texts", lambda texts: [[0.0] * 384])
    captured = {}

    def fake_search(vector, top_k, source_type, location):
        captured.update(top_k=top_k, source_type=source_type, location=location)
        return [
            {
                "id": "doc-1",
                "location": "Chicago, IL",
                "headline": "Rain",
                "source_type": "forecast",
                "chunk_text": "Rain is likely.",
                "similarity": 0.83,
            }
        ]

    monkeypatch.setattr(app_module.weather_repository, "semantic_search", fake_search)
    response = client.post(
        "/weather/search",
        json={"query": "flooding", "top_k": 999, "location": " Chicago "},
    )
    assert response.status_code == 200
    assert response.get_json()["count"] == 1
    assert captured == {"top_k": 20, "source_type": None, "location": "Chicago"}
