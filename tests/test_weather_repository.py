import pytest

import weather_repository


def test_semantic_search_requires_384_dimensions():
    with pytest.raises(ValueError, match="384-dimensional"):
        weather_repository.semantic_search([0.1, 0.2], top_k=5)


def test_semantic_search_uses_cosine_operator_and_parameters(monkeypatch):
    captured = {}

    def fake_query(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return [{"similarity": 0.75}]

    monkeypatch.setattr(weather_repository.lakebase, "run_query", fake_query)
    rows = weather_repository.semantic_search(
        [0.0] * 384,
        top_k=7,
        source_type="alert",
        location="Austin",
    )
    assert "<=>" in captured["sql"]
    assert captured["params"][-1] == 7
    assert rows[0]["similarity"] == 0.75


def test_document_upsert_supplies_synced_at_expression(monkeypatch):
    captured = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        def commit(self):
            captured["committed"] = True

    class FakeConnectionContext:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, *args):
            return None

    def fake_execute_values(cursor, sql, values, template, page_size):
        captured.update(sql=sql, values=values, template=template, page_size=page_size)

    monkeypatch.setattr(
        weather_repository.lakebase,
        "get_connection",
        lambda: FakeConnectionContext(),
    )
    monkeypatch.setattr(weather_repository, "execute_values", fake_execute_values)

    document = {
        "id": "doc-1",
        "source_id": "source-1",
        "location": "Chicago, IL",
        "latitude": 41.88,
        "longitude": -87.63,
        "source_type": "forecast",
        "headline": "Rain likely",
        "narrative_text": "Rain is likely this afternoon.",
        "issued_at": "2026-08-09T10:00:00Z",
        "effective_at": "2026-08-09T12:00:00Z",
        "expires_at": "2026-08-09T18:00:00Z",
        "content_hash": "abc123",
        "payload": {"period": 1},
    }

    assert weather_repository.upsert_documents([document]) == 1
    assert len(captured["values"][0]) == 13
    assert captured["template"].count("%s") == 13
    assert captured["template"].endswith("now())")
    assert captured["committed"] is True
