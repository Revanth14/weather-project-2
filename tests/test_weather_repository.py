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

