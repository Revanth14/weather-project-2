from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/sync_weather_documents.py",
        "notebooks/ingest_weather_embeddings.py",
    ],
)
def test_job_entrypoint_executes_without_file_global(relative_path, monkeypatch):
    """Mirror the Databricks runner, which execs files without defining __file__."""

    monkeypatch.chdir(PROJECT_ROOT)
    script_path = PROJECT_ROOT / relative_path
    namespace = {"__name__": "databricks_job_test"}
    exec(compile(script_path.read_bytes(), str(script_path), "exec"), namespace)
    assert namespace["PROJECT_ROOT"] == PROJECT_ROOT
