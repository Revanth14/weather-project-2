import run_app


def test_launcher_resolves_databricks_port(monkeypatch):
    captured = {}

    class FakeApplication:
        def __init__(self, application, options):
            captured["application"] = application
            captured["options"] = options

        def run(self):
            captured["ran"] = True

    monkeypatch.setenv("DATABRICKS_APP_PORT", "9123")
    monkeypatch.setattr(run_app, "WeatherApplication", FakeApplication)
    run_app.main()

    assert captured["options"]["bind"] == "0.0.0.0:9123"
    assert captured["options"]["workers"] == 1
    assert captured["ran"] is True
