"""Production launcher that resolves the Databricks Apps port at runtime."""

from __future__ import annotations

import os
from typing import Any

from gunicorn.app.base import BaseApplication

from app import app


class WeatherApplication(BaseApplication):
    def __init__(self, application: Any, options: dict[str, Any]) -> None:
        self.application = application
        self.options = options
        super().__init__()

    def load_config(self) -> None:
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key, value)

    def load(self) -> Any:
        return self.application


def main() -> None:
    port = int(
        os.environ.get(
            "DATABRICKS_APP_PORT",
            os.environ.get("FLASK_RUN_PORT", "8000"),
        )
    )
    WeatherApplication(
        app,
        {
            "bind": f"0.0.0.0:{port}",
            "workers": 1,
            "threads": 4,
            "worker_class": "gthread",
            "timeout": 180,
            "accesslog": "-",
            "errorlog": "-",
            "capture_output": True,
        },
    ).run()


if __name__ == "__main__":
    main()

