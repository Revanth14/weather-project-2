from copy import deepcopy

from weather_client import GEOCODER_URL, ResolvedLocation, WeatherClient


LOCATION = ResolvedLocation("Chicago, IL", 41.8781, -87.6298)


def test_geocoder_uses_the_public_anonymous_endpoint():
    assert GEOCODER_URL.startswith("https://geocode.arcgis.com/")


def test_alert_normalization_combines_description_and_instruction():
    feature = {
        "id": "https://api.weather.gov/alerts/test-alert",
        "properties": {
            "event": "Flash Flood Warning",
            "headline": "Flash Flood Warning issued for Cook County",
            "description": "Heavy rain may flood low-lying roads.",
            "instruction": "Move to higher ground.",
            "sent": "2026-08-09T10:00:00Z",
            "effective": "2026-08-09T10:00:00Z",
            "expires": "2026-08-09T12:00:00Z",
        },
    }
    document = WeatherClient._normalize_alert(LOCATION, feature)
    assert document["source_type"] == "alert"
    assert "Instructions: Move to higher ground." in document["narrative_text"]
    assert document["payload"] == feature
    assert len(document["content_hash"]) == 64


def test_forecast_id_stays_stable_if_period_number_changes():
    payload = {
        "properties": {
            "updateTime": "2026-08-09T10:00:00Z",
            "periods": [],
        }
    }
    period = {
        "number": 1,
        "name": "Tonight",
        "startTime": "2026-08-09T18:00:00-05:00",
        "endTime": "2026-08-10T06:00:00-05:00",
        "shortForecast": "Showers likely",
        "detailedForecast": "Showers likely, with locally heavy rainfall.",
    }
    first = WeatherClient._normalize_forecast(LOCATION, payload, period)
    updated_period = deepcopy(period)
    updated_period["number"] = 3
    updated_period["detailedForecast"] = "Showers and thunderstorms likely."
    second = WeatherClient._normalize_forecast(LOCATION, payload, updated_period)

    assert first["id"] == second["id"]
    assert first["content_hash"] != second["content_hash"]


def test_blank_forecast_is_not_stored():
    assert WeatherClient._normalize_forecast(
        LOCATION,
        {"properties": {}},
        {"startTime": "2026-08-09T18:00:00Z", "detailedForecast": ""},
    ) is None
