"""National Weather Service harvesting and normalization client."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


NWS_BASE_URL = "https://api.weather.gov"
GEOCODER_URL = (
    "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/"
    "findAddressCandidates"
)
DEFAULT_TIMEOUT_SECONDS = 20


class WeatherClientError(RuntimeError):
    """A user-safe error raised for geocoding or NWS failures."""


@dataclass(frozen=True)
class ResolvedLocation:
    label: str
    latitude: float
    longitude: float


class WeatherClient:
    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent or os.environ.get(
            "NWS_USER_AGENT",
            "weather-intelligence-homework/1.0 (https://github.com/Revanth14)",
        )
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update(
            {"User-Agent": self.user_agent, "Accept": "application/geo+json"}
        )

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise WeatherClientError(f"Weather service request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise WeatherClientError("Weather service returned an unexpected response")
        return payload

    def resolve_location(self, location: str | dict[str, Any]) -> ResolvedLocation:
        if isinstance(location, dict):
            label = str(location.get("name") or location.get("location") or "").strip()
            try:
                latitude = float(location["latitude"])
                longitude = float(location["longitude"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WeatherClientError(
                    "Coordinate locations require numeric latitude and longitude"
                ) from exc
            self._validate_coordinates(latitude, longitude)
            return ResolvedLocation(label or f"{latitude:.4f},{longitude:.4f}", latitude, longitude)

        if not isinstance(location, str) or not location.strip():
            raise WeatherClientError("Each location must be a city/state string or coordinate object")

        value = location.strip()
        coordinate_parts = [part.strip() for part in value.split(",")]
        if len(coordinate_parts) == 2:
            try:
                latitude, longitude = map(float, coordinate_parts)
            except ValueError:
                pass
            else:
                self._validate_coordinates(latitude, longitude)
                return ResolvedLocation(value, latitude, longitude)
        return self._geocode(value)

    @lru_cache(maxsize=128)
    def _geocode(self, query: str) -> ResolvedLocation:
        payload = self._get_json(
            GEOCODER_URL,
            {
                "SingleLine": query,
                "f": "json",
                "countryCode": "USA",
                "maxLocations": 1,
                "outFields": "Match_addr,Addr_type",
            },
        )
        candidates = payload.get("candidates") or []
        if not candidates:
            raise WeatherClientError(f"Could not resolve location: {query}")
        candidate = candidates[0]
        if float(candidate.get("score") or 0) < 70:
            raise WeatherClientError(f"Location match was too uncertain: {query}")
        point = candidate.get("location") or {}
        try:
            latitude = float(point["y"])
            longitude = float(point["x"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherClientError("Geocoder returned an invalid coordinate") from exc
        self._validate_coordinates(latitude, longitude)
        label = candidate.get("address") or query
        return ResolvedLocation(str(label), latitude, longitude)

    @staticmethod
    def _validate_coordinates(latitude: float, longitude: float) -> None:
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise WeatherClientError("Latitude or longitude is outside its valid range")

    def fetch_documents(
        self,
        location_input: str | dict[str, Any],
        limit: int = 50,
    ) -> tuple[ResolvedLocation, list[dict[str, Any]]]:
        location = self.resolve_location(location_input)
        lat_lon = f"{location.latitude:.4f},{location.longitude:.4f}"
        point = self._get_json(f"{NWS_BASE_URL}/points/{lat_lon}")
        properties = point.get("properties") or {}
        forecast_url = properties.get("forecast")
        if not forecast_url:
            raise WeatherClientError(f"NWS has no forecast grid for {location.label}")

        alerts_payload = self._get_json(
            f"{NWS_BASE_URL}/alerts/active",
            {"point": lat_lon},
        )
        forecast_payload = self._get_json(str(forecast_url))

        documents: list[dict[str, Any]] = []
        for feature in alerts_payload.get("features") or []:
            document = self._normalize_alert(location, feature)
            if document:
                documents.append(document)
        for period in (forecast_payload.get("properties") or {}).get("periods") or []:
            document = self._normalize_forecast(location, forecast_payload, period)
            if document:
                documents.append(document)
        return location, documents[:limit]

    @staticmethod
    def _normalize_alert(
        location: ResolvedLocation, feature: dict[str, Any]
    ) -> dict[str, Any] | None:
        properties = feature.get("properties") or {}
        description = str(properties.get("description") or "").strip()
        instruction = str(properties.get("instruction") or "").strip()
        narrative = description
        if instruction:
            narrative = f"{description}\n\nInstructions: {instruction}" if description else instruction
        if not narrative:
            return None

        source_id = str(feature.get("id") or properties.get("id") or "")
        if not source_id:
            source_id = hashlib.sha256(
                f"{properties.get('event')}|{properties.get('sent')}|{narrative}".encode()
            ).hexdigest()
        document_id = "nws-alert-" + hashlib.sha256(source_id.encode()).hexdigest()[:32]
        return _document(
            document_id=document_id,
            source_id=source_id,
            location=location,
            source_type="alert",
            headline=str(properties.get("headline") or properties.get("event") or "Weather alert"),
            narrative=narrative,
            issued_at=properties.get("sent"),
            effective_at=properties.get("effective") or properties.get("onset"),
            expires_at=properties.get("expires") or properties.get("ends"),
            payload=feature,
        )

    @staticmethod
    def _normalize_forecast(
        location: ResolvedLocation,
        forecast_payload: dict[str, Any],
        period: dict[str, Any],
    ) -> dict[str, Any] | None:
        narrative = str(period.get("detailedForecast") or "").strip()
        if not narrative:
            return None
        start_time = str(period.get("startTime") or "")
        source_id = (
            f"{location.latitude:.4f},{location.longitude:.4f}|"
            f"{start_time}"
        )
        document_id = "nws-forecast-" + hashlib.sha256(source_id.encode()).hexdigest()[:32]
        forecast_properties = forecast_payload.get("properties") or {}
        headline_parts = [period.get("name"), period.get("shortForecast")]
        headline = ": ".join(str(part) for part in headline_parts if part) or "Weather forecast"
        payload = {
            "period": period,
            "forecast_generated_at": forecast_properties.get("generatedAt"),
            "forecast_update_time": forecast_properties.get("updateTime"),
        }
        return _document(
            document_id=document_id,
            source_id=source_id,
            location=location,
            source_type="forecast",
            headline=headline,
            narrative=narrative,
            issued_at=forecast_properties.get("updateTime") or forecast_properties.get("generatedAt"),
            effective_at=period.get("startTime"),
            expires_at=period.get("endTime"),
            payload=payload,
        )


def _document(
    *,
    document_id: str,
    source_id: str,
    location: ResolvedLocation,
    source_type: str,
    headline: str,
    narrative: str,
    issued_at: str | None,
    effective_at: str | None,
    expires_at: str | None,
    payload: dict[str, Any],
) -> dict[str, Any]:
    content_hash = hashlib.sha256(narrative.encode("utf-8")).hexdigest()
    return {
        "id": document_id,
        "source_id": source_id,
        "location": location.label,
        "latitude": location.latitude,
        "longitude": location.longitude,
        "source_type": source_type,
        "headline": headline,
        "narrative_text": narrative,
        "issued_at": issued_at,
        "effective_at": effective_at,
        "expires_at": expires_at,
        "content_hash": content_hash,
        "payload": payload,
        "resolved_location": asdict(location),
    }
