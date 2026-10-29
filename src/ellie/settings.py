"""Add-on options, Supervisor MQTT credentials and owner-adjustable runtime settings."""

import json
import logging
import math
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)

SUPERVISOR_MQTT_URL = "http://supervisor/services/mqtt"

RUNTIME_DEFAULTS: dict[str, object] = {
    "confidence_threshold": 0.80,
    "merge_gap_s": 60,
    "deciding_analyser": "gemini:pee_v1",
    "validation_mode": True,
    "pad_change_days": 3,
}
RUNTIME_RANGES: dict[str, tuple[float, float]] = {
    "confidence_threshold": (0.50, 0.99),
    "merge_gap_s": (15, 300),
    "pad_change_days": (1, 7),
}


class SettingsError(Exception):
    """Options or MQTT service data are missing or invalid."""


class Options(BaseModel):
    """Add-on options from /data/options.json (defaults mirror config.yaml)."""

    model_config = ConfigDict(extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    area_x1: float = 0.244
    area_y1: float = 0.041
    area_x2: float = 0.753
    area_y2: float = 0.795
    frigate_url: str = "http://ccab4aaf-frigate:5000"
    go2rtc_url: str = "http://ccab4aaf-frigate:1984"
    camera: str = "ellie"
    log_level: str = "info"
    usd_to_eur: float = 0.9

    @property
    def analysis_area(self) -> tuple[float, float, float, float]:
        return (self.area_x1, self.area_y1, self.area_x2, self.area_y2)


def load_options(path: str = "/data/options.json") -> Options:
    """Load options; a missing file gives the defaults."""
    file = Path(path)
    if not file.exists():
        return Options()
    try:
        return Options.model_validate(json.loads(file.read_text()))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise SettingsError(f"invalid options file {path}: {type(exc).__name__}") from exc


class MqttService(BaseModel):
    host: str
    port: int
    username: str
    password: str


async def fetch_mqtt_service(client: httpx.AsyncClient, token: str | None) -> MqttService:
    """Read the broker credentials from the Supervisor MQTT service."""
    if not token:
        raise SettingsError("SUPERVISOR_TOKEN is not set; cannot read the MQTT service")
    try:
        response = await client.get(
            SUPERVISOR_MQTT_URL, headers={"Authorization": f"Bearer {token}"}
        )
    except httpx.HTTPError as exc:
        raise SettingsError(f"MQTT service request failed: {type(exc).__name__}") from exc
    if response.status_code != 200:
        raise SettingsError(f"MQTT service returned HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise SettingsError("MQTT service returned invalid JSON") from exc
    if not isinstance(body, dict) or body.get("result") != "ok":
        raise SettingsError("MQTT service result is not ok")
    data = body.get("data")
    try:
        return MqttService.model_validate(data)
    except ValidationError as exc:
        # Report field names only: error details can echo input values.
        missing = sorted({str(e["loc"][0]) for e in exc.errors() if e["loc"]})
        raise SettingsError(f"MQTT service data missing or invalid fields: {missing}") from None


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().upper() in ("ON", "OFF"):
        return value.strip().upper() == "ON"
    raise ValueError(f"invalid validation_mode: {value!r}")


def clamp_setting(key: str, value: object) -> tuple[object, bool]:
    """Coerce to the default's type and clamp to the range; returns (value, was_clamped)."""
    if key not in RUNTIME_DEFAULTS:
        raise KeyError(key)
    default = RUNTIME_DEFAULTS[key]
    if isinstance(default, bool):
        return _to_bool(value), False
    if isinstance(default, str):
        return str(value), False
    low, high = RUNTIME_RANGES[key]
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{key} must be a finite number")
    clamped = min(max(number, low), high)
    result: object = round(clamped) if isinstance(default, int) else clamped
    if clamped != number:
        logger.warning(
            "setting %s=%s outside [%s, %s]; clamped to %s", key, value, low, high, result
        )
    return result, clamped != number
