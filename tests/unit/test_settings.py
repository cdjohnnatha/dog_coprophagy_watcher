import json

import httpx
import pytest

from ellie.settings import (
    MqttService,
    Options,
    SettingsError,
    clamp_setting,
    fetch_mqtt_service,
    load_options,
)

PASSWORD = "s3cret-pw"
GOOD = {"host": "core-mosquitto", "port": 1883, "username": "u", "password": PASSWORD}


def test_defaults():
    o = Options()
    assert o.gemini_model == "gemini-3.5-flash-lite"
    assert o.camera == "ellie"
    assert o.analysis_area == (0.244, 0.041, 0.753, 0.795)


def test_missing_file_gives_defaults(tmp_path):
    assert load_options(str(tmp_path / "none.json")) == Options()


def test_load_from_file_ignores_extra(tmp_path):
    f = tmp_path / "o.json"
    f.write_text(json.dumps({"camera": "x", "area_x1": 0.1, "unknown": 1}))
    o = load_options(str(f))
    assert o.camera == "x"
    assert o.analysis_area[0] == 0.1


def test_invalid_json(tmp_path):
    f = tmp_path / "o.json"
    f.write_text("{nope")
    with pytest.raises(SettingsError):
        load_options(str(f))


@pytest.mark.parametrize(
    ("key", "value", "expected", "clamped"),
    [
        ("confidence_threshold", 0.1, 0.50, True),
        ("confidence_threshold", 5, 0.99, True),
        ("confidence_threshold", 0.7, 0.7, False),
        ("merge_gap_s", 5, 15, True),
        ("merge_gap_s", 999, 300, True),
        ("merge_gap_s", 90.0, 90, False),
        ("merge_gap_s", "120", 120, False),
        ("pad_change_days", 0, 1, True),
        ("pad_change_days", 9, 7, True),
        ("pad_change_days", 3, 3, False),
        ("validation_mode", "off", False, False),
        ("validation_mode", "ON", True, False),
        ("validation_mode", False, False, False),
        ("deciding_analyser", "local:x", "local:x", False),
    ],
)
def test_clamp(key, value, expected, clamped):
    result, was = clamp_setting(key, value)
    assert result == expected
    assert type(result) is type(expected)
    assert was is clamped


def test_clamp_unknown_key():
    with pytest.raises(KeyError):
        clamp_setting("nope", 1)


def test_clamp_bad_bool():
    with pytest.raises(ValueError):
        clamp_setting("validation_mode", "maybe")


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_success():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["Authorization"]
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"result": "ok", "data": {**GOOD, "ssl": False}})

    async with _client(handler) as c:
        svc = await fetch_mqtt_service(c, "tok")
    assert svc == MqttService(**GOOD)
    assert seen == {"auth": "Bearer tok", "url": "http://supervisor/services/mqtt"}


async def test_fetch_no_token():
    async with _client(lambda r: httpx.Response(200)) as c:
        with pytest.raises(SettingsError, match="SUPERVISOR_TOKEN"):
            await fetch_mqtt_service(c, None)


async def test_fetch_bad_status():
    async with _client(lambda r: httpx.Response(401)) as c:
        with pytest.raises(SettingsError, match="401"):
            await fetch_mqtt_service(c, "tok")


async def test_fetch_result_not_ok():
    async with _client(lambda r: httpx.Response(200, json={"result": "error"})) as c:
        with pytest.raises(SettingsError, match="not ok"):
            await fetch_mqtt_service(c, "tok")


async def test_fetch_missing_fields_hides_password():
    data = {"host": "h", "port": "not-a-number", "password": PASSWORD}
    async with _client(lambda r: httpx.Response(200, json={"result": "ok", "data": data})) as c:
        with pytest.raises(SettingsError) as info:
            await fetch_mqtt_service(c, "tok")
    assert "username" in str(info.value)
    assert PASSWORD not in str(info.value)
    assert PASSWORD not in repr(info.value)


async def test_fetch_network_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    async with _client(handler) as c:
        with pytest.raises(SettingsError):
            await fetch_mqtt_service(c, "tok")
