"""Contract tests for the MQTT discovery payloads."""

import json

import jinja2
import pytest

from ellie.discovery import discovery_messages, removal_messages
from ellie.settings import RUNTIME_RANGES

ANALYSERS = ["gemini", "claude"]
IDS = {
    "sensor.ellie_pee_today",
    "sensor.ellie_last_pee",
    "sensor.ellie_pee_daily_history",
    "sensor.ellie_review_queue",
    "sensor.ellie_pee_accuracy",
    "sensor.ellie_last_visit",
    "sensor.ellie_analysis_calls_month",
    "binary_sensor.ellie_at_tray",
    "number.ellie_confidence_threshold",
    "number.ellie_merge_gap",
    "select.ellie_deciding_analyser",
    "switch.ellie_validation_mode",
    "sensor.ellie_pad_changed_at",
    "sensor.ellie_pad_change_due",
    "number.ellie_pad_change_days",
    "image.ellie_last_pee_photo",
    "image.ellie_review_before",
    "image.ellie_review_after",
}
ABBREV = {
    "dev",
    "ids",
    "uniq_id",
    "avty_t",
    "stat_t",
    "val_tpl",
    "json_attr_t",
    "json_attr_tpl",
    "stat_cla",
    "dev_cla",
    "ent_cat",
    "cmd_t",
    "def_ent_id",
}

FULL = {
    "pee_today": 3,
    "visits_today": 5,
    "uncertain_open": 1,
    "at_tray": True,
    "last_pee": {"at": "2026-10-29T08:00:00Z", "visit_id": "v1", "image": "a.jpg", "reason": "r"},
    "last_visit": {
        "visit_id": "v1",
        "auto_outcome": "pee",
        "final_outcome": "pee",
        "started_at": "2026-10-29T07:59:00Z",
        "ended_at": "2026-10-29T08:00:00Z",
        "reason": "r",
        "poop_observed": False,
        "needs_review": False,
        "deciding_analyser": "gemini",
    },
    "review_queue": {
        "count": 2,
        "items": [
            {
                "visit_id": "v2",
                "time": "t",
                "auto_outcome": "uncertain",
                "reason": "r",
                "before": "b.jpg",
                "after": "a.jpg",
            }
        ],
    },
    "daily_history": {"days": [{"date": "2026-10-28", "pees": 4, "visits": 6, "corrected": 0}]},
    "accuracy": {
        "pct": 92.5,
        "window_days": 30,
        "analysers": {"gemini": {"labelled": 10, "agree": 9, "accuracy": 0.9}},
    },
    "analysis_calls": {
        "calls": 12,
        "tokens_in": 100,
        "tokens_out": 50,
        "estimated_cost_eur": 0.5,
        "estimated_cost_eur_total": 1.5,
    },
    "settings": {
        "confidence_threshold": 0.8,
        "merge_gap_s": 60,
        "deciding_analyser": "gemini",
        "validation_mode": True,
        "pad_change_days": 3,
    },
    "pad": {
        "changed_at": "2026-10-27T10:00:00Z",
        "due_at": "2026-10-30T10:00:00Z",
        "source": "manual",
        "days_in_use": 2.0,
        "pees_since": 7,
        "pad_color": "blue",
        "image": "p.jpg",
        "history": [],
    },
}
NULLED = {**FULL, "last_pee": None, "last_visit": None, "pad": None}

MSGS = discovery_messages("1.2.3", ANALYSERS)
BY_ID = {p["default_entity_id"]: p for _, p in MSGS}
ENV = jinja2.Environment()
ENV.policies["json.dumps_function"] = json.dumps


def _render(template: str, state: dict) -> str:
    return ENV.from_string(template).render(value_json=state).strip()


def test_entities_exist_once() -> None:
    ids = [p["default_entity_id"] for _, p in MSGS]
    assert len(ids) == len(set(ids))
    assert set(ids) == IDS


def test_unique_ids_and_topics() -> None:
    uids = [p["unique_id"] for _, p in MSGS]
    assert len(uids) == len(set(uids))
    for topic, p in MSGS:
        assert p["unique_id"].startswith("ellie_v3_")
        component = p["default_entity_id"].split(".")[0]
        assert topic == f"homeassistant/{component}/ellie/{p['unique_id']}/config"
    assert removal_messages() == [t for t, _ in MSGS]


def test_common_keys_and_no_abbreviations() -> None:
    for _, p in MSGS:
        for key in ("device", "availability_topic", "origin", "name"):
            assert key in p
        if p["default_entity_id"].startswith("image."):  # bytes come from image_topic
            assert p["image_topic"].startswith("ellie/image/") and "state_topic" not in p
        else:
            assert "state_topic" in p
        assert p["origin"]["sw_version"] == "1.2.3"
        assert not ABBREV & set(p)
        json.dumps(p)


def test_select_and_number_ranges() -> None:
    assert BY_ID["select.ellie_deciding_analyser"]["options"] == ANALYSERS
    for key, eid in [
        ("confidence_threshold", "number.ellie_confidence_threshold"),
        ("merge_gap_s", "number.ellie_merge_gap"),
        ("pad_change_days", "number.ellie_pad_change_days"),
    ]:
        p = BY_ID[eid]
        assert (p["min"], p["max"]) == RUNTIME_RANGES[key]
        assert p["command_topic"] == f"ellie/set/{key}"


@pytest.mark.parametrize("state", [FULL, NULLED], ids=["full", "nulls"])
def test_templates_render(state: dict) -> None:
    for p in BY_ID.values():
        if "value_template" in p:
            _render(p["value_template"], state)
        if "json_attributes_template" in p:
            assert isinstance(json.loads(_render(p["json_attributes_template"], state)), dict)


def test_rendered_values() -> None:
    def val(eid: str, state: dict = FULL) -> str:
        return _render(BY_ID[eid]["value_template"], state)

    assert val("sensor.ellie_pee_today") == "3"
    assert val("sensor.ellie_review_queue") == "2"
    assert val("sensor.ellie_pee_accuracy") == "92.5"
    assert val("binary_sensor.ellie_at_tray") == "ON"
    assert val("switch.ellie_validation_mode") == "ON"
    assert val("sensor.ellie_last_pee") == "2026-10-29T08:00:00Z"
    assert val("sensor.ellie_last_pee", NULLED) == "None"
    attrs = json.loads(
        _render(BY_ID["sensor.ellie_review_queue"]["json_attributes_template"], FULL)
    )
    assert attrs["items"][0]["visit_id"] == "v2"
