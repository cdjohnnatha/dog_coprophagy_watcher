import json
import os
import time
from datetime import UTC, datetime, timedelta

import pytest
from hasim import DASH, HA, LISBON, SCRIPTS, Ha, automation, entity_ids_in

from ellie import discovery
from helpers import FakeClock, build_rig, make_jpeg, make_visit

NOW = datetime(2026, 7, 10, 20, 0, 0, tzinfo=UTC)  # summer: Lisbon = UTC+1


@pytest.fixture
def rig(tmp_path):
    clock = FakeClock(NOW)
    rig = build_rig(tmp_path, clock)
    yield rig
    rig.store.close()


def topics(rig, name):
    return [b for t, b, _ in rig.published if t == name]


def last_state(rig):
    return topics(rig, "ellie/state")[-1]


def put_visit(rig, vid="v_20260710_190000", at="2026-07-10T19:00:00Z", **kw):
    rig.store.insert_visit(make_visit(id=vid, started_at=at, ended_at=at, **kw))
    d = rig.evidence.visit_dir(vid)
    rig.evidence.write(d / "before.jpg", make_jpeg((1, 1, 1)))
    rig.evidence.write(d / "after.jpg", make_jpeg((9, 9, 9)))


def ha_from_state(state, now=NOW):
    """Entity states HA would hold after MQTT discovery renders the real ellie/state."""
    states = {}
    for topic, p in discovery.discovery_messages("t", ["gemini:pee_v1"]):
        if "value_template" not in p:
            continue
        env = Ha({}, now).env
        raw = env.from_string(p["value_template"]).render(value_json=state).strip()
        attrs = {}
        if "json_attributes_template" in p:
            attrs = json.loads(
                env.from_string(p["json_attributes_template"]).render(value_json=state)
            )
        states[p["default_entity_id"]] = (None if raw == "None" else raw, attrs)
    return states


async def cmd(rig, name, **data):
    await rig.service.on_message(f"ellie/cmd/{name}", json.dumps(data).encode())
    await rig.service.wait_idle()


def render_payload(script_name, ha, **fields):
    """Render the mqtt.publish step of a script like HA would; returns (topic, payload str)."""
    for step in SCRIPTS[script_name]["sequence"]:
        if "variables" in step:
            for k, v in step["variables"].items():
                fields[k] = (
                    Ha.__dict__["_native"](ha.render(v))
                    if False
                    else ha.env.compile_expression(v.strip("{} ").strip(), undefined_to_none=False)(
                        **fields
                    )
                )
        if step.get("action") == "mqtt.publish":
            return step["data"]["topic"], ha.render(step["data"]["payload"], **fields)
    raise AssertionError


# ---------- sanity: things that must hold (regression guards) ----------


def test_every_entity_id_used_in_yaml_exists():
    known = {p["default_entity_id"] for _, p in discovery.discovery_messages("t", ["a"])}
    known |= {f"script.{k}" for k in SCRIPTS}
    known |= {
        "input_number.ellie_no_pee_alert_hours",
        "input_number.ellie_gemini_credit_eur",
        "input_datetime.ellie_waking_start",
        "input_datetime.ellie_waking_end",
    }
    text = "".join(
        (HA / f).read_text()
        for f in (
            "custom/automations/ellie-pee-counter.yaml",
            "docs/ellie-pee-counter/dashboard-cards.yaml",
            "custom/scripts/ellie-pee-counter.yaml",
        )
    )
    used = {e for e in entity_ids_in(text) if ".ellie_" in e}
    assert used - known == set()


async def test_discovery_templates_render_on_real_state(rig):
    put_visit(rig)
    await rig.service.publish_state()
    states = ha_from_state(last_state(rig))
    assert states["sensor.ellie_pee_today"][0] == "1"
    assert states["sensor.ellie_last_pee"][0] == "2026-07-10T19:00:00Z"
    assert states["sensor.ellie_pad_changed_at"][0] is None  # no pad yet: unknown, not "None"


async def test_manual_pee_with_datetime_selector_format(rig):
    ha = Ha({}, NOW)
    _, payload = render_payload("ellie_add_manual_pee", ha, at="2026-07-10 18:30:00")
    await rig.service.on_command("manual_pee", payload.encode())
    (v,) = rig.store.visits_started_between("2026-07-10T00:00:00Z", "2026-07-11T00:00:00Z")
    assert v["started_at"] == "2026-07-10T17:30:00Z"  # 18:30 Lisbon summer


async def test_label_from_phone_button_and_dashboard(rig):
    put_visit(rig, needs_review=True, final_outcome=None, auto_outcome="uncertain")
    await rig.service.publish_state()
    st = ha_from_state(last_state(rig))
    ha = Ha(st, NOW)
    _, payload = render_payload("ellie_label_current_review", ha, label="pee")
    await rig.service.on_command("label", payload.encode())
    assert rig.store.get_visit("v_20260710_190000")["final_outcome"] == "pee"
    act = automation("review_action")["actions"][1]["data"]["payload"]
    ev = {"data": {"action": "ELLIE_PEE_NO::v_20260710_190000", "device_id": "abc"}}

    class Ev(dict):
        pass

    payload = Ha({}, NOW).render(
        act, parts=["ELLIE_PEE_NO", "v_20260710_190000"], trigger={"event": ev}
    )
    await rig.service.on_command("label", payload.encode())
    assert rig.store.get_visit("v_20260710_190000")["final_outcome"] == "no_pee"


# ---------- defects ----------


def test_dashboard_undo_confirmation_is_under_tap_action():
    """Button card `confirmation` must live inside tap_action; at card level HA ignores it."""

    def walk(x):
        if isinstance(x, dict):
            yield x
            for v in x.values():
                yield from walk(v)
        elif isinstance(x, list):
            for v in x:
                yield from walk(v)

    for node in walk(DASH):
        if node.get("type") == "button" and "confirmation" in node:
            pytest.fail(
                "card-level `confirmation` is not a button-card option; undo runs unconfirmed"
            )


def test_dashboard_has_no_literal_magic_line_token():
    text = (HA / "docs/ellie-pee-counter/dashboard-cards.yaml").read_text()
    assert "magic___" not in text, "rendered literally in the markdown card"


def _tz(name):
    os.environ["TZ"] = name
    time.tzset()


@pytest.mark.parametrize("proc_tz", ["UTC"])
def test_no_pee_alert_time_is_lisbon_even_if_process_tz_is_utc(proc_tz):
    """`astimezone()` without argument uses the OS zone, not HA's configured zone (use as_local)."""
    old = os.environ.get("TZ")
    _tz(proc_tz)
    try:
        ha = Ha({"sensor.ellie_last_pee": ("2026-07-10T20:30:00+00:00", {})}, NOW)
        step = automation("no_pee_alert")["actions"][0]["variables"]
        last = ha.render(step["last"])

        msg = ha.env.from_string(
            automation("no_pee_alert")["actions"][1]["data"]["message"]
        ).render(last=ha._as_datetime(last))
        assert msg == "Último xixi às 21:30."
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_pad_reminder_once_per_day_uses_ha_zone():
    """last_triggered 23:30 Lisbon-local day 10 (22:30Z) vs now 00:10 Lisbon day 11 -> new day OK;
    but with a UTC process zone `.astimezone().date()` says day 10 for a 00:10 Lisbon firing
    (23:10Z day 10) and the next 07:00 is still 'same day' only by luck; assert HA-zone logic."""
    cond = automation("pad_reminder")["conditions"][2]["value_template"]
    old = os.environ.get("TZ")
    _tz("UTC")
    try:
        # fired 2026-07-10 23:30 Lisbon = 22:30Z ; now = 2026-07-11 00:10 Lisbon (local day 11)
        ha = Ha({}, datetime(2026, 7, 11, 0, 10, tzinfo=LISBON))
        ha.this = {"attributes": {"last_triggered": datetime(2026, 7, 10, 22, 30, tzinfo=UTC)}}
        # In the owner's zone the reminder fired on day 10 (23:30) -> day 11 allowed.
        # With process zone UTC the fired date is day 10, now.date() day 11 -> allowed (ok).
        # Reverse case: fired 2026-07-11 00:30 Lisbon (= 07-10 23:30Z), now 07-11 07:00 Lisbon.
        ha.this = {"attributes": {"last_triggered": datetime(2026, 7, 10, 23, 30, tzinfo=UTC)}}
        ha.now = datetime(2026, 7, 11, 7, 0, tzinfo=LISBON)
        assert ha.render(cond) == "False"  # already reminded today (local), must not repeat
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


def test_credit_warning_does_not_refire_when_attribute_is_briefly_missing():
    """Template trigger fires on every False->True edge: a restart/unavailable blip (attribute
    None -> 0) re-arms it and the 80 % notification is sent again."""
    trig = automation("gemini_credit_low")["triggers"][0]["value_template"]
    full = {
        "sensor.ellie_analysis_calls_month": ("5", {"estimated_cost_eur_total": 4.0}),
        "input_number.ellie_gemini_credit_eur": ("4.5", {}),
    }
    blip = {
        "sensor.ellie_analysis_calls_month": ("unavailable", {}),
        "input_number.ellie_gemini_credit_eur": ("4.5", {}),
    }
    assert Ha(full, NOW).render(trig) == "True"
    # the trigger may re-arm after a blip, so the CONDITIONS must stop a duplicate warning
    conditions = automation("gemini_credit_low")["conditions"]
    during_blip = Ha(blip, NOW)
    assert during_blip.render(conditions[0]["value_template"]) == "False"  # sensor unavailable
    recent = Ha(full, NOW)
    recent.this = {"attributes": {"last_triggered": NOW - timedelta(days=1)}}
    assert recent.render(conditions[1]["value_template"]) == "False"  # already warned this week
    old_warning = Ha(full, NOW)
    old_warning.this = {"attributes": {"last_triggered": NOW - timedelta(days=8)}}
    assert old_warning.render(conditions[1]["value_template"]) == "True"


async def test_reanalysis_that_decides_clears_the_stale_phone_notification(rig):
    rig.service.cfg["validation_mode"] = False
    put_visit(
        rig,
        vid="v_20260710_190000",
        state="DECIDED",
        auto_outcome="unknown",
        final_outcome="unknown",
        needs_review=True,
    )
    await cmd(rig, "reanalyse", visit_id="v_20260710_190000")
    assert rig.store.get_visit("v_20260710_190000")["final_outcome"] == "pee"
    assert {"visit_id": "v_20260710_190000"} in topics(rig, "ellie/review/resolved"), (
        "notification 'não consegui analisar' stays on the phones"
    )


async def test_last_pee_photo_follows_last_pee_after_manual_pee(rig):
    put_visit(rig)
    await rig.service.publish_state()
    await cmd(rig, "manual_pee", at="2026-07-10 20:30:00")
    st = last_state(rig)
    assert st["last_pee"]["visit_id"] != "v_20260710_190000"
    assert st["last_pee"]["image"] is None
    # HA keeps showing the retained photo of the OLDER visit under 'Último xixi'
    imgs = [b for t, b, _ in rig.published if t == "ellie/image/last_pee"]
    assert imgs[-2] != imgs[-1], (
        "dashboard still shows the photo of a different pee"
    )  # a placeholder replaces it


async def test_double_tap_on_pad_confirmation_makes_undo_ineffective(rig):
    rig.store.insert_presence(
        {"id": "s_20260710_190000", "started_at": "2026-07-10T19:00:00Z", "state": "DECIDED"}
    )
    rig.store.update_presence("s_20260710_190000", ended_at="2026-07-10T19:00:30Z")
    await cmd(rig, "pad_label", presence_id="s_20260710_190000", label="replaced")
    await cmd(rig, "pad_label", presence_id="s_20260710_190000", label="replaced")  # second phone
    await cmd(rig, "pad_undo")
    assert last_state(rig)["pad"] is None, "undo left a duplicate change current"


async def test_back_dated_pad_change_is_silently_ignored_when_a_newer_exists(rig):
    await cmd(rig, "pad_changed", at="2026-07-10 10:00:00")
    await cmd(rig, "pad_changed", at="2026-07-06 10:00:00")  # owner corrects: it was 4 days ago
    # design: the most recent change is the current pad; to correct it the owner undoes it first
    assert last_state(rig)["pad"]["changed_at"] == "2026-07-10T09:00:00Z"
    await cmd(rig, "pad_undo")
    assert last_state(rig)["pad"]["changed_at"] == "2026-07-06T09:00:00Z"
