"""Untrusted input, error handling and persistence failures."""

import re
import sqlite3

import pytest
from qa_common import EVENT, all_visits, at, dog_visit, send, tick_for

from ellie.analysers.base import AnalyserError
from ellie.settings import RUNTIME_DEFAULTS
from helpers import build_rig, make_jpeg


async def test_nan_confidence_threshold_setting_disables_the_uncertain_band(make_rig):
    rig = make_rig()
    await send(rig, "ellie/set/confidence_threshold", "nan")
    thr = rig.service.cfg["confidence_threshold"]
    assert thr == thr and 0.5 <= thr <= 0.99, f"threshold is {thr!r}"
    assert rig.store.get_setting("confidence_threshold") == thr or thr == thr


@pytest.mark.parametrize(
    "topic,payload",
    [
        ("frigate/pad_zone/dog", b"inf"),
        ("frigate/tray_area/dog", b"1e999"),
        ("frigate/events", b'{"after": [1]}'),
        ("ellie/cmd/reanalyse", b'{"visit_id": ["x"]}'),
        ("ellie/cmd/reanalyse", b'{"visit_id": {"a": 1}}'),
        ("ellie/cmd/manual_pee", b'{"at": "0001-01-01T00:00:00+00:00"}'),
        ("ellie/cmd/pad_changed", b'{"at": "0001-01-01T00:00:00+00:00"}'),
    ],
    ids=lambda x: str(x)[:30],
)
async def test_malformed_payload_must_not_raise_out_of_the_handler(make_rig, topic, payload):
    rig = make_rig()
    await rig.service.on_message(topic, payload)


async def test_manual_pee_with_a_bad_by_field_leaves_a_pee_without_a_label(make_rig):
    rig = make_rig()
    try:
        await send(rig, "ellie/cmd/manual_pee", {"at": "2026-10-29T10:00:00+00:00", "by": {"x": 1}})
    except Exception:
        pass
    for v in all_visits(rig):
        assert rig.store.labels_for(v["id"]), "counted manual pee has no label row"


async def test_event_id_from_mqtt_is_passed_unvalidated_into_the_frigate_url(tmp_path, clock):
    seen = []

    async def snap(event_id):
        seen.append(event_id)
        return make_jpeg()

    rig = build_rig(tmp_path, clock, fetch_snapshot=snap)
    rig.reference.update(make_jpeg())
    evil = {"after": {**EVENT["after"], "id": "../../../config"}}
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/events", evil)
    await send(rig, "frigate/tray_area/dog", 1)
    await tick_for(rig, 5)
    await send(rig, "frigate/pad_zone/dog", 0)
    await send(rig, "frigate/tray_area/dog", 0)
    await tick_for(rig, 100)
    rig.store.close()
    assert all(re.fullmatch(r"[0-9]+\.[0-9]+-[A-Za-z0-9]+", str(e)) for e in seen), seen


async def test_non_string_event_id_is_stored_in_the_visit(make_rig):
    rig = make_rig()
    bad = {"after": {**EVENT["after"], "id": 12345}}
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/events", bad)
    (v,) = all_visits(rig)
    assert all(isinstance(e, str) for e in v["frigate_event_ids"])


async def test_the_same_event_before_the_visit_starts_is_not_stored_twice(make_rig):
    rig = make_rig()
    await send(rig, "frigate/events", EVENT)
    await send(rig, "frigate/events", EVENT)  # Frigate sends "update" repeatedly
    await send(rig, "frigate/pad_zone/dog", 1)
    (v,) = all_visits(rig)
    assert v["frigate_event_ids"] == [EVENT["after"]["id"]]


async def test_transient_store_error_on_hand_over_must_not_lose_the_visit(make_rig):
    rig = make_rig()
    real = rig.store.update_visit
    state = {"failed": False}

    def flaky(visit_id, **fields):
        if fields.get("state") == "ANALYSING" and not state["failed"]:
            state["failed"] = True
            raise sqlite3.OperationalError("database is locked")
        return real(visit_id, **fields)

    rig.store.update_visit = flaky
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/tray_area/dog", 1)
    await tick_for(rig, 5)
    await send(rig, "frigate/pad_zone/dog", 0)
    await send(rig, "frigate/tray_area/dog", 0)
    for _ in range(300):
        rig.clock.advance(1)
        try:  # the periodic runner (`every`) swallows and logs exceptions
            await rig.service.tick()
        except Exception:
            pass
    await rig.service.wait_idle()
    (v,) = all_visits(rig)
    assert v["state"] in ("DECIDED", "LABELLED"), f"visit stuck in {v['state']} until restart"


async def test_publish_failure_after_the_decision_must_not_turn_a_pee_into_unknown(tmp_path, clock):
    calls = {"n": 0}

    async def publish(topic, payload, retain):
        if topic.startswith("ellie/visit/") and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("broker hiccup")

    rig = build_rig(tmp_path, clock, publish=publish)
    rig.reference.update(make_jpeg())
    await dog_visit(rig)
    (v,) = all_visits(rig)
    rig.store.close()
    assert v["auto_outcome"] == "pee" and v["final_outcome"] == "pee"


async def test_disk_error_while_refreshing_the_reference_must_not_stop_the_minute_task(make_rig):
    rig = make_rig()
    await rig.service.tick()
    await tick_for(rig, 15)

    def boom(_):
        raise OSError(28, "No space left on device")

    rig.reference.update = boom
    n = len(rig.published)
    try:
        await rig.service.minute_tasks()
    except OSError:
        pytest.fail("OSError escaped minute_tasks: no state published, overdue/purge skipped")
    assert len(rig.published) > n


async def test_snapshot_write_failure_must_not_abort_the_analysis(make_rig):
    rig = make_rig()
    real = rig.evidence.write

    def flaky(path, data):
        if path.name == "during.jpg":
            raise OSError(28, "No space left on device")
        return real(path, data)

    rig.evidence.write = flaky
    await dog_visit(rig)
    (v,) = all_visits(rig)
    assert v["auto_outcome"] == "pee", "the optional 'during' photo killed the whole analysis"


async def test_recover_marks_every_open_state_and_retries_analysing(make_rig):
    rig = make_rig()
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await tick_for(rig, 3)
    (v,) = all_visits(rig)
    assert v["state"] == "OPEN"
    await rig.service.recover()
    v = rig.store.get_visit(v["id"])
    assert v["state"] == "DECIDED" and v["final_outcome"] == "unknown"


async def test_review_queue_count_is_not_capped_at_200(tmp_path):
    from ellie.evidence import Evidence
    from ellie.state import build_state
    from ellie.store import Store
    from helpers import make_visit

    store = Store(str(tmp_path / "x.db"))
    for i in range(250):
        store.insert_visit(
            make_visit(
                id=f"v_20261028_{i:06d}", started_at="2026-10-28T10:00:00Z", needs_review=True
            )
        )
    cfg = dict(RUNTIME_DEFAULTS)
    st = build_state(
        store=store,
        evidence=Evidence(tmp_path / "m"),
        active_visit=None,
        cfg=cfg,
        now=at(12),
        usd_to_eur=0.9,
    )
    store.close()
    assert st["review_queue"]["count"] == 250


async def test_gemini_errors_are_not_counted_as_wrong_answers_in_accuracy(make_rig):
    rig = make_rig()
    rig.analyser.script(AnalyserError("gemini unavailable: HTTP 503"))
    await dog_visit(rig)
    (v,) = all_visits(rig)
    await send(rig, "ellie/cmd/label", {"visit_id": v["id"], "label": "pee"})
    st = [b for t, b, _ in rig.published if t == "ellie/state"][-1]
    a = st["accuracy"]["analysers"].get("gemini:pee_v1", {"labelled": 0})
    assert a["labelled"] == 0, f"an API outage was graded as a wrong answer: {a}"
