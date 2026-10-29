"""US7 end to end: a person at the tray, the pad analysis, the reminder facts and the commands."""

import json

import pytest
from test_replay_visit import build, last_state, one_visit, topics

from ellie.analysers.base import AnalyserError
from helpers import FakeClock, make_jpeg, make_pad_result


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rig(tmp_path, clock):
    rig = build(tmp_path, clock)
    rig.reference.update(make_jpeg((200, 200, 200)))
    yield rig
    rig.store.close()


async def command(rig, name, **data):
    await rig.service.on_message(f"ellie/cmd/{name}", json.dumps(data).encode())


async def person_at_the_tray(rig, stay=60):
    """Someone stands at the tray `stay` seconds and leaves; time runs until the analysis ends."""
    await rig.service.tick()
    await rig.service.on_message("frigate/tray_area/person", b"1")
    for _ in range(stay):
        rig.clock.advance(1)
        await rig.service.tick()
    await rig.service.on_message("frigate/tray_area/person", b"0")
    for _ in range(40):
        rig.clock.advance(1)
        await rig.service.tick()
    await rig.service.wait_idle()


def presences(rig):
    rows = rig.store._db.execute("SELECT id FROM presences ORDER BY id").fetchall()
    return [rig.store.get_presence(r["id"]) for r in rows]


async def test_a_confident_replacement_restarts_the_countdown_with_no_tap(rig):
    rig.analyser.script_pad(make_pad_result(pad_replaced=True, pad_replaced_confidence=0.95, pad_color="grey"))  # fmt: skip
    await person_at_the_tray(rig)
    (presence,) = presences(rig)
    assert presence["auto_outcome"] == presence["final_outcome"] == "replaced"
    assert presence["needs_review"] is False
    for name in ("before.jpg", "after.jpg"):
        assert (rig.evidence.presence_dir(presence["id"]) / name).exists()
    (change,) = rig.store.pad_changes()
    assert change["source"] == "detected" and change["presence_id"] == presence["id"]
    pad = last_state(rig)["pad"]
    assert pad["source"] == "detected" and pad["pad_color"] == "grey" and pad["pees_since"] == 0
    assert pad["image"].endswith("after.jpg") and pad["changed_at"] == presence["ended_at"]
    assert pad["due_at"] == "2026-11-01T12:01:00Z"  # three days after the change
    assert topics(rig, "ellie/pad/changed")[-1]["source"] == "detected"
    assert topics(rig, "ellie/review/request") == []  # a confident change is silent


async def test_picking_up_poop_is_not_a_pad_change(rig):
    rig.analyser.script_pad(make_pad_result(pad_replaced=False, poop_removed=True))
    await person_at_the_tray(rig)
    (presence,) = presences(rig)
    assert presence["final_outcome"] == "not_replaced" and presence["poop_removed"] is True
    assert rig.store.pad_changes() == [] and last_state(rig)["pad"] is None
    assert topics(rig, "ellie/review/request") == []


async def test_an_unsure_answer_asks_the_owner(rig):
    rig.analyser.script_pad(make_pad_result(pad_replaced=True, pad_replaced_confidence=0.6))
    await person_at_the_tray(rig)
    (presence,) = presences(rig)
    (request,) = topics(rig, "ellie/review/request")
    assert request["kind"] == "pad_uncertain" and request["visit_id"] == presence["id"]
    assert request["title"] == "Ellie: você trocou o tapete?"
    assert presence["needs_review"] is True and rig.store.pad_changes() == []

    await command(rig, "pad_label", presence_id=presence["id"], label="replaced", by="iphone")
    (change,) = rig.store.pad_changes()
    assert change["source"] == "confirmed" and change["at"] == presence["ended_at"]
    assert rig.store.get_presence(presence["id"])["needs_review"] is False
    assert topics(rig, "ellie/review/resolved")[-1] == {"visit_id": presence["id"]}


async def test_saying_not_replaced_records_nothing(rig):
    rig.analyser.script_pad(make_pad_result(pad_replaced_confidence=0.5))
    await person_at_the_tray(rig)
    (presence,) = presences(rig)
    await command(rig, "pad_label", presence_id=presence["id"], label="not_replaced")
    assert rig.store.pad_changes() == []
    assert rig.store.get_presence(presence["id"])["final_outcome"] == "not_replaced"


async def test_a_short_pass_costs_nothing_and_leaves_no_files(rig):
    await person_at_the_tray(rig, stay=4)
    assert rig.analyser.pad_calls == []
    (presence,) = presences(rig)
    assert presence["state"] == "DECIDED" and presence["reason"] == "too_short"
    assert not rig.evidence.presence_dir(presence["id"]).exists()


async def test_a_gemini_failure_asks_the_owner_and_credit_errors_alert(rig):
    rig.analyser.script_pad(AnalyserError("gemini request failed: HTTP 402"))
    await person_at_the_tray(rig)
    assert [a["kind"] for a in topics(rig, "ellie/alert")] == ["credit_exhausted"]
    (request,) = topics(rig, "ellie/review/request")
    assert request["kind"] == "pad_uncertain" and "créditos do Gemini" in request["reason"]


async def test_manual_mark_undo_and_history(rig):
    await command(rig, "pad_changed", at="2026-10-27T09:00:00", by="owner")
    await command(rig, "pad_changed")  # now
    pad = last_state(rig)["pad"]
    assert pad["source"] == "manual" and pad["changed_at"] == "2026-10-29T12:00:00Z"
    assert len(pad["history"]) == 2
    await command(rig, "pad_undo")
    pad = last_state(rig)["pad"]
    assert pad["changed_at"] == "2026-10-27T09:00:00Z" and pad["days_in_use"] == 2.1
    assert [h["undone_at"] is not None for h in pad["history"]] == [True, False]  # both are kept
    assert topics(rig, "ellie/pad/changed")[-1]["undone"] is True
    await command(rig, "pad_undo")
    await command(rig, "pad_undo")  # nothing left: ignored
    assert last_state(rig)["pad"] is None


async def test_the_interval_setting_moves_the_due_date(rig):
    await command(rig, "pad_changed")
    assert last_state(rig)["pad"]["due_at"] == "2026-11-01T12:00:00Z"
    await rig.service.on_message("ellie/set/pad_change_days", b"5")
    assert last_state(rig)["pad"]["due_at"] == "2026-11-03T12:00:00Z"
    await rig.service.on_message("ellie/set/pad_change_days", b"99")  # clamped to 7
    assert last_state(rig)["pad"]["due_at"] == "2026-11-05T12:00:00Z"


async def test_pees_are_counted_per_pad(rig):
    await command(rig, "pad_changed", at="2026-10-29T11:00:00")
    await one_visit(rig)
    assert last_state(rig)["pad"]["pees_since"] == 1
    await command(rig, "pad_changed")
    assert last_state(rig)["pad"]["pees_since"] == 0


async def test_a_visit_with_a_person_and_a_swapped_pad_records_the_change(rig):
    from helpers import make_result

    rig.analyser.script(make_result(pad_replaced=True))
    await rig.service.tick()
    await rig.service.on_message("frigate/pad_zone/dog", b"1")
    await rig.service.on_message("frigate/tray_area/person", b"1")
    for _ in range(3):
        rig.clock.advance(1)
        await rig.service.tick()
    await rig.service.on_message("frigate/pad_zone/dog", b"0")
    await rig.service.on_message("frigate/tray_area/person", b"0")
    for _ in range(100):
        rig.clock.advance(1)
        await rig.service.tick()
    await rig.service.wait_idle()
    (change,) = rig.store.pad_changes()
    assert change["source"] == "visit" and change["visit_id"]
    assert presences(rig) == []  # the dog visit owned the area: no separate presence


async def test_a_swapped_pad_without_a_person_is_not_trusted(rig):
    from helpers import make_result

    rig.analyser.script(make_result(pad_replaced=True))
    await one_visit(rig)
    assert rig.store.pad_changes() == []


async def test_bad_pad_commands_are_ignored(rig):
    await command(rig, "pad_label", presence_id="nope", label="replaced")
    await command(rig, "pad_changed", at="2026-12-01T09:00:00")  # future
    await command(rig, "pad_changed", at="banana")
    assert rig.store.pad_changes() == []


async def test_presence_analyses_are_billed_in_the_monthly_usage(rig):
    rig.analyser.last_usage = {"in": 1000, "out": 100}
    await person_at_the_tray(rig)
    usage = last_state(rig)["analysis_calls"]
    assert usage["calls"] == 1 and usage["tokens_in"] == 1000


async def test_restart_recovery_for_presences(rig):
    rig.store.insert_presence({"id": "s_20261029_100000", "started_at": "2026-10-29T10:00:00Z", "state": "PRESENT"})  # fmt: skip
    rig.store.insert_presence({"id": "s_20261029_110000", "started_at": "2026-10-29T11:00:00Z", "ended_at": "2026-10-29T11:01:00Z", "state": "ANALYSING"})  # fmt: skip
    directory = rig.evidence.presence_dir("s_20261029_110000")
    rig.evidence.write(directory / "before.jpg", make_jpeg())
    rig.evidence.write(directory / "after.jpg", make_jpeg())
    await rig.service.recover()
    await rig.service.wait_idle()
    assert rig.store.get_presence("s_20261029_100000")["reason"] == "restart"
    assert rig.store.get_presence("s_20261029_110000")["auto_outcome"] == "not_replaced"
