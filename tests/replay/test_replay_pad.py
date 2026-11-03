"""US7 end to end: a person at the tray, the pad analysis, the reminder facts and the commands."""

import json
from datetime import datetime, timedelta

import pytest
from test_replay_visit import build, last_state, one_visit, topics

from ellie.analysers.base import AnalyserError
from helpers import FakeClock, make_jpeg, make_pad_result, make_result


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
    assert topics(rig, "ellie/pad/changed")[-1]["image"].endswith(
        "after.jpg"
    )  # for the phone notice
    assert pad["image_at"] == pad["changed_at"]
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
    rig.analyser.script(make_result(pee=False, pee_confidence=0.9))  # nothing new on the pad
    await person_at_the_tray(rig)
    usage = last_state(rig)["analysis_calls"]
    # two Gemini questions per presence (pad changed? new pee/poop?), billed on the presence
    assert usage["calls"] == 1 and usage["tokens_in"] == 2000


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


async def test_a_manual_mark_after_a_detection_keeps_showing_the_detected_photo(rig):
    rig.analyser.script_pad(make_pad_result(pad_replaced=True, pad_replaced_confidence=0.95))
    await person_at_the_tray(rig)
    (presence,) = presences(rig)
    later = (datetime.fromisoformat(presence["ended_at"]) + timedelta(minutes=1)).isoformat()
    await command(rig, "pad_changed", at=later)
    pad = last_state(rig)["pad"]
    assert pad["source"] == "manual" and pad["changed_at"] > presence["ended_at"]
    assert pad["image"].endswith("after.jpg")  # the photo of the detection
    assert pad["image_at"] == presence["ended_at"]  # and the card can say when it was taken
    photo, photo_at = rig.service.pad_photo()  # also what the image entity shows
    assert photo.name == "after.jpg" and photo_at == presence["ended_at"]


async def test_the_big_photo_shows_the_clean_pad_after_a_change_until_the_next_pee(rig):
    def big_photo() -> bytes:
        return [b for t, b, _ in rig.published if t == "ellie/image/last_pee"][-1]

    await one_visit(rig)  # a counted pee: its photo is on show
    (visit,) = [v for v in rig.store._db.execute("SELECT id FROM visits")]
    assert big_photo() == (rig.evidence.visit_dir(visit["id"]) / "after.jpg").read_bytes()
    rig.analyser.script_pad(make_pad_result(pad_replaced=True, pad_replaced_confidence=0.95))
    await person_at_the_tray(rig)
    (presence,) = [p for p in presences(rig) if p["final_outcome"] == "replaced"]
    clean = (rig.evidence.presence_dir(presence["id"]) / "after.jpg").read_bytes()
    assert big_photo() == clean  # the stained photo is replaced by the clean pad
    await command(rig, "pad_undo")
    assert big_photo() == (rig.evidence.visit_dir(visit["id"]) / "after.jpg").read_bytes()


async def test_a_restart_between_the_pad_decision_and_the_promotion_does_not_lose_the_pee(rig):
    """T093: the "person" left a pee; the app died after deciding "pad not replaced"."""
    rig.store.insert_presence({"id": "s_20261102_100000", "started_at": "2026-11-02T10:00:00Z", "ended_at": "2026-11-02T10:01:00Z", "state": "DECIDED", "auto_outcome": "not_replaced", "final_outcome": "not_replaced", "promotion_pending": True})  # fmt: skip
    directory = rig.evidence.presence_dir("s_20261102_100000")
    rig.evidence.write(directory / "before.jpg", make_jpeg())
    rig.evidence.write(directory / "after.jpg", make_jpeg())
    rig.analyser.script(make_result(pee=True, pee_confidence=0.95))
    await rig.service.recover()
    await rig.service.wait_idle()
    (visit,) = [v for v in one_visit_rows(rig) if v["person_present"]]
    assert visit["final_outcome"] == "pee" and visit["started_at"] == "2026-11-02T10:00:00Z"
    assert rig.store.get_presence("s_20261102_100000")["promotion_pending"] is False
    await rig.service.recover()  # a second restart must not count it again
    await rig.service.wait_idle()
    assert len([v for v in one_visit_rows(rig) if v["person_present"]]) == 1
    assert len(rig.analyser.calls) == 1


def one_visit_rows(rig):
    rows = rig.store._db.execute("SELECT id FROM visits ORDER BY id").fetchall()
    return [rig.store.get_visit(r["id"]) for r in rows]


def _pee(rig, n, at):
    rig.store.insert_visit({"id": f"v_{n}", "source": "auto", "started_at": at, "ended_at": at, "entries": 1, "person_present": False, "frigate_event_ids": [], "state": "DECIDED", "final_outcome": "pee"})  # fmt: skip


async def test_the_average_of_pees_per_finished_pad(rig):
    """T096: pads A (3 pees) and B (1 pee) are finished, C is in use and does not count yet."""
    for pad, at in (("a", "2026-10-20T10:00:00Z"), ("b", "2026-10-23T10:00:00Z"), ("c", "2026-10-25T10:00:00Z")):  # fmt: skip
        rig.store.insert_pad_change({"id": f"p_{pad}", "at": at, "source": "manual"})
    for n, at in enumerate(("2026-10-20T12:00:00Z", "2026-10-21T12:00:00Z", "2026-10-22T12:00:00Z", "2026-10-24T12:00:00Z", "2026-10-26T12:00:00Z")):  # fmt: skip
        _pee(rig, n, at)
    pad = rig.service.pad_state(rig.clock())
    assert [h["pees"] for h in pad["history"]] == [
        None,
        1,
        3,
    ]  # newest first; the current pad is open
    assert pad["pees_since"] == 1 and pad["pads_counted"] == 2 and pad["avg_pees_per_pad"] == 2.0


async def test_one_finished_pad_is_too_few_for_an_average_and_an_undone_change_is_skipped(rig):
    rig.store.insert_pad_change({"id": "p_a", "at": "2026-10-20T10:00:00Z", "source": "manual"})
    rig.store.insert_pad_change({"id": "p_b", "at": "2026-10-23T10:00:00Z", "source": "manual", "undone_at": "2026-10-23T10:05:00Z"})  # fmt: skip
    rig.store.insert_pad_change({"id": "p_c", "at": "2026-10-25T10:00:00Z", "source": "manual"})
    _pee(rig, 1, "2026-10-24T12:00:00Z")  # falls on pad A, because B was undone
    pad = rig.service.pad_state(rig.clock())
    assert [h["pees"] for h in pad["history"]] == [None, None, 1]
    assert pad["pads_counted"] == 1 and pad["avg_pees_per_pad"] is None
