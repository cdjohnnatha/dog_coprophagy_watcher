"""The poop outcome, the "poop present" state and the ways it ends (ADR-0014)."""

import json
from pathlib import Path

import pytest
from test_replay_visit import build, last_state, one_visit, topics

from ellie.service import POOP_PRESENT_TIMEOUT_H
from ellie.watch import PadWatch
from helpers import FakeClock, make_result

FIXTURES = Path(__file__).parent.parent / "fixtures" / "poop_0613"
BEFORE = (FIXTURES / "before.jpg").read_bytes()
AFTER = (FIXTURES / "after.jpg").read_bytes()


@pytest.fixture
def clock():
    return FakeClock()


def make_rig(tmp_path, clock, shown=None):
    shown = shown if shown is not None else {"jpeg": AFTER}

    async def fetch_frame():
        return shown["jpeg"]

    rig = build(tmp_path, clock, fetch_frame=fetch_frame, watch=PadWatch(clock))
    rig.shown = shown
    rig.reference.update(BEFORE)
    return rig


@pytest.fixture
def rig(tmp_path, clock):
    rig = make_rig(tmp_path, clock)
    rig.analyser.script(
        make_result(pee=False, pee_confidence=0.95, poop="beside_tray", poop_confidence=0.95)
    )
    yield rig
    rig.store.close()


async def command(rig, name, **data):
    await rig.service.on_message(f"ellie/cmd/{name}", json.dumps(data).encode())


def visits(rig):
    return rig.store.visits_started_between("2026", "2028")


async def minute(rig, jpeg):
    """One minute passes and the minute task takes a clear frame that shows `jpeg`."""
    rig.shown["jpeg"] = jpeg
    for _ in range(2):
        rig.clock.advance(30)
        await rig.service.tick()
    await rig.service.minute_tasks()
    await rig.service.wait_idle()


async def poop_visit(rig):
    await one_visit(rig)
    (visit,) = visits(rig)
    return visit


async def test_the_poop_of_2026_11_05_0613_is_counted_and_present(rig):
    visit = await poop_visit(rig)
    assert visit["auto_poop"] == visit["final_poop"] == "poop"
    assert visit["final_outcome"] == "no_pee" and visit["poop_cleared_at"] is None
    state = last_state(rig)
    assert state["poop_today"] == 1 and state["pee_today"] == 0
    assert state["poops_by_day"][-1]["poops"] == 1 and len(state["poops_by_day"]) == 90
    present = state["poop_present"]
    assert present["on"] is True and present["count"] == 1
    assert present["where"] == "beside_tray" and present["visit_id"] == visit["id"]
    assert present["image"] and present["since"] == visit["started_at"]
    assert (
        state["last_poop"]["visit_id"] == visit["id"] and state["last_poop"]["cleared_at"] is None
    )
    assert rig.service.watch.tracked_poops() == [visit["id"]]
    assert topics(rig, "ellie/review/request") == []  # a confident answer never asks
    assert any(t == "ellie/image/poop" for t, _, _ in rig.published)


async def test_a_confident_no_poop_is_not_present(tmp_path, clock):
    rig = make_rig(tmp_path, clock, {"jpeg": BEFORE})  # nothing new on the floor either
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["final_poop"] == "no_poop"
    assert last_state(rig)["poop_today"] == 0 and last_state(rig)["poop_present"]["on"] is False
    rig.store.close()


async def test_the_watch_clears_it_when_the_floor_is_clean_again(rig):
    visit = await poop_visit(rig)
    await minute(rig, AFTER)  # still there
    await minute(rig, AFTER)
    assert rig.store.get_visit(visit["id"])["poop_cleared_at"] is None
    await minute(rig, BEFORE)  # one clear frame is not enough
    assert rig.store.get_visit(visit["id"])["poop_cleared_at"] is None
    await minute(rig, BEFORE)
    cleared = rig.store.get_visit(visit["id"])
    assert cleared["poop_cleared_by"] == "watch" and cleared["poop_cleared_at"]
    assert cleared["final_poop"] == "poop"  # it still counts
    state = last_state(rig)
    assert state["poop_present"]["on"] is False and state["poop_today"] == 1
    assert state["last_poop"]["cleared_by"] == "watch"


async def test_a_blip_does_not_clear_it(rig):
    visit = await poop_visit(rig)
    await minute(rig, BEFORE)
    await minute(rig, AFTER)  # back: the streak starts again
    await minute(rig, BEFORE)
    assert rig.store.get_visit(visit["id"])["poop_cleared_at"] is None


async def test_the_floor_blob_of_a_counted_poop_does_not_open_another_visit(rig):
    await poop_visit(rig)
    for _ in range(6):
        await minute(rig, AFTER)
    assert len(visits(rig)) == 1


async def test_it_stops_being_present_after_the_timeout(rig):
    visit = await poop_visit(rig)
    rig.clock.advance(POOP_PRESENT_TIMEOUT_H * 3600 - 600)
    await rig.service.minute_tasks()
    assert rig.store.get_visit(visit["id"])["poop_cleared_at"] is None
    rig.clock.advance(1200)
    await rig.service.minute_tasks()
    cleared = rig.store.get_visit(visit["id"])
    assert (
        cleared["poop_cleared_by"] == "timeout" and last_state(rig)["poop_present"]["on"] is False
    )


async def test_the_owner_clears_one_or_all(rig):
    visit = await poop_visit(rig)
    await command(rig, "poop_clear", visit_id=5)  # a bad id changes nothing
    await command(rig, "poop_clear", visit_id="v_nope")
    assert last_state(rig)["poop_present"]["on"] is True
    await command(rig, "poop_clear")
    assert rig.store.get_visit(visit["id"])["poop_cleared_by"] == "owner"
    assert last_state(rig)["poop_present"]["on"] is False
    assert rig.service.watch.tracked_poops() == []


async def test_answering_no_poop_clears_it_and_never_touches_the_pee(rig):
    visit = await poop_visit(rig)
    await command(rig, "label", visit_id=visit["id"], kind="poop", label="no_poop", by="iphone")
    after = rig.store.get_visit(visit["id"])
    assert after["final_poop"] == "no_poop" and after["poop_cleared_by"] == "owner"
    assert (after["final_outcome"], after["state"]) == (visit["final_outcome"], visit["state"])
    assert after["needs_review"] == visit["needs_review"]
    assert last_state(rig)["poop_today"] == 0
    assert rig.store.labels_for(visit["id"]) == []  # no pee label was written
    (label,) = rig.store.labels_for(visit["id"], "poop")
    assert (label["label"], label["previous_final_outcome"]) == ("no_poop", "poop")
    assert topics(rig, "ellie/review/resolved")[-1] == {"visit_id": visit["id"], "kind": "poop"}


async def test_a_pee_answer_never_touches_the_poop(rig):
    visit = await poop_visit(rig)
    await command(rig, "label", visit_id=visit["id"], kind="pee", label="pee")
    await command(rig, "label", visit_id=visit["id"], label="no_pee")  # kind defaults to pee
    after = rig.store.get_visit(visit["id"])
    assert after["final_poop"] == "poop" and after["poop_cleared_at"] is None
    assert after["final_outcome"] == "no_pee"
    assert last_state(rig)["poop_present"]["on"] is True


async def test_bad_label_commands_are_ignored(rig):
    visit = await poop_visit(rig)
    for bad in (
        {"kind": "poop", "label": "pee"},
        {"kind": "pee", "label": "poop"},
        {"kind": "x", "label": "poop"},
        {"kind": 3, "label": "poop"},
        {"kind": "poop", "label": "poop", "channel": "manual_add"},
    ):
        await command(rig, "label", visit_id=visit["id"], **bad)
    assert rig.store.labels_for(visit["id"], "poop") == []
    assert rig.store.labels_for(visit["id"]) == []


async def test_an_owner_poop_answer_beats_a_later_analysis(rig):
    visit = await poop_visit(rig)
    await command(rig, "label", visit_id=visit["id"], kind="poop", label="no_poop")
    rig.store.update_visit(visit["id"], state="ANALYSING")
    await rig.service._analyse(visit["id"], retry=True)
    assert rig.store.get_visit(visit["id"])["final_poop"] == "no_poop"


async def test_an_owner_poop_on_a_no_poop_visit_makes_it_present(tmp_path, clock):
    rig = make_rig(tmp_path, clock)
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["final_poop"] != "poop"  # no_poop, or asked because of the blob
    await command(rig, "label", visit_id=visit["id"], kind="poop", label="poop")
    state = last_state(rig)
    assert state["poop_today"] == 1 and state["poop_present"]["on"] is True
    assert rig.service.watch.tracked_poops() == [visit["id"]]
    rig.store.close()


async def test_a_pad_change_clears_only_a_poop_on_the_pad(tmp_path, clock):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    on_pad = make_rig(tmp_path / "a", clock)
    on_pad.analyser.script(make_result(pee=False, poop="on_pad", poop_confidence=0.95))
    await one_visit(on_pad)
    await command(on_pad, "pad_changed")
    (v,) = visits(on_pad)
    assert on_pad.store.get_visit(v["id"])["poop_cleared_by"] == "pad_change"
    beside = make_rig(tmp_path / "b", clock)
    beside.analyser.script(make_result(pee=False, poop="beside_tray", poop_confidence=0.95))
    await one_visit(beside)
    await command(beside, "pad_changed")
    (v,) = visits(beside)
    assert beside.store.get_visit(v["id"])["poop_cleared_at"] is None
    on_pad.store.close()
    beside.store.close()


async def test_an_unsure_poop_asks_once_and_waits_in_the_review_list(tmp_path, clock):
    rig = make_rig(tmp_path, clock)
    rig.service.cfg["validation_mode"] = False  # only a real doubt reaches the review list
    rig.analyser.script(make_result(pee=False, pee_confidence=0.95, poop="unsure"))
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["auto_poop"] == "uncertain" and visit["final_poop"] is None
    assert last_state(rig)["poop_today"] == 0 and last_state(rig)["poop_present"]["on"] is False
    (ask,) = topics(rig, "ellie/review/request")
    assert ask["kind"] == "poop" and ask["title"] == "Ellie: fez cocô?"
    queue = last_state(rig)["review_queue"]
    assert queue["count"] == 1
    item = queue["items"][0]
    assert (item["poop_question"], item["pee_question"]) == (True, False)
    await rig.service.publish_state()  # more publishes do not ask again
    assert len(topics(rig, "ellie/review/request")) == 1
    await command(rig, "label", visit_id=visit["id"], kind="poop", label="poop")
    assert last_state(rig)["review_queue"]["count"] == 0
    assert last_state(rig)["poop_present"]["on"] is True
    rig.store.close()


async def test_a_low_confidence_poop_is_uncertain_too(tmp_path, clock):
    rig = make_rig(tmp_path, clock)
    rig.analyser.script(
        make_result(pee=False, pee_confidence=0.95, poop="beside_tray", poop_confidence=0.5)
    )
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["auto_poop"] == "uncertain" and last_state(rig)["poop_today"] == 0
    rig.store.close()


async def test_an_unanswered_poop_question_expires_the_next_day(tmp_path, clock):
    rig = make_rig(tmp_path, clock)
    rig.service.cfg["validation_mode"] = False  # only a real doubt reaches the review list
    rig.analyser.script(make_result(pee=False, pee_confidence=0.95, poop="unsure"))
    await one_visit(rig)
    clock.advance(36 * 3600)
    await rig.service.minute_tasks()
    (visit,) = visits(rig)
    assert visit["final_poop"] == "unknown" and last_state(rig)["review_queue"]["count"] == 0
    rig.store.close()


async def test_a_restart_keeps_following_the_poop(tmp_path, clock):
    first = make_rig(tmp_path, clock)
    first.analyser.script(make_result(pee=False, poop="beside_tray", poop_confidence=0.95))
    await one_visit(first)
    (visit,) = visits(first)
    first.store.close()
    second = make_rig(tmp_path, clock)  # same database and photos, a new process
    assert second.service.watch.tracked_poops() == []
    await second.service.recover()
    assert second.service.watch.tracked_poops() == [visit["id"]]
    await minute(second, BEFORE)
    await minute(second, BEFORE)
    assert second.store.get_visit(visit["id"])["poop_cleared_by"] == "watch"
    second.store.close()


async def test_a_failed_analysis_has_no_poop_answer(tmp_path, clock):
    from ellie.analysers.base import AnalyserError

    rig = make_rig(tmp_path, clock)
    rig.analyser.script(AnalyserError("gemini unavailable: HTTP 503"))
    await one_visit(rig)
    (visit,) = visits(rig)
    assert (visit["auto_poop"], visit["final_poop"]) == ("unknown", "unknown")
    assert last_state(rig)["poop_today"] == 0
    rig.store.close()


async def test_a_blob_gemini_missed_asks_the_owner(tmp_path, clock):
    """2026-11-05 06:13: the poop is in the after photo, Gemini answered "none"."""
    rig = make_rig(tmp_path, clock)  # after = the real 06:13 photo
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["auto_poop"] == "uncertain" and visit["final_poop"] is None
    asked = [b for b in topics(rig, "ellie/review/request") if b.get("kind") == "poop"]
    assert asked and asked[0]["visit_id"] == visit["id"]
    rig.store.close()
