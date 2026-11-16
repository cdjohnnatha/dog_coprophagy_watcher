"""ADR-0020 (2026-11-14): failures of 2026-11-13/14 and the fixes in app 3.10.0."""

import json
from datetime import timedelta

import pytest

from ellie.census import FOUND_PEE_REASON, found_pee_times, missing_pees
from ellie.frames import FrameError
from ellie.visits import iso
from ellie.watch import Candidate, new_stain_area, pee_stain
from helpers import FakeClock, build_rig, make_jpeg, make_pad_result, make_result, make_visit

EVENT = {
    "after": {
        "id": "1793288511.990085-yohln0",
        "camera": "ellie",
        "label": "dog",
        "entered_zones": ["pad_zone", "tray_area"],
    }
}
CLEAN = make_jpeg((200, 200, 200))
STAINED = make_jpeg((200, 200, 200), box=(160, 70, 190, 90, (60, 60, 60)))  # new patch on the pad
DOG = make_jpeg((10, 10, 10))


@pytest.fixture
def clock():
    return FakeClock()


def _camera(frames):
    async def fetch_frame():
        return frames.pop(0) if len(frames) > 1 else frames[0]

    return fetch_frame


def build(tmp_path, clock, frames, **overrides):
    rig = build_rig(tmp_path, clock, fetch_frame=_camera(list(frames)), **overrides)
    rig.reference.update(CLEAN)
    return rig


async def send(rig, topic, value):
    await rig.service.on_message(topic, str(value).encode())


async def advance(rig, seconds):
    for _ in range(int(seconds)):
        rig.clock.advance(1)
        await rig.service.tick()
    await rig.service.wait_idle()


async def one_visit(rig, stay=8, person=False):
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/tray_area/dog", 1)
    await rig.service.on_message("frigate/events", json.dumps(EVENT).encode())
    await advance(rig, stay)
    await send(rig, "frigate/pad_zone/dog", 0)
    await send(rig, "frigate/tray_area/dog", 0)  # exit frame
    await advance(rig, 20)
    if person:
        await send(rig, "frigate/tray_area/person", 1)
        await advance(rig, 30)
        await send(rig, "frigate/tray_area/person", 0)
    await advance(rig, 100)


def visits(rig):
    return rig.store.visits_started_between("2026", "2028")


def topics(rig, name):
    return [body for topic, body, _ in rig.published if topic == name]


# --- 1. the exit frame that still shows her (2026-11-14 09:37) ---


async def test_the_later_frame_is_judged_when_the_person_changed_nothing(tmp_path, clock):
    late = make_jpeg((150, 150, 150))
    rig = build(tmp_path, clock, [DOG, DOG, late])  # during, exit frame (she is still there), late
    await one_visit(rig, person=True)
    (visit,) = visits(rig)
    directory = rig.evidence.visit_dir(visit["id"])
    assert (directory / "after.jpg").read_bytes() == late
    assert rig.analyser.afters == [late]
    assert rig.analyser.pad_calls == [{"before_len": len(DOG), "after_len": len(late)}]  # once


async def test_the_exit_frame_stays_when_the_person_removed_a_poop(tmp_path, clock):
    exit_frame, late = make_jpeg((60, 40, 20)), make_jpeg((150, 150, 150))
    rig = build(tmp_path, clock, [DOG, exit_frame, late])
    rig.analyser.script_pad(make_pad_result(poop_removed=True))
    await one_visit(rig, person=True)
    (visit,) = visits(rig)
    assert (rig.evidence.visit_dir(visit["id"]) / "after.jpg").read_bytes() == exit_frame
    assert len(rig.analyser.pad_calls) == 1


async def test_a_failed_person_check_keeps_the_exit_frame(tmp_path, clock):
    from ellie.analysers.base import AnalyserError

    exit_frame, late = make_jpeg((60, 40, 20)), make_jpeg((150, 150, 150))
    rig = build(tmp_path, clock, [DOG, exit_frame, late])
    rig.analyser.script_pad(AnalyserError("timeout"))
    await one_visit(rig, person=True)
    (visit,) = visits(rig)
    assert (rig.evidence.visit_dir(visit["id"]) / "after.jpg").read_bytes() == exit_frame
    assert visit["final_outcome"] == "pee"


# --- 2. a pee that widens an old stain (2026-11-13 23:58) ---


def test_the_stain_area_sees_a_new_patch_and_ignores_light():
    assert pee_stain(CLEAN, STAINED) == pytest.approx(0.05, abs=0.02)
    assert pee_stain(CLEAN, CLEAN) is None
    lamp_off = make_jpeg((200, 200, 200), box=(0, 130, 320, 180, (90, 90, 90)))  # floor darker
    _area, floor = new_stain_area(CLEAN, lamp_off)
    assert floor > 0.06 and pee_stain(CLEAN, lamp_off) is None
    assert new_stain_area(b"not a jpeg", CLEAN) is None


async def test_a_new_stain_turns_a_confident_no_pee_into_a_question(tmp_path, clock):
    rig = build(tmp_path, clock, [DOG, STAINED])
    rig.analyser.script(make_result(pee=False, pee_confidence=0.95, reason="tapete igual"))
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["auto_outcome"] == "uncertain" and visit["final_outcome"] is None
    assert visit["needs_review"] is True
    assert "mancha nova" in visit["reason"]
    (request,) = topics(rig, "ellie/review/request")
    assert request["kind"] == "uncertain"


async def test_no_question_while_the_last_pee_may_still_be_spreading(tmp_path, clock):
    rig = build(tmp_path, clock, [DOG, STAINED])
    ten_min_ago = iso(clock() - timedelta(minutes=10))
    rig.store.insert_visit(make_visit(id="v_20261029_115000", started_at=ten_min_ago,
                                      ended_at=ten_min_ago))  # fmt: skip
    rig.analyser.script(make_result(pee=False, pee_confidence=0.95))
    await one_visit(rig)
    visit = visits(rig)[-1]
    assert visit["auto_outcome"] == "no_pee"


async def test_a_watch_visit_gets_no_stain_opinion(tmp_path, clock):
    rig = build(tmp_path, clock, [STAINED])
    rig.analyser.script(make_result(pee=False, pee_confidence=0.95))
    found = Candidate(at=clock(), before_at=clock() - timedelta(minutes=1), before_jpeg=CLEAN,
                      after_jpeg=STAINED, pad_step=0.2, ctrl_step=0.02, norm=[], brightness=120)  # fmt: skip
    await rig.service._open_watch_visit(found)
    await rig.service.wait_idle()
    (visit,) = visits(rig)
    assert visit["auto_outcome"] == "no_pee"


async def test_gemini_gets_the_minutes_since_the_last_pee(tmp_path, clock):
    rig = build(tmp_path, clock, [DOG, CLEAN])
    rig.store.insert_pad_change({"id": "p_1", "at": iso(clock() - timedelta(hours=5)),
                                 "source": "manual"})  # fmt: skip
    two_h = iso(clock() - timedelta(hours=2))
    rig.store.insert_visit(make_visit(id="v_20261029_100000", started_at=two_h, ended_at=two_h))
    await one_visit(rig)
    (context,) = rig.analyser.contexts
    assert "put in 5.0 hours before BEFORE" in context
    assert "last counted pee was 120 minutes before BEFORE" in context


async def test_gemini_hears_when_no_pee_is_on_the_new_pad(tmp_path, clock):
    rig = build(tmp_path, clock, [DOG, CLEAN])
    two_h = iso(clock() - timedelta(hours=2))
    rig.store.insert_visit(make_visit(id="v_20261029_100000", started_at=two_h, ended_at=two_h))
    rig.store.insert_pad_change({"id": "p_1", "at": iso(clock() - timedelta(hours=1)),
                                 "source": "manual"})  # fmt: skip
    await one_visit(rig)
    assert "No pee was counted on this pad yet." in rig.analyser.contexts[0]


# --- 3. the stain census (owner's idea; 2026-11-13 evening) ---


def test_missing_pees_and_their_times():
    assert missing_pees(None, 0) == 0 and missing_pees(2, 2) == 0 and missing_pees(1, 3) == 0
    assert missing_pees(3, 0) == 3 and missing_pees(9, 0) == 3  # capped
    start = FakeClock()()
    times = found_pee_times(start, start + timedelta(hours=4), 3)
    assert [t - start for t in times] == [timedelta(hours=h) for h in (1, 2, 3)]


async def test_the_census_counts_the_pees_nobody_saw(tmp_path, clock):
    rig = build(tmp_path, clock, [DOG, CLEAN])
    rig.store.insert_pad_change({"id": "p_1", "at": iso(clock() - timedelta(hours=4)),
                                 "source": "detected"})  # fmt: skip
    rig.analyser.script(make_result(pee=False, pee_confidence=0.95, poop="on_pad",
                                    poop_confidence=0.95, pad_stains=3))  # fmt: skip
    await one_visit(rig)
    found = [v for v in visits(rig) if (v["reason"] or "").startswith(FOUND_PEE_REASON)]
    assert len(found) == 3
    assert all(v["final_outcome"] == "pee" and v["needs_review"] for v in found)
    assert all(v["started_at"] < iso(clock()) for v in found)
    assert rig.store.pees_since(iso(clock() - timedelta(hours=5))) == 3
    (alert,) = topics(rig, "ellie/alert")
    assert alert["kind"].startswith("census_") and "3 manchas" in alert["message"]

    # the owner then enters the real time of one of them: it takes the found pee's place
    real = (clock() - timedelta(hours=3, minutes=10)).astimezone().isoformat()
    await rig.service.on_message("ellie/cmd/manual_pee", json.dumps({"at": real}).encode())
    assert rig.store.pees_since(iso(clock() - timedelta(hours=5))) == 3
    taken = [v for v in visits(rig) if v["reason"].startswith("adicionado manualmente")]
    assert len(taken) == 1 and taken[0]["started_at"] == iso(
        clock() - timedelta(hours=3, minutes=10)
    )

    # a later visit seeing the same three stains adds nothing
    rig.analyser.script(make_result(pee=False, pee_confidence=0.95, pad_stains=3))
    clock.advance(3600)
    await one_visit(rig)
    assert rig.store.pees_since(iso(clock() - timedelta(hours=6))) == 3


async def test_the_census_needs_a_known_pad_change_and_a_clear_answer(tmp_path, clock):
    rig = build(tmp_path, clock, [DOG, CLEAN])
    rig.analyser.script(make_result(pee=False, pee_confidence=0.95, pad_stains=3))
    await one_visit(rig)  # no pad change known
    assert rig.store.pees_since("2026") == 0
    rig.store.insert_pad_change({"id": "p_1", "at": iso(clock() - timedelta(hours=4)),
                                 "source": "detected"})  # fmt: skip
    rig.analyser.script(make_result(pee=False, pee_confidence=0.5, pad_stains=3))  # unsure
    clock.advance(3600)
    await one_visit(rig)
    assert rig.store.pees_since("2026") == 0


# --- 4. a floor change hidden behind her own short "presence" (2026-11-14 09:59) ---


async def test_a_presence_does_not_explain_a_floor_change(tmp_path, clock):
    rig = build(tmp_path, clock, [CLEAN])
    now = clock()
    rig.store.insert_presence({"id": "s_1", "started_at": iso(now - timedelta(seconds=20)),
                               "ended_at": iso(now - timedelta(seconds=18)),
                               "state": "DECIDED"})  # fmt: skip
    found = Candidate(at=now, before_at=now - timedelta(minutes=1), before_jpeg=CLEAN,
                      after_jpeg=STAINED, pad_step=0.02, ctrl_step=0.02, norm=[], brightness=120,
                      kind="floor", floor_peak=3.6)  # fmt: skip
    await rig.service._open_watch_visit(found)
    await rig.service.wait_idle()
    assert len(visits(rig)) == 1
    pad = Candidate(at=now, before_at=now - timedelta(minutes=1), before_jpeg=CLEAN,
                    after_jpeg=STAINED, pad_step=0.2, ctrl_step=0.02, norm=[], brightness=120)  # fmt: skip
    await rig.service._open_watch_visit(pad)  # a pad change near a presence is still explained
    assert len(visits(rig)) == 1


# --- 5. the same event entered twice ---


async def test_a_manual_poop_next_to_a_counted_one_is_not_added(tmp_path, clock):
    rig = build(tmp_path, clock, [CLEAN])
    at = "2026-10-29T11:01:58Z"
    rig.store.insert_visit(make_visit(id="v_20261029_110158", started_at=at, ended_at=at,
                                      final_outcome="no_pee", final_poop="poop"))  # fmt: skip
    payload = json.dumps({"at": "2026-10-29T11:02:00+00:00"}).encode()
    await rig.service.on_message("ellie/cmd/manual_poop", payload)
    assert len(visits(rig)) == 1
    payload = json.dumps({"at": "2026-10-29T08:00:00+00:00"}).encode()
    await rig.service.on_message("ellie/cmd/manual_poop", payload)
    assert len(visits(rig)) == 2


async def test_two_detected_pad_changes_a_minute_apart_are_one(tmp_path, clock):
    rig = build(tmp_path, clock, [CLEAN])
    first = await rig.service._record_pad_change("detected", "2026-10-29T10:37:49Z")
    second = await rig.service._record_pad_change("detected", "2026-10-29T10:38:35Z")
    assert first == second and len(rig.store.pad_changes(10)) == 1
    await rig.service._record_pad_change("detected", "2026-10-29T11:00:00Z")
    assert len(rig.store.pad_changes(10)) == 2


# --- 6. the camera failing for hours (2026-11-13 17:55-21:07) ---


async def test_frigates_frame_stands_in_when_go2rtc_fails(tmp_path, clock):
    async def broken():
        raise FrameError("go2rtc returned HTTP 500")

    fallback_frame = make_jpeg((180, 180, 180))

    async def latest():
        return fallback_frame

    rig = build_rig(tmp_path, clock, fetch_frame=broken, fetch_fallback_frame=latest)
    rig.reference.update(CLEAN)
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["final_outcome"] == "pee"
    assert (rig.evidence.visit_dir(visit["id"]) / "after.jpg").read_bytes() == fallback_frame
    assert topics(rig, "ellie/alert") == []
    clock.advance(16 * 60)
    await rig.service.minute_tasks()
    (alert,) = topics(rig, "ellie/alert")
    assert alert["kind"] == "camera" and "Frigate" in alert["message"]
    await rig.service.minute_tasks()
    assert len(topics(rig, "ellie/alert")) == 1  # rate-limited


async def test_without_any_frame_the_visit_fails_as_before(tmp_path, clock):
    async def broken():
        raise FrameError("go2rtc returned HTTP 500")

    rig = build_rig(tmp_path, clock, fetch_frame=broken, fetch_fallback_frame=broken)
    rig.reference.update(CLEAN)
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["final_outcome"] == "unknown"


async def test_a_manual_pee_for_this_same_visit_is_not_the_last_pee(tmp_path, clock):
    rig = build(tmp_path, clock, [DOG, CLEAN])
    three_h, one_min = iso(clock() - timedelta(hours=3)), iso(clock() - timedelta(minutes=1))
    rig.store.insert_visit(make_visit(id="v_20261029_090000", started_at=three_h, ended_at=three_h))
    rig.store.insert_visit(make_visit(id="v_20261029_115900", source="manual", started_at=one_min,
                                      ended_at=one_min))  # fmt: skip
    await one_visit(rig)
    assert "last counted pee was 180 minutes before BEFORE" in rig.analyser.contexts[0]


# --- review findings (2026-11-14) ---


def test_frames_of_different_sizes_get_no_stain_opinion():
    small = make_jpeg((200, 200, 200), size=(160, 90), box=(80, 35, 95, 45, (60, 60, 60)))
    assert new_stain_area(CLEAN, small) is None


async def _census_rig(tmp_path, clock, stains=1):
    rig = build(tmp_path, clock, [DOG, CLEAN])
    rig.store.insert_pad_change({"id": "p_1", "at": iso(clock() - timedelta(hours=4)),
                                 "source": "detected"})  # fmt: skip
    rig.analyser.script(make_result(pee=False, pee_confidence=0.95, pad_stains=stains))
    return rig


def _found(rig):
    return [v for v in visits(rig) if v["found_by"] and v["final_outcome"] == "pee"]


async def test_a_visit_waiting_for_the_owner_is_one_of_the_stains(tmp_path, clock):
    rig = await _census_rig(tmp_path, clock)
    two_h = iso(clock() - timedelta(hours=2))
    rig.store.insert_visit(make_visit(id="v_20261029_100000", started_at=two_h, ended_at=two_h,
                                      auto_outcome="uncertain", final_outcome=None,
                                      needs_review=True))  # fmt: skip
    await one_visit(rig)
    assert _found(rig) == []


async def test_labelling_the_census_visit_as_pee_drops_its_found_pee(tmp_path, clock):
    rig = await _census_rig(tmp_path, clock)
    await one_visit(rig)
    (found,) = _found(rig)
    census = rig.store.get_visit(found["found_by"])
    assert census["final_outcome"] == "no_pee"
    payload = json.dumps({"visit_id": census["id"], "label": "pee"}).encode()
    await rig.service.on_message("ellie/cmd/label", payload)
    assert _found(rig) == [] and rig.store.pees_since("2026") == 1  # one stain, one pee


async def test_a_manual_pee_outside_the_census_window_is_added(tmp_path, clock):
    rig = await _census_rig(tmp_path, clock)
    await one_visit(rig)
    clock.advance(1800)
    later = (clock() - timedelta(minutes=5)).astimezone().isoformat()  # after the census visit
    await rig.service.on_message("ellie/cmd/manual_pee", json.dumps({"at": later}).encode())
    assert len(_found(rig)) == 1 and rig.store.pees_since("2026") == 2


async def test_a_real_pee_wins_over_a_found_one_next_to_it(tmp_path, clock):
    rig = build(tmp_path, clock, [CLEAN])
    t = clock() - timedelta(hours=1)
    rig.store.insert_visit(make_visit(id="v_found", started_at=iso(t), ended_at=iso(t),
                                      found_by="v_x", found_from=iso(t - timedelta(hours=1))))  # fmt: skip
    real = t + timedelta(minutes=20)
    rig.store.insert_visit(make_visit(id="v_real", started_at=iso(real), ended_at=iso(real)))
    entry = (t + timedelta(minutes=15)).astimezone().isoformat()
    await rig.service.on_message("ellie/cmd/manual_pee", json.dumps({"at": entry}).encode())
    assert rig.store.get_visit("v_found")["started_at"] == iso(t)  # not moved
    assert rig.store.pees_since("2026") == 2


async def test_undoing_the_pad_change_drops_the_found_pees(tmp_path, clock):
    rig = await _census_rig(tmp_path, clock, stains=2)
    await one_visit(rig)
    assert len(_found(rig)) == 2
    await rig.service.on_message("ellie/cmd/pad_undo", b"{}")
    assert _found(rig) == []
