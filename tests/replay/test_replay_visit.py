"""End-to-end flow of the service with fakes: Frigate messages in, published topics out."""

import json

import pytest

from ellie.analysers.base import AnalyserError
from ellie.frames import FrameError
from helpers import FakeClock, build_rig, make_jpeg, make_pad_result, make_result, make_visit

EVENT = {
    "after": {
        "id": "1793288511.990085-yohln0",
        "camera": "ellie",
        "label": "dog",
        "entered_zones": ["pad_zone", "tray_area"],
    }
}


def build(tmp_path, clock, analyser=None, **overrides):
    return build_rig(tmp_path, clock, analyser, **overrides)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rig(tmp_path, clock):
    rig = build(tmp_path, clock)
    rig.reference.update(make_jpeg((200, 200, 200)))  # a clear "before" frame exists
    yield rig
    rig.store.close()


async def send(rig, topic, value):
    await rig.service.on_message(topic, str(value).encode())


async def advance(rig, seconds):
    for _ in range(int(seconds)):
        rig.clock.advance(1)
        await rig.service.tick()
    await rig.service.wait_idle()


async def one_visit(rig, stay=8):
    """The dog enters, stays `stay` seconds and leaves; time then runs until the analysis ends."""
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/tray_area/dog", 1)
    await rig.service.on_message("frigate/events", json.dumps(EVENT).encode())
    await advance(rig, stay)
    await send(rig, "frigate/pad_zone/dog", 0)
    await send(rig, "frigate/tray_area/dog", 0)
    await advance(rig, 75)


def topics(rig, name):
    return [body for topic, body, _ in rig.published if topic == name]


def last_state(rig):
    return topics(rig, "ellie/state")[-1]


async def test_a_confident_pee_is_counted_with_evidence(rig):
    await one_visit(rig)
    (visit,) = rig.store.visits_started_between("2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z")
    assert visit["auto_outcome"] == visit["final_outcome"] == "pee"
    assert visit["state"] == "DECIDED" and visit["entries"] == 1
    assert visit["frigate_event_ids"] == [EVENT["after"]["id"]]
    assert visit["deciding_analyser"] == "gemini:pee_v1"
    assert visit["needs_review"] is True  # validation mode is ON by default
    for name in ("before.jpg", "after.jpg", "during.jpg"):
        assert (rig.evidence.visit_dir(visit["id"]) / name).exists()
    state = last_state(rig)
    assert state["pee_today"] == 1 and state["last_pee"]["visit_id"] == visit["id"]
    assert state["last_pee"]["image"].startswith("/media/local/ellie/visits/")
    assert topics(rig, "ellie/visit/" + visit["id"])[-1]["final_outcome"] == "pee"
    assert topics(rig, "ellie/review/request") == []  # confident: no phone notification
    assert rig.analyser.calls[0]["during_len"] is not None
    assert len(rig.store.analyses_for(visit["id"])) == 1


async def test_the_after_frame_becomes_the_next_before_reference(rig):
    await one_visit(rig)
    (visit,) = rig.store.open_visits() or rig.store.visits_started_between("2026", "2028")
    after = (rig.evidence.visit_dir(visit["id"]) / "after.jpg").read_bytes()
    assert rig.reference.latest()[0] == after


async def test_an_uncertain_visit_asks_the_owner_and_is_not_counted(rig):
    rig.analyser.script(make_result(pee=True, pee_confidence=0.6))
    await one_visit(rig)
    (request,) = topics(rig, "ellie/review/request")
    assert request["kind"] == "uncertain" and request["title"] == "Ellie: fez xixi?"
    assert request["message"].startswith("Visita às ") and request["image"].endswith("after.jpg")
    visit = rig.store.get_visit(request["visit_id"])
    assert visit["auto_outcome"] == "uncertain" and visit["final_outcome"] is None
    assert visit["needs_review"] is True and last_state(rig)["pee_today"] == 0
    assert last_state(rig)["uncertain_open"] == 1


async def test_a_gemini_credit_error_becomes_unknown_and_alerts_once(rig):
    rig.analyser.script(AnalyserError("gemini request failed: HTTP 402"))
    await one_visit(rig)
    await one_visit(rig)
    visits = rig.store.visits_started_between("2026", "2028")
    assert [v["auto_outcome"] for v in visits] == ["unknown", "unknown"]
    assert "créditos do Gemini acabaram" in visits[0]["reason"]
    (alert,) = topics(rig, "ellie/alert")  # rate-limited: two visits, one alert
    assert alert["kind"] == "credit_exhausted" and "aistudio.google.com/billing" in alert["message"]
    requests = topics(rig, "ellie/review/request")
    assert [r["kind"] for r in requests] == ["unknown", "unknown"]
    assert requests[0]["title"] == "Ellie: não consegui analisar"
    assert last_state(rig)["pee_today"] == 0


async def test_other_analysis_errors_do_not_alert(rig):
    rig.analyser.script(AnalyserError("gemini unavailable: HTTP 503"))
    await one_visit(rig)
    assert topics(rig, "ellie/alert") == []
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["reason"] == "a análise falhou"
    assert rig.store.analyses_for(visit["id"])[0]["error"] == "gemini unavailable: HTTP 503"


async def test_no_analyser_configured_ends_unknown(tmp_path, clock):
    rig = build(tmp_path, clock, analysers={})
    rig.reference.update(make_jpeg())
    await one_visit(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["auto_outcome"] == "unknown" and "não está configurada" in visit["reason"]


async def test_a_camera_failure_ends_unknown(tmp_path, clock):
    async def broken():
        raise FrameError("camera offline")

    rig = build(tmp_path, clock, fetch_frame=broken)
    rig.reference.update(make_jpeg())
    await one_visit(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["auto_outcome"] == "unknown" and "imagem da câmera" in visit["reason"]


async def test_a_visit_without_a_reference_frame_ends_unknown(tmp_path, clock):
    rig = build(tmp_path, clock)  # nothing was ever captured as "before"
    await one_visit(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["auto_outcome"] == "unknown" and "referência" in visit["reason"]
    assert visit["before_age_s"] is None


async def test_circling_is_one_visit(rig):
    await rig.service.tick()
    for _ in range(3):
        await send(rig, "frigate/pad_zone/dog", 1)
        await send(rig, "frigate/tray_area/dog", 1)
        await advance(rig, 4)
        await send(rig, "frigate/pad_zone/dog", 0)
        await send(rig, "frigate/tray_area/dog", 0)
        await advance(rig, 6)
    await advance(rig, 80)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["entries"] == 3 and rig.analyser.calls.__len__() == 1


async def test_a_person_during_the_visit_is_recorded(rig):
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/tray_area/person", 1)
    await advance(rig, 3)
    await send(rig, "frigate/pad_zone/dog", 0)
    await send(rig, "frigate/tray_area/person", 0)
    await advance(rig, 100)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["person_present"] is True


@pytest.mark.parametrize(
    "event",
    [
        {"after": {**EVENT["after"], "camera": "garage"}},
        {"after": {**EVENT["after"], "label": "person"}},
        {"after": {**EVENT["after"], "entered_zones": ["tray_pad"]}},
        {"before": {}},
    ],
)
async def test_unrelated_frigate_events_are_ignored(rig, event):
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await rig.service.on_message("frigate/events", json.dumps(event).encode())
    assert rig.service.tracker.active["frigate_event_ids"] == []


async def test_garbage_messages_never_raise(rig):
    for topic, payload in [
        ("frigate/pad_zone/dog", b"banana"),
        ("frigate/events", b"{not json"),
        ("ellie/cmd/label", b"[1,2]"),
        ("ellie/cmd/nothing", b"{}"),
        ("ellie/set/confidence_threshold", b"abc"),
        ("ellie/set/unknown_setting", b"1"),
    ]:
        await rig.service.on_message(topic, payload)
    assert rig.service.cfg["confidence_threshold"] == 0.8


async def test_frigate_going_offline_ends_the_visit_unknown(rig):
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/available", "offline")
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["auto_outcome"] == "unknown" and "Frigate" in visit["reason"]


async def test_settings_apply_persist_and_clamp(tmp_path, clock):
    rig = build(tmp_path, clock)
    rig.reference.update(make_jpeg())
    await send(rig, "ellie/set/confidence_threshold", "0.99")
    await send(rig, "ellie/set/merge_gap_s", "5")  # below the minimum: clamped to 15
    await send(rig, "ellie/set/validation_mode", "OFF")
    await send(rig, "ellie/set/deciding_analyser", "does-not-exist")  # ignored
    assert rig.service.cfg["confidence_threshold"] == 0.99
    assert rig.service.cfg["merge_gap_s"] == 15 and rig.service.tracker.merge_gap_s == 15
    assert rig.service.cfg["validation_mode"] is False
    assert rig.service.cfg["deciding_analyser"] == "gemini:pee_v1"
    assert last_state(rig)["settings"]["confidence_threshold"] == 0.99

    again = build(tmp_path, clock)  # same database: a restart keeps the settings
    assert again.service.cfg["confidence_threshold"] == 0.99
    assert again.service.tracker.merge_gap_s == 15

    await one_visit(rig)  # 0.95 confidence is below 0.99 now
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["auto_outcome"] == "uncertain" and visit["needs_review"] is True


async def test_validation_mode_off_does_not_flag_confident_visits(rig):
    await send(rig, "ellie/set/validation_mode", "OFF")
    await one_visit(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["final_outcome"] == "pee" and visit["needs_review"] is False


async def test_connecting_publishes_discovery_and_state(rig):
    await rig.service.on_connect()
    configs = [(t, r) for t, _, r in rig.published if t.startswith("homeassistant/")]
    # 25 = 20 + poop today, last poop, poop history, poop present and the poop photo (3.5.0)
    assert len(configs) == 30 and all(retained for _, retained in configs)
    state = [(b, r) for t, b, r in rig.published if t == "ellie/state"][-1]
    assert state[1] is True and state[0]["pee_today"] == 0
    assert {
        "last_pee",
        "review_queue",
        "daily_history",
        "accuracy",
        "analysis_calls",
        "settings",
    } <= set(state[0])
    assert len(state[0]["daily_history"]["days"]) == 90


async def test_the_count_survives_a_restart(tmp_path, clock):
    first = build(tmp_path, clock)
    first.reference.update(make_jpeg())
    await one_visit(first)
    assert last_state(first)["pee_today"] == 1
    again = build(tmp_path, clock)  # new process, same database
    await again.service.recover()
    await again.service.publish_state()
    assert last_state(again)["pee_today"] == 1


async def test_recovery_after_a_restart(rig):
    store = rig.store
    store.insert_visit(make_visit(id="v_20261029_090000", state="OPEN", auto_outcome=None, final_outcome=None, needs_review=False))  # fmt: skip
    store.insert_visit(make_visit(id="v_20261029_100000", state="ANALYSING", auto_outcome=None, final_outcome=None, needs_review=False))  # fmt: skip
    store.insert_visit(make_visit(id="v_20261029_110000", state="ANALYSING", auto_outcome=None, final_outcome=None, needs_review=False))  # fmt: skip
    directory = rig.evidence.visit_dir("v_20261029_100000")
    rig.evidence.write(directory / "before.jpg", make_jpeg())
    rig.evidence.write(directory / "after.jpg", make_jpeg())  # this analysis can be retried
    await rig.service.recover()
    await rig.service.wait_idle()
    lost, retried, no_files = (store.get_visit(i) for i in ("v_20261029_090000", "v_20261029_100000", "v_20261029_110000"))  # fmt: skip
    assert lost["auto_outcome"] == "unknown" and "reiniciou" in lost["reason"]
    assert retried["auto_outcome"] == "pee" and retried["state"] == "DECIDED"
    assert no_files["auto_outcome"] == "unknown"


async def test_the_reference_is_refreshed_only_while_the_area_is_clear(rig):
    await rig.service.tick()
    await advance(rig, 15)
    before = rig.reference.latest()[1]
    rig.clock.advance(60)
    await rig.service.minute_tasks()
    assert rig.reference.latest()[1] > before
    await send(rig, "frigate/tray_area/person", 1)
    stamp = rig.reference.latest()[1]
    rig.clock.advance(60)
    await rig.service.minute_tasks()
    assert rig.reference.latest()[1] == stamp  # busy area: the old reference is kept


async def test_an_event_that_only_entered_tray_area_belongs_to_the_visit(rig):
    """2026-10-30 08:24: she was seen only in tray_area; its clip and snapshot are still evidence."""
    await rig.service.tick()
    await send(rig, "frigate/tray_area/dog", 1)
    event = {"after": {**EVENT["after"], "entered_zones": ["tray_area"]}}
    await rig.service.on_message("frigate/events", json.dumps(event).encode())
    assert rig.service.tracker.active["frigate_event_ids"] == [EVENT["after"]["id"]]


DOG = make_jpeg((10, 10, 10))  # the own "during" frame, taken while she is there


def _camera(rig_frames):
    """A camera that shows each queued frame in turn (the last one keeps repeating)."""

    async def fetch_frame():
        return rig_frames.pop(0) if len(rig_frames) > 1 else rig_frames[0]

    return fetch_frame


async def _poop_then_cleaned(rig, person_in_between=True):
    """2026-11-09 08:22: she poops, leaves, the owner cleans before the after frame is due."""
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/tray_area/dog", 1)
    await advance(rig, 30)
    await send(rig, "frigate/pad_zone/dog", 0)
    await send(rig, "frigate/tray_area/dog", 0)  # exit frame: the poop is there
    await advance(rig, 20)
    if person_in_between:
        await send(rig, "frigate/tray_area/person", 1)
        await advance(rig, 50)
        await send(rig, "frigate/tray_area/person", 0)
    await advance(rig, 100)


async def test_the_exit_frame_is_judged_when_a_person_came_before_the_after_frame(tmp_path, clock):
    exit_frame, cleaned = make_jpeg((60, 40, 20)), make_jpeg((200, 200, 200))
    rig = build(tmp_path, clock, fetch_frame=_camera([DOG, exit_frame, cleaned]))
    rig.reference.update(make_jpeg((200, 200, 200)))
    rig.analyser.script(make_result(pee=False, poop="on_pad", poop_confidence=0.95))
    rig.analyser.script_pad(make_pad_result(pad_replaced=False, poop_removed=True))
    await _poop_then_cleaned(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    directory = rig.evidence.visit_dir(visit["id"])
    assert (directory / "after.jpg").read_bytes() == exit_frame  # Gemini saw the poop
    assert (directory / "after_person.jpg").read_bytes() == cleaned
    assert rig.analyser.calls[0]["after_len"] == len(exit_frame)
    assert visit["final_poop"] == "poop"  # counted
    assert visit["poop_cleared_by"] == "owner"  # and already cleaned by the person
    assert rig.analyser.pad_calls == [{"before_len": len(exit_frame), "after_len": len(cleaned)}]


async def test_a_pad_swap_by_the_person_after_her_is_recorded(tmp_path, clock):
    rig = build(tmp_path, clock, fetch_frame=_camera([DOG, make_jpeg((60, 40, 20)), make_jpeg()]))
    rig.reference.update(make_jpeg((200, 200, 200)))
    rig.analyser.script_pad(make_pad_result(pad_replaced=True, pad_replaced_confidence=0.95))
    await _poop_then_cleaned(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert rig.store.current_pad_change()["visit_id"] == visit["id"]


async def test_without_a_person_the_late_frame_is_judged_as_before(tmp_path, clock):
    exit_frame, late = make_jpeg((60, 40, 20)), make_jpeg((150, 150, 150))
    rig = build(tmp_path, clock, fetch_frame=_camera([DOG, exit_frame, late]))
    rig.reference.update(make_jpeg((200, 200, 200)))
    await _poop_then_cleaned(rig, person_in_between=False)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    directory = rig.evidence.visit_dir(visit["id"])
    assert (directory / "after.jpg").read_bytes() == late
    assert not (directory / "after_person.jpg").exists()
    assert rig.analyser.pad_calls == []


async def test_an_own_during_frame_replaces_a_missing_frigate_snapshot(tmp_path, clock):
    """2026-11-09 08:23: Frigate dropped the low-score dog, its snapshot was a 404."""

    async def no_snapshot(event_id):
        return None

    rig = build(tmp_path, clock, fetch_frame=_camera([DOG, make_jpeg()]),
                fetch_snapshot=no_snapshot)  # fmt: skip
    rig.reference.update(make_jpeg((200, 200, 200)))
    await one_visit(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    directory = rig.evidence.visit_dir(visit["id"])
    assert (directory / "during.jpg").read_bytes() == DOG
    assert rig.analyser.calls[0]["during_len"] == len(DOG)
    rig.store.close()


async def test_a_person_removing_a_poop_gemini_did_not_see_counts_it(tmp_path, clock):
    rig = build(tmp_path, clock, fetch_frame=_camera([DOG, make_jpeg((60, 40, 20)), make_jpeg()]))
    rig.reference.update(make_jpeg((200, 200, 200)))
    rig.analyser.script(make_result(pee=False, poop="none"))
    rig.analyser.script_pad(make_pad_result(pad_replaced=False, poop_removed=True))
    await _poop_then_cleaned(rig)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["final_poop"] == "poop" and visit["poop_cleared_by"] == "owner"
    assert visit["needs_review"] is True
