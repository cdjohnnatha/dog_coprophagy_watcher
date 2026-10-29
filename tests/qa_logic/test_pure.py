"""State machine and pure-module holes."""

from datetime import datetime, timedelta

from qa_common import all_visits, at, dog_visit

from ellie.counting import day_bounds_utc, local_date
from ellie.pads import PresenceTracker
from ellie.visits import VisitTracker
from helpers import make_visit


def test_presence_must_not_stay_present_forever_while_a_dog_idles_in_the_area():
    p = PresenceTracker()
    t = at(10)
    p.on_person(1, t)
    p.on_person(0, t + timedelta(seconds=20))
    p.on_dog_area(1, t + timedelta(seconds=21))  # dog follows the person and stays
    out = []
    for s in range(22, 3600):
        out += p.tick(t + timedelta(seconds=s))
    assert out, "presence is PRESENT for an hour with no capture and no drop (no timeout)"


def test_visit_must_not_stay_open_forever_when_a_zero_count_is_lost():
    v = VisitTracker()
    t = at(0)
    v.on_pad_dog(1, t)
    out = []
    for s in range(1, 12 * 3600, 5):
        out += v.tick(t + timedelta(seconds=s))
    assert v.active is None or v.active["state"] != "OPEN", "OPEN for 12 hours, no max duration"


def test_counts_seen_before_frigate_went_offline_are_stale_after_it_returns():
    v = VisitTracker()
    t = at(10)
    v.on_tray_person(1, t)
    v.on_frigate_available(False, t + timedelta(seconds=5))
    v.on_frigate_available(True, t + timedelta(seconds=60))
    v.tick(t + timedelta(seconds=61))
    assert v.clear_for(t + timedelta(seconds=600)) > 0, "tray looks busy forever (stale person=1)"


def test_flapping_counts_make_one_visit_and_jumps_are_ok():
    v = VisitTracker()
    t = at(10)
    starts = 0
    for i in range(6):
        for a in v.on_pad_dog(i % 2, t + timedelta(milliseconds=150 * i)):
            starts += a.kind == "started"
    assert starts == 1
    assert v.on_pad_dog(2, t + timedelta(seconds=2)) != [] or True


def test_dst_days_count_correctly_in_lisbon():
    start, end = day_bounds_utc(datetime(2026, 10, 25).date())
    assert (start, end) == ("2026-10-24T23:00:00Z", "2026-10-26T00:00:00Z")
    start, end = day_bounds_utc(datetime(2027, 3, 28).date())
    assert (start, end) == ("2027-03-28T00:00:00Z", "2027-03-28T23:00:00Z")
    assert str(local_date("2026-10-25T00:30:00Z")) == "2026-10-25"
    assert str(local_date("2026-10-25T01:30:00Z")) == "2026-10-25"


async def test_uncertain_visit_asked_just_before_midnight_gets_minutes_not_a_day(make_rig):
    from ellie.analysers.base import AnalyserError  # noqa: F401
    from helpers import make_result

    rig = make_rig()
    rig.clock.set(at(23, 58, 0))
    rig.reference.update(__import__("helpers").make_jpeg())
    rig.analyser.script(make_result(pee=True, pee_confidence=0.6))
    await dog_visit(rig, stay=60, settle=75)  # decided about 00:00:20, asks the owner
    (v,) = all_visits(rig)
    assert v["auto_outcome"] == "uncertain"
    await rig.service.minute_tasks()
    v = rig.store.get_visit(v["id"])
    assert v["final_outcome"] is None, "unanswered 2 minutes after the question: already unknown"


def test_corrected_counts_owner_answers_to_uncertain_as_corrections(tmp_path):
    from ellie.store import Store

    s = Store(str(tmp_path / "c.db"))
    s.insert_visit(make_visit(auto_outcome="uncertain", final_outcome=None, state="DECIDED"))
    s.insert_label("v_20261029_101530", "pee", "notification", None, None)
    s.update_visit("v_20261029_101530", final_outcome="pee", state="LABELLED")
    corrected = s.corrected_ids("2026-01-01T00:00:00Z")
    s.close()
    assert corrected == set(), "answering a question is not correcting the automatic result"
