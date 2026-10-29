from datetime import timedelta

import pytest

from ellie.visits import VisitTracker, iso
from helpers import FakeClock


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def tracker():
    return VisitTracker(merge_gap_s=60, clear_s=30, capture_timeout_s=300)


def kinds(actions):
    return [a.kind for a in actions]


def run(tracker, clock, seconds, step=1):
    """Tick once per `step` seconds and collect every action."""
    out = []
    for _ in range(int(seconds / step)):
        clock.advance(step)
        out += tracker.tick(clock())
    return out


def enter(tracker, clock):
    clock.advance(0)
    actions = tracker.on_pad_dog(1, clock())
    tracker.on_tray_dog(1, clock())
    return actions


def leave(tracker, clock):
    actions = tracker.on_pad_dog(0, clock())
    tracker.on_tray_dog(0, clock())
    return actions


def test_single_visit_from_entry_to_capture(tracker, clock):
    tracker.tick(clock())
    started = enter(tracker, clock)
    assert kinds(started) == ["started"]
    row = started[0].visit
    assert row["id"] == "v_20261029_120000"
    assert row["state"] == "OPEN" and row["entries"] == 1 and row["source"] == "auto"
    assert row["started_at"] == "2026-10-29T12:00:00Z"

    clock.advance(10)
    assert kinds(leave(tracker, clock)) == ["updated"]
    assert tracker.active["state"] == "CLOSING"
    assert tracker.active["ended_at"] == "2026-10-29T12:00:10Z"

    actions = run(tracker, clock, 59)
    assert "capture_after" not in kinds(actions)  # merge gap not over yet
    actions = run(tracker, clock, 2)
    assert kinds(actions).count("capture_after") == 1
    capture = next(a for a in actions if a.kind == "capture_after")
    assert capture.visit["state"] == "ANALYSING"
    assert tracker.active is None


def test_reentry_inside_the_gap_is_one_visit(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    clock.advance(5)
    leave(tracker, clock)
    clock.advance(20)
    again = enter(tracker, clock)
    assert kinds(again) == ["updated"]
    assert tracker.active["entries"] == 2 and tracker.active["state"] == "OPEN"
    clock.advance(5)
    leave(tracker, clock)
    clock.advance(10)
    enter(tracker, clock)
    assert tracker.active["entries"] == 3
    clock.advance(5)
    leave(tracker, clock)
    actions = run(tracker, clock, 100)
    assert kinds(actions).count("capture_after") == 1


def test_reentry_while_waiting_for_a_clear_area_reopens_the_same_visit(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    clock.advance(2)
    leave(tracker, clock)
    tracker.on_tray_person(1, clock())  # someone keeps the area busy
    run(tracker, clock, 65)
    assert tracker.active["state"] == "CAPTURING"
    enter(tracker, clock)
    assert tracker.active["state"] == "OPEN" and tracker.active["entries"] == 2


def test_after_capture_a_new_entry_is_a_new_visit(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    clock.advance(2)
    leave(tracker, clock)
    first = run(tracker, clock, 70)
    first_id = next(a for a in first if a.kind == "capture_after").visit["id"]
    second = enter(tracker, clock)
    assert kinds(second) == ["started"]
    assert second[0].visit["id"] != first_id


def test_area_never_clear_ends_unknown_after_five_minutes(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    clock.advance(2)
    leave(tracker, clock)
    tracker.on_tray_person(1, clock())
    actions = run(tracker, clock, 400)
    unknown = [a for a in actions if a.kind == "unknown"]
    assert len(unknown) == 1 and unknown[0].reason == "area_not_clear"
    assert unknown[0].visit["state"] == "DECIDED"
    assert "capture_after" not in kinds(actions)
    assert tracker.active is None


def test_flapping_counts_do_not_trigger_an_early_capture(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    clock.advance(2)
    leave(tracker, clock)
    run(tracker, clock, 55)
    # the person count flaps 1/0/1/0 inside one second, like the real log
    for count in (1, 0, 1, 0):
        tracker.on_tray_person(count, clock())
    actions = run(tracker, clock, 20)
    assert "capture_after" not in kinds(actions)  # clear window restarted at the last 0
    actions = run(tracker, clock, 15)
    assert kinds(actions).count("capture_after") == 1


def test_dog_still_in_tray_area_delays_the_capture(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    clock.advance(2)
    tracker.on_pad_dog(0, clock())  # pad_zone says she left, tray_area says she is still near
    actions = run(tracker, clock, 90)
    assert "capture_after" not in kinds(actions)
    tracker.on_tray_dog(0, clock())
    actions = run(tracker, clock, 31)
    assert kinds(actions).count("capture_after") == 1


def test_person_during_the_visit_is_flagged_once(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    assert tracker.active["person_present"] is False
    first = tracker.on_tray_person(1, clock())
    assert kinds(first) == ["updated"] and tracker.active["person_present"] is True
    assert tracker.on_tray_person(2, clock()) == []


def test_frigate_event_ids_attach_dedupe_and_survive_arriving_early(tracker, clock):
    tracker.tick(clock())
    tracker.on_event("1793288511.990085-yohln0", clock())  # arrives before the count topic
    enter(tracker, clock)
    assert tracker.active["frigate_event_ids"] == ["1793288511.990085-yohln0"]
    assert tracker.on_event("1793288511.990085-yohln0", clock()) == []
    assert kinds(tracker.on_event("1793288520.1-abc", clock())) == ["updated"]
    assert tracker.active["frigate_event_ids"] == ["1793288511.990085-yohln0", "1793288520.1-abc"]


def test_an_old_event_is_not_attached_to_a_later_visit(tracker, clock):
    tracker.tick(clock())
    tracker.on_event("old-event", clock())
    run(tracker, clock, 30)
    enter(tracker, clock)
    assert tracker.active["frigate_event_ids"] == []


def test_frigate_going_offline_ends_the_visit_unknown(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    actions = tracker.on_frigate_available(False, clock())
    assert [(a.kind, a.reason) for a in actions] == [("unknown", "frigate_offline")]
    assert tracker.active is None
    assert tracker.on_frigate_available(False, clock()) == []


def test_seconds_on_the_pad_are_accumulated(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    tracker.on_tray_pad_dog(1, clock())
    clock.advance(4)
    tracker.on_tray_pad_dog(0, clock())
    clock.advance(3)
    tracker.on_tray_pad_dog(1, clock())
    clock.advance(2.5)
    leave(tracker, clock)  # leaving flushes the open interval
    assert tracker.active["on_pad_s"] == 6.5


def test_changing_the_merge_gap_at_runtime(tracker, clock):
    tracker.tick(clock())
    enter(tracker, clock)
    clock.advance(1)
    leave(tracker, clock)
    tracker.merge_gap_s = 15
    actions = run(tracker, clock, 40)
    assert kinds(actions).count("capture_after") == 1


def test_two_visits_in_the_same_second_get_different_ids(tracker, clock):
    tracker.tick(clock())
    first = enter(tracker, clock)[0].visit["id"]
    tracker.on_frigate_available(False, clock())
    leave(tracker, clock)
    second = enter(tracker, clock)[0].visit["id"]
    assert first != second


def test_iso_is_utc_with_z():
    clock = FakeClock()
    assert iso(clock() + timedelta(hours=1, seconds=5)) == "2026-10-29T13:00:05Z"
