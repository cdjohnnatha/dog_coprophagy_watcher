import pytest

from ellie.pads import PresenceTracker
from helpers import FakeClock


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def tracker():
    return PresenceTracker()


def kinds(actions):
    return [a.kind for a in actions]


def run(tracker, clock, seconds):
    out = []
    for _ in range(int(seconds)):
        clock.advance(1)
        out += tracker.tick(clock())
    return out


def test_a_person_staying_a_while_is_captured_after_a_clear_period(tracker, clock):
    tracker.tick(clock())
    started = tracker.on_person(1, clock())
    assert kinds(started) == ["started"] and started[0].presence["id"] == "s_20261029_120000"
    clock.advance(40)
    tracker.on_person(0, clock())
    assert run(tracker, clock, 29) == []  # not yet 30 s clear
    (capture,) = run(tracker, clock, 2)
    assert capture.kind == "capture" and capture.presence["state"] == "ANALYSING"
    assert capture.presence["ended_at"] == "2026-10-29T12:00:40Z"  # last time seen, not now
    assert not tracker.present


def test_a_short_pass_is_dropped_without_analysis(tracker, clock):
    tracker.tick(clock())
    tracker.on_person(1, clock())
    clock.advance(5)
    tracker.on_person(0, clock())
    (dropped,) = run(tracker, clock, 31)
    assert dropped.kind == "dropped" and dropped.reason == "too_short"


def test_flapping_counts_do_not_split_or_end_a_presence(tracker, clock):
    tracker.tick(clock())
    tracker.on_person(1, clock())
    clock.advance(20)
    for count in (0, 1, 0, 1, 0):  # the real log flipped like this inside one second
        tracker.on_person(count, clock())
    assert run(tracker, clock, 20) == []  # the clear window restarted at the last 0
    (capture,) = run(tracker, clock, 12)
    assert capture.kind == "capture"


def test_a_dog_around_keeps_the_area_busy(tracker, clock):
    tracker.tick(clock())
    tracker.on_person(1, clock())
    clock.advance(30)
    tracker.on_person(0, clock())
    tracker.on_dog_area(1, clock())
    assert run(tracker, clock, 90) == []
    tracker.on_dog_area(0, clock())
    assert kinds(run(tracker, clock, 31)) == ["capture"]


def test_a_dog_visit_takes_over_and_drops_the_presence(tracker, clock):
    tracker.tick(clock())
    tracker.on_person(1, clock())
    clock.advance(15)
    (dropped,) = tracker.set_visit_active(True, clock())
    assert dropped.kind == "dropped" and dropped.reason == "dog_visit"
    assert tracker.on_person(1, clock()) == []  # a person during a visit starts no presence
    assert tracker.set_visit_active(False, clock()) == []


def test_presence_ids_are_unique_in_the_same_second(tracker, clock):
    tracker.tick(clock())
    first = tracker.on_person(1, clock())[0].presence["id"]
    tracker.set_visit_active(True, clock())
    tracker.set_visit_active(False, clock())
    second = tracker.on_person(0, clock()) or tracker.on_person(1, clock())
    assert second[0].presence["id"] != first


# --- ADR-0022: a real person's event that entered tray_area keeps the presence going ---


def test_a_cleanup_mostly_outside_the_zone_is_captured_while_the_event_lasts(tracker, clock):
    """2026-11-18 12:14: zone count 2 s, the person event 18 s; the cleanup must be analysed."""
    tracker.tick(clock())
    tracker.on_person_event("e1", True, clock())
    tracker.on_person(1, clock())
    clock.advance(2)
    tracker.on_person(0, clock())  # bent over the tray: the box left the zone
    assert run(tracker, clock, 16) == []
    tracker.on_person_event("e1", False, clock())  # the event ends: they left
    (capture,) = run(tracker, clock, 31)
    assert capture.kind == "capture"
    assert capture.presence["ended_at"] == "2026-10-29T12:00:18Z"


def test_an_event_with_a_lost_end_does_not_keep_the_presence_open(tracker, clock):
    tracker.tick(clock())
    tracker.on_person_event("e1", True, clock())
    tracker.on_person(1, clock())
    clock.advance(20)
    tracker.on_person(0, clock())
    actions = run(tracker, clock, 700)
    assert "capture" in kinds(actions) and not tracker.present


def test_the_end_of_an_unknown_event_changes_nothing(tracker, clock):
    tracker.tick(clock())
    assert tracker.on_person_event("other", False, clock()) == []
    assert not tracker.present
