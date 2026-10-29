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
