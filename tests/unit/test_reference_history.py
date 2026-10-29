from datetime import timedelta

from ellie.frames import HISTORY_SIZE, ReferenceKeeper
from helpers import FakeClock, make_jpeg


def keeper(tmp_path):
    clock = FakeClock()
    return ReferenceKeeper(tmp_path / "ref", clock), clock


def fill(keeper_, clock, shades):
    """Store one reference per minute; returns {shade: (jpeg, taken_at)}."""
    taken = {}
    for shade in shades:
        jpeg = make_jpeg((shade, shade, shade))
        keeper_.update(jpeg)
        taken[shade] = (jpeg, clock())
        clock.advance(60)
    return taken


def test_before_skips_a_reference_taken_just_before_the_start(tmp_path):
    keeper_, clock = keeper(tmp_path)
    taken = fill(keeper_, clock, [10, 20, 30])
    start = taken[30][1] + timedelta(
        seconds=5
    )  # the newest was taken 5 s before: maybe a person entering
    assert keeper_.before(start) == taken[20]


def test_before_takes_the_newest_when_it_is_old_enough(tmp_path):
    keeper_, clock = keeper(tmp_path)
    taken = fill(keeper_, clock, [10, 20, 30])
    assert keeper_.before(taken[30][1] + timedelta(seconds=45)) == taken[30]


def test_before_falls_back_to_the_newest_when_nothing_is_old_enough(tmp_path):
    keeper_, clock = keeper(tmp_path)
    taken = fill(keeper_, clock, [10])
    assert keeper_.before(taken[10][1] + timedelta(seconds=2)) == taken[10]


def test_before_without_any_reference_is_none(tmp_path):
    keeper_, _ = keeper(tmp_path)
    assert keeper_.before(FakeClock()()) is None


def test_the_history_is_capped(tmp_path):
    keeper_, clock = keeper(tmp_path)
    taken = fill(keeper_, clock, list(range(10, 10 + 8 * 10, 10)))
    oldest_kept = max(taken)  # the newest shade is the last one
    start = taken[oldest_kept][1] + timedelta(seconds=60 * (HISTORY_SIZE + 2))
    assert keeper_.before(start) == taken[oldest_kept]
    assert len(keeper_._history) == HISTORY_SIZE


def test_after_a_restart_only_the_latest_survives(tmp_path):
    keeper_, clock = keeper(tmp_path)
    taken = fill(keeper_, clock, [10, 20])
    reborn = ReferenceKeeper(tmp_path / "ref", clock)
    assert reborn.before(taken[20][1] + timedelta(seconds=5)) == taken[20]  # the disk copy
