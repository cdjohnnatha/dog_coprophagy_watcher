"""Poop habits (ADR-0015): usual time, interval, hours since. Pure, with the real poops of the week."""

from datetime import UTC, datetime, timedelta

from ellie.poop_habits import compute

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)  # winter: Lisbon == UTC


def at(day: str, hhmm: str) -> datetime:
    return datetime.fromisoformat(f"{day}T{hhmm}:00+00:00")


REAL = [
    at("2026-10-29", "20:18"),
    at("2026-10-30", "06:53"),
    at("2026-10-31", "16:12"),
    at("2026-11-01", "09:05"),
    at("2026-11-01", "09:37"),
    at("2026-11-02", "02:43"),
    at("2026-11-02", "09:00"),
    at("2026-11-03", "10:00"),
    at("2026-11-04", "07:28"),
    at("2026-11-04", "19:48"),
    at("2026-11-05", "06:13"),
]


def test_the_real_week_gives_a_usual_morning_time() -> None:
    h = compute(REAL, NOW)
    assert h["n"] == 11 and h["ready"] is True
    assert h["last_at"] == "2026-11-05T06:13:00Z" and h["hours_since"] == 5.8
    assert "07:30" <= h["typical_time"] <= "10:00"  # the median is a morning time
    assert h["typical_from"] < h["typical_time"] < h["typical_to"]
    assert h["periods"] == {"madrugada": 1, "manha": 7, "tarde": 1, "noite": 2}
    assert h["per_day"] == 1.5  # 11 poops in the 7.5 days since the first one
    assert 10 < h["interval_hours"] < 30


def test_the_day_starts_at_three_so_a_night_poop_sorts_with_the_early_morning() -> None:
    times = [at(f"2026-11-0{d}", hhmm) for d, hhmm in ((1, "02:43"), (2, "06:13"), (3, "06:53"))]
    times += [at("2026-11-04", "07:28"), at("2026-11-05", "07:50")]
    h = compute(times, NOW)
    assert h["ready"] and h["typical_time"] == "07:28"
    assert h["typical_from"] == "06:53" and h["typical_to"] == "07:50"


def test_no_poop_at_all() -> None:
    h = compute([], NOW)
    assert h["n"] == 0 and h["ready"] is False
    assert h["hours_since"] is None and h["last_at"] is None and h["typical_time"] is None
    assert h["interval_hours"] is None and h["per_day"] is None
    assert set(h["periods"]) == {"madrugada", "manha", "tarde", "noite"}


def test_one_poop_gives_hours_since_but_no_habits() -> None:
    h = compute([at("2026-11-05", "06:13")], NOW)
    assert h["n"] == 1 and h["hours_since"] == 5.8 and h["ready"] is False
    assert h["typical_time"] is None and h["interval_hours"] is None


def test_four_poops_are_not_enough_for_a_usual_time_but_give_an_interval() -> None:
    h = compute(REAL[-4:], NOW)
    assert h["n"] == 4 and h["ready"] is False
    assert h["typical_time"] is None and h["typical_from"] is None and h["typical_to"] is None
    assert h["interval_hours"] is not None


def test_hours_since_looks_beyond_the_window() -> None:
    old = at("2026-10-01", "08:00")  # 35 days before NOW: outside the 14-day window
    h = compute([old], NOW)
    assert h["n"] == 0 and h["last_at"] == "2026-10-01T08:00:00Z"
    assert h["hours_since"] == round((NOW - old).total_seconds() / 3600, 1)


def test_a_poop_in_the_future_is_ignored() -> None:
    h = compute([*REAL, NOW + timedelta(hours=2)], NOW)
    assert h["n"] == 11 and h["hours_since"] == 5.8


def test_hours_since_grows_with_the_clock() -> None:
    first = compute(REAL, NOW)["hours_since"]
    later = compute(REAL, NOW + timedelta(hours=7))["hours_since"]
    assert later == round(first + 7, 1)


def test_poops_across_midnight_land_in_their_local_period() -> None:
    # 23:50 on 2026-11-03 and 00:10 on 2026-11-04 are noite and madrugada
    h = compute([at("2026-11-03", "23:50"), at("2026-11-04", "00:10")], NOW)
    assert h["periods"] == {"madrugada": 1, "manha": 0, "tarde": 0, "noite": 1}
    assert h["interval_hours"] is None  # two poops: one gap is not a median


def test_the_summer_time_end_does_not_shift_the_local_clock() -> None:
    """The clocks went back on 2026-10-25 at 01:00 UTC: 07:30 local is 06:30 UTC before and
    07:30 UTC after, and all five are 'manha' at the same local time."""
    times = [
        datetime(2026, 10, 23, 6, 30, tzinfo=UTC),  # 07:30 WEST
        datetime(2026, 10, 24, 6, 30, tzinfo=UTC),  # 07:30 WEST
        datetime(2026, 10, 26, 7, 30, tzinfo=UTC),  # 07:30 WET
        datetime(2026, 10, 27, 7, 30, tzinfo=UTC),
        datetime(2026, 10, 28, 7, 30, tzinfo=UTC),
    ]
    h = compute(times, datetime(2026, 10, 29, 12, 0, tzinfo=UTC))
    assert h["ready"] and h["typical_time"] == "07:30"
    assert h["typical_from"] == "07:30" and h["typical_to"] == "07:30"
    assert h["periods"]["manha"] == 5
