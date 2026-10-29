from datetime import UTC, date, datetime

import pytest

from ellie.counting import (
    LOCAL_TZ,
    Counting,
    daily_summary,
    day_bounds_utc,
    history,
    local_date,
    pees_on,
)
from ellie.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "ellie.db"))
    yield s
    s.close()


def mk(vid, started, outcome="pee", **over):
    base = {
        "id": vid,
        "source": "auto",
        "started_at": started,
        "state": "DECIDED",
        "auto_outcome": outcome,
        "final_outcome": outcome if outcome in ("pee", "no_pee", "unknown") else None,
    }
    return {**base, **over}


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def test_winter_boundary(store):
    store.insert_visit(mk("a", "2026-01-10T23:59:30Z"))
    store.insert_visit(mk("b", "2026-01-11T00:00:30Z"))
    c = Counting(store)
    assert c.pees_today(utc(2026, 1, 10, 23, 59, 40)) == 1
    assert c.pees_today(utc(2026, 1, 11, 0, 1)) == 1
    assert c.visits_today(utc(2026, 1, 11, 0, 1)) == 1


def test_summer_boundary_differs_from_utc(store):
    # 23:59:30 local (WEST) = 22:59:30Z; 00:00:30 local = 23:00:30Z
    store.insert_visit(mk("a", "2026-07-10T22:59:30Z"))
    store.insert_visit(mk("b", "2026-07-10T23:00:30Z"))
    assert local_date("2026-07-10T22:59:30Z") == date(2026, 7, 10)
    assert local_date("2026-07-10T23:00:30Z") == date(2026, 7, 11)
    c = Counting(store)
    assert c.pees_today(utc(2026, 7, 10, 23, 30)) == 1
    assert c.pees_today(utc(2026, 7, 10, 22, 59, 50)) == 1
    assert (
        pees_on(
            store.visits_started_between("2026-07-01T00:00:00Z", "2026-08-01T00:00:00Z"),
            date(2026, 7, 10),
        )
        == 1
    )


@pytest.mark.parametrize(
    ("day", "hours"), [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25), (date(2026, 1, 5), 24)]
)
def test_dst_day_lengths(day, hours):
    start, end = day_bounds_utc(day)
    delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    assert delta.total_seconds() == hours * 3600


def test_dst_days_count_once(store):
    start, end = day_bounds_utc(date(2026, 10, 25))
    store.insert_visit(mk("a", start))
    store.insert_visit(mk("b", "2026-10-25T01:30:00Z"))
    store.insert_visit(mk("c", "2026-10-25T23:59:59Z"))
    store.insert_visit(mk("d", end))
    c = Counting(store)
    assert c.pees_today(utc(2026, 10, 25, 12)) == 3
    assert c.pees_today(utc(2026, 10, 26, 12)) == 1
    s = day_bounds_utc(date(2026, 3, 29))[0]
    store.insert_visit(mk("e", s))
    store.insert_visit(mk("f", "2026-03-29T22:59:59Z"))
    assert c.pees_today(utc(2026, 3, 29, 12)) == 2


def test_uncertain_never_counted_manual_counted(store):
    store.insert_visit(mk("a", "2026-01-10T10:00:00Z", "uncertain"))
    store.insert_visit(mk("b", "2026-01-10T11:00:00Z", source="manual"))
    c = Counting(store)
    now = utc(2026, 1, 10, 12)
    assert c.pees_today(now) == 1
    assert c.uncertain_open(now) == 1
    assert c.summary(now) == {
        "date": "2026-01-10",
        "pees": 1,
        "visits": 2,
        "corrected": 0,
        "uncertain_open": 1,
        "unknown": 0,
    }


def test_flip_and_corrected(store):
    store.insert_visit(mk("a", "2026-01-10T10:00:00Z"))
    c = Counting(store)
    now = utc(2026, 1, 10, 12)
    assert c.pees_today(now) == 1
    store.update_visit("a", final_outcome="no_pee")
    assert c.pees_today(now) == 0
    visits = store.visits_started_between("2026-01-10T00:00:00Z", "2026-01-11T00:00:00Z")
    assert daily_summary(visits, date(2026, 1, 10), corrected_ids={"a"})["corrected"] == 1


def test_backdated_visit_own_day_only(store):
    store.insert_visit(mk("a", "2026-01-08T10:00:00Z", source="manual"))
    c = Counting(store)
    assert c.pees_today(utc(2026, 1, 10, 12)) == 0
    hist = c.history_days(utc(2026, 1, 10, 12), days=5)
    assert {h["date"]: h["pees"] for h in hist}["2026-01-08"] == 1
    assert sum(h["pees"] for h in hist) == 1


def test_history_shape(store):
    store.insert_visit(mk("a", "2026-01-08T10:00:00Z"))
    hist = Counting(store).history_days(utc(2026, 1, 10, 12), days=5)
    assert [h["date"] for h in hist] == [
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-10",
    ]
    assert hist[0] == {"date": "2026-01-06", "pees": 0, "visits": 0, "corrected": 0}
    assert len(history([], date(2026, 1, 10))) == 90


def test_resolve_overdue_uncertain(store):
    store.insert_visit(mk("old", "2026-01-09T10:00:00Z", "uncertain"))
    store.insert_visit(mk("today", "2026-01-10T10:00:00Z", "uncertain"))
    store.insert_visit(mk("answered", "2026-01-09T11:00:00Z", "uncertain", final_outcome="pee"))
    c = Counting(store)
    now = utc(2026, 1, 10, 12)
    assert c.resolve_overdue_uncertain(now) == ["old"]
    old = store.get_visit("old")
    assert old["final_outcome"] == "unknown"
    assert old["needs_review"] is True
    assert store.get_visit("today")["final_outcome"] is None
    assert store.get_visit("answered")["final_outcome"] == "pee"
    assert c.resolve_overdue_uncertain(now) == []


def test_last_pee(store):
    assert Counting(store).last_pee(utc(2026, 1, 10, 12)) is None
    store.insert_visit(mk("a", "2026-01-09T10:00:00Z"))
    store.insert_visit(mk("b", "2026-01-10T09:00:00Z"))
    store.insert_visit(mk("c", "2026-01-10T11:00:00Z", "no_pee"))
    assert Counting(store).last_pee(utc(2026, 1, 10, 12))["id"] == "b"


def test_tz_default():
    assert str(LOCAL_TZ) == "Europe/Lisbon"
