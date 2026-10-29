"""Pee counting by local calendar day (pure helpers plus a Store-backed wrapper)."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from zoneinfo import ZoneInfo

from ellie.store import Store

LOCAL_TZ = ZoneInfo("Europe/Lisbon")
_ISO_Z = "%Y-%m-%dT%H:%M:%SZ"


def local_date(ts: str, tz: tzinfo = LOCAL_TZ) -> date:
    """Local calendar date of a UTC ISO-8601 timestamp."""
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(tz).date()


def _midnight_utc(day: date, tz: tzinfo) -> datetime:
    return datetime.combine(day, time.min, tzinfo=tz).astimezone(UTC)


def day_bounds_utc(day: date, tz: tzinfo = LOCAL_TZ) -> tuple[str, str]:
    """[start, end) of a local day as UTC ISO strings ending in 'Z'."""
    start = _midnight_utc(day, tz)
    end = _midnight_utc(day + timedelta(days=1), tz)
    return start.strftime(_ISO_Z), end.strftime(_ISO_Z)


def pees_on(visits: Iterable[dict], day: date, tz: tzinfo = LOCAL_TZ) -> int:
    """Visits with final_outcome 'pee' that started on the given local day."""
    return sum(
        1
        for v in visits
        if v.get("final_outcome") == "pee" and local_date(v["started_at"], tz) == day
    )


def daily_summary(
    visits: Iterable[dict],
    day: date,
    tz: tzinfo = LOCAL_TZ,
    corrected_ids: Collection[str] = (),
) -> dict:
    """Summary of the visits that started on the given local day."""
    todays = [v for v in visits if local_date(v["started_at"], tz) == day]
    return {
        "date": day.isoformat(),
        "pees": sum(1 for v in todays if v.get("final_outcome") == "pee"),
        "visits": len(todays),
        "corrected": sum(1 for v in todays if v["id"] in corrected_ids),
        "uncertain_open": sum(
            1
            for v in todays
            if v.get("auto_outcome") == "uncertain" and v.get("final_outcome") is None
        ),
        "unknown": sum(1 for v in todays if v.get("final_outcome") == "unknown"),
    }


def history(
    visits: Iterable[dict],
    today: date,
    days: int = 90,
    tz: tzinfo = LOCAL_TZ,
    corrected_ids: Collection[str] = (),
) -> list[dict]:
    """Per-day {date, pees, visits, corrected} for the last `days` days, oldest first."""
    by_day: dict[date, list[dict]] = {}
    for v in visits:
        by_day.setdefault(local_date(v["started_at"], tz), []).append(v)
    result = []
    for offset in range(days - 1, -1, -1):
        day = today - timedelta(days=offset)
        s = daily_summary(by_day.get(day, []), day, tz, corrected_ids)
        result.append({k: s[k] for k in ("date", "pees", "visits", "corrected")})
    return result


class Counting:
    """Store-backed counting queries relative to a given 'now' (aware UTC)."""

    def __init__(self, store: Store, tz: tzinfo = LOCAL_TZ) -> None:
        self._store = store
        self._tz = tz

    def _today(self, now: datetime) -> date:
        return now.astimezone(self._tz).date()

    def _visits_on(self, day: date) -> list[dict]:
        return self._store.visits_started_between(*day_bounds_utc(day, self._tz))

    def _visits_since(self, first_day: date, today: date) -> list[dict]:
        start = day_bounds_utc(first_day, self._tz)[0]
        end = day_bounds_utc(today, self._tz)[1]
        return self._store.visits_started_between(start, end)

    def pees_today(self, now: datetime) -> int:
        return pees_on(self._visits_on(self._today(now)), self._today(now), self._tz)

    def visits_today(self, now: datetime) -> int:
        return len(self._visits_on(self._today(now)))

    def uncertain_open(self, now: datetime) -> int:
        return self.summary(now)["uncertain_open"]

    def summary(self, now: datetime) -> dict:
        today = self._today(now)
        return daily_summary(self._visits_on(today), today, self._tz)

    def history_days(
        self, now: datetime, days: int = 90, corrected_ids: Collection[str] = ()
    ) -> list[dict]:
        today = self._today(now)
        visits = self._visits_since(today - timedelta(days=days - 1), today)
        return history(visits, today, days, self._tz, corrected_ids)

    def last_pee(self, now: datetime, lookback_days: int = 30) -> dict | None:
        today = self._today(now)
        visits = self._visits_since(today - timedelta(days=lookback_days), today)
        pees = [v for v in visits if v.get("final_outcome") == "pee"]
        return max(pees, key=lambda v: v["started_at"], default=None)

    def resolve_overdue_uncertain(self, now: datetime) -> list[str]:
        """Turn unanswered 'uncertain' visits of earlier days into 'unknown'."""
        today = self._today(now)
        start = day_bounds_utc(today - timedelta(days=7), self._tz)[0]
        end = day_bounds_utc(today, self._tz)[0]
        resolved = []
        for v in self._store.visits_started_between(start, end):
            if v.get("auto_outcome") == "uncertain" and v.get("final_outcome") is None:
                self._store.update_visit(v["id"], final_outcome="unknown", needs_review=True)
                resolved.append(v["id"])
        return resolved
