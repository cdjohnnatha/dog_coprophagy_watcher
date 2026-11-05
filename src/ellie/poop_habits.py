"""Ellie's poop habits: usual time, interval and hours since the last one (ADR-0015).

Pure functions: the clock and the list of counted poops are passed in. The window is the last
14 local days; the typical time is the median of the time of day with the day starting at 03:00
(so a 02:43 and a 06:13 poop sort together instead of sitting at both ends of the clock).
There is deliberately no prediction and no link with the feeder.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from statistics import median

from ellie.counting import LOCAL_TZ

WINDOW_DAYS = 14
MIN_POOPS = 5  # typical time and window only from this many poops in the window
MIN_FOR_INTERVAL = 3  # the median gap needs at least two gaps
DAY_START_MIN = 3 * 60  # the "day" of the time-of-day statistics starts at 03:00 local
PERIODS = (("madrugada", 0, 6), ("manha", 6, 12), ("tarde", 12, 18), ("noite", 18, 24))


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..1) of a sorted, non-empty sequence."""
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def _clock_text(shifted_minutes: float) -> str:
    """Minutes since 03:00 -> 'HH:MM' local."""
    minutes = int(round(shifted_minutes) + DAY_START_MIN) % 1440
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _empty(now: datetime) -> dict:
    return {
        "window_days": WINDOW_DAYS,
        "n": 0,
        "ready": False,
        "last_at": None,
        "hours_since": None,
        "typical_time": None,
        "typical_from": None,
        "typical_to": None,
        "interval_hours": None,
        "per_day": None,
        "periods": {name: 0 for name, _, _ in PERIODS},
    }


def compute(poop_times_utc: Sequence[datetime], now: datetime) -> dict:
    """The `poop_habits` object of `ellie/state` from the start times of every counted poop."""
    result = _empty(now)
    past = sorted(t for t in poop_times_utc if t <= now)  # a poop "in the future" is ignored
    if not past:
        return result

    last = past[-1]
    result["last_at"] = last.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    result["hours_since"] = round((now - last).total_seconds() / 3600, 1)

    now_local = now.astimezone(LOCAL_TZ)
    start = (now_local - timedelta(days=WINDOW_DAYS - 1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    window = [t for t in past if t.astimezone(LOCAL_TZ) >= start]
    result["n"] = len(window)
    if not window:
        return result

    # per day: from the day of the first poop in the window (data starts there), at most 14 days
    first_day = window[0].astimezone(LOCAL_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    days = max(1.0, (now_local - max(first_day, start)).total_seconds() / 86400)
    result["per_day"] = round(len(window) / days, 1)

    for t in window:
        hour = t.astimezone(LOCAL_TZ).hour
        for name, low, high in PERIODS:
            if low <= hour < high:
                result["periods"][name] += 1

    if len(window) >= MIN_FOR_INTERVAL:
        gaps = [(b - a).total_seconds() / 3600 for a, b in pairwise(window)]
        result["interval_hours"] = round(median(gaps), 1)

    if len(window) >= MIN_POOPS:
        shifted = sorted(
            (t.astimezone(LOCAL_TZ).hour * 60 + t.astimezone(LOCAL_TZ).minute - DAY_START_MIN)
            % 1440
            for t in window
        )
        result["ready"] = True
        result["typical_time"] = _clock_text(median(shifted))
        result["typical_from"] = _clock_text(_percentile(shifted, 0.25))
        result["typical_to"] = _clock_text(_percentile(shifted, 0.75))
    return result
