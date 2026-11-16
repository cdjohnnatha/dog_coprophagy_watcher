"""Stain census (ADR-0020): the pad shows more separate stains than pees counted on it.

The owner's idea (2026-11-14): a clean pad that suddenly carries two or three separate stains
took two or three pees, whatever the visits saw. Gemini (prompt pee_v5) counts the separate pee
stains on the pad in every AFTER frame (`pad_stains`). Stains that touch merge into one, so the
number is a LOWER bound of the pees the pad took: when it is higher than the pees counted on this
pad since it was put in, the difference are pees nobody saw (2026-11-13: pad put in at 18:37, the
camera failed at 19:36, Frigate missed 21:07 and 22:31, and the 23:01 visit's AFTER showed three
separate stains while none was counted).

A found pee is counted at once (the pee count stays right without the owner), marked for review,
and placed evenly in the time window in which it must have happened. A manual pee of the owner
for the same pad later takes the found pee's place instead of adding one more.
"""

from __future__ import annotations

from datetime import datetime

CENSUS_MAX_FOUND = 3  # pees added by one census at most (a wrong count must not run away)
FOUND_PEE_REASON = "xixi achado no tapete"


def missing_pees(stains: int | None, counted_on_pad: int) -> int:
    """Pees the stains prove beyond those counted on this pad (0 when the census says nothing)."""
    if stains is None or stains <= counted_on_pad:
        return 0
    return min(stains - counted_on_pad, CENSUS_MAX_FOUND)


def found_pee_times(start: datetime, end: datetime, n: int) -> list[datetime]:
    """`n` times spread evenly inside (start, end), oldest first."""
    if n <= 0:
        return []
    if end <= start:
        return [end] * n
    step = (end - start) / (n + 1)
    return [start + step * (i + 1) for i in range(n)]


def is_found_pee(visit: dict) -> bool:
    """Added by a census and not yet taken over by the owner's manual pee."""
    return bool(visit.get("found_by"))
