"""Pure tracker of a person at the tray (US7): when did someone stand there, and for how long.

Same rules as the visit tracker, from the real Frigate capture (research R15): counts flap, so
a presence ends only after the tray area has been clear of people and dogs for `clear_s`, and
it must last `min_presence_s` to be worth analysing. While a dog visit is open the visit owns
the area (its analysis also reports `pad_replaced`), so no presence is tracked.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from typing import Any, Literal

from ellie.counting import LOCAL_TZ
from ellie.visits import iso

PresenceKind = Literal["started", "dropped", "capture"]


@dataclass(frozen=True)
class PresenceAction:
    kind: PresenceKind
    presence: dict[str, Any]
    reason: str | None = None


class PresenceTracker:
    def __init__(
        self, min_presence_s: float = 10, clear_s: float = 30, tz: tzinfo = LOCAL_TZ
    ) -> None:
        self.min_presence_s = min_presence_s
        self.clear_s = clear_s
        self._tz = tz
        self._person = 0
        self._dog = 0
        self._visit_active = False
        self._clear_since: datetime | None = None
        self._started: datetime | None = None
        self._last_person: datetime | None = None
        self._id: str | None = None
        self._used: set[str] = set()

    @property
    def present(self) -> bool:
        return self._started is not None

    def on_person(self, count: int, now: datetime) -> list[PresenceAction]:
        """`frigate/tray_area/person`."""
        previous, self._person = self._person, max(count, 0)
        if self._person == 0 and previous > 0 and self._started is not None:
            self._last_person = now  # counts are only published on change: this is when they left
        if self._person > 0:
            self._last_person, self._clear_since = now, None
            if self._started is None and not self._visit_active:
                self._started, self._id = now, self._new_id(now)
                return [PresenceAction("started", self._row(None))]
            return []
        self._update_clear(now)
        return []

    def on_dog_area(self, count: int, now: datetime) -> list[PresenceAction]:
        """`frigate/tray_area/dog`: a dog around also keeps the area "busy"."""
        self._dog = max(count, 0)
        self._update_clear(now)
        return []

    def set_visit_active(self, active: bool, now: datetime) -> list[PresenceAction]:
        """A dog visit takes over: a presence in progress is dropped without analysis."""
        self._visit_active = active
        if active and self._started is not None:
            return [self._drop("dog_visit")]
        return []

    def tick(self, now: datetime) -> list[PresenceAction]:
        if self._clear_since is None and self._person == 0 and self._dog == 0:
            self._clear_since = now
        if self._started is None or self._clear_since is None:
            return []
        if (now - self._clear_since).total_seconds() < self.clear_s:
            return []
        stay = (self._last_person - self._started).total_seconds()
        if stay < self.min_presence_s:
            return [self._drop("too_short")]
        row = self._row(self._last_person)
        self._reset()
        return [PresenceAction("capture", row)]

    def _update_clear(self, now: datetime) -> None:
        if self._person > 0 or self._dog > 0:
            self._clear_since = None
        elif self._clear_since is None:
            self._clear_since = now

    def _drop(self, reason: str) -> PresenceAction:
        row = self._row(self._last_person)
        self._reset()
        return PresenceAction("dropped", row, reason)

    def _reset(self) -> None:
        self._started = self._last_person = self._id = None

    def _row(self, ended: datetime | None) -> dict[str, Any]:
        return {
            "id": self._id,
            "started_at": iso(self._started),
            "ended_at": iso(ended) if ended else None,
            "state": "PRESENT" if ended is None else "ANALYSING",
        }

    def _new_id(self, now: datetime) -> str:
        stamp = now.astimezone(self._tz)
        presence_id = f"s_{stamp:%Y%m%d_%H%M%S}"
        while presence_id in self._used:
            stamp += timedelta(seconds=1)
            presence_id = f"s_{stamp:%Y%m%d_%H%M%S}"
        self._used.add(presence_id)
        return presence_id
