"""Pure visit state machine: Frigate zone counts in, persistence and capture actions out.

No I/O and no clock of its own: every input carries `now` (aware datetime), so tests drive
it with a fake clock. Behaviour is based on what Frigate really publishes (research R15):
counts flap within a second, so every "gone" or "clear" decision is debounced.

One visit is active at a time:

    OPEN      dog in pad_zone or tray_area (ADR-0011)
    CLOSING   dog left both zones, waiting `merge_gap_s` for a re-entry (she circles)
    CAPTURING gap elapsed, waiting for tray_area to be clear of dog and person for `clear_s`

A re-entry while CLOSING or CAPTURING reopens the same visit: its after frame has not been
taken yet, so one before/after pair still covers the whole stay. Once the after frame is due
the visit is handed over (`capture_after`) and the tracker is free for the next one.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any, Literal

from ellie.counting import LOCAL_TZ

ActionKind = Literal["started", "updated", "capture_after", "unknown"]

PENDING_EVENT_TTL_S = 15


@dataclass(frozen=True)
class Action:
    """Something the caller must do. `visit` holds store column values (see `_Visit.row`)."""

    kind: ActionKind
    visit: dict[str, Any]
    reason: str | None = None


@dataclass
class _Visit:
    id: str
    started_at: datetime
    ended_at: datetime | None = None
    entries: int = 1
    person_present: bool = False
    event_ids: list[str] = field(default_factory=list)
    state: str = "OPEN"
    on_pad_s: float = 0.0
    capturing_since: datetime | None = None
    open_since: datetime | None = None

    def row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": "auto",
            "started_at": iso(self.started_at),
            "ended_at": iso(self.ended_at) if self.ended_at else None,
            "entries": self.entries,
            "person_present": self.person_present,
            "frigate_event_ids": list(self.event_ids),
            "state": self.state,
            "on_pad_s": round(self.on_pad_s, 1),
        }


def iso(moment: datetime) -> str:
    """UTC ISO-8601 with a Z suffix, the format the store and counting use."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class VisitTracker:
    def __init__(
        self,
        merge_gap_s: float = 60,
        clear_s: float = 30,
        capture_timeout_s: float = 300,
        tz: tzinfo = LOCAL_TZ,
        max_open_s: float = 7200,
    ) -> None:
        self.merge_gap_s = merge_gap_s
        self.clear_s = clear_s
        self.capture_timeout_s = capture_timeout_s
        self.max_open_s = max_open_s  # a lost 'left' message must not keep a visit open for ever
        self._frigate_online = True
        self._tz = tz
        self._active: _Visit | None = None
        self._pad_dog = False
        self._area_dog = 0
        self._area_person = 0
        self._clear_since: datetime | None = None
        self._on_pad_since: datetime | None = None
        self._pending_events: list[tuple[str, datetime]] = []
        self._recent_ids: deque[str] = deque(maxlen=20)
        self.id_taken: Callable[[str], bool] | None = None  # the store: manual visits, restarts

    @property
    def active(self) -> dict[str, Any] | None:
        return self._active.row() if self._active else None

    def clear_for(self, now: datetime) -> float:
        """Seconds tray_area has had no dog and no person (0 when it is busy)."""
        return (now - self._clear_since).total_seconds() if self._clear_since else 0.0

    # --- inputs: Frigate plain count topics (never the /active variants) ---

    def on_pad_dog(self, count: int, now: datetime) -> list[Action]:
        """`frigate/pad_zone/dog`: one of the two signals that she is at the tray."""
        self._pad_dog = count > 0
        return self._dog_moved(now)

    def on_tray_dog(self, count: int, now: datetime) -> list[Action]:
        """`frigate/tray_area/dog`: the other signal. She often stands in front of the tray with
        her feet just outside pad_zone (seen live on 2026-10-30: a pee there was missed)."""
        self._area_dog = max(count, 0)
        self._update_clear(now)
        return self._dog_moved(now)

    def _dog_moved(self, now: datetime) -> list[Action]:
        """Start, reopen or close the visit from "is she in pad_zone or tray_area"."""
        inside = self._pad_dog or self._area_dog > 0
        visit = self._active
        if inside:
            self._clear_since = None
            if visit is None:
                self._active = visit = self._new_visit(now)
                return [Action("started", visit.row())]
            if visit.state == "OPEN":
                return []  # still there (the two zones flap independently)
            visit.state, visit.entries, visit.capturing_since = "OPEN", visit.entries + 1, None
            visit.open_since = now
            return [Action("updated", visit.row())]
        if visit is None or visit.state != "OPEN":
            return []
        self._flush_on_pad(now)
        visit.state, visit.ended_at = "CLOSING", now
        return [Action("updated", visit.row())]

    def on_tray_person(self, count: int, now: datetime) -> list[Action]:
        """`frigate/tray_area/person`: flags the visit and blocks the "clear" state."""
        self._area_person = max(count, 0)
        self._update_clear(now)
        visit = self._active
        if count > 0 and visit and not visit.person_present:
            visit.person_present = True
            return [Action("updated", visit.row())]
        return []

    def on_tray_pad_dog(self, count: int, now: datetime) -> list[Action]:
        """`frigate/tray_pad/dog`: seconds she really stood on the pad (kept as evidence)."""
        if count > 0:
            if self._on_pad_since is None:
                self._on_pad_since = now
        else:
            self._flush_on_pad(now)
            self._on_pad_since = None
        return []

    def on_event(self, event_id: str, now: datetime) -> list[Action]:
        """A Frigate dog event that entered pad_zone. Keep the full id (it has a suffix)."""
        visit = self._active
        if visit is None:
            if all(e != event_id for e, _ in self._pending_events):
                self._pending_events.append((event_id, now))
            return []
        if event_id in visit.event_ids:
            return []
        visit.event_ids.append(event_id)
        return [Action("updated", visit.row())]

    def on_frigate_available(self, online: bool, now: datetime) -> list[Action]:
        """`frigate/available`: a visit we cannot watch cannot be judged.

        Frigate publishes counts only on change, so after an outage the cached counts are stale:
        forget them on the way down and again on the way back.
        """
        if online == self._frigate_online:
            return []
        self._frigate_online = online
        self._area_dog = self._area_person = 0
        self._pad_dog = False
        self._clear_since = None
        if online or self._active is None:
            return []
        return self._fail(now, "frigate_offline")

    # --- time ---

    def tick(self, now: datetime) -> list[Action]:
        self._pending_events = [
            (e, t)
            for e, t in self._pending_events
            if (now - t).total_seconds() <= PENDING_EVENT_TTL_S
        ]
        if self._clear_since is None and self._area_dog == 0 and self._area_person == 0:
            self._clear_since = now
        visit = self._active
        actions: list[Action] = []
        if visit is None:
            return actions
        if visit.state == "OPEN" and (now - visit.open_since).total_seconds() >= self.max_open_s:
            visit.state, visit.ended_at = "CLOSING", now  # treat it as if she had left
            self._pad_dog, self._area_dog = False, 0
            actions.append(Action("updated", visit.row()))
        if visit.state == "CLOSING" and (now - visit.ended_at).total_seconds() >= self.merge_gap_s:
            visit.state, visit.capturing_since = "CAPTURING", now
            actions.append(Action("updated", visit.row()))
        if visit.state != "CAPTURING":
            return actions
        if self._clear_since and (now - self._clear_since).total_seconds() >= self.clear_s:
            self._flush_on_pad(now)
            visit.state = "ANALYSING"
            self._active = None
            actions.append(Action("capture_after", visit.row()))
        elif (now - visit.capturing_since).total_seconds() >= self.capture_timeout_s:
            actions.extend(self._fail(now, "area_not_clear"))
        return actions

    # --- internals ---

    def _new_visit(self, now: datetime) -> _Visit:
        stamp = now.astimezone(self._tz)
        visit_id = f"v_{stamp:%Y%m%d_%H%M%S}"
        while visit_id in self._recent_ids or (self.id_taken and self.id_taken(visit_id)):
            stamp += timedelta(seconds=1)
            visit_id = f"v_{stamp:%Y%m%d_%H%M%S}"
        self._recent_ids.append(visit_id)
        # counts come only on change: a person already there sends nothing new
        visit = _Visit(id=visit_id, started_at=now, open_since=now,
                       person_present=self._area_person > 0)  # fmt: skip
        visit.event_ids = [e for e, _ in self._pending_events]
        self._pending_events = []
        if self._on_pad_since is not None:
            self._on_pad_since = max(self._on_pad_since, now)
        return visit

    def _update_clear(self, now: datetime) -> None:
        if self._area_dog > 0 or self._area_person > 0:
            self._clear_since = None
        elif self._clear_since is None:
            self._clear_since = now

    def _flush_on_pad(self, now: datetime) -> None:
        visit, since = self._active, self._on_pad_since
        if visit and since is not None:
            visit.on_pad_s += max((now - max(since, visit.started_at)).total_seconds(), 0)
            self._on_pad_since = now

    def _fail(self, now: datetime, reason: str) -> list[Action]:
        visit = self._active
        self._active = None
        visit.state, visit.ended_at = "DECIDED", visit.ended_at or now
        return [Action("unknown", visit.row(), reason)]
