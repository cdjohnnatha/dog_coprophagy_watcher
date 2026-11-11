"""Ellie seen as a "person": a small person box inside tray_area is her (ADR-0019).

Frigate's generic detector, looking from above, often calls her "person" when she squats to poop
(2026-10-30 to 2026-11-10: 16 events, 0.71-0.84). A person box she makes is at most ~36000 px at
the 896x504 detect size; real people are 38771 px and up (an arm at the image edge), usually over
70000 px. Zone topics only carry counts, so this reads `frigate/events` (box, area, zones) and
tells the service how many of the tray_area "persons" are really her.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

DETECT_W, DETECT_H = 896, 504  # Frigate detect size of camera ellie (Frigate config, not in git)
ELLIE_MAX_AREA = 36000  # px at the detect size; her largest "person" box was 35530
EDGE_PX = 8  # a box this close to the image edge is someone coming in, not her at the tray
BIG_PERSON_AREA = 38000  # a real person: small boxes are not trusted while one is around
BIG_PERSON_HOLD = timedelta(seconds=60)


class SmallPersonFilter:
    def __init__(self, camera: str) -> None:
        self.camera = camera
        self._max_area: dict[str, float] = {}  # person id -> largest area seen
        self._ellie: set[str] = set()  # person ids that are her, now in tray_area
        self._big: set[str] = set()
        self._big_seen: datetime | None = None

    @property
    def count(self) -> int:
        """Persons in tray_area that are really Ellie."""
        return len(self._ellie)

    def is_ellie(self, event_id: str) -> bool:
        return event_id in self._ellie

    def reset(self) -> None:
        """Frigate went away or came back: forget every object."""
        self._max_area.clear()
        self._ellie.clear()
        self._big.clear()

    def on_event(self, payload: bytes, now: datetime) -> bool:
        """`frigate/events`. Returns True when the number of persons that are her changed."""
        before = self.count
        try:
            message = json.loads(payload) or {}
            after = message.get("after") or {}
            if after.get("camera") != self.camera or after.get("label") != "person":
                return False
            event_id = after.get("id")
            if not isinstance(event_id, str):
                return False
            if message.get("type") == "end" or after.get("end_time"):
                self._forget(event_id, now)
                return self.count != before
            area = float(after.get("area") or 0)
            box = [float(v) for v in after.get("box") or ()]
            zones = set(after.get("current_zones") or ())
        except (ValueError, AttributeError, TypeError):  # not JSON, wrong shape
            return False
        area = max(area, self._max_area.get(event_id, 0.0))
        self._max_area[event_id] = area
        if area >= BIG_PERSON_AREA:
            self._big.add(event_id)
            self._big_seen = now
            self._ellie.discard(event_id)
            return self.count != before
        if event_id not in self._ellie and self._big_around(now):
            return False  # the owner is there: a small box may be their hand at the pad
        if area <= ELLIE_MAX_AREA and "tray_area" in zones and len(box) == 4 and not _at_edge(box):
            self._ellie.add(event_id)
        else:
            self._ellie.discard(event_id)
        return self.count != before

    def _big_around(self, now: datetime) -> bool:
        return bool(self._big) or (
            self._big_seen is not None and now - self._big_seen < BIG_PERSON_HOLD
        )

    def _forget(self, event_id: str, now: datetime) -> None:
        if event_id in self._big:
            self._big.discard(event_id)
            self._big_seen = now  # they just left: keep distrusting small boxes for a while
        self._ellie.discard(event_id)
        self._max_area.pop(event_id, None)


def _at_edge(box: list[float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= EDGE_PX or y1 <= EDGE_PX or x2 >= DETECT_W - EDGE_PX or y2 >= DETECT_H - EDGE_PX
