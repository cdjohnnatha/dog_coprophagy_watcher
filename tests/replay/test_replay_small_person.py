"""Ellie seen as a small "person" at the tray opens a visit, not a presence (ADR-0019)."""

import json

import pytest
from test_replay_visit import advance, build, send

from helpers import FakeClock, make_jpeg

HER = {"id": "1794300000.1-abcdef", "camera": "ellie", "label": "person",
       "box": [367, 55, 520, 172], "area": 17516, "current_zones": ["tray_area"],
       "entered_zones": ["tray_area"]}  # fmt: skip
OWNER = {**HER, "id": "1794300100.2-ghijkl", "box": [120, 40, 560, 470], "area": 125000}


@pytest.fixture
def rig(tmp_path):
    rig = build(tmp_path, FakeClock())
    rig.reference.update(make_jpeg((200, 200, 200)))
    yield rig
    rig.store.close()


async def event(rig, after, kind="update"):
    await rig.service.on_message(
        "frigate/events", json.dumps({"type": kind, "after": after}).encode()
    )


async def test_her_squatting_as_a_person_opens_a_visit(rig):
    """2026-11-10 07:37: Frigate saw only a "person" at the tray; it was her pooping."""
    await rig.service.tick()
    await send(rig, "frigate/tray_area/person", 1)  # the count may come before the event
    await event(rig, HER, "new")
    await advance(rig, 20)
    assert rig.service.tracker.active is not None
    assert not rig.service.presence.present
    await send(rig, "frigate/tray_area/person", 0)
    await event(rig, {**HER, "current_zones": [], "end_time": 1794300030.0}, "end")
    await advance(rig, 100)
    (visit,) = rig.store.visits_started_between("2026", "2028")
    assert visit["person_present"] is False
    assert HER["id"] in visit["frigate_event_ids"]
    assert visit["auto_outcome"] is not None


async def test_a_real_person_at_the_tray_is_still_a_presence(rig):
    await rig.service.tick()
    await event(rig, OWNER, "new")
    await send(rig, "frigate/tray_area/person", 1)
    await advance(rig, 20)
    assert rig.service.tracker.active is None
    assert rig.service.presence.present
