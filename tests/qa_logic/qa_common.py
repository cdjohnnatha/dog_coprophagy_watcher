import asyncio
import json
from datetime import UTC, datetime

from helpers import build_rig, make_jpeg  # noqa: F401

EVENT = {
    "after": {
        "id": "1793288511.990085-yohln0",
        "camera": "ellie",
        "label": "dog",
        "entered_zones": ["pad_zone", "tray_area"],
    }
}


async def send(rig, topic, value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value)
    if isinstance(value, bytes):
        await rig.service.on_message(topic, value)
    else:
        await rig.service.on_message(topic, str(value).encode())


async def tick_for(rig, seconds, idle=True):
    for _ in range(int(seconds)):
        rig.clock.advance(1)
        await rig.service.tick()
        await asyncio.sleep(0)
    if idle:
        await rig.service.wait_idle()


async def dog_visit(rig, stay=8, settle=75, idle=True, event=True):
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/tray_area/dog", 1)
    if event:
        await send(rig, "frigate/events", EVENT)
    await tick_for(rig, stay)
    await send(rig, "frigate/pad_zone/dog", 0)
    await send(rig, "frigate/tray_area/dog", 0)
    await tick_for(rig, settle, idle=idle)


def all_visits(rig):
    return rig.store.visits_started_between("2000", "2100")


def at(h, m=0, s=0, day=29, month=10):
    return datetime(2026, month, day, h, m, s, tzinfo=UTC)
