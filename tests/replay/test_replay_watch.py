"""The pad watch opens a visit for a change at the pad that Frigate never reported (ADR-0012)."""

import io
import json

import pytest
from PIL import Image
from test_replay_visit import build, last_state

from ellie.watch import PAD_BOX, PadWatch
from helpers import FakeClock


def frame(stained: bool = False, size: int = 200) -> bytes:
    img = Image.new("L", (size, size), 120)
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = 120 + ((x // 3) * 7 + (y // 3) * 13) % 6 * 8  # busy floor texture
    if stained:
        x1, y1, x2, y2 = PAD_BOX
        cx, cy = int((x1 + x2) / 2 * size), int((y1 + y2) / 2 * size)
        for y in range(cy - 25, cy + 25):
            for x in range(cx - 25, cx + 25):
                px[x, y] = 60
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rig(tmp_path, clock):
    shown = {"jpeg": frame()}

    async def fetch_frame():
        return shown["jpeg"]

    rig = build(tmp_path, clock, fetch_frame=fetch_frame, watch=PadWatch(clock))
    rig.shown = shown
    yield rig
    rig.store.close()


async def minute(rig, jpeg):
    """One minute passes and the minute task takes a clear frame that shows `jpeg`."""
    rig.shown["jpeg"] = jpeg
    for _ in range(2):  # the tracker needs a tick to start counting "area clear" time
        rig.clock.advance(30)
        await rig.service.tick()
    await rig.service.minute_tasks()
    await rig.service.wait_idle()


def auto_visits(rig):
    return [dict(r) for r in rig.store._db.execute("SELECT * FROM visits WHERE source = 'auto'")]


async def test_a_confirmed_change_opens_a_visit_that_gemini_decides(rig):
    clean, stained = frame(), frame(stained=True)
    await minute(rig, clean)  # baseline
    await minute(rig, stained)  # candidate
    assert auto_visits(rig) == []  # not before it is confirmed
    await minute(rig, stained)  # confirmed
    (visit,) = auto_visits(rig)
    assert visit["state"] == "DECIDED" and visit["final_outcome"] == "pee"
    assert visit["frigate_event_ids"] in ("[]", [])
    directory = rig.evidence.visit_dir(visit["id"])
    assert (directory / "before.jpg").read_bytes() == clean
    assert (directory / "after.jpg").read_bytes() == stained
    assert len(rig.analyser.calls) == 1 and rig.analyser.calls[0]["before_len"] == len(clean)
    assert last_state(rig)["pee_today"] == 1


async def test_a_change_already_explained_by_a_visit_is_not_analysed_again(rig):
    await rig.service.on_command("manual_pee", json.dumps({"at": rig.clock().isoformat()}).encode())
    await minute(rig, frame())
    await minute(rig, frame(stained=True))
    await minute(rig, frame(stained=True))
    assert auto_visits(rig) == [] and rig.analyser.calls == []


async def test_the_daily_cap_stops_a_runaway(rig, monkeypatch):
    monkeypatch.setattr("ellie.service.WATCH_DAILY_CAP", 1)
    await minute(rig, frame())
    await minute(rig, frame(stained=True))
    await minute(rig, frame(stained=True))  # first visit opened
    for _ in range(5):  # quiet minutes, so the next change is outside the explained window
        await minute(rig, frame(stained=True))
    await minute(rig, frame())  # the pad is clean again: a change
    await minute(rig, frame())  # confirmed, but today's cap is used
    assert len(auto_visits(rig)) == 1
