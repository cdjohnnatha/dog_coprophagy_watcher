"""The real walk-through of 2026-10-29 replayed through the whole service.

Fixture: Frigate's zone counts and (trimmed) events recorded while the owner stood at the tray
and then Ellie walked through with treats. It carries every flap the real camera produced.
Timeline (UTC = Lisbon that day): a person at the tray 15:40:37-15:41:01, again from 15:41:38;
Ellie enters pad_zone at 15:41:52 (2 entries inside 4 s) with the person still around.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest
from test_replay_visit import build, last_state

from helpers import FakeClock, make_jpeg

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mqtt_walkthrough_2026-10-29.tsv"


def messages():
    for line in FIXTURE.read_text().splitlines():
        stamp, topic, payload = line.split("\t", 2)
        yield datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S%z"), topic, payload.encode()


@pytest.fixture
async def replayed(tmp_path):
    first = next(messages())[0]
    clock = FakeClock(first.replace(second=first.second - 5 if first.second >= 5 else 0))
    rig = build(tmp_path, clock)
    rig.reference.update(make_jpeg((200, 200, 200)))
    await rig.service.tick()
    for moment, topic, payload in messages():
        while clock() < moment:  # one tick per second, like the real service
            clock.advance(1)
            await rig.service.tick()
        await rig.service.on_message(topic, payload)
    for _ in range(200):
        clock.advance(1)
        await rig.service.tick()
    await rig.service.wait_idle()
    yield rig
    rig.store.close()


def visits(rig):
    return rig.store.visits_started_between("2026", "2028")


def presences(rig):
    rows = rig.store._db.execute("SELECT id FROM presences ORDER BY id").fetchall()
    return [rig.store.get_presence(r["id"]) for r in rows]


async def test_ellie_is_one_visit_despite_the_flapping_zone(replayed):
    (visit,) = visits(replayed)
    assert visit["started_at"] == "2026-10-29T15:41:52Z"
    assert visit["entries"] == 2  # pad_zone/dog went 1, 0, 1, 0 inside four seconds
    assert visit["person_present"] is True  # the owner was beside the tray
    assert visit["frigate_event_ids"] == [
        "1793288511.990085-yohln0"
    ]  # the person events are not hers
    assert (
        visit["state"] == "DECIDED" and visit["auto_outcome"] == "pee"
    )  # the fake analyser says so


async def test_the_after_photo_waits_for_the_area_to_be_really_clear(replayed):
    # last person flap 15:42:04 + 30 s clear = 15:42:34; the merge gap after her last exit
    # (15:41:56 + 60 s = 15:42:56) is what finally allows the capture
    assert len(replayed.analyser.calls) == 1
    (visit,) = visits(replayed)
    assert (replayed.evidence.visit_dir(visit["id"]) / "after.jpg").exists()


async def test_the_person_before_the_dog_is_one_presence_and_the_one_during_the_visit_is_dropped(
    replayed,
):
    first, second = presences(replayed)
    assert (
        first["started_at"] == "2026-10-29T15:40:37Z"
        and first["ended_at"] == "2026-10-29T15:41:01Z"
    )
    assert first["auto_outcome"] == "not_replaced"  # analysed once, the fake says "same pad"
    assert (
        second["state"] == "DECIDED" and second["reason"] == "dog_visit"
    )  # the visit owns the area
    assert len(replayed.analyser.pad_calls) == 1


async def test_nothing_in_the_walk_through_counts_as_a_pad_change(replayed):
    assert replayed.store.pad_changes() == []
    assert last_state(replayed)["pad"] is None


async def test_the_day_ends_with_one_counted_visit_and_no_alerts(replayed):
    state = last_state(replayed)
    assert state["pee_today"] == 1 and state["visits_today"] == 1
    assert [t for t, *_ in replayed.published if t in ("ellie/alert", "ellie/review/request")] == []
    json.dumps(state)
