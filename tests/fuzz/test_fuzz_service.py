"""Random sequences thrown at the whole service; rules that must hold once it settles.

Not a replacement for the scenario tests: it looks for states nobody thought of. A failing seed is
reproducible (`pytest -k "seed-17"`), so it can be turned into a normal test.
Set FUZZ_SEEDS=500 for a long run (default 40 keeps the normal suite fast).
"""

import asyncio
import json
import os
import random
from datetime import timedelta

import pytest

from ellie.analysers.base import AnalyserError
from ellie.counting import LOCAL_TZ
from ellie.frames import FrameError
from helpers import FakeClock, make_jpeg, make_pad_result, make_result
from helpers import build_rig as build

SEEDS = int(os.environ.get("FUZZ_SEEDS", "40"))
COUNT_TOPICS = [
    "frigate/pad_zone/dog",
    "frigate/tray_area/dog",
    "frigate/tray_area/person",
    "frigate/tray_pad/dog",
]
ERRORS = [
    "gemini unavailable: HTTP 503",
    "gemini request failed: HTTP 402",
    "gemini request failed: HTTP 401",
    "gemini returned an empty answer",
]


class ChaosAnalyser:
    id = "gemini:pee_v1"

    def __init__(self, rnd):
        self.rnd = rnd
        self.last_usage = {"in": 1000, "out": 80}

    def _maybe_fail(self):
        roll = self.rnd.random()
        if roll < 0.10:
            raise AnalyserError(self.rnd.choice(ERRORS))
        if roll < 0.13:
            raise RuntimeError("unexpected")

    async def analyse(self, before, after, during):
        self._maybe_fail()
        return make_result(
            pee=self.rnd.random() < 0.5,
            pee_confidence=self.rnd.choice([0.3, 0.79, 0.8, 0.95, 1.0]),
            poop=self.rnd.choice(["none", "none", "on_pad", "beside_tray", "unsure"]),
            pad_replaced=self.rnd.random() < 0.1,
            image_quality=self.rnd.choice(["ok"] * 8 + ["occluded", "too_dark"]),
        )

    async def analyse_pad(self, before, after):
        self._maybe_fail()
        return make_pad_result(
            pad_replaced=self.rnd.random() < 0.4,
            pad_replaced_confidence=self.rnd.choice([0.3, 0.8, 0.95]),
            image_quality=self.rnd.choice(["ok"] * 8 + ["occluded"]),
        )


def make_rig(tmp_path, clock, rnd):
    async def fetch_frame():
        if rnd.random() < 0.05:
            raise FrameError("camera glitch")
        return make_jpeg((rnd.randint(0, 255),) * 3)

    async def fetch_snapshot(event_id):
        return make_jpeg() if rnd.random() < 0.8 else None

    return build(
        tmp_path,
        clock,
        analyser=ChaosAnalyser(rnd),
        fetch_frame=fetch_frame,
        fetch_snapshot=fetch_snapshot,
    )


def ids(rig, table):
    return [r["id"] for r in rig.store._db.execute(f"SELECT id FROM {table}")]


async def random_command(rig, rnd, clock):
    visit_ids, presence_ids = ids(rig, "visits"), ids(rig, "presences")
    pick = lambda pool: (
        rnd.choice(pool)
        if pool and rnd.random() < 0.85
        else rnd.choice(["", "x", "../../etc/passwd", None, 7])
    )
    local = clock().astimezone(LOCAL_TZ)
    name = rnd.choice(
        [
            "label",
            "manual_pee",
            "reject_last",
            "reanalyse",
            "pad_label",
            "pad_changed",
            "pad_undo",
            "bogus",
        ]
    )
    data = {
        "label": {
            "visit_id": pick(visit_ids),
            "label": rnd.choice(["pee", "no_pee", "maybe", None, 3]),
            "channel": rnd.choice(["dashboard", "notification", "manual_add", None]),
        },
        "manual_pee": {
            "at": rnd.choice(
                [
                    (local - timedelta(hours=rnd.randint(0, 120))).strftime("%Y-%m-%d %H:%M:%S"),
                    (local + timedelta(hours=3)).isoformat(),
                    "banana",
                    None,
                    5,
                ]
            )
        },
        "reject_last": {},
        "reanalyse": {"visit_id": pick(visit_ids)},
        "pad_label": {
            "presence_id": pick(presence_ids),
            "label": rnd.choice(["replaced", "not_replaced", "x", None]),
        },
        "pad_changed": rnd.choice([{}, {"at": "2026-10-01T10:00:00"}, {"at": "nope"}]),
        "pad_undo": {},
        "bogus": {"a": 1},
    }[name]
    payload = (
        rnd.choice([json.dumps(data).encode(), b"", b"not json", b"[1,2]", b"null"])
        if rnd.random() < 0.15
        else json.dumps(data).encode()
    )
    await rig.service.on_message(f"ellie/cmd/{name}", payload)


async def run(rig_holder, tmp_path, rnd, clock, errors):
    async def deliver(topic, payload):
        try:
            await rig_holder[0].service.on_message(topic, payload)
        except Exception as exc:  # noqa: BLE001 - collecting what escapes the handler
            errors.append((topic, payload[:40], repr(exc)))

    async def tick(seconds):
        for _ in range(seconds):
            clock.advance(1)
            try:
                await rig_holder[0].service.tick()
                if int(clock().timestamp()) % 60 == 0:
                    await rig_holder[0].service.minute_tasks()
            except Exception as exc:  # noqa: BLE001
                errors.append(("tick", b"", repr(exc)))

    await tick(2)
    for _ in range(rnd.randint(30, 140)):
        op = rnd.choices(
            ["count", "event", "avail", "advance", "cmd", "set", "restart", "garbage"],
            weights=[40, 8, 2, 28, 10, 4, 2, 4],
        )[0]
        if op == "count":
            await deliver(rnd.choice(COUNT_TOPICS), str(rnd.choice([0, 0, 1, 1, 2])).encode())
        elif op == "event":
            event = {
                "after": {
                    "id": f"{rnd.randint(1, 9)}.{rnd.randint(1, 99)}-abc",
                    "camera": rnd.choice(["ellie", "ellie", "garage"]),
                    "label": rnd.choice(["dog", "dog", "person"]),
                    "entered_zones": rnd.choice(
                        [["pad_zone"], ["tray_area"], [], ["pad_zone", "tray_area"]]
                    ),
                }
            }
            await deliver("frigate/events", json.dumps(event).encode())
        elif op == "avail":
            await deliver("frigate/available", rnd.choice([b"online", b"offline"]))
        elif op == "advance":
            await tick(rnd.choice([1, 2, 5, 20, 45, 90]))
        elif op == "cmd":
            try:
                await random_command(rig_holder[0], rnd, clock)
            except Exception as exc:  # noqa: BLE001
                errors.append(("cmd", b"", repr(exc)))
        elif op == "set":
            key = rnd.choice(
                [
                    "confidence_threshold",
                    "merge_gap_s",
                    "validation_mode",
                    "deciding_analyser",
                    "pad_change_days",
                    "nope",
                ]
            )
            await deliver(
                f"ellie/set/{key}",
                rnd.choice(
                    [
                        b"0.5",
                        b"0.99",
                        b"15",
                        b"300",
                        b"ON",
                        b"OFF",
                        b"7",
                        b"1",
                        b"gemini:pee_v1",
                        b"abc",
                        b"",
                        b"-5",
                        b"1e9",
                    ]
                ),
            )
        elif op == "garbage":
            await deliver(
                rnd.choice(COUNT_TOPICS + ["frigate/events", "ellie/cmd/label", "ellie/set/x"]),
                bytes(rnd.getrandbits(8) for _ in range(rnd.randint(0, 12))),
            )
        else:  # restart: the old process dies, a new one opens the same database
            old = rig_holder[0]
            for task in list(old.service._tasks):
                task.cancel()
            await asyncio.gather(*list(old.service._tasks), return_exceptions=True)
            old.store.close()
            rig_holder[0] = make_rig(tmp_path, clock, rnd)
            await rig_holder[0].service.recover()

    # settle: nothing at the tray any more, time passes
    for topic in COUNT_TOPICS:
        await deliver(topic, b"0")
    await deliver("frigate/available", b"online")
    await tick(1500)
    await rig_holder[0].service.wait_idle()
    await tick(5)
    await rig_holder[0].service.wait_idle()


async def _async_state(rig):
    from ellie.state import build_state

    return build_state(
        store=rig.store,
        evidence=rig.evidence,
        active_visit=rig.service.tracker.active,
        cfg=rig.service.cfg,
        now=rig.service._clock(),
        usd_to_eur=0.9,
        pad=rig.service.pad_state(rig.service._clock()),
    )


@pytest.mark.parametrize("seed", range(SEEDS), ids=lambda s: f"seed-{s}")
async def test_random_sequences_settle_in_a_consistent_state(tmp_path, seed):
    rnd = random.Random(seed)
    clock = FakeClock()
    holder = [make_rig(tmp_path, clock, rnd)]
    holder[0].reference.update(make_jpeg())
    errors: list = []
    await run(holder, tmp_path, rnd, clock, errors)
    rig = holder[0]
    try:
        await _check(rig, errors, seed)
    finally:
        rig.store.close()


async def _check(rig, errors, seed):
    store, service = rig.store, rig.service
    problems = [f"escaped exception: {e}" for e in errors]
    visits = store.visits_started_between("2000", "2100")
    if store.open_visits():
        problems.append(
            f"visits stuck open: {[v['id'] + ':' + v['state'] for v in store.open_visits()]}"
        )
    if store.analysing_visits():
        problems.append(f"visits stuck analysing: {[v['id'] for v in store.analysing_visits()]}")
    if store.analysing_presences():
        problems.append(
            f"presences stuck: {[p['id'] + ':' + p['state'] for p in store.analysing_presences()]}"
        )
    if service.tracker.active is not None:
        problems.append(f"tracker still has an active visit: {service.tracker.active}")
    if service.presence.present:
        problems.append("presence tracker still present")
    if service._tasks:
        problems.append(f"tasks still pending: {len(service._tasks)}")
    for v in visits:
        labels = store.labels_for(v["id"])
        tag = f"visit {v['id']} auto={v['auto_outcome']} final={v['final_outcome']} state={v['state']} src={v['source']}"
        if v["state"] not in ("DECIDED", "LABELLED"):
            problems.append(f"{tag}: not in a terminal state")
        if v["source"] == "auto" and v["auto_outcome"] is None:
            problems.append(f"{tag}: auto visit without an outcome")
        if labels and v["final_outcome"] != labels[-1]["label"]:
            problems.append(f"{tag}: final differs from the latest label {labels[-1]['label']}")
        if (
            not labels
            and v["auto_outcome"] in ("pee", "no_pee")
            and v["final_outcome"] != v["auto_outcome"]
        ):
            problems.append(f"{tag}: confident outcome not copied to final")
        if not labels and v["auto_outcome"] == "unknown" and v["final_outcome"] != "unknown":
            problems.append(f"{tag}: unknown outcome with a final outcome and no label")
        if v["ended_at"] and v["ended_at"] < v["started_at"]:
            problems.append(f"{tag}: ended before it started")
    state = await _async_state(rig)
    try:
        json.dumps(state)
    except (TypeError, ValueError) as exc:
        problems.append(f"state is not serialisable: {exc}")
    from ellie.counting import Counting

    if state["pee_today"] != Counting(store).pees_today(service._clock()):
        problems.append("pee_today differs from Counting")
    if state["review_queue"]["count"] != len(store.review_queue(200)):
        problems.append("review queue count differs from the store")
    current = store.current_pad_change()
    if (state["pad"] is None) != (current is None):
        problems.append("pad state and the current pad change disagree")
    assert not problems, f"seed {seed}:\n  " + "\n  ".join(problems)
