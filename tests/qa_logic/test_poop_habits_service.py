"""The poop habits in `ellie/state` follow counted poops and the clock (ADR-0015)."""

import json

from qa_common import send, tick_for

from ellie.counting import LOCAL_TZ


def habits(rig):
    body = [b for t, b, _ in rig.published if t == "ellie/state"][-1]
    return (json.loads(body) if isinstance(body, str | bytes) else body)["poop_habits"]


async def test_no_poop_yet_means_unknown_hours_not_zero(make_rig):
    rig = make_rig()
    await rig.service.publish_state()
    h = habits(rig)
    assert h["n"] == 0 and h["hours_since"] is None and h["ready"] is False


async def test_a_manual_poop_updates_the_habits_and_the_hours_grow_with_the_clock(make_rig):
    rig = make_rig()
    local = rig.clock().astimezone(LOCAL_TZ)
    await send(rig, "ellie/cmd/manual_poop", {"at": local.strftime("%Y-%m-%d %H:%M:%S")})
    await tick_for(rig, 1)
    first = habits(rig)
    assert first["n"] == 1 and first["last_at"] is not None
    assert first["hours_since"] == 0.0
    rig.clock.advance(3 * 3600)
    await rig.service.minute_tasks()  # the minute task republishes the state with no new poop
    assert habits(rig)["hours_since"] == 3.0


async def test_an_owner_answer_poop_counts_and_no_poop_removes_it(make_rig):
    rig = make_rig()
    local = rig.clock().astimezone(LOCAL_TZ)
    await send(rig, "ellie/cmd/manual_poop", {"at": local.strftime("%Y-%m-%d %H:%M:%S")})
    await tick_for(rig, 1)
    (visit,) = rig.store.visits_started_between("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")
    await send(
        rig, "ellie/cmd/label", {"visit_id": visit["id"], "kind": "poop", "label": "no_poop"}
    )
    await tick_for(rig, 1)
    h = habits(rig)
    assert h["n"] == 0 and h["hours_since"] is None
