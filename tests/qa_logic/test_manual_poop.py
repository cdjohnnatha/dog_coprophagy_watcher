"""`ellie/cmd/manual_poop` (app 3.5.1): a poop the system missed, added by the owner."""

import json

from qa_common import send, tick_for

from ellie.counting import LOCAL_TZ
from helpers import make_result


def state(rig):
    body = [b for t, b, _ in rig.published if t == "ellie/state"][-1]
    return json.loads(body) if isinstance(body, str | bytes) else body


async def test_a_manual_poop_is_counted_but_never_present_and_never_a_pee(make_rig):
    rig = make_rig()
    local = rig.clock().astimezone(LOCAL_TZ)
    await send(rig, "ellie/cmd/manual_poop", {"at": local.strftime("%Y-%m-%d %H:%M:%S")})
    await tick_for(rig, 1)
    st = state(rig)
    assert st["poop_today"] == 1 and st["pee_today"] == 0
    assert st["poop_present"]["on"] is False
    (visit,) = rig.store.visits_started_between("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")
    assert visit["source"] == "manual" and visit["final_poop"] == "poop"
    assert visit["final_outcome"] == "no_pee" and visit["poop_cleared_by"] == "owner"
    assert rig.store.labels_for(visit["id"]) == []  # no pee answer was given
    assert [lb["label"] for lb in rig.store.labels_for(visit["id"], "poop")] == ["poop"]


async def test_a_manual_poop_is_not_a_pee_in_the_accuracy_or_the_review_queue(make_rig):
    rig = make_rig()
    rig.analyser.script(make_result(pee=False))
    await send(rig, "ellie/cmd/manual_poop", {"at": "2026-10-29 10:00:00"})
    await tick_for(rig, 1)
    assert rig.store.review_count() == 0
    assert state(rig)["pee_today"] == 0


async def test_a_bad_time_or_place_is_ignored_or_defaulted(make_rig):
    rig = make_rig()
    for payload in ({"at": "banana"}, {"at": None}, {"at": "2999-01-01 10:00:00"}, {}):
        await send(rig, "ellie/cmd/manual_poop", payload)
    assert rig.store.visits_started_between("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z") == []
    await send(rig, "ellie/cmd/manual_poop", {"at": "2026-10-29 10:00:00", "where": []})
    (visit,) = rig.store.visits_started_between("2000-01-01T00:00:00Z", "2100-01-01T00:00:00Z")
    assert visit["poop_observed"] == "none"
