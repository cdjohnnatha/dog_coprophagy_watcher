"""QA of 3.3.0 (ADR-0011): person presences promoted to visits, tray_area as a visit trigger."""

import json

from qa_common import all_visits, send, tick_for

from helpers import make_pad_result, make_result


def presences(rig):
    rows = rig.store._db.execute("SELECT id FROM presences ORDER BY id").fetchall()
    return [rig.store.get_presence(r["id"]) for r in rows]


async def _person_at_tray(rig, stay=20):
    await rig.service.tick()
    await send(rig, "frigate/tray_area/person", 1)
    await tick_for(rig, stay, idle=False)
    await send(rig, "frigate/tray_area/person", 0)
    await tick_for(rig, 40)


async def test_owner_saying_pad_replaced_must_not_keep_the_promoted_pee(make_rig):
    """Pad answer unsure → owner is asked. A new pad explains any "new mark", so the pee question
    waits for the answer: "Troquei" asks nothing more and counts nothing."""
    rig = make_rig()
    rig.analyser.script_pad(make_pad_result(pad_replaced=True, pad_replaced_confidence=0.6))
    rig.analyser.script(make_result(pee=True, pee_confidence=0.95))
    await _person_at_tray(rig)
    (presence,) = presences(rig)
    assert all_visits(rig) == [] and rig.analyser.calls == []
    await send(rig, "ellie/cmd/pad_label", {"presence_id": presence["id"], "label": "replaced"})
    await tick_for(rig, 1)
    assert all_visits(rig) == [] and rig.analyser.calls == []


async def test_owner_saying_pad_not_replaced_runs_the_pee_question(make_rig):
    """Same unsure presence, answered "Não troquei": then the new mark was Ellie's."""
    rig = make_rig()
    rig.analyser.script_pad(make_pad_result(pad_replaced=True, pad_replaced_confidence=0.6))
    rig.analyser.script(make_result(pee=True, pee_confidence=0.95))
    await _person_at_tray(rig)
    (presence,) = presences(rig)
    await send(rig, "ellie/cmd/pad_label", {"presence_id": presence["id"], "label": "not_replaced"})
    await tick_for(rig, 1)
    (visit,) = all_visits(rig)
    assert visit["final_outcome"] == "pee" and len(rig.analyser.calls) == 1
    await send(rig, "ellie/cmd/pad_label", {"presence_id": presence["id"], "label": "not_replaced"})
    await tick_for(rig, 1)
    assert len(all_visits(rig)) == 1  # a double tap is ignored


async def test_a_failure_while_promoting_must_not_undo_the_pad_decision(make_rig):
    rig = make_rig()
    rig.analyser.script_pad(make_pad_result(pad_replaced=False, pad_replaced_confidence=0.95))
    rig.analyser.script(make_result(pee=True, pee_confidence=0.95))
    real_write = rig.evidence.write

    def write(path, data):
        if "/v_" in str(path):
            raise OSError(28, "No space left on device")
        return real_write(path, data)

    rig.evidence.write = write
    await _person_at_tray(rig)
    (presence,) = presences(rig)
    assert presence["final_outcome"] == "not_replaced", presence
    assert all(v["state"] != "ANALYSING" for v in all_visits(rig))


async def test_dog_following_the_owner_must_not_lose_the_pad_change(make_rig):
    """The owner swaps the pad; Ellie walks into tray_area (now a visit trigger), which drops
    the presence. The visit sees the swap, but it only records it when it knows a person was there."""
    rig = make_rig()
    rig.analyser.script(make_result(pad_replaced=True, pee=False, pee_confidence=0.9))
    await rig.service.tick()
    await send(rig, "frigate/tray_area/person", 1)
    await tick_for(rig, 15, idle=False)
    await send(rig, "frigate/tray_area/dog", 1)
    await tick_for(rig, 10, idle=False)
    await send(rig, "frigate/tray_area/dog", 0)
    await send(rig, "frigate/tray_area/person", 0)
    await tick_for(rig, 120)
    assert rig.store.current_pad_change() is not None, [json.dumps(p) for p in presences(rig)]
