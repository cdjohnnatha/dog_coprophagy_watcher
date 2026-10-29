"""Owner commands racing the analysis tasks."""

from qa_common import all_visits, dog_visit, send, tick_for

from ellie.analysers.base import AnalyserError
from helpers import FakeAnalyser, make_pad_result, make_result


async def _until_analysing(rig, stay=8):
    """Run a visit up to the moment the analysis task is in flight (analyser is slow)."""
    await rig.service.tick()
    await send(rig, "frigate/pad_zone/dog", 1)
    await send(rig, "frigate/tray_area/dog", 1)
    await tick_for(rig, stay, idle=False)
    await send(rig, "frigate/pad_zone/dog", 0)
    await send(rig, "frigate/tray_area/dog", 0)
    await tick_for(rig, 95, idle=False)
    (v,) = all_visits(rig)
    assert v["state"] == "ANALYSING", v["state"]
    return v["id"]


async def test_owner_label_given_while_analysing_is_overwritten_by_late_analysis(make_rig):
    rig = make_rig(analyser=FakeAnalyser(id="gemini:pee_v1", delay_s=0.3))
    rig.analyser.script(make_result(pee=True, pee_confidence=0.95))
    vid = await _until_analysing(rig)
    await send(rig, "ellie/cmd/label", {"visit_id": vid, "label": "no_pee"})
    await rig.service.wait_idle()
    v = rig.store.get_visit(vid)
    labels = rig.store.labels_for(vid)
    if labels:  # a label that was accepted must still decide the outcome
        assert v["final_outcome"] == labels[-1]["label"], "owner label overwritten by late analysis"
        assert v["state"] == "LABELLED" and v["needs_review"] is False


async def test_owner_label_given_while_analysing_is_destroyed_by_failed_analysis(make_rig):
    rig = make_rig(analyser=FakeAnalyser(id="gemini:pee_v1", delay_s=0.3))
    rig.analyser.script(AnalyserError("gemini unavailable: HTTP 503"))
    vid = await _until_analysing(rig)
    await send(rig, "ellie/cmd/label", {"visit_id": vid, "label": "pee"})
    await rig.service.wait_idle()
    v = rig.store.get_visit(vid)
    labels = rig.store.labels_for(vid)
    if labels:
        assert v["final_outcome"] == labels[-1]["label"], (
            f"label replaced by {v['final_outcome']!r}"
        )


async def test_double_reanalyse_of_a_failed_visit_calls_the_analyser_twice(make_rig):
    rig = make_rig()
    rig.analyser.script(AnalyserError("gemini unavailable: HTTP 503"))
    await dog_visit(rig)
    (v,) = all_visits(rig)
    assert v["final_outcome"] == "unknown"
    rig.analyser.delay_s = 0.1
    rig.analyser.script(make_result())
    before = len(rig.analyser.calls)
    await send(rig, "ellie/cmd/reanalyse", {"visit_id": v["id"]})
    await send(rig, "ellie/cmd/reanalyse", {"visit_id": v["id"]})
    await rig.service.wait_idle()
    assert len(rig.analyser.calls) - before == 1, (
        "two concurrent analyses of one visit (double Gemini cost)"
    )


async def test_reanalyse_while_first_analysis_is_in_flight_runs_a_second_gemini_call(make_rig):
    rig = make_rig(analyser=FakeAnalyser(id="gemini:pee_v1", delay_s=0.3))
    vid = await _until_analysing(rig)
    await send(rig, "ellie/cmd/reanalyse", {"visit_id": vid})
    await rig.service.wait_idle()
    assert len(rig.analyser.calls) == 1


async def test_pad_label_given_while_analysing_is_overwritten(make_rig):
    rig = make_rig(analyser=FakeAnalyser(id="gemini:pee_v1", delay_s=0.3))
    rig.analyser.script_pad(make_pad_result(pad_replaced=False, pad_replaced_confidence=0.95))
    await rig.service.tick()
    await send(rig, "frigate/tray_area/person", 1)
    await tick_for(rig, 15, idle=False)
    await send(rig, "frigate/tray_area/person", 0)
    await tick_for(rig, 40, idle=False)
    (row,) = [rig.store.get_presence(p) for p in _presence_ids(rig)]
    assert row["state"] == "ANALYSING"
    await send(rig, "ellie/cmd/pad_label", {"presence_id": row["id"], "label": "replaced"})
    await rig.service.wait_idle()
    p = rig.store.get_presence(row["id"])
    confirmed = [c for c in rig.store.pad_changes(10) if c["source"] == "confirmed"]
    if confirmed:  # an accepted answer must not be overwritten by the late analysis
        assert p["final_outcome"] == "replaced"


def _presence_ids(rig):
    rows = rig.store._db.execute("SELECT id FROM presences").fetchall()
    return [r["id"] for r in rows]
