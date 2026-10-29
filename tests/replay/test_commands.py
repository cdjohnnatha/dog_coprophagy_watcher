"""Owner commands (labels, manual entries, undo, reanalyse) and what hangs off them."""

import json
from datetime import timedelta

import pytest
from test_replay_visit import EVENT, advance, build, last_state, one_visit, topics

from ellie.analysers.base import AnalyserError
from helpers import FakeClock, make_jpeg, make_result


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rig(tmp_path, clock):
    rig = build(tmp_path, clock)
    rig.reference.update(make_jpeg((200, 200, 200)))
    yield rig
    rig.store.close()


async def command(rig, name, **data):
    await rig.service.on_message(f"ellie/cmd/{name}", json.dumps(data).encode())


def visits(rig):
    return rig.store.visits_started_between("2026", "2028")


async def uncertain_visit(rig):
    rig.analyser.script(make_result(pee=True, pee_confidence=0.6))
    await one_visit(rig)
    return visits(rig)[-1]["id"]


async def test_labelling_an_uncertain_visit_counts_it_and_feeds_the_dataset(rig):
    visit_id = await uncertain_visit(rig)
    assert last_state(rig)["pee_today"] == 0
    await command(rig, "label", visit_id=visit_id, label="pee", channel="notification", by="iphone")
    visit = rig.store.get_visit(visit_id)
    assert visit["final_outcome"] == "pee" and visit["auto_outcome"] == "uncertain"
    assert visit["needs_review"] is False and visit["state"] == "LABELLED"
    assert last_state(rig)["pee_today"] == 1
    assert topics(rig, "ellie/review/resolved")[-1] == {"visit_id": visit_id}
    (label,) = rig.store.labels_for(visit_id)
    assert (label["label"], label["channel"], label["labelled_by"]) == (
        "pee",
        "notification",
        "iphone",
    )
    assert label["previous_final_outcome"] is None
    root = rig.evidence.root / "dataset"
    assert sorted(p.name for p in (root / "pee" / visit_id).iterdir()) == [
        "after.jpg",
        "before.jpg",
        "during.jpg",
    ]
    line = json.loads((root / "labels.jsonl").read_text().splitlines()[-1])
    assert line["visit_id"] == visit_id and line["label"] == "pee"


async def test_relabelling_moves_the_example_and_the_latest_label_wins(rig):
    visit_id = await uncertain_visit(rig)
    await command(rig, "label", visit_id=visit_id, label="pee", channel="dashboard")
    await command(rig, "label", visit_id=visit_id, label="no_pee", channel="dashboard")
    assert rig.store.get_visit(visit_id)["final_outcome"] == "no_pee"
    assert last_state(rig)["pee_today"] == 0
    assert len(rig.store.labels_for(visit_id)) == 2
    assert rig.store.labels_for(visit_id)[1]["previous_final_outcome"] == "pee"
    root = rig.evidence.root / "dataset"
    assert not (root / "pee" / visit_id).exists() and (root / "no_pee" / visit_id).exists()
    assert len((root / "labels.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize(
    "data",
    [
        {"visit_id": "nope", "label": "pee"},
        {"label": "pee"},
        {"visit_id": "x", "label": "maybe"},
        {"visit_id": "x", "label": "pee", "channel": "manual_add"},
    ],
)
async def test_bad_label_commands_are_ignored(rig, data):
    visit_id = await uncertain_visit(rig)
    data = {**data, "visit_id": visit_id if data.get("visit_id") == "x" else data.get("visit_id")}
    await command(rig, "label", **{k: v for k, v in data.items() if v is not None})
    assert rig.store.labels_for(visit_id) == []


async def test_a_manual_pee_for_yesterday_lands_on_yesterday_only(rig):
    await command(rig, "manual_pee", at="2026-10-28T21:00:00", by="owner")
    (visit,) = visits(rig)
    assert visit["source"] == "manual" and visit["id"] == "v_20261028_210000"
    assert visit["final_outcome"] == "pee" and visit["needs_review"] is False
    assert rig.store.labels_for(visit["id"])[0]["channel"] == "manual_add"
    state = last_state(rig)
    days = {d["date"]: d for d in state["daily_history"]["days"]}
    assert days["2026-10-28"]["pees"] == 1 and days["2026-10-28"]["corrected"] == 1
    assert state["pee_today"] == 0 and days["2026-10-29"]["pees"] == 0


async def test_manual_pee_rejects_the_future_and_bad_times_and_keeps_ids_unique(rig):
    await command(rig, "manual_pee", at="2026-10-30T09:00:00")
    await command(rig, "manual_pee", at="banana")
    assert visits(rig) == []
    await command(rig, "manual_pee", at="2026-10-29T10:00:00")
    await command(rig, "manual_pee", at="2026-10-29T10:00:00")
    ids = [v["id"] for v in visits(rig)]
    assert len(ids) == len(set(ids)) == 2 and last_state(rig)["pee_today"] == 2


async def test_manual_pee_with_a_timezone_offset(rig):
    await command(rig, "manual_pee", at="2026-10-29T09:00:00+01:00")
    assert visits(rig)[0]["started_at"] == "2026-10-29T08:00:00Z"


async def test_reject_last_undoes_todays_latest_pee_only(rig):
    await command(rig, "manual_pee", at="2026-10-28T21:00:00")  # yesterday: must stay
    await one_visit(rig)
    assert last_state(rig)["pee_today"] == 1
    await command(rig, "reject_last", by="owner")
    state = last_state(rig)
    assert state["pee_today"] == 0
    todays = next(v for v in visits(rig) if v["source"] == "auto")
    assert todays["final_outcome"] == "no_pee" and todays["auto_outcome"] == "pee"
    assert rig.store.labels_for(todays["id"])[0]["channel"] == "reject_last"
    days = {d["date"]: d["pees"] for d in state["daily_history"]["days"]}
    assert days["2026-10-28"] == 1
    await command(rig, "reject_last")  # nothing left today: ignored
    assert [v["final_outcome"] for v in visits(rig)].count("pee") == 1


async def test_reanalysing_a_failed_visit_decides_it(rig):
    rig.analyser.script(AnalyserError("gemini request failed: HTTP 402"))
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["auto_outcome"] == "unknown"
    rig.analyser.script(make_result())  # credit restored
    await command(rig, "reanalyse", visit_id=visit["id"])
    await rig.service.wait_idle()
    visit = rig.store.get_visit(visit["id"])
    assert visit["auto_outcome"] == "pee" and visit["final_outcome"] == "pee"
    assert last_state(rig)["pee_today"] == 1


async def test_reanalysing_a_decided_visit_only_records_the_new_answer(rig):
    await one_visit(rig)
    (visit,) = visits(rig)
    rig.analyser.script(make_result(pee=False, pee_confidence=0.9))
    await command(rig, "reanalyse", visit_id=visit["id"])
    await rig.service.wait_idle()
    same = rig.store.get_visit(visit["id"])
    assert same["final_outcome"] == "pee"  # untouched
    analyses = rig.store.analyses_for(visit["id"])
    assert [(a["role"], a["outcome"]) for a in analyses] == [
        ("deciding", "pee"),
        ("shadow", "no_pee"),
    ]


async def test_reanalyse_without_images_or_with_an_unknown_id_is_ignored(rig):
    await command(rig, "reanalyse", visit_id="v_20200101_000000")
    await command(rig, "reanalyse")
    await rig.service.wait_idle()
    assert rig.analyser.calls == []


async def test_an_unanswered_uncertain_visit_is_not_counted_and_stays_in_review(rig, clock):
    visit_id = await uncertain_visit(rig)
    clock.advance(24 * 3600)  # next local day
    await rig.service.minute_tasks()
    visit = rig.store.get_visit(visit_id)
    assert visit["final_outcome"] == "unknown" and visit["needs_review"] is True
    assert visit_id in [v["id"] for v in rig.store.review_queue()]
    assert last_state(rig)["pee_today"] == 0
    await command(
        rig, "label", visit_id=visit_id, label="pee", channel="dashboard"
    )  # still answerable
    assert rig.store.get_visit(visit_id)["final_outcome"] == "pee"


async def test_the_poop_clip_is_requested_only_when_poop_was_seen(tmp_path, clock):
    requested = []

    async def download_clip(event_id, visit_id):
        requested.append((event_id, visit_id))
        return True

    rig = build(tmp_path, clock, download_clip=download_clip)
    rig.reference.update(make_jpeg())
    rig.analyser.script(make_result(poop="beside_tray", poop_confidence=0.9))
    await one_visit(rig)
    (visit,) = visits(rig)
    assert visit["poop_observed"] == "beside_tray"
    assert requested == [(EVENT["after"]["id"], visit["id"])]
    rig.analyser.script(make_result(poop="none"))
    await one_visit(rig)
    assert len(requested) == 1


async def test_retention_runs_once_a_day_after_0330_local(rig, clock):
    calls = []
    rig.evidence.purge = lambda now: calls.append(now) or {"deleted_dirs": 0, "freed_bytes": 0}
    clock.set(clock().replace(hour=3, minute=29))
    await rig.service.minute_tasks()
    assert calls == []
    clock.advance(120)
    await rig.service.minute_tasks()
    await rig.service.minute_tasks()
    assert len(calls) == 1
    clock.advance(24 * 3600)
    await rig.service.minute_tasks()
    assert len(calls) == 2


async def test_accuracy_and_corrections_show_up_in_the_state(rig):
    await one_visit(rig)  # confident pee, will be confirmed
    await advance(rig, 10)
    await one_visit(rig)  # confident pee, will be rejected
    first, second = (v["id"] for v in visits(rig))
    await command(rig, "label", visit_id=first, label="pee", channel="dashboard")
    await command(rig, "label", visit_id=second, label="no_pee", channel="dashboard")
    accuracy = last_state(rig)["accuracy"]
    assert accuracy["pct"] == 50.0
    assert accuracy["analysers"]["gemini:pee_v1"] == {"labelled": 2, "agree": 1, "accuracy": 50.0}
    assert accuracy["period"]["confirmations"] == 1 and accuracy["period"]["rejections"] == 1
    today = last_state(rig)["daily_history"]["days"][-1]
    assert today["visits"] == 2 and today["corrected"] == 1


async def test_a_visit_started_by_a_person_only_never_becomes_a_pee(rig):
    # a person at the tray (pad change) is not a dog visit: no visit row, no analysis
    await rig.service.tick()
    await rig.service.on_message("frigate/tray_area/person", b"1")
    await advance(rig, 120)
    await rig.service.on_message("frigate/tray_area/person", b"0")
    await advance(rig, 120)
    assert visits(rig) == [] and rig.analyser.calls == []
    assert rig.clock() - timedelta(minutes=5) < rig.clock()


async def test_photos_go_to_the_image_entities_only_when_they_change(rig):
    visit_id = await uncertain_visit(rig)
    directory = rig.evidence.visit_dir(visit_id)
    sent = {t: (b, r) for t, b, r in rig.published if t.startswith("ellie/image/")}
    assert sent["ellie/image/review_before"] == ((directory / "before.jpg").read_bytes(), True)
    assert sent["ellie/image/review_after"] == ((directory / "after.jpg").read_bytes(), True)
    placeholder = sent["ellie/image/last_pee"][0]  # nothing counted yet: a flat placeholder
    assert placeholder != (directory / "after.jpg").read_bytes() and placeholder[:2] == b"\xff\xd8"

    count = len([t for t, *_ in rig.published if t.startswith("ellie/image/")])
    await rig.service.minute_tasks()
    await rig.service.minute_tasks()
    assert len([t for t, *_ in rig.published if t.startswith("ellie/image/")]) == count

    await command(rig, "label", visit_id=visit_id, label="pee", channel="dashboard")
    assert topics(rig, "ellie/image/last_pee")[-1] == (directory / "after.jpg").read_bytes()


async def test_the_before_photo_is_older_than_the_moment_someone_entered(tmp_path, clock):
    rig = build(tmp_path, clock)
    old, fresh = make_jpeg((10, 10, 10)), make_jpeg((250, 250, 250))
    rig.reference.update(old)
    clock.advance(55)
    rig.reference.update(fresh)  # taken 3 s before the dog is counted: may show her entering
    clock.advance(3)
    await rig.service.on_message("frigate/pad_zone/dog", b"1")
    (visit,) = visits(rig)
    assert (rig.evidence.visit_dir(visit["id"]) / "before.jpg").read_bytes() == old
    assert visit["before_age_s"] == 58


async def test_a_manual_pee_in_the_same_second_does_not_collide_with_a_visit(rig, clock):
    stamp = clock().astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    await command(rig, "manual_pee", at=stamp)  # takes the id of this very second
    await rig.service.on_message("frigate/pad_zone/dog", b"1")  # the dog enters in the same second
    ids = [v["id"] for v in visits(rig)]
    assert len(ids) == len(set(ids)) == 2
    assert rig.service.tracker.active["id"] in ids
