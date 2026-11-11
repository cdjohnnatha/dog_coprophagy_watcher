"""A small "person" box inside tray_area is Ellie squatting (ADR-0019)."""

import json
from datetime import UTC, datetime, timedelta

from ellie.small_person import BIG_PERSON_HOLD, SmallPersonFilter

T0 = datetime(2026, 11, 10, 7, 37, tzinfo=UTC)
# 2026-11-10 07:37:14: her, squatting on the pad, "person" 0.79, 17516 px
HER_BOX = [367, 55, 520, 172]


def event(eid="1794.1-abc", label="person", box=None, area=17516, zones=("tray_area",),
          kind="update", camera="ellie", end=None) -> bytes:  # fmt: skip
    return json.dumps(
        {
            "type": kind,
            "after": {
                "id": eid,
                "camera": camera,
                "label": label,
                "box": box or HER_BOX,
                "area": area,
                "current_zones": list(zones),
                "end_time": end,
            },
        }
    ).encode()


def test_her_squatting_as_a_person_is_her():
    f = SmallPersonFilter("ellie")
    assert f.on_event(event(kind="new"), T0) is True
    assert f.count == 1 and f.is_ellie("1794.1-abc")
    assert f.on_event(event(kind="end", end=1794.9), T0) is True
    assert f.count == 0


def test_a_box_that_grows_to_person_size_is_a_person():
    f = SmallPersonFilter("ellie")
    f.on_event(event(), T0)
    f.on_event(event(area=60000), T0)
    assert f.count == 0
    f.on_event(event(area=17000), T0)  # the largest area seen counts
    assert f.count == 0


def test_a_small_box_while_a_person_is_around_is_not_trusted():
    """The owner's hand at the pad must stay a person (a pad change)."""
    f = SmallPersonFilter("ellie")
    f.on_event(event(eid="big", area=125000, box=[100, 50, 500, 480]), T0)
    f.on_event(event(eid="hand"), T0)
    assert f.count == 0
    f.on_event(event(eid="big", area=125000, kind="end", end=1.0), T0)
    f.on_event(event(eid="hand2"), T0 + BIG_PERSON_HOLD - timedelta(seconds=1))
    assert f.count == 0
    f.on_event(event(eid="later"), T0 + BIG_PERSON_HOLD + timedelta(seconds=1))
    assert f.count == 1


def test_she_stays_herself_when_the_owner_comes_in():
    f = SmallPersonFilter("ellie")
    f.on_event(event(eid="her"), T0)
    f.on_event(event(eid="big", area=125000, box=[100, 50, 500, 480]), T0)
    f.on_event(event(eid="her"), T0 + timedelta(seconds=2))
    assert f.is_ellie("her")


def test_edges_other_zones_cameras_and_labels_are_not_her():
    f = SmallPersonFilter("ellie")
    f.on_event(event(eid="arm", box=[0, 120, 130, 420], area=38771 - 5000), T0)  # 09/11 10:10
    f.on_event(event(eid="mat", zones=()), T0)  # on the mat below the tray
    f.on_event(event(eid="g", camera="garage"), T0)
    f.on_event(event(eid="d", label="dog"), T0)
    assert f.count == 0


def test_she_leaving_tray_area_and_garbage():
    f = SmallPersonFilter("ellie")
    f.on_event(event(), T0)
    assert f.on_event(event(zones=()), T0) is True and f.count == 0
    for junk in (b"", b"{", b"[]", b'{"after": 3}', json.dumps({"after": {"label": "person",
                 "camera": "ellie", "id": "x", "box": "ab", "area": "big"}}).encode()):  # fmt: skip
        assert f.on_event(junk, T0) is False
    f.on_event(event(eid="y"), T0)
    f.reset()
    assert f.count == 0


def test_a_box_between_her_size_and_a_person_is_neither():
    f = SmallPersonFilter("ellie")
    f.on_event(event(area=37000), T0)
    assert f.count == 0
