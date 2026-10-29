"""Tests for the shared harness helpers."""

import io
from datetime import UTC, datetime

import pytest
from PIL import Image

from ellie.analysers.base import AnalyserError
from helpers import FakeAnalyser, FakeClock, make_jpeg, make_result, make_visit


def test_clock_advance_and_set():
    c = FakeClock()
    assert c() == c.now() == datetime(2026, 10, 29, 12, tzinfo=UTC)
    assert c.advance(90).isoformat() == "2026-10-29T12:01:30+00:00"
    c.set(datetime(2027, 1, 1, tzinfo=UTC))
    assert c.now().year == 2027


def test_clock_rejects_naive():
    with pytest.raises(ValueError):
        FakeClock(datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        FakeClock().set(datetime(2026, 1, 1))


async def test_analyser_sequence_repeat_and_calls():
    a, b = make_result(pee=True), make_result(pee=False)
    fake = FakeAnalyser([a, b])
    assert await fake.analyse(b"1", b"22", None) is a
    assert await fake.analyse(b"1", b"22", b"333") is b
    assert await fake.analyse(b"", b"", None) is b
    assert fake.calls[1] == {"before_len": 1, "after_len": 2, "during_len": 3}
    assert fake.calls[0]["during_len"] is None


async def test_analyser_default_and_exception():
    assert (await FakeAnalyser().analyse(b"", b"", None)).pee is True
    fake = FakeAnalyser([AnalyserError("boom")])
    with pytest.raises(AnalyserError):
        await fake.analyse(b"", b"", None)
    assert len(fake.calls) == 1


def test_make_jpeg_size_and_box():
    data = make_jpeg(size=(64, 48), box=(10, 10, 30, 30, (255, 0, 0)))
    img = Image.open(io.BytesIO(data))
    assert img.format == "JPEG"
    assert img.size == (64, 48)
    r, g, b = img.convert("RGB").getpixel((20, 20))
    assert r > 200 and g < 60 and b < 60


def test_make_visit_round_trip(tmp_store):
    visit = make_visit()
    tmp_store.insert_visit(visit)
    assert tmp_store.get_visit(visit["id"]) == visit
