import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from PIL import Image

from ellie.frames import FrameError, ReferenceKeeper, crop, fetch_frame, fetch_snapshot

RED, GREEN, BLUE, WHITE = (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)


def make_jpeg() -> bytes:
    img = Image.new("RGB", (200, 100))
    img.paste(RED, (0, 0, 100, 50))
    img.paste(GREEN, (100, 0, 200, 50))
    img.paste(BLUE, (0, 50, 100, 100))
    img.paste(WHITE, (100, 50, 200, 100))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=95)
    return out.getvalue()


def close(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    return all(abs(x - y) < 40 for x, y in zip(a, b, strict=True))


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_frame_ok() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/frame.jpeg"
        assert req.url.params["src"] == "cam"
        return httpx.Response(200, content=b"jpg")

    async with client(handler) as c:
        assert await fetch_frame(c, "http://g:1984", "cam") == b"jpg"


@pytest.mark.parametrize("resp", [httpx.Response(404), httpx.Response(200, content=b"")])
async def test_fetch_frame_bad_response(resp: httpx.Response) -> None:
    async with client(lambda req: resp) as c:
        with pytest.raises(FrameError):
            await fetch_frame(c, "http://g:1984", "cam")


async def test_fetch_frame_network_error() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    async with client(handler) as c:
        with pytest.raises(FrameError):
            await fetch_frame(c, "http://g:1984", "cam")


async def test_fetch_snapshot_ok() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/events/e1/snapshot.jpg"
        assert req.url.params["crop"] == "0"
        return httpx.Response(200, content=b"snap")

    async with client(handler) as c:
        assert await fetch_snapshot(c, "http://f:5000", "e1") == b"snap"


async def test_fetch_snapshot_failures_return_none() -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    for handler in (
        lambda req: httpx.Response(404),
        lambda req: httpx.Response(200, content=b""),
        boom,
    ):
        async with client(handler) as c:
            assert await fetch_snapshot(c, "http://f:5000", "e1") is None


def test_crop_middle_columns() -> None:
    out = Image.open(io.BytesIO(crop(make_jpeg(), (0.25, 0.0, 0.75, 1.0))))
    assert out.size == (100, 100)
    assert close(out.getpixel((10, 10)), RED)
    assert close(out.getpixel((90, 10)), GREEN)
    assert close(out.getpixel((10, 90)), BLUE)
    assert close(out.getpixel((90, 90)), WHITE)


def test_crop_clamps_out_of_range() -> None:
    out = Image.open(io.BytesIO(crop(make_jpeg(), (-0.5, -1.0, 2.0, 0.5))))
    assert out.size == (200, 50)


@pytest.mark.parametrize("area", [(0.5, 0.0, 0.5, 1.0), (0.8, 0.0, 0.2, 1.0), (1.5, 0, 2, 1)])
def test_crop_empty_box(area: tuple[float, float, float, float]) -> None:
    with pytest.raises(FrameError):
        crop(make_jpeg(), area)


def test_crop_not_an_image() -> None:
    with pytest.raises(FrameError):
        crop(b"nope", (0, 0, 1, 1))


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def test_reference_keeper(tmp_path: Path) -> None:
    clock = FakeClock()
    keeper = ReferenceKeeper(tmp_path / "ref", clock)
    assert keeper.latest() is None
    assert keeper.age_s() is None

    keeper.update(b"one")
    t0 = clock.now
    clock.now += timedelta(seconds=75)
    assert keeper.latest() == (b"one", t0)
    assert keeper.age_s() == 75

    fresh = ReferenceKeeper(tmp_path / "ref", clock)
    assert fresh.latest() == (b"one", t0)
    assert fresh.age_s() == 75
    assert {p.name for p in (tmp_path / "ref").iterdir()} == {"latest.jpg", "latest.json"}


def test_reference_keeper_corrupt_sidecar(tmp_path: Path) -> None:
    clock = FakeClock()
    ReferenceKeeper(tmp_path, clock).update(b"one")
    (tmp_path / "latest.json").write_text("{not json")
    assert ReferenceKeeper(tmp_path, clock).latest() is None
    (tmp_path / "latest.json").write_text(json.dumps({"x": 1}))
    assert ReferenceKeeper(tmp_path, clock).age_s() is None
