import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from ellie.evidence import Evidence

VID = "v_20260315_071530"
NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)


def _visit(ev: Evidence, vid: str = VID, names: tuple[str, ...] = ("before.jpg", "after.jpg")):
    for n in names:
        ev.write(ev.visit_dir(vid) / n, n.encode())


def _lines(ev: Evidence) -> list[dict]:
    text = (ev.root / "dataset" / "labels.jsonl").read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines()]


def test_export_copies_existing(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    _visit(ev)
    files = ev.export_dataset({"id": VID, "poop_observed": True}, "pee", night=False, now=NOW)
    assert files == ["before.jpg", "after.jpg"]
    assert not (tmp_path / "dataset/pee" / VID / "during.jpg").exists()
    assert (tmp_path / "dataset/pee" / VID / "after.jpg").read_bytes() == b"after.jpg"
    assert _lines(ev) == [
        {
            "visit_id": VID,
            "label": "pee",
            "poop_observed": True,
            "night": False,
            "labelled_at": "2026-06-01T12:00:00Z",
            "files": files,
        }
    ]


def test_relabel_moves(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    _visit(ev)
    ev.export_dataset({"id": VID, "poop_observed": False}, "pee", now=NOW)
    ev.export_dataset({"id": VID, "poop_observed": False}, "no_pee", now=NOW)
    assert not (tmp_path / "dataset/pee" / VID).exists()
    assert (tmp_path / "dataset/no_pee" / VID / "before.jpg").exists()
    assert [r["label"] for r in _lines(ev)] == ["pee", "no_pee"]


def test_export_no_images_and_invalid(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    assert ev.export_dataset({"id": VID, "poop_observed": False}, "no_pee", now=NOW) == []
    assert _lines(ev)[0]["files"] == []
    with pytest.raises(ValueError):
        ev.export_dataset({"id": VID}, "maybe")


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_clip_success(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    async with _client(lambda r: httpx.Response(200, content=b"mp4")) as c:
        assert await ev.download_clip(c, "http://f", "e1", VID)
    assert ev.clip_path(VID).read_bytes() == b"mp4"
    assert [p.name for p in ev.clip_path(VID).parent.iterdir()] == ["clip.mp4"]


async def test_clip_retry_then_success(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, content=b"x")

    async with _client(handler) as c:
        assert await ev.download_clip(c, "http://f", "e1", VID)
    assert calls == ["http://f/api/events/e1/clip.mp4"] * 2


@pytest.mark.parametrize("kind", ["404", "empty", "exc"])
async def test_clip_failure(tmp_path: Path, kind: str) -> None:
    ev = Evidence(tmp_path)

    def handler(req: httpx.Request) -> httpx.Response:
        if kind == "exc":
            raise httpx.ReadTimeout("t")
        return httpx.Response(404) if kind == "404" else httpx.Response(200, content=b"")

    async with _client(handler) as c:
        assert not await ev.download_clip(c, "http://f", "e1", VID)
    assert not ev.clip_path(VID).exists()
    d = ev.clip_path(VID).parent
    assert not d.exists() or list(d.iterdir()) == []


def _mk(ev: Evidence, kind: str, ident: str, size: int = 10) -> Path:
    d = ev.visit_dir(ident) if kind == "visits" else ev.presence_dir(ident)
    ev.write(d / "f.bin", b"x" * size)
    return d


def test_purge_age_rule(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    old = _mk(ev, "visits", "v_20260101_000000")  # 151 days
    edge = _mk(ev, "visits", "v_20260303_000000")  # 90 days -> kept
    pad_old = _mk(ev, "pad", "s_20250101_000000")
    res = ev.purge(NOW)
    assert res == {"deleted_dirs": 2, "freed_bytes": 20}
    assert not old.exists() and not pad_old.exists() and edge.exists()
    assert not (tmp_path / "pad").exists() or not (tmp_path / "pad/2025").exists()
    assert not (tmp_path / "visits/2026/01").exists()
    assert ev.purge(NOW) == {"deleted_dirs": 0, "freed_bytes": 0}


def test_purge_89_vs_91(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    a = _mk(ev, "visits", "v_20260304_000000")  # 89 days
    b = _mk(ev, "visits", "v_20260302_000000")  # 91 days
    ev.purge(NOW)
    assert a.exists() and not b.exists()


def test_purge_size_rule(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    v1 = _mk(ev, "visits", "v_20260510_000000", 100)
    p1 = _mk(ev, "pad", "s_20260511_000000", 100)
    v2 = _mk(ev, "visits", "v_20260520_000000", 100)
    ev.write(tmp_path / "dataset/pee/v_20260101_000000/before.jpg", b"x" * 100)
    ev.write(tmp_path / "dataset/labels.jsonl", b"{}\n")
    ev.write(tmp_path / "reference/r.jpg", b"x" * 100)
    ev.write(tmp_path / "models/m.bin", b"x" * 100)
    total = 603
    res = ev.purge(NOW, max_bytes=total - 150)
    assert not v1.exists() and not p1.exists() and v2.exists()
    assert res["deleted_dirs"] == 2
    assert (tmp_path / "dataset/pee/v_20260101_000000").exists()
    res = ev.purge(NOW, max_bytes=0)
    assert not v2.exists()
    assert not (tmp_path / "dataset/pee/v_20260101_000000").exists()
    assert (tmp_path / "dataset/labels.jsonl").exists()
    assert (tmp_path / "reference/r.jpg").exists() and (tmp_path / "models/m.bin").exists()
    assert ev.purge(NOW, max_bytes=0)["deleted_dirs"] == 0


def test_purge_missing_root(tmp_path: Path) -> None:
    assert Evidence(tmp_path / "nope").purge(NOW) == {"deleted_dirs": 0, "freed_bytes": 0}
