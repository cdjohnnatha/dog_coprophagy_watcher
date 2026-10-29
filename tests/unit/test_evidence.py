from pathlib import Path

import pytest

from ellie.evidence import Evidence


def test_visit_paths(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    d = ev.visit_dir("v_20260315_071530")
    assert d == tmp_path / "visits/2026/03/15/v_20260315_071530"
    assert ev.clip_path("v_20260315_071530") == d / "clip.mp4"


def test_presence_path(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    assert ev.presence_dir("s_20261231_235959") == tmp_path / "pad/2026/12/31/s_20261231_235959"


@pytest.mark.parametrize("bad", ["", "v_2026_1", "s_20260315_071530", "v_20260315_0715300", "x"])
def test_visit_malformed(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        Evidence(tmp_path).visit_dir(bad)


def test_presence_malformed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Evidence(tmp_path).presence_dir("v_20260315_071530")


def test_write_atomic_creates_dirs(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    path = ev.visit_dir("v_20260315_071530") / "before.jpg"
    ev.write(path, b"a")
    ev.write(path, b"b")
    assert path.read_bytes() == b"b"
    assert [p.name for p in path.parent.iterdir()] == ["before.jpg"]


def test_write_failure_leaves_no_temp(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    with pytest.raises(TypeError):
        ev.write(tmp_path / "x.jpg", "not bytes")  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []


def test_local_url(tmp_path: Path) -> None:
    ev = Evidence(tmp_path)
    p = ev.visit_dir("v_20260315_071530") / "after.jpg"
    assert ev.local_url(p) == "/media/local/ellie/visits/2026/03/15/v_20260315_071530/after.jpg"


def test_local_url_outside_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Evidence(tmp_path / "root").local_url(tmp_path / "other.jpg")
