"""Frame capture, cropping and the clear-area reference frame."""

import io
import json
import logging
import os
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import httpx
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)


class FrameError(Exception):
    """A frame could not be fetched or cropped."""


async def fetch_frame(
    client: httpx.AsyncClient, go2rtc_url: str, camera: str, timeout: float = 10.0
) -> bytes:
    """Fetch a full-resolution JPEG frame from go2rtc."""
    url = f"{go2rtc_url}/api/frame.jpeg"
    try:
        resp = await client.get(url, params={"src": camera}, timeout=timeout)
    except httpx.HTTPError as exc:
        raise FrameError(f"go2rtc request failed: {exc!r}") from exc
    if resp.status_code != 200:
        raise FrameError(f"go2rtc returned HTTP {resp.status_code}")
    if not resp.content:
        raise FrameError("go2rtc returned an empty body")
    return resp.content


async def fetch_snapshot(
    client: httpx.AsyncClient, frigate_url: str, event_id: str, timeout: float = 10.0
) -> bytes | None:
    """Fetch the Frigate event snapshot; best effort, None on any failure."""
    url = f"{frigate_url}/api/events/{event_id}/snapshot.jpg"
    try:
        resp = await client.get(url, params={"crop": 0}, timeout=timeout)
    except httpx.HTTPError:
        return None
    if resp.status_code != 200 or not resp.content:
        return None
    return resp.content


def crop(jpeg: bytes, area: tuple[float, float, float, float], quality: int = 90) -> bytes:
    """Crop a JPEG to a normalised (x1, y1, x2, y2) area, without resizing."""
    try:
        img = Image.open(io.BytesIO(jpeg))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise FrameError(f"not a valid image: {exc!r}") from exc
    w, h = img.size
    x1, y1, x2, y2 = area
    box = (
        max(0, min(w, round(x1 * w))),
        max(0, min(h, round(y1 * h))),
        max(0, min(w, round(x2 * w))),
        max(0, min(h, round(y2 * h))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise FrameError(f"empty crop box {area}")
    out = io.BytesIO()
    img.convert("RGB").crop(box).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ReferenceKeeper:
    """Holds the latest clear-area crop and when it was taken, persisted on disk."""

    def __init__(self, directory: Path, clock: Callable[[], datetime]) -> None:
        self._dir = directory
        self._clock = clock
        self._cache: tuple[bytes, datetime] | None = None

    def update(self, jpeg_crop: bytes) -> None:
        """Store a new reference crop stamped with the current time."""
        now = self._clock()
        self._dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self._dir / "latest.jpg", jpeg_crop)
        _atomic_write(self._dir / "latest.json", json.dumps({"at": now.isoformat()}).encode())
        self._cache = (jpeg_crop, now)

    def latest(self) -> tuple[bytes, datetime] | None:
        """Return the reference and its time, from memory or disk, else None."""
        if self._cache is not None:
            return self._cache
        try:
            data = (self._dir / "latest.jpg").read_bytes()
            meta = json.loads((self._dir / "latest.json").read_text())
            at = datetime.fromisoformat(meta["at"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not data or at.tzinfo is None:
            return None
        self._cache = (data, at)
        return self._cache

    def age_s(self) -> int | None:
        """Seconds since the reference was stored, or None if there is none."""
        ref = self.latest()
        if ref is None:
            return None
        return int((self._clock() - ref[1]).total_seconds())
