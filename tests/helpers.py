"""Plain importable test helpers (no fixtures)."""

import asyncio
import io
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from PIL import Image, ImageDraw

from ellie.analysers.base import AnalysisResult, PadChangeResult


def _check_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(UTC)


class FakeClock:
    """Controllable timezone-aware UTC clock."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = _check_aware(start or datetime(2026, 10, 29, 12, 0, 0, tzinfo=UTC))

    def now(self) -> datetime:
        return self._now

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> datetime:
        self._now += timedelta(seconds=seconds)
        return self._now

    def set(self, dt: datetime) -> None:
        self._now = _check_aware(dt)


def make_result(**overrides: Any) -> AnalysisResult:
    """Valid AnalysisResult with optional field overrides."""
    data: dict[str, Any] = {
        "pee": True,
        "pee_confidence": 0.95,
        "poop": "none",
        "poop_confidence": 0.9,
        "pad_replaced": False,
        "image_quality": "ok",
        "reason": "mancha nova no centro do tapete",
    }
    return AnalysisResult(**{**data, **overrides})


def make_pad_result(**overrides: Any) -> PadChangeResult:
    """Valid PadChangeResult with optional field overrides."""
    data: dict[str, Any] = {
        "pad_replaced": False,
        "pad_replaced_confidence": 0.9,
        "poop_removed": False,
        "image_quality": "ok",
        "reason": "só recolheu o cocô",
    }
    return PadChangeResult(**{**data, **overrides})


class FakeAnalyser:
    """Scripted Analyser: returns or raises queued items, repeating the last."""

    def __init__(
        self,
        results: Sequence[AnalysisResult | Exception] | None = None,
        id: str = "fake:pee_v1",
        delay_s: float = 0.0,
    ) -> None:
        self.id = id
        self.delay_s = delay_s
        self._results = list(results or [])
        self.calls: list[dict[str, int | None]] = []
        self._pad_results: list[PadChangeResult | Exception] = []
        self.pad_calls: list[dict[str, int]] = []

    def script_pad(self, *results: PadChangeResult | Exception) -> None:
        """Queue answers for `analyse_pad` (the last one keeps repeating)."""
        self._pad_results = list(results)

    async def analyse_pad(self, before: bytes, after: bytes) -> PadChangeResult:
        self.pad_calls.append({"before_len": len(before), "after_len": len(after)})
        if not self._pad_results:
            item: PadChangeResult | Exception = make_pad_result()
        elif len(self._pad_results) > 1:
            item = self._pad_results.pop(0)
        else:
            item = self._pad_results[0]
        await asyncio.sleep(self.delay_s)
        if isinstance(item, Exception):
            raise item
        return item

    def script(self, *results: AnalysisResult | Exception) -> None:
        """Replace the queued results (the last one keeps repeating)."""
        self._results = list(results)

    async def analyse(self, before: bytes, after: bytes, during: bytes | None) -> AnalysisResult:
        self.calls.append(
            {
                "before_len": len(before),
                "after_len": len(after),
                "during_len": None if during is None else len(during),
            }
        )
        item: AnalysisResult | Exception
        if not self._results:
            item = make_result()
        elif len(self._results) > 1:
            item = self._results.pop(0)
        else:
            item = self._results[0]
        await asyncio.sleep(self.delay_s)
        if isinstance(item, Exception):
            raise item
        return item


def make_jpeg(
    color: tuple[int, int, int] = (200, 200, 200),
    size: tuple[int, int] = (320, 180),
    box: tuple[int, int, int, int, tuple[int, int, int]] | None = None,
) -> bytes:
    """JPEG bytes; `box=(x1, y1, x2, y2, color)` draws a filled rectangle."""
    img = Image.new("RGB", size, color)
    if box is not None:
        ImageDraw.Draw(img).rectangle(box[:4], fill=box[4])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_visit(**overrides: Any) -> dict:
    """Dict valid for Store.insert_visit."""
    base: dict[str, Any] = {
        "id": "v_20261029_101530",
        "source": "auto",
        "started_at": "2026-10-29T10:15:30Z",
        "ended_at": "2026-10-29T10:16:10Z",
        "entries": 1,
        "person_present": False,
        "frigate_event_ids": [],
        "before_age_s": 30,
        "state": "DECIDED",
        "auto_outcome": "pee",
        "final_outcome": "pee",
        "poop_observed": "none",
        "reason": "x",
        "needs_review": False,
        "deciding_analyser": "fake:pee_v1",
        "on_pad_s": 2.5,
    }
    return {**base, **overrides}


def build_rig(tmp_path, clock, analyser=None, **overrides):
    """A whole Service wired to fakes: real store/evidence on tmp_path, fake camera and publisher."""
    import json
    from types import SimpleNamespace

    from ellie.evidence import Evidence
    from ellie.frames import ReferenceKeeper
    from ellie.service import Service
    from ellie.store import Store
    from ellie.visits import VisitTracker

    store = Store(str(tmp_path / "ellie.db"))
    published: list[tuple[str, object, bool]] = []

    async def publish(topic, payload, retain):
        is_json = isinstance(payload, str) and payload.startswith(("{", "["))
        body = json.loads(payload) if is_json else payload
        published.append((topic, body, retain))

    async def fetch_frame():
        return make_jpeg((120, 120, 120))

    async def fetch_snapshot(event_id):
        return make_jpeg((90, 90, 90))

    analyser = analyser or FakeAnalyser(id="gemini:pee_v1")
    parts = {
        "store": store,
        "evidence": Evidence(tmp_path / "media"),
        "reference": ReferenceKeeper(tmp_path / "ref", clock),
        "tracker": VisitTracker(),
        "analysers": {analyser.id: analyser},
        "fetch_frame": fetch_frame,
        "fetch_snapshot": fetch_snapshot,
        "crop": lambda jpeg: jpeg,
        "publish": publish,
        "clock": clock,
        **overrides,
    }
    service = Service(**parts)
    return SimpleNamespace(service=service, store=store, published=published, analyser=analyser,
                           evidence=parts["evidence"], reference=parts["reference"], clock=clock)  # fmt: skip
