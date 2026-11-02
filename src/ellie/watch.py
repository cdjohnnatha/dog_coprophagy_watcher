"""Safety net for pees that Frigate never saw (2026-10-31 08:59, 2026-11-01 18:05).

Every minute the service already takes a clear crop of the tray area. This module compares that
crop with the previous one: a pee on the pad is a step in the pad region that stays, while the
floor around it does not change. Light (sun, lamps, the infrared switch) changes both regions, so
the floor is the control.

LOG-ONLY for now (ADR-0013): it writes one CSV row per minute and never opens a visit or calls
Gemini, so the thresholds can be tuned on real days before anything depends on them.

Pure of I/O except the optional CSV append; the clock is injected.
"""

import io
import logging
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

SIZE = 48  # both frames are reduced to SIZE x SIZE grey pixels
# `tray_pad` (0.405,0.195 .. 0.645,0.570 of the frame) in coordinates of the analysis crop
# (0.244,0.041 .. 0.753,0.795). Re-derive both if either zone changes.
PAD_BOX = (0.32, 0.19, 0.79, 0.70)
CONTROL_MARGIN = 0.05  # floor this far outside the pad box (the tray rim) is not used
LIGHT_JUMP = 12.0  # mean grey levels in one minute: the whole scene changed (day/infrared)
STEP_MIN = 0.20  # normalised mean |difference| in the pad region
STEP_RATIO = 2.5  # the pad must change this many times more than the floor
CSV_HEADER = "at,pad_step,ctrl_step,brightness,light_jump,candidate,confirmed\n"


def _pixels(jpeg: bytes) -> list[float]:
    try:
        img = Image.open(io.BytesIO(jpeg))
        img.draft("L", (SIZE * 2, SIZE * 2))  # cheap JPEG decode at reduced size
        img = img.convert("L").resize((SIZE, SIZE))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"not a valid image: {exc!r}") from exc
    return [float(p) for p in img.getdata()]


def _regions() -> tuple[list[int], list[int]]:
    x1, y1, x2, y2 = PAD_BOX
    m = CONTROL_MARGIN
    pad, control = [], []
    for row in range(SIZE):
        for col in range(SIZE):
            fx, fy = (col + 0.5) / SIZE, (row + 0.5) / SIZE
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                pad.append(row * SIZE + col)
            elif not (x1 - m <= fx <= x2 + m and y1 - m <= fy <= y2 + m):
                control.append(row * SIZE + col)
    return pad, control


_PAD, _CONTROL = _regions()


def _normalise(px: list[float]) -> list[float]:
    """Remove overall brightness and contrast, so a lamp or the sun is not a difference."""
    mean = sum(px) / len(px)
    var = sum((p - mean) ** 2 for p in px) / len(px)
    std = var**0.5 or 1.0
    return [(p - mean) / std for p in px]


def _mean_abs(a: list[float], b: list[float], idx: list[int]) -> float:
    return sum(abs(a[i] - b[i]) for i in idx) / len(idx)


class PadWatch:
    def __init__(self, clock: Callable[[], datetime], log_path: Path | None = None) -> None:
        self._clock = clock
        self._log_path = log_path
        self._frames: deque[tuple[list[float], float]] = deque(maxlen=3)  # (normalised, mean)
        self._pending: dict | None = None  # a candidate waiting for the next frame to confirm it
        self.candidates_today = 0
        self._day = None

    def observe(self, crop_jpeg: bytes) -> dict | None:
        """Compare with the previous clear crop; returns the metrics row, or None on a bad frame."""
        try:
            px = _pixels(crop_jpeg)
        except ValueError as exc:
            log.warning("watch: %s", exc)
            return None
        now = self._clock()
        brightness = sum(px) / len(px)
        norm = _normalise(px)
        row = None
        if self._frames:
            prev, prev_brightness = self._frames[-1]
            pad_step = _mean_abs(norm, prev, _PAD)
            ctrl_step = _mean_abs(norm, prev, _CONTROL)
            jump = abs(brightness - prev_brightness) >= LIGHT_JUMP
            candidate = (
                not jump and pad_step >= STEP_MIN and pad_step >= STEP_RATIO * max(ctrl_step, 0.02)
            )
            confirmed = False
            if self._pending is not None:  # did last minute's step stay (the pad did not revert)?
                confirmed = pad_step < STEP_MIN / 2 and not jump
                self._pending["confirmed"] = confirmed
                self._write(self._pending)
                if confirmed:
                    self._count(now)
                    log.info("watch: a change at the pad that Frigate did not report (log only)")
                self._pending = None
            row = {
                "at": now.isoformat(timespec="seconds"),
                "pad_step": round(pad_step, 3),
                "ctrl_step": round(ctrl_step, 3),
                "brightness": round(brightness, 1),
                "light_jump": jump,
                "candidate": candidate,
                "confirmed": False,
            }
            if candidate:
                self._pending = row  # written next minute, once `confirmed` is known
            else:
                self._write(row)
        self._frames.append((norm, brightness))
        return row

    def _count(self, now: datetime) -> None:
        if self._day != now.date():
            self._day, self.candidates_today = now.date(), 0
        self.candidates_today += 1

    def _write(self, row: dict) -> None:
        if self._log_path is None:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            new = not self._log_path.exists()
            with self._log_path.open("a", encoding="utf-8") as f:
                if new:
                    f.write(CSV_HEADER)
                f.write(
                    f"{row['at']},{row['pad_step']},{row['ctrl_step']},{row['brightness']},"
                    f"{int(row['light_jump'])},{int(row['candidate'])},{int(row['confirmed'])}\n"
                )
        except OSError as exc:
            log.warning("watch: could not write %s: %s", self._log_path, exc)
