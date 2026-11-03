"""Safety net for pees that Frigate never saw (2026-10-31 08:59, 2026-11-01 18:05, 2026-11-03 14:46).

Every minute the service already takes a clear crop of the tray area. This module compares that
crop with the previous one: a pee on the pad is a step in the pad region that stays, while the
floor around it does not change. Light (sun, lamps, the infrared switch) changes both regions, so
the floor is the control.

A step becomes a CANDIDATE. It is CONFIRMED when a later crop still shows the pad as it was in the
candidate frame (not as the previous minute: a lost frame or someone at the tray must not break
it), within `PENDING_TTL_S`. The confirmed candidate is returned to the service, which opens a
visit from it (ADR-0012). Pixel difference is only a trigger: Gemini decides.

Pure of I/O except the optional CSV append; the clock is injected.
"""

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

SIZE = 48  # both frames are reduced to SIZE x SIZE grey pixels
# `tray_pad` (0.405,0.195 .. 0.645,0.570 of the frame) in coordinates of the analysis crop
# (0.244,0.041 .. 0.753,0.795). Re-derive both if either zone changes.
PAD_BOX = (0.32, 0.19, 0.79, 0.70)
CONTROL_MARGIN = 0.05  # floor this far outside the pad box (the tray rim) is not used
LIGHT_JUMP = 12.0  # mean grey levels in one minute: the whole scene changed (day/infrared)
# Calibrated on the 19 labelled visits (before/after) plus the 2026-11-03 14:46 frames: pees on an
# already stained pad step 0.12-0.21 at the pad and 0.03-0.07 at the floor, quiet minutes are about
# 0.03 on both, lighting events move the floor by 0.15 or more. Few data: re-check with watch.csv.
STEP_MIN = 0.10  # normalised mean |difference| in the pad region
STEP_RATIO = 2.0  # the pad must change this many times more than the floor
CTRL_MAX = 0.10  # more than this at the floor is light, not a pee
CONFIRM_MAX = 0.06  # a later frame confirms when the pad is within this of the candidate frame
PENDING_TTL_S = 300  # a candidate waits this long for a frame that confirms it
CSV_HEADER = "at,pad_step,ctrl_step,brightness,light_jump,candidate,confirmed\n"


@dataclass
class Candidate:
    """A step at the pad, with the two frames Gemini will compare."""

    at: datetime  # when the frame with the change was taken
    before_at: datetime  # when the previous clear frame was taken
    before_jpeg: bytes
    after_jpeg: bytes
    pad_step: float
    ctrl_step: float
    norm: list[float]
    brightness: float
    seen: int = 0  # later frames that did not confirm it yet


def _pixels(jpeg: bytes) -> list[float]:
    try:
        img = Image.open(io.BytesIO(jpeg))
        img.draft("L", (SIZE * 2, SIZE * 2))  # cheap JPEG decode at reduced size
        img = img.convert("L").resize((SIZE, SIZE))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"not a valid image: {exc!r}") from exc
    flat = img.get_flattened_data() if hasattr(img, "get_flattened_data") else img.getdata()
    return [float(p) for p in flat]


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
    """Remove brightness and contrast measured on the FLOOR, so a lamp or the sun is not a
    difference. The floor is used (not the whole frame) because a pee or a new pad changes the
    pad region a lot, and that must not make the floor look different too."""
    floor = [px[i] for i in _CONTROL]
    mean = sum(floor) / len(floor)
    var = sum((p - mean) ** 2 for p in floor) / len(floor)
    std = var**0.5 or 1.0
    return [(p - mean) / std for p in px]


def _mean_abs(a: list[float], b: list[float], idx: list[int]) -> float:
    return sum(abs(a[i] - b[i]) for i in idx) / len(idx)


class PadWatch:
    def __init__(self, clock: Callable[[], datetime], log_path: Path | None = None) -> None:
        self._clock = clock
        self._log_path = log_path
        # last clear frame: (normalised, mean grey level, jpeg, time)
        self._prev: tuple[list[float], float, bytes, datetime] | None = None
        self._pending: Candidate | None = None
        self._pending_row: dict | None = None
        self.last_row: dict | None = None  # the metrics of the latest observation (for tests/logs)
        self.confirmed_total = 0

    def rebase(self, crop_jpeg: bytes) -> None:
        """A visit or a presence was handled: its after frame is the new baseline.

        What changed at the pad is already explained, so a pending candidate is dropped and the
        next minute is compared with this frame, not with the one before the visit.
        """
        try:
            px = _pixels(crop_jpeg)
        except ValueError as exc:
            log.warning("watch: %s", exc)
            return
        if self._pending is not None:
            log.info("watch: candidate dropped, a visit explains it")
            self._pending = self._pending_row = None
        self._prev = (_normalise(px), sum(px) / len(px), crop_jpeg, self._clock())

    def observe(self, crop_jpeg: bytes) -> Candidate | None:
        """Compare with what came before; returns a CONFIRMED candidate, else None."""
        try:
            px = _pixels(crop_jpeg)
        except ValueError as exc:
            log.warning("watch: %s", exc)
            return None
        now = self._clock()
        brightness = sum(px) / len(px)
        norm = _normalise(px)
        confirmed: Candidate | None = None
        if self._pending is not None:
            confirmed = self._resolve_pending(norm, brightness, now)
            self._prev = (norm, brightness, crop_jpeg, now)
            return confirmed
        if self._prev is not None:
            prev_norm, prev_brightness, prev_jpeg, prev_at = self._prev
            pad_step = _mean_abs(norm, prev_norm, _PAD)
            ctrl_step = _mean_abs(norm, prev_norm, _CONTROL)
            jump = abs(brightness - prev_brightness) >= LIGHT_JUMP
            candidate = (
                not jump
                and pad_step >= STEP_MIN
                and ctrl_step <= CTRL_MAX
                and pad_step >= STEP_RATIO * max(ctrl_step, 0.02)
            )
            row = {
                "at": now.isoformat(timespec="seconds"),
                "pad_step": round(pad_step, 3),
                "ctrl_step": round(ctrl_step, 3),
                "brightness": round(brightness, 1),
                "light_jump": jump,
                "candidate": candidate,
                "confirmed": False,
            }
            self.last_row = row
            if candidate:
                self._pending_row = row  # written once it is confirmed or given up
                self._pending = Candidate(
                    at=now,
                    before_at=prev_at,
                    before_jpeg=prev_jpeg,
                    after_jpeg=crop_jpeg,
                    pad_step=pad_step,
                    ctrl_step=ctrl_step,
                    norm=norm,
                    brightness=brightness,
                )
                log.info("watch: candidate at the pad (pad %.2f, floor %.2f)", pad_step, ctrl_step)
            else:
                self._write(row)
        self._prev = (norm, brightness, crop_jpeg, now)
        return None

    def _resolve_pending(
        self, norm: list[float], brightness: float, now: datetime
    ) -> Candidate | None:
        pending, row = self._pending, self._pending_row
        assert pending is not None and row is not None
        pending.seen += 1
        same = (
            _mean_abs(norm, pending.norm, _PAD) < CONFIRM_MAX
            and abs(brightness - pending.brightness) < LIGHT_JUMP
        )
        expired = now - pending.at > timedelta(seconds=PENDING_TTL_S)
        if same:
            row["confirmed"] = True
            self._write(row)
            self._pending = self._pending_row = None
            self.confirmed_total += 1
            log.info("watch: change at the pad confirmed, Frigate did not report it")
            return pending
        if expired:
            self._write(row)  # given up: confirmed stays False
            self._pending = self._pending_row = None
            log.info("watch: candidate not confirmed, dropped")
        return None

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
