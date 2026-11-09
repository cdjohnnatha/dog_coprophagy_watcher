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
# A later frame whose pad still differs this much from the frame BEFORE the candidate confirms it
# too (the pee stays, she does not). Pees on a stained pad step 0.12-0.21, quiet minutes ~0.03.
LASTING_MIN = 0.10
# Poop on the floor or the tray rim (2026-11-03 10:00, 11-04 07:28, 11-05 06:13): a small DARK blob
# appears outside the pad. It is a few pixels wide, so the mean over the floor hides it; the peak of
# the darkening over a 2x2 block of cells does not. Calibrated on the stored before/after pairs:
# real poops 1.19 and 1.67 with a quiet floor, the worst light event 1.15 with a noisy floor.
FLOOR_PEAK_MIN = 1.0
FLOOR_CTRL_MAX = 0.06  # mean darkening of the whole floor: more is light or a person, not poop
FLOOR_RATIO = 20.0  # the blob must stand out this many times above the mean darkening
# A counted poop stays "present" until the watch sees the blob gone (ADR-0014): the darkening of its
# block against the frame before the visit must stay under POOP_GONE_RATIO of the original peak for
# POOP_GONE_FRAMES clear frames in a row.
POOP_TRACK_MIN_PEAK = 0.6  # a weaker blob cannot be told from noise: not tracked
POOP_GONE_RATIO = 0.35
POOP_GONE_FRAMES = 2
CSV_HEADER = "at,pad_step,ctrl_step,brightness,light_jump,candidate,confirmed,floor_peak,kind\n"


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
    kind: str = "pad"  # "pad": a pee-like step at the pad; "floor": a dark blob outside it
    floor_peak: float = 0.0
    before_norm: list[float] | None = None  # normalised frame before it
    before_brightness: float | None = None
    lasting: bool = False  # confirmed by a lasting change, not by a frame like the candidate one


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
_PAD_SET = frozenset(_PAD)
# 2x2 blocks of cells that touch no pad cell: the floor and the tray rim
_FLOOR_BLOCKS = [
    (r * SIZE + c, r * SIZE + c + 1, (r + 1) * SIZE + c, (r + 1) * SIZE + c + 1)
    for r in range(SIZE - 1)
    for c in range(SIZE - 1)
    if not _PAD_SET.intersection((r * SIZE + c, r * SIZE + c + 1, (r + 1) * SIZE + c,
                                  (r + 1) * SIZE + c + 1))
]  # fmt: skip


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


def _floor_peak(prev: list[float], now: list[float]) -> tuple[float, float, int]:
    """(peak over 2x2 blocks, mean over the control floor, top-left cell of the peak block).

    Only darkening counts: a poop is darker than the tiles, and someone cleaning it up (brighter)
    must not open a visit.
    """
    dark = [max(0.0, p - n) for p, n in zip(prev, now, strict=True)]
    peak, where = -1.0, _FLOOR_BLOCKS[0][0]
    for block in _FLOOR_BLOCKS:
        value = sum(dark[i] for i in block) / 4
        if value > peak:
            peak, where = value, block[0]
    return peak, sum(dark[i] for i in _CONTROL) / len(_CONTROL), where


def _floor_darkening(prev: list[float], now: list[float]) -> tuple[float, float]:
    peak, mean, _ = _floor_peak(prev, now)
    return peak, mean


def _block_darkening(prev: list[float], now: list[float], cell: int) -> float:
    """Darkening of `now` against `prev` over the 2x2 block at `cell` and its 8 neighbours (max).

    The tolerance of one cell keeps a poop that the dog nudged, or JPEG noise, from looking gone.
    """
    row, col = divmod(cell, SIZE)
    best = 0.0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r, c = row + dr, col + dc
            if not (0 <= r < SIZE - 1 and 0 <= c < SIZE - 1):
                continue
            idx = (r * SIZE + c, r * SIZE + c + 1, (r + 1) * SIZE + c, (r + 1) * SIZE + c + 1)
            best = max(best, sum(max(0.0, prev[i] - now[i]) for i in idx) / 4)
    return best


def _near(a: int, b: int, reach: int = 3) -> bool:
    (ra, ca), (rb, cb) = divmod(a, SIZE), divmod(b, SIZE)
    return abs(ra - rb) <= reach and abs(ca - cb) <= reach


def floor_blob(before_jpeg: bytes, after_jpeg: bytes) -> float | None:
    """Peak of a new dark blob on the floor or tray rim between two visit frames, else None.

    A second opinion on Gemini's "no poop" (2026-11-05 06:13: the poop was in the photo, Gemini
    missed it). Same rule as a floor candidate. On the 55 stored visits it fires for the poops
    beside the tray (1.19, 1.67, 3.45) and once without a poop (1.03). It cannot see a poop ON the
    pad: there a pee darkens as much (up to 3.7).
    """
    try:
        before = _normalise(_pixels(before_jpeg))
        after = _normalise(_pixels(after_jpeg))
    except ValueError as exc:
        log.warning("watch: cannot compare the visit frames: %s", exc)
        return None
    peak, floor_dark, _ = _floor_peak(before, after)
    blob = (
        peak >= FLOOR_PEAK_MIN
        and floor_dark <= FLOOR_CTRL_MAX
        and peak >= FLOOR_RATIO * max(floor_dark, 0.02)
    )
    return round(peak, 2) if blob else None


@dataclass
class PoopTrack:
    """One counted poop the watch keeps an eye on, until it is gone."""

    visit_id: str
    cell: int  # top-left cell of the 2x2 block that darkened the most
    peak: float
    before_norm: list[float]
    clear_streak: int = 0


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
        self._poops: dict[str, PoopTrack] = {}
        self._cleared: list[str] = []

    # --- counted poops (ADR-0014) ---

    def track_poop(self, visit_id: str, before_jpeg: bytes, after_jpeg: bytes) -> bool:
        """Start following the blob that appeared between `before_jpeg` and `after_jpeg`.

        Returns False (nothing tracked) when no block darkened enough to follow, for instance a
        poop on the pad at night: only the owner, a pad change or the timeout clear that one.
        """
        try:
            before, after = _normalise(_pixels(before_jpeg)), _normalise(_pixels(after_jpeg))
        except ValueError as exc:
            log.warning("watch: cannot track the poop of %s: %s", visit_id, exc)
            return False
        dark = [max(0.0, b - a) for b, a in zip(before, after, strict=True)]
        best, cell = -1.0, 0
        for r in range(SIZE - 1):
            for c in range(SIZE - 1):
                i = r * SIZE + c
                value = (dark[i] + dark[i + 1] + dark[i + SIZE] + dark[i + SIZE + 1]) / 4
                if value > best:
                    best, cell = value, i
        if best < POOP_TRACK_MIN_PEAK:
            log.info("watch: poop of %s too faint to follow (peak %.2f)", visit_id, best)
            return False
        self._poops[visit_id] = PoopTrack(visit_id, cell, best, before)
        log.info("watch: following the poop of %s (peak %.2f)", visit_id, best)
        return True

    def untrack_poop(self, visit_id: str) -> None:
        self._poops.pop(visit_id, None)

    def tracked_poops(self) -> list[str]:
        return list(self._poops)

    def take_cleared(self) -> list[str]:
        """Visit ids whose poop is gone since the last call."""
        cleared, self._cleared = self._cleared, []
        return cleared

    def _follow_poops(self, norm: list[float], jump: bool) -> None:
        for track in list(self._poops.values()):
            if jump:  # the whole scene changed: say nothing about the blob
                track.clear_streak = 0
                continue
            now = _block_darkening(track.before_norm, norm, track.cell)
            track.clear_streak = track.clear_streak + 1 if now < POOP_GONE_RATIO * track.peak else 0
            if track.clear_streak >= POOP_GONE_FRAMES:
                del self._poops[track.visit_id]
                self._cleared.append(track.visit_id)
                log.info("watch: the poop of %s is gone (%.2f of %.2f)", track.visit_id, now,
                         track.peak)  # fmt: skip

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
        if self._poops:
            jump_now = self._prev is not None and abs(brightness - self._prev[1]) >= LIGHT_JUMP
            self._follow_poops(norm, jump_now)
        if self._pending is not None:
            confirmed = self._resolve_pending(norm, brightness, now, crop_jpeg)
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
            peak, floor_dark, peak_cell = _floor_peak(prev_norm, norm)
            floor = (
                not candidate
                and not jump
                and not any(_near(peak_cell, t.cell) for t in self._poops.values())
                and peak >= FLOOR_PEAK_MIN
                and floor_dark <= FLOOR_CTRL_MAX
                and peak >= FLOOR_RATIO * max(floor_dark, 0.02)
            )
            row = {
                "at": now.isoformat(timespec="seconds"),
                "pad_step": round(pad_step, 3),
                "ctrl_step": round(ctrl_step, 3),
                "brightness": round(brightness, 1),
                "light_jump": jump,
                "candidate": candidate or floor,
                "confirmed": False,
                "floor_peak": round(peak, 2),
                "kind": "floor" if floor else "pad",
            }
            self.last_row = row
            if candidate or floor:
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
                    kind="floor" if floor else "pad",
                    floor_peak=peak,
                    before_norm=prev_norm,
                    before_brightness=prev_brightness,
                )
                if floor:
                    log.info("watch: dark blob outside the pad (peak %.2f, floor %.3f)", peak,
                             floor_dark)  # fmt: skip
                else:
                    log.info(
                        "watch: candidate at the pad (pad %.2f, floor %.2f)", pad_step, ctrl_step
                    )
            else:
                self._write(row)
        self._prev = (norm, brightness, crop_jpeg, now)
        return None

    def _resolve_pending(
        self, norm: list[float], brightness: float, now: datetime, crop_jpeg: bytes
    ) -> Candidate | None:
        pending, row = self._pending, self._pending_row
        assert pending is not None and row is not None
        pending.seen += 1
        if pending.kind == "floor" and pending.before_norm is not None:
            # the blob is still there (darker than before the candidate) and the floor is as in
            # the candidate frame
            peak, _ = _floor_darkening(pending.before_norm, norm)
            same = (
                peak >= FLOOR_PEAK_MIN * 0.6
                and _mean_abs(norm, pending.norm, _CONTROL) < CONFIRM_MAX
                and abs(brightness - pending.brightness) < LIGHT_JUMP
            )
        else:
            same = (
                _mean_abs(norm, pending.norm, _PAD) < CONFIRM_MAX
                and abs(brightness - pending.brightness) < LIGHT_JUMP
            )
            if not same and self._lasting(pending, norm, brightness):
                # the candidate frame still had her in it (2026-11-07 08:52: Frigate did not see
                # her, no later frame matched it): judge this clear frame instead
                pending.after_jpeg, pending.lasting, same = crop_jpeg, True, True
                row["kind"] = "pad_lasting"
        expired = now - pending.at > timedelta(seconds=PENDING_TTL_S)
        if same:
            row["confirmed"] = True
            self._write(row)
            self._pending = self._pending_row = None
            self.confirmed_total += 1
            log.info("watch: %s change confirmed, Frigate did not report it", pending.kind)
            return pending
        if expired:
            self._write(row)  # given up: confirmed stays False
            self._pending = self._pending_row = None
            log.info("watch: candidate not confirmed, dropped")
        return None

    @staticmethod
    def _lasting(pending: Candidate, norm: list[float], brightness: float) -> bool:
        """The pad still differs from the frame BEFORE the candidate, with a quiet floor and no
        light change: something lasting happened at the pad (a pee), whatever the candidate frame
        looked like."""
        if pending.before_norm is None or pending.before_brightness is None:
            return False
        return (
            _mean_abs(norm, pending.before_norm, _PAD) >= LASTING_MIN
            and _mean_abs(norm, pending.before_norm, _CONTROL) <= CTRL_MAX
            and abs(brightness - pending.before_brightness) < LIGHT_JUMP
        )

    def _write(self, row: dict) -> None:
        if self._log_path is None:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._log_path.exists() and not self._header_ok():
                self._log_path.rename(self._log_path.with_suffix(".old.csv"))  # older columns
            new = not self._log_path.exists()
            with self._log_path.open("a", encoding="utf-8") as f:
                if new:
                    f.write(CSV_HEADER)
                f.write(
                    f"{row['at']},{row['pad_step']},{row['ctrl_step']},{row['brightness']},"
                    f"{int(row['light_jump'])},{int(row['candidate'])},{int(row['confirmed'])},"
                    f"{row['floor_peak']},{row['kind']}\n"
                )
        except OSError as exc:
            log.warning("watch: could not write %s: %s", self._log_path, exc)

    def _header_ok(self) -> bool:
        try:
            with self._log_path.open("rb") as f:
                first = f.readline().decode("utf-8", "ignore").lstrip("\x00")
        except OSError:
            return True
        return first == CSV_HEADER
