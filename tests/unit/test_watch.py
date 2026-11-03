"""The pad watch: a step at the pad that stays is a candidate; light and noise are not."""

import io
from datetime import UTC, datetime, timedelta

from PIL import Image

from ellie.watch import LIGHT_JUMP, PAD_BOX, PENDING_TTL_S, PadWatch

START = datetime(2026, 11, 3, 14, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def frame(base: int = 120, pad: int | None = None, size: int = 200) -> bytes:
    """A floor with a little texture and a pad box, optionally with a darker patch on the pad."""
    img = Image.new("L", (size, size), base)
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (
                base + ((x // 3) * 7 + (y // 3) * 13) % 6 * 8
            )  # busy floor texture, like a photo
    if pad is not None:
        x1, y1, x2, y2 = PAD_BOX
        cx, cy = int((x1 + x2) / 2 * size), int((y1 + y2) / 2 * size)
        for y in range(cy - 25, cy + 25):
            for x in range(cx - 25, cx + 25):
                px[x, y] = pad
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def person_at_the_tray() -> bytes:
    """The pad and the floor around it both change a lot (someone walks in)."""
    img = Image.open(io.BytesIO(frame(pad=60)))
    px = img.load()
    for y in range(img.height):
        for x in range(60):
            px[x, y] = 30
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def watch(tmp_path=None) -> tuple[PadWatch, Clock]:
    clock = Clock()
    return PadWatch(clock, None if tmp_path is None else tmp_path / "w" / "watch.csv"), clock


def minute(w: PadWatch, clock: Clock, jpeg: bytes):
    clock.advance(60)
    return w.observe(jpeg)


def test_the_first_frame_has_nothing_to_compare() -> None:
    w, clock = watch()
    assert minute(w, clock, frame()) is None and w.last_row is None


def test_an_unchanged_scene_is_not_a_candidate() -> None:
    w, clock = watch()
    minute(w, clock, frame())
    assert minute(w, clock, frame()) is None
    assert w.last_row["candidate"] is False and w.last_row["pad_step"] < 0.05


def test_a_patch_that_stays_is_confirmed_and_carries_both_frames(tmp_path) -> None:
    w, clock = watch(tmp_path)
    clean, stained = frame(), frame(pad=60)
    minute(w, clock, clean)
    assert minute(w, clock, stained) is None  # a candidate, waiting for a later frame
    assert w.last_row["candidate"] is True and w.last_row["ctrl_step"] < w.last_row["pad_step"]
    found = minute(w, clock, stained)  # still there a minute later
    assert found is not None and found.before_jpeg == clean and found.after_jpeg == stained
    assert (found.at - found.before_at).total_seconds() == 60 and w.confirmed_total == 1
    lines = (tmp_path / "w" / "watch.csv").read_text().splitlines()
    assert lines[0].startswith("at,pad_step") and lines[-1].endswith(",1,1")


def test_a_lost_frame_or_someone_at_the_tray_does_not_break_the_confirmation() -> None:
    w, clock = watch()
    minute(w, clock, frame())
    minute(w, clock, frame(pad=60))  # candidate
    assert minute(w, clock, person_at_the_tray()) is None  # a person: not the same pad
    assert minute(w, clock, frame(pad=60)) is not None  # gone again, the pad is as in the candidate


def test_a_patch_that_disappears_is_dropped_after_the_wait(tmp_path) -> None:
    w, clock = watch(tmp_path)
    minute(w, clock, frame())
    minute(w, clock, frame(pad=60))  # a shadow or a paw
    assert minute(w, clock, frame()) is None  # the pad is clean again
    clock.advance(PENDING_TTL_S)
    assert minute(w, clock, frame()) is None
    csv = (tmp_path / "w" / "watch.csv").read_text().strip()
    assert w.confirmed_total == 0 and csv.endswith(",1,0")


def test_a_uniform_brightness_change_is_light_not_a_pee() -> None:
    w, clock = watch()
    minute(w, clock, frame(base=100))
    assert minute(w, clock, frame(base=100 + int(LIGHT_JUMP) + 20)) is None
    assert w.last_row["candidate"] is False


def test_a_visit_rebases_the_watch() -> None:
    w, clock = watch()
    minute(w, clock, frame())
    minute(w, clock, frame(pad=60))  # candidate
    w.rebase(frame(pad=60))  # a Frigate visit explained it: its after frame is the baseline
    assert minute(w, clock, frame(pad=60)) is None and w.last_row["candidate"] is False
    assert w.confirmed_total == 0


def test_a_bad_frame_is_skipped_without_raising() -> None:
    w, clock = watch()
    assert minute(w, clock, b"not a jpeg") is None
