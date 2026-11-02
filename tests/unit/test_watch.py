"""The pad watch: a step at the pad that stays is a candidate; light and noise are not."""

import io
from datetime import UTC, datetime

from PIL import Image

from ellie.watch import LIGHT_JUMP, PAD_BOX, PadWatch

NOW = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)


def frame(base: int = 120, pad: int | None = None, size: int = 200) -> bytes:
    """A floor with a little texture and a pad box, optionally with a darker patch on the pad."""
    img = Image.new("L", (size, size), base)
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = base + ((x // 10 + y // 10) % 2) * 8  # tile pattern
    if pad is not None:
        x1, y1, x2, y2 = PAD_BOX
        cx, cy = int((x1 + x2) / 2 * size), int((y1 + y2) / 2 * size)
        for y in range(cy - 25, cy + 25):
            for x in range(cx - 25, cx + 25):
                px[x, y] = pad
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def watch(tmp_path=None) -> PadWatch:
    return PadWatch(lambda: NOW, None if tmp_path is None else tmp_path / "w" / "watch.csv")


def test_the_first_frame_has_nothing_to_compare() -> None:
    assert watch().observe(frame()) is None


def test_an_unchanged_scene_is_not_a_candidate() -> None:
    w = watch()
    w.observe(frame())
    row = w.observe(frame())
    assert row is not None and row["candidate"] is False and row["pad_step"] < 0.05


def test_a_patch_on_the_pad_that_stays_is_confirmed(tmp_path) -> None:
    w = watch(tmp_path)
    w.observe(frame())
    row = w.observe(frame(pad=60))  # a dark patch appears on the pad only
    assert row["candidate"] is True and row["ctrl_step"] < row["pad_step"]
    w.observe(frame(pad=60))  # and it is still there a minute later
    assert w.candidates_today == 1
    lines = (tmp_path / "w" / "watch.csv").read_text().splitlines()
    assert lines[0].startswith("at,pad_step") and lines[-1].endswith(",1,1")


def test_a_patch_that_disappears_is_not_confirmed() -> None:
    w = watch()
    w.observe(frame())
    w.observe(frame(pad=60))  # a shadow or the dog's paw
    w.observe(frame())  # gone
    assert w.candidates_today == 0


def test_a_uniform_brightness_change_is_light_not_a_pee() -> None:
    w = watch()
    w.observe(frame(base=100))
    row = w.observe(frame(base=100 + int(LIGHT_JUMP) + 20))
    assert row["candidate"] is False


def test_a_bad_frame_is_skipped_without_raising() -> None:
    assert watch().observe(b"not a jpeg") is None
