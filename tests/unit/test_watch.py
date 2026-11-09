"""The pad watch: a step at the pad that stays is a candidate; light and noise are not."""

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image

from ellie.watch import LIGHT_JUMP, PAD_BOX, PENDING_TTL_S, PadWatch, floor_blob

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
    assert lines[0].startswith("at,pad_step") and lines[-1].split(",")[5:7] == ["1", "1"]


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
    csv = (tmp_path / "w" / "watch.csv").read_text().strip().splitlines()[-1]
    assert w.confirmed_total == 0 and csv.split(",")[5:7] == ["1", "0"]


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


# --- poop on the floor (T107c) ---

FIXTURES = Path(__file__).parent.parent / "fixtures"


def blob(jpeg: bytes, at=(20, 150), size=5, level=20) -> bytes:
    """A small dark blob on the floor, far from the pad box."""
    img = Image.open(io.BytesIO(jpeg))
    px = img.load()
    for y in range(at[1], at[1] + size):
        for x in range(at[0], at[0] + size):
            px[x, y] = level
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


def test_a_dark_blob_on_the_floor_that_stays_is_a_confirmed_floor_candidate() -> None:
    w, clock = watch()
    clean = frame()
    poop = blob(clean)
    minute(w, clock, clean)
    assert minute(w, clock, poop) is None
    assert w.last_row["kind"] == "floor" and w.last_row["candidate"] is True
    found = minute(w, clock, poop)
    assert found is not None and found.kind == "floor" and found.floor_peak >= 1.0


def test_a_floor_blob_that_leaves_again_is_dropped() -> None:
    w, clock = watch()
    clean = frame()
    minute(w, clock, clean)
    minute(w, clock, blob(clean))  # candidate
    for _ in range(PENDING_TTL_S // 60 + 1):
        assert minute(w, clock, clean) is None
    assert w.confirmed_total == 0


def test_cleaning_the_blob_up_is_not_a_candidate() -> None:
    w, clock = watch()
    clean = frame()
    poop = blob(clean)
    minute(w, clock, poop)
    assert minute(w, clock, clean) is None  # brighter, not darker
    assert w.last_row["candidate"] is False


def test_the_real_poop_of_2026_11_05_0613_is_a_floor_candidate() -> None:
    w, clock = watch()
    before = (FIXTURES / "poop_0613" / "before.jpg").read_bytes()
    after = (FIXTURES / "poop_0613" / "after.jpg").read_bytes()
    minute(w, clock, before)
    assert minute(w, clock, after) is None
    assert w.last_row["kind"] == "floor" and w.last_row["candidate"] is True
    found = minute(w, clock, after)
    assert found is not None and found.kind == "floor"


def test_a_real_lighting_change_is_not_a_floor_candidate() -> None:
    """2026-11-02 21:36: the lights changed (peak 1.15 but a noisy floor)."""
    w, clock = watch()
    minute(w, clock, (FIXTURES / "light_0211" / "before.jpg").read_bytes())
    assert minute(w, clock, (FIXTURES / "light_0211" / "after.jpg").read_bytes()) is None
    assert w.last_row["candidate"] is False


def test_an_old_csv_with_other_columns_is_set_aside(tmp_path) -> None:
    path = tmp_path / "w" / "watch.csv"
    path.parent.mkdir()
    path.write_text("at,pad_step,ctrl_step,brightness,light_jump,candidate,confirmed\nold\n")
    w = PadWatch(Clock(), path)
    clock = w._clock
    minute(w, clock, frame())
    minute(w, clock, frame())
    assert (tmp_path / "w" / "watch.old.csv").exists()
    assert path.read_text().splitlines()[0].endswith("floor_peak,kind")


# --- following a counted poop (ADR-0014) ---


def test_a_faint_blob_is_not_followed() -> None:
    w, _ = watch()
    clean = frame()
    assert w.track_poop("v1", clean, clean) is False and w.tracked_poops() == []


def test_a_followed_poop_is_gone_after_two_clear_frames() -> None:
    w, clock = watch()
    clean = frame()
    poop = blob(clean)
    assert w.track_poop("v1", clean, poop) is True
    minute(w, clock, poop)
    minute(w, clock, poop)
    assert w.take_cleared() == []
    minute(w, clock, clean)
    assert w.take_cleared() == []  # one clear frame
    minute(w, clock, clean)
    assert w.take_cleared() == ["v1"] and w.tracked_poops() == []
    assert w.take_cleared() == []  # drained


def test_a_light_jump_resets_the_clear_count() -> None:
    w, clock = watch()
    clean = frame()
    poop = blob(clean)
    w.track_poop("v1", clean, poop)
    minute(w, clock, poop)
    minute(w, clock, clean)
    minute(w, clock, frame(base=120 + int(LIGHT_JUMP * 2)))  # the lights changed
    assert w.take_cleared() == []
    minute(w, clock, clean)  # a jump back: no clear either
    assert w.take_cleared() == []
    minute(w, clock, clean)
    minute(w, clock, clean)
    assert w.take_cleared() == ["v1"]


def test_a_tracked_blob_does_not_make_a_new_floor_candidate() -> None:
    w, clock = watch()
    clean = frame()
    poop = blob(clean)
    minute(w, clock, clean)
    w.track_poop("v1", clean, poop)
    minute(w, clock, poop)  # the blob appears: it is already known
    assert w.last_row["candidate"] is False
    other = blob(poop, at=(150, 30))  # a second, different blob far away is a candidate
    minute(w, clock, other)
    assert w.last_row["kind"] == "floor" and w.last_row["candidate"] is True


def test_untracking_forgets_the_poop() -> None:
    w, clock = watch()
    clean = frame()
    w.track_poop("v1", clean, blob(clean))
    w.untrack_poop("v1")
    minute(w, clock, clean)
    minute(w, clock, clean)
    assert w.take_cleared() == []


def test_a_candidate_frame_with_her_in_it_is_confirmed_by_the_lasting_change() -> None:
    """2026-11-07 08:52: Frigate did not see her, the candidate frame had her at the pad and no
    later frame matched it. The pad still differs from the frame before: confirm on that frame."""
    w, clock = watch()
    clean, with_her, stain = frame(), frame(pad=15), frame(pad=95)
    minute(w, clock, clean)
    assert minute(w, clock, with_her) is None  # candidate
    found = minute(w, clock, stain)  # she left, the stain stays
    assert found is not None and found.before_jpeg == clean and found.after_jpeg == stain


def test_a_candidate_that_went_back_to_the_frame_before_is_not_lasting() -> None:
    w, clock = watch()
    minute(w, clock, frame())
    minute(w, clock, frame(pad=15))  # a paw, a shadow
    assert minute(w, clock, frame()) is None  # the pad is as before: nothing lasting
    assert w.confirmed_total == 0


def test_floor_blob_sees_the_poop_gemini_missed_and_nothing_on_a_quiet_floor() -> None:
    fixtures = Path(__file__).parent.parent / "fixtures" / "poop_0613"
    before, after = (fixtures / "before.jpg").read_bytes(), (fixtures / "after.jpg").read_bytes()
    assert floor_blob(before, after) is not None  # 2026-11-05 06:13, beside the tray
    assert floor_blob(before, before) is None
    assert floor_blob(frame(), frame(pad=60)) is None  # a pee on the pad is not a floor blob
    assert floor_blob(b"not a jpeg", before) is None
