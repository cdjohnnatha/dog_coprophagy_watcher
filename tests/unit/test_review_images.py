"""Pad close-ups for the review popup (ADR-0023)."""

from __future__ import annotations

import io
import os

from PIL import Image, ImageDraw

from ellie import review_images as ri

SIZE = (400, 300)
PAD = (150, 95, 270, 195)  # inside PAD_CROP of a 400x300 frame


def frame(floor=(150, 110, 80), pad=(225, 230, 228), stain=None) -> Image.Image:
    """Brown floor, light pad, an optional stain (x1, y1, x2, y2, colour) on the pad."""
    img = Image.new("RGB", SIZE, floor)
    draw = ImageDraw.Draw(img)
    draw.rectangle(PAD, fill=pad)
    if stain:
        draw.rectangle(stain[:4], fill=stain[4])
    return img


def save(img: Image.Image, path) -> None:
    img.save(path, format="JPEG", quality=95)


def grey(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


def mean(img: Image.Image, channel: int) -> float:
    px = img.get_flattened_data()
    return sum(p[channel] for p in px) / len(px)


def test_infrared_frames_are_told_from_colour_frames():
    assert ri.is_colour(frame())
    assert not ri.is_colour(grey(frame()))


def test_white_balance_removes_a_cyan_cast_on_the_clean_pad():
    cast = Image.new("RGB", (60, 60), (190, 235, 235))
    balanced = ri.white_balance(cast)
    r, g, b = (mean(balanced, c) for c in range(3))
    assert max(r, g, b) - min(r, g, b) < 6


def test_before_and_after_share_one_contrast_stretch():
    dark = Image.new("RGB", (50, 50), (100, 100, 100))
    light = Image.new("RGB", (50, 50), (160, 160, 160))
    together = ri.enhance([dark, light])
    alone = ri.enhance([light])
    # stretched together the light frame stays lighter than the dark one, not pushed to white alone
    assert mean(together[1], 0) > mean(together[0], 0)
    assert mean(together[1], 0) != mean(alone[0], 0)


def test_the_yellow_map_is_brighter_on_a_yellow_stain_than_on_white_glare():
    img = Image.new("RGB", (120, 60), (235, 235, 235))  # white glare / clean pad
    ImageDraw.Draw(img).rectangle((60, 0, 119, 59), fill=(200, 180, 90))  # yellow pee
    (yellow,) = ri.yellow_map([img])
    left = yellow.crop((5, 5, 50, 55)).convert("L")
    right = yellow.crop((70, 5, 115, 55)).convert("L")
    assert mean(right.convert("RGB"), 0) > mean(left.convert("RGB"), 0) + 50


def test_colour_visit_gets_all_six_close_ups(tmp_path):
    save(frame(), tmp_path / "before.jpg")
    save(frame(stain=(170, 120, 230, 170, (200, 180, 90))), tmp_path / "after.jpg")
    paths = ri.write_variants(tmp_path)
    assert all(paths[k] is not None for k in paths), paths
    pad = Image.open(paths["after_pad"])
    w, h = SIZE
    assert pad.size == (int(0.705 * w) - int(0.345 * w), int(0.675 * h) - int(0.285 * h))


def test_infrared_frame_gets_only_its_pad_crop(tmp_path):
    save(grey(frame()), tmp_path / "before.jpg")
    save(frame(), tmp_path / "after.jpg")
    paths = ri.write_variants(tmp_path)
    assert paths["before_pad"] is not None
    assert paths["before_enh"] is None and paths["before_yellow"] is None
    assert paths["after_enh"] is not None and paths["after_yellow"] is not None


def test_close_ups_are_written_once_and_redone_when_the_photo_changes(tmp_path):
    save(frame(), tmp_path / "before.jpg")
    save(frame(), tmp_path / "after.jpg")
    first = ri.write_variants(tmp_path)
    stamp = first["after_enh"].stat().st_mtime_ns
    assert ri.write_variants(tmp_path)["after_enh"].stat().st_mtime_ns == stamp

    for f in tmp_path.iterdir():  # everything so far is 10 s old
        old = f.stat().st_mtime - 10
        os.utime(f, (old, old))
    save(grey(frame()), tmp_path / "after.jpg")  # replaced by an infrared frame
    again = ri.write_variants(tmp_path)
    assert again["after_enh"] is None and again["after_yellow"] is None
    assert again["after_pad"] is not None
    # unchanged infrared frame: nothing rewritten the next time
    pad_stamp = again["after_pad"].stat().st_mtime_ns
    assert ri.write_variants(tmp_path)["after_pad"].stat().st_mtime_ns == pad_stamp


def test_missing_or_broken_photos_never_raise(tmp_path):
    assert all(v is None for v in ri.write_variants(tmp_path).values())
    (tmp_path / "before.jpg").write_bytes(b"not a jpeg")
    assert all(v is None for v in ri.write_variants(tmp_path).values())


def test_close_ups_are_jpeg(tmp_path):
    save(frame(), tmp_path / "before.jpg")
    paths = ri.write_variants(tmp_path)
    data = paths["before_enh"].read_bytes()
    assert data[:2] == b"\xff\xd8" and Image.open(io.BytesIO(data)).format == "JPEG"
