"""Pad close-ups shown to the owner when a review photo is tapped (ADR-0023, FR-031).

For BEFORE and AFTER of a visit, three files next to the photos:

- ``<frame>_pad.jpg``: the pad cut out of the stored area crop, no change;
- ``<frame>_enh.jpg``: white balance on the clean pad, one contrast stretch measured on both
  frames together (so they stay comparable), a moderate colour boost;
- ``<frame>_yellow.jpg``: CIE LAB b* (blue → yellow), one stretch for both frames, painted
  dark blue → grey → yellow. Pee is yellow; glare is white, so it stays dark.

An infrared frame has no colour: it gets only ``_pad`` (owner, 2026-11-19). These files are for
the owner only; Gemini keeps receiving the plain frames. Pillow only.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, ImageCms, ImageEnhance, ImageFilter, ImageOps

log = logging.getLogger(__name__)

FRAMES = ("before", "after")
KINDS = ("pad", "enh", "yellow")
# The pad inside the tray, as fractions of the stored area crop (1368x1146 since 2026-10-29).
# Must follow the tray if it is moved (T152).
PAD_CROP = (0.345, 0.285, 0.705, 0.675)
COLOUR_MIN = 12.0  # mean max-min channel spread of the whole frame; infrared frames are ~0
COLOUR_BOOST = 1.5
STRETCH_CUT = 0.01  # share of darkest and brightest pixels clipped by the contrast stretch
QUALITY = 92

_TO_LAB = ImageCms.buildTransform(
    ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB"
)


def variant_path(visit_dir: Path, frame: str, kind: str) -> Path:
    return visit_dir / f"{frame}_{kind}.jpg"


def is_colour(img: Image.Image) -> bool:
    """False for an infrared (black-and-white) frame."""
    px = img.convert("RGB").resize((48, 36)).get_flattened_data()
    return sum(max(p) - min(p) for p in px) / len(px) > COLOUR_MIN


def pad_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = PAD_CROP
    box = (
        int(x1 * w),
        int(y1 * h),
        max(int(x2 * w), int(x1 * w) + 1),
        max(int(y2 * h), int(y1 * h) + 1),
    )
    return img.convert("RGB").crop(box)


def white_balance(img: Image.Image) -> Image.Image:
    """Make the brightest third (clean pad paper) neutral grey: removes a cyan/green cast."""
    px = img.resize((96, 72)).get_flattened_data()
    bright = sorted(px, key=sum)[-max(len(px) // 3, 1) :]
    avg = [sum(p[c] for p in bright) / len(bright) for c in range(3)]
    grey = sum(avg) / 3
    lut: list[int] = []
    for c in range(3):
        k = grey / max(avg[c], 1.0)
        lut += [min(255, int(i * k)) for i in range(256)]
    return img.point(lut)


def _cut_points(hist: list[int]) -> tuple[int, int]:
    n = sum(hist)
    acc, lo = 0, 0
    for i, v in enumerate(hist):
        acc += v
        if acc >= STRETCH_CUT * n:
            lo = i
            break
    acc, hi = 0, 255
    for i in range(255, -1, -1):
        acc += hist[i]
        if acc >= STRETCH_CUT * n:
            hi = i
            break
    return lo, max(hi, lo + 1)


def _stretch_lut(lo: int, hi: int) -> list[int]:
    scale = 255 / (hi - lo)
    return [max(0, min(255, int((i - lo) * scale))) for i in range(256)]


def _joint_hist(images: list[Image.Image], mode: str = "L") -> list[int]:
    total = [0] * 256
    for img in images:
        for i, v in enumerate(img.convert(mode).histogram()[:256]):
            total[i] += v
    return total


def enhance(crops: list[Image.Image]) -> list[Image.Image]:
    """White balance each, then one contrast stretch for all, then the colour boost."""
    balanced = [white_balance(c) for c in crops]
    lut = _stretch_lut(*_cut_points(_joint_hist(balanced))) * 3
    return [ImageEnhance.Color(b.point(lut)).enhance(COLOUR_BOOST) for b in balanced]


def yellow_map(crops: list[Image.Image]) -> list[Image.Image]:
    """LAB b* with one stretch for all crops, painted dark blue → grey → yellow."""
    bs = [
        ImageCms.applyTransform(c, _TO_LAB).getchannel("B").filter(ImageFilter.GaussianBlur(3))
        for c in crops
    ]
    lut = _stretch_lut(*_cut_points(_joint_hist(bs)))
    return [
        ImageOps.colorize(b.point(lut), black="#101040", mid="#7a7a7a", white="#ffd000") for b in bs
    ]


def _save(img: Image.Image, path: Path) -> None:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=QUALITY)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)


def _fresh(path: Path, source: Path) -> bool:
    return path.exists() and path.stat().st_mtime >= source.stat().st_mtime


def write_variants(visit_dir: Path) -> dict[str, Path | None]:
    """Write the missing or outdated close-ups; return {"before_pad": path or None, ...}.

    Colour frames share one stretch, so BEFORE and AFTER stay comparable. An infrared frame
    gets only its pad crop. Errors are logged, never raised (the review must still work).
    """
    result: dict[str, Path | None] = {f"{f}_{k}": None for f in FRAMES for k in KINDS}
    sources = {f: visit_dir / f"{f}.jpg" for f in FRAMES if (visit_dir / f"{f}.jpg").exists()}
    if not sources:
        return result
    if all(
        _fresh(variant_path(visit_dir, f, "pad"), s)
        and (
            _fresh(variant_path(visit_dir, f, "enh"), s)
            or not variant_path(visit_dir, f, "enh").exists()
            and _is_ir_marker(visit_dir, f)
        )
        for f, s in sources.items()
    ):
        return _existing(visit_dir, result)
    try:
        images = {f: Image.open(s).convert("RGB") for f, s in sources.items()}
        crops = {f: pad_crop(img) for f, img in images.items()}
        colour = [f for f in sources if is_colour(images[f])]
        for f, crop in crops.items():
            _save(crop, variant_path(visit_dir, f, "pad"))
            if f in colour:
                _ir_marker(visit_dir, f).unlink(missing_ok=True)
            else:
                _mark_ir(visit_dir, f)
        if colour:
            for f, img in zip(colour, enhance([crops[f] for f in colour]), strict=True):
                _save(img, variant_path(visit_dir, f, "enh"))
            for f, img in zip(colour, yellow_map([crops[f] for f in colour]), strict=True):
                _save(img, variant_path(visit_dir, f, "yellow"))
        for f in sources:
            if (
                f not in colour
            ):  # a frame replaced by an infrared one must not keep old colour views
                for k in ("enh", "yellow"):
                    variant_path(visit_dir, f, k).unlink(missing_ok=True)
    except Exception:  # a broken frame must not break the review
        log.exception("could not write the review close-ups of %s", visit_dir.name)
    return _existing(visit_dir, result)


def _ir_marker(visit_dir: Path, frame: str) -> Path:
    return visit_dir / f".{frame}_ir"


def _mark_ir(visit_dir: Path, frame: str) -> None:
    _ir_marker(visit_dir, frame).touch()


def _is_ir_marker(visit_dir: Path, frame: str) -> bool:
    marker = _ir_marker(visit_dir, frame)
    return (
        marker.exists() and marker.stat().st_mtime >= (visit_dir / f"{frame}.jpg").stat().st_mtime
    )


def _existing(visit_dir: Path, result: dict[str, Path | None]) -> dict[str, Path | None]:
    for key in result:
        frame, kind = key.split("_", 1)
        path = variant_path(visit_dir, frame, kind)
        result[key] = path if path.exists() else None
    return result
