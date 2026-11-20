"""Prototype of the 2x2 review image (offline, not used by the app yet).

Top row: BEFORE and AFTER pad crops with the same contrast stretch and colour boost.
Bottom row: yellow maps (CIE LAB b*) of both, with a red outline on the AFTER map where
yellow grew. Infrared frames have no colour: the bottom row then shows the grey change.

Usage: python tools/review_grid.py OUT_DIR VISIT_DIR [VISIT_DIR ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageCms, ImageDraw, ImageEnhance, ImageFilter, ImageOps

CROP = (0.345, 0.285, 0.705, 0.675)  # pad inside the tray, fractions of the frame
TILE = 420
YELLOW_GROWTH = 10  # b* levels (0-255 scale) a pixel must gain to count as "more yellow"

_LAB = ImageCms.buildTransform(
    ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB"
)


def _crop(path: Path) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    x1, y1, x2, y2 = CROP
    c = im.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
    return c.resize((TILE, int(TILE * c.height / c.width)))


def is_colour(im: Image.Image) -> bool:
    px = im.resize((48, 36)).get_flattened_data()
    return sum(max(p) - min(p) for p in px) / len(px) > 12


def _white_balance(im: Image.Image) -> Image.Image:
    """Make the brightest third of the pad (clean pad paper) neutral grey: removes the cyan/green cast."""
    small = im.resize((96, 72)).get_flattened_data()
    bright = sorted(small, key=sum)[-len(small) // 3 :]
    avg = [sum(p[c] for p in bright) / len(bright) for c in range(3)]
    grey = sum(avg) / 3
    lut = []
    for c in range(3):
        k = grey / max(avg[c], 1)
        lut += [min(255, int(i * k)) for i in range(256)]
    return im.point(lut)


def _stretch_pair(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Same contrast stretch for both frames (cut points measured on both together)."""
    both = Image.new("RGB", (a.width, a.height + b.height))
    both.paste(a, (0, 0))
    both.paste(b, (0, a.height))
    hist = both.convert("L").histogram()
    n, acc, lo, hi = sum(hist), 0, 0, 255
    for i, v in enumerate(hist):
        acc += v
        if acc >= 0.01 * n:
            lo = i
            break
    acc = 0
    for i in range(255, -1, -1):
        acc += hist[i]
        if acc >= 0.01 * n:
            hi = i
            break
    scale = 255 / max(hi - lo, 1)
    lut = [max(0, min(255, int((i - lo) * scale))) for i in range(256)] * 3
    return tuple(
        ImageEnhance.Color(x.point(lut)).enhance(1.5)
        for x in (_white_balance(a), _white_balance(b))
    )


def _yellow(im: Image.Image) -> Image.Image:
    return ImageCms.applyTransform(im, _LAB).getchannel("B").filter(ImageFilter.GaussianBlur(3))


def _yellow_maps(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    ya, yb = _yellow(a), _yellow(b)
    lo = min(ya.getextrema()[0], yb.getextrema()[0])
    hi = max(ya.getextrema()[1], yb.getextrema()[1])
    scale = 255 / max(hi - lo, 1)
    lut = [max(0, min(255, int((i - lo) * scale))) for i in range(256)]
    paint = lambda y: ImageOps.colorize(
        y.point(lut), black="#101040", mid="#7a7a7a", white="#ffd000"
    )
    grown = ImageChops.subtract(yb, ya).point(lambda v: 255 if v >= YELLOW_GROWTH else 0)
    grown = grown.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))  # drop specks
    return paint(ya), paint(yb), grown


def _grey_change(a: Image.Image, b: Image.Image) -> tuple[Image.Image, Image.Image, Image.Image]:
    ga, gb = (
        ImageOps.autocontrast(a.convert("L"), cutoff=1),
        ImageOps.autocontrast(b.convert("L"), cutoff=1),
    )
    darker = ImageChops.subtract(ga, gb).filter(ImageFilter.GaussianBlur(3))
    grown = darker.point(lambda v: 255 if v >= 30 else 0)
    grown = grown.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))
    return ga.convert("RGB"), gb.convert("RGB"), grown


def _outline(base: Image.Image, mask: Image.Image) -> Image.Image:
    edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(5)), mask)
    out = base.copy()
    out.paste((255, 0, 0), (0, 0), edge)
    return out


def grid(visit: Path) -> Image.Image:
    a, b = _crop(visit / "before.jpg"), _crop(visit / "after.jpg")
    ea, eb = _stretch_pair(a, b)
    if is_colour(a) and is_colour(b):
        ma, mb, grown = _yellow_maps(a, b)
        bottom = "amarelo"
    else:
        ma, mb, grown = _grey_change(a, b)
        bottom = "cinza (infravermelho)"
    mb = _outline(mb, grown)
    share = sum(1 for v in grown.get_flattened_data() if v) / (grown.width * grown.height)
    h = a.height
    g = Image.new("RGB", (2 * TILE + 8, 2 * h + 8 + 24), "white")
    for i, t in enumerate((ea, eb, ma, mb)):
        g.paste(t, ((i % 2) * (TILE + 8), 24 + (i // 2) * (h + 8)))
    ImageDraw.Draw(g).text(
        (4, 6),
        f"{visit.name}   ANTES | DEPOIS   embaixo: {bottom}, vermelho = aumentou ({share:.1%})",
        fill="black",
    )
    return g


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for v in map(Path, sys.argv[2:]):
        grid(v).save(out / f"{v.name}_grid.jpg", quality=88)
        print(out / f"{v.name}_grid.jpg")


if __name__ == "__main__":
    main()
