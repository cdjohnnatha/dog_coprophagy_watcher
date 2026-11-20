"""Visits the owner answered in doubt (3.10.8): a summary plus a 2x2 grid of each one.

Reads /media/ellie/dataset/labels.jsonl (latest line per visit wins), keeps the visits whose
latest answer carries `doubt`, counts them by reason and by bad photo, and writes one
review grid per visit (tools/review_grid.py) into OUT_DIR.

Usage: python tools/doubt_report.py OUT_DIR [--since YYYY-MM-DD] [--root /media/ellie]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # works with python -I too
from review_grid import grid

REASONS = {"glare": "reflexo ou luz forte", "old_stain": "mancha antiga confunde", "other": "outro"}
PHOTOS = {"before": "antes", "after": "depois", "both": "as duas", "none": "nenhuma"}


def latest_labels(labels: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for line in labels.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        latest[row["visit_id"]] = row
    return latest


def visit_dir(root: Path, visit_id: str) -> Path:
    stamp = visit_id[2:10]
    return root / "visits" / stamp[:4] / stamp[4:6] / stamp[6:8] / visit_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--since", default="")
    ap.add_argument("--root", default="/media/ellie")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    since = args.since.replace("-", "")
    rows = [
        r
        for r in latest_labels(root / "dataset" / "labels.jsonl").values()
        if r.get("doubt") and r["visit_id"][2:10] >= since
    ]
    print(f"{len(rows)} visit(s) answered in doubt")
    print("by reason:", dict(Counter(REASONS.get(r["doubt"]["reason"], "?") for r in rows)))
    print("bad photo:", dict(Counter(PHOTOS.get(r["doubt"]["photos"], "?") for r in rows)))
    for r in sorted(rows, key=lambda r: r["visit_id"]):
        folder = visit_dir(root, r["visit_id"])
        made = ""
        if (folder / "before.jpg").exists() and (folder / "after.jpg").exists():
            grid(folder).save(out / f"{r['visit_id']}_grid.jpg", quality=88)
            made = " (grid)"
        d = r["doubt"]
        print(
            f"{r['visit_id']} {r['label']:6} {PHOTOS.get(d['photos'])} · {REASONS.get(d['reason'])}{made}"
        )


if __name__ == "__main__":
    main()
