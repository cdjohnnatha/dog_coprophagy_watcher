"""Evidence file layout on /media. Paths are derived from ids, never stored."""

import json
import logging
import os
import re
import shutil
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_VISIT_RE = re.compile(r"^v_(\d{4})(\d{2})(\d{2})_\d{6}$")
_PRESENCE_RE = re.compile(r"^s_(\d{4})(\d{2})(\d{2})_\d{6}$")
_LOCAL_PREFIX = "/media/local/ellie/"


def _dated(root: Path, kind: str, pattern: re.Pattern[str], item_id: str) -> Path:
    m = pattern.match(item_id)
    if m is None:
        raise ValueError(f"malformed id: {item_id!r}")
    year, month, day = m.groups()
    return root / kind / year / month / day / item_id


class Evidence:
    """Paths and atomic writes for visit and pad evidence."""

    def __init__(self, root: Path = Path("/media/ellie")) -> None:
        self.root = root

    def visit_dir(self, visit_id: str) -> Path:
        return _dated(self.root, "visits", _VISIT_RE, visit_id)

    def presence_dir(self, presence_id: str) -> Path:
        return _dated(self.root, "pad", _PRESENCE_RE, presence_id)

    def clip_path(self, visit_id: str) -> Path:
        return self.visit_dir(visit_id) / "clip.mp4"

    def write(self, path: Path, data: bytes) -> None:
        """Write atomically (temp file in the same directory, then rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def local_url(self, path: Path) -> str:
        """URL under which Home Assistant serves a file in /media."""
        try:
            rel = path.relative_to(self.root)
        except ValueError:
            raise ValueError(f"{path} is outside {self.root}") from None
        return _LOCAL_PREFIX + rel.as_posix()

    def export_dataset(
        self,
        visit: dict,
        label: str,
        night: bool | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        """Copy a visit's images into dataset/<label>/<visit_id>/ and log the label."""
        if label not in ("pee", "no_pee"):
            raise ValueError(f"invalid label: {label!r}")
        visit_id = visit["id"]
        src = self.visit_dir(visit_id)
        dataset = self.root / "dataset"
        other = "no_pee" if label == "pee" else "pee"
        shutil.rmtree(dataset / other / visit_id, ignore_errors=True)
        dest = dataset / label / visit_id
        files: list[str] = []
        for name in ("before.jpg", "after.jpg", "during.jpg"):
            if (src / name).is_file():
                self.write(dest / name, (src / name).read_bytes())
                files.append(name)
        stamp = (now or datetime.now(UTC)).astimezone(UTC)
        record = {
            "visit_id": visit_id,
            "label": label,
            "poop_observed": visit.get("poop_observed"),
            "night": night,
            "labelled_at": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": files,
        }
        dataset.mkdir(parents=True, exist_ok=True)
        with (dataset / "labels.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return files

    async def download_clip(
        self,
        client: httpx.AsyncClient,
        frigate_url: str,
        event_id: str,
        visit_id: str,
        retries: int = 1,
    ) -> bool:
        """Fetch the Frigate clip for a visit; return False instead of raising."""
        url = f"{frigate_url}/api/events/{event_id}/clip.mp4"
        for _ in range(retries + 1):
            try:
                resp = await client.get(url, timeout=60)
            except httpx.HTTPError:
                continue
            if resp.status_code == 200 and resp.content:
                self.write(self.clip_path(visit_id), resp.content)
                return True
        logger.warning("clip download failed for visit %s (event %s)", visit_id, event_id)
        return False

    def _id_dirs(self, kind: str) -> list[tuple[date, Path]]:
        base = self.root / kind
        found: list[tuple[date, Path]] = []
        for p in base.glob("*/*/*/*"):
            if not p.is_dir():
                continue
            try:
                d = date(*(int(x) for x in p.relative_to(base).parts[:3]))
            except ValueError:
                continue
            found.append((d, p))
        return found

    def _size(self, path: Path) -> int:
        total = 0
        for dirpath, _dirs, names in os.walk(path):
            for n in names:
                try:
                    total += os.lstat(os.path.join(dirpath, n)).st_size
                except OSError:
                    pass
        return total

    def _remove(self, path: Path, stats: dict) -> None:
        stats["freed_bytes"] += self._size(path)
        shutil.rmtree(path, ignore_errors=True)
        stats["deleted_dirs"] += 1
        stop = self.root
        parent = path.parent
        while parent != stop and parent.parent != stop:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def purge(self, now: datetime, *, max_age_days: int = 90, max_bytes: int = 5 * 1024**3) -> dict:
        """Delete old evidence, then oldest-first until under the size limit."""
        stats = {"deleted_dirs": 0, "freed_bytes": 0}
        if not self.root.is_dir():
            return stats
        cutoff = now.date() - timedelta(days=max_age_days)
        evidence = sorted(
            self._id_dirs("visits") + self._id_dirs("pad"), key=lambda t: (t[0], t[1].name)
        )
        remaining: list[Path] = []
        for d, p in evidence:
            if d < cutoff:
                self._remove(p, stats)
            else:
                remaining.append(p)
        total = self._size(self.root)
        for p in remaining:
            if total <= max_bytes:
                return stats
            size = self._size(p)
            self._remove(p, stats)
            total -= size
        datasets = sorted(
            (
                p
                for lbl in ("pee", "no_pee")
                for p in (self.root / "dataset" / lbl).glob("*")
                if p.is_dir()
            ),
            key=lambda p: (p.name, p.parent.name),
        )
        for p in datasets:
            if total <= max_bytes:
                break
            size = self._size(p)
            stats["freed_bytes"] += size
            shutil.rmtree(p, ignore_errors=True)
            stats["deleted_dirs"] += 1
            total -= size
        return stats
