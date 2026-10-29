"""Evidence file layout on /media. Paths are derived from ids, never stored."""

import os
import re
import tempfile
from pathlib import Path

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
