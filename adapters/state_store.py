"""
Simple JSON state store (persistent across add-on restarts).

This module is intentionally tiny and dependency-free.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional


class JsonStateStore:
    """Persist small JSON blobs to disk with atomic writes."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> Dict[str, Any]:
        """Load JSON from disk, returning {} if missing/corrupt."""
        try:
            if not os.path.exists(self.path):
                return {}
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save(self, data: Dict[str, Any]) -> None:
        """Atomic save to disk (write temp + replace)."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        tmp_fd: Optional[int] = None
        tmp_path: Optional[str] = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=os.path.dirname(self.path) or ".")
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            tmp_fd = None
            os.replace(tmp_path, self.path)
        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except Exception:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

