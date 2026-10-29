"""SQLite persistence for visits, analyses and settings.

All SQL lives here. Times are UTC ISO-8601 text; JSON columns are encoded and
decoded at this boundary so callers only see Python lists, dicts and bools.
"""

import json
import sqlite3
from typing import Any

MIGRATIONS: list[str] = [
    # 1: visits, analyses, settings
    """
    CREATE TABLE visits (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL CHECK (source IN ('auto', 'manual')),
        started_at TEXT NOT NULL,
        ended_at TEXT,
        entries INTEGER NOT NULL DEFAULT 1 CHECK (entries >= 1),
        person_present INTEGER NOT NULL DEFAULT 0,
        frigate_event_ids TEXT NOT NULL DEFAULT '[]',
        before_age_s INTEGER,
        state TEXT NOT NULL CHECK (state IN
            ('IDLE', 'OPEN', 'CLOSING', 'CAPTURING', 'ANALYSING', 'DECIDED', 'LABELLED')),
        auto_outcome TEXT CHECK (auto_outcome IN ('pee', 'no_pee', 'uncertain', 'unknown')),
        final_outcome TEXT CHECK (final_outcome IN ('pee', 'no_pee', 'unknown')),
        poop_observed TEXT NOT NULL DEFAULT 'none'
            CHECK (poop_observed IN ('none', 'on_pad', 'beside_tray', 'unsure')),
        reason TEXT,
        needs_review INTEGER NOT NULL DEFAULT 0,
        deciding_analyser TEXT
    );
    CREATE INDEX idx_visits_started_at ON visits(started_at);
    CREATE INDEX idx_visits_needs_review ON visits(needs_review);
    CREATE TABLE analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id TEXT NOT NULL REFERENCES visits(id),
        analyser TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('deciding', 'shadow')),
        result TEXT,
        outcome TEXT,
        latency_ms INTEGER,
        error TEXT,
        cost_units TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    );
    CREATE INDEX idx_analyses_visit ON analyses(visit_id);
    CREATE TABLE settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """,
    # 2: seconds she stood on the pad itself (zone tray_pad), kept as evidence
    "ALTER TABLE visits ADD COLUMN on_pad_s REAL;",
]

_VISIT_COLUMNS = (
    "id",
    "source",
    "started_at",
    "ended_at",
    "entries",
    "person_present",
    "frigate_event_ids",
    "before_age_s",
    "state",
    "auto_outcome",
    "final_outcome",
    "poop_observed",
    "reason",
    "needs_review",
    "deciding_analyser",
    "on_pad_s",
)
_BOOL_FIELDS = ("person_present", "needs_review")
_JSON_FIELDS = ("frigate_event_ids",)


def _encode(field: str, value: Any) -> Any:
    if field in _JSON_FIELDS:
        return json.dumps(value)
    if field in _BOOL_FIELDS and value is not None:
        return int(bool(value))
    return value


def _visit_from_row(row: sqlite3.Row) -> dict:
    visit = dict(row)
    for field in _BOOL_FIELDS:
        if visit[field] is not None:
            visit[field] = bool(visit[field])
    for field in _JSON_FIELDS:
        if visit[field] is not None:
            visit[field] = json.loads(visit[field])
    return visit


def _loads(text: str | None) -> Any:
    return None if text is None else json.loads(text)


def _dumps(value: Any) -> str | None:
    return None if value is None else json.dumps(value)


class Store:
    """Single-connection SQLite store; migrations run on open."""

    def __init__(self, path: str) -> None:
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        self._db.close()

    def schema_version(self) -> int:
        return self._db.execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        for version in range(self.schema_version(), len(MIGRATIONS)):
            # executescript commits implicitly; user_version is set last so a
            # failed migration is retried on the next start.
            self._db.executescript(MIGRATIONS[version])
            self._db.execute(f"PRAGMA user_version = {version + 1}")
        self._db.commit()

    def insert_visit(self, visit: dict) -> None:
        unknown = set(visit) - set(_VISIT_COLUMNS)
        if unknown:
            raise ValueError(f"unknown visit fields: {sorted(unknown)}")
        columns = list(visit)
        sql = (
            f"INSERT INTO visits ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})"
        )
        with self._db:
            self._db.execute(sql, [_encode(c, visit[c]) for c in columns])

    def update_visit(self, visit_id: str, **fields: Any) -> None:
        unknown = set(fields) - set(_VISIT_COLUMNS)
        if unknown or "id" in fields:
            raise ValueError(
                f"unknown or immutable visit fields: {sorted(unknown | {'id'} & set(fields))}"
            )
        if not fields:
            if self.get_visit(visit_id) is None:
                raise KeyError(visit_id)
            return
        assignments = ", ".join(f"{c} = ?" for c in fields)
        sql = f"UPDATE visits SET {assignments} WHERE id = ?"
        with self._db:
            cur = self._db.execute(sql, [*(_encode(c, v) for c, v in fields.items()), visit_id])
        if cur.rowcount == 0:
            raise KeyError(visit_id)

    def get_visit(self, visit_id: str) -> dict | None:
        row = self._db.execute("SELECT * FROM visits WHERE id = ?", (visit_id,)).fetchone()
        return None if row is None else _visit_from_row(row)

    def _visits(self, where: str, params: tuple = ()) -> list[dict]:
        sql = f"SELECT * FROM visits WHERE {where} ORDER BY started_at"
        return [_visit_from_row(r) for r in self._db.execute(sql, params)]

    def visits_started_between(self, start_iso: str, end_iso: str) -> list[dict]:
        """Visits with start <= started_at < end, oldest first."""
        return self._visits("started_at >= ? AND started_at < ?", (start_iso, end_iso))

    def open_visits(self) -> list[dict]:
        return self._visits("state IN ('OPEN', 'CLOSING', 'CAPTURING')")

    def analysing_visits(self) -> list[dict]:
        return self._visits("state = 'ANALYSING'")

    def review_queue(self, limit: int = 20) -> list[dict]:
        """Visits flagged for review, newest first."""
        sql = "SELECT * FROM visits WHERE needs_review = 1 ORDER BY started_at DESC LIMIT ?"
        return [_visit_from_row(r) for r in self._db.execute(sql, (limit,))]

    def latest_visit(self) -> dict | None:
        row = self._db.execute("SELECT * FROM visits ORDER BY started_at DESC LIMIT 1").fetchone()
        return None if row is None else _visit_from_row(row)

    def usage_since(self, start_iso: str) -> dict:
        """Billed analysis calls and tokens since `start_iso` (cost_units holds in/out tokens)."""
        row = self._db.execute(
            "SELECT COUNT(*) AS calls,"
            " COALESCE(SUM(json_extract(cost_units, '$.in')), 0) AS tokens_in,"
            " COALESCE(SUM(json_extract(cost_units, '$.out')), 0) AS tokens_out"
            " FROM analyses WHERE created_at >= ? AND cost_units IS NOT NULL",
            (start_iso,),
        ).fetchone()
        return {k: int(row[k]) for k in ("calls", "tokens_in", "tokens_out")}

    def insert_analysis(
        self,
        visit_id: str,
        analyser: str,
        role: str,
        result: dict | None,
        outcome: str,
        latency_ms: int | None,
        error: str | None,
        cost_units: dict | None,
    ) -> int:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO analyses (visit_id, analyser, role, result, outcome, latency_ms,"
                " error, cost_units) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    visit_id,
                    analyser,
                    role,
                    _dumps(result),
                    outcome,
                    latency_ms,
                    error,
                    _dumps(cost_units),
                ),
            )
        return int(cur.lastrowid)

    def analyses_for(self, visit_id: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM analyses WHERE visit_id = ? ORDER BY id", (visit_id,)
        )
        out = []
        for row in rows:
            item = dict(row)
            item["result"] = _loads(item["result"])
            item["cost_units"] = _loads(item["cost_units"])
            out.append(item)
        return out

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self._db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def set_setting(self, key: str, value: Any) -> None:
        with self._db:
            self._db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )
