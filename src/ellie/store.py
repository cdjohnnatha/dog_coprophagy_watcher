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
    # 3: the owner's answers; the latest label of a visit decides its final outcome
    """
    CREATE TABLE labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id TEXT NOT NULL REFERENCES visits(id),
        label TEXT NOT NULL CHECK (label IN ('pee', 'no_pee')),
        channel TEXT NOT NULL
            CHECK (channel IN ('notification', 'dashboard', 'manual_add', 'reject_last')),
        labelled_by TEXT,
        at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
        previous_final_outcome TEXT
            CHECK (previous_final_outcome IN ('pee', 'no_pee', 'unknown'))
    );
    CREATE INDEX idx_labels_visit ON labels(visit_id);
    """,
    # 4: person presences at the tray and the pad changes derived from them (US7)
    """
    CREATE TABLE presences (
        id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        state TEXT NOT NULL CHECK (state IN ('PRESENT', 'ANALYSING', 'DECIDED')),
        auto_outcome TEXT
            CHECK (auto_outcome IN ('replaced', 'not_replaced', 'uncertain', 'unknown')),
        final_outcome TEXT CHECK (final_outcome IN ('replaced', 'not_replaced', 'unknown')),
        poop_removed INTEGER,
        reason TEXT,
        analysis TEXT,
        cost_units TEXT,
        needs_review INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE pad_changes (
        id TEXT PRIMARY KEY,
        at TEXT NOT NULL,
        source TEXT NOT NULL CHECK (source IN ('detected', 'confirmed', 'visit', 'manual')),
        presence_id TEXT REFERENCES presences(id),
        visit_id TEXT REFERENCES visits(id),
        pad_color TEXT,
        undone_at TEXT
    );
    CREATE INDEX idx_pad_changes_at ON pad_changes(at);
    """,
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
_PRESENCE_COLUMNS = (
    "id",
    "started_at",
    "ended_at",
    "state",
    "auto_outcome",
    "final_outcome",
    "poop_removed",
    "reason",
    "analysis",
    "cost_units",
    "needs_review",
)
_PAD_CHANGE_COLUMNS = ("id", "at", "source", "presence_id", "visit_id", "pad_color", "undone_at")
_BOOL_FIELDS = ("person_present", "needs_review", "poop_removed")
_JSON_FIELDS = ("frigate_event_ids", "analysis", "cost_units")


def _encode(field: str, value: Any) -> Any:
    if field in _JSON_FIELDS:
        return json.dumps(value)
    if field in _BOOL_FIELDS and value is not None:
        return int(bool(value))
    return value


def _visit_from_row(row: sqlite3.Row) -> dict:
    visit = dict(row)
    for field in ("person_present", "needs_review"):
        if visit[field] is not None:
            visit[field] = bool(visit[field])
    if visit["frigate_event_ids"] is not None:
        visit["frigate_event_ids"] = json.loads(visit["frigate_event_ids"])
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

    def review_count(self) -> int:
        return int(
            self._db.execute("SELECT COUNT(*) FROM visits WHERE needs_review = 1").fetchone()[0]
        )

    def latest_visit(self) -> dict | None:
        row = self._db.execute("SELECT * FROM visits ORDER BY started_at DESC LIMIT 1").fetchone()
        return None if row is None else _visit_from_row(row)

    def usage_since(self, start_iso: str) -> dict:
        """Billed analysis calls (visits and presences) and tokens since `start_iso`."""
        row = self._db.execute(
            "SELECT COUNT(*) AS calls,"
            " COALESCE(SUM(json_extract(cu, '$.in')), 0) AS tokens_in,"
            " COALESCE(SUM(json_extract(cu, '$.out')), 0) AS tokens_out FROM ("
            " SELECT created_at AS at, cost_units AS cu FROM analyses WHERE cost_units IS NOT NULL"
            " UNION ALL"
            " SELECT started_at AS at, cost_units AS cu FROM presences WHERE cost_units IS NOT NULL"
            ") WHERE at >= ?",
            (start_iso,),
        ).fetchone()
        return {k: int(row[k]) for k in ("calls", "tokens_in", "tokens_out")}

    def _insert(self, table: str, columns: tuple[str, ...], row: dict) -> None:
        unknown = set(row) - set(columns)
        if unknown:
            raise ValueError(f"unknown {table} fields: {sorted(unknown)}")
        names = list(row)
        sql = f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})"
        with self._db:
            self._db.execute(sql, [_encode(c, row[c]) for c in names])

    def _update(self, table: str, columns: tuple[str, ...], key: str, fields: dict) -> None:
        unknown = set(fields) - set(columns) | ({"id"} & set(fields))
        if unknown or not fields:
            raise ValueError(f"unknown or immutable {table} fields: {sorted(unknown)}")
        assignments = ", ".join(f"{c} = ?" for c in fields)
        sql = f"UPDATE {table} SET {assignments} WHERE id = ?"
        with self._db:
            cur = self._db.execute(sql, [*(_encode(c, v) for c, v in fields.items()), key])
        if cur.rowcount == 0:
            raise KeyError(key)

    @staticmethod
    def _decoded(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        for field in _BOOL_FIELDS:
            if item.get(field) is not None:
                item[field] = bool(item[field])
        for field in _JSON_FIELDS:
            if item.get(field) is not None:
                item[field] = json.loads(item[field])
        return item

    def insert_presence(self, presence: dict) -> None:
        self._insert("presences", _PRESENCE_COLUMNS, presence)

    def update_presence(self, presence_id: str, **fields: Any) -> None:
        self._update("presences", _PRESENCE_COLUMNS, presence_id, fields)

    def get_presence(self, presence_id: str) -> dict | None:
        row = self._db.execute("SELECT * FROM presences WHERE id = ?", (presence_id,)).fetchone()
        return self._decoded(row)

    def analysing_presences(self) -> list[dict]:
        rows = self._db.execute("SELECT * FROM presences WHERE state IN ('PRESENT', 'ANALYSING')")
        return [self._decoded(r) for r in rows]

    def insert_pad_change(self, change: dict) -> None:
        self._insert("pad_changes", _PAD_CHANGE_COLUMNS, change)

    def pad_changes(self, limit: int = 10) -> list[dict]:
        """Newest first, undone ones included (the history keeps both, US7-6)."""
        rows = self._db.execute("SELECT * FROM pad_changes ORDER BY at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def current_pad_change(self) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM pad_changes WHERE undone_at IS NULL ORDER BY at DESC LIMIT 1"
        ).fetchone()
        return None if row is None else dict(row)

    def undo_current_pad_change(self, undone_at: str) -> dict | None:
        current = self.current_pad_change()
        if current is None:
            return None
        self._update("pad_changes", _PAD_CHANGE_COLUMNS, current["id"], {"undone_at": undone_at})
        return {**current, "undone_at": undone_at}

    def pees_since(self, start_iso: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM visits WHERE final_outcome = 'pee' AND started_at >= ?",
            (start_iso,),
        ).fetchone()
        return int(row["n"])

    def insert_label(
        self,
        visit_id: str,
        label: str,
        channel: str,
        by: str | None = None,
        previous_final_outcome: str | None = None,
    ) -> int:
        with self._db:
            cur = self._db.execute(
                "INSERT INTO labels (visit_id, label, channel, labelled_by, previous_final_outcome)"
                " VALUES (?, ?, ?, ?, ?)",
                (visit_id, label, channel, by, previous_final_outcome),
            )
        return int(cur.lastrowid)

    def labels_for(self, visit_id: str) -> list[dict]:
        rows = self._db.execute("SELECT * FROM labels WHERE visit_id = ? ORDER BY id", (visit_id,))
        return [dict(r) for r in rows]

    def corrected_ids(self, start_iso: str) -> set[str]:
        """Visits since `start_iso` whose latest label differs from the automatic outcome."""
        rows = self._db.execute(
            "SELECT v.id FROM visits v JOIN labels l ON l.visit_id = v.id"
            " AND l.id = (SELECT MAX(id) FROM labels WHERE visit_id = v.id)"
            " WHERE v.started_at >= ? AND (v.auto_outcome IS NULL OR"
            " (v.auto_outcome NOT IN ('uncertain', 'unknown') AND v.auto_outcome != l.label))",
            (start_iso,),
        )
        return {r["id"] for r in rows}

    def analysis_label_pairs(self, since_iso: str) -> list[dict]:
        """One row per (labelled visit, analyser): the analyser's latest outcome and the owner's
        latest label. Labelled visits without any analysis (manual entries) have analyser None."""
        rows = self._db.execute(
            "SELECT v.id AS visit_id, v.source, v.auto_outcome, l.label,"
            " a.analyser, a.role, a.outcome"
            " FROM visits v"
            " JOIN labels l ON l.visit_id = v.id"
            "  AND l.id = (SELECT MAX(id) FROM labels WHERE visit_id = v.id)"
            " LEFT JOIN analyses a ON a.visit_id = v.id"
            "  AND a.error IS NULL"
            "  AND a.id = (SELECT MAX(id) FROM analyses WHERE visit_id = v.id"
            "              AND analyser = a.analyser AND error IS NULL)"
            " WHERE v.started_at >= ? ORDER BY v.started_at",
            (since_iso,),
        )
        return [dict(r) for r in rows]

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
