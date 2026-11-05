import sqlite3

import pytest

from ellie.store import MIGRATIONS, Store


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "ellie.db"))
    yield s
    s.close()


def visit(vid="v_20260101_100000", **over):
    base = {
        "id": vid,
        "source": "auto",
        "started_at": "2026-01-01T10:00:00Z",
        "state": "OPEN",
        "entries": 1,
        "person_present": False,
        "frigate_event_ids": ["a", "b"],
    }
    return {**base, **over}


def test_migration_creates_tables(store):
    assert store.schema_version() == len(MIGRATIONS) == 6
    names = {r[0] for r in store._db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"visits", "analyses", "settings"} <= names


def test_migration_twice_is_noop(tmp_path):
    path = str(tmp_path / "x.db")
    Store(path).close()
    s = Store(path)
    assert s.schema_version() == 6
    s.close()


@pytest.mark.parametrize(
    "bad",
    [{"source": "x"}, {"entries": 0}, {"state": "NOPE"}, {"final_outcome": "uncertain"}],
)
def test_check_constraints(store, bad):
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_visit(visit(**bad))


def test_roundtrip(store):
    store.insert_visit(visit(person_present=True))
    got = store.get_visit("v_20260101_100000")
    assert got["frigate_event_ids"] == ["a", "b"]
    assert got["person_present"] is True
    assert got["needs_review"] is False
    assert got["poop_observed"] == "none"
    assert got["auto_outcome"] is None
    assert store.get_visit("nope") is None


def test_update_visit(store):
    store.insert_visit(visit())
    store.update_visit("v_20260101_100000", state="DECIDED", needs_review=True, entries=2)
    got = store.get_visit("v_20260101_100000")
    assert (got["state"], got["needs_review"], got["entries"]) == ("DECIDED", True, 2)
    with pytest.raises(KeyError):
        store.update_visit("missing", state="OPEN")
    with pytest.raises(ValueError):
        store.update_visit("v_20260101_100000", bogus=1)


def test_visits_started_between(store):
    for i, t in enumerate(["09:59:59", "10:00:00", "10:30:00", "11:00:00"]):
        store.insert_visit(visit(f"v{i}", started_at=f"2026-01-01T{t}Z"))
    got = store.visits_started_between("2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z")
    assert [v["id"] for v in got] == ["v1", "v2"]


def test_state_queries(store):
    for vid, state, review in [
        ("a", "OPEN", 0),
        ("b", "CLOSING", 0),
        ("c", "CAPTURING", 1),
        ("d", "ANALYSING", 1),
        ("e", "DECIDED", 0),
    ]:
        store.insert_visit(visit(vid, state=state, needs_review=bool(review)))
    assert [v["id"] for v in store.open_visits()] == ["a", "b", "c"]
    assert [v["id"] for v in store.analysing_visits()] == ["d"]
    store.update_visit("c", started_at="2026-01-02T00:00:00Z")
    assert [v["id"] for v in store.review_queue()] == ["c", "d"]
    assert [v["id"] for v in store.review_queue(limit=1)] == ["c"]


def test_analyses(store):
    store.insert_visit(visit())
    aid = store.insert_analysis(
        "v_20260101_100000",
        "claude",
        "deciding",
        {"pee": True},
        "pee",
        900,
        None,
        {"in": 1128, "out": 80},
    )
    store.insert_analysis(
        "v_20260101_100000", "local", "shadow", None, "unknown", None, "boom", None
    )
    rows = store.analyses_for("v_20260101_100000")
    assert rows[0]["id"] == aid
    assert rows[0]["result"] == {"pee": True}
    assert rows[0]["cost_units"] == {"in": 1128, "out": 80}
    assert rows[1]["result"] is None and rows[1]["error"] == "boom"
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_analysis("ghost", "claude", "deciding", None, "pee", None, None, None)
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_analysis("v_20260101_100000", "c", "other", None, "pee", None, None, None)


def test_settings(store):
    assert store.get_setting("k") is None
    assert store.get_setting("k", 5) == 5
    for value in (1.5, True, "text", {"a": [1]}):
        store.set_setting("k", value)
        assert store.get_setting("k") == value


def test_reopen_keeps_data(tmp_path):
    path = str(tmp_path / "r.db")
    s = Store(path)
    s.insert_visit(visit())
    s.set_setting("k", 1)
    s.close()
    s = Store(path)
    assert s.get_visit("v_20260101_100000") is not None
    assert s.get_setting("k") == 1
    s.close()


def test_migration_6_backfills_old_visits_and_keeps_old_labels(tmp_path) -> None:
    """A database at version 5 gets final_poop from poop_observed; old poops are not "present"."""
    import sqlite3

    from ellie.store import MIGRATIONS

    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    for script in MIGRATIONS[:5]:
        db.executescript(script)
    db.execute("PRAGMA user_version = 5")

    def add(visit_id, poop, state="DECIDED", source="auto", outcome="no_pee"):
        db.execute(
            "INSERT INTO visits (id, source, started_at, ended_at, state, auto_outcome,"
            " final_outcome, poop_observed) VALUES (?, ?, '2026-11-01T09:00:00Z',"
            " '2026-11-01T09:01:00Z', ?, ?, ?, ?)",
            (visit_id, source, state, outcome, outcome, poop),
        )

    add("a", "beside_tray")
    add("b", "none")
    add("c", "unsure")
    add("d", "beside_tray", source="manual", state="LABELLED")
    add("e", "beside_tray", state="OPEN", outcome=None)
    db.execute("INSERT INTO labels (visit_id, label, channel) VALUES ('a', 'pee', 'dashboard')")
    db.commit()
    db.close()

    store = Store(path)
    assert store.schema_version() == 6
    got = {v["id"]: v for v in store.visits_started_between("2026", "2027")}
    assert (got["a"]["auto_poop"], got["a"]["final_poop"]) == ("poop", "poop")
    assert got["a"]["poop_cleared_by"] == "migration" and store.open_poops() == []
    assert (got["b"]["auto_poop"], got["b"]["final_poop"]) == ("no_poop", "no_poop")
    assert (got["c"]["auto_poop"], got["c"]["final_poop"]) == ("uncertain", "unknown")
    assert store.poop_review_open() == []  # an old doubt is not asked retroactively
    assert got["d"]["final_poop"] is None and got["e"]["final_poop"] is None
    assert store.poops_between("2026-11-01T00:00:00Z", "2026-11-02T00:00:00Z") == 1
    (label,) = store.labels_for("a")
    assert label["kind"] == "pee" and label["label"] == "pee"
    store.insert_label("a", "poop", "dashboard", None, "poop", kind="poop")
    assert [x["label"] for x in store.labels_for("a", "poop")] == ["poop"]
    assert [x["label"] for x in store.labels_for("a")] == ["pee"]
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_label("a", "poop", "dashboard", None, None, kind="pee-ish")
    store.close()
