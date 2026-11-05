"""QA of 3.5.0 (ADR-0014): poop outcome, "poop present" and migration 6."""

import sqlite3

import pytest
from qa_common import tick_for

from ellie import store as store_module
from ellie.store import MIGRATIONS, Store


@pytest.mark.parametrize("kind", [[], {}, ["poop"], 7, None])
async def test_a_label_command_with_a_strange_kind_must_not_raise(make_rig, kind):
    """`kind` comes from MQTT: an unhashable value must be ignored, not kill the handler."""
    rig = make_rig()
    await rig.service.on_command("label", __import__("json").dumps(
        {"visit_id": "v_x", "kind": kind, "label": "poop"}).encode())  # fmt: skip
    await tick_for(rig, 1)


def _v5_database(path) -> None:
    db = sqlite3.connect(path)
    for script in MIGRATIONS[:5]:
        db.executescript(script)
    db.execute("PRAGMA user_version = 5")
    db.execute(
        "INSERT INTO visits (id, source, started_at, ended_at, state, auto_outcome,"
        " final_outcome, poop_observed) VALUES ('v1','auto','2026-11-04T19:45:29Z',"
        "'2026-11-04T19:46:36Z','DECIDED','pee','pee','beside_tray')"
    )
    db.execute("INSERT INTO labels (visit_id, label, channel) VALUES ('v1','pee','dashboard')")
    db.commit()
    db.close()


def test_migration_6_keeps_visits_and_labels_and_does_not_show_old_poops_as_present(tmp_path):
    path = str(tmp_path / "ellie.db")
    _v5_database(path)
    store = Store(path)
    (label,) = store.labels_for("v1")
    assert label["label"] == "pee" and label["kind"] == "pee"
    assert store.open_poops() == []  # old poops are marked cleared by the migration
    assert store.get_visit("v1")["final_poop"] == "poop"
    store.close()


def test_migration_6_must_be_retried_cleanly_after_a_crash_in_the_middle(tmp_path, monkeypatch):
    """A power cut or a failed statement half way through: the next start must not crash on
    "duplicate column name" (the app would loop on restart forever)."""
    path = str(tmp_path / "ellie.db")
    _v5_database(path)
    good = MIGRATIONS[5]
    monkeypatch.setattr(
        store_module, "MIGRATIONS", [*MIGRATIONS[:5], good + "\nSELECT * FROM table_that_is_gone;"]
    )
    with pytest.raises(sqlite3.Error):
        Store(path)
    monkeypatch.setattr(store_module, "MIGRATIONS", [*MIGRATIONS[:5], good])
    store = Store(path)  # must succeed
    assert store.schema_version() == 6
    store.close()
