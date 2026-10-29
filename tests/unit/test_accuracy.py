from ellie.accuracy import summarize


def row(vid, outcome, label, analyser="a", source="auto", auto="pee", role="deciding"):
    return {
        "visit_id": vid,
        "source": source,
        "auto_outcome": auto,
        "label": label,
        "analyser": analyser,
        "role": role,
        "outcome": outcome,
    }


def manual(vid):
    return row(vid, None, "pee", analyser=None, source="manual", auto=None, role=None)


def test_empty():
    r = summarize([], "a")
    assert r["pct"] is None and r["analysers"] == {} and r["window_days"] == 14
    assert r["period"] == {
        "visits": 0,
        "auto_pees": 0,
        "confirmations": 0,
        "rejections": 0,
        "manual_additions": 0,
        "missed": 0,
    }


def test_perfect():
    r = summarize([row(1, "pee", "pee"), row(2, "no_pee", "no_pee", auto="no_pee")], "a")
    assert r["pct"] == 100.0
    assert r["analysers"]["a"] == {"labelled": 2, "agree": 2, "accuracy": 100.0}


def test_one_disagreement():
    r = summarize([row(1, "pee", "pee"), row(2, "pee", "no_pee")], "a")
    assert r["pct"] == 50.0
    assert r["period"]["confirmations"] == 1 and r["period"]["rejections"] == 1


def test_uncertain_unknown_labelled_not_agree():
    r = summarize(
        [
            row(1, "uncertain", "pee", auto="uncertain"),
            row(2, "unknown", "no_pee", auto="unknown"),
            row(3, "pee", "pee"),
        ],
        "a",
    )
    assert r["analysers"]["a"] == {"labelled": 3, "agree": 1, "accuracy": 33.3}


def test_shadow_vs_deciding():
    rows = [
        row(1, "pee", "pee"),
        row(2, "pee", "pee"),
        row(1, "no_pee", "pee", analyser="s", role="shadow"),
        row(2, "pee", "pee", analyser="s", role="shadow"),
    ]
    r = summarize(rows, "a")
    assert r["analysers"]["a"]["accuracy"] == 100.0
    assert r["analysers"]["s"]["accuracy"] == 50.0
    assert r["pct"] == 100.0


def test_manual_lowers_pct():
    r = summarize([row(1, "pee", "pee"), manual(2)], "a")
    assert r["pct"] == 50.0
    assert r["period"]["manual_additions"] == 1 and r["period"]["visits"] == 2
    assert r["analysers"]["a"]["labelled"] == 1


def test_period_counters_mixed():
    rows = [
        row(1, "pee", "pee"),
        row(2, "pee", "no_pee"),
        row(3, "no_pee", "pee", auto="no_pee"),
        row(4, "no_pee", "no_pee", auto="no_pee"),
        row(5, "uncertain", "pee", auto="uncertain"),
        manual(6),
    ]
    assert summarize(rows, "a")["period"] == {
        "visits": 6,
        "auto_pees": 2,
        "confirmations": 1,
        "rejections": 1,
        "manual_additions": 1,
        "missed": 1,
    }


def test_duplicates_and_multi_analyser_count_once():
    rows = [
        row(1, "pee", "pee"),
        row(1, "pee", "pee"),
        row(1, "pee", "pee", analyser="s", role="shadow"),
    ]
    r = summarize(rows, "a")
    assert r["analysers"]["a"]["labelled"] == 1
    assert r["period"]["visits"] == 1 and r["period"]["auto_pees"] == 1


def test_deciding_without_rows():
    r = summarize([manual(1)], "a")
    assert r["pct"] == 0.0 and r["analysers"] == {}
    assert summarize([row(1, "pee", "pee", analyser="s")], "a")["pct"] is None


def test_rounding():
    rows = [row(1, "pee", "pee"), row(2, "pee", "pee"), row(3, "pee", "no_pee")]
    assert summarize(rows, "a")["pct"] == 66.7
