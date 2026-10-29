"""Accuracy summary of analysers against the owner's labels (pure)."""

from collections.abc import Iterable


def _pct(agree: int, labelled: int) -> float | None:
    return round(100 * agree / labelled, 1) if labelled else None


def summarize(rows: Iterable[dict], deciding_analyser: str, window_days: int = 14) -> dict:
    """Summarise per-analyser accuracy and period totals from label/analyser rows."""
    pairs: dict[tuple[int | str, str], bool] = {}
    visits: dict[int | str, dict] = {}
    for row in rows:
        visits.setdefault(row["visit_id"], row)
        analyser = row.get("analyser")
        if analyser is not None:
            pairs[(row["visit_id"], analyser)] = row.get("outcome") == row["label"]

    counts: dict[str, list[int]] = {}
    for (_, analyser), agrees in pairs.items():
        c = counts.setdefault(analyser, [0, 0])
        c[0] += 1
        c[1] += agrees
    analysers = {
        a: {"labelled": c[0], "agree": c[1], "accuracy": _pct(c[1], c[0])}
        for a, c in counts.items()
    }

    deciding = counts.get(deciding_analyser, [0, 0])
    manual = sum(1 for v in visits.values() if v["source"] == "manual")
    period = {
        "visits": len(visits),
        "auto_pees": 0,
        "confirmations": 0,
        "rejections": 0,
        "manual_additions": manual,
        "missed": 0,
    }
    for v in visits.values():
        if v["source"] != "auto":
            continue
        if v.get("auto_outcome") == "pee":
            period["auto_pees"] += 1
            period["confirmations" if v["label"] == "pee" else "rejections"] += 1
        elif v.get("auto_outcome") == "no_pee" and v["label"] == "pee":
            period["missed"] += 1

    return {
        "pct": _pct(deciding[1], deciding[0] + manual),
        "window_days": window_days,
        "analysers": analysers,
        "period": period,
    }
