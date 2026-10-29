"""Builds the one JSON object published to `ellie/state` (its shape is documented in discovery.py)."""

from datetime import datetime, timedelta
from typing import Any

from ellie.accuracy import summarize
from ellie.counting import LOCAL_TZ, Counting
from ellie.evidence import Evidence
from ellie.store import Store
from ellie.visits import iso

# gemini-3.5-flash-lite, USD per 1M tokens (ADR-0003). Re-check when the model changes.
PRICE_IN_USD_PER_M = 0.30
PRICE_OUT_USD_PER_M = 2.50
BEGINNING = "1970-01-01T00:00:00Z"


def estimate_cost_eur(tokens_in: int, tokens_out: int, usd_to_eur: float) -> float:
    usd = (tokens_in * PRICE_IN_USD_PER_M + tokens_out * PRICE_OUT_USD_PER_M) / 1_000_000
    return round(usd * usd_to_eur, 4)


def local_text(ts: str | None) -> str | None:
    """UTC ISO string -> local 'YYYY-MM-DD HH:MM:SS' for display."""
    if not ts:
        return None
    return datetime.fromisoformat(ts).astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _url_if_exists(evidence: Evidence, visit_id: str, name: str) -> str | None:
    path = evidence.visit_dir(visit_id) / name
    return evidence.local_url(path) if path.exists() else None


def build_state(
    *,
    store: Store,
    evidence: Evidence,
    active_visit: dict | None,
    cfg: dict[str, Any],
    now: datetime,
    usd_to_eur: float,
    pad: dict | None = None,
) -> dict[str, Any]:
    counting = Counting(store)
    summary = counting.summary(now)

    last_pee = counting.last_pee(now)
    last_visit = store.latest_visit()
    queue = store.review_queue(200)
    month_start = now.astimezone(LOCAL_TZ).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month = store.usage_since(iso(month_start))
    total = store.usage_since(BEGINNING)
    window_days = 14
    accuracy = summarize(
        store.analysis_label_pairs(iso(now - timedelta(days=window_days))),
        cfg["deciding_analyser"],
        window_days,
    )
    corrected = store.corrected_ids(iso(now - timedelta(days=90)))

    return {
        "pee_today": summary["pees"],
        "visits_today": summary["visits"],
        "uncertain_open": summary["uncertain_open"],
        "at_tray": bool(active_visit and active_visit["state"] in ("OPEN", "CLOSING")),
        "last_pee": None
        if last_pee is None
        else {
            "at": last_pee["started_at"],
            "visit_id": last_pee["id"],
            "image": _url_if_exists(evidence, last_pee["id"], "after.jpg"),
            "reason": last_pee["reason"],
        },
        "last_visit": None
        if last_visit is None
        else {
            "visit_id": last_visit["id"],
            "auto_outcome": last_visit["auto_outcome"],
            "final_outcome": last_visit["final_outcome"],
            "started_at": last_visit["started_at"],
            "ended_at": last_visit["ended_at"],
            "reason": last_visit["reason"],
            "poop_observed": last_visit["poop_observed"],
            "needs_review": last_visit["needs_review"],
            "deciding_analyser": last_visit["deciding_analyser"],
        },
        "review_queue": {
            "count": store.review_count(),
            "items": [
                {
                    "visit_id": v["id"],
                    "time": local_text(v["started_at"]),
                    "auto_outcome": v["auto_outcome"],
                    "reason": v["reason"],
                    "before": _url_if_exists(evidence, v["id"], "before.jpg"),
                    "after": _url_if_exists(evidence, v["id"], "after.jpg"),
                }
                for v in queue[:20]
            ],
        },
        "daily_history": {"days": counting.history_days(now, days=90, corrected_ids=corrected)},
        "accuracy": accuracy,
        "analysis_calls": {
            "calls": month["calls"],
            "tokens_in": month["tokens_in"],
            "tokens_out": month["tokens_out"],
            "estimated_cost_eur": estimate_cost_eur(
                month["tokens_in"], month["tokens_out"], usd_to_eur
            ),
            "estimated_cost_eur_total": estimate_cost_eur(
                total["tokens_in"], total["tokens_out"], usd_to_eur
            ),
        },
        "settings": dict(cfg),
        "pad": pad,
    }
