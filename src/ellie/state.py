"""Builds the one JSON object published to `ellie/state` (its shape is documented in discovery.py)."""

from datetime import datetime, timedelta
from typing import Any

from ellie.accuracy import summarize
from ellie.counting import LOCAL_TZ, Counting
from ellie.evidence import Evidence
from ellie.poop_habits import compute as compute_poop_habits
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


def _poop_view(evidence: Evidence, visit: dict | None) -> dict | None:
    if visit is None:
        return None
    return {
        "at": visit["started_at"],
        "visit_id": visit["id"],
        "where": visit["poop_observed"]
        if visit["poop_observed"] in ("on_pad", "beside_tray")
        else None,
        "image": _url_if_exists(evidence, visit["id"], "after.jpg"),
        "reason": visit["reason"],
        "cleared_at": visit["poop_cleared_at"],
        "cleared_by": visit["poop_cleared_by"],
    }


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
    open_poops = store.open_poops()
    last_poop = counting.last_poop(now)
    poop_habits = compute_poop_habits(
        [datetime.fromisoformat(t) for t in store.counted_poop_times()], now
    )

    return {
        "pee_today": summary["pees"],
        "poop_today": summary["poops"],
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
        "last_poop": _poop_view(evidence, last_poop),
        "poop_habits": poop_habits,
        "poop_present": {
            "on": bool(open_poops),
            "count": len(open_poops),
            "since": open_poops[0]["started_at"] if open_poops else None,
            **{
                k: (_poop_view(evidence, open_poops[0]) or {}).get(k) if open_poops else None
                for k in ("visit_id", "where", "image")
            },
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
            "auto_poop": last_visit["auto_poop"],
            "final_poop": last_visit["final_poop"],
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
                    "final_outcome": v["final_outcome"],
                    "pee_question": bool(v["needs_review"]),
                    "poop_question": v["auto_poop"] == "uncertain" and v["final_poop"] is None,
                    "poop_observed": v["poop_observed"],
                    "before": _url_if_exists(evidence, v["id"], "before.jpg"),
                    "after": _url_if_exists(evidence, v["id"], "after.jpg"),
                }
                for v in queue[:20]
            ],
        },
        "daily_history": {"days": counting.history_days(now, days=90, corrected_ids=corrected)},
        "poops_by_day": counting.poop_history_days(now, days=90),
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
