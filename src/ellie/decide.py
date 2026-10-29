"""Pure decision rules: turn an analyser answer and a threshold into an outcome."""

from typing import Literal

from ellie.analysers.base import AnalysisResult, PadChangeResult

VisitOutcome = Literal["pee", "no_pee", "uncertain", "unknown"]
PadOutcome = Literal["replaced", "not_replaced", "uncertain", "unknown"]


def decide_visit(result: AnalysisResult | None, threshold: float) -> VisitOutcome:
    """Decide a visit outcome; the confidence boundary is inclusive."""
    if result is None:
        return "unknown"
    if result.image_quality != "ok":
        return "uncertain"
    # A pad swap during the visit makes the before/after comparison unreliable.
    if result.pad_replaced:
        return "uncertain"
    if result.pee_confidence < threshold:
        return "uncertain"
    return "pee" if result.pee else "no_pee"


def decide_pad_change(result: PadChangeResult | None, threshold: float) -> PadOutcome:
    """Decide a pad change outcome; the confidence boundary is inclusive."""
    if result is None:
        return "unknown"
    if result.image_quality != "ok":
        return "uncertain"
    if result.pad_replaced_confidence < threshold:
        return "uncertain"
    return "replaced" if result.pad_replaced else "not_replaced"


def needs_phone_notification(outcome: str) -> bool:
    """True when a person has to look at the outcome."""
    return outcome in ("uncertain", "unknown")
