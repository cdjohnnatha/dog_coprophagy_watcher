"""Unit tests for the pure decision rules."""

import pytest

from ellie.analysers.base import AnalysisResult, PadChangeResult
from ellie.decide import (
    decide_pad_change,
    decide_poop,
    decide_visit,
    needs_phone_notification,
)

T = 0.80


def visit(
    pee: bool = True,
    conf: float = 0.95,
    quality: str = "ok",
    pad_replaced: bool = False,
) -> AnalysisResult:
    return AnalysisResult(
        pee=pee,
        pee_confidence=conf,
        poop="none",
        poop_confidence=0.9,
        pad_replaced=pad_replaced,
        image_quality=quality,  # type: ignore[arg-type]
        reason="test",
    )


def pad(replaced: bool = True, conf: float = 0.95, quality: str = "ok") -> PadChangeResult:
    return PadChangeResult(
        pad_replaced=replaced,
        pad_replaced_confidence=conf,
        poop_removed=False,
        image_quality=quality,  # type: ignore[arg-type]
        reason="test",
    )


@pytest.mark.parametrize(
    ("result", "threshold", "expected"),
    [
        (visit(True, 0.95), T, "pee"),
        (visit(False, 0.95), T, "no_pee"),
        (visit(True, 0.5), T, "uncertain"),
        (visit(False, 0.5), T, "uncertain"),
        (visit(True, 0.80), T, "pee"),
        (visit(False, 0.80), T, "no_pee"),
        (visit(True, 0.7999), T, "uncertain"),
        (visit(False, 0.7999), T, "uncertain"),
        (visit(True, 1.0, "occluded"), T, "uncertain"),
        (visit(True, 1.0, "too_dark"), T, "uncertain"),
        (visit(False, 1.0, "camera_moved"), T, "uncertain"),
        (visit(True, 1.0, pad_replaced=True), T, "uncertain"),
        (visit(False, 1.0, pad_replaced=True), T, "uncertain"),
        (None, T, "unknown"),
        (visit(True, 0.6), 0.5, "pee"),
        (visit(True, 0.95), 0.99, "uncertain"),
        (visit(False, 0.99), 0.99, "no_pee"),
    ],
)
def test_decide_visit(result: AnalysisResult | None, threshold: float, expected: str) -> None:
    assert decide_visit(result, threshold) == expected


@pytest.mark.parametrize(
    ("result", "threshold", "expected"),
    [
        (pad(True, 0.95), T, "replaced"),
        (pad(False, 0.95), T, "not_replaced"),
        (pad(True, 0.5), T, "uncertain"),
        (pad(False, 0.5), T, "uncertain"),
        (pad(True, 0.80), T, "replaced"),
        (pad(False, 0.80), T, "not_replaced"),
        (pad(True, 0.7999), T, "uncertain"),
        (pad(False, 0.7999), T, "uncertain"),
        (pad(True, 1.0, "occluded"), T, "uncertain"),
        (pad(True, 1.0, "too_dark"), T, "uncertain"),
        (pad(False, 1.0, "camera_moved"), T, "uncertain"),
        (None, T, "unknown"),
        (pad(True, 0.6), 0.5, "replaced"),
        (pad(True, 0.95), 0.99, "uncertain"),
    ],
)
def test_decide_pad_change(result: PadChangeResult | None, threshold: float, expected: str) -> None:
    assert decide_pad_change(result, threshold) == expected


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("uncertain", True),
        ("unknown", True),
        ("pee", False),
        ("no_pee", False),
        ("replaced", False),
        ("not_replaced", False),
    ],
)
def test_needs_phone_notification(outcome: str, expected: bool) -> None:
    assert needs_phone_notification(outcome) is expected


def poop_result(poop="none", conf=0.9, quality="ok", pad_replaced=False) -> AnalysisResult:
    return AnalysisResult(
        pee=False,
        pee_confidence=0.95,
        poop=poop,  # type: ignore[arg-type]
        poop_confidence=conf,
        pad_replaced=pad_replaced,
        image_quality=quality,  # type: ignore[arg-type]
        reason="test",
    )


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (poop_result("on_pad"), "poop"),
        (poop_result("beside_tray"), "poop"),
        (poop_result("none"), "no_poop"),
        (poop_result("unsure"), "uncertain"),
        (poop_result("beside_tray", conf=0.79), "uncertain"),
        (poop_result("none", conf=0.8), "no_poop"),  # the boundary is inclusive
        (poop_result("none", conf=0.5), "uncertain"),
        (poop_result("beside_tray", quality="occluded"), "uncertain"),
        (poop_result("on_pad", pad_replaced=True), "uncertain"),
        (poop_result("beside_tray", pad_replaced=True), "poop"),  # the floor is not the pad
        (None, "unknown"),
    ],
)
def test_decide_poop(result, expected) -> None:
    assert decide_poop(result, T) == expected


def test_the_pee_and_the_poop_decisions_are_independent() -> None:
    result = AnalysisResult(
        pee=True,
        pee_confidence=0.5,
        poop="beside_tray",
        poop_confidence=0.95,
        pad_replaced=False,
        image_quality="ok",
        reason="test",
    )
    assert decide_visit(result, T) == "uncertain" and decide_poop(result, T) == "poop"
