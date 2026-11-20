"""Analyser interface and the shapes every analyser returns.

The models mirror docs/contracts/analysis-result.schema.json and
docs/contracts/pad-change-result.schema.json (additionalProperties: false).
A local analyser must produce the same shapes, so visits, counting and review
never depend on which analyser answered.
"""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

ImageQuality = Literal["ok", "occluded", "too_dark", "camera_moved"]
PoopObserved = Literal["none", "on_pad", "beside_tray", "unsure"]
PadColor = Literal["white", "grey", "unknown"]
PeeWhere = Literal["none", "on_pad", "off_pad", "both"]


class AnalyserError(Exception):
    """The analyser could not answer (network, timeout, invalid answer, quota)."""


class AnalysisResult(BaseModel):
    """Answer for one dog visit (prompt pee_v1)."""

    model_config = ConfigDict(extra="forbid")

    pee: bool
    pee_confidence: float = Field(ge=0, le=1)
    poop: PoopObserved
    poop_confidence: float = Field(ge=0, le=1)
    pad_replaced: bool
    image_quality: ImageQuality
    night_mode: bool | None = None
    pee_where: PeeWhere | None = None  # pee_v4: on the pad, outside it (pad full), or both
    pad_stains: int | None = None  # pee_v5: separate stains in AFTER, clamped to 0..30
    reason: str = Field(max_length=140)

    @field_validator("pad_stains")
    @classmethod
    def _clamp_stains(cls, value: int | None) -> int | None:
        return None if value is None else max(0, min(value, 30))  # a wild count must not fail it


class PadChangeResult(BaseModel):
    """Answer for one person presence at the tray (prompt pad_change_v2)."""

    model_config = ConfigDict(extra="forbid")

    pad_replaced: bool
    pad_replaced_confidence: float = Field(ge=0, le=1)
    pad_color: PadColor | None = None
    poop_removed: bool
    image_quality: ImageQuality
    night_mode: bool | None = None
    reason: str = Field(max_length=140)


class Analyser(Protocol):
    """A deciding or shadow analyser. `id` is stored with every analysis."""

    id: str

    async def analyse(
        self, before: bytes, after: bytes, during: bytes | None, context: str | None = None
    ) -> AnalysisResult:
        """Analyse JPEG bytes of the before/after crops and the optional during frame.

        `context` is a short text about the pad (age, minutes since the last pee) that a prompt
        may use (pee_v5); an analyser that does not use it ignores it.

        Raises AnalyserError when no valid answer could be produced.
        """
        ...
