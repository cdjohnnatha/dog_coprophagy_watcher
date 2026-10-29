"""Analyser interface and the shapes every analyser returns.

The models mirror docs/contracts/analysis-result.schema.json and
docs/contracts/pad-change-result.schema.json (additionalProperties: false).
A local analyser must produce the same shapes, so visits, counting and review
never depend on which analyser answered.
"""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

ImageQuality = Literal["ok", "occluded", "too_dark", "camera_moved"]
PoopObserved = Literal["none", "on_pad", "beside_tray", "unsure"]
PadColor = Literal["white", "grey", "unknown"]


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
    reason: str = Field(max_length=140)


class PadChangeResult(BaseModel):
    """Answer for one person presence at the tray (prompt pad_change_v1)."""

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
        self, before: bytes, after: bytes, during: bytes | None
    ) -> AnalysisResult:
        """Analyse JPEG bytes of the before/after crops and the optional during frame.

        Raises AnalyserError when no valid answer could be produced.
        """
        ...
