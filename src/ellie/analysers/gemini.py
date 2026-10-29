"""Gemini analyser, through the Interactions API (ADR-0003).

Calls are stateless (`store=False`: otherwise Google keeps each interaction for 55 days),
use JSON-schema output, `high` resolution for the before/after crops, `low` for the posture
frame, and thinking `minimal` (thinking tokens are billed as output). 503/429 and timeouts
are retried with backoff inside a time budget, because the real API answered 503 "high
demand" in the first smoke test; every other error is final.
"""

import asyncio
import base64
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from importlib import resources
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from ellie.analysers.base import AnalyserError, AnalysisResult, PadChangeResult

log = logging.getLogger(__name__)

ANALYSER_ID = "gemini:pee_v1"
PAD_ANALYSER_ID = "gemini:pad_change_v1"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_RETRY_DELAYS = (15.0, 60.0, 180.0)

_IMAGE_QUALITY = ["ok", "occluded", "too_dark", "camera_moved"]
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pee": {"type": "boolean", "description": "A NEW pee mark appears in AFTER (old stains do not count)."},
        "pee_confidence": {"type": "number", "description": "0-1, confidence in the `pee` answer."},
        "poop": {"type": "string", "enum": ["none", "on_pad", "beside_tray", "unsure"]},
        "poop_confidence": {"type": "number", "description": "0-1, confidence in the `poop` answer."},
        "pad_replaced": {"type": "boolean", "description": "The pad itself was changed or the tray cleaned."},
        "image_quality": {"type": "string", "enum": _IMAGE_QUALITY},
        "night_mode": {"type": "boolean", "description": "Images are infrared/black-and-white."},
        "reason": {"type": "string", "description": "Short explanation in Brazilian Portuguese, at most 140 characters."},
    },
    "required": [
        "pee", "pee_confidence", "poop", "poop_confidence",
        "pad_replaced", "image_quality", "reason",
    ],
}  # fmt: skip
PAD_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pad_replaced": {"type": "boolean", "description": "AFTER shows a different pad than BEFORE."},
        "pad_replaced_confidence": {"type": "number", "description": "0-1, confidence in `pad_replaced`."},
        "pad_color": {"type": "string", "enum": ["white", "grey", "unknown"]},
        "poop_removed": {"type": "boolean", "description": "Poop in BEFORE is gone in AFTER."},
        "image_quality": {"type": "string", "enum": _IMAGE_QUALITY},
        "night_mode": {"type": "boolean"},
        "reason": {"type": "string", "description": "Short explanation in Brazilian Portuguese, at most 140 characters."},
    },
    "required": ["pad_replaced", "pad_replaced_confidence", "poop_removed", "image_quality", "reason"],
}  # fmt: skip


def load_prompt(name: str) -> str:
    return (resources.files("ellie") / "prompts" / f"{name}.md").read_text(encoding="utf-8")


def _text(label: str) -> dict[str, Any]:
    return {"type": "text", "text": label}


def _image(data: bytes, resolution: str) -> dict[str, Any]:
    return {
        "type": "image",
        "mime_type": "image/jpeg",
        "data": base64.b64encode(data).decode("ascii"),
        "resolution": resolution,
    }


def classify_error(exc: Exception) -> tuple[bool, str]:
    """(retryable, short description). Never includes the exception text: it may echo input."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS, f"HTTP {status}"
    if isinstance(exc, TimeoutError | httpx.TimeoutException):
        return True, "timeout"
    if (
        isinstance(exc, httpx.TransportError | ConnectionError)
        or type(exc).__name__ == "NoResponseError"
    ):
        return True, "network error"
    return False, type(exc).__name__


class GeminiAnalyser:
    """Implements the Analyser protocol; `analyse_pad` answers the pad change question."""

    id = ANALYSER_ID

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        client: Any = None,
        fallback_model: str | None = None,
        timeout_s: float = 90.0,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
        budget_s: float = 300.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if client is None:
            from google import genai  # imported late: tests inject a fake client

            client = genai.Client(api_key=api_key)
        self._client = client
        self._model = model
        self._fallback = fallback_model
        self._timeout_s = timeout_s
        self._delays = tuple(retry_delays)
        self._budget_s = budget_s
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = (
            asyncio.Lock()
        )  # one call at a time: the quota is shared and usage is read after
        self._pee_prompt = load_prompt("pee_v1")
        self._pad_prompt = load_prompt("pad_change_v1")
        self.last_model: str | None = None
        self.last_usage: dict[str, int] = {"in": 0, "out": 0}

    async def analyse(self, before: bytes, after: bytes, during: bytes | None) -> AnalysisResult:
        parts = [_text("BEFORE:"), _image(before, "high"), _text("AFTER:"), _image(after, "high")]
        if during:
            parts += [_text("DURING:"), _image(during, "low")]
        answer = await self._generate(self._pee_prompt, parts, RESPONSE_SCHEMA)
        return self._parse(AnalysisResult, answer)

    async def analyse_pad(self, before: bytes, after: bytes) -> PadChangeResult:
        parts = [_text("BEFORE:"), _image(before, "high"), _text("AFTER:"), _image(after, "high")]
        answer = await self._generate(self._pad_prompt, parts, PAD_RESPONSE_SCHEMA)
        return self._parse(PadChangeResult, answer)

    def _model_for(self, attempt: int) -> str:
        return self._fallback if self._fallback and attempt % 2 == 1 else self._model

    async def _generate(self, system: str, parts: list[dict], schema: dict) -> str:
        started = self._monotonic()
        attempts = len(self._delays) + 1
        why = "no attempt"
        async with self._lock:
            for attempt in range(attempts):
                model = self._model_for(attempt)
                try:
                    async with asyncio.timeout(self._timeout_s):
                        interaction = await self._client.aio.interactions.create(
                            model=model,
                            input=parts,
                            system_instruction=system,
                            response_format={
                                "type": "text",
                                "mime_type": "application/json",
                                "schema": schema,
                            },
                            generation_config={"thinking_level": "minimal"},
                            store=False,
                            timeout=self._timeout_s,
                        )
                except Exception as exc:  # noqa: BLE001 - the SDK may raise anything; classified below
                    retryable, why = classify_error(exc)
                    if not retryable:
                        raise AnalyserError(f"gemini request failed: {why}") from None
                    if attempt == attempts - 1:
                        break
                    delay = self._delays[attempt]
                    if self._monotonic() - started + delay > self._budget_s:
                        break
                    log.warning("gemini %s on %s, retrying in %.0f s", why, model, delay)
                    await self._sleep(delay)
                    continue
                self.last_model = model
                usage = getattr(interaction, "usage", None)
                self.last_usage = {
                    "in": int(getattr(usage, "total_input_tokens", 0) or 0),
                    "out": int(getattr(usage, "total_output_tokens", 0) or 0),
                }
                return getattr(interaction, "output_text", None) or ""
        raise AnalyserError(f"gemini unavailable: {why}")

    @staticmethod
    def _parse[M: BaseModel](model_cls: type[M], answer: str) -> M:
        if not answer.strip():
            raise AnalyserError("gemini returned an empty answer")
        try:
            return model_cls.model_validate(json.loads(answer))
        except (json.JSONDecodeError, ValidationError):
            raise AnalyserError(
                "gemini returned an answer that does not match the schema"
            ) from None
