import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from ellie.analysers.base import AnalyserError
from ellie.analysers.gemini import (
    PAD_RESPONSE_SCHEMA,
    RESPONSE_SCHEMA,
    GeminiAnalyser,
    classify_error,
)
from helpers import make_jpeg

PEE_JSON = {
    "pee": True,
    "pee_confidence": 0.93,
    "poop": "none",
    "poop_confidence": 0.9,
    "pad_replaced": False,
    "image_quality": "ok",
    "reason": "mancha nova e úmida no centro do tapete",
}
PAD_JSON = {
    "pad_replaced": True,
    "pad_replaced_confidence": 0.95,
    "poop_removed": False,
    "image_quality": "ok",
    "reason": "tapete novo: manchas antigas sumiram",
}


class StatusError(Exception):
    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code


def answer(payload: dict | str, tokens_in: int = 1128, tokens_out: int = 80):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    usage = SimpleNamespace(total_input_tokens=tokens_in, total_output_tokens=tokens_out)
    return SimpleNamespace(output_text=text, usage=usage)


class FakeClient:
    """Stands in for genai.Client: `client.aio.interactions.create(**kwargs)`."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []
        self.aio = SimpleNamespace(interactions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make(outcomes, **kwargs):
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = FakeClient(outcomes)
    analyser = GeminiAnalyser("secret-key", "model-a", client=client, sleep=fake_sleep, **kwargs)
    return analyser, client, sleeps


async def test_request_shape_and_parsed_result():
    analyser, client, _ = make([answer(PEE_JSON)])
    before, after, during = make_jpeg((1, 2, 3)), make_jpeg((4, 5, 6)), make_jpeg((7, 8, 9))
    result = await analyser.analyse(before, after, during)
    assert result.pee is True and result.pee_confidence == 0.93
    call = client.calls[0]
    assert call["model"] == "model-a" and call["store"] is False
    assert call["generation_config"] == {"thinking_level": "minimal"}
    assert call["response_format"]["mime_type"] == "application/json"
    assert call["response_format"]["schema"] is RESPONSE_SCHEMA
    assert "Brazilian Portuguese" in call["system_instruction"]
    kinds = [(p["type"], p.get("text") or p.get("resolution")) for p in call["input"]]
    assert kinds == [
        ("text", "BEFORE:"), ("image", "high"),
        ("text", "AFTER:"), ("image", "high"),
        ("text", "DURING:"), ("image", "low"),
    ]  # fmt: skip
    assert base64.b64decode(call["input"][1]["data"]) == before
    assert call["input"][1]["mime_type"] == "image/jpeg"
    assert analyser.last_usage == {"in": 1128, "out": 80} and analyser.last_model == "model-a"


async def test_without_the_during_frame_only_two_images_are_sent():
    analyser, client, _ = make([answer(PEE_JSON)])
    await analyser.analyse(make_jpeg(), make_jpeg(), None)
    assert len(client.calls[0]["input"]) == 4


async def test_pad_change_uses_its_own_prompt_and_schema():
    analyser, client, _ = make([answer(PAD_JSON)])
    result = await analyser.analyse_pad(make_jpeg(), make_jpeg())
    assert result.pad_replaced is True
    call = client.calls[0]
    assert call["response_format"]["schema"] is PAD_RESPONSE_SCHEMA
    assert "REPLACED the pad" in call["system_instruction"]
    assert len(call["input"]) == 4


async def test_503_is_retried_with_the_backoff_delays():
    analyser, client, sleeps = make([StatusError(503), StatusError(503), answer(PEE_JSON)])
    result = await analyser.analyse(make_jpeg(), make_jpeg(), None)
    assert result.pee is True
    assert len(client.calls) == 3 and sleeps == [15.0, 60.0]


async def test_fallback_model_is_tried_on_the_second_attempt():
    analyser, client, _ = make([StatusError(503), answer(PEE_JSON)], fallback_model="model-b")
    await analyser.analyse(make_jpeg(), make_jpeg(), None)
    assert [c["model"] for c in client.calls] == ["model-a", "model-b"]
    assert analyser.last_model == "model-b"


async def test_gives_up_after_all_attempts():
    analyser, client, sleeps = make([StatusError(503)])
    with pytest.raises(AnalyserError, match="HTTP 503"):
        await analyser.analyse(make_jpeg(), make_jpeg(), None)
    assert len(client.calls) == 4 and sleeps == [15.0, 60.0, 180.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_client_errors_are_final(status):
    analyser, client, sleeps = make([StatusError(status)])
    with pytest.raises(AnalyserError, match=f"HTTP {status}"):
        await analyser.analyse(make_jpeg(), make_jpeg(), None)
    assert len(client.calls) == 1 and sleeps == []


async def test_stops_retrying_when_the_time_budget_would_be_exceeded():
    ticks = iter(range(0, 10_000, 200))  # every monotonic() call jumps 200 s
    analyser, client, sleeps = make([StatusError(503)], budget_s=300, monotonic=lambda: next(ticks))
    with pytest.raises(AnalyserError):
        await analyser.analyse(make_jpeg(), make_jpeg(), None)
    assert len(client.calls) < 4 and sum(sleeps) < 255


async def test_a_timeout_is_retried():
    analyser, client, sleeps = make([TimeoutError(), answer(PEE_JSON)])
    await analyser.analyse(make_jpeg(), make_jpeg(), None)
    assert len(client.calls) == 2 and sleeps == [15.0]


@pytest.mark.parametrize(
    "bad",
    ["", "not json", json.dumps({"pee": "maybe"}), json.dumps({**PEE_JSON, "pee_confidence": 3})],
)
async def test_unusable_answers_become_analyser_errors(bad):
    analyser, client, _ = make([answer(bad)])
    with pytest.raises(AnalyserError):
        await analyser.analyse(make_jpeg(), make_jpeg(), None)
    assert len(client.calls) == 1


async def test_error_messages_never_contain_the_key_or_the_exception_text():
    analyser, _, _ = make([StatusError(401, "bad key secret-key")])
    with pytest.raises(AnalyserError) as err:
        await analyser.analyse(make_jpeg(), make_jpeg(), None)
    assert "secret-key" not in str(err.value) and "bad key" not in str(err.value)


async def test_calls_are_serialised():
    import asyncio

    running = 0
    peak = 0

    class SlowClient(FakeClient):
        async def _create(self, **kwargs):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return answer(PEE_JSON)

    client = SlowClient([answer(PEE_JSON)])
    analyser = GeminiAnalyser("k", "m", client=client)
    await asyncio.gather(*(analyser.analyse(make_jpeg(), make_jpeg(), None) for _ in range(3)))
    assert peak == 1


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (StatusError(429), (True, "HTTP 429")),
        (StatusError(500), (True, "HTTP 500")),
        (StatusError(400), (False, "HTTP 400")),
        (TimeoutError(), (True, "timeout")),
        (httpx.ConnectError("x"), (True, "network error")),
        (ValueError("x"), (False, "ValueError")),
    ],
)
def test_classify_error(exc, expected):
    assert classify_error(exc) == expected
