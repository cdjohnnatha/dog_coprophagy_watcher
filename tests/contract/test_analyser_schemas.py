"""The schemas in the code, the pydantic models and docs/contracts/*.json must agree."""

import json
from pathlib import Path

import jsonschema
import pytest

from ellie.analysers.base import AnalysisResult, PadChangeResult
from ellie.analysers.gemini import PAD_RESPONSE_SCHEMA, RESPONSE_SCHEMA
from helpers import make_pad_result, make_result

CONTRACTS = Path(__file__).resolve().parents[2] / "docs" / "contracts"


def contract(name: str) -> dict:
    return json.loads((CONTRACTS / name).read_text())


@pytest.mark.parametrize(
    ("code_schema", "file"),
    [
        (RESPONSE_SCHEMA, "analysis-result.schema.json"),
        (PAD_RESPONSE_SCHEMA, "pad-change-result.schema.json"),
    ],
)
def test_gemini_response_schema_matches_the_contract_file(code_schema, file):
    doc = contract(file)
    assert set(code_schema["properties"]) == set(doc["properties"])
    assert set(code_schema["required"]) == set(doc["required"])
    for name, spec in doc["properties"].items():
        assert code_schema["properties"][name]["type"] == spec["type"], name
        assert code_schema["properties"][name].get("enum") == spec.get("enum"), name


@pytest.mark.parametrize(
    ("model", "file", "example"),
    [
        (AnalysisResult, "analysis-result.schema.json", make_result),
        (PadChangeResult, "pad-change-result.schema.json", make_pad_result),
    ],
)
def test_models_accept_what_the_contract_accepts_and_reject_what_it_rejects(model, file, example):
    schema = contract(file)
    good = example().model_dump(exclude_none=True)
    jsonschema.validate(good, schema)
    model.model_validate(good)

    for bad in (
        {**good, "extra_field": 1},
        {k: v for k, v in good.items() if k == "reason"},
        {**good, "image_quality": "blurry"},
        {**good, "reason": "x" * 141},
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, schema)
        with pytest.raises(ValueError):
            model.model_validate(bad)
