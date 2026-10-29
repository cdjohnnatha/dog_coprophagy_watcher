"""Shared pytest fixtures."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ellie.evidence import Evidence
from ellie.store import Store
from helpers import FakeAnalyser, FakeClock, make_jpeg


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake_analyser() -> FakeAnalyser:
    return FakeAnalyser()


@pytest.fixture
def tmp_store(tmp_path: Path) -> Iterator[Store]:
    store = Store(str(tmp_path / "ellie.db"))
    yield store
    store.close()


@pytest.fixture
def tmp_evidence(tmp_path: Path) -> Evidence:
    return Evidence(tmp_path / "media")


@pytest.fixture
def jpeg_factory() -> Callable[..., bytes]:
    return make_jpeg
