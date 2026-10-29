import pytest

from helpers import FakeClock, build_rig, make_jpeg


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_rig(tmp_path, clock):
    rigs = []

    def factory(**kw):
        rig = build_rig(tmp_path, clock, **kw)
        rig.reference.update(make_jpeg((200, 200, 200)))
        rigs.append(rig)
        return rig

    yield factory
    for r in rigs:
        r.store.close()
