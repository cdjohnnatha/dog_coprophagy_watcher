import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# These tests read the Home Assistant YAML next to the app. Where that repo is absent (GitHub
# Actions), do not even import them: hasim.py loads the YAML at import time.
if not Path("/homeassistant/custom/automations/ellie-pee-counter.yaml").exists():
    collect_ignore_glob = ["test_*.py"]


def pytest_collection_modifyitems(config, items):
    """These tests read the Home Assistant YAML next to the app; skip them where it is absent."""
    import pytest

    if not (Path("/homeassistant/custom/automations/ellie-pee-counter.yaml")).exists():
        skip = pytest.mark.skip(reason="the Home Assistant config repo is not mounted here")
        for item in items:
            if "qa_contract" in str(item.fspath):
                item.add_marker(skip)
