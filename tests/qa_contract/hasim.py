"""Tiny Home Assistant template/state simulator for the QA contract tests (not part of the app)."""

import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import jinja2
import yaml

HA = Path("/homeassistant")
LISBON = ZoneInfo("Europe/Lisbon")
AUTOMATIONS = yaml.safe_load((HA / "custom/automations/ellie-pee-counter.yaml").read_text())
SCRIPTS = yaml.safe_load((HA / "custom/scripts/ellie-pee-counter.yaml").read_text())
DASH = yaml.safe_load((HA / "docs/ellie-pee-counter/dashboard-cards.yaml").read_text())


def automation(alias_part):
    return next(a for a in AUTOMATIONS if alias_part in a["alias"] or a["id"].endswith(alias_part))


class Ha:
    """states: {entity_id: (state, attrs)}"""

    def __init__(self, states, now):
        self.states, self.now, self.this = states, now, {"attributes": {"last_triggered": None}}
        env = jinja2.Environment()
        env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
        env.filters["as_datetime"] = self._as_datetime
        env.filters["as_local"] = lambda dt: dt.astimezone(ZoneInfo("Europe/Lisbon"))
        env.globals.update(
            states=self._states,
            state_attr=self._attr,
            is_state=lambda e, s: self._states(e) == s,
            now=lambda: self.now,
            relative_time=lambda d: str(self.now - d),
        )
        self.env = env

    def _states(self, e):
        return self.states.get(e, ("unknown", {}))[0]

    def _attr(self, e, a):
        return self.states.get(e, ("unknown", {}))[1].get(a)

    @staticmethod
    def _as_datetime(value, default=None):
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return default

    def render(self, template, **variables):
        variables.setdefault("this", self.this)
        return self.env.from_string(template).render(**variables).strip()


def parse_native(text):
    """What HA does with a rendered string result (literal types)."""
    if text in ("True", "true"):
        return True
    if text in ("False", "false"):
        return False
    return text


def entity_ids_in(text):
    return set(
        re.findall(
            r"\b(?:sensor|binary_sensor|number|select|switch|image|input_number|input_datetime|script)"
            r"\.[a-z0-9_]+",
            text,
        )
    )
