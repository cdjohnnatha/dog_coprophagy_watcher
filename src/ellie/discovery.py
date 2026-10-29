"""Home Assistant MQTT discovery payloads for Ellie.

Contract for the single JSON object published (retained) on ``ellie/state``::

    pee_today: int, visits_today: int, uncertain_open: int, at_tray: bool,
    last_pee: null | {at: ISO-8601 UTC, visit_id, image, reason},
    last_visit: null | {visit_id, auto_outcome, final_outcome, started_at, ended_at,
                        reason, poop_observed, needs_review, deciding_analyser},
    review_queue: {count: int, items: [{visit_id, time, auto_outcome, reason, before, after}]},
    daily_history: {days: [{date, pees, visits, corrected}]},
    accuracy: {pct: float|null, window_days: int,
               analysers: {id: {labelled, agree, accuracy}}},
    analysis_calls: {calls: int, tokens_in: int, tokens_out: int, estimated_cost_eur: float,
                     estimated_cost_eur_total: float},
    settings: {confidence_threshold: float, merge_gap_s: int, deciding_analyser: str,
               validation_mode: bool, pad_change_days: int},
    pad: null | {changed_at, due_at | null, source, days_in_use: float, pees_since: int,
                 pad_color: str | null, image: str | null, history: [...]}

Commands arrive on ``ellie/set/<setting>``; availability on ``ellie/status``.
Three image entities get raw JPEG bytes (retained) on the topics in ``IMAGE_TOPICS``: HA serves
them with its own auth, which a plain /media/local URL in a dashboard card cannot do.
"""

from collections.abc import Sequence
from typing import Any

from ellie.settings import RUNTIME_RANGES

DISCOVERY_PREFIX = "homeassistant"
STATE_TOPIC = "ellie/state"
STATUS_TOPIC = "ellie/status"
SET_TOPIC_PREFIX = "ellie/set"
IMAGE_TOPICS = {
    "last_pee_photo": "ellie/image/last_pee",
    "review_before": "ellie/image/review_before",
    "review_after": "ellie/image/review_after",
}
DEVICE: dict[str, Any] = {
    "identifiers": ["ellie_v3"],
    "name": "Ellie",
    "manufacturer": "Ellie Watcher",
    "model": "v3",
}

_NONE = "None"


def _attrs(**fields: str) -> str:
    """Build a json_attributes_template from attribute name -> Jinja expression."""
    body = ", ".join(f"'{k}': {v}" for k, v in fields.items())
    return "{{ {" + body + "} | tojson }}"


def _val(path: str) -> str:
    return "{{ value_json." + path + " }}"


def _opt(obj: str, key: str) -> str:
    """Null-safe lookup of ``value_json.<obj>.<key>``."""
    return f"(value_json.{obj}.{key} if value_json.{obj} else none)"


def _specs(analysers: Sequence[str]) -> list[tuple[str, str, dict[str, Any]]]:
    """(component, object, extra payload) for every entity."""
    last_visit_keys = [
        "visit_id",
        "auto_outcome",
        "final_outcome",
        "started_at",
        "ended_at",
        "reason",
        "poop_observed",
        "needs_review",
        "deciding_analyser",
    ]
    pad_keys = ["source", "days_in_use", "pees_since", "pad_color", "image", "history"]
    lo_c, hi_c = RUNTIME_RANGES["confidence_threshold"]
    lo_m, hi_m = RUNTIME_RANGES["merge_gap_s"]
    lo_p, hi_p = RUNTIME_RANGES["pad_change_days"]

    def cmd(setting: str) -> dict[str, Any]:
        return {"command_topic": f"{SET_TOPIC_PREFIX}/{setting}", "entity_category": "config"}

    return [
        (
            "sensor",
            "pee_today",
            {
                "name": "Xixis hoje",
                "state_class": "measurement",
                "unit_of_measurement": "xixis",
                "value_template": _val("pee_today"),
                "json_attributes_topic": STATE_TOPIC,
                "json_attributes_template": _attrs(
                    visits_today="value_json.visits_today",
                    uncertain_open="value_json.uncertain_open",
                ),
            },
        ),
        (
            "sensor",
            "last_pee",
            {
                "name": "Último xixi",
                "device_class": "timestamp",
                "value_template": "{{ " + _opt("last_pee", "at") + " }}",
                "json_attributes_topic": STATE_TOPIC,
                "json_attributes_template": _attrs(
                    **{k: _opt("last_pee", k) for k in ("visit_id", "image", "reason")}
                ),
            },
        ),
        (
            "sensor",
            "pee_daily_history",
            {
                "name": "Histórico diário de xixi",
                "value_template": "{{ value_json.daily_history.days | length }}",
                "json_attributes_topic": STATE_TOPIC,
                "json_attributes_template": _attrs(days="value_json.daily_history.days"),
            },
        ),
        (
            "sensor",
            "review_queue",
            {
                "name": "Fila de revisão",
                "value_template": _val("review_queue.count"),
                "json_attributes_topic": STATE_TOPIC,
                "json_attributes_template": _attrs(items="value_json.review_queue['items']"),
            },
        ),
        (
            "sensor",
            "pee_accuracy",
            {
                "name": "Precisão da detecção",
                "unit_of_measurement": "%",
                "value_template": _val("accuracy.pct"),
                "json_attributes_topic": STATE_TOPIC,
                "json_attributes_template": _attrs(
                    analysers="value_json.accuracy.analysers",
                    window_days="value_json.accuracy.window_days",
                ),
            },
        ),
        (
            "sensor",
            "last_visit",
            {
                "name": "Última visita",
                "value_template": "{{ " + _opt("last_visit", "auto_outcome") + " }}",
                "json_attributes_topic": STATE_TOPIC,
                "json_attributes_template": _attrs(
                    **{k: _opt("last_visit", k) for k in last_visit_keys}
                ),
            },
        ),
        (
            "sensor",
            "analysis_calls_month",
            {
                "name": "Chamadas de análise no mês",
                "state_class": "total_increasing",
                "value_template": _val("analysis_calls.calls"),
                "json_attributes_topic": STATE_TOPIC,
                "json_attributes_template": _attrs(
                    tokens_in="value_json.analysis_calls.tokens_in",
                    tokens_out="value_json.analysis_calls.tokens_out",
                    estimated_cost_eur="value_json.analysis_calls.estimated_cost_eur",
                    estimated_cost_eur_total="value_json.analysis_calls.estimated_cost_eur_total",
                ),
            },
        ),
        (
            "binary_sensor",
            "at_tray",
            {
                "name": "No tabuleiro",
                "device_class": "occupancy",
                "payload_on": "ON",
                "payload_off": "OFF",
                "value_template": "{{ 'ON' if value_json.at_tray else 'OFF' }}",
            },
        ),
        (
            "number",
            "confidence_threshold",
            {
                "name": "Limiar de confiança",
                "min": lo_c,
                "max": hi_c,
                "step": 0.01,
                "mode": "box",
                "value_template": _val("settings.confidence_threshold"),
                **cmd("confidence_threshold"),
            },
        ),
        (
            "number",
            "merge_gap",
            {
                "name": "Intervalo para juntar visitas",
                "min": lo_m,
                "max": hi_m,
                "step": 5,
                "unit_of_measurement": "s",
                "value_template": _val("settings.merge_gap_s"),
                **cmd("merge_gap_s"),
            },
        ),
        (
            "select",
            "deciding_analyser",
            {
                "name": "Analisador que decide",
                "options": list(analysers),
                "value_template": _val("settings.deciding_analyser"),
                **cmd("deciding_analyser"),
            },
        ),
        (
            "switch",
            "validation_mode",
            {
                "name": "Modo de validação",
                "payload_on": "ON",
                "payload_off": "OFF",
                "state_on": "ON",
                "state_off": "OFF",
                "value_template": "{{ 'ON' if value_json.settings.validation_mode else 'OFF' }}",
                **cmd("validation_mode"),
            },
        ),
        (
            "sensor",
            "pad_changed_at",
            {
                "name": "Tapete trocado em",
                "device_class": "timestamp",
                "value_template": "{{ " + _opt("pad", "changed_at") + " }}",
                "json_attributes_topic": STATE_TOPIC,
                "json_attributes_template": _attrs(**{k: _opt("pad", k) for k in pad_keys}),
            },
        ),
        (
            "sensor",
            "pad_change_due",
            {
                "name": "Próxima troca do tapete",
                "device_class": "timestamp",
                "value_template": "{{ " + _opt("pad", "due_at") + " }}",
            },
        ),
        (
            "number",
            "pad_change_days",
            {
                "name": "Dias entre trocas do tapete",
                "min": lo_p,
                "max": hi_p,
                "step": 1,
                "unit_of_measurement": "d",
                "value_template": _val("settings.pad_change_days"),
                **cmd("pad_change_days"),
            },
        ),
        *[
            (
                "image",
                obj,
                {"name": name, "image_topic": IMAGE_TOPICS[obj], "content_type": "image/jpeg"},
            )
            for obj, name in (
                ("last_pee_photo", "Foto do último xixi"),
                ("review_before", "Revisão: foto antes"),
                ("review_after", "Revisão: foto depois"),
            )
        ],
    ]


def _topic(component: str, unique_id: str) -> str:
    return f"{DISCOVERY_PREFIX}/{component}/ellie/{unique_id}/config"


def discovery_messages(version: str, analysers: Sequence[str]) -> list[tuple[str, dict]]:
    """Return one (topic, payload) per entity; the caller publishes them retained."""
    messages: list[tuple[str, dict]] = []
    for component, obj, extra in _specs(analysers):
        unique_id = f"ellie_v3_{obj}"
        payload: dict[str, Any] = {
            "name": extra["name"],
            "unique_id": unique_id,
            "default_entity_id": f"{component}.ellie_{obj}",
            "device": DEVICE,
            "origin": {"name": "Ellie", "sw_version": version},
            "availability_topic": STATUS_TOPIC,
        }
        if component != "image":  # image entities read bytes from image_topic instead
            payload["state_topic"] = STATE_TOPIC
        payload.update({k: v for k, v in extra.items() if k != "name"})
        messages.append((_topic(component, unique_id), payload))
    return messages


def removal_messages() -> list[str]:
    """Discovery topics to clear (publish an empty retained payload to each)."""
    return [topic for topic, _ in discovery_messages("", [])]
