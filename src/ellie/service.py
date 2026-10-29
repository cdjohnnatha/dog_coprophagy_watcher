"""Orchestrates a visit from Frigate counts to a decided outcome, plus settings and alerts.

All I/O is injected (store, camera fetchers, analysers, publish, clock), so the whole flow
runs in tests without MQTT, the camera or the network. Commands from Home Assistant
(labels, manual entries, ...) are registered in `self.commands`.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from datetime import datetime
from typing import Any

from ellie import discovery
from ellie.analysers.base import Analyser, AnalyserError
from ellie.counting import Counting
from ellie.decide import decide_visit
from ellie.evidence import Evidence
from ellie.frames import FrameError, ReferenceKeeper
from ellie.settings import RUNTIME_DEFAULTS, clamp_setting
from ellie.state import build_state, local_text
from ellie.store import Store
from ellie.visits import Action, VisitTracker

log = logging.getLogger(__name__)

Publish = Callable[[str, str, bool], Awaitable[object]]
REFERENCE_CLEAR_S = 10  # the area must be clear this long before a reference frame is kept
ALERT_EVERY_S = 6 * 3600

# Owner-facing reasons (Brazilian Portuguese) for visits that could not be decided.
REASONS = {
    "frigate_offline": "o Frigate ficou fora do ar durante a visita",
    "area_not_clear": "o tabuleiro não ficou livre para a foto final",
    "restart": "o serviço reiniciou durante a visita",
    "no_reference": "faltou a foto de referência do tabuleiro",
    "frame_error": "não consegui pegar a imagem da câmera",
    "analyser_not_configured": "a análise do Gemini não está configurada",
    "analysis_failed": "a análise falhou",
    "credit_exhausted": "os créditos do Gemini acabaram",
    "gemini_key": "o Gemini recusou a chave",
}
# HTTP status -> (reason code, alert title, alert message) for problems the owner must fix.
FIXABLE = {
    "HTTP 402": (
        "credit_exhausted",
        "Ellie: créditos do Gemini acabaram",
        "As visitas novas ficam sem análise até você colocar crédito em aistudio.google.com/billing.",
    ),
    "HTTP 401": (
        "gemini_key",
        "Ellie: chave do Gemini recusada",
        "Confira a chave nas opções do app.",
    ),
    "HTTP 403": (
        "gemini_key",
        "Ellie: chave do Gemini recusada",
        "Confira a chave nas opções do app.",
    ),
}


class Service:
    def __init__(
        self,
        *,
        store: Store,
        evidence: Evidence,
        reference: ReferenceKeeper,
        tracker: VisitTracker,
        analysers: dict[str, Analyser],
        fetch_frame: Callable[[], Awaitable[bytes]],
        fetch_snapshot: Callable[[str], Awaitable[bytes | None]],
        crop: Callable[[bytes], bytes],
        publish: Publish,
        clock: Callable[[], datetime],
        camera: str = "ellie",
        usd_to_eur: float = 0.9,
        version: str = "3.0.0",
    ) -> None:
        self.store, self.evidence, self.reference, self.tracker = (
            store,
            evidence,
            reference,
            tracker,
        )
        self.analysers = analysers
        self._fetch_frame, self._fetch_snapshot, self._crop = fetch_frame, fetch_snapshot, crop
        self._publish_fn, self._clock = publish, clock
        self.camera, self.usd_to_eur, self.version = camera, usd_to_eur, version
        self.cfg: dict[str, Any] = {
            key: clamp_setting(key, store.get_setting(key, default))[0]
            for key, default in RUNTIME_DEFAULTS.items()
        }
        self.tracker.merge_gap_s = self.cfg["merge_gap_s"]
        self.commands: dict[str, Callable[[dict], Awaitable[None]]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._alerts_sent: dict[str, datetime] = {}

    # --- MQTT entry points ---

    async def on_connect(self) -> None:
        """(Re)publish discovery and the full state after every (re)connection."""
        options = list(dict.fromkeys([*self.analysers, self.cfg["deciding_analyser"]]))
        for topic, payload in discovery.discovery_messages(self.version, options):
            await self._publish(topic, json.dumps(payload, ensure_ascii=False), True)
        await self.publish_state()

    async def on_message(self, topic: str, payload: bytes) -> None:
        now = self._clock()
        actions: list[Action] = []
        if topic == "frigate/pad_zone/dog":
            actions = self._count(payload, lambda n: self.tracker.on_pad_dog(n, now))
        elif topic == "frigate/tray_area/dog":
            actions = self._count(payload, lambda n: self.tracker.on_tray_dog(n, now))
        elif topic == "frigate/tray_area/person":
            actions = self._count(payload, lambda n: self.tracker.on_tray_person(n, now))
        elif topic == "frigate/tray_pad/dog":
            actions = self._count(payload, lambda n: self.tracker.on_tray_pad_dog(n, now))
        elif topic == "frigate/events":
            event_id = self._dog_event_id(payload)
            actions = self.tracker.on_event(event_id, now) if event_id else []
        elif topic == "frigate/available":
            actions = self.tracker.on_frigate_available(payload.decode().strip() == "online", now)
        elif topic.startswith("ellie/set/"):
            await self.on_set(topic.removeprefix("ellie/set/"), payload.decode().strip())
        elif topic.startswith("ellie/cmd/"):
            await self.on_command(topic.removeprefix("ellie/cmd/"), payload)
        await self._apply(actions)

    async def tick(self) -> None:
        await self._apply(self.tracker.tick(self._clock()))

    async def minute_tasks(self) -> None:
        now = self._clock()
        if self.tracker.active is None and self.tracker.clear_for(now) >= REFERENCE_CLEAR_S:
            try:
                self.reference.update(self._crop(await self._fetch_frame()))
            except FrameError as exc:
                log.warning("reference refresh failed: %s", exc)
        Counting(self.store).resolve_overdue_uncertain(now)
        await self.publish_state()

    async def recover(self) -> None:
        """After a restart: visits we were watching are lost; interrupted analyses are retried."""
        for visit in self.store.open_visits():
            await self._fail(visit["id"], "restart")
        for visit in self.store.analysing_visits():
            if (self.evidence.visit_dir(visit["id"]) / "after.jpg").exists():
                self._spawn(self._analyse(visit["id"], retry=True))
            else:
                await self._fail(visit["id"], "restart")

    async def wait_idle(self) -> None:
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # --- settings ---

    async def on_set(self, key: str, raw: str) -> None:
        try:
            value, _ = clamp_setting(key, raw)
        except (KeyError, ValueError):
            log.warning("ignored setting %s=%r", key, raw)
            return
        if key == "deciding_analyser" and value not in self.analysers:
            log.warning("unknown analyser %r, setting ignored", value)
            return
        self.cfg[key] = value
        self.store.set_setting(key, value)
        if key == "merge_gap_s":
            self.tracker.merge_gap_s = value
        await self.publish_state()

    async def on_command(self, name: str, payload: bytes) -> None:
        handler = self.commands.get(name)
        if handler is None:
            log.warning("unknown command %s", name)
            return
        try:
            data = json.loads(payload or b"{}")
        except json.JSONDecodeError:
            log.warning("command %s: payload is not JSON", name)
            return
        if not isinstance(data, dict):
            log.warning("command %s: payload is not an object", name)
            return
        await handler(data)

    # --- visit flow ---

    async def _apply(self, actions: list[Action]) -> None:
        for action in actions:
            visit = action.visit
            if action.kind == "started":
                self._insert_started(visit)
            elif action.kind in ("updated", "capture_after"):
                self._update_tracked(visit)
                if action.kind == "capture_after":
                    self._spawn(self._analyse(visit["id"]))
            elif action.kind == "unknown":
                self._update_tracked(visit)
                await self._fail(visit["id"], action.reason or "analysis_failed")
        if actions:
            await self.publish_state()

    def _insert_started(self, visit: dict) -> None:
        age = self.reference.age_s()
        reference = self.reference.latest()
        self.store.insert_visit({**visit, "before_age_s": age, "needs_review": False})
        if reference:
            self.evidence.write(self.evidence.visit_dir(visit["id"]) / "before.jpg", reference[0])
        if age is not None and age > 900:
            log.warning("visit %s: the before reference is %s s old", visit["id"], age)

    def _update_tracked(self, visit: dict) -> None:
        fields = {k: visit[k] for k in ("state", "ended_at", "entries", "person_present",
                                        "frigate_event_ids", "on_pad_s")}  # fmt: skip
        self.store.update_visit(visit["id"], **fields)

    async def _analyse(self, visit_id: str, retry: bool = False) -> None:
        try:
            await self._analyse_inner(visit_id, retry)
        except Exception:
            log.exception("analysis of %s crashed", visit_id)
            await self._fail(visit_id, "analysis_failed")

    async def _analyse_inner(self, visit_id: str, retry: bool) -> None:
        directory = self.evidence.visit_dir(visit_id)
        before_path, after_path = directory / "before.jpg", directory / "after.jpg"
        if not retry:
            try:
                after = self._crop(await self._fetch_frame())
            except FrameError as exc:
                log.warning("visit %s: %s", visit_id, exc)
                return await self._fail(visit_id, "frame_error")
            self.evidence.write(after_path, after)
            self.reference.update(after)  # the next visit starts from this clear frame
        if not before_path.exists():
            return await self._fail(visit_id, "no_reference")
        analyser = self.analysers.get(self.cfg["deciding_analyser"])
        if analyser is None:
            return await self._fail(visit_id, "analyser_not_configured")

        events = self.store.get_visit(visit_id)["frigate_event_ids"]
        during = await self._fetch_snapshot(events[0]) if events else None
        if during:
            self.evidence.write(directory / "during.jpg", during)

        started = time.monotonic()
        try:
            result = await analyser.analyse(
                before_path.read_bytes(), after_path.read_bytes(), during
            )
        except AnalyserError as exc:
            self.store.insert_analysis(visit_id, analyser.id, "deciding", None, "unknown",
                                       int((time.monotonic() - started) * 1000), str(exc), None)  # fmt: skip
            return await self._fail(visit_id, self._reason_for(str(exc)))
        outcome = decide_visit(result, self.cfg["confidence_threshold"])
        usage = getattr(analyser, "last_usage", None)
        self.store.insert_analysis(visit_id, analyser.id, "deciding",
                                   result.model_dump(exclude_none=True), outcome,
                                   int((time.monotonic() - started) * 1000), None, usage)  # fmt: skip
        decided = outcome in ("pee", "no_pee")
        self.store.update_visit(
            visit_id,
            state="DECIDED",
            auto_outcome=outcome,
            final_outcome=outcome if decided else None,
            poop_observed=result.poop,
            reason=result.reason,
            needs_review=(not decided) or bool(self.cfg["validation_mode"]),
            deciding_analyser=analyser.id,
        )
        await self._visit_decided(visit_id, "uncertain" if outcome == "uncertain" else None)

    def _reason_for(self, error: str) -> str:
        for marker, (code, _title, _message) in FIXABLE.items():
            if marker in error:
                return code
        return "analysis_failed"

    async def _fail(self, visit_id: str, code: str) -> None:
        self.store.update_visit(
            visit_id,
            state="DECIDED",
            auto_outcome="unknown",
            final_outcome="unknown",
            needs_review=True,
            reason=REASONS.get(code, REASONS["analysis_failed"]),
        )
        for fix_code, title, message in FIXABLE.values():
            if fix_code == code:
                await self._alert(code, title, message)
                break
        await self._visit_decided(visit_id, "unknown")

    async def _visit_decided(self, visit_id: str, review_kind: str | None) -> None:
        visit = self.store.get_visit(visit_id)
        await self._publish(f"ellie/visit/{visit_id}", json.dumps(visit, ensure_ascii=False), False)
        if review_kind:
            await self._request_review(visit, review_kind)
        await self.publish_state()

    async def _request_review(self, visit: dict, kind: str) -> None:
        hhmm = (local_text(visit["started_at"]) or "")[11:16]
        after = self.evidence.visit_dir(visit["id"]) / "after.jpg"
        if kind == "uncertain":
            title, message = "Ellie: fez xixi?", f"Visita às {hhmm} — {visit['reason']}"
        else:
            title = "Ellie: não consegui analisar"
            message = f"Visita às {hhmm} ({visit['reason']}). Toque para revisar."
        payload = {
            "visit_id": visit["id"],
            "kind": kind,
            "title": title,
            "message": message,
            "image": self.evidence.local_url(after) if after.exists() else None,
            "reason": visit["reason"],
        }
        await self._publish("ellie/review/request", json.dumps(payload, ensure_ascii=False), False)

    async def _alert(self, kind: str, title: str, message: str) -> None:
        """Owner-facing problem (credits, key). Rate-limited, HA turns it into a notification."""
        now = self._clock()
        last = self._alerts_sent.get(kind)
        if last and (now - last).total_seconds() < ALERT_EVERY_S:
            return
        self._alerts_sent[kind] = now
        payload = {"kind": kind, "title": title, "message": message}
        await self._publish("ellie/alert", json.dumps(payload, ensure_ascii=False), False)

    # --- helpers ---

    async def publish_state(self) -> None:
        state = build_state(
            store=self.store,
            evidence=self.evidence,
            active_visit=self.tracker.active,
            cfg=self.cfg,
            now=self._clock(),
            usd_to_eur=self.usd_to_eur,
        )
        await self._publish(discovery.STATE_TOPIC, json.dumps(state, ensure_ascii=False), True)

    async def _publish(self, topic: str, payload: str, retain: bool) -> None:
        await self._publish_fn(topic, payload, retain)

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    def _count(payload: bytes, handler: Callable[[int], list[Action]]) -> list[Action]:
        try:
            count = int(float(payload.decode().strip()))
        except ValueError:
            log.warning("ignored non-numeric count %r", payload[:20])
            return []
        return handler(count)

    def _dog_event_id(self, payload: bytes) -> str | None:
        try:
            after = (json.loads(payload) or {}).get("after") or {}
        except (json.JSONDecodeError, AttributeError):
            return None
        if after.get("camera") != self.camera or after.get("label") != "dog":
            return None
        if "pad_zone" not in (after.get("entered_zones") or []):
            return None
        return after.get("id")
