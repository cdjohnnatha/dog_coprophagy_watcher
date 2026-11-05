"""Orchestrates a visit from Frigate counts to a decided outcome, plus settings and alerts.

All I/O is injected (store, camera fetchers, analysers, publish, clock), so the whole flow
runs in tests without MQTT, the camera or the network. Commands from Home Assistant
(labels, manual entries, ...) are registered in `self.commands`.
"""

import asyncio
import json
import logging
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable, Coroutine
from datetime import date, datetime, timedelta
from typing import Any

from ellie import discovery
from ellie.analysers.base import Analyser, AnalyserError
from ellie.counting import LOCAL_TZ, Counting, local_date
from ellie.decide import decide_poop, decide_visit
from ellie.evidence import Evidence
from ellie.frames import FrameError, ReferenceKeeper
from ellie.pads import PresenceTracker
from ellie.pads_flow import PadsMixin
from ellie.settings import RUNTIME_DEFAULTS, clamp_setting
from ellie.state import build_state, local_text
from ellie.store import Store
from ellie.visits import Action, VisitTracker, iso
from ellie.watch import Candidate, PadWatch

log = logging.getLogger(__name__)

Publish = Callable[[str, str | bytes, bool], Awaitable[object]]
EVENT_ID = re.compile(
    r"^[0-9]+\.[0-9]+-[A-Za-z0-9]+$"
)  # Frigate event id, e.g. 1793288511.990085-yohln0
MAX_BACKDATE_DAYS = 366  # manual entries and pad changes older than this are rejected
LABEL_CHANNELS = ("notification", "dashboard")
REFERENCE_CLEAR_S = 10  # the area must be clear this long before a reference frame is kept
WATCH_DAILY_CAP = 6  # visits the pad watch may open per local day (worst case about 0.008 EUR)
WATCH_EXPLAIN_S = 180  # a visit or presence this close to a candidate already explains it
ALERT_EVERY_S = 6 * 3600
POOP_PRESENT_TIMEOUT_H = 12  # a counted poop nobody cleared stops being "present" after this
POOP_TRACK_LOOKBACK_DAYS = 2  # after a restart, open poops this recent are followed again
POOP_LABELS = ("poop", "no_poop")
PEE_LABELS = ("pee", "no_pee")

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


def _placeholder_jpeg() -> bytes:
    """A flat grey picture for "no photo for this pee" (manual entry, nothing counted yet)."""
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (640, 360), (70, 70, 70)).save(out, "JPEG", quality=60)
    return out.getvalue()


class Service(PadsMixin):
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
        download_clip: Callable[[str, str], Awaitable[bool]] | None = None,
        presence: PresenceTracker | None = None,
        watch: PadWatch | None = None,
    ) -> None:
        self.watch = watch
        self._watch_day: date | None = None
        self._watch_opened = 0
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
        self.tracker.id_taken = lambda i: self.store.get_visit(i) is not None
        self._download_clip = download_clip
        self.presence = presence or PresenceTracker()
        self.presence.id_taken = lambda i: self.store.get_presence(i) is not None
        self.commands: dict[str, Callable[[dict], Awaitable[None]]] = {
            "label": self._cmd_label,
            "manual_pee": self._cmd_manual_pee,
            "reject_last": self._cmd_reject_last,
            "reanalyse": self._cmd_reanalyse,
            "pad_label": self._cmd_pad_label,
            "pad_changed": self._cmd_pad_changed,
            "pad_undo": self._cmd_pad_undo,
            "poop_clear": self._cmd_poop_clear,
            "manual_poop": self._cmd_manual_poop,
        }
        self._tasks: set[asyncio.Task] = set()
        self._alerts_sent: dict[str, datetime] = {}
        self._last_purge: date | None = None
        self._inflight: set[str] = set()  # visit/presence ids with an analysis running
        self._image_keys: dict[str, str] = {}
        self._poop_asked: set[str] = set()  # visits whose poop question was already sent

    # --- MQTT entry points ---

    async def on_connect(self) -> None:
        """(Re)publish discovery and the full state after every (re)connection."""
        self._image_keys.clear()  # photos are sent again on every connection (the broker may have lost them)
        options = list(dict.fromkeys([*self.analysers, self.cfg["deciding_analyser"]]))
        for topic, payload in discovery.discovery_messages(self.version, options):
            await self._publish(topic, json.dumps(payload, ensure_ascii=False), True)
        await self.publish_state()

    async def on_message(self, topic: str, payload: bytes) -> None:
        now = self._clock()
        actions: list[Action] = []
        presence_actions = self._sync_presence(now)
        if topic == "frigate/pad_zone/dog":
            actions = self._count(payload, lambda n: self.tracker.on_pad_dog(n, now))
        elif topic == "frigate/tray_area/dog":
            actions = self._count(payload, lambda n: self.tracker.on_tray_dog(n, now))
            self._count(payload, lambda n: self.presence.on_dog_area(n, now))
        elif topic == "frigate/tray_area/person":
            actions = self._count(payload, lambda n: self.tracker.on_tray_person(n, now))
            presence_actions += self._count(payload, lambda n: self.presence.on_person(n, now))
        elif topic == "frigate/tray_pad/dog":
            actions = self._count(payload, lambda n: self.tracker.on_tray_pad_dog(n, now))
        elif topic == "frigate/events":
            event_id = self._dog_event_id(payload)
            actions = self.tracker.on_event(event_id, now) if event_id else []
        elif topic == "frigate/available":
            actions = self.tracker.on_frigate_available(
                payload.decode(errors="replace").strip() == "online", now
            )
            self.presence.reset(now)
        elif topic.startswith("ellie/set/"):
            await self.on_set(
                topic.removeprefix("ellie/set/"), payload.decode(errors="replace").strip()
            )
        elif topic.startswith("ellie/cmd/"):
            await self.on_command(topic.removeprefix("ellie/cmd/"), payload)
        await self._apply(actions)
        presence_actions += self._sync_presence(now)  # a visit may have just started
        await self._apply_presence(presence_actions)

    async def tick(self) -> None:
        now = self._clock()
        await self._apply(self.tracker.tick(now))
        await self._apply_presence(self._sync_presence(now) + self.presence.tick(now))

    async def minute_tasks(self) -> None:
        now = self._clock()
        if self.tracker.active is None and self.tracker.clear_for(now) >= REFERENCE_CLEAR_S:
            try:
                crop = self._crop(await self._fetch_frame())
                confirmed = self.watch.observe(crop) if self.watch is not None else None
                self.reference.update(crop)
                if confirmed is not None:  # a change Frigate never reported (ADR-0012)
                    await self._open_watch_visit(confirmed)
                for visit_id in self.watch.take_cleared() if self.watch is not None else ():
                    await self._clear_poops("watch", visit_id=visit_id)  # the blob is gone
            except (FrameError, OSError) as exc:
                log.warning("reference refresh failed: %s", exc)
        await self._expire_poops(now)
        Counting(self.store).resolve_overdue_uncertain(now)
        local = now.astimezone(LOCAL_TZ)
        if (local.hour, local.minute) >= (3, 30) and self._last_purge != local.date():
            self._last_purge = local.date()  # once a day, after 03:30 local
            stats = await asyncio.to_thread(self.evidence.purge, now)
            log.info("retention: %s", stats)
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
        for presence in self.store.analysing_presences():
            retryable = presence["state"] == "ANALYSING"
            if retryable and (self.evidence.presence_dir(presence["id"]) / "after.jpg").exists():
                self._spawn(self._analyse_presence(presence["id"], retry=True))
            else:  # we lost track of this person: nothing to judge
                self.store.update_presence(presence["id"], state="DECIDED", reason="restart")
        self._follow_open_poops()
        analyser = self.analysers.get(self.cfg["deciding_analyser"])
        for presence in self.store.pending_promotions():  # cut short between decision and check
            if analyser is not None and hasattr(analyser, "analyse"):
                self._spawn(self._promote_if_eliminated(presence, analyser))
            else:
                self.store.update_presence(presence["id"], promotion_pending=False)

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
        except ValueError:  # JSONDecodeError and UnicodeDecodeError
            log.warning("command %s: payload is not JSON", name)
            return
        if not isinstance(data, dict):
            log.warning("command %s: payload is not an object", name)
            return
        await handler(data)

    # --- owner commands (contracts/mqtt.md) ---

    async def _cmd_label(self, data: dict) -> None:
        """`ellie/cmd/label`: the owner answers for one visit (notification or dashboard)."""
        visit_id, label = data.get("visit_id"), data.get("label")
        channel, kind = data.get("channel", "dashboard"), data.get("kind", "pee")
        visit = self.store.get_visit(visit_id) if isinstance(visit_id, str) else None
        valid = (
            {"pee": PEE_LABELS, "poop": POOP_LABELS}.get(kind, ()) if isinstance(kind, str) else ()
        )
        if visit is None or label not in valid or channel not in LABEL_CHANNELS:
            log.warning(
                "ignored label command %r", {k: data.get(k) for k in ("visit_id", "kind", "label")}
            )
            return
        if kind == "poop":  # a poop answer never touches the pee outcome (ADR-0014)
            await self._apply_poop_label(visit, label, channel, self._by(data))
        else:
            await self._apply_label(visit, label, channel, self._by(data))

    def _manual_time(self, command: str, data: dict) -> datetime | None:
        """The time of a manual entry (local if it has no zone); None, with a log, if unusable."""
        try:
            at = datetime.fromisoformat(str(data.get("at")))
            if at.tzinfo is None:
                at = at.replace(tzinfo=LOCAL_TZ)
            at.astimezone(LOCAL_TZ)
        except (ValueError, OverflowError):
            log.warning("%s: invalid time %r", command, data.get("at"))
            return None
        if at > self._clock() + timedelta(minutes=5):
            log.warning("%s: time is in the future", command)
            return None
        if at < self._clock() - timedelta(days=MAX_BACKDATE_DAYS):
            log.warning("%s: time is too far in the past", command)
            return None
        return at

    def _manual_visit_id(self, at: datetime) -> str:
        stamp = at.astimezone(LOCAL_TZ)
        visit_id = f"v_{stamp:%Y%m%d_%H%M%S}"
        while self.store.get_visit(visit_id):
            stamp += timedelta(seconds=1)
            visit_id = f"v_{stamp:%Y%m%d_%H%M%S}"
        return visit_id

    async def _cmd_manual_pee(self, data: dict) -> None:
        """`ellie/cmd/manual_pee`: a pee the system missed, possibly on an earlier day."""
        at = self._manual_time("manual_pee", data)
        if at is None:
            return
        visit_id = self._manual_visit_id(at)
        self.store.insert_visit(
            {
                "id": visit_id,
                "source": "manual",
                "started_at": iso(at),
                "ended_at": iso(at),
                "entries": 1,
                "person_present": False,
                "frigate_event_ids": [],
                "state": "LABELLED",
                "final_outcome": "pee",
                "poop_observed": "none",
                "reason": "adicionado manualmente",
                "needs_review": False,
            }
        )
        self.store.insert_label(visit_id, "pee", "manual_add", self._by(data), None)
        await self._visit_decided(visit_id, None)

    async def _cmd_manual_poop(self, data: dict) -> None:
        """`ellie/cmd/manual_poop`: a poop the system missed (ADR-0014 amendment).

        Counted for its day but never "present": the owner saw it and handles it, so it is
        recorded as already cleared. It says nothing about the pee of that visit.
        """
        at = self._manual_time("manual_poop", data)
        if at is None:
            return
        where = data.get("where") if data.get("where") in ("on_pad", "beside_tray") else "none"
        visit_id = self._manual_visit_id(at)
        self.store.insert_visit(
            {
                "id": visit_id,
                "source": "manual",
                "started_at": iso(at),
                "ended_at": iso(at),
                "entries": 1,
                "person_present": False,
                "frigate_event_ids": [],
                "state": "LABELLED",
                "final_outcome": "no_pee",
                "auto_poop": None,
                "final_poop": "poop",
                "poop_cleared_at": iso(at),
                "poop_cleared_by": "owner",
                "poop_observed": where,
                "reason": "cocô adicionado manualmente",
                "needs_review": False,
            }
        )
        self.store.insert_label(visit_id, "poop", "manual_add", self._by(data), None, kind="poop")
        await self._visit_decided(visit_id, None)

    async def _open_watch_visit(self, found: Candidate) -> None:
        """Open a visit for a confirmed change at the pad that Frigate did not report.

        Skipped when a visit or presence already explains it, when something is happening at the
        tray, or when today's cap is used. Gemini decides like for any visit; with validation mode
        on it also lands in the review list.
        """
        local = found.at.astimezone(LOCAL_TZ)
        if self._watch_day != local.date():
            self._watch_day, self._watch_opened = local.date(), 0
        # a poop often comes a minute or two after a visit closed, so for the floor only a visit
        # that overlaps the frames themselves explains it (ADR-0012 amendment)
        window = timedelta(seconds=0 if found.kind == "floor" else WATCH_EXPLAIN_S)
        if self.tracker.active is not None or self.store.activity_between(
            iso(found.before_at - window), iso(found.at + window)
        ):
            log.info("watch: a visit or presence already explains the change, no new visit")
            return
        if self._watch_opened >= WATCH_DAILY_CAP:
            log.warning("watch: %s visits already opened today, ignoring a change", WATCH_DAILY_CAP)
            return
        if self.analysers.get(self.cfg["deciding_analyser"]) is None:
            log.warning("watch: no analyser configured, the change is not analysed")
            return
        stamp = local
        visit_id = f"v_{stamp:%Y%m%d_%H%M%S}"
        while self.store.get_visit(visit_id):
            stamp += timedelta(seconds=1)
            visit_id = f"v_{stamp:%Y%m%d_%H%M%S}"
        directory = self.evidence.visit_dir(visit_id)
        self.evidence.write(directory / "before.jpg", found.before_jpeg)
        self.evidence.write(directory / "after.jpg", found.after_jpeg)
        self.store.insert_visit(
            {
                "id": visit_id,
                "source": "auto",
                "started_at": iso(found.at),
                "ended_at": iso(found.at),
                "entries": 1,
                "person_present": False,
                "frigate_event_ids": [],
                "before_age_s": int((found.at - found.before_at).total_seconds()),
                "state": "ANALYSING",
                "poop_observed": "none",
                "reason": (
                    "mancha escura fora do tapete sem detecção do Frigate"
                    if found.kind == "floor"
                    else "mudança no tapete sem detecção do Frigate"
                ),
                "needs_review": False,
            }
        )
        self._watch_opened += 1
        log.info("watch: %s visit %s opened (pad %.2f, floor %.2f, peak %.2f)", found.kind,
                 visit_id, found.pad_step, found.ctrl_step, found.floor_peak)  # fmt: skip
        self._spawn(self._analyse(visit_id, retry=True))  # the frames are already on disk
        await self.publish_state()

    async def _cmd_reject_last(self, data: dict) -> None:
        """`ellie/cmd/reject_last`: undo the most recent pee counted today."""
        now = self._clock()
        last = Counting(self.store).last_pee(now, lookback_days=2)
        if last is None or local_date(last["started_at"]) != local_date(iso(now)):
            log.warning("reject_last: no counted pee today")
            return
        await self._apply_label(last, "no_pee", "reject_last", self._by(data))

    async def _cmd_reanalyse(self, data: dict) -> None:
        """`ellie/cmd/reanalyse`: run the deciding analyser again on the stored images."""
        visit_id = data.get("visit_id")
        visit = self.store.get_visit(visit_id) if isinstance(visit_id, str) and visit_id else None
        if visit is None:
            log.warning("reanalyse: unknown visit")
            return
        directory = self.evidence.visit_dir(visit["id"])
        if not (directory / "before.jpg").exists() or not (directory / "after.jpg").exists():
            log.warning("reanalyse %s: the stored images are missing", visit["id"])
            return
        if visit["id"] in self._inflight:
            log.info("reanalyse %s: an analysis is already running", visit["id"])
            return
        failed = visit["final_outcome"] == "unknown" and not self.store.labels_for(visit["id"])
        if failed:  # never decided: this is a retry
            self.store.update_visit(visit["id"], state="ANALYSING")
            self._spawn(self._analyse(visit["id"], retry=True))
        else:  # already decided or labelled: record the new answer, change nothing else
            self._spawn(self._rerun(visit["id"], data.get("analyser")))

    async def _apply_label(self, visit: dict, label: str, channel: str, by: str | None) -> None:
        visit_id = visit["id"]
        if visit["state"] not in ("DECIDED", "LABELLED"):
            log.warning("label for %s ignored: the visit is still %s", visit_id, visit["state"])
            return
        self.store.insert_label(visit_id, label, channel, by, visit["final_outcome"])
        self.store.update_visit(visit_id, final_outcome=label, needs_review=False, state="LABELLED")
        visit = self.store.get_visit(visit_id)
        analyses = self.store.analyses_for(visit_id)
        night = ((analyses[-1]["result"] or {}).get("night_mode")) if analyses else None
        try:
            self.evidence.export_dataset(visit, label, night=night)
        except OSError:
            log.exception("dataset export failed for %s", visit_id)
        await self._publish("ellie/review/resolved", json.dumps({"visit_id": visit_id}), False)
        await self._visit_decided(visit_id, None)

    # --- poop (ADR-0014) ---

    @staticmethod
    def _poop_where(visit: dict) -> str | None:
        return (
            visit["poop_observed"] if visit["poop_observed"] in ("on_pad", "beside_tray") else None
        )

    def _poop_fields(self, result) -> dict:
        """auto_poop and final_poop from an analyser answer (None final = the owner is asked)."""
        outcome = decide_poop(result, self.cfg["confidence_threshold"])
        final = outcome if outcome in ("poop", "no_poop") else None
        return {"auto_poop": outcome, "final_poop": "unknown" if outcome == "unknown" else final}

    def _follow_poop(self, visit: dict) -> None:
        """Hand a counted, uncleared poop to the pad watch so it can see when it is gone."""
        if self.watch is None:
            return
        directory = self.evidence.visit_dir(visit["id"])
        before, after = directory / "before.jpg", directory / "after.jpg"
        if before.exists() and after.exists():
            self.watch.track_poop(visit["id"], before.read_bytes(), after.read_bytes())

    def _follow_open_poops(self) -> None:
        cutoff = iso(self._clock() - timedelta(days=POOP_TRACK_LOOKBACK_DAYS))
        for visit in self.store.open_poops():
            if visit["started_at"] >= cutoff:
                self._follow_poop(visit)

    async def _clear_poops(
        self, by: str, visit_id: str | None = None, where: str | None = None
    ) -> int:
        """Mark open poops as cleared (`by`: watch, pad_change, owner or timeout)."""
        stamp, cleared = iso(self._clock()), 0
        for visit in self.store.open_poops():
            if (visit_id and visit["id"] != visit_id) or (
                where and self._poop_where(visit) != where
            ):
                continue
            self.store.update_visit(visit["id"], poop_cleared_at=stamp, poop_cleared_by=by)
            if self.watch is not None:
                self.watch.untrack_poop(visit["id"])
            cleared += 1
        if cleared:
            log.info("poop cleared (%s): %s visit(s)", by, cleared)
            await self.publish_state()
        return cleared

    async def _expire_poops(self, now: datetime) -> None:
        limit = timedelta(hours=POOP_PRESENT_TIMEOUT_H)
        for visit in self.store.open_poops():
            seen = datetime.fromisoformat(visit["ended_at"] or visit["started_at"])
            if now - seen >= limit:
                await self._clear_poops("timeout", visit_id=visit["id"])

    async def _cmd_poop_clear(self, data: dict) -> None:
        """`ellie/cmd/poop_clear`: "Limpei". One visit, or every open poop without `visit_id`."""
        visit_id = data.get("visit_id")
        if visit_id is not None and not isinstance(visit_id, str):
            log.warning("ignored poop_clear command: bad visit_id")
            return
        await self._clear_poops("owner", visit_id=visit_id or None)

    async def _apply_poop_label(
        self, visit: dict, label: str, channel: str, by: str | None
    ) -> None:
        visit_id = visit["id"]
        if visit["state"] not in ("DECIDED", "LABELLED"):
            log.warning(
                "poop label for %s ignored: the visit is still %s", visit_id, visit["state"]
            )
            return
        self.store.insert_label(visit_id, label, channel, by, visit["final_poop"], kind="poop")
        fields: dict[str, Any] = {"final_poop": label}
        if label == "no_poop" and visit["poop_cleared_at"] is None:
            fields |= {"poop_cleared_at": iso(self._clock()), "poop_cleared_by": "owner"}
        elif label == "poop" and visit["final_poop"] != "poop":  # a new poop, present until cleared
            fields |= {"poop_cleared_at": None, "poop_cleared_by": None}
        self.store.update_visit(visit_id, **fields)
        visit = self.store.get_visit(visit_id)
        if self.watch is not None:
            self.watch.untrack_poop(visit_id)
        if label == "poop" and visit["poop_cleared_at"] is None:
            self._follow_poop(visit)
        payload = json.dumps({"visit_id": visit_id, "kind": "poop"})
        await self._publish("ellie/review/resolved", payload, False)
        await self._visit_decided(visit_id, None)

    async def _ask_poop(self, visit: dict) -> None:
        """Ask the owner about a poop Gemini was unsure of (once per visit)."""
        if (
            visit["auto_poop"] != "uncertain"
            or visit["final_poop"] is not None
            or visit["id"] in self._poop_asked
        ):
            return
        self._poop_asked.add(visit["id"])
        await self._request_review(visit, "poop")

    async def _rerun(self, visit_id: str, analyser_id: object = None) -> None:
        if visit_id in self._inflight:
            return
        self._inflight.add(visit_id)
        try:
            await self._rerun_inner(visit_id, analyser_id)
        finally:
            self._inflight.discard(visit_id)

    async def _rerun_inner(self, visit_id: str, analyser_id: object = None) -> None:
        """Rerun one analyser (default: the deciding one) and keep its answer as a shadow."""
        key = analyser_id if isinstance(analyser_id, str) else self.cfg["deciding_analyser"]
        analyser = self.analysers.get(key)
        directory = self.evidence.visit_dir(visit_id)
        if analyser is None:
            return
        during = directory / "during.jpg"
        started = time.monotonic()
        try:
            result = await analyser.analyse(
                (directory / "before.jpg").read_bytes(),
                (directory / "after.jpg").read_bytes(),
                during.read_bytes() if during.exists() else None,
            )
        except AnalyserError as exc:
            log.warning("reanalyse %s failed: %s", visit_id, exc)
            return
        outcome = decide_visit(result, self.cfg["confidence_threshold"])
        log.info(
            "reanalyse %s with %s: %s (pee %s %.2f, poop %s %.2f) %s",
            visit_id, analyser.id, outcome, result.pee, result.pee_confidence,
            result.poop, result.poop_confidence, result.reason,
        )  # fmt: skip
        self.store.insert_analysis(
            visit_id, analyser.id, "shadow", result.model_dump(exclude_none=True), outcome,
            int((time.monotonic() - started) * 1000), None, getattr(analyser, "last_usage", None),
        )  # fmt: skip
        await self._visit_decided(visit_id, None)

    # --- visit flow ---

    async def _apply(self, actions: list[Action]) -> None:
        for action in actions:
            visit = action.visit
            if action.kind == "capture_after":
                self._spawn(self._analyse(visit["id"]))  # first: a store error must not strand it
            try:
                if action.kind == "started":
                    self._insert_started(visit)
                elif action.kind in ("updated", "capture_after"):
                    self._update_tracked(visit)
                elif action.kind == "unknown":
                    self._update_tracked(visit)
                    await self._fail(visit["id"], action.reason or "analysis_failed")
            except sqlite3.Error:
                log.exception("could not persist %s for %s", action.kind, visit["id"])
        if actions:
            await self.publish_state()

    def _insert_started(self, visit: dict) -> None:
        started = datetime.fromisoformat(visit["started_at"])
        reference = self.reference.before(started)
        age = int((started - reference[1]).total_seconds()) if reference else None
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
        if visit_id in self._inflight:
            log.info("analysis of %s already running, skipped", visit_id)
            return
        self._inflight.add(visit_id)
        try:
            await self._analyse_inner(visit_id, retry)
        except Exception:  # a visit must never stay ANALYSING forever
            log.exception("analysis of %s crashed", visit_id)
            await self._fail(visit_id, "analysis_failed")
        finally:
            self._inflight.discard(visit_id)

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
            if self.watch is not None:
                self.watch.rebase(after)  # what changed at the pad is explained by this visit
        if not before_path.exists():
            return await self._fail(visit_id, "no_reference")
        analyser = self.analysers.get(self.cfg["deciding_analyser"])
        if analyser is None:
            return await self._fail(visit_id, "analyser_not_configured")

        events = self.store.get_visit(visit_id)["frigate_event_ids"]
        during = await self._fetch_snapshot(events[0]) if events else None
        if during:
            try:
                self.evidence.write(directory / "during.jpg", during)
            except OSError:
                log.warning(
                    "could not store the during photo of %s, analysing without it", visit_id
                )

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
        fields = {
            "auto_outcome": outcome,
            "poop_observed": result.poop,
            "reason": result.reason,
            "deciding_analyser": analyser.id,
        }
        if not self.store.labels_for(visit_id, "poop"):  # the owner's poop answer wins too
            fields |= self._poop_fields(result) | {"poop_cleared_at": None, "poop_cleared_by": None}
        if not self.store.labels_for(visit_id):  # an owner's answer always wins
            fields |= {
                "state": "DECIDED",
                "final_outcome": outcome if decided else None,
                "needs_review": (not decided) or bool(self.cfg["validation_mode"]),
            }
        self.store.update_visit(visit_id, **fields)
        visit = self.store.get_visit(visit_id)
        if visit["final_poop"] == "poop" and visit["poop_cleared_at"] is None:
            self._follow_poop(visit)
        if (
            result.pad_replaced and visit["person_present"]
        ):  # a person was there: the pad was swapped
            await self._record_pad_change(
                "visit", visit["ended_at"] or visit["started_at"], visit_id=visit_id
            )
        if result.poop != "none" and events and self._download_clip:
            self._spawn(self._download_clip(events[0], visit_id))  # raw material for spec 004/005
        if retry and decided:
            await self._publish("ellie/review/resolved", json.dumps({"visit_id": visit_id}), False)
        await self._visit_decided(visit_id, "uncertain" if outcome == "uncertain" else None)

    @staticmethod
    def _reason_text(code: str) -> str:
        return REASONS.get(code, REASONS["analysis_failed"])

    @staticmethod
    def _fixable() -> dict:
        return FIXABLE

    def _reason_for(self, error: str) -> str:
        for marker, (code, _title, _message) in FIXABLE.items():
            if marker in error:
                return code
        return "analysis_failed"

    async def _fail(self, visit_id: str, code: str) -> None:
        current = self.store.get_visit(visit_id)
        if (
            current
            and current["state"] in ("DECIDED", "LABELLED")
            and current["auto_outcome"] not in (None, "unknown")
        ):
            log.warning("visit %s was already decided, not turning it into unknown", visit_id)
            return
        poop = {} if self.store.labels_for(visit_id, "poop") else {
            "auto_poop": "unknown", "final_poop": "unknown",
        }  # fmt: skip
        self.store.update_visit(
            visit_id,
            state="DECIDED",
            auto_outcome="unknown",
            final_outcome="unknown",
            needs_review=True,
            reason=REASONS.get(code, REASONS["analysis_failed"]),
            **poop,
        )
        for fix_code, title, message in FIXABLE.values():
            if fix_code == code:
                await self._alert(code, title, message)
                break
        await self._visit_decided(visit_id, "unknown")

    async def _visit_decided(self, visit_id: str, review_kind: str | None) -> None:
        visit = self.store.get_visit(visit_id)
        log.info(
            "visit %s: auto=%s final=%s poop=%s entries=%s on_pad=%ss (%s)",
            visit_id, visit["auto_outcome"], visit["final_outcome"], visit["poop_observed"],
            visit["entries"], visit["on_pad_s"], visit["reason"],
        )  # fmt: skip
        await self._publish(f"ellie/visit/{visit_id}", json.dumps(visit, ensure_ascii=False), False)
        if review_kind:
            await self._request_review(visit, review_kind)
        await self._ask_poop(visit)
        await self.publish_state()

    async def _request_review(self, visit: dict, kind: str) -> None:
        hhmm = (local_text(visit["started_at"]) or "")[11:16]
        after = self.evidence.visit_dir(visit["id"]) / "after.jpg"
        if kind == "poop":
            title, message = "Ellie: fez cocô?", f"Visita às {hhmm} — {visit['reason']}"
        elif kind == "uncertain":
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
            pad=self.pad_state(self._clock()),
        )
        await self._publish(discovery.STATE_TOPIC, json.dumps(state, ensure_ascii=False), True)
        await self._publish_images(state)

    async def _publish_images(self, state: dict) -> None:
        """Send a photo to its image entity only when it changed (a few hundred KB, retained)."""
        topics = discovery.IMAGE_TOPICS
        last = state["last_pee"]
        current = (state["review_queue"]["items"] or [None])[0]
        wanted = {
            topics["review_before"]: (current and current["visit_id"], "before.jpg"),
            topics["review_after"]: (current and current["visit_id"], "after.jpg"),
        }
        for topic, (visit_id, name) in wanted.items():
            path = self.evidence.visit_dir(visit_id) / name if visit_id else None
            if not visit_id or self._image_keys.get(topic) == visit_id:
                continue
            if path is None or not path.exists():
                continue
            sent = await self._publish(topic, path.read_bytes(), True)
            if sent is not False:  # a dropped publish (no MQTT connection yet) must be retried
                self._image_keys[topic] = visit_id
        top = self._top_photo(last)
        top_topic = topics["last_pee_photo"]
        if (
            top is not None
            and self._image_keys.get(top_topic) != top[0]
            and await self._publish(top_topic, top[1], True) is not False
        ):
            self._image_keys[top_topic] = top[0]
        poop_id = (state["poop_present"] or {}).get("visit_id") or (state["last_poop"] or {}).get(
            "visit_id"
        )
        poop_path = self.evidence.visit_dir(poop_id) / "after.jpg" if poop_id else None
        poop_topic = topics["poop_photo"]
        if (
            poop_path is not None
            and poop_path.exists()
            and self._image_keys.get(poop_topic) != poop_id
            and await self._publish(poop_topic, poop_path.read_bytes(), True) is not False
        ):
            self._image_keys[poop_topic] = poop_id
        pad_path, _ = self.pad_photo()  # the photo of the current pad (see pads_flow.pad_photo)
        pad_topic = topics["pad_photo"]
        if (
            pad_path is not None
            and self._image_keys.get(pad_topic) != str(pad_path)
            and await self._publish(pad_topic, pad_path.read_bytes(), True) is not False
        ):
            self._image_keys[pad_topic] = str(pad_path)

    def _top_photo(self, last: dict | None) -> tuple[str, bytes] | None:
        """(key, jpeg) for the big "Foto do tapete": the latest of the last pee and a pad change.

        After a pad change the clean pad replaces the stained photo until the next pee. A manual
        mark has no photo, so it waits for a reference frame taken after the mark (the owner may
        tap right after swapping, when the newest reference still shows the old pad).
        """
        change = self.store.current_pad_change()
        if change is not None and (
            last is None
            or datetime.fromisoformat(change["at"]) > datetime.fromisoformat(last["at"])
        ):
            photo = self._change_photo(change)
            if photo is not None:
                return f"pad:{change['id']}", photo.read_bytes()
            reference = self.reference.latest()
            if reference is not None and reference[1] >= datetime.fromisoformat(change["at"]):
                return f"pad:{change['id']}", reference[0]
            return None  # keep what is shown; the next minute's reference will do
        if last is not None:
            path = self.evidence.visit_dir(last["visit_id"]) / "after.jpg"
            if path.exists():
                return last["visit_id"], path.read_bytes()
        return "placeholder", _placeholder_jpeg()  # a manual or undone pee has no photo

    async def _publish(self, topic: str, payload: str | bytes, retain: bool) -> object:
        return await self._publish_fn(topic, payload, retain)

    def _spawn(self, coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    def _by(data: dict) -> str | None:
        """Who answered: only a short string is kept (the field is informational)."""
        by = data.get("by")
        return by[:80] if isinstance(by, str) else None

    @staticmethod
    def _count(payload: bytes, handler: Callable[[int], list[Action]]) -> list[Action]:
        try:
            count = int(float(payload.decode().strip()))
        except (ValueError, OverflowError):
            log.warning("ignored non-numeric count %r", payload[:20])
            return []
        return handler(count)

    def _dog_event_id(self, payload: bytes) -> str | None:
        try:
            after = (json.loads(payload) or {}).get("after") or {}
            if after.get("camera") != self.camera or after.get("label") != "dog":
                return None
            if not {"pad_zone", "tray_area"} & set(after.get("entered_zones") or []):
                return None
            event_id = after.get("id")
        except (ValueError, AttributeError, TypeError):  # not JSON, not UTF-8, wrong shape
            return None
        # the id ends up in a Frigate API URL: accept only the shape Frigate really uses
        return event_id if isinstance(event_id, str) and EVENT_ID.match(event_id) else None
