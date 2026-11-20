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

from ellie import discovery, review_images
from ellie.analysers.base import Analyser, AnalyserError
from ellie.census import FOUND_PEE_REASON, found_pee_times, is_found_pee, missing_pees
from ellie.counting import LOCAL_TZ, Counting, local_date
from ellie.decide import decide_pad_change, decide_poop, decide_visit
from ellie.evidence import Evidence
from ellie.frames import FrameError, ReferenceKeeper
from ellie.pads import PresenceTracker
from ellie.pads_flow import FOUND_POOP_WINDOW, PadsMixin
from ellie.settings import RUNTIME_DEFAULTS, clamp_setting
from ellie.small_person import SmallPersonFilter
from ellie.state import build_state, local_text
from ellie.store import Store
from ellie.visits import Action, VisitTracker, iso
from ellie.watch import Candidate, PadWatch, floor_blob, pee_stain

log = logging.getLogger(__name__)

Publish = Callable[[str, str | bytes, bool], Awaitable[object]]
EVENT_ID = re.compile(
    r"^[0-9]+\.[0-9]+-[A-Za-z0-9]+$"
)  # Frigate event id, e.g. 1793288511.990085-yohln0
MAX_BACKDATE_DAYS = 366  # manual entries and pad changes older than this are rejected
LABEL_CHANNELS = ("notification", "dashboard")
REFERENCE_CLEAR_S = 10  # the area must be clear this long before a reference frame is kept
WATCH_DAILY_CAP = 6  # visits the pad watch may open per local day (worst case about 0.008 EUR)
LASTING_QUIET = timedelta(minutes=30)  # no activity this long before a lasting-change confirm
WATCH_EXPLAIN_S = 180  # a visit or presence this close to a candidate already explains it
# A pee stain spreads for a few minutes while the pad absorbs it (the night minutes after 2026-11-13
# 03:46 settle within 5 min) and Ellie never pees twice in a short time (shortest gap between two
# real pees in 45: 84 min). So growth within SPREAD_MIN of a pee is that pee; later it is a new one
# (ADR-0020). SAME_PEE is the window in which two entries are one and the same pee or poop.
# 2026-11-19: a big pee was still spreading 43 min later (12:13 → 12:56, owner asked for nothing);
# shortest gap between real pees in 52 is 87 min, so the window is 60 min (ADR-0020 amendment).
# Every skipped question is logged to SPREAD_LOG to check the number against the owner's list.
SPREAD_MIN = timedelta(minutes=60)
SPREAD_LOG = "spread_skips.jsonl"
SPREAD_PHOTO = "after_spread.jpg"  # the latest frame of a pee that kept spreading (display only)
SAME_EVENT = timedelta(minutes=30)
FOUND_MATCH = timedelta(hours=6)  # a manual pee takes over a found pee of the same pad this close
CAMERA_ALERT_AFTER = timedelta(minutes=15)  # go2rtc failing this long: tell the owner
ALERT_EVERY_S = 6 * 3600
POOP_PRESENT_TIMEOUT_H = 12  # a counted poop nobody cleared stops being "present" after this
POOP_TRACK_LOOKBACK_DAYS = 2  # after a restart, open poops this recent are followed again
POOP_LABELS = ("poop", "no_poop")
DURING_OWN = "during_own.jpg"  # own frame while she is there, used when Frigate has no snapshot
AFTER_PERSON = "after_person.jpg"  # the late frame when a person came before it (after = exit)
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


DOUBT_PHOTOS = ("before", "after", "both", "none")
DOUBT_REASONS = ("glare", "old_stain", "other")


def _doubt(data: dict) -> dict | None:
    """`doubt: {photos, reason}` of a label command (owner answered unsure), else None."""
    doubt = data.get("doubt")
    if not isinstance(doubt, dict):
        return None
    photos, reason = doubt.get("photos") or "none", doubt.get("reason") or "other"
    if photos not in DOUBT_PHOTOS or reason not in DOUBT_REASONS:
        log.warning("label: ignored an invalid doubt %r", doubt)
        return None
    return {"photos": photos, "reason": reason}


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
        fetch_fallback_frame: Callable[[], Awaitable[bytes]] | None = None,
    ) -> None:
        self.watch = watch
        self._fetch_fallback = fetch_fallback_frame
        self._camera_failing_since: datetime | None = None
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
        # tray_area counts as Frigate sends them, and the persons that are really her (ADR-0019)
        self.small_person = SmallPersonFilter(camera)
        self._raw_area = {"dog": 0, "person": 0}
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
        self._presence_retakes: dict[str, int] = {}  # occluded presences: retakes done so far
        self._pad_full_sent: set[str] = set()  # visits whose "pad full" notice went out
        self._presence_chain: tuple[bytes, datetime] | None = None  # before + end of last presence
        self._census_lock = asyncio.Lock()
        self._person_checks: dict[str, Any] = {}  # visit id -> pad answer for the person after her

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
        elif topic in ("frigate/tray_area/dog", "frigate/tray_area/person"):
            kind = topic.rsplit("/", 1)[1]
            raw = self._count(payload, lambda n: [n])
            if raw:
                self._raw_area[kind] = max(raw[0], 0)
                actions, more = self._area_counts(now, (kind,))
                presence_actions += more
        elif topic == "frigate/tray_pad/dog":
            actions = self._count(payload, lambda n: self.tracker.on_tray_pad_dog(n, now))
        elif topic == "frigate/events":
            if self.small_person.on_event(payload, now):
                actions, more = self._area_counts(now, ("dog", "person"))
                presence_actions += more
            event_id = self._dog_event_id(payload)
            actions += self.tracker.on_event(event_id, now) if event_id else []
            held = self._person_event(payload)
            if held is not None:
                presence_actions += self.presence.on_person_event(*held, now)
        elif topic == "frigate/available":
            actions = self.tracker.on_frigate_available(
                payload.decode(errors="replace").strip() == "online", now
            )
            self.presence.reset(now)
            self.small_person.reset()
            self._raw_area = {"dog": 0, "person": 0}  # stale after an outage, like presence's
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
                crop, fallback = await self._frame()
                # a lower-resolution fallback frame would look like a change to the watch
                watching = self.watch is not None and not fallback
                confirmed = self.watch.observe(crop) if watching else None
                self.reference.update(crop)
                if confirmed is not None:  # a change Frigate never reported (ADR-0012)
                    await self._open_watch_visit(confirmed)
                for visit_id in self.watch.take_cleared() if self.watch is not None else ():
                    await self._clear_poops("watch", visit_id=visit_id)  # the blob is gone
            except (FrameError, OSError) as exc:
                log.warning("reference refresh failed: %s", exc)
        for presence_id in self._due_retakes(now):
            self._spawn(self._analyse_presence(presence_id))  # a new after frame, same before
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
            await asyncio.sleep(0)  # let finished tasks run their discard callback (no busy loop)

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
            await self._apply_label(visit, label, channel, self._by(data), _doubt(data))

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

    @staticmethod
    def _closest(visits: list[dict], at: datetime) -> dict | None:
        return min(
            visits,
            key=lambda v: abs((datetime.fromisoformat(v["started_at"]) - at).total_seconds()),
            default=None,
        )

    def _counted_near(self, at: datetime, kind: str) -> dict | None:
        """The closest counted pee (or poop) within SAME_EVENT of `at`: the same event, twice.

        For pees a real one wins over a census pee, so a manual entry next to a real pee is
        never turned into a second pee by taking over a found one.
        """
        near = self.store.visits_started_between(iso(at - SAME_EVENT), iso(at + SAME_EVENT))
        if kind == "pee":
            pees = [v for v in near if v["final_outcome"] == "pee"]
            real = [v for v in pees if not is_found_pee(v)]
            return self._closest(real or pees, at)
        return self._closest([v for v in near if v["final_poop"] == "poop"], at)

    def _found_pees_covering(self, at: datetime, census_from: str | None = None) -> list[dict]:
        """Census pees still counted whose window (found_from .. the census visit) holds `at`."""
        stamp = iso(at)
        found = []
        for v in self.store.visits_started_between(iso(at - FOUND_MATCH), iso(at + FOUND_MATCH)):
            if not is_found_pee(v) or v["final_outcome"] != "pee":
                continue
            census = self.store.get_visit(v["found_by"])
            window_end = census["started_at"] if census else v["started_at"]
            if (v["found_from"] or "") <= stamp <= window_end and (
                census_from is None or window_end >= census_from
            ):
                found.append(v)
        return found

    def _found_pee_for(self, at: datetime) -> dict | None:
        """The census pee whose window holds `at`, closest first (FOUND_MATCH at most)."""
        return self._closest(self._found_pees_covering(at), at)

    async def _cmd_manual_pee(self, data: dict) -> None:
        """`ellie/cmd/manual_pee`: a pee the system missed, possibly on an earlier day."""
        at = self._manual_time("manual_pee", data)
        if at is None:
            return
        same = self._counted_near(at, "pee")
        if same is not None and not is_found_pee(same):
            log.warning("manual_pee at %s: the pee of %s is already counted (Ellie never pees "
                        "twice within %s min), not added", iso(at), same["id"],
                        int(SAME_EVENT.total_seconds() // 60))  # fmt: skip
            return
        found = same if same is not None else self._found_pee_for(at)
        if found is not None:  # the owner knows the real time of a pee the census added
            self.store.update_visit(
                found["id"], started_at=iso(at), ended_at=iso(at), needs_review=False,
                state="LABELLED", found_by=None, found_from=None,
                reason="adicionado manualmente (era xixi achado no tapete)",
            )  # fmt: skip
            self.store.insert_label(found["id"], "pee", "manual_add", self._by(data), "pee")
            log.info("manual_pee at %s takes the place of the found pee %s", iso(at), found["id"])
            await self._visit_decided(found["id"], None)
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
        same = self._counted_near(at, "poop")
        if same is not None:
            log.warning("manual_poop at %s: the poop of %s is already counted, not added",
                        iso(at), same["id"])  # fmt: skip
            return
        walk = data.get("where") == "walk"  # seen on a walk: counted, but nothing at the tray
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
                "reason": "cocô no passeio" if walk else "cocô adicionado manualmente",
                "poop_walk": walk,
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
        # a presence never counts a poop, and Ellie seen as a person for 2 s is such a presence
        # (2026-11-14 09:59): only a visit explains a floor change (ADR-0020)
        tables = ("visits",) if found.kind == "floor" else ("visits", "presences")
        if self.tracker.active is not None or self.store.activity_between(
            iso(found.before_at - window), iso(found.at + window), tables
        ):
            log.info("watch: a visit or presence already explains the change, no new visit")
            return
        # A held frame was taken after the activity window checked above, so a person who left
        # before it cannot explain the change (2026-11-20 12:38 pee and 12:45 poop, Frigate saw
        # no dog, dropped here because the owner had been at the tray at 12:30-12:34).
        if (
            found.lasting
            and not found.held
            and self._recent_activity(found.before_at - LASTING_QUIET)
        ):
            # a tray lifted, cleaned and put back, or a puddle drying, also leaves a lasting change
            # (2026-11-09 12:59, 13:17, 13:28): only a quiet tray may confirm this way
            log.info("watch: lasting change right after activity at the tray, no new visit")
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

    def _recent_activity(self, since: datetime) -> bool:
        """A visit, a presence or a pad change since `since`."""
        now = iso(self._clock())
        if self.store.activity_between(iso(since), now):
            return True
        return any(c["at"] >= iso(since) for c in self.store.pad_changes(3))

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

    async def _apply_label(
        self, visit: dict, label: str, channel: str, by: str | None, doubt: dict | None = None
    ) -> None:
        visit_id = visit["id"]
        if visit["state"] not in ("DECIDED", "LABELLED"):
            log.warning("label for %s ignored: the visit is still %s", visit_id, visit["state"])
            return
        self.store.insert_label(visit_id, label, channel, by, visit["final_outcome"], doubt=doubt)
        if doubt:
            log.info("visit %s: answered %s in doubt (photos %s, %s)", visit_id, label,
                     doubt["photos"], doubt["reason"])  # fmt: skip
        self.store.update_visit(visit_id, final_outcome=label, needs_review=False, state="LABELLED")
        if label == "pee" and visit["final_outcome"] != "pee" and not is_found_pee(visit):
            await self._absorb_found_pee(visit)
        visit = self.store.get_visit(visit_id)
        analyses = self.store.analyses_for(visit_id)
        night = ((analyses[-1]["result"] or {}).get("night_mode")) if analyses else None
        try:
            self.evidence.export_dataset(visit, label, night=night, doubt=doubt)
        except OSError:
            log.exception("dataset export failed for %s", visit_id)
        await self._publish("ellie/review/resolved", json.dumps({"visit_id": visit_id}), False)
        await self._visit_decided(visit_id, None)

    async def _absorb_found_pee(self, visit: dict) -> None:
        """A visit the owner turned into a pee was one of the stains a census counted as missing:
        drop that census pee, so one stain stays one pee (ADR-0020)."""
        at = datetime.fromisoformat(visit["started_at"])
        found = self._closest(self._found_pees_covering(at, census_from=visit["started_at"]), at)
        if found is None:
            return
        hhmm = (local_text(visit["started_at"]) or "")[11:16]
        self.store.update_visit(
            found["id"], final_outcome="no_pee", needs_review=False,
            reason=f"{FOUND_PEE_REASON}: era a visita das {hhmm}",
        )  # fmt: skip
        log.info(
            "the found pee %s was the pee of %s, not counted any more", found["id"], visit["id"]
        )
        await self._visit_decided(found["id"], None)

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
                context=self._pee_context(self.store.get_visit(visit_id)),
            )
        except AnalyserError as exc:
            log.warning("reanalyse %s failed: %s", visit_id, exc)
            return
        outcome = decide_visit(result, self.cfg["confidence_threshold"])
        log.info(
            "reanalyse %s with %s: %s (pee %s %.2f, poop %s %.2f, stains %s) %s",
            visit_id, analyser.id, outcome, result.pee, result.pee_confidence,
            result.poop, result.poop_confidence, result.pad_stains, result.reason,
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
            if action.kind == "dog_left":
                self._spawn(self._capture_frame(visit["id"], f"left_{visit['entries']}.jpg"))
                continue
            if action.kind == "during":
                self._spawn(self._capture_frame(visit["id"], DURING_OWN))
                continue
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
        self._presence_chain = None  # a dog visit between two presences breaks the chain
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

    def _second_poop_opinion(self, visit_id: str, before_path, after_path) -> dict:
        """Gemini saw no poop, but a new dark blob on the floor says otherwise: ask the owner."""
        peak = floor_blob(before_path.read_bytes(), after_path.read_bytes())
        if peak is None:
            return {}
        log.info("visit %s: Gemini saw no poop but the floor has a new dark blob (peak %.2f), "
                 "asking the owner", visit_id, peak)  # fmt: skip
        return {"auto_poop": "uncertain", "final_poop": None}

    @staticmethod
    def _before_at(visit: dict) -> datetime:
        """When the visit's BEFORE frame was taken."""
        started = datetime.fromisoformat(visit["started_at"])
        return started - timedelta(seconds=visit.get("before_age_s") or 0)

    def _last_pee_before(self, visit: dict) -> dict | None:
        """The last counted pee that started before this visit (this visit excluded).

        A manual pee just before the visit is the owner's correction of this same visit, not an
        earlier pee (Ellie never pees twice within SAME_EVENT).
        """
        started = datetime.fromisoformat(visit["started_at"])
        same = iso(started - SAME_EVENT)
        start = iso(started - timedelta(days=3))
        pees = [
            p for p in self.store.pees_list_between(start, visit["started_at"])
            if p["id"] != visit["id"] and not (p["source"] == "manual" and p["started_at"] >= same)
        ]  # fmt: skip
        return pees[-1] if pees else None

    def _pad_change_for(self, stamp: str) -> dict | None:
        """The pad change in effect at `stamp` (the newest not undone at or before it)."""
        changes = [c for c in self.store.pad_changes(50) if c["undone_at"] is None]
        return max((c for c in changes if c["at"] <= stamp), key=lambda c: c["at"], default=None)

    def _pee_context(self, visit: dict | None) -> str | None:
        """CONTEXT for the pee prompt: the history of the pad up to BEFORE (pee_v5, pee_v6).

        Pad age, every pee and poop counted on this pad (time, how it was counted), poops not
        cleared yet, people at the tray since the last event, and the minutes from the last pee
        to BEFORE. Written facts only: Gemini still judges the images (owner, 2026-11-20).
        """
        if not visit:
            return None
        before_at = self._before_at(visit)
        before_iso = iso(before_at)
        hhmm = lambda at: (local_text(at) or "")[11:16]
        parts = []
        change = self._pad_change_for(visit["started_at"])
        if change:
            hours = (before_at - datetime.fromisoformat(change["at"])).total_seconds() / 3600
            parts.append(
                f"The pad was put in {max(hours, 0):.1f} hours before BEFORE (at {hhmm(change['at'])})."
            )
        start = change["at"] if change else iso(before_at - timedelta(days=1))
        earlier = [
            v for v in self.store.visits_started_between(start, before_iso)
            if v["id"] != visit["id"]
        ]  # fmt: skip
        pees = [v for v in earlier if v.get("final_outcome") == "pee"]
        how = {"manual": "added by the owner"}
        if pees:
            listed = ", ".join(
                f"{hhmm(p['started_at'])} ({how.get(p.get('source'), 'found by a stain count' if p.get('found_by') else 'seen by the camera')})"
                for p in pees
            )
            parts.append(
                f"{len(pees)} pee(s) were counted on this pad before BEFORE: {listed}. Their marks "
                "are old; they may have merged into fewer stains."
            )
        poops = [v for v in earlier if v.get("final_poop") == "poop"]
        for p in poops:
            where = {"on_pad": "on the pad", "beside_tray": "beside the tray"}.get(
                p.get("poop_observed"), "near the tray"
            )
            state = (
                f"removed at {hhmm(p['poop_cleared_at'])}"
                if p.get("poop_cleared_at")
                else "not removed yet (it is old if you see it in BEFORE)"
            )
            parts.append(f"A poop {where} was counted at {hhmm(p['started_at'])}, {state}.")
        last = self._last_pee_before(visit)
        if last is None or (change and last["started_at"] < change["at"]):
            parts.append("No pee was counted on this pad yet.")
        else:
            last_at = datetime.fromisoformat(last["ended_at"] or last["started_at"])
            minutes = (before_at - last_at).total_seconds() / 60
            if minutes < 0:
                parts.append("The last counted pee came after BEFORE was taken.")
            else:
                parts.append(f"The last counted pee was {minutes:.0f} minutes before BEFORE.")
        since = last["started_at"] if last is not None and last["started_at"] > start else start
        people = self.store.presences_between(since, before_iso)
        if people:
            times = ", ".join(f"{hhmm(p['started_at'])}-{hhmm(p['ended_at'])}" for p in people[-3:])
            parts.append(
                f"A person was at the tray after that ({times}) and did not change the pad: the "
                "tray may have been moved or cleaned of poop."
            )
        return " ".join(parts)

    def _second_pee_opinion(
        self, visit_id: str, visit: dict, before_path, after_path
    ) -> float | None:
        """Gemini saw no pee, but a new stain covers part of the pad: the stained share, else None.

        Not used when a pee came less than SPREAD_MIN (60 min) before the BEFORE frame: that stain
        may still be spreading (ADR-0020); the skip is logged to SPREAD_LOG.
        """
        try:
            stain = pee_stain(before_path.read_bytes(), after_path.read_bytes())
        except OSError:
            return None
        if stain is None:
            return None
        last = self._last_pee_before(visit)
        if last is not None:
            last_at = datetime.fromisoformat(last["ended_at"] or last["started_at"])
            since = self._before_at(visit) - last_at
            if since < SPREAD_MIN:
                log.info("visit %s: new stain on %.0f%% of the pad, but the pee of %s (%.0f min "
                         "before) may still be spreading", visit_id, stain * 100, last["id"],
                         since.total_seconds() / 60)  # fmt: skip
                self._log_spread_skip(visit_id, last["id"], since, stain)
                self._keep_spread_photo(last["id"], after_path)
                return None
        log.info("visit %s: Gemini saw no pee but %.1f%% of the pad has a new stain, asking the "
                 "owner", visit_id, stain * 100)  # fmt: skip
        return stain

    def _keep_spread_photo(self, pee_id: str, after_path) -> None:
        """Show the spread stain as the pee's pad photo (owner, 2026-11-19).

        Copied to SPREAD_PHOTO next to the pee's own frames; after.jpg stays untouched (Gemini,
        labels and the dataset use it).
        """
        try:
            folder = self.evidence.visit_dir(pee_id)
            folder.mkdir(parents=True, exist_ok=True)  # a manual pee has no folder yet
            target = folder / SPREAD_PHOTO
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(after_path.read_bytes())
            tmp.replace(target)
        except (OSError, ValueError):  # ValueError: malformed id
            log.warning("could not keep the spread photo of %s", pee_id)

    def _log_spread_skip(self, visit_id: str, pee_id: str, since: timedelta, stain: float) -> None:
        """One JSON line per question not asked because the last pee may still be spreading."""
        row = {
            "at": iso(self._clock()),
            "visit_id": visit_id,
            "last_pee_id": pee_id,
            "minutes_since_pee": round(since.total_seconds() / 60, 1),
            "stain_pct": round(stain * 100, 1),
        }
        try:
            with (self.evidence.root / SPREAD_LOG).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            log.warning("could not write %s", SPREAD_LOG)

    async def _census(self, visit: dict, result) -> None:
        """More separate stains on the pad than pees counted on it: count the missing ones."""
        stains = getattr(result, "pad_stains", None)
        if stains is None or result.image_quality != "ok" or result.pad_replaced:
            return
        change = self.store.current_pad_change()
        if change is None or change["at"] >= visit["started_at"]:
            return
        async with self._census_lock:  # two visits decided at once must not both add the same pees
            await self._census_locked(visit, stains, change)

    async def _census_locked(self, visit: dict, stains: int, change: dict) -> None:
        end = iso(datetime.fromisoformat(visit["started_at"]) + timedelta(seconds=1))
        on_pad = [v for v in self.store.pees_list_between(change["at"], end)
                  if v["pee_where"] != "off_pad"]  # fmt: skip
        # a visit still waiting for the owner may be one of the stains: never count it twice
        waiting = [v for v in self.store.visits_started_between(change["at"], end)
                   if v["state"] == "DECIDED" and v["final_outcome"] is None]  # fmt: skip
        missing = missing_pees(stains, len(on_pad) + len(waiting))
        if not missing:
            return
        earlier = [v for v in on_pad if v["id"] != visit["id"]]
        start = datetime.fromisoformat(earlier[-1]["started_at"] if earlier else change["at"])
        hhmm = (local_text(visit["started_at"]) or "")[11:16]
        log.info("visit %s: the pad shows %s separate stains but %s pees were counted on it, "
                 "adding %s found pee(s)", visit["id"], stains, len(on_pad), missing)  # fmt: skip
        for at in found_pee_times(start, datetime.fromisoformat(visit["started_at"]), missing):
            found_id = self._manual_visit_id(at)
            self.store.insert_visit(
                {
                    "id": found_id,
                    "source": "auto",
                    "started_at": iso(at),
                    "ended_at": iso(at),
                    "entries": 1,
                    "person_present": False,
                    "frigate_event_ids": [],
                    "state": "DECIDED",
                    "auto_outcome": "pee",
                    "final_outcome": "pee",
                    "pee_where": "on_pad",
                    "poop_observed": "none",
                    "reason": f"{FOUND_PEE_REASON} na visita das {hhmm} "
                              f"({stains} manchas, {len(on_pad)} contados); hora estimada",
                    "needs_review": True,
                    "found_by": visit["id"],
                    "found_from": iso(start),
                }
            )  # fmt: skip
            await self._visit_decided(found_id, None)
        await self._alert(
            f"census_{visit['id']}",
            "Ellie: xixi achado no tapete",
            f"Na visita das {hhmm} o tapete tinha {stains} manchas separadas, mas só "
            f"{len(on_pad)} xixi(s) contado(s) desde a troca. Contei {missing} a mais, com hora "
            "estimada. Se souber a hora certa, adicione o xixi manual: ele substitui o estimado.",
        )

    def _poop_explained(self, since: str) -> bool:
        """A removed poop is explained by one we know: still open, or cleared since `since`."""
        if self.store.open_poops():
            return True
        start = iso(datetime.fromisoformat(since) - timedelta(days=1))
        return any(
            v["final_poop"] == "poop" and (v["poop_cleared_at"] or "") >= since
            for v in self.store.visits_started_between(start, iso(self._clock()))
        )

    async def _frame(self) -> tuple[bytes, bool]:
        """A cropped camera frame and whether it came from the fallback (Frigate's latest frame).

        go2rtc is tried first (full resolution). When it fails and a fallback is configured, the
        fallback frame is used, and after CAMERA_ALERT_AFTER of go2rtc failures the owner is told.
        Raises FrameError when no frame could be had.
        """
        now = self._clock()
        try:
            frame = await self._fetch_frame()
        except FrameError as exc:
            if self._camera_failing_since is None:
                self._camera_failing_since = now
            elif now - self._camera_failing_since >= CAMERA_ALERT_AFTER:
                await self._alert(
                    "camera",
                    "Ellie: câmera falhando",
                    "O go2rtc não entrega a imagem da câmera da Ellie desde "
                    + (local_text(iso(self._camera_failing_since)) or "")[11:16] + ". "
                    + ("Estou usando a imagem do Frigate (resolução menor)."
                       if self._fetch_fallback else "As visitas ficam sem foto.")
                    + " Se continuar, reinicie a câmera ou o Frigate.",
                )  # fmt: skip
            if self._fetch_fallback is None:
                raise
            try:
                frame = await self._fetch_fallback()
            except FrameError as fallback_exc:
                raise FrameError(f"{exc}; fallback: {fallback_exc}") from fallback_exc
            log.warning("go2rtc failed (%s), using Frigate's latest frame", exc)
            return self._crop(frame), True
        if self._camera_failing_since is not None:
            log.info("go2rtc frames are back")
            self._camera_failing_since = None
        return self._crop(frame), False

    async def _capture_frame(self, visit_id: str, name: str) -> None:
        """Keep a frame of a visit in progress: while she is there, or right after she left
        (before anyone can clean up)."""
        try:
            frame, _ = await self._frame()
            self.evidence.write(self.evidence.visit_dir(visit_id) / name, frame)
        except (FrameError, OSError) as exc:
            log.warning("visit %s: no %s frame: %s", visit_id, name, exc)

    def _left_frame(self, visit_id: str) -> bytes | None:
        """The frame from her last exit, when a person came into the area after it."""
        visit = self.store.get_visit(visit_id)
        if not visit or not visit["person_present"]:
            return None
        path = self.evidence.visit_dir(visit_id) / f"left_{visit['entries']}.jpg"
        return path.read_bytes() if path.exists() else None

    async def _person_check(self, visit_id: str, left: bytes, late: bytes):
        """Pad answer for her exit frame against the late frame (what the person did), or None."""
        analyser = self.analysers.get(self.cfg["deciding_analyser"])
        if analyser is None or not hasattr(analyser, "analyse_pad"):
            return None
        try:
            return await analyser.analyse_pad(left, late)
        except AnalyserError as exc:
            log.warning("visit %s: could not check what the person did: %s", visit_id, exc)
            return None

    def _person_changed_nothing(self, result) -> bool:
        if result is None:
            return False
        outcome = decide_pad_change(result, self.cfg["confidence_threshold"])
        return outcome == "not_replaced" and not result.poop_removed

    async def _after_person(self, visit: dict, directory) -> None:
        """Act on what the person did between her exit frame and the late frame: pad swap, cleanup."""
        if visit["id"] in self._person_checks:
            result = self._person_checks.pop(visit["id"])
        else:  # e.g. analysed after a restart: ask now
            try:
                left = directory / f"left_{visit['entries']}.jpg"
                result = await self._person_check(
                    visit["id"], left.read_bytes(), (directory / AFTER_PERSON).read_bytes()
                )
            except OSError as exc:
                log.warning("visit %s: could not check what the person did: %s", visit["id"], exc)
                return
        if result is None:
            return
        outcome = decide_pad_change(result, self.cfg["confidence_threshold"])
        log.info("visit %s: the person after her: %s, poop removed %s (%s)",
                 visit["id"], outcome, result.poop_removed, result.reason)  # fmt: skip
        if outcome == "replaced":
            await self._record_pad_change(
                "visit", visit["ended_at"] or visit["started_at"], visit_id=visit["id"]
            )
        if (outcome == "replaced" or result.poop_removed) and visit["final_poop"] == "poop":
            await self._clear_poops("owner", visit_id=visit["id"])
        elif result.poop_removed and not self.store.labels_for(visit["id"], "poop"):
            since = iso(datetime.fromisoformat(visit["started_at"]) - FOUND_POOP_WINDOW)
            if not self._poop_explained(since):  # a poop we never counted, or a misread
                # Ask, never count: on 2026-11-16 01:20 a fresh IR pee stain that spread was read
                # as "poop removed" (the only firing since 06/11, and it was wrong). 3.10.1.
                log.info("visit %s: the person may have removed a poop Gemini had not seen, "
                         "asking the owner", visit["id"])  # fmt: skip
                self.store.update_visit(
                    visit["id"], auto_poop="uncertain", final_poop=None, needs_review=True,
                )  # fmt: skip

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
            self._person_checks.pop(visit_id, None)

    async def _analyse_inner(self, visit_id: str, retry: bool) -> None:
        directory = self.evidence.visit_dir(visit_id)
        before_path, after_path = directory / "before.jpg", directory / "after.jpg"
        if not retry:
            try:
                after, fallback = await self._frame()
            except FrameError as exc:
                log.warning("visit %s: %s", visit_id, exc)
                return await self._fail(visit_id, "frame_error")
            left = self._left_frame(visit_id)
            chosen = after
            if left is not None:  # a person came before this frame: it may show a cleaned pad
                self.evidence.write(directory / AFTER_PERSON, after)
                self.evidence.write(after_path, left)  # a restart during the check keeps a pair
                person = await self._person_check(visit_id, left, after)
                self._person_checks[visit_id] = person
                if self._person_changed_nothing(person):
                    # the exit frame can still show her when Frigate followed her as a "person"
                    # (2026-11-14 09:37): the later frame shows the same pad, without anyone
                    log.info("visit %s: the person after her changed nothing, judging the "
                             "later frame", visit_id)  # fmt: skip
                else:
                    chosen = left
                    log.info("visit %s: a person came before the after frame, using the frame "
                             "taken when she left", visit_id)  # fmt: skip
            self.evidence.write(after_path, chosen)
            self.reference.update(after)  # the next visit starts from this clear frame
            if self.watch is not None and not fallback:
                self.watch.rebase(after)  # what changed at the pad is explained by this visit
        if not before_path.exists():
            return await self._fail(visit_id, "no_reference")
        analyser = self.analysers.get(self.cfg["deciding_analyser"])
        if analyser is None:
            return await self._fail(visit_id, "analyser_not_configured")

        events = self.store.get_visit(visit_id)["frigate_event_ids"]
        during = await self._fetch_snapshot(events[0]) if events else None
        if during is None and (directory / DURING_OWN).exists():  # Frigate dropped the event
            during = (directory / DURING_OWN).read_bytes()
        if during:
            try:
                self.evidence.write(directory / "during.jpg", during)
            except OSError:
                log.warning(
                    "could not store the during photo of %s, analysing without it", visit_id
                )

        started = time.monotonic()
        visit_row = self.store.get_visit(visit_id)
        try:
            result = await analyser.analyse(
                before_path.read_bytes(), after_path.read_bytes(), during,
                context=self._pee_context(visit_row),
            )  # fmt: skip
        except AnalyserError as exc:
            self.store.insert_analysis(visit_id, analyser.id, "deciding", None, "unknown",
                                       int((time.monotonic() - started) * 1000), str(exc), None)  # fmt: skip
            return await self._fail(visit_id, self._reason_for(str(exc)))
        outcome = decide_visit(result, self.cfg["confidence_threshold"])
        usage = getattr(analyser, "last_usage", None)
        self.store.insert_analysis(visit_id, analyser.id, "deciding",
                                   result.model_dump(exclude_none=True), outcome,
                                   int((time.monotonic() - started) * 1000), None, usage)  # fmt: skip
        stain = None
        if outcome == "no_pee" and result.poop == "none" and visit_row["frigate_event_ids"]:
            stain = self._second_pee_opinion(visit_id, visit_row, before_path, after_path)
            if stain is not None:
                outcome = "uncertain"  # the owner decides
        decided = outcome in ("pee", "no_pee")
        fields = {
            "auto_outcome": outcome,
            "pee_where": result.pee_where,
            "poop_observed": result.poop,
            "reason": result.reason
            if stain is None
            else (f"{result.reason} (mas {stain:.0%} do tapete tem mancha nova)"),
            "deciding_analyser": analyser.id,
        }
        if not self.store.labels_for(visit_id, "poop"):  # the owner's poop answer wins too
            fields |= self._poop_fields(result) | {"poop_cleared_at": None, "poop_cleared_by": None}
            if fields["auto_poop"] == "no_poop":
                fields |= self._second_poop_opinion(visit_id, before_path, after_path)
        if not self.store.labels_for(visit_id):  # an owner's answer always wins
            fields |= {
                "state": "DECIDED",
                "final_outcome": outcome if decided else None,
                "needs_review": (not decided) or bool(self.cfg["validation_mode"]),
            }
        self.store.update_visit(visit_id, **fields)
        visit = self.store.get_visit(visit_id)
        if outcome in ("pee", "no_pee"):
            await self._census(visit, result)
        await self._pad_full_if_outside(visit)
        if visit["final_poop"] == "poop" and visit["poop_cleared_at"] is None:
            self._follow_poop(visit)
        if (directory / AFTER_PERSON).exists():  # the person's work is in that later frame
            if not retry:  # a reanalysis must not record the same pad change twice
                await self._after_person(visit, directory)
        elif (
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
            if not visit_id or self._image_keys.get(topic) == visit_id:
                continue
            path = self._photo_dir(visit_id) / name
            # never leave the photos of an older visit on screen (2026-11-20 07:41 found pee
            # showed the pad before the change): placeholder when there is no photo at all
            body = path.read_bytes() if path.exists() else _placeholder_jpeg()
            sent = await self._publish(topic, body, True)
            if sent is not False:  # a dropped publish (no MQTT connection yet) must be retried
                self._image_keys[topic] = visit_id
        await self._publish_review_closeups(current and current["visit_id"])
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

    def _photo_dir(self, visit_id: str):
        """Folder with the BEFORE/AFTER to show for a visit: its own, else the visit that found it.

        A pee found by the stain census has no frames; the frames of the visit that found it
        show the stain. Falls back to the visit's own (possibly empty) folder.
        """
        own = self.evidence.visit_dir(visit_id)
        if (own / "after.jpg").exists() or (own / "before.jpg").exists():
            return own
        visit = self.store.get_visit(visit_id)
        finder = visit.get("found_by") if visit else None
        if finder:
            try:
                return self.evidence.visit_dir(finder)
            except ValueError:
                pass
        return own

    async def _publish_review_closeups(self, visit_id: str | None) -> None:
        """Pad close-ups of the review photos (ADR-0023): written once, sent when the head changes.

        A view that does not exist (infrared frame, missing photo) gets the flat placeholder, so
        the popup never shows the close-up of an older visit.
        """
        topics = discovery.IMAGE_TOPICS
        keys = [k for k in topics if k.startswith("review_") and k.count("_") == 2]
        if not visit_id or all(self._image_keys.get(topics[k]) == visit_id for k in keys):
            return
        paths = await asyncio.to_thread(review_images.write_variants, self._photo_dir(visit_id))
        for key in keys:
            topic = topics[key]
            if self._image_keys.get(topic) == visit_id:
                continue
            path = paths.get(key.removeprefix("review_"))
            body = path.read_bytes() if path is not None else _placeholder_jpeg()
            if await self._publish(topic, body, True) is not False:
                self._image_keys[topic] = visit_id

    def _top_photo(self, last: dict | None) -> tuple[str, bytes] | None:
        """(key, jpeg) for the big "Foto do tapete": the latest of the last pee and a pad change.

        A pee that kept spreading shows its latest spread frame (SPREAD_PHOTO).

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
            folder = self.evidence.visit_dir(last["visit_id"])
            spread = folder / SPREAD_PHOTO  # the same pee, photographed after it spread
            if spread.exists():
                return f"{last['visit_id']}:{spread.stat().st_mtime_ns}", spread.read_bytes()
            path = self._photo_dir(last["visit_id"]) / "after.jpg"
            if path.exists():
                return last["visit_id"], path.read_bytes()
            # a manual pee has no photo: show the pad as the camera saw it after the pee
            # (2026-11-20 07:41 manual pee kept the placeholder)
            reference = self.reference.latest()
            if reference is not None and reference[1] >= datetime.fromisoformat(last["at"]):
                return f"{last['visit_id']}:ref", reference[0]
        return "placeholder", _placeholder_jpeg()  # no photo yet

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

    def _area_counts(self, now: datetime, kinds: tuple[str, ...]) -> tuple[list[Action], list[Any]]:
        """Pass tray_area counts on, with the persons that are her moved to the dog count."""
        her = min(self.small_person.count, self._raw_area["person"])
        counts = {"dog": self._raw_area["dog"] + her, "person": self._raw_area["person"] - her}
        actions: list[Action] = []
        presence: list[Any] = []
        # persons first: a visit that starts on the dog count must not see her as a person
        if "person" in kinds:
            actions += self.tracker.on_tray_person(counts["person"], now)
            presence += self.presence.on_person(counts["person"], now)
        if "dog" in kinds:
            actions += self.tracker.on_tray_dog(counts["dog"], now)
            presence += self.presence.on_dog_area(counts["dog"], now)
        return actions, presence

    def _person_event(self, payload: bytes) -> tuple[str, bool] | None:
        """(event id, still going) of a real person who entered tray_area; None otherwise."""
        try:
            message = json.loads(payload) or {}
            after = message.get("after") or {}
            if after.get("camera") != self.camera or after.get("label") != "person":
                return None
            event_id = after.get("id")
            if not isinstance(event_id, str):
                return None
            if message.get("type") == "end" or after.get("end_time"):
                return event_id, False
            if "tray_area" not in set(after.get("entered_zones") or []):
                return None
        except (ValueError, AttributeError, TypeError):
            return None
        return (event_id, True) if self.small_person.is_big(event_id) else None

    def _dog_event_id(self, payload: bytes) -> str | None:
        try:
            after = (json.loads(payload) or {}).get("after") or {}
            if after.get("camera") != self.camera:
                return None
            if after.get("label") != "dog" and not (
                after.get("label") == "person" and self.small_person.is_ellie(str(after.get("id")))
            ):
                return None
            if not {"pad_zone", "tray_area"} & set(after.get("entered_zones") or []):
                return None
            event_id = after.get("id")
        except (ValueError, AttributeError, TypeError):  # not JSON, not UTF-8, wrong shape
            return None
        # the id ends up in a Frigate API URL: accept only the shape Frigate really uses
        return event_id if isinstance(event_id, str) and EVENT_ID.match(event_id) else None
