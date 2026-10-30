"""Pad change flow of the service (US7), mixed into `Service`.

A person at the tray is analysed before/after with the pad-change prompt. Frigate sometimes labels
Ellie herself as a person (squatting, seen from above: 2026-10-30 06:53, a poop), so a presence
whose pad was not replaced is also asked the visit question; new pee or poop turns it into a visit. A confident
"replaced" restarts the countdown with no tap; unsure answers ask the owner; the owner can also
mark a change ("Troquei agora") or undo one. Reminders live in Home Assistant (a time trigger on
`sensor.ellie_pad_change_due`), so this module only keeps the facts.
"""

import json
import logging
import shutil
from datetime import datetime, timedelta
from typing import Any

from ellie.analysers.base import AnalyserError
from ellie.counting import LOCAL_TZ
from ellie.decide import decide_pad_change, decide_visit
from ellie.frames import FrameError
from ellie.pads import PresenceAction
from ellie.state import local_text
from ellie.visits import iso

log = logging.getLogger(__name__)
PAD_REASONS = {
    "no_reference": "faltou a foto de antes do tabuleiro",
    "frame_error": "não consegui pegar a imagem da câmera",
    "analyser_not_configured": "a análise do Gemini não está configurada",
    "analysis_failed": "a análise falhou",
}


class PadsMixin:
    """Uses the attributes and helpers defined by `Service`."""

    # --- state published to HA ---

    def pad_state(self, now: datetime) -> dict[str, Any] | None:
        change = self.store.current_pad_change()
        if change is None:
            return None  # no change known yet: no reminder, the dashboard says so
        at = datetime.fromisoformat(change["at"])
        due = at + timedelta(days=self.cfg["pad_change_days"])
        image = None
        for directory in (
            self.evidence.presence_dir(change["presence_id"]) if change["presence_id"] else None,
            self.evidence.visit_dir(change["visit_id"]) if change["visit_id"] else None,
        ):
            if directory is not None and (directory / "after.jpg").exists():
                image = self.evidence.local_url(directory / "after.jpg")
        return {
            "changed_at": change["at"],
            "due_at": iso(due),
            "source": change["source"],
            "days_in_use": round((now - at).total_seconds() / 86400, 1),
            "pees_since": self.store.pees_since(change["at"]),
            "pad_color": change["pad_color"],
            "image": image,
            "history": [
                {"id": c["id"], "at": c["at"], "source": c["source"], "undone_at": c["undone_at"]}
                for c in self.store.pad_changes(10)
            ],
        }

    # --- presences ---

    def _sync_presence(self, now: datetime) -> list[PresenceAction]:
        """A dog visit owns the area; tell the presence tracker about it."""
        return self.presence.set_visit_active(self.tracker.active is not None, now)

    async def _apply_presence(self, actions: list[PresenceAction]) -> None:
        for action in actions:
            row = action.presence
            presence_id = row["id"]
            directory = self.evidence.presence_dir(presence_id)
            if action.kind == "started":
                self.store.insert_presence(
                    {"id": presence_id, "started_at": row["started_at"], "state": "PRESENT"}
                )
                reference = self.reference.before(datetime.fromisoformat(row["started_at"]))
                if reference:
                    self.evidence.write(directory / "before.jpg", reference[0])
            elif action.kind == "dropped":
                self.store.update_presence(
                    presence_id, ended_at=row["ended_at"], state="DECIDED", reason=action.reason
                )
                shutil.rmtree(directory, ignore_errors=True)  # nothing worth keeping
            else:
                self.store.update_presence(presence_id, ended_at=row["ended_at"], state="ANALYSING")
                self._spawn(self._analyse_presence(presence_id))

    async def _analyse_presence(self, presence_id: str, retry: bool = False) -> None:
        if presence_id in self._inflight:
            return
        self._inflight.add(presence_id)
        try:
            await self._analyse_presence_inner(presence_id, retry)
        except Exception:  # a presence must never stay ANALYSING forever
            log.exception("analysis of %s crashed", presence_id)
            await self._presence_unknown(presence_id, "analysis_failed")
        finally:
            self._inflight.discard(presence_id)

    async def _analyse_presence_inner(self, presence_id: str, retry: bool) -> None:
        directory = self.evidence.presence_dir(presence_id)
        before_path, after_path = directory / "before.jpg", directory / "after.jpg"
        if not retry:
            try:
                after = self._crop(await self._fetch_frame())
            except FrameError as exc:
                log.warning("presence %s: %s", presence_id, exc)
                return await self._presence_unknown(presence_id, "frame_error")
            self.evidence.write(after_path, after)
            self.reference.update(after)
        if not before_path.exists():
            return await self._presence_unknown(presence_id, "no_reference")
        analyser = self.analysers.get(self.cfg["deciding_analyser"])
        if analyser is None or not hasattr(analyser, "analyse_pad"):
            return await self._presence_unknown(presence_id, "analyser_not_configured")
        try:
            result = await analyser.analyse_pad(before_path.read_bytes(), after_path.read_bytes())
        except AnalyserError as exc:
            return await self._presence_unknown(presence_id, self._reason_for(str(exc)))
        outcome = decide_pad_change(result, self.cfg["confidence_threshold"])
        decided = outcome in ("replaced", "not_replaced")
        self.store.update_presence(
            presence_id,
            state="DECIDED",
            auto_outcome=outcome,
            final_outcome=outcome if decided else None,
            poop_removed=result.poop_removed,
            reason=result.reason,
            analysis=result.model_dump(exclude_none=True),
            cost_units=getattr(analyser, "last_usage", None),
            needs_review=not decided,
        )
        presence = self.store.get_presence(presence_id)
        log.info("presence %s: %s (%s)", presence_id, outcome, result.reason)
        if outcome == "replaced":
            await self._record_pad_change(
                "detected",
                presence["ended_at"],
                presence_id=presence_id,
                pad_color=result.pad_color,
            )
        elif outcome != "not_replaced":
            await self._request_pad_review(presence)
        if outcome != "replaced":  # a new pad would make "new mark" meaningless
            await self._promote_if_eliminated(presence, before_path, after_path, analyser)
        await self.publish_state()

    async def _promote_if_eliminated(
        self, presence: dict, before_path, after_path, analyser
    ) -> None:
        """If the "person" left new pee or poop, it was Ellie: record it as a normal visit."""
        try:
            result = await analyser.analyse(before_path.read_bytes(), after_path.read_bytes(), None)
        except AnalyserError as exc:
            log.warning("presence %s: elimination check failed: %s", presence["id"], exc)
            return
        peed = result.pee and result.pee_confidence >= 0.5
        pooped = result.poop in ("on_pad", "beside_tray")
        if not (peed or pooped):
            # nothing new: bill this second call to the presence (a promoted visit bills itself)
            usage = getattr(analyser, "last_usage", None) or {}
            spent = self.store.get_presence(presence["id"])["cost_units"] or {}
            self.store.update_presence(
                presence["id"],
                cost_units={k: int(spent.get(k, 0)) + int(usage.get(k, 0)) for k in ("in", "out")},
            )
            return
        stamp = datetime.fromisoformat(presence["started_at"]).astimezone(LOCAL_TZ)
        visit_id = f"v_{stamp:%Y%m%d_%H%M%S}"
        while self.store.get_visit(visit_id):
            stamp += timedelta(seconds=1)
            visit_id = f"v_{stamp:%Y%m%d_%H%M%S}"
        self.store.insert_visit(
            {
                "id": visit_id,
                "source": "auto",
                "started_at": presence["started_at"],
                "ended_at": presence["ended_at"],
                "entries": 1,
                "person_present": True,
                "frigate_event_ids": [],
                "state": "ANALYSING",
                "needs_review": False,
            }
        )
        directory = self.evidence.visit_dir(visit_id)
        self.evidence.write(directory / "before.jpg", before_path.read_bytes())
        self.evidence.write(directory / "after.jpg", after_path.read_bytes())
        outcome = decide_visit(result, self.cfg["confidence_threshold"])
        self.store.insert_analysis(visit_id, analyser.id, "deciding",
                                   result.model_dump(exclude_none=True), outcome, None, None,
                                   getattr(analyser, "last_usage", None))  # fmt: skip
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
        log.info(
            "presence %s became visit %s: %s (Frigate saw a person)",
            presence["id"],
            visit_id,
            outcome,
        )
        await self._visit_decided(visit_id, "uncertain" if outcome == "uncertain" else None)

    async def _presence_unknown(self, presence_id: str, code: str) -> None:
        reason = PAD_REASONS.get(code) or self._reason_text(code)
        self.store.update_presence(
            presence_id,
            state="DECIDED",
            auto_outcome="unknown",
            final_outcome=None,
            reason=reason,
            needs_review=True,
        )
        for fix_code, title, message in self._fixable().values():
            if fix_code == code:
                await self._alert(code, title, message)
                break
        await self._request_pad_review(self.store.get_presence(presence_id))
        await self.publish_state()

    async def _request_pad_review(self, presence: dict) -> None:
        hhmm = (local_text(presence["started_at"]) or "")[11:16]
        after = self.evidence.presence_dir(presence["id"]) / "after.jpg"
        payload = {
            "visit_id": presence["id"],
            "kind": "pad_uncertain",
            "title": "Ellie: você trocou o tapete?",
            "message": f"Movimento no tabuleiro às {hhmm} — {presence['reason']}",
            "image": self.evidence.local_url(after) if after.exists() else None,
            "reason": presence["reason"],
        }
        await self._publish("ellie/review/request", json.dumps(payload, ensure_ascii=False), False)

    # --- pad changes ---

    async def _record_pad_change(
        self,
        source: str,
        at: str,
        presence_id: str | None = None,
        visit_id: str | None = None,
        pad_color: str | None = None,
    ) -> str:
        stamp = datetime.fromisoformat(at).astimezone(LOCAL_TZ)
        change_id = f"p_{stamp:%Y%m%d_%H%M%S}"
        while any(c["id"] == change_id for c in self.store.pad_changes(50)):
            stamp += timedelta(seconds=1)
            change_id = f"p_{stamp:%Y%m%d_%H%M%S}"
        self.store.insert_pad_change(
            {
                "id": change_id,
                "at": at,
                "source": source,
                "presence_id": presence_id,
                "visit_id": visit_id,
                "pad_color": pad_color,
            }
        )
        log.info("pad change %s at %s (%s)", change_id, at, source)
        message = {"pad_change_id": change_id, "at": at, "source": source}
        await self._publish("ellie/pad/changed", json.dumps(message), False)
        await self.publish_state()
        return change_id

    async def _cmd_pad_label(self, data: dict) -> None:
        """`ellie/cmd/pad_label`: the owner answers "Você trocou o tapete?"."""
        presence = self.store.get_presence(str(data.get("presence_id")))
        label = data.get("label")
        if presence is None or label not in ("replaced", "not_replaced"):
            log.warning("ignored pad_label command %r", data.get("presence_id"))
            return
        if presence["state"] != "DECIDED" or presence["final_outcome"] is not None:
            log.warning("pad_label for %s ignored: not waiting for an answer", presence["id"])
            return  # still analysing, or already answered (a double tap)
        self.store.update_presence(presence["id"], final_outcome=label, needs_review=False)
        if label == "replaced":
            await self._record_pad_change(
                "confirmed",
                presence["ended_at"] or presence["started_at"],
                presence_id=presence["id"],
            )
        await self._publish(
            "ellie/review/resolved", json.dumps({"visit_id": presence["id"]}), False
        )
        await self.publish_state()

    async def _cmd_pad_changed(self, data: dict) -> None:
        """`ellie/cmd/pad_changed`: "Troquei agora" / "Já troquei"."""
        raw = data.get("at")
        try:
            at = datetime.fromisoformat(str(raw)) if raw else self._clock()
            if at.tzinfo is None:
                at = at.replace(tzinfo=LOCAL_TZ)
            stamp = iso(at)
        except (ValueError, OverflowError):
            log.warning("pad_changed: invalid time %r", raw)
            return
        if at > self._clock() + timedelta(minutes=5):
            log.warning("pad_changed: time is in the future")
            return
        if at < self._clock() - timedelta(days=366):
            log.warning("pad_changed: time is too far in the past")
            return
        current = self.store.current_pad_change()
        if current and stamp < current["at"]:
            log.info("pad_changed: older than the current change, it will not become current")
        await self._record_pad_change("manual", stamp)

    async def _cmd_pad_undo(self, data: dict) -> None:
        """`ellie/cmd/pad_undo`: "Não troquei"; the previous change becomes current again."""
        undone = self.store.undo_current_pad_change(iso(self._clock()))
        if undone is None:
            log.warning("pad_undo: no pad change to undo")
            return
        message = {"pad_change_id": undone["id"], "at": undone["at"], "source": undone["source"]}
        await self._publish("ellie/pad/changed", json.dumps({**message, "undone": True}), False)
        await self.publish_state()
