"""Pad change flow of the service (US7), mixed into `Service`.

A person at the tray is analysed before/after with the pad-change prompt. Frigate sometimes labels
Ellie herself as a person (squatting, seen from above: 2026-10-30 06:53, a poop), so a presence
whose pad was not replaced (by Gemini, or by the owner when Gemini was unsure) is also asked the
visit question; new pee or poop turns it into a visit. A confident
"replaced" restarts the countdown with no tap; unsure answers ask the owner; the owner can also
mark a change ("Troquei agora") or undo one. Reminders live in Home Assistant (a time trigger on
`sensor.ellie_pad_change_due`), so this module only keeps the facts.
"""

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ellie.analysers.base import AnalyserError
from ellie.counting import LOCAL_TZ
from ellie.decide import decide_pad_change, decide_visit
from ellie.frames import FrameError
from ellie.pads import PresenceAction
from ellie.state import local_text
from ellie.visits import iso

log = logging.getLogger(__name__)
PRESENCE_RETAKES = 3  # occluded after frames taken again before the owner is asked
RETAKE_CLEAR_S = 60  # the tray must be clear this long before a retake
FOUND_POOP_WINDOW = timedelta(minutes=15)  # a poop cleared this long before a cleanup explains it
PADS_AVERAGED = 5  # finished pads in the "pees per pad" average
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
        photo, image_at = self.pad_photo()
        history = self._pad_history()
        return {
            "changed_at": change["at"],
            "due_at": iso(due),
            "source": change["source"],
            "image_at": image_at,
            "days_in_use": round((now - at).total_seconds() / 86400, 1),
            "pees_since": self.store.pees_since(change["at"]),
            "pad_color": change["pad_color"],
            "image": self.evidence.local_url(photo) if photo else None,
            "history": history,
            **self._pad_averages(history),
        }

    def _pad_history(self) -> list[dict[str, Any]]:
        """Newest first. `pees` is how many pees the pad took until the next (not undone) change.

        Derived from the visits, not stored: an owner correction later also corrects the average.
        The current pad has no end yet (`pees_since` says how many it has taken so far). `days`
        is how long the pad was in use; `overflow_pee` is the number of the first pee that went
        outside the pad (pee_v4, ADR-0017), None when none did.
        """
        history, newer_at = [], None
        now = iso(self._clock())
        for c in self.store.pad_changes(10):
            counted = c["undone_at"] is None
            entry = {
                "id": c["id"],
                "at": c["at"],
                "source": c["source"],
                "undone_at": c["undone_at"],
                "pees": None,
                "days": None,
                "pees_per_day": None,
                "overflow_pee": None,
            }
            if counted:
                pees = self.store.pees_list_between(c["at"], newer_at)
                end = newer_at or now
                days = (
                    datetime.fromisoformat(end) - datetime.fromisoformat(c["at"])
                ).total_seconds()
                days /= 86400
                entry["days"] = round(days, 1)
                entry["overflow_pee"] = next(
                    (n for n, v in enumerate(pees, 1) if v["pee_where"] in ("off_pad", "both")),
                    None,
                )
                if newer_at:  # finished: the pad's totals are final
                    entry["pees"] = len(pees)
                    entry["pees_per_day"] = round(len(pees) / days, 1) if days >= 0.1 else None
                newer_at = c["at"]
            history.append(entry)
        return history

    def _pad_averages(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Averages of the last `PADS_AVERAGED` finished pads (once at least two exist) and the
        pads used in the last 30 days (undone changes do not count)."""
        finished = [h for h in history if h["pees"] is not None][:PADS_AVERAGED]
        enough = len(finished) >= 2

        def mean(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 1) if enough and values else None

        days = mean([h["days"] for h in finished])
        overflows = [h["overflow_pee"] for h in history if h["overflow_pee"] is not None]
        since = iso(self._clock() - timedelta(days=30))
        return {
            "avg_pees_per_pad": mean([h["pees"] for h in finished]),
            "pads_counted": len(finished),
            "avg_days_per_pad": days,
            "pads_per_month": round(30 / days, 1) if days else None,
            "pads_last_30d": sum(
                1 for c in self.store.pad_changes_since(since) if c["undone_at"] is None
            ),
            "overflow_pees": overflows,  # first pee outside the pad, per pad that overflowed
        }

    async def _pad_full_if_outside(self, visit: dict) -> None:
        """A counted pee outside the pad: the pad is full (ADR-0017). Once per visit."""
        if (
            visit["final_outcome"] != "pee"
            or visit["pee_where"] not in ("off_pad", "both")
            or visit["id"] in self._pad_full_sent
        ):
            return
        self._pad_full_sent.add(visit["id"])
        change = self.store.current_pad_change()
        after = self.evidence.visit_dir(visit["id"]) / "after.jpg"
        payload = {
            "visit_id": visit["id"],
            "at": visit["started_at"],
            "pees_since": self.store.pees_since(change["at"]) if change else None,
            "image": self.evidence.local_url(after) if after.exists() else None,
            "reason": visit["reason"],
        }
        log.info("visit %s: pee outside the pad, the pad is full", visit["id"])
        await self._publish("ellie/pad/full", json.dumps(payload, ensure_ascii=False), False)

    def pad_photo(self) -> tuple[Path | None, str | None]:
        """Photo of the current pad change and its time.

        A manual mark has no photo of its own: use the latest photographed change (the automatic
        detection that usually came first) and say when it was taken.
        """
        current = self.store.current_pad_change()
        if current is None:
            return None, None
        for change in [current, *self.store.pad_changes(10)]:
            if change["undone_at"] is None and (photo := self._change_photo(change)):
                return photo, change["at"]
        return None, None

    def _change_photo(self, change: dict) -> Path | None:
        """The `after.jpg` that came with a pad change, if there is one."""
        found = None
        for directory in (
            self.evidence.presence_dir(change["presence_id"]) if change["presence_id"] else None,
            self.evidence.visit_dir(change["visit_id"]) if change["visit_id"] else None,
        ):
            if directory is not None and (directory / "after.jpg").exists():
                found = directory / "after.jpg"
        return found

    def _change_image(self, change: dict) -> str | None:
        """`/media/local` URL of a pad change's photo, for notifications."""
        photo = self._change_photo(change)
        return self.evidence.local_url(photo) if photo else None

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
            if self.watch is not None:
                self.watch.rebase(after)  # the presence explains what changed at the pad
        if not before_path.exists():
            return await self._presence_unknown(presence_id, "no_reference")
        analyser = self.analysers.get(self.cfg["deciding_analyser"])
        if analyser is None or not hasattr(analyser, "analyse_pad"):
            return await self._presence_unknown(presence_id, "analyser_not_configured")
        try:
            result = await analyser.analyse_pad(before_path.read_bytes(), after_path.read_bytes())
        except AnalyserError as exc:
            return await self._presence_unknown(presence_id, self._reason_for(str(exc)))
        if result.image_quality == "occluded" and self._retake_presence(presence_id):
            return  # stays ANALYSING: a new after frame is taken once the tray is clear
        self._presence_retakes.pop(presence_id, None)
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
            promotion_pending=outcome == "not_replaced",  # cleared when the check has run
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
        if result.poop_removed:
            await self._found_poop(presence)
        if outcome == "not_replaced":  # unsure: wait for the owner (a new pad fakes a "new mark")
            await self._promote_if_eliminated(presence, analyser)
        await self.publish_state()

    def _retake_presence(self, presence_id: str) -> bool:
        """An occluded frame (someone still at the tray): take the after frame again later
        instead of asking the owner (2026-11-09 13:08 and 13:10). False when the retakes are used."""
        done = self._presence_retakes.get(presence_id, 0)
        if done >= PRESENCE_RETAKES:
            self._presence_retakes.pop(presence_id, None)
            return False
        self._presence_retakes[presence_id] = done + 1
        after = self.evidence.presence_dir(presence_id) / "after.jpg"
        if after.exists():
            self.reference.discard(after.read_bytes())  # never the "before" of what comes next
        log.info("presence %s: occluded, taking the after frame again when the tray is clear "
                 "(%s of %s)", presence_id, done + 1, PRESENCE_RETAKES)  # fmt: skip
        return True

    def _due_retakes(self, now: datetime) -> list[str]:
        """Presences waiting for a new after frame, once the tray has been clear a while."""
        if not self._presence_retakes or self.tracker.active is not None:
            return []
        if self.presence.present or self.tracker.clear_for(now) < RETAKE_CLEAR_S:
            return []
        return [p for p in self._presence_retakes if p not in self._inflight]

    async def _found_poop(self, presence: dict) -> None:
        """The person removed a poop the system never counted: count it now (cleared already).

        Since 3.5.0 every "poop removed" answer matched a poop that was still open (2026-11-07
        09:22, 11-08 12:23), so one with no open poop is a poop Frigate or Gemini missed. The time
        is when the cleanup started (the poop came earlier); the visit lands in the review list.
        """
        since = iso(datetime.fromisoformat(presence["started_at"]) - FOUND_POOP_WINDOW)
        if self._poop_explained(since):
            return
        directory = self.evidence.presence_dir(presence["id"])
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
                "state": "DECIDED",
                "auto_outcome": None,
                "final_outcome": "no_pee",
                "auto_poop": "poop",
                "final_poop": "poop",
                "poop_observed": "none",
                "poop_cleared_at": presence["ended_at"],
                "poop_cleared_by": "owner",
                "reason": "cocô achado na limpeza (aconteceu antes desta hora)",
                "needs_review": True,
            }
        )
        target = self.evidence.visit_dir(visit_id)
        for name in ("before.jpg", "after.jpg"):
            if (directory / name).exists():
                self.evidence.write(target / name, (directory / name).read_bytes())
        log.info("presence %s removed a poop nobody counted: visit %s", presence["id"], visit_id)
        await self._visit_decided(visit_id, None)

    async def _promote_if_eliminated(self, presence: dict, analyser) -> None:
        """If the "person" left new pee or poop, it was Ellie: record it as a normal visit.

        Runs once the pad is known not to be replaced (by Gemini or by the owner). A failure here
        must not undo the pad decision nor leave a visit ANALYSING.
        """
        directory = self.evidence.presence_dir(presence["id"])
        before_path, after_path = directory / "before.jpg", directory / "after.jpg"
        if before_path.exists() and after_path.exists():
            created: list[str] = []
            try:
                await self._promote(presence, before_path, after_path, analyser, created)
            except Exception:
                log.exception("promotion of presence %s failed", presence["id"])
                if created:
                    await self._fail(created[0], "analysis_failed")
        # Only a restart in the middle leaves the marker set; `recover()` then runs the check again.
        self.store.update_presence(presence["id"], promotion_pending=False)

    async def _promote(self, presence: dict, before_path, after_path, analyser, created) -> None:
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
        twin = self.store.get_visit(visit_id)
        if twin and twin["started_at"] == presence["started_at"] and twin["person_present"]:
            return  # a run cut short by a restart had already made this visit
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
        created.append(visit_id)
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
            pee_where=result.pee_where,
            reason=result.reason,
            needs_review=(not decided) or bool(self.cfg["validation_mode"]),
            deciding_analyser=analyser.id,
            **self._poop_fields(result),
        )
        promoted = self.store.get_visit(visit_id)
        await self._pad_full_if_outside(promoted)
        if promoted["final_poop"] == "poop":
            self._follow_poop(promoted)
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
        message = {
            "pad_change_id": change_id,
            "at": at,
            "source": source,
            "image": self._change_image({"presence_id": presence_id, "visit_id": visit_id}),
        }
        await self._publish("ellie/pad/changed", json.dumps(message), False)
        await self._clear_poops("pad_change", where="on_pad")  # the poop went with the pad
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
        else:  # the pad stayed: a new mark on it was made by Ellie (seen as a person)
            self.store.update_presence(presence["id"], promotion_pending=True)
            analyser = self.analysers.get(self.cfg["deciding_analyser"])
            if analyser is not None and hasattr(analyser, "analyse"):
                self._spawn(self._promote_if_eliminated(presence, analyser))
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
