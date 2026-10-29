"""
Domain services - Orchestration logic for pee/poop detection.

Design goals for this v2-in-place refactor:
- Event-driven: start work from `frigate/events` dog episodes.
- Post-episode confirmation on a stable Pad ROI (Frigate zone bbox).
- Reliable pee vs poop using image diffs (no LLM, no extra hardware).
- Keep existing MQTT contract used by Home Assistant:
  - home/ellie/state (string)
  - home/ellie/poop_present (retained JSON)
  - home/ellie/poop_event (JSON)
  - home/ellie/elimination_event (JSON)
  - home/ellie/elimination_stats (retained JSON)
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from settings import Settings
from adapters.state_store import JsonStateStore
from domain.models import (
    EliminationKind,
    EliminationLocation,
    FrigateEventUpdate,
    Point,
    ROI,
    ShapeParams,
)
from domain import heuristics


@dataclass
class _Episode:
    event_id: str
    zones: list[str]
    start_ts: float
    baseline_roi: Optional[np.ndarray]
    pad_roi: Optional[ROI]
    mono: bool
    has_clip: Optional[bool]


class EllieWatcherService:
    """Main service orchestrating Frigate-episode based elimination monitoring."""

    def __init__(
        self,
        settings: Settings,
        clock,  # Clock interface
        cv_ops,  # adapters.cv_ops
        frigate_client,  # adapters.frigate_client.FrigateClient
        mqtt_client,  # adapters.mqtt_client.MQTTClient
        state_store: Optional[JsonStateStore] = None,
    ):
        self.settings = settings
        self.clock = clock
        self.cv = cv_ops
        self.frigate = frigate_client
        self.mqtt = mqtt_client
        self.state_store = state_store

        self._lock = threading.Lock()

        # Shape params for poop blob filtering (anti-pee).
        self.shape_params = ShapeParams(
            min_blob_w=settings.min_blob_w,
            min_blob_h=settings.min_blob_h,
            min_circularity=settings.min_circularity,
            min_solidity=settings.min_solidity,
            hsv_s_min=settings.hsv_s_min,
            tex_min_mono=settings.tex_min_mono,
            spec_max_mono=settings.spec_max_mono,
        )

        # Episode state.
        self._active: Optional[_Episode] = None
        self._confirming_event_id: Optional[str] = None
        self._last_seen_dog_ts: float = 0.0

        # Pad ROI and pad clean reference (best-effort).
        self._pad_roi: Optional[ROI] = None
        self._pad_ref_bgr: Optional[np.ndarray] = None
        self._pending_pad_ref_ts: float = 0.0
        self._last_pad_refresh_log_ts: float = 0.0
        self._load_pad_reference()
        self._load_last_confirmed_pee()

        # Continuous baseline for episode detection (captured when dog is absent).
        self._continuous_baseline_roi: Optional[np.ndarray] = None
        self._continuous_baseline_ts: float = 0.0
        self._continuous_baseline_mono: bool = False
        self._continuous_baseline_pad_roi: Optional[ROI] = None

        # Poop present tracking (for cleanup reminders).
        self._poop_present: bool = False
        self._poop_pad_roi: Optional[ROI] = None
        self._poop_baseline_roi: Optional[np.ndarray] = None
        self._poop_mono: bool = False

        # De-dup (avoid double counting due to repeated end/update).
        self._last_confirm_ts: float = 0.0
        self._last_confirm_centroid: Optional[Point] = None
        self._last_confirm_kind: Optional[str] = None
        self._last_confirmed_pee_roi: Optional[np.ndarray] = None
        self._last_confirmed_pee_ts: float = 0.0
        self._last_confirmed_pee_centroid: Optional[Point] = None

        # History for typical time-of-day stats.
        self._history_window_days = 30
        self._history: list[dict] = []
        self._load_history()

    # ---------------------------------------------------------------------
    # MQTT handlers entry points
    # ---------------------------------------------------------------------

    def handle_person_event(self, upd: FrigateEventUpdate) -> None:
        """
        Person presence is used as a hint that the pad may have been replaced.
        We refresh a clean pad reference once the dog is absent.
        """
        if self.settings.toilet_zone and (self.settings.toilet_zone not in (upd.zones or [])):
            return
        with self._lock:
            self._pending_pad_ref_ts = float(self.clock.now())

    def handle_dog_event(self, upd: FrigateEventUpdate) -> None:
        if upd.label != "dog":
            return

        now_t = float(self.clock.now())
        with self._lock:
            self._last_seen_dog_ts = now_t

        # If we want to monitor only the toilet zone, ignore dog events elsewhere.
        if (not self.settings.monitor_all_zones) and self.settings.toilet_zone:
            if self.settings.toilet_zone not in (upd.zones or []):
                return

        # Track zones as they evolve during the event.
        if upd.type == "update":
            with self._lock:
                if self._active and self._active.event_id == upd.event_id:
                    self._active.zones = list(upd.zones or [])
            return

        if upd.type == "new":
            self._start_episode(upd)
            return

        if upd.type == "end":
            self._end_episode(upd)
            return

    # ---------------------------------------------------------------------
    # Main loop tick
    # ---------------------------------------------------------------------

    def check_episode_timeout(self) -> None:
        """
        Called from the runner loop every ~1s.

        Keeps pad reference updated, and opportunistically updates pad wetness metrics
        when the dog is absent (cheap and robust).
        """
        self._maybe_refresh_pad_reference()
        # Opportunistic: when dog is absent, refresh wetness metrics (optional context).
        now_t = float(self.clock.now())
        if (now_t - float(self._last_seen_dog_ts)) >= float(self.settings.pad_ref_dog_absent_s):
            self._update_pad_wetness_state()
            # Also capture a continuous baseline for better episode detection.
            self._capture_continuous_baseline()

    # ---------------------------------------------------------------------
    # Episode orchestration
    # ---------------------------------------------------------------------

    def _start_episode(self, upd: FrigateEventUpdate) -> None:
        now_t = float(self.clock.now())
        with self._lock:
            # Avoid restarting the same episode.
            if self._active and self._active.event_id == upd.event_id:
                return
            # Ignore if we are already confirming this event_id.
            if self._confirming_event_id == upd.event_id:
                return

            self.mqtt.publish_state("EPISODE_ACTIVE")

        # Try to use continuous baseline if it's recent (within 5 minutes).
        baseline: Optional[np.ndarray] = None
        pad_roi: Optional[ROI] = None
        mono: bool = False
        
        with self._lock:
            if self._continuous_baseline_roi is not None and self._continuous_baseline_ts > 0.0:
                age_s = now_t - self._continuous_baseline_ts
                if age_s <= 300.0:  # 5 minutes max age
                    baseline = self._continuous_baseline_roi.copy()
                    pad_roi = self._continuous_baseline_pad_roi
                    mono = self._continuous_baseline_mono
                    self.mqtt.log(f"📸 Using continuous baseline (age={age_s:.1f}s)")
        
        # Fall back to capturing baseline at episode start if continuous baseline is stale.
        if baseline is None:
            baseline, pad_roi, mono = self._capture_pad_baseline()

        with self._lock:
            self._active = _Episode(
                event_id=upd.event_id,
                zones=list(upd.zones or []),
                start_ts=now_t,
                baseline_roi=baseline,
                pad_roi=pad_roi,
                mono=bool(mono),
                has_clip=upd.has_clip,
            )

    def _end_episode(self, upd: FrigateEventUpdate) -> None:
        with self._lock:
            ep = self._active
            if ep is None or ep.event_id != upd.event_id:
                return
            if self._confirming_event_id == upd.event_id:
                return
            self._confirming_event_id = upd.event_id

        th = threading.Thread(target=self._confirm_episode_worker, args=(ep,), daemon=True)
        th.start()

    def _confirm_episode_worker(self, ep: _Episode) -> None:
        try:
            self.mqtt.publish_state("CONFIRMING")

            baseline = ep.baseline_roi
            pad_roi = ep.pad_roi
            mono = bool(ep.mono)

            # Determine baseline source for evidence/debugging.
            # Priority: continuous_baseline > episode_start > confirm_first_frame
            baseline_source = "unknown"
            if baseline is not None and baseline.size > 0:
                # Check if this baseline came from continuous capture (within 5 min of episode start).
                with self._lock:
                    if self._continuous_baseline_ts > 0.0 and (ep.start_ts - self._continuous_baseline_ts) <= 300.0:
                        baseline_source = "continuous_baseline"
                    else:
                        baseline_source = "episode_start"
            
            # If we failed to capture a baseline at start, fall back to first frame in window.
            if baseline is None or baseline.size == 0 or pad_roi is None:
                baseline_source = "confirm_first_frame"

            start = float(self.clock.now())
            max_window = float(
                max(
                    float(self.settings.residue_confirm_window_s),
                    float(self.settings.pee_confirm_window_s) if self.settings.pee_enabled else 0.0,
                )
            )

            poop_last_ok = 0.0
            poop_best_centroid: Optional[Point] = None
            poop_best_area = 0

            pee_last_ok = 0.0
            pee_best_centroid: Optional[Point] = None
            pee_best_area = 0
            pee_best_mass = 0

            fps = float(max(1.0, float(self.settings.snapshot_fps)))
            sleep_s = max(0.25, 1.0 / fps)

            while float(self.clock.now()) - start <= max_window:
                try:
                    img = self.frigate.fetch_snapshot()
                except Exception as e:
                    self.mqtt.log(f"⚠️ snapshot fail (confirm): {e}")
                    self.clock.sleep(sleep_s)
                    continue

                # Resolve pad ROI (cached) and crop.
                h, w = img.shape[:2]
                if pad_roi is None:
                    pad_roi = self._get_pad_roi(frame_w=w, frame_h=h, fallback_full=True)
                    if pad_roi is None:
                        self.clock.sleep(sleep_s)
                        continue

                cur_roi = self.cv.safe_roi(img, pad_roi)
                if cur_roi is None or cur_roi.size == 0:
                    self.clock.sleep(sleep_s)
                    continue

                # Initialize baseline if missing.
                if baseline is None:
                    baseline = cur_roi.copy()
                    mono = self.cv.is_monochrome_roi(baseline, self.settings.sat_mono_global)
                    baseline_source = "confirm_first_frame"

                now_t = float(self.clock.now())

                # 1) Poop candidate (solid blob) - priority over pee.
                blob = self.cv.diff_blob(
                    baseline,
                    cur_roi,
                    self.shape_params,
                    monochrome=mono,
                    min_area=int(self.settings.residue_min_area),
                )
                poop_detected_this_frame = blob is not None
                if poop_detected_this_frame:
                    if poop_last_ok == 0.0:
                        poop_last_ok = now_t
                        x, y, _, _ = pad_roi
                        poop_best_centroid = (x + int(blob.centroid[0]), y + int(blob.centroid[1]))
                        poop_best_area = int(blob.area)
                    elif (now_t - poop_last_ok) >= float(self.settings.residue_static_sec):
                        self._handle_elimination_confirmed(
                            kind=EliminationKind.POOP,
                            centroid=poop_best_centroid,
                            area=int(poop_best_area),
                            zones=list(ep.zones or []),
                            pad_roi=pad_roi,
                            baseline_roi=baseline,
                            mono=mono,
                            event_id=ep.event_id,
                            meta={"baseline_source": baseline_source},
                        )
                        return
                else:
                    poop_last_ok = 0.0

                # 2) Pee candidate (wetness delta/mass).
                if self.settings.pee_enabled and (not poop_detected_this_frame):
                    pee_cand = self.cv.diff_wet_candidate(
                        baseline,
                        cur_roi,
                        min_area=int(self.settings.pee_min_area),
                        delta_l_min=int(self.settings.pee_delta_l_min),
                        delta_l_bright_min=int(self.settings.pee_delta_l_bright_min),
                        delta_b_min=int(self.settings.pee_delta_b_min),
                        mass_min=int(self.settings.pee_mass_min),
                    )
                    if pee_cand is not None:
                        if pee_last_ok == 0.0:
                            pee_last_ok = now_t
                            x, y, _, _ = pad_roi
                            pee_best_centroid = (x + int(pee_cand.blob.centroid[0]), y + int(pee_cand.blob.centroid[1]))
                            pee_best_area = int(pee_cand.blob.area)
                            pee_best_mass = int(pee_cand.mass)
                        elif (now_t - pee_last_ok) >= float(self.settings.pee_static_sec):
                            inc_ok, inc_meta = self._passes_pee_increment_gate(
                                cur_roi=cur_roi,
                                now_t=now_t,
                                centroid=pee_best_centroid,
                            )
                            if inc_meta.get("increment_gate") in ("passed", "blocked_small_delta"):
                                self.mqtt.publish_increment_gate_debug(inc_meta)
                            if not inc_ok:
                                self.mqtt.log(
                                    f"⏸️ pee increment gate blocked "
                                    f"(mass={int(inc_meta.get('last_image_mass', 0))}, "
                                    f"min={int(inc_meta.get('min_mass', 0))})"
                                )
                                self.clock.sleep(sleep_s)
                                continue
                            self._handle_elimination_confirmed(
                                kind=EliminationKind.PEE,
                                centroid=pee_best_centroid,
                                area=int(pee_best_area),
                                zones=list(ep.zones or []),
                                pad_roi=pad_roi,
                                baseline_roi=baseline,
                                mono=mono,
                                event_id=ep.event_id,
                                meta={
                                    "baseline_source": baseline_source,
                                    "pee_mass": int(pee_best_mass),
                                    **inc_meta,
                                },
                                confirmed_roi=cur_roi,
                            )
                            return
                    else:
                        pee_last_ok = 0.0

                self.clock.sleep(sleep_s)

            # No confirmation.
            self.mqtt.publish_state("IDLE")
        finally:
            with self._lock:
                if self._confirming_event_id == ep.event_id:
                    self._confirming_event_id = None
                if self._active and self._active.event_id == ep.event_id:
                    self._active = None

    # ---------------------------------------------------------------------
    # Confirmation + publishing
    # ---------------------------------------------------------------------

    def _handle_elimination_confirmed(
        self,
        kind: EliminationKind,
        centroid: Optional[Point],
        area: int,
        zones: list[str],
        pad_roi: ROI,
        baseline_roi: np.ndarray,
        mono: bool,
        event_id: str,
        meta: Optional[dict] = None,
        confirmed_roi: Optional[np.ndarray] = None,
    ) -> None:
        if centroid is None:
            return

        now_t = float(self.clock.now())

        proper_zone = self.settings.proper_zone_name
        # Prefer blob centroid inside pad ROI (where residue landed) over dog zones.
        in_proper = None
        location_source = "zones"
        if pad_roi and len(pad_roi) == 4:
            in_proper = heuristics.point_in_roi(centroid, pad_roi)
            location_source = "centroid_in_roi"
        if in_proper is None and proper_zone:
            in_proper = (proper_zone in (zones or []))
            location_source = "zones"

        if in_proper is True:
            location = EliminationLocation.PROPER
        elif in_proper is False:
            location = EliminationLocation.OUTSIDE
        else:
            location = EliminationLocation.UNKNOWN

        # De-dup within a short window to avoid double counts on repeated end/update.
        with self._lock:
            if self._last_confirm_ts > 0.0 and self._last_confirm_centroid is not None:
                dt = float(now_t - self._last_confirm_ts)
                dist = float(heuristics.distance_px(centroid, self._last_confirm_centroid))

                cooldown_s = float(self.settings.ep_cooldown_s)
                radius_px = int(self.settings.merge_radius_px)
                if kind == EliminationKind.PEE:
                    cooldown_s = float(self.settings.pee_cooldown_s)
                    radius_px = int(self.settings.pee_merge_radius_px)
                
                # AGGRESSIVE POOP MERGE: If we already confirmed poop recently, assume ANY new poop
                # is just a second piece or the same event (split poop).
                # Merge all within 10 min regardless of distance (one "poop session").
                if kind == EliminationKind.POOP and self._last_confirm_kind == "poop":
                    cooldown_s = 600.0  # 10 minutes
                    radius_px = 2000    # Merge regardless of distance within cooldown

                if (self._last_confirm_kind == kind.value) and heuristics.should_merge_episode(dt, dist, cooldown_s, radius_px):
                    self.mqtt.log(
                        f"⏭️ dedup confirmed kind={kind.value} dt={dt:.1f}s dist={dist:.1f}px (cooldown={cooldown_s}s radius={radius_px}px)"
                    )
                    # For poop, still refresh poop-present baseline to keep cleanup monitor stable.
                    if kind == EliminationKind.POOP:
                        self._mark_poop_present(pad_roi, baseline_roi, mono)
                    return

            self._last_confirm_ts = now_t
            self._last_confirm_centroid = centroid
            self._last_confirm_kind = kind.value

        payload = {
            "ts": self.clock.now_iso(),
            "camera": self.settings.camera_name,
            "zones": zones or [],
            "kind": kind.value,
            "location": location.value,
            "proper_zone": proper_zone,
            "in_proper_zone": in_proper,
            "centroid": [int(centroid[0]), int(centroid[1])],
            "area": int(area),
            "confidence": None,
            "event_id": event_id,
        }

        # Attach Frigate URLs when we have an event id.
        if event_id:
            payload.update(self.frigate.get_event_urls(event_id))

        # Optional pad wetness context from a clean reference.
        if self._pad_ref_bgr is not None and self._pad_roi is not None:
            try:
                cur = self.frigate.fetch_snapshot()
                cur_roi = self.cv.safe_roi(cur, self._pad_roi)
                wm = self.cv.wetness_metrics(
                    self._pad_ref_bgr,
                    cur_roi,
                    delta_l_min=int(self.settings.pee_delta_l_min),
                    delta_l_bright_min=int(self.settings.pee_delta_l_bright_min),
                    delta_b_min=int(self.settings.pee_delta_b_min),
                )
                if isinstance(wm, dict):
                    payload["pad_wet_coverage"] = float(wm.get("wet_coverage", 0.0))
                    payload["pad_wet_mass"] = int(wm.get("wet_mass", 0))
            except Exception:
                pass

        self.mqtt.publish_elimination_event(payload)
        self._record_and_publish_stats(kind=kind, location=location, ts_unix=now_t)

        if kind == EliminationKind.PEE:
            # Keep the last confirmed pee frame as a reference for incremental
            # wetness comparison (pee-on-pee counting).
            if confirmed_roi is not None and confirmed_roi.size > 0:
                with self._lock:
                    self._last_confirmed_pee_roi = confirmed_roi.copy()
                    self._last_confirmed_pee_ts = now_t
                    self._last_confirmed_pee_centroid = centroid
            else:
                try:
                    snap = self.frigate.fetch_snapshot()
                    roi_img = self.cv.safe_roi(snap, pad_roi)
                    if roi_img is not None and roi_img.size > 0:
                        with self._lock:
                            self._last_confirmed_pee_roi = roi_img.copy()
                            self._last_confirmed_pee_ts = now_t
                            self._last_confirmed_pee_centroid = centroid
                except Exception:
                    pass
            # Persist metadata (ts, centroid) so after restart we do not report first_confirmed_pee.
            if self.state_store is not None and centroid is not None:
                try:
                    raw = self.state_store.load()
                    raw["last_confirmed_pee"] = {
                        "ts": float(now_t),
                        "centroid": [int(centroid[0]), int(centroid[1])],
                    }
                    self.state_store.save(raw)
                except Exception:
                    pass

        if kind == EliminationKind.POOP:
            # Publish legacy poop_present state for cleanup tracking.
            zone_out = proper_zone if location == EliminationLocation.PROPER else "outside"
            self.mqtt.publish_poop_present(
                {
                    "value": True,
                    "zone": zone_out,
                    "centroid": [int(centroid[0]), int(centroid[1])],
                    "area": int(area),
                    "ts": self.clock.now_iso(),
                }
            )
            # Optional: emit poop_event used by an existing HA automation to build a daily list.
            self.mqtt.publish_poop_event({"ts": self.clock.now_iso(), "event_id": event_id})
            # Mark as present and start cleanup monitoring.
            self._mark_poop_present(pad_roi, baseline_roi, mono)
            self._start_poop_cleanup_monitor()

        self.mqtt.publish_state("IDLE")

        if meta:
            self.mqtt.log(f"✅ confirmed kind={kind.value} loc={location.value} meta={meta} location_source={location_source}")
        else:
            self.mqtt.log(f"✅ confirmed kind={kind.value} loc={location.value} location_source={location_source}")

    # ---------------------------------------------------------------------
    # Poop cleanup monitoring (no coprophagy logic)
    # ---------------------------------------------------------------------

    def _mark_poop_present(self, pad_roi: ROI, baseline_roi: np.ndarray, mono: bool) -> None:
        with self._lock:
            self._poop_present = True
            self._poop_pad_roi = pad_roi
            self._poop_baseline_roi = None if baseline_roi is None else baseline_roi.copy()
            self._poop_mono = bool(mono)

    def _start_poop_cleanup_monitor(self) -> None:
        # One monitor thread at a time.
        th = threading.Thread(target=self._poop_cleanup_worker, daemon=True)
        th.start()

    def _poop_cleanup_worker(self) -> None:
        miss = 0
        while True:
            with self._lock:
                if not self._poop_present or self._poop_pad_roi is None or self._poop_baseline_roi is None:
                    return
                pad_roi = self._poop_pad_roi
                baseline = self._poop_baseline_roi
                mono = self._poop_mono

            self.clock.sleep(float(self.settings.check_interval_s))

            try:
                img = self.frigate.fetch_snapshot()
            except Exception:
                continue

            cur_roi = self.cv.safe_roi(img, pad_roi)
            if cur_roi is None or cur_roi.size == 0:
                continue

            blob = self.cv.diff_blob(
                baseline,
                cur_roi,
                self.shape_params,
                monochrome=mono,
                min_area=int(self.settings.residue_min_area),
            )

            if blob is None:
                miss += 1
            else:
                miss = 0

            if miss >= 3:
                with self._lock:
                    self._poop_present = False
                    self._poop_pad_roi = None
                    self._poop_baseline_roi = None
                self.mqtt.publish_poop_present(
                    {"value": False, "zone": self.settings.proper_zone_name, "centroid": [], "area": 0, "ts": self.clock.now_iso()}
                )
                return

    # ---------------------------------------------------------------------
    # Pad ROI + pad reference
    # ---------------------------------------------------------------------

    def _get_pad_roi(self, frame_w: int, frame_h: int, fallback_full: bool) -> Optional[ROI]:
        with self._lock:
            roi = self._pad_roi
        if roi is not None:
            x, y, rw, rh = roi
            if rw > 0 and rh > 0 and x >= 0 and y >= 0 and (x + rw) <= frame_w and (y + rh) <= frame_h:
                return roi

        roi = self.frigate.get_zone_bbox_for_snapshot(self.settings.toilet_zone, frame_w, frame_h)
        if roi is None and fallback_full:
            roi = (0, 0, int(frame_w), int(frame_h))

        with self._lock:
            self._pad_roi = roi
        return roi

    def _capture_pad_baseline(self) -> tuple[Optional[np.ndarray], Optional[ROI], bool]:
        """
        Capture a baseline ROI for the pad (median of a few snapshots).

        This baseline is used to detect new poop/pee *during this episode*, even if
        the pad already has previous pee stains.
        """
        frames: list[np.ndarray] = []
        pad_roi: Optional[ROI] = None

        for _ in range(3):
            try:
                img = self.frigate.fetch_snapshot()
            except Exception:
                self.clock.sleep(0.2)
                continue

            h, w = img.shape[:2]
            pad_roi = self._get_pad_roi(frame_w=w, frame_h=h, fallback_full=False)
            if pad_roi is None:
                self.clock.sleep(0.2)
                continue

            roi_img = self.cv.safe_roi(img, pad_roi)
            if roi_img is None or roi_img.size == 0:
                self.clock.sleep(0.2)
                continue

            frames.append(roi_img)
            self.clock.sleep(0.2)

        if not frames:
            return None, pad_roi, False

        # Median stack in uint8 space (robust to transient shadows).
        try:
            stack = np.stack(frames, axis=0).astype(np.float32)
            med = np.median(stack, axis=0).astype(np.uint8)
        except Exception:
            med = frames[-1].copy()

        mono = bool(self.cv.is_monochrome_roi(med, self.settings.sat_mono_global))
        return med, pad_roi, mono

    def _capture_continuous_baseline(self) -> None:
        """
        Capture a continuous baseline when the dog is absent.
        
        This baseline is used as a clean reference for episode detection, avoiding the
        problem of capturing the baseline AFTER elimination has started.
        
        Called periodically from check_episode_timeout when dog is absent.
        """
        # Don't capture if dog is present or if we have a recent baseline.
        now_t = float(self.clock.now())
        if (now_t - float(self._last_seen_dog_ts)) < float(self.settings.pad_ref_dog_absent_s):
            return
        
        # Don't capture too frequently (every 30s is enough).
        if self._continuous_baseline_ts > 0.0 and (now_t - self._continuous_baseline_ts) < 30.0:
            return
        
        try:
            img = self.frigate.fetch_snapshot()
            h, w = img.shape[:2]
            pad_roi = self._get_pad_roi(frame_w=w, frame_h=h, fallback_full=False)
            if pad_roi is None:
                return
            
            roi_img = self.cv.safe_roi(img, pad_roi)
            if roi_img is None or roi_img.size == 0:
                return
            
            mono = bool(self.cv.is_monochrome_roi(roi_img, self.settings.sat_mono_global))
            
            with self._lock:
                self._continuous_baseline_roi = roi_img.copy()
                self._continuous_baseline_ts = now_t
                self._continuous_baseline_mono = mono
                self._continuous_baseline_pad_roi = pad_roi
        except Exception:
            pass

    def _load_pad_reference(self) -> None:
        if self.state_store is None:
            return
        try:
            raw = self.state_store.load()
            pad = raw.get("pad") if isinstance(raw.get("pad"), dict) else {}
            ref_path = str(pad.get("ref_path") or self.settings.pad_ref_image_path)
            if ref_path and os.path.exists(ref_path):
                import cv2

                img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    self._pad_ref_bgr = img
        except Exception:
            self._pad_ref_bgr = None

    def _load_last_confirmed_pee(self) -> None:
        """
        Restore last confirmed pee metadata (ts, centroid) from state store.
        ROI is not persisted; first pee after restart will use last_image_stale due to age.
        """
        if self.state_store is None:
            return
        try:
            raw = self.state_store.load()
            lcp = raw.get("last_confirmed_pee")
            if not isinstance(lcp, dict):
                return
            ts = lcp.get("ts")
            centroid = lcp.get("centroid")
            if ts is None or not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
                return
            with self._lock:
                self._last_confirmed_pee_ts = float(ts)
                self._last_confirmed_pee_centroid = (int(centroid[0]), int(centroid[1]))
        except Exception:
            pass

    def _maybe_refresh_pad_reference(self) -> None:
        """
        Best-effort capture of a clean pad reference image after a person is detected.
        """
        if self.state_store is None:
            return

        with self._lock:
            pending = float(self._pending_pad_ref_ts or 0.0)
        if pending <= 0.0:
            return

        now_t = float(self.clock.now())
        if (now_t - pending) > float(self.settings.pad_ref_person_window_s):
            with self._lock:
                self._pending_pad_ref_ts = 0.0
            return

        # Wait until dog is absent.
        if (now_t - float(self._last_seen_dog_ts)) < float(self.settings.pad_ref_dog_absent_s):
            return

        # Ensure pad ROI.
        if self._pad_roi is None:
            try:
                img = self.frigate.fetch_snapshot()
            except Exception:
                return
            h, w = img.shape[:2]
            self._pad_roi = self._get_pad_roi(frame_w=w, frame_h=h, fallback_full=False)
        if self._pad_roi is None:
            return

        # Capture median of a few frames to reduce noise.
        frames: list[np.ndarray] = []
        for _ in range(5):
            try:
                img = self.frigate.fetch_snapshot()
            except Exception:
                self.clock.sleep(0.2)
                continue
            roi_img = self.cv.safe_roi(img, self._pad_roi)
            if roi_img is None or roi_img.size == 0:
                self.clock.sleep(0.2)
                continue
            frames.append(roi_img)
            self.clock.sleep(0.2)

        if not frames:
            return

        try:
            stack = np.stack(frames, axis=0).astype(np.float32)
            med = np.median(stack, axis=0).astype(np.uint8)
        except Exception:
            med = frames[-1].copy()

        # Persist to disk.
        import cv2

        ok, buf = cv2.imencode(".jpg", med, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return

        path = str(self.settings.pad_ref_image_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(buf.tobytes())

        # Refresh in-memory reference and metadata.
        self._pad_ref_bgr = med.copy()
        raw = self.state_store.load()
        raw["pad"] = {"ref_path": str(path), "ts": self.clock.now_iso(), "last_pad_reset_ts": float(now_t)}
        self.state_store.save(raw)

        # Throttle log to at most once per 5 minutes to reduce spam.
        if (now_t - self._last_pad_refresh_log_ts) >= 300.0:
            self.mqtt.log("🧼 Pad reference refreshed")
            self._last_pad_refresh_log_ts = now_t

        with self._lock:
            self._pending_pad_ref_ts = 0.0
            # The pad likely changed/was cleaned. Reset incremental pee compare state.
            self._last_confirmed_pee_roi = None
            self._last_confirmed_pee_ts = 0.0
            self._last_confirmed_pee_centroid = None

    def _passes_pee_increment_gate(
        self,
        cur_roi: Optional[np.ndarray],
        now_t: float,
        centroid: Optional[Point],
    ) -> tuple[bool, dict]:
        """
        Secondary gate for pee confirmation using last confirmed pee image.

        This helps differentiate "same image/no new pee" from real incremental
        wetness increase (e.g., pee count 2 -> 3). When the last confirmed pee
        image is older than last_image_max_age_s (default 240s), the gate allows
        the new pee without running the last-vs-current comparison, to avoid
        blocking real pees after long idle.
        """
        if not bool(self.settings.pee_increment_enabled):
            return True, {"increment_gate": "disabled"}

        if cur_roi is None or cur_roi.size == 0:
            return True, {"increment_gate": "cur_roi_missing"}

        with self._lock:
            last_roi = None if self._last_confirmed_pee_roi is None else self._last_confirmed_pee_roi.copy()
            last_ts = float(self._last_confirmed_pee_ts or 0.0)
            last_centroid = self._last_confirmed_pee_centroid

        if last_roi is None or last_ts <= 0.0:
            return True, {"increment_gate": "first_confirmed_pee"}

        age_s = float(now_t - last_ts)
        max_age_s = float(self.settings.last_image_max_age_s)
        if age_s > max_age_s:
            return True, {"increment_gate": "last_image_stale", "last_image_age_s": round(age_s, 1)}

        # If the new centroid is far from the previous confirmed pee, allow.
        if centroid is not None and last_centroid is not None:
            dist = float(heuristics.distance_px(centroid, last_centroid))
            if dist >= float(self.settings.pee_merge_radius_px):
                return True, {"increment_gate": "far_from_last_pee", "last_dist_px": int(dist)}

        wm = self.cv.wetness_metrics(
            last_roi,
            cur_roi,
            delta_l_min=int(self.settings.pee_delta_l_min),
            delta_l_bright_min=int(self.settings.pee_delta_l_bright_min),
            delta_b_min=int(self.settings.pee_delta_b_min),
        )
        if not isinstance(wm, dict):
            # Fail open to preserve recall if metrics are unavailable.
            return True, {"increment_gate": "delta_unavailable"}

        mass = int(wm.get("wet_mass", 0))
        coverage = float(wm.get("wet_coverage", 0.0))
        min_mass = int(self.settings.pee_increment_mass_min)
        if mass >= min_mass:
            return True, {
                "increment_gate": "passed",
                "last_image_mass": int(mass),
                "min_mass": int(min_mass),
                "last_image_wet_coverage": round(coverage, 6),
            }

        return False, {
            "increment_gate": "blocked_small_delta",
            "last_image_mass": int(mass),
            "min_mass": int(min_mass),
            "last_image_wet_coverage": round(coverage, 6),
        }

    def _update_pad_wetness_state(self) -> None:
        """
        Update persisted pad wetness metrics from the clean reference (best-effort).
        This does not affect counts; it's only helpful context and for pad-change debugging.
        """
        if self.state_store is None or self._pad_ref_bgr is None or self._pad_roi is None:
            return
        try:
            img = self.frigate.fetch_snapshot()
            cur_roi = self.cv.safe_roi(img, self._pad_roi)
            wm = self.cv.wetness_metrics(
                self._pad_ref_bgr,
                cur_roi,
                delta_l_min=int(self.settings.pee_delta_l_min),
                delta_l_bright_min=int(self.settings.pee_delta_l_bright_min),
                delta_b_min=int(self.settings.pee_delta_b_min),
            )
            if not isinstance(wm, dict):
                return
            raw = self.state_store.load()
            raw["pad_wetness"] = {"ts": self.clock.now_iso(), **wm}
            self.state_store.save(raw)
        except Exception:
            return

    # ---------------------------------------------------------------------
    # History + stats publishing
    # ---------------------------------------------------------------------

    def _load_history(self) -> None:
        if self.state_store is None:
            return
        try:
            raw = self.state_store.load()
            hist = raw.get("elimination_history")
            if isinstance(hist, list):
                self._history = [x for x in hist if isinstance(x, dict)]
        except Exception:
            self._history = []

    def _record_and_publish_stats(self, kind: EliminationKind, location: EliminationLocation, ts_unix: float) -> None:
        """
        Keep a rolling 30d history and publish aggregate stats to MQTT (retained).
        """
        # Record
        entry = {
            "ts": self.clock.now_iso(),
            "ts_unix": float(ts_unix),
            "kind": kind.value,
            "location": location.value,
        }
        self._history.append(entry)

        # Prune > window days.
        cutoff = float(ts_unix) - float(self._history_window_days * 86400)
        self._history = [e for e in self._history if float(e.get("ts_unix") or 0.0) >= cutoff]

        if self.state_store is not None:
            try:
                raw = self.state_store.load()
                raw["elimination_history"] = self._history[-5000:]  # hard cap
                self.state_store.save(raw)
            except Exception:
                pass

        # Build stats payload.
        stats = self._build_stats_payload()
        self.mqtt.publish_elimination_stats(stats)

    def _build_stats_payload(self) -> dict:
        now_iso = self.clock.now_iso()
        # counts by kind and location
        def counts_for(kind: str) -> dict:
            proper = 0
            outside = 0
            for e in self._history:
                if e.get("kind") != kind:
                    continue
                loc = e.get("location")
                if loc == "proper":
                    proper += 1
                elif loc == "outside":
                    outside += 1
            return {"total": proper + outside, "proper": proper, "outside": outside}

        # Typical time of day (mean clock time) over the window.
        def mean_time_of_day(kind: str) -> dict:
            minutes: list[float] = []
            for e in self._history:
                if e.get("kind") != kind:
                    continue
                ts = e.get("ts")
                try:
                    # ISO string parse: rely on Home Assistant format (timezone-aware).
                    from datetime import datetime

                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    m = dt.hour * 60 + dt.minute + (dt.second / 60.0)
                    minutes.append(float(m))
                except Exception:
                    continue

            if not minutes:
                return {"mean_minutes": None, "mean_time": None}

            # Circular mean on 24h clock.
            angles = [2.0 * math.pi * (m / 1440.0) for m in minutes]
            x = sum(math.cos(a) for a in angles) / float(len(angles))
            y = sum(math.sin(a) for a in angles) / float(len(angles))
            mean_a = math.atan2(y, x)
            if mean_a < 0:
                mean_a += 2.0 * math.pi
            mean_m = (mean_a / (2.0 * math.pi)) * 1440.0
            hh = int(mean_m // 60) % 24
            mm = int(mean_m % 60)
            return {"mean_minutes": float(mean_m), "mean_time": f"{hh:02d}:{mm:02d}"}

        return {
            "ts": now_iso,
            "window_days": int(self._history_window_days),
            "poop": {"count": counts_for("poop"), "typical_time": mean_time_of_day("poop")},
            "pee": {"count": counts_for("pee"), "typical_time": mean_time_of_day("pee")},
        }

