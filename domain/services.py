"""
Domain services - Orchestration logic for business workflows.
Coordinates between domain logic and adapters without direct I/O.
"""
import threading
from typing import Optional, List, Callable
import numpy as np

from .models import (
    TrackState, DogDetection, EllieState, Blob, 
    ShapeParams, CoprophagyThresholds, Point, ROI
)
from .fsm import EllieFSM, Signal, Command
from . import heuristics
from ..settings import Settings


class EllieWatcherService:
    """Main service orchestrating Ellie's behavior monitoring."""
    
    def __init__(
        self,
        settings: Settings,
        clock,  # Clock interface
        cv_ops,  # CV operations module
        frigate_client,  # FrigateClient
        mqtt_client,  # MQTTClient
    ):
        self.settings = settings
        self.clock = clock
        self.cv = cv_ops
        self.frigate = frigate_client
        self.mqtt = mqtt_client
        
        # State
        self.state = TrackState()
        self.fsm = EllieFSM()
        self.last_seen_dog_ts = 0.0
        
        # Shape parameters
        self.shape_params = ShapeParams(
            min_blob_w=settings.min_blob_w,
            min_blob_h=settings.min_blob_h,
            min_circularity=settings.min_circularity,
            min_solidity=settings.min_solidity,
            hsv_s_min=settings.hsv_s_min,
            tex_min_mono=settings.tex_min_mono,
            spec_max_mono=settings.spec_max_mono,
        )
        
        # Coprophagy thresholds
        self.copro_thresholds = CoprophagyThresholds(
            copro_drop_frac=settings.copro_drop_frac,
            cum_drop_frac=settings.cum_drop_frac,
            cum_window_s=settings.cum_window_s,
        )
    
    def handle_dog_detection(self, detection: DogDetection) -> None:
        """
        Handle incoming dog detection from Frigate.
        
        Args:
            detection: Dog detection data
        """
        self.mqtt.log(
            f"🐶 Zones={detection.zones}, ratio={detection.ratio:.2f}, "
            f"stationary={detection.stationary}, motionless={detection.motionless_count}, "
            f"speed={detection.speed:.2f}"
        )
        
        # Update tracking
        self.state.last_ts = detection.timestamp
        self.last_seen_dog_ts = detection.timestamp
        self.state.last_bbox = detection.bbox
        
        # Score squat
        squat_score = heuristics.score_squat(
            ratio=detection.ratio,
            stationary=detection.stationary,
            speed=detection.speed,
            motionless_count=detection.motionless_count,
            squat_thresh=self.settings.squat_score_thresh,
            motionless_min=self.settings.motionless_min,
            speed_max_still=self.settings.speed_max_still,
        )
        
        self.mqtt.log(f"Squat score: {squat_score.score:.3f}, squatting={squat_score.is_squatting}")
        
        if squat_score.is_squatting:
            self._handle_squatting(detection)
        else:
            self._handle_not_squatting()
    
    def _handle_squatting(self, detection: DogDetection) -> None:
        """Handle dog in squatting position."""
        # Fetch snapshot
        try:
            img = self.frigate.fetch_snapshot()
        except Exception as e:
            self.mqtt.log(f"⚠️ Snapshot fail: {e}")
            return
        
        t = self.clock.now()
        
        if not self.state.in_squat:
            # Start squat
            self.state.squat_start = t
            self.state.in_squat = True
            
            transition = self.fsm.transition(
                Signal.SQUAT_START,
                {"timestamp": t}
            )
            self._execute_commands(transition.commands)
            
            # Open episode if needed
            if not self.state.episode_active:
                self.state.episode_active = True
                self.state.episode_started_ts = t
        else:
            # Continue squat
            duration = t - self.state.squat_start
            if duration >= self.settings.squat_min_duration_s:
                transition = self.fsm.transition(
                    Signal.SQUAT_CONTINUE,
                    {
                        "duration": duration,
                        "min_duration": self.settings.squat_min_duration_s,
                    }
                )
                self._execute_commands(transition.commands)
        
        # Update ROI anchor and background
        frame_h, frame_w = img.shape[:2]
        self.state.roi_anchor = heuristics.calculate_roi_from_bbox(
            detection.bbox,
            frame_w,
            frame_h,
            pad=20
        )
        
        roi_quick = self.cv.safe_roi(img, self.state.roi_anchor)
        self.state.bg_roi = None if roi_quick is None else roi_quick.copy()
    
    def _handle_not_squatting(self) -> None:
        """Handle dog not in squatting position."""
        if self.state.in_squat:
            self.state.in_squat = False
            
            transition = self.fsm.transition(
                Signal.SQUAT_END,
                {"timestamp": self.clock.now()}
            )
            self._execute_commands(transition.commands)
    
    def handle_person_detection(self, timestamp: float) -> None:
        """
        Handle person detection in zone.
        
        Args:
            timestamp: Detection timestamp
        """
        self.state.last_person_ts = timestamp
        self.mqtt.log(f"🧍 Person in zone {self.settings.toilet_zone}")
    
    def start_confirmation_window(self) -> None:
        """Start residue confirmation window in background thread."""
        threading.Thread(
            target=self._confirmation_window_worker,
            daemon=True
        ).start()
    
    def _confirmation_window_worker(self) -> None:
        """Worker for residue confirmation window."""
        start = self.clock.now()
        last_ok = 0.0
        best_centroid: Optional[Point] = None
        best_area = 0
        
        if self.state.roi_anchor is None:
            self.mqtt.log("⚠️ No roi_anchor for confirmation window")
            self.mqtt.publish_state(EllieState.IDLE.value)
            return
        
        self.mqtt.log("🔍 Starting residue confirmation window")
        baseline: Optional[np.ndarray] = None
        
        while self.clock.now() - start <= self.settings.residue_confirm_window_s:
            try:
                img = self.frigate.fetch_snapshot()
            except Exception as e:
                self.mqtt.log(f"⚠️ Snapshot fail (confirm): {e}")
                self.clock.sleep(max(0.1, 1.0 / max(1.0, self.settings.snapshot_fps)))
                continue
            
            cur_roi = self.cv.safe_roi(img, self.state.roi_anchor)
            if cur_roi is None or cur_roi.size == 0:
                self.clock.sleep(max(0.1, 1.0 / max(1.0, self.settings.snapshot_fps)))
                continue
            
            # Capture baseline
            if baseline is None:
                baseline = cur_roi.copy()
                self.state.residue_bg = baseline.copy()
                self.state.mono_flag = self.cv.is_monochrome_roi(
                    baseline,
                    self.settings.sat_mono_global
                )
                self.mqtt.log(f"📸 Baseline captured, mono={self.state.mono_flag}")
                self.clock.sleep(max(0.1, 1.0 / max(1.0, self.settings.snapshot_fps)))
                continue
            
            # Check for blob
            blob = self.cv.diff_blob(
                baseline,
                cur_roi,
                self.shape_params,
                monochrome=self.state.mono_flag,
                min_area=self.settings.residue_min_area,
            )
            
            if blob:
                if last_ok == 0.0:
                    last_ok = self.clock.now()
                    x, y, w, h = self.state.roi_anchor
                    best_centroid = (x + blob.centroid[0], y + blob.centroid[1])
                    best_area = blob.area
                    self.mqtt.log(f"🟡 Possible residue: area={blob.area}")
                
                elif self.clock.now() - last_ok >= self.settings.residue_static_sec:
                    # Confirmed!
                    self._handle_residue_confirmed(best_centroid, best_area)
                    return
            else:
                if last_ok != 0.0:
                    self.mqtt.log("🔄 Blob disappeared before stabilization")
                last_ok = 0.0
            
            self.clock.sleep(max(0.1, 1.0 / max(1.0, self.settings.snapshot_fps)))
        
        # Window ended without confirmation
        self.mqtt.log("⏹️ Confirmation window ended without residue")
        transition = self.fsm.transition(
            Signal.RESIDUE_NOT_FOUND,
            {"timestamp": self.clock.now_iso()}
        )
        self._execute_commands(transition.commands)
    
    def _handle_residue_confirmed(
        self,
        centroid: Optional[Point],
        area: int
    ) -> None:
        """Handle confirmed residue detection."""
        if centroid is None:
            return
        
        # Check for episode merging
        if self.state.episode_active and self.state.last_confirm_ts > 0:
            dt = self.clock.now() - self.state.last_confirm_ts
            dist = 1e9
            
            if self.state.last_confirm_centroid is not None:
                dist = heuristics.distance_px(centroid, self.state.last_confirm_centroid)
            
            if heuristics.should_merge_episode(
                dt,
                dist,
                self.settings.ep_cooldown_s,
                self.settings.merge_radius_px
            ):
                self.mqtt.log(f"➿ Merged to same episode (dt={dt:.1f}s, dist={int(dist)}px)")
                self._update_poop_state(centroid, area)
                self._start_monitoring()
                return
        
        # New confirmation
        self.state.last_confirm_ts = self.clock.now()
        self.state.last_confirm_centroid = centroid
        
        self._update_poop_state(centroid, area)
        
        # Find nearby event
        evt_id = self.frigate.find_nearby_dog_event(self.clock.now())
        
        # Transition FSM
        transition = self.fsm.transition(
            Signal.RESIDUE_CONFIRMED,
            {
                "zone": self.settings.toilet_zone,
                "centroid": list(centroid),
                "area": area,
                "timestamp": self.clock.now_iso(),
                "event_id": evt_id,
                "camera": self.settings.camera_name,
                "sub_label": self.settings.frigate_event_poop_sub_label,
            }
        )
        self._execute_commands(transition.commands)
    
    def _update_poop_state(self, centroid: Point, area: int) -> None:
        """Update poop state tracking."""
        self.state.poop_present = True
        self.state.poop_roi = self.state.roi_anchor
        self.state.poop_centroid = centroid
        self.state.poop_area0 = area
        self.state.cum_drop_ref_ts = self.clock.now()
        self.state.cum_area_min = area
        self.state.risk_announced = False
        self.state.near_since = 0.0
        self.state.was_near = False
        self.state.visit_person_flag = False
        self.state.last_near_end_ts = 0.0
    
    def _start_monitoring(self) -> None:
        """Start monitoring threads."""
        threading.Thread(target=self._monitor_poop_present, daemon=True).start()
        threading.Thread(target=self._monitor_coprophagy, daemon=True).start()
    
    def _monitor_poop_present(self) -> None:
        """Monitor for poop cleanup."""
        miss = 0
        self.mqtt.log("🧼 Monitoring poop cleanup...")
        
        while self.state.poop_present and self.state.poop_roi is not None:
            self.clock.sleep(self.settings.check_interval_s)
            
            try:
                img = self.frigate.fetch_snapshot()
            except Exception:
                continue
            
            cur_roi = self.cv.safe_roi(img, self.state.poop_roi)
            if cur_roi is None or cur_roi.size == 0:
                continue
            
            baseline = self.state.residue_bg
            blob = self.cv.diff_blob(
                baseline,
                cur_roi,
                self.shape_params,
                monochrome=self.state.mono_flag,
                min_area=self.settings.residue_min_area,
            )
            
            if blob is None:
                miss += 1
            else:
                miss = 0
            
            if miss >= 3:
                # Check if dog was present
                dog_near = self.frigate.check_recent_dog_presence(15.0)
                
                if dog_near:
                    self.mqtt.log("🍽️ Coprophagy detected (dog present during removal)")
                    self.mqtt.publish_coprophagy_event({
                        "value": True,
                        "zone": self.settings.toilet_zone,
                        "ts": self.clock.now_iso(),
                    })
                    
                    evt_id = self.frigate.find_nearby_dog_event(self.clock.now())
                    if evt_id:
                        self.frigate.update_event_sub_label(
                            evt_id,
                            self.settings.frigate_event_coprophagy_sub_label
                        )
                
                # Poop cleaned
                self.state.poop_present = False
                self.mqtt.log("🧹 Poop removed → IDLE")
                
                transition = self.fsm.transition(
                    Signal.POOP_CLEANED,
                    {
                        "zone": self.settings.toilet_zone,
                        "timestamp": self.clock.now_iso(),
                        "dog_present": dog_near,
                        "event_id": self.frigate.find_nearby_dog_event(self.clock.now()) if dog_near else None,
                    }
                )
                self._execute_commands(transition.commands)
                return
    
    def _monitor_coprophagy(self) -> None:
        """Monitor for coprophagy behavior."""
        self.mqtt.log("👀 Monitoring coprophagy (proximity/area)...")
        
        area0 = max(1, self.state.poop_area0)
        visit_near = False
        visit_start = 0.0
        visit_person = False
        last_near = False
        
        while self.state.poop_present and self.state.poop_roi is not None:
            self.clock.sleep(1.0)
            
            # Check if dog is near
            fresh = (
                self.clock.now() - self.last_seen_dog_ts <= 2.0
                and self.state.last_bbox is not None
            )
            
            near = False
            if fresh and self.state.poop_centroid is not None:
                dog_center = (
                    (self.state.last_bbox[0] + self.state.last_bbox[2]) // 2,
                    (self.state.last_bbox[1] + self.state.last_bbox[3]) // 2,
                )
                near = heuristics.is_near_poop(
                    dog_center,
                    self.state.poop_centroid,
                    self.settings.near_radius_px
                )
            
            # Track person during visit
            if near and (self.clock.now() - self.state.last_person_ts) <= 3.0:
                visit_person = True
            
            # Visit start
            if near and not last_near:
                visit_near = True
                visit_start = self.clock.now()
                visit_person = False
                self.mqtt.log("➡️ Started visit to poop")
            
            # Risk announcement
            if (
                near
                and visit_near
                and not self.state.risk_announced
                and (self.clock.now() - visit_start) >= self.settings.risk_dur_s
            ):
                self.state.risk_announced = True
                self.mqtt.publish_coprophagy_risk({
                    "ts": self.clock.now_iso(),
                    "zone": self.settings.toilet_zone,
                    "centroid": list(self.state.poop_centroid) if self.state.poop_centroid else [],
                    "since": int(visit_start),
                    "duration_s": int(self.clock.now() - visit_start),
                })
                self.mqtt.log("⚠️ Coprophagy risk published")
            
            # Visit end - measure area
            if (not near) and last_near:
                self.state.last_near_end_ts = self.clock.now()
                measured = self._measure_poop_area_post_visit()
                
                drop_frac = max(0.0, float(area0 - measured) / float(area0))
                
                # Update cumulative tracking
                now_t = self.clock.now()
                if self.state.cum_drop_ref_ts == 0.0:
                    self.state.cum_drop_ref_ts = now_t
                
                self.state.cum_area_min = min(
                    self.state.cum_area_min or measured or area0,
                    measured or area0
                )
                
                cum_drop = max(0.0, float(area0 - self.state.cum_area_min) / float(area0))
                window_elapsed = now_t - self.state.cum_drop_ref_ts
                
                self.mqtt.log(
                    f"⤵️ Post-visit: area_now={measured} (A0={area0}) "
                    f"drop={drop_frac:.2f} cum={cum_drop:.2f} person={visit_person}"
                )
                
                # Check for coprophagy
                if heuristics.is_coprophagy(
                    area0,
                    measured,
                    self.state.cum_area_min,
                    visit_person,
                    self.copro_thresholds,
                    window_elapsed,
                ):
                    evt_id = self.frigate.find_nearby_dog_event(
                        self.state.last_near_end_ts or self.clock.now()
                    )
                    
                    transition = self.fsm.transition(
                        Signal.COPROPHAGY_CONFIRMED,
                        {
                            "camera": self.settings.camera_name,
                            "zone": self.settings.toilet_zone,
                            "timestamp": self.clock.now_iso(),
                            "event_id": evt_id,
                            "manual_event_id": None,
                        }
                    )
                    self._execute_commands(transition.commands)
                    self.mqtt.log("✅ Coprophagy confirmed")
            
            last_near = near
    
    def _measure_poop_area_post_visit(self) -> int:
        """Measure poop area after dog leaves."""
        end_deadline = self.clock.now() + self.settings.area_post_leave_s
        measured = 0
        
        while self.clock.now() <= end_deadline:
            try:
                img = self.frigate.fetch_snapshot()
                cur_roi = self.cv.safe_roi(img, self.state.poop_roi)
                base = self.state.residue_bg
                
                measured = self.cv.diff_area(
                    base,
                    cur_roi,
                    self.shape_params,
                    ignore_color=self.settings.ignore_sat_night,
                    min_area=self.settings.residue_min_area,
                )
                break
            except Exception:
                self.clock.sleep(0.4)
        
        return measured
    
    def _execute_commands(self, commands: List[Command]) -> None:
        """Execute commands from FSM."""
        for cmd in commands:
            if cmd.type == "publish_state":
                self.mqtt.publish_state(cmd.data["state"])
            
            elif cmd.type == "open_episode":
                # Already handled in state
                pass
            
            elif cmd.type == "start_confirmation_window":
                self.start_confirmation_window()
            
            elif cmd.type == "publish_poop_present":
                self.mqtt.publish_poop_present(cmd.data)
            
            elif cmd.type == "publish_poop_event":
                # Build full event data
                event_data = cmd.data.copy()
                if cmd.data.get("event_id"):
                    urls = self.frigate.get_event_urls(cmd.data["event_id"])
                    event_data.update(urls)
                self.mqtt.publish_poop_event(event_data)
            
            elif cmd.type == "update_frigate_sub_label":
                self.frigate.update_event_sub_label(
                    cmd.data["event_id"],
                    cmd.data["sub_label"]
                )
            
            elif cmd.type == "start_poop_monitor":
                self._start_monitoring()
            
            elif cmd.type == "start_coprophagy_monitor":
                # Already started in start_poop_monitor
                pass
            
            elif cmd.type == "publish_coprophagy_event":
                event_data = cmd.data.copy()
                if cmd.data.get("event_id"):
                    urls = self.frigate.get_event_urls(cmd.data["event_id"])
                    event_data.update(urls)
                self.mqtt.publish_coprophagy_event(event_data)
    
    def check_episode_timeout(self) -> None:
        """Check if episode should be closed due to dog absence."""
        if (
            self.state.episode_active
            and (self.clock.now() - self.last_seen_dog_ts) >= self.settings.leave_timeout_s
        ):
            self.state.episode_active = False
            self.mqtt.log("Episode closed (dog left)")

