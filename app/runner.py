"""
Application runner - Dependency injection and wiring.
"""
import time
from ..settings import load_settings
from ..adapters.clock import SystemClock
from ..adapters import cv_ops
from ..adapters.frigate_client import FrigateClient
from ..adapters.mqtt_client import MQTTClient
from ..domain.services import EllieWatcherService
from .handlers import MQTTHandlers


class Application:
    """Main application orchestrator."""
    
    def __init__(self):
        """Initialize application with all dependencies."""
        # Load settings
        self.settings = load_settings()
        
        # Initialize adapters
        self.clock = SystemClock()
        self.frigate = FrigateClient(self.settings)
        self.mqtt = MQTTClient(
            self.settings,
            debug_enabled=self.settings.enable_debug_watcher
        )
        
        # Initialize service
        self.service = EllieWatcherService(
            settings=self.settings,
            clock=self.clock,
            cv_ops=cv_ops,
            frigate_client=self.frigate,
            mqtt_client=self.mqtt,
        )
        
        # Initialize handlers
        self.handlers = MQTTHandlers(self.service, self.settings)
        
        # Wire MQTT callbacks
        self.mqtt.set_on_connect(self.handlers.on_connect)
        self.mqtt.set_on_message(self.handlers.on_message)
    
    def run(self) -> None:
        """Run the application."""
        # Log startup
        self.mqtt.log(
            "[ellie-watcher] running with:",
            f"camera={self.settings.camera_name}",
            f"zone={self.settings.toilet_zone}",
            f"mqtt={self.settings.mqtt_host}:{self.settings.mqtt_port}"
        )
        self.mqtt.log(
            f"Heuristics: squat_thr={self.settings.squat_score_thresh} "
            f"min_dur={self.settings.squat_min_duration_s}s"
        )
        
        if self.settings.enable_debug_watcher:
            self._log_all_settings()
        
        # Connect and start MQTT
        self.mqtt.connect()
        self.mqtt.start()
        
        # Main loop - check for episode timeouts
        try:
            while True:
                time.sleep(1)
                self.service.check_episode_timeout()
        except KeyboardInterrupt:
            self.mqtt.log("Shutting down...")
            self.mqtt.stop()
    
    def _log_all_settings(self) -> None:
        """Log all settings for debugging."""
        self.mqtt.log(
            f"All settings: "
            f"SQUAT_THR={self.settings.squat_score_thresh}, "
            f"SQUAT_MIN_S={self.settings.squat_min_duration_s}, "
            f"RES_WIN_S={self.settings.residue_confirm_window_s}, "
            f"RES_MIN_A={self.settings.residue_min_area}, "
            f"RES_STATIC_S={self.settings.residue_static_sec}, "
            f"SNAP_FPS={self.settings.snapshot_fps}, "
            f"CHECK_INT_S={self.settings.check_interval_s}, "
            f"EP_COOLDOWN_S={self.settings.ep_cooldown_s}, "
            f"LEAVE_TIMEOUT_S={self.settings.leave_timeout_s}, "
            f"MERGE_RADIUS_PX={self.settings.merge_radius_px}, "
            f"MIN_CIRCULARITY={self.settings.min_circularity}, "
            f"MIN_SOLIDITY={self.settings.min_solidity}, "
            f"HSV_S_MIN={self.settings.hsv_s_min}, "
            f"MOTIONLESS_MIN={self.settings.motionless_min}, "
            f"SPEED_MAX_STILL={self.settings.speed_max_still}"
        )


def run() -> None:
    """Entry point for running the application."""
    app = Application()
    app.run()

