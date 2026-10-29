"""
Application runner - Dependency injection and wiring.
"""
import time
from settings import load_settings
from adapters.clock import SystemClock
from adapters import cv_ops
from adapters.frigate_client import FrigateClient
from adapters.mqtt_client import MQTTClient
from adapters.state_store import JsonStateStore
from domain.services import EllieWatcherService
from app.handlers import MQTTHandlers


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
        self.state_store = JsonStateStore(self.settings.state_store_path)
        
        # Initialize service
        self.service = EllieWatcherService(
            settings=self.settings,
            clock=self.clock,
            cv_ops=cv_ops,
            frigate_client=self.frigate,
            mqtt_client=self.mqtt,
            state_store=self.state_store,
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
            "Heuristics:",
            f"poop_min_area={self.settings.residue_min_area}",
            f"pee_min_area={self.settings.pee_min_area}",
            f"confirm_win_s=max({self.settings.residue_confirm_window_s},{self.settings.pee_confirm_window_s})",
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
            f"RES_WIN_S={self.settings.residue_confirm_window_s}, "
            f"RES_MIN_A={self.settings.residue_min_area}, "
            f"RES_STATIC_S={self.settings.residue_static_sec}, "
            f"SNAP_FPS={self.settings.snapshot_fps}, "
            f"CHECK_INT_S={self.settings.check_interval_s}, "
            f"EP_COOLDOWN_S={self.settings.ep_cooldown_s}, "
            f"MERGE_RADIUS_PX={self.settings.merge_radius_px}, "
            f"MIN_CIRCULARITY={self.settings.min_circularity}, "
            f"MIN_SOLIDITY={self.settings.min_solidity}, "
            f"HSV_S_MIN={self.settings.hsv_s_min}, "
            f"PEE_MIN_A={self.settings.pee_min_area}, "
            f"PEE_STATIC_S={self.settings.pee_static_sec}, "
            f"PEE_MASS_MIN={self.settings.pee_mass_min}, "
            f"PEE_INC_EN={self.settings.pee_increment_enabled}, "
            f"PEE_INC_MASS_MIN={self.settings.pee_increment_mass_min}, "
            f"LAST_IMG_MAX_AGE_S={self.settings.last_image_max_age_s}"
        )


def run() -> None:
    """Entry point for running the application."""
    app = Application()
    app.run()

