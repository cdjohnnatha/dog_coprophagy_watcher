"""
Settings module - Centralizes all environment variables using Pydantic.
"""
import os
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # MQTT Configuration
    mqtt_host: str = Field(default="localhost", alias="MQTT_HOST")
    mqtt_port: int = Field(default=1883, alias="MQTT_PORT")
    mqtt_user: str = Field(default="", alias="MQTT_USER")
    mqtt_pass: str = Field(default="", alias="MQTT_PASS")
    mqtt_prefix: str = Field(default="frigate", alias="MQTT_PREFIX")
    
    # Frigate Configuration
    frigate_base_url: str = Field(default="http://frigate:5000", alias="FRIGATE_BASE_URL")
    frigate_public_url: str = Field(default="", alias="FRIGATE_PUBLIC_URL")
    camera_name: str = Field(default="dog", alias="CAMERA_NAME")
    toilet_zone: str = Field(default="poop_zone", alias="TOILET_ZONE")
    
    # Squat Detection Thresholds
    squat_score_thresh: float = Field(default=0.65, alias="SQUAT_SCORE_THRESH")
    squat_min_duration_s: float = Field(default=5.0, alias="SQUAT_MIN_DURATION_S")
    
    # Residue Confirmation
    residue_confirm_window_s: float = Field(default=20.0, alias="RESIDUE_CONFIRM_WINDOW_S")
    residue_min_area: int = Field(default=140, alias="RESIDUE_MIN_AREA")
    residue_static_sec: float = Field(default=2.0, alias="RESIDUE_STATIC_SEC")
    
    # Snapshot Configuration
    snapshot_fps: float = Field(default=4.0, alias="SNAPSHOT_FPS")
    check_interval_s: float = Field(default=15.0, alias="CHECK_INTERVAL_S")
    
    # Debug
    enable_debug_watcher: bool = Field(default=False, alias="ENABLE_DEBUG_WATCHER")
    
    # Episode Management
    ep_cooldown_s: float = Field(default=180.0, alias="EP_COOLDOWN_S")
    leave_timeout_s: float = Field(default=8.0, alias="LEAVE_TIMEOUT_S")
    merge_radius_px: int = Field(default=90, alias="MERGE_RADIUS_PX")
    
    # Anti-Pee / Shape Filtering
    min_blob_w: int = Field(default=14, alias="MIN_BLOB_W")
    min_blob_h: int = Field(default=14, alias="MIN_BLOB_H")
    min_circularity: float = Field(default=0.28, alias="MIN_CIRCULARITY")
    min_solidity: float = Field(default=0.65, alias="MIN_SOLIDITY")
    hsv_s_min: int = Field(default=25, alias="HSV_S_MIN")
    sat_mono_global: int = Field(default=12, alias="SAT_MONO_GLOBAL")
    tex_min_mono: float = Field(default=25.0, alias="TEX_MIN_MONO")
    spec_max_mono: float = Field(default=0.18, alias="SPEC_MAX_MONO")
    
    # Still Position Detection
    motionless_min: int = Field(default=3, alias="MOTIONLESS_MIN")
    speed_max_still: float = Field(default=0.15, alias="SPEED_MAX_STILL")
    
    # Frigate Event Labels
    frigate_event_poop_sub_label: str = Field(default="poop", alias="FRIGATE_EVENT_POOP_SUB_LABEL")
    frigate_event_coprophagy_sub_label: str = Field(default="coprophagy", alias="FRIGATE_EVENT_COPROPHAGY_SUB_LABEL")
    
    # Coprophagy Detection
    near_radius_px: int = Field(default=80, alias="NEAR_RADIUS_PX")
    risk_dur_s: float = Field(default=8.0, alias="RISK_DUR_S")
    leave_near_gap_s: float = Field(default=2.0, alias="LEAVE_NEAR_GAP_S")
    copro_drop_frac: float = Field(default=0.50, alias="COPRO_DROP_FRAC")
    cum_drop_frac: float = Field(default=0.70, alias="CUM_DROP_FRAC")
    cum_window_s: float = Field(default=900.0, alias="CUM_WINDOW_S")
    area_post_leave_s: float = Field(default=3.0, alias="AREA_POST_LEAVE_S")
    ignore_sat_night: bool = Field(default=True, alias="IGNORE_SAT_NIGHT")

    # ML Configuration
    ml_enabled: bool = True
    poop_model_path: str = "/data/models/poop_residue_mnv3.tflite"
    copro_model_path: str = "/data/models/copro_pair_mnv3.tflite"
    poop_thresh: float = 0.65        # p_poop minimum to confirm residue
    copro_thresh: float = 0.70       # p_eaten minimum to confirm coprophagia
    copro_area_drop_min: float = 0.20  # it is required to drop >= 20% (robustness)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def fr_public(self) -> str:
        """Public Frigate URL, falls back to base URL."""
        return self.frigate_public_url or self.frigate_base_url
    
    def get_topic(self, name: str) -> str:
        """Get MQTT topic by name."""
        topics = {
            "events": f"{self.mqtt_prefix}/events",
            "state": "home/ellie/state",
            "poop": "home/ellie/poop_present",
            "poop_event": "home/ellie/poop_event",
            "copro_risk": "home/ellie/coprophagy_risk",
            "copro_event": "home/ellie/coprophagy_event",
            "health": "home/ellie/health",
            "debug_log": "home/ellie/debug/log",
        }
        return topics.get(name, f"home/ellie/{name}")


def _maybe_apply_thresholds_from_file(s: Settings) -> None:
    """Override ML thresholds from a JSON file if present.
    Order of precedence:
      1) THRESHOLDS_JSON env var path
      2) /data/thresholds.json (add-on writable data)
      3) /config/ellie_thresholds.json (HA config share)
    Keys: poop_threshold, copro_threshold
    """
    import json
    candidates = [
        os.environ.get("THRESHOLDS_JSON", ""),
        "/data/thresholds.json",
        "/config/ellie_thresholds.json",
    ]
    for path in candidates:
        if not path:
            continue
        try:
            if os.path.isfile(path):
                with open(path, "r") as f:
                    j = json.load(f)
                if isinstance(j, dict):
                    if "poop_threshold" in j:
                        s.poop_thresh = float(j["poop_threshold"])  # type: ignore[attr-defined]
                    if "copro_threshold" in j:
                        s.copro_thresh = float(j["copro_threshold"])  # type: ignore[attr-defined]
                break
        except Exception:
            # ignore malformed files
            continue


def load_settings() -> Settings:
    """Load and return application settings."""
    s = Settings()
    _maybe_apply_thresholds_from_file(s)
    return s

