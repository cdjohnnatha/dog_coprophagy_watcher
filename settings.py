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
    proper_zone: str = Field(default="", alias="PROPER_ZONE")
    monitor_all_zones: bool = Field(default=True, alias="MONITOR_ALL_ZONES")
    
    # Residue Confirmation
    residue_confirm_window_s: float = Field(default=20.0, alias="RESIDUE_CONFIRM_WINDOW_S")
    residue_min_area: int = Field(default=140, alias="RESIDUE_MIN_AREA")
    residue_static_sec: float = Field(default=2.0, alias="RESIDUE_STATIC_SEC")
    
    # Snapshot Configuration
    snapshot_fps: float = Field(default=4.0, alias="SNAPSHOT_FPS")
    check_interval_s: float = Field(default=15.0, alias="CHECK_INTERVAL_S")
    
    # Debug
    enable_debug_watcher: bool = Field(default=False, alias="ENABLE_DEBUG_WATCHER")

    # Persistence (add-on /data is a persistent volume)
    state_store_path: str = Field(default="/data/ellie_watcher_state.json", alias="STATE_STORE_PATH")
    pad_ref_image_path: str = Field(default="/data/pad_clean_ref.jpg", alias="PAD_REF_IMAGE_PATH")
    evidence_enabled: bool = Field(default=True, alias="EVIDENCE_ENABLED")
    evidence_dir: str = Field(default="/data/evidence", alias="EVIDENCE_DIR")

    # Pad reference capture heuristics (used to detect pad replacement/reset).
    pad_ref_person_window_s: float = Field(default=180.0, alias="PAD_REF_PERSON_WINDOW_S")
    pad_ref_dog_absent_s: float = Field(default=5.0, alias="PAD_REF_DOG_ABSENT_S")

    # Pee Detection (confirmation after squat ends)
    pee_enabled: bool = Field(default=True, alias="PEE_ENABLED")
    pee_confirm_window_s: float = Field(default=25.0, alias="PEE_CONFIRM_WINDOW_S")
    pee_min_area: int = Field(default=220, alias="PEE_MIN_AREA")
    pee_static_sec: float = Field(default=1.0, alias="PEE_STATIC_SEC")
    pee_delta_l_min: int = Field(default=10, alias="PEE_DELTA_L_MIN")
    # Pee can also show up as yellowing on white pads (b* in LAB), and in some
    # lighting conditions as brightening/specular highlights.
    pee_delta_l_bright_min: int = Field(default=12, alias="PEE_DELTA_L_BRIGHT_MIN")
    pee_delta_b_min: int = Field(default=10, alias="PEE_DELTA_B_MIN")
    # Integrated wetness change threshold to detect pee-on-pee even if area grows little.
    pee_mass_min: int = Field(default=1200, alias="PEE_MASS_MIN")
    # Optional secondary gate: compare with last confirmed pee image to validate
    # an incremental increase (e.g. count 2 -> 3).
    pee_increment_enabled: bool = Field(default=True, alias="PEE_INCREMENT_ENABLED")
    pee_increment_mass_min: int = Field(default=500, alias="PEE_INCREMENT_MASS_MIN")
    last_image_max_age_s: float = Field(default=240.0, alias="LAST_IMAGE_MAX_AGE_S")
    
    # Episode Management
    ep_cooldown_s: float = Field(default=180.0, alias="EP_COOLDOWN_S")
    merge_radius_px: int = Field(default=90, alias="MERGE_RADIUS_PX")

    # Pee episode de-dup (pee can happen multiple times close together)
    pee_cooldown_s: float = Field(default=45.0, alias="PEE_COOLDOWN_S")
    pee_merge_radius_px: int = Field(default=70, alias="PEE_MERGE_RADIUS_PX")
    
    # Anti-Pee / Shape Filtering
    min_blob_w: int = Field(default=14, alias="MIN_BLOB_W")
    min_blob_h: int = Field(default=14, alias="MIN_BLOB_H")
    min_circularity: float = Field(default=0.28, alias="MIN_CIRCULARITY")
    min_solidity: float = Field(default=0.65, alias="MIN_SOLIDITY")
    hsv_s_min: int = Field(default=25, alias="HSV_S_MIN")
    sat_mono_global: int = Field(default=12, alias="SAT_MONO_GLOBAL")
    tex_min_mono: float = Field(default=25.0, alias="TEX_MIN_MONO")
    spec_max_mono: float = Field(default=0.18, alias="SPEC_MAX_MONO")
    
    # Frigate sub-label (optional; used only if you want to tag dog events as poop/pee)
    frigate_event_poop_sub_label: str = Field(default="poop", alias="FRIGATE_EVENT_POOP_SUB_LABEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def fr_public(self) -> str:
        """Public Frigate URL, falls back to base URL."""
        return self.frigate_public_url or self.frigate_base_url

    @property
    def proper_zone_name(self) -> str:
        """Zone that represents the proper elimination area (pad zone)."""
        return self.proper_zone or self.toilet_zone
    
    def get_topic(self, name: str) -> str:
        """Get MQTT topic by name."""
        topics = {
            "events": f"{self.mqtt_prefix}/events",
            "state": "home/ellie/state",
            "poop": "home/ellie/poop_present",
            "poop_event": "home/ellie/poop_event",
            "elimination_start": "home/ellie/elimination_start",
            "elimination_event": "home/ellie/elimination_event",
            "elimination_stats": "home/ellie/elimination_stats",
            "health": "home/ellie/health",
            "debug_log": "home/ellie/debug/log",
            "increment_gate": "home/ellie/debug/increment_gate",
        }
        return topics.get(name, f"home/ellie/{name}")


def load_settings() -> Settings:
    """Load and return application settings."""
    return Settings()

