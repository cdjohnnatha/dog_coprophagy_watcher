"""
MQTT message handlers - Transform MQTT messages into service calls.
"""
import json
from domain.models import FrigateEventUpdate


class MQTTHandlers:
    """Handlers for MQTT callbacks."""
    
    def __init__(self, service, settings):
        """
        Initialize handlers.
        
        Args:
            service: EllieWatcherService instance
            settings: Settings instance
        """
        self.service = service
        self.settings = settings
    
    def on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        """
        Handle MQTT connection.
        
        Args:
            client: MQTT client
            userdata: User data
            flags: Connection flags
            rc: Return code
            properties: Connection properties (MQTT v5)
        """
        # Connection is handled by MQTTClient adapter
        pass
    
    def on_message(self, client, userdata, msg) -> None:
        """
        Handle incoming MQTT message.
        
        Args:
            client: MQTT client
            userdata: User data
            msg: MQTT message
        """
        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return
        
        # Extract event data
        after = data.get("after", {}) or {}
        before = data.get("before", {}) or {}
        evt_type = str(data.get("type") or "")
        label = str(after.get("label") or "")
        cam = str(after.get("camera") or "")
        event_id = str(after.get("id") or "")
        
        # Filter by camera
        if cam != self.settings.camera_name:
            return
        
        # Filter by label
        if label not in ("dog", "person"):
            return
        
        zones = (
            after.get("entered_zones")
            or after.get("current_zones")
            or after.get("zones")
            or []
        )
        if not isinstance(zones, list):
            zones = []

        # Zone filter:
        # - legacy behavior: only process detections in `toilet_zone`
        # - new behavior (recommended): monitor all zones to detect accidents,
        #   while using `proper_zone` only for compliance classification.
        if (not self.settings.monitor_all_zones) and self.settings.toilet_zone:
            if self.settings.toilet_zone not in zones:
                return

        try:
            st = float(after.get("start_time")) if after.get("start_time") is not None else None
        except Exception:
            st = None
        try:
            et = float(after.get("end_time")) if after.get("end_time") is not None else None
        except Exception:
            et = None

        has_clip = after.get("has_clip")
        if isinstance(has_clip, bool):
            hc = has_clip
        else:
            hc = None

        upd = FrigateEventUpdate(
            type=evt_type,
            camera=cam,
            label=label,
            event_id=event_id,
            zones=[str(z) for z in zones if z is not None],
            start_time=st,
            end_time=et,
            has_clip=hc,
        )

        # Extra hint: some Frigate versions set end_time but keep type=update; pass both.
        ended_now = bool(upd.end_time) and (before.get("end_time") in (None, 0, 0.0, ""))
        if ended_now and upd.type not in ("end",):
            upd.type = "end"

        if upd.label == "person":
            self.service.handle_person_event(upd)
            return

        self.service.handle_dog_event(upd)

