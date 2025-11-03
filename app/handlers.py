"""
MQTT message handlers - Transform MQTT messages into service calls.
"""
import json
from typing import Optional
from ..domain.models import DogDetection, BBox


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
        label = after.get("label")
        cam = after.get("camera")
        
        # Filter by camera
        if cam != self.settings.camera_name:
            return
        
        # Filter by label
        if label not in ("dog", "person"):
            return
        
        # Filter by zone
        zones = (
            after.get("entered_zones")
            or after.get("current_zones")
            or after.get("zones")
            or []
        )
        
        if self.settings.toilet_zone and self.settings.toilet_zone not in zones:
            return
        
        # Handle person detection
        if label == "person":
            import time
            self.service.handle_person_detection(time.time())
            return
        
        # Handle dog detection
        bbox = self._extract_bbox(after)
        if not bbox:
            return
        
        detection = self._build_dog_detection(after, bbox)
        if detection:
            self.service.handle_dog_detection(detection)
    
    def _extract_bbox(self, after: dict) -> Optional[BBox]:
        """
        Extract bounding box from event data.
        
        Args:
            after: Event 'after' data
        
        Returns:
            BBox tuple or None
        """
        # Try box format [x, y, w, h]
        if "box" in after and isinstance(after["box"], list) and len(after["box"]) == 4:
            x, y, w, h = after["box"]
            return (int(x), int(y), int(x + w), int(y + h))
        
        # Try coordinate format
        if all(k in after for k in ("top", "left", "bottom", "right")):
            return (
                int(after["left"]),
                int(after["top"]),
                int(after["right"]),
                int(after["bottom"])
            )
        
        return None
    
    def _build_dog_detection(self, after: dict, bbox: BBox) -> Optional[DogDetection]:
        """
        Build DogDetection from event data.
        
        Args:
            after: Event 'after' data
            bbox: Bounding box
        
        Returns:
            DogDetection or None
        """
        import time
        
        zones = (
            after.get("entered_zones")
            or after.get("current_zones")
            or after.get("zones")
            or []
        )
        
        return DogDetection(
            bbox=bbox,
            ratio=float(after.get("ratio", 1.0)),
            stationary=bool(after.get("stationary", False)),
            speed=float(after.get("current_estimated_speed", 0.0)),
            motionless_count=int(after.get("motionless_count", 0)),
            timestamp=time.time(),
            zones=zones,
        )

