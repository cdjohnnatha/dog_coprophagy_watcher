"""
MQTT client adapter for publishing and subscribing.
"""
import json
import paho.mqtt.client as mqtt
from typing import Callable, Optional, Dict, Any
from ..settings import Settings


class MQTTClient:
    """Wrapper for MQTT operations."""
    
    def __init__(self, settings: Settings, debug_enabled: bool = False):
        self.settings = settings
        self.debug_enabled = debug_enabled
        
        # Create MQTT client
        try:
            self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:
            # Fallback for older paho-mqtt versions
            self.client = mqtt.Client()
        
        # Set credentials if provided
        if settings.mqtt_user:
            self.client.username_pw_set(settings.mqtt_user, settings.mqtt_pass)
        
        self._on_connect_callback: Optional[Callable] = None
        self._on_message_callback: Optional[Callable] = None
        
        # Set internal callbacks
        self.client.on_connect = self._handle_connect
        self.client.on_message = self._handle_message
    
    def set_on_connect(self, callback: Callable) -> None:
        """Set callback for connection events."""
        self._on_connect_callback = callback
    
    def set_on_message(self, callback: Callable) -> None:
        """Set callback for incoming messages."""
        self._on_message_callback = callback
    
    def _handle_connect(self, client, userdata, flags, rc, properties=None) -> None:
        """Internal connection handler."""
        self.log(f"🔌 MQTT connected, rc={rc}")
        
        # Subscribe to events topic
        events_topic = self.settings.get_topic("events")
        client.subscribe(events_topic, qos=0)
        
        # Publish health status
        self.publish_health()
        
        # Call user callback if set
        if self._on_connect_callback:
            self._on_connect_callback(client, userdata, flags, rc, properties)
    
    def _handle_message(self, client, userdata, msg) -> None:
        """Internal message handler."""
        if self._on_message_callback:
            self._on_message_callback(client, userdata, msg)
    
    def connect(self) -> None:
        """Connect to MQTT broker."""
        self.client.connect(
            self.settings.mqtt_host,
            self.settings.mqtt_port,
            keepalive=60
        )
    
    def start(self) -> None:
        """Start MQTT loop in background."""
        self.client.loop_start()
    
    def stop(self) -> None:
        """Stop MQTT loop."""
        self.client.loop_stop()
    
    def publish(self, topic: str, payload: Any, retain: bool = False) -> None:
        """
        Publish message to topic.
        
        Args:
            topic: MQTT topic
            payload: Message payload (dict/list will be JSON-encoded)
            retain: Whether to retain message
        """
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, separators=(",", ":"))
        
        self.client.publish(topic, payload, qos=0, retain=retain)
    
    def publish_state(self, state: str) -> None:
        """Publish Ellie's state."""
        topic = self.settings.get_topic("state")
        self.publish(topic, state)
    
    def publish_poop_present(self, data: Dict[str, Any]) -> None:
        """Publish poop presence status."""
        topic = self.settings.get_topic("poop")
        self.publish(topic, data, retain=True)
    
    def publish_poop_event(self, data: Dict[str, Any]) -> None:
        """Publish poop detection event."""
        topic = self.settings.get_topic("poop_event")
        self.publish(topic, data)
    
    def publish_coprophagy_risk(self, data: Dict[str, Any]) -> None:
        """Publish coprophagy risk alert."""
        topic = self.settings.get_topic("copro_risk")
        self.publish(topic, data)
    
    def publish_coprophagy_event(self, data: Dict[str, Any]) -> None:
        """Publish coprophagy confirmation event."""
        topic = self.settings.get_topic("copro_event")
        self.publish(topic, data)
    
    def publish_health(self) -> None:
        """Publish health status."""
        from .clock import SystemClock
        clock = SystemClock()
        
        topic = self.settings.get_topic("health")
        payload = {
            "ok": True,
            "ts": clock.now_iso(),
            "camera": self.settings.camera_name,
            "zone": self.settings.toilet_zone,
        }
        self.publish(topic, payload)
    
    def log(self, message: str) -> None:
        """
        Log debug message.
        
        Args:
            message: Log message
        """
        if not self.debug_enabled:
            return
        
        print(message, flush=True)
        
        try:
            topic = self.settings.get_topic("debug_log")
            self.publish(topic, message)
        except Exception:
            pass

