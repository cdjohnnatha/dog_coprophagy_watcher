"""
Frigate API client for interacting with Frigate NVR.
"""
import requests
import numpy as np
from typing import Optional, List, Dict, Any
from ..settings import Settings
from .cv_ops import decode_image


class FrigateClient:
    """Client for Frigate API operations."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.frigate_base_url
        self.camera = settings.camera_name
        self.zone = settings.toilet_zone
    
    def fetch_snapshot(self) -> Optional[np.ndarray]:
        """
        Fetch latest snapshot from camera.
        
        Returns:
            Decoded image or None on error
        """
        try:
            url = f"{self.base_url}/api/{self.camera}/latest.jpg"
            r = requests.get(url, timeout=2.0)
            r.raise_for_status()
            return decode_image(r.content)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch snapshot: {e}")
    
    def create_event(
        self,
        camera: str,
        label: str,
        sub_label: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a manual event in Frigate.
        
        Note: Recent Frigate versions don't support POST /api/events,
        so this returns None.
        
        Args:
            camera: Camera name
            label: Event label
            sub_label: Optional sub-label
        
        Returns:
            Event ID or None
        """
        # Frigate API doesn't support creating events via POST
        return None
    
    def update_event_sub_label(self, event_id: str, sub_label: str) -> bool:
        """
        Update the sub_label of an existing event.
        
        Args:
            event_id: Event ID
            sub_label: New sub-label
        
        Returns:
            True if successful
        """
        try:
            url = f"{self.base_url}/api/events/{event_id}"
            r = requests.patch(url, json={"sub_label": sub_label}, timeout=3.0)
            r.raise_for_status()
            return True
        except Exception as e:
            print(f"[frigate] Error updating sub_label ({sub_label}) for {event_id}: {e}")
            return False
    
    def get_event_urls(self, event_id: str) -> Dict[str, str]:
        """
        Get URLs for event clip and thumbnail.
        
        Args:
            event_id: Event ID
        
        Returns:
            Dictionary with clip_url, thumb_url, snapshot_url, ha_clip, ha_thumb
        """
        base_api = f"{self.base_url}/api/events/{event_id}"
        return {
            "clip_url": f"{base_api}/clip.mp4",
            "thumb_url": f"{base_api}/thumbnail.jpg",
            "snapshot_url": f"{base_api}/snapshot.jpg",
            "ha_clip": f"/api/frigate/notifications/{event_id}/clip.mp4",
            "ha_thumb": f"/api/frigate/notifications/{event_id}/thumbnail.jpg",
        }
    
    def find_nearby_dog_event(
        self,
        timestamp: float,
        before_window: float = 45.0,
        after_window: float = 120.0,
    ) -> Optional[str]:
        """
        Find a dog event near the given timestamp.
        
        Args:
            timestamp: Reference timestamp
            before_window: Seconds before timestamp to search
            after_window: Seconds after timestamp to search
        
        Returns:
            Event ID or None
        """
        try:
            after = int(timestamp - before_window)
            before = int(timestamp + after_window)
            
            url = f"{self.base_url}/api/events"
            params = {
                "camera": self.camera,
                "label": "dog",
                "has_clip": 1,
                "after": after,
                "before": before,
                "limit": 15,
            }
            
            if self.zone:
                params["zone"] = self.zone
            
            r = requests.get(url, params=params, timeout=3.0)
            r.raise_for_status()
            
            items = r.json() if isinstance(r.json(), list) else []
            if not items:
                return None
            
            # Sort by proximity to timestamp
            def score(ev: Dict[str, Any]) -> float:
                st = float(ev.get("start_time", 0))
                et = float(ev.get("end_time", st))
                ref = et if et > 0 else st
                return abs(timestamp - ref)
            
            items.sort(key=score)
            return items[0].get("id")
        
        except Exception:
            return None
    
    def check_recent_dog_presence(self, seconds: float = 15.0) -> bool:
        """
        Check if dog was present in recent events.
        
        Args:
            seconds: How many seconds back to check
        
        Returns:
            True if dog was detected recently
        """
        try:
            import time
            after = int(time.time() - seconds)
            
            url = f"{self.base_url}/api/events"
            params = {
                "camera": self.camera,
                "label": "dog",
                "after": after,
                "limit": 5,
            }
            
            r = requests.get(url, params=params, timeout=2.5)
            r.raise_for_status()
            
            items = r.json() if isinstance(r.json(), list) else []
            return len(items) > 0
        
        except Exception:
            return False

