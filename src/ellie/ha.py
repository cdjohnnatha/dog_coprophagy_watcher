"""MQTT bridge: one connection with a last will, reconnection and a message dispatcher.

Topics and payloads: docs/contracts/mqtt.md. Entities are created by discovery.py.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence

import aiomqtt

from ellie.discovery import STATUS_TOPIC

log = logging.getLogger(__name__)

FRIGATE_TOPICS = (
    "frigate/pad_zone/dog",
    "frigate/tray_area/dog",
    "frigate/tray_area/person",
    "frigate/tray_pad/dog",
    "frigate/events",
    "frigate/available",
)
COMMAND_TOPICS = ("ellie/cmd/#", "ellie/set/#")
MAX_RECONNECT_DELAY_S = 30


class Bridge:
    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self._conn = {"hostname": host, "port": port, "username": username, "password": password}
        self._client: aiomqtt.Client | None = None

    async def publish(self, topic: str, payload: str | bytes, retain: bool = False) -> bool:
        """Publish if connected. Anything lost while offline is republished on reconnect."""
        client = self._client
        if client is None:
            log.debug("not connected, dropped publish to %s", topic)
            return False
        try:
            await client.publish(topic, payload, qos=1, retain=retain)
        except aiomqtt.MqttError as exc:
            log.warning("publish to %s failed: %s", topic, exc)
            return False
        return True

    async def run(
        self,
        topics: Sequence[str],
        on_connect: Callable[[], Awaitable[None]],
        on_message: Callable[[str, bytes], Awaitable[None]],
    ) -> None:
        """Connect, subscribe and dispatch forever. A failing handler never stops the loop."""
        delay = 1
        while True:
            try:
                async with aiomqtt.Client(
                    **self._conn,
                    identifier="ellie-v3",
                    keepalive=30,
                    will=aiomqtt.Will(STATUS_TOPIC, "offline", qos=1, retain=True),
                ) as client:
                    self._client = client
                    for topic in topics:
                        await client.subscribe(topic)
                    await client.publish(STATUS_TOPIC, "online", qos=1, retain=True)
                    log.info("mqtt connected")
                    await on_connect()
                    delay = 1
                    async for message in client.messages:
                        try:
                            await on_message(str(message.topic), bytes(message.payload or b""))
                        except Exception:
                            log.exception("handler failed for %s", message.topic)
            except aiomqtt.MqttError as exc:
                log.warning("mqtt connection lost (%s), retrying in %s s", exc, delay)
            except asyncio.CancelledError:
                await self._say_offline()
                raise
            finally:
                self._client = None
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_RECONNECT_DELAY_S)

    async def _say_offline(self) -> None:
        """Graceful stop: the last will only fires on unexpected disconnects."""
        if self._client is not None:
            try:
                await self._client.publish(STATUS_TOPIC, "offline", qos=1, retain=True)
            except aiomqtt.MqttError:
                pass
