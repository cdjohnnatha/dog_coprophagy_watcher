"""Process entry point: wiring only. Behaviour lives in service.py."""

import asyncio
import logging
import os
import signal
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ellie import __version__, frames
from ellie.analysers.gemini import GeminiAnalyser
from ellie.evidence import Evidence
from ellie.ha import COMMAND_TOPICS, FRIGATE_TOPICS, Bridge
from ellie.service import Service
from ellie.settings import fetch_mqtt_service, load_options
from ellie.store import Store
from ellie.visits import VisitTracker

log = logging.getLogger("ellie")
DB_PATH = "/data/ellie.db"
MEDIA_ROOT = Path("/media/ellie")


def utcnow() -> datetime:
    return datetime.now(UTC)


async def every(seconds: float, step) -> None:
    while True:
        try:
            await step()
        except Exception:
            log.exception("periodic task failed")
        await asyncio.sleep(seconds)


async def main() -> None:
    options = load_options()
    logging.basicConfig(
        level=options.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log.info("Ellie %s starting", __version__)

    async with httpx.AsyncClient(timeout=30) as http:
        mqtt = await fetch_mqtt_service(http, os.environ.get("SUPERVISOR_TOKEN"))
        store = Store(DB_PATH)
        analysers = {}
        if options.gemini_api_key:
            gemini = GeminiAnalyser(options.gemini_api_key, options.gemini_model)
            analysers[gemini.id] = gemini
        else:
            log.warning("gemini_api_key is not set: visits will end as 'unknown'")

        bridge = Bridge(mqtt.host, mqtt.port, mqtt.username, mqtt.password)

        async def publish(topic: str, payload: str, retain: bool) -> bool:
            return await bridge.publish(topic, payload, retain)

        service = Service(
            store=store,
            evidence=Evidence(MEDIA_ROOT),
            reference=frames.ReferenceKeeper(MEDIA_ROOT / "reference", utcnow),
            tracker=VisitTracker(),
            analysers=analysers,
            fetch_frame=lambda: frames.fetch_frame(http, options.go2rtc_url, options.camera),
            fetch_snapshot=lambda event_id: frames.fetch_snapshot(
                http, options.frigate_url, event_id
            ),
            crop=lambda jpeg: frames.crop(jpeg, options.analysis_area),
            publish=publish,
            clock=utcnow,
            camera=options.camera,
            usd_to_eur=options.usd_to_eur,
            version=__version__,
        )
        await service.recover()

        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)

        tasks = [
            asyncio.create_task(
                bridge.run(
                    [*FRIGATE_TOPICS, *COMMAND_TOPICS], service.on_connect, service.on_message
                )
            ),
            asyncio.create_task(every(1, service.tick)),
            asyncio.create_task(every(60, service.minute_tasks)),
        ]
        await stop.wait()
        log.info("stopping")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
