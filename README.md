# Ellie (v3)

Home Assistant app that **counts Ellie's pees per day**. It replaces the old pixel-heuristic watcher (v2, archived on branch `legacy/v2`, tag `v2.4.3-final`).

## What it does

- Frigate already tells us when Ellie is in the tray zone (`pad_zone`). The service turns that into a **visit** (entries within 60 s are one visit).
- For each visit it keeps a **before** crop and takes an **after** crop of the analysis area (the tray plus a margin), plus a frame of her during the visit. Gemini compares them and answers with a small JSON (`docs/contracts/analysis-result.schema.json`). A pure rule turns that into `pee`, `no_pee`, `uncertain` or `unknown`.
- Uncertain visits go to the owner's phone ("Fez xixi" / "Não fez"). Every answer is a label for a later local classifier.
- It also notices when the pad was **changed** (a person at the tray) and drives the "change the pad every N days" reminder.
- Poop is recorded silently for now.

Design and decisions: `docs/adr/` and the spec in the Home Assistant config repo (`specs/003-ellie-pee-counter/`, [spec](https://github.com/cdjohnnatha/home-assistant-jetsons/blob/003-ellie-pee-counter/specs/003-ellie-pee-counter/spec.md)).

## Install (Home Assistant OS)

1. Put this repository at `/local_apps/ellie` on the Home Assistant host.
2. Settings → Apps → App store → ⋮ → **Check for updates**, then install **Ellie** from the local apps.
3. Set the options below and start it. The device **Ellie** appears in Home Assistant through MQTT discovery. See `docs/contracts/ha-entities.md`.

## Options

| Option | Meaning |
|---|---|
| `gemini_api_key` | Google AI Studio key (paid tier). Stored by the Supervisor, never in git |
| `gemini_model` | Pinned model id, for example `gemini-3.5-flash-lite` |
| `area_x1`, `area_y1`, `area_x2`, `area_y2` | Analysis area, normalised 0–1 (pad zone + 20 % margin). Must match the Frigate zone `tray_area` |
| `frigate_url`, `go2rtc_url`, `camera` | Frigate API, go2rtc frame API and camera name |
| `log_level` | `debug`, `info`, `warning` or `error` |

MQTT credentials are not options: they come from the Supervisor MQTT service.

Tunables changed at runtime from Home Assistant (confidence threshold, merge gap, deciding analyser, validation mode, pad change interval) are entities of the device, not options.

## Where things are

- `/data/ellie.db`: SQLite (visits, analyses, labels, pad changes, settings). Included in app backups.
- `/media/ellie/`: evidence images, poop clips, dataset and (later) models. Budget 5 GB, 90 days.
- Topics and payloads: `docs/contracts/mqtt.md`.

## Develop and test

```sh
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check .
```

The pure modules (`visits.py`, `decide.py`, `counting.py`, `pads.py`) have no I/O and take the clock as an argument, so tests run without Home Assistant, MQTT or the network.

## Layout

```text
src/ellie/       service code (see docs/adr/0002)
tests/           unit, contract and replay tests
docs/adr/        architecture decision records (MADR 4)
docs/contracts/  MQTT topics, Home Assistant entities, JSON schemas
training/        Phase 2 only (Mac training), not used yet
```
