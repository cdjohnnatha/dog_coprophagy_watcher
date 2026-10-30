# Contract: MQTT topics

Broker: `core-mosquitto`. Credentials come from the Supervisor MQTT service (research R11). New root: **`ellie/`**. The old watcher's `home/ellie/*` topics are neither read nor written by v3 (FR-023).

## Consumed (from Frigate)

| Topic | Payload | Use |
|---|---|---|
| `frigate/pad_zone/dog` | int (count) | With `tray_area/dog`: she is at the tray when either is `>0`; the visit closes when both are 0 (ADR-0011) |
| `frigate/tray_area/dog`, `frigate/tray_area/person` | int | `person_present` flag and "area clear" for before/after frames. `person > 0` with no dog visit open starts a person presence (US7, research R14) |
| `frigate/tray_pad/dog` | int | Seconds on the pad itself (`on_pad_s`). Zone tight around the tray, added 2026-10-29 |
| `frigate/events` | Frigate event JSON | `after.camera == "ellie"`, `after.label == "dog"`, `"pad_zone" in after.entered_zones` → append `after.id` to the open visit |
| `frigate/available` | `online`/`offline` | Offline during a visit → visit becomes `unknown` |

## Published by the service

| Topic | Retained | Payload |
|---|---|---|
| `ellie/status` | yes | `online` / `offline` (MQTT last will). Availability topic for all entities |
| `ellie/state` | yes | JSON state for all sensors (one message, entities pick fields via `value_template`) — see [ha-entities.md](ha-entities.md) |
| `ellie/visit/<visit_id>` | no | The stored visit row as JSON after each decision (no file paths: those are derived from the id) |
| `ellie/review/request` | no | `{visit_id, kind: "uncertain"|"unknown"|"pad_uncertain", title, message, image: "/media/local/ellie/…/after.jpg", reason}`. For `pad_uncertain`, `visit_id` holds the presence id (`s_…`). Titles and messages are pt-BR |
| `ellie/review/resolved` | no | `{visit_id}`. HA clears the notification tag on all devices |
| `ellie/pad/changed` | no | `{pad_change_id, at, source}` after any recorded pad change (or undo, with `undone: true`). HA clears the `ellie_pad_reminder` notification on all devices |
| `ellie/alert` | no | `{kind: "credit_exhausted"|"gemini_key", title, message}`. Problems only the owner can fix (Gemini credit ran out on HTTP 402, key refused on 401/403). The app sends at most one per kind every 6 h; HA turns it into a phone + persistent notification |
| `ellie/image/last_pee`, `ellie/image/review_before`, `ellie/image/review_after` | yes | Raw JPEG bytes for the three `image` entities, sent only when the photo changes (last counted pee; before/after of the newest visit in the review queue) |
| `homeassistant/<component>/ellie/<object_id>/config` | yes | MQTT discovery configs |

## Commands (HA → service)

All payloads are JSON. Invalid payloads are logged and ignored (never crash).

| Topic | Payload | Effect |
|---|---|---|
| `ellie/cmd/label` | `{visit_id, label: "pee"|"no_pee", channel, by?}` | Adds a Label (data-model). Publishes `ellie/review/resolved` |
| `ellie/cmd/manual_pee` | `{at: ISO-8601 local, by?}` | Creates a `source=manual` visit with `final_outcome=pee` |
| `ellie/cmd/reject_last` | `{by?}` | Labels the most recent counted pee (today) as `no_pee` |
| `ellie/cmd/reanalyse` | `{visit_id}` | Re-runs the deciding analyser on stored frames (for example after a prompt change) |
| `ellie/cmd/pad_label` | `{presence_id, label: "replaced"|"not_replaced", by?}` | Resolves an uncertain presence. `replaced` creates a Pad change (`source=confirmed`). Publishes `ellie/review/resolved` |
| `ellie/cmd/pad_changed` | `{at?: ISO-8601 local, by?}` | Pad change `source=manual` (default `at` = now) |
| `ellie/cmd/pad_undo` | `{by?}` | Sets `undone_at` on the current pad change; the previous one becomes current |
| `ellie/set/<setting>` | value | `confidence_threshold` (0.5–0.99), `merge_gap_s` (15–300), `deciding_analyser` (id), `validation_mode` (`ON`/`OFF`), `pad_change_days` (1–7) |

## Notification action ids (HA side)

`ELLIE_PEE_YES::<visit_id>` and `ELLIE_PEE_NO::<visit_id>`. The HA automation maps them to `ellie/cmd/label` with `channel: "notification"`. Button titles: "Fez xixi" / "Não fez".

`ELLIE_PAD_YES::<presence_id>` / `ELLIE_PAD_NO::<presence_id>` → `ellie/cmd/pad_label` ("Troquei" / "Não troquei"). `ELLIE_PAD_DONE` (reminder button "Já troquei") → `ellie/cmd/pad_changed`.
