# Contract: Home Assistant entities and UI

All entities are created by MQTT discovery under one device **"Ellie"** (`identifiers: ["ellie_v3"]`), with `unique_id = ellie_v3_<object_id>` and availability `ellie/status`. Names are pt-BR (visible text); entity ids are English.

## Entities published by the service

| Entity id | Kind | State / attributes |
|---|---|---|
| `sensor.ellie_pee_today` | sensor, `state_class: measurement`, unit `xixis` | Pees today (FR-011). Attr: `visits_today`, `uncertain_open` |
| `sensor.ellie_last_pee` | sensor, `device_class: timestamp` | Start time of the most recent counted pee. Attr: `visit_id`, `image` (`/media/local/...after.jpg`), `reason` |
| `sensor.ellie_pee_daily_history` | sensor | State = days with data. Attr `days`: `[{date, pees, visits, corrected}]`, last 90 days (back-dated corrections included) |
| `sensor.ellie_review_queue` | sensor | Count of visits with `needs_review`. Attr `items`: `[{visit_id, time, auto_outcome, reason, before, after}]` (max 20, newest first) |
| `sensor.ellie_pee_accuracy` | sensor, `%` | Deciding analyser accuracy over the window (default 14 days). Attr `analysers`: `{id: {labelled, agree, accuracy}}`, `window_days` |
| `sensor.ellie_last_visit` | sensor | `auto_outcome` of the latest visit. Attr: full visit JSON |
| `sensor.ellie_analysis_calls_month` | sensor, `state_class: total_increasing` | Cloud calls this month. Attr: `tokens_in`, `tokens_out`, `estimated_cost_eur` |
| `binary_sensor.ellie_at_tray` | binary_sensor, `occupancy` | Visit OPEN/CLOSING |
| `number.ellie_confidence_threshold` | number 0.50–0.99 | Tunable 1 (FR-008) |
| `number.ellie_merge_gap` | number 15–300 s | Tunable 2 |
| `select.ellie_deciding_analyser` | select | Tunable 3 (FR-010b). Options = analysers available |
| `switch.ellie_validation_mode` | switch | Tunable 4 (research R9) |
| `sensor.ellie_pad_changed_at` | sensor, `device_class: timestamp` | Current pad's change time (US7). Attr: `source`, `days_in_use`, `pees_since`, `pad_color`, `image`, `history` (last 10 changes incl. undone) |
| `sensor.ellie_pad_change_due` | sensor, `device_class: timestamp` | `changed_at + pad_change_days`. `unknown` until a first change exists |
| `number.ellie_pad_change_days` | number 1–7 d, default 3 | Tunable 6 (FR-027) |

## Defined in the HA repo (`/homeassistant`)

| Item | File | Notes |
|---|---|---|
| Review notification + action handling + clear-on-resolve | `custom/automations/ellie-pee-counter.yaml` | Sends to `notify.common_mobile_tag`, `tag: ellie_review_<id>`, image + 2 actions, `mode: queued` |
| "No pee for a long time" alert (FR-020) | same file | Uses `input_number.ellie_no_pee_alert_hours` (tunable 5, default 8) and `input_datetime.ellie_waking_start` / `_end` (07:00 / 23:00). Fires once per gap |
| Pad change reminder (FR-027) + pad notification actions + clear on `ellie/pad/changed` | same file | Triggers: `sensor.ellie_pad_change_due` (time trigger on a timestamp sensor) and `input_datetime.ellie_waking_start`. Conditions: due ≤ now, inside waking hours, not reminded today. `tag: ellie_pad_reminder` |
| Scripts `script.ellie_label_visit` (`visit_id`, `label`), `script.ellie_add_manual_pee` (`at`), `script.ellie_reject_last_pee`, `script.ellie_mark_pad_changed` (`at?`), `script.ellie_undo_pad_change` | `custom/scripts/ellie-pee-counter.yaml` | Publish to `ellie/cmd/*`. `unique_id` on each |
| Helpers above | `custom/helpers/…` (input_datetime) and `configuration.yaml` inline `input_number` (repo convention) | `unique_id`-capable definitions |

## Ellie dashboard (`/dashboard-ellie`, storage mode)

It is a UI (storage) dashboard, so `.storage` is never hand-edited (constitution). The new cards are kept as YAML in `docs/ellie-pee-counter/dashboard-cards.yaml` (HA repo) and applied through the UI raw editor or the `lovelace/config/save` WebSocket call after a backup.

Cards (all texts pt-BR):

1. **Hoje**: today's count, "último xixi há X h" with the photo, uncertain count.
2. **Revisar**: list from `sensor.ellie_review_queue` (`auto-entities`/`button-card`), each with before/after thumbnails and buttons "Fez xixi" / "Não fez" → `script.ellie_label_visit`.
3. **Adicionar / desfazer**: "Adicionar xixi" (time picker → `script.ellie_add_manual_pee`), "Desfazer último".
4. **Histórico**: `apexcharts-card` columns from `sensor.ellie_pee_daily_history` `days` via `data_generator`, 30 days by default.
5. **Precisão**: accuracy %, per-analyser table, validation mode switch, threshold.
6. **Tapete**: "Trocado há 2 dias (seg 27/10, 09:14 · detectado)", "7 xixis neste tapete", "Próxima troca: qui 30/10, 09:14" (red when overdue), buttons "Troquei agora" → `script.ellie_mark_pad_changed`, "Não troquei" → `script.ellie_undo_pad_change`, interval `number.ellie_pad_change_days`. With no change yet: "Troca ainda não registrada".

## Notification texts (pt-BR)

- Uncertain: title "Ellie: fez xixi?" · message "Visita às {HH:MM} — {reason}" · actions "Fez xixi" / "Não fez".
- Unknown: title "Ellie: não consegui analisar" · message "Visita às {HH:MM} ({motivo}). Toque para revisar."
- No pee: title "Ellie: sem xixi há {N} h" · message "Último xixi às {HH:MM}."
- Pad reminder: title "Ellie: hora de trocar o tapete" · message "Tapete em uso há {N} dias ({X} xixis)." · action "Já troquei".
- Pad uncertain: title "Ellie: você trocou o tapete?" · message "Movimento no tabuleiro às {HH:MM} — {reason}" · actions "Troquei" / "Não troquei".
- A detected pad change sends **no** notification (it only clears the reminder).
