# Contract: Home Assistant entities and UI

All entities are created by MQTT discovery under one device **"Ellie"** (`identifiers: ["ellie_v3"]`), with `unique_id = ellie_v3_<object_id>` and availability `ellie/status`. Names are pt-BR (visible text); entity ids are English.

## Entities published by the service

| Entity id | Kind | State / attributes |
|---|---|---|
| `sensor.ellie_pee_today` | sensor, `state_class: measurement`, unit `xixis` | Pees today (FR-011). Attr: `visits_today`, `uncertain_open` |
| `sensor.ellie_last_pee` | sensor, `device_class: timestamp` | Start time of the most recent counted pee. Attr: `visit_id`, `image` (`/media/local/...after.jpg`), `reason` |
| `sensor.ellie_poop_today` | sensor, `state_class: measurement`, unit `cocôs` | Poops counted today (`final_poop = poop`), 3.5.0 |
| `sensor.ellie_last_poop` | sensor, `device_class: timestamp` | Start time of the most recent counted poop. Attr: `visit_id`, `where` (`on_pad` / `beside_tray` / null), `image`, `reason`, `cleared_at`, `cleared_by` (`watch` / `pad_change` / `owner` / `timeout` / `migration`) |
| `sensor.ellie_poop_daily_history` | sensor | State = days with data (90). Attr `days`: `[{date, poops}]`, oldest first |
| `sensor.ellie_poop_hours_since` | sensor, `state_class: measurement`, unit `h` | Hours since the last counted poop (ever, not only the 14-day window; 1 decimal; unknown when there is none). Grows every minute with no new poop. Attr: `last_at` (UTC). 3.6.0, ADR-0015 |
| `sensor.ellie_poop_typical_time` | sensor | Usual time of day of the poops (median, local `HH:MM`) over the last 14 days, or `coletando dados` until 5 poops. Attr: `typical_from` / `typical_to` (25 % and 75 %), `n`, `ready`, `periods` (`{madrugada, manha, tarde, noite}` counts for 00-06, 06-12, 12-18, 18-24), `window_days` |
| `sensor.ellie_poop_interval_hours` | sensor, unit `h` | Median gap between consecutive poops in the window (unknown below 3 poops) |
| `sensor.ellie_poop_per_day` | sensor, unit `cocôs/dia` | Average per day, counted from the day of the first poop in the window (at most 14 days) |
| `binary_sensor.ellie_poop_present` | binary_sensor (`ON`/`OFF`) | ON while a counted poop is not cleared (ADR-0014). Attr: `count`, `since`, `visit_id`, `where`, `image` (newest open poop). Clears itself: the pad watch sees the blob gone, a pad change (on-pad poops), `ellie/cmd/poop_clear`, an owner `no_poop`, or 12 h |
| `image.ellie_poop_photo` | image (MQTT) | `after.jpg` of the poop present now, else of the last counted one |
| `sensor.ellie_pee_daily_history` | sensor | State = days with data. Attr `days`: `[{date, pees, visits, corrected}]`, last 90 days (back-dated corrections included) |
| `sensor.ellie_review_queue` | sensor | Count of visits with `needs_review` or an open poop question. Attr `items`: `[{visit_id, time, auto_outcome, final_outcome, reason, pee_question, poop_question, poop_observed, before, after}]` (max 20, newest first) |
| `sensor.ellie_pee_accuracy` | sensor, `%` | Deciding analyser accuracy over the window (default 14 days). Attr `analysers`: `{id: {labelled, agree, accuracy}}`, `window_days` |
| `sensor.ellie_last_visit` | sensor | `auto_outcome` of the latest visit. Attr: full visit JSON |
| `sensor.ellie_analysis_calls_month` | sensor, `state_class: total_increasing` | Cloud calls this month. Attr: `tokens_in`, `tokens_out`, `estimated_cost_eur` |
| `binary_sensor.ellie_at_tray` | binary_sensor, `occupancy` | Visit OPEN/CLOSING |
| `number.ellie_confidence_threshold` | number 0.50–0.99 | Tunable 1 (FR-008) |
| `number.ellie_merge_gap` | number 15–300 s | Tunable 2 |
| `select.ellie_deciding_analyser` | select | Tunable 3 (FR-010b). Options = analysers available |
| `switch.ellie_validation_mode` | switch | Tunable 4 (research R9) |
| `sensor.ellie_pad_changed_at` | sensor, `device_class: timestamp` | Current pad's change time (US7). Attr: `source`, `days_in_use`, `pees_since`, `pad_color`, `image`, `history` (last 10 changes incl. undone) |
| `sensor.ellie_pees_per_pad` | sensor, `state_class: measurement` | Average pees of the last 5 finished pads (T096); `unknown` until 2 pads have finished. Attr: `pads_counted`. Per-pad `pees` also in `sensor.ellie_pad_changed_at` attribute `history` (newest first, `null` for the pad in use and for undone changes); derived from the visits, so an owner correction also corrects it. |
| `sensor.ellie_pad_change_due` | sensor, `device_class: timestamp` | `changed_at + pad_change_days`. `unknown` until a first change exists |
| `number.ellie_pad_change_days` | number 1–7 d, default 3 | Tunable 6 (FR-027) |
| `image.ellie_last_pee_photo`, `image.ellie_review_before`, `image.ellie_review_after` | image (MQTT) | Photos for the dashboard. A `/media/local/...` URL in a card needs a signed URL, so the app sends the bytes and HA serves them with its own auth |

## Defined in the HA repo (`/homeassistant`)

| Item | File | Notes |
|---|---|---|
| Review notification + action handling + clear-on-resolve | `custom/automations/ellie-pee-counter.yaml` | Sends to `notify.common_mobile_tag`, `tag: ellie_review_<id>`, image + 2 actions, `mode: queued` |
| "No pee for a long time" alert (FR-020) | same file | Uses `input_number.ellie_no_pee_alert_hours` (tunable 5, default 8) and `input_datetime.ellie_waking_start` / `_end` (07:00 / 23:00). Fires once per gap |
| Pad change reminder (FR-027) + pad notification actions + clear on `ellie/pad/changed` | same file | Triggers: `sensor.ellie_pad_change_due` (time trigger on a timestamp sensor) and `input_datetime.ellie_waking_start`. Conditions: due ≤ now, inside waking hours, not reminded today. `tag: ellie_pad_reminder` |
| Scripts `script.ellie_label_visit` (`visit_id`, `label`), `script.ellie_label_current_review` (`label`: answers the newest pending visit, which is what the Revisar card shows), `script.ellie_add_manual_pee` (`at`), `script.ellie_reject_last_pee`, `script.ellie_reanalyse_visit` (`visit_id`; run it by hand after topping up credit), `script.ellie_mark_pad_changed` (`at?`), `script.ellie_undo_pad_change` | `custom/scripts/ellie-pee-counter.yaml` | Publish to `ellie/cmd/*`. Scripts defined in YAML have no `unique_id`; the automations are named by HA from their alias (for example `automation.ellie_v3_sem_xixi_ha_muito_tempo`) |
| Helpers above | `custom/helpers/…` (input_datetime) and `configuration.yaml` inline `input_number` (repo convention) | `unique_id`-capable definitions |
| Gemini credit alerts | same automations file + `input_number.ellie_gemini_credit_eur` | `ellie/alert` -> notification (hard alert). `ellie_v3_gemini_credit_low` warns at 80 % of the credit declared in the helper (total credit ever added, euros), using `estimated_cost_eur_total` of `sensor.ellie_analysis_calls_month`. The helper has no `initial`: set it once to what was paid (about 4.5 for R$ 30) |

## Ellie dashboard (`/dashboard-ellie`, storage mode)

It is a UI (storage) dashboard, so `.storage` is never hand-edited (constitution). The new cards are kept as YAML in `docs/ellie-pee-counter/dashboard-cards.yaml` (HA repo) and applied through the UI raw editor or the `lovelace/config/save` WebSocket call after a backup.

Cards (all texts pt-BR):

1. **Hoje**: today's count, "último xixi há X h" with the photo, uncertain count.
2. **Revisar**: shows the newest visit of `sensor.ellie_review_queue` (its `image.ellie_review_before/after` photos and the reason, in a markdown card) with buttons "Fez xixi" / "Não fez" → `script.ellie_label_current_review`. Answering shows the next one.
3. **Adicionar / desfazer**: "Adicionar xixi" (time picker → `script.ellie_add_manual_pee`), "Desfazer último".
4. **Histórico**: `apexcharts-card` columns from `sensor.ellie_pee_daily_history` `days` via `data_generator`, 30 days by default.
5. **Precisão**: accuracy %, per-analyser table, validation mode switch, threshold.
6. **Tapete**: "Trocado há 2 dias (seg 27/10, 09:14 · detectado)", "7 xixis neste tapete", "Próxima troca: qui 30/10, 09:14" (red when overdue), buttons "Troquei agora" → `script.ellie_mark_pad_changed`, "Não troquei" → `script.ellie_undo_pad_change`, interval `number.ellie_pad_change_days`. With no change yet: "Troca ainda não registrada".

## Notification texts (pt-BR)

- Uncertain: title "Ellie: fez xixi?" · message "Visita às {HH:MM} — {reason}" · actions "Fez xixi" / "Não fez".
- Poop uncertain (3.5.0): title "Ellie: fez cocô?" · message "Visita às {HH:MM} — {reason}" · actions "Teve cocô" / "Não teve" (`ELLIE_POOP_YES` / `ELLIE_POOP_NO`). Never sent for a confident answer.
- Unknown: title "Ellie: não consegui analisar" · message "Visita às {HH:MM} ({motivo}). Toque para revisar."
- Gemini: "Ellie: créditos do Gemini acabaram" (hard, HTTP 402), "Ellie: chave do Gemini recusada" (401/403), "Ellie: crédito do Gemini acabando" (80 % of the declared credit).
- No pee: title "Ellie: sem xixi há {N} h" · message "Último xixi às {HH:MM}."
- Pad reminder: title "Ellie: hora de trocar o tapete" · message "Tapete em uso há {N} dias ({X} xixis)." · action "Já troquei".
- Pad uncertain: title "Ellie: você trocou o tapete?" · message "Movimento no tabuleiro às {HH:MM} — {reason}" · actions "Troquei" / "Não troquei".
- A detected pad change sends **no** notification (it only clears the reminder).
