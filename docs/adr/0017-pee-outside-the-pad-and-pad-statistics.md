---
status: accepted (2026-11-09)
date: 2026-11-09
decision-makers: Claudio
---

# A pee outside the pad means the pad is full; pad statistics per month

## Context and Problem Statement

On 2026-11-09 at 12:53 Ellie peed on the floor just below the tray, on the 6th pee of a pad that had been in use for 1.2 days. The owner says she does that when the pad is full (rarely by mistake). Gemini saw it ("vazando no chão logo abaixo da bandeja"), but the answer had no field for where the pee was, so nothing could use it. The owner also wants to see the pad pattern: how many pees a pad takes, how long it lasts, and how many pads a month he uses (a pack of 30 should last about 2 months). He does not want a new reminder.

## Decision Drivers

* Only automatic metrics (no manual tracking).
* No extra Gemini calls.
* No new reminder: show the pattern, and say "pad full" only when it happens.

## Considered Options

* Ask Gemini where the pee is, in the same call (chosen).
* Detect a puddle with the pad watch: a clear, shiny puddle does not darken the tiles, so the floor rule misses it.
* A pee-count reminder ("5 pees or 3 days"): rejected by the owner.

## Decision Outcome

1. **`pee_where`** (`none`, `on_pad`, `off_pad`, `both`) in the analysis answer. It is optional, so older prompts leave it out. Prompt `pee_v4` is `pee_v3` plus rule 5b. Migration 7 adds `visits.pee_where`. The deciding analyser is switched to `gemini:pee_v4` with the existing `select.ellie_deciding_analyser`.
2. **Pad full.** A counted pee with `pee_where` `off_pad` or `both` publishes `ellie/pad/full` once per visit. HA notifies "Ellie: tapete cheio" with the photo and a "Já troquei" button. At night it waits for the waking time, and is dropped if the pad was changed in the meantime. The next pad change clears it.
3. **Pad statistics**, derived from the visits and pad changes (nothing new is stored): per pad `days`, `pees_per_day`, `overflow_pee` (number of the first pee outside the pad), plus `avg_days_per_pad` and `pads_per_month` (last 5 finished pads, from 2 on), `pads_last_30d` and `overflow_pees`. New sensor `sensor.ellie_pads_last_30d`. The dashboard pad card opens a "Tapetes: histórico" popup with these numbers, how long a pack of 30 lasts, and a table of the last pads.

### Consequences

* Good, because the owner gets the "pad full" notice right when it happens, and over time `overflow_pees` shows after how many pees she starts to miss.
* Bad, because only visits judged by `pee_v4` carry `pee_where`. Older pads show no overflow (the 12:53 one only if reanalysed with v4).
* Bad, because an owner label "pee" on a visit Gemini said `no_pee` has no `pee_where`, so that pee never counts as an overflow.

### Confirmation

Tests: `tests/replay/test_replay_pad.py::test_a_pee_outside_the_pad_says_the_pad_is_full_and_pads_get_stats`, `tests/unit/test_store.py::test_migration_7_adds_pee_where_and_checks_it`, `tests/contract/test_discovery.py`. Live: log line "pee outside the pad, the pad is full", and the popup numbers against the owner's own count.

## More Information

Spec 003 tasks T121-T125. Same release (3.8.0): ADR-0016 amendment (a lasting-change confirm needs 30 quiet minutes; occluded presence frames are taken again).
