---
status: proposed
date: 2026-11-02
decision-makers: Claudio (approved "Vamos às correções que você sugeriu" after the 30/10-02/11 scorecard)
---

# Prompt `pee_v2` and a log-only pad watch for pees Frigate never saw

(Numbered 0012 here; T088 had reserved 0012 for the poop outcome, which now takes the next free number.)

## Context and Problem Statement

The scorecard of 30/10-02/11 (`validation.md`) found two kinds of pee errors after v3.3:

1. **No visit at all** (31/10 08:59, 01/11 18:05). Frigate had no event of any label and kept no recording, so the app never got a dog count. The miss is upstream of the app.
2. **Wrong decision** (02/11 00:31). Frigate saw her; Gemini said `no_pee`. In the infrared `before.jpg` the pad was already heavily stained and the new pee fell on an old stain (darker and larger patch).

## Decision

1. **`pee_v2` prompt**, selectable as analyser `gemini:pee_v2` next to `gemini:pee_v1`. It adds: a pee often lands on an old stain, so compare the same spot and count growth or darkening; at night the difference is subtle; the DURING posture supports a subtle change. `ellie/cmd/reanalyse` accepts an optional `analyser` to rerun a visit with either version as a shadow, without changing the decision. Switching the deciding analyser stays a manual choice in `select.ellie_deciding_analyser` after the comparison.
2. **Pad watch (`watch.py`), log only.** Each minute, with the clear crop the service already fetches, compare the normalised grey image with the previous one. A candidate is a step in the `tray_pad` region that is at least 2.5x the step in the floor around it (so light, sun and the infrared switch do not count), and it is **confirmed** only if it stays the next minute. It writes `/media/ellie/watch/watch.csv` and a log line; it opens no visit and calls no Gemini. Pixel difference is only a possible trigger, never the decider, so ADR-0005 is not reopened.

## Consequences

- Cost on the Pi is one small image decode per minute from a frame that was fetched anyway. No Gemini cost in this step.
- The thresholds (`STEP_MIN` 0.20, `STEP_RATIO` 2.5, `LIGHT_JUMP` 12) are first guesses. After 2-3 days compare `watch.csv` with the owner's Tapo times; only then decide to let a confirmed candidate open a visit (with a daily cap, about 6, worst case about 0.008 EUR per day).
- `pee_v2` is not validated yet: first rerun the 02/11 00:31 visit and the labelled visits of the week with both versions and compare in `sensor.ellie_pee_accuracy.analysers`.
- Frigate side (not done): continuous recording for 1-2 days and a lower `dog` threshold are separate, owner-approved changes.

## Result of the prompt comparison (2026-11-02, `reanalyse` on stored frames, 16 labelled visits)

Third variant added during the test: `gemini:pee_v2+medium` (same prompt as `pee_v2`, `thinking_level` medium instead of minimal). Registered next to the others; `pee_v1` still decides.

| Case | v1 (live) | v2 | v2+medium |
|---|---|---|---|
| 02/11 00:31, pee on an old stain (owner: pee) | no_pee 0.90 ✘ | no_pee 0.90 ✘ | **pee 0.85 ✔** |
| 02/11 02:43, poop only (owner: no pee) | no_pee | pee 0.90 ✘ | no_pee 0.80 ✔ |
| 01/11 21:38, ten minutes after a pee (owner: no pee) | no_pee | no_pee | pee 0.90 ✘ |
| 29/10 20:18, pad replaced (owner: no pee) | uncertain | not rerun | uncertain |
| the other 11 visits | agree | agree | agree |

- Prompt text alone (`pee_v2`) did **not** fix 00:31: the model needs more thinking to compare overlapping stains.
- `pee_v2+medium` fixes 00:31 but adds one false pee (21:38), probably the stain of the previous visit seen again. Totals: `pee_v2` 13/15 agree, `pee_v2+medium` 14/16 (counting the 00:31 case), `pee_v1` 13/15 (accuracy sensor). The sample is far too small to call a winner.
- Cost: 46 calls this month cost 0.0527 EUR (about 0.001 EUR per call); `medium` does not show a visible increase in the counters.
- **Decision:** keep `gemini:pee_v1` as the decider. Keep `pee_v2` and `pee_v2+medium` selectable and compare them on new labelled visits (both run in shadow through `reanalyse`). Revisit after about 30 labelled visits, or sooner if the owner prefers fewer missed pees over an occasional false one.

## Amendment 2026-11-03: the pad watch now opens visits (owner approved after the 14:46 catch)

First real catch: 2026-11-03 14:46, a pee on the new pad with no Frigate event (motion recording of 14:45:46-14:47:03 had 0 objects; `watch.csv` row 14:46:05 `pad 0.422`, `floor 0.071`, baseline about 0.03). It was not confirmed because the next frame failed (go2rtc HTTP 500) and the frame after showed someone at the tray.

Changes (app 3.4.1, written, not yet deployed):

- **Confirmation** compares a later crop with the **candidate frame** (pad region still as in the candidate, no light jump), within 5 minutes, instead of with the previous minute. A lost frame or someone at the tray no longer breaks it.
- **A confirmed candidate opens a visit** (`source = auto`, reason "mudança no tapete sem detecção do Frigate") with `before.jpg` = the crop before the step and `after.jpg` = the candidate crop, and Gemini decides it like any visit. With validation mode on it also lands in the review list.
- **Dedupe:** no visit if a visit or presence overlaps the window (3 min around the candidate) or the tray is active; `rebase()` makes the after frame of every visit and presence the new baseline, so counted visits do not trigger the watch.
- **Cap:** at most 6 watch visits per local day (`WATCH_DAILY_CAP`), worst case about 0.008 EUR per day. The cap resets on restart.
- **go2rtc frame fetch retries once** after an error or HTTP 500 (the endpoint answers 500 or takes 2-8 s now and then).

Open: the thresholds stay as they were; watch `watch.csv` for false candidates (light, owner cleaning) and tune `STEP_MIN`, `STEP_RATIO`, `LIGHT_JUMP` if needed.

### Calibration of the pad watch with real frames (2026-11-03, tests run)

The first thresholds came from one example (the 14:46 catch, `pad 0.422`, global brightness normalisation). Running the watch on the 19 labelled visits (`before.jpg` -> `after.jpg`) showed that a pee on an already stained pad steps only 0.12-0.21 at the pad (floor 0.03-0.07), so the old limit (0.20) would have missed most pees. Two changes, both checked against the same real frames:

- **Brightness is normalised on the floor around the pad**, not on the whole frame: a big pee or a new pad changed the global mean and made the floor look different (ratio dropped under the limit in a synthetic test).
- **Thresholds:** `STEP_MIN` 0.10 (was 0.20), `STEP_RATIO` 2.0 (was 2.5), new `CTRL_MAX` 0.10 (more than this at the floor is light), `CONFIRM_MAX` 0.06.

Result on the real data (in-sample, 19 visits plus the 14:46 frames from the Frigate recording): 7 of 9 pee visits and the 14:46 case become candidates; the 2 missed are the 06:48 dawn light change and a borderline 2.0 ratio; 0 of the 10 "no pee" visits do (the 21:36 one is dropped by `CTRL_MAX`). It is a small sample and the thresholds were tuned on it, so keep reading `watch.csv` and the log for false candidates; the daily cap (6) and Gemini's decision bound the cost of a wrong one.

Tests: the whole suite passes (383), `ruff check` and `ruff format` are clean. New: `tests/unit/test_watch.py`, `tests/replay/test_replay_watch.py`, two tests in `tests/unit/test_frames.py`.

## Amendment 2026-11-05: prompt `pee_v3` (poop) and a floor watch for poops

Scorecard 03/11-05/11: `pee_v1` saw 0 of 4 poops (two had no visit, one came 2 minutes after the visit closed, one lay in the photo and Gemini answered `none`).

1. **`pee_v3`** = `pee_v1` with a poop rule that scans the whole floor around the tray and describes a small dark lump. Tested on the 28 labelled visits: 8 poops found, all real (checked on the photos), 0 false, pee decisions unchanged. It became the deciding analyser on 2026-11-05 (`select.ellie_deciding_analyser`); `pee_v1` stays selectable.
2. **Floor watch** (`watch.py`, app 3.4.2): next to the pad step, a new dark blob outside the pad box (peak darkening over a 2x2 block of the 48x48 grid, only darkening counts so cleaning it up is ignored) becomes a `floor` candidate, confirmed like a pad one by a later frame that still shows the blob. Thresholds from the stored pairs: real poops 1.19 and 1.67 with a quiet floor; the worst light event 1.15 with a noisy floor (`FLOOR_PEAK_MIN 1.0`, `FLOOR_CTRL_MAX 0.06`, `FLOOR_RATIO 20`). Only two positives, so check `watch.csv` (new columns `floor_peak,kind`; an old file is renamed `watch.old.csv`).
3. **Poop right after a visit**: a floor candidate is explained only by a visit or presence that overlaps its own two frames, not by the 3-minute window used for the pad (the 19:48 poop came 2 minutes after the visit of 19:45 closed). This replaces a separate "second photo" step: the watch already compares each minute with the visit's after frame.
4. Known limits: poop under strong sun or shadows (01/11 09:05 and 09:37 would not trigger, they had visits anyway); poop on the pad at night (02/11 02:43) is the visit's business; a dog lying on the floor makes a candidate that Gemini will call `none` (cost only, capped at 6 visits per day with the pad ones).
