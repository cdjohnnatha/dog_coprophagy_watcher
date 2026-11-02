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
