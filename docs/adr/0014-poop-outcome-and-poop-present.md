---
status: accepted (2026-11-05)
date: 2026-11-05
---

# The poop outcome, its confirmation and a "poop present" state that clears itself

## Context and problem

Scorecard 03/11-05/11 (`validation.md`): the app saw 0 of 4 poops with `pee_v1`. Prompt `pee_v3` (ADR-0012 amendment) found 8 of 8 real poops on the labelled visits with 0 false ones, and the floor watch opens visits for poops Frigate never saw. What was missing: a counted poop (daily counter, history), a way for the owner to confirm or deny it, and a state that says "there is poop in the area now" and goes away by itself once it is cleaned up.

## Decision

1. **`final_poop` next to `final_outcome`.** `auto_poop` = `decide_poop(result, threshold)`: `on_pad` or `beside_tray` is `poop`, `none` is `no_poop`, `unsure` is `uncertain`; same inclusive confidence rule as the pee, with `poop_confidence`; a bad image is `uncertain`; a swapped pad makes only an `on_pad` poop `uncertain`. `final_poop` is `poop` / `no_poop` when decided, `NULL` while the owner is asked, `unknown` when the analysis failed or the question went unanswered for the day (like `final_outcome`). Migration 6 adds `auto_poop`, `final_poop`, `poop_cleared_at`, `poop_cleared_by` and backfills old automatic visits from `poop_observed` (those poops are marked cleared by `migration`, so history counts them but nothing is "present").
2. **Two questions, two answers.** `labels` gets a `kind` (`pee` / `poop`; old rows are `pee`). `ellie/cmd/label` takes `kind` (default `pee`). A poop answer never changes `final_outcome`, `state` or `needs_review`, and a pee answer never changes `final_poop` (tests both ways). The owner's answer wins over a later analysis, per kind. Accuracy and the dataset still use the pee labels only.
3. **Asking.** Buttons only on the review screen and in one notification (`kind: "poop"`, title "Ellie: fez cocô?", action ids `ELLIE_POOP_YES::<id>` / `ELLIE_POOP_NO::<id>`, texts "Teve cocô" / "Não teve") and only when the poop is `uncertain`, once per visit. A confident answer sends nothing. The review list shows a visit while a pee **or** a poop question is open (`pee_question`, `poop_question`).
4. **Counters.** `poop_today`, `poops_by_day` (90 days) and `last_poop` are derived from `final_poop = poop` like the pee ones, so a correction lands on the right day.
5. **`poop_present`.** ON while at least one counted poop is not cleared. It goes OFF by itself when: (a) the pad watch sees the blob gone: at the end of the visit the watch finds the 2x2 block that darkened most between `before.jpg` and `after.jpg` (pad included) and follows it; two clear frames in a row, with no light jump, whose darkening of that block (or its neighbours) against the "before" frame is under 35 % of the original peak, clear it (`poop_cleared_by = watch`); (b) a pad change is recorded and the poop was `on_pad` (`pad_change`); (c) the owner sends `ellie/cmd/poop_clear` or answers `no_poop` (`owner`); (d) 12 h pass (`timeout`). A poop too faint to follow (peak under 0.6) can only end by (b), (c) or (d). After a restart `recover()` follows the open poops of the last 2 days again from the stored photos.
6. **No double counting.** While a poop is tracked, a floor candidate whose peak block lies near the tracked block does not open another visit.

## Consequences

- One more analysis question per visit costs nothing: it is the same Gemini call.
- Clearing by the watch depends on a quiet floor: sun patches or a dog lying on the spot keep it "present" until the timeout. Thresholds (0.6, 35 %, 2 frames) come from two real poops; check `watch.csv` and the log lines `watch: the poop of ... is gone`.
- A poop that is on the pad at night is usually not followed (low contrast): it ends by the owner, the pad change or the timeout.
- `poop_cleared_by = migration` rows are never "present".
