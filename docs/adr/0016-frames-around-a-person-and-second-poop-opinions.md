---
status: accepted (2026-11-09)
date: 2026-11-09
decision-makers: Claudio
---

# Own frames around a visit, a person who cleans up, and second opinions on "no poop"

## Context and Problem Statement

The 06/11-09/11 scorecard (`validation.md`) caught 3 of 4 poops. The miss, 09/11 08:22, was a timing gap: the owner cleaned at about 08:24:30 and the after frame came at 08:25:25 (area clear + 30 s), so before and after were both clean. Frigate had also dropped the dog of that visit (its score never reached the 0.6 threshold while she squatted), so the during snapshot was a 404, which had happened in 9 of 38 visits. Earlier misses had the same roots: a poop Gemini did not see although it was in the photo (05/11 06:13), poops with no visit at all (03/11 10:00, 04/11 07:28), and a pee whose watch candidate frame still had her in it (07/11 08:52).

## Decision Drivers

* Only automatic metrics (no manual tracking). The owner answers yes/no questions at most.
* Reuse what runs already (the pad watch maths, the pad-change prompt, the review flow).
* Do not restart or retune Frigate for this (Frigate restarts preceded host freezes, T103).

## Considered Options

* Lower Frigate's dog threshold so squatting stays an event, and shorten `clear_s`.
* App-side frames and checks (chosen).
* A new Gemini prompt that gets more frames.

## Decision Outcome

Chosen option: "app-side frames and checks", because it fixes all four cases without touching Frigate, and it costs at most one extra Gemini call and only when a person came right after her.

1. **Exit frame.** When she leaves both zones and no person is in `tray_area`, the tracker emits `dog_left` and the service stores `left_<entries>.jpg`. If a person was flagged on the visit by the time the after frame is due, the exit frame of the last entry becomes `after.jpg` (what Gemini judges), and the late frame is kept as `after_person.jpg`. The late frame still becomes the reference and the watch baseline.
2. **What the person did.** With `after_person.jpg`, the pad-change prompt compares the exit and late frames: `replaced` records a pad change (source `visit`), and `replaced` or `poop_removed` clears a counted poop (by `owner`). When Gemini saw no poop but the person removed one that no open or recently cleared (15 min) poop explains, the visit's poop is counted (`auto_poop = final_poop = poop`, cleared by `owner`, `needs_review`).
3. **Found poop at a cleanup.** A presence whose pad answer says `poop_removed`, with no open poop and none cleared in the 15 min before it, becomes a poop-only visit at the presence time (`final_outcome = no_pee`, `final_poop = poop`, cleared by `owner`, `needs_review`, reason "cocô achado na limpeza"). The real poop time is earlier, so the usual-time statistics (ADR-0015) are biased a little late for these.
4. **Second opinion on "no poop".** When Gemini's poop is a confident `no_poop`, `watch.floor_blob(before, after)` applies the floor-candidate rule (peak >= 1.0, floor mean <= 0.06, ratio >= 20). A blob makes the poop `uncertain`, so the owner gets the existing "Ellie: fez cocô?" question. On the 55 stored visits it fires for the 3 poops beside the tray (1.19, 1.67, 3.45) and once without a poop (1.03). It cannot judge the pad itself, where a pee darkens as much as a poop (up to 3.7).
5. **Lasting change confirms a pad candidate.** A pad candidate is also confirmed by a later frame whose pad still differs from the frame before the candidate by >= 0.10, with the floor within `CTRL_MAX` and no light jump. That later frame is then the one Gemini judges. The CSV row kind is `pad_lasting`.
6. **Own during frame.** 8 s into a visit the tracker emits `during` and the service stores `during_own.jpg`. It is sent to Gemini only when the Frigate snapshot is missing.

### Consequences

* Good, because the 09/11 08:22 case now counts the poop and marks it cleaned, and the 05/11 06:13 case asks the owner.
* Good, because cleanups the system never saw still add the poop to the history.
* Bad, because there are two more frame grabs per visit (go2rtc, about 0.5 MB each on disk, removed by the same retention).
* Bad, because the exit frame may still show part of her if Frigate's count drops early. Gemini sees the before and during frames too.
* Bad, because the floor-blob rule asks about 1 in 55 visits for nothing.
* Bad, because a lasting-change confirmation can also come from a moved pad or tray. Gemini still decides, and the daily cap of 6 watch visits holds.

### Confirmation

Tests: `tests/replay/test_replay_visit.py` (exit frame, pad swap, found poop on a visit, own during frame), `tests/replay/test_replay_pad.py` (found poop at a cleanup), `tests/replay/test_replay_poop.py` (the real 06:13 photos ask the owner), `tests/unit/test_watch.py` (lasting change, `floor_blob`), `tests/unit/test_visits.py` (`dog_left`). Live: log lines "using the frame taken when she left", "the person after her", "floor has a new dark blob", "removed a poop nobody counted", and watch rows of kind `pad_lasting`. Re-check at the next scorecard.

## Amendment 2026-11-09 (app 3.8.0)

On the first day of 3.7.0, rule 5 opened 3 false visits (12:59, 13:17, 13:28). They were a puddle drying after a visit, and the owner lifting, cleaning and putting back the tray. The owner was asked about a pee twice. Pixel signs cannot tell these apart from a pee (13:17 and 13:28 darken the pad like one). So a lasting-change confirmation now opens a visit only when there was no visit, presence or pad change in the 30 minutes before the candidate (`LASTING_QUIET`). The cost is that 07/11 08:52 (a presence at 08:42) would also have been skipped.

The same day, two presences were "uncertain" because the owner was still in the after frame (13:08). That occluded frame then became the "before" of the next presence (13:10). Now an `occluded` pad answer drops the frame from the reference history and takes the after frame again once the tray has been clear for 60 s, up to 3 times, before the owner is asked (`PRESENCE_RETAKES`). An app restart in the middle falls back to the old behaviour (the presence is reanalysed and asks).

## More Information

Spec 003 tasks T115-T119. Related: ADR-0009 (reopen and debounce), ADR-0012 (pad watch), ADR-0014 (poop outcome), ADR-0015 (poop habits). `Service.wait_idle` now yields once per round: a task that had finished but was not yet discarded made it spin.
