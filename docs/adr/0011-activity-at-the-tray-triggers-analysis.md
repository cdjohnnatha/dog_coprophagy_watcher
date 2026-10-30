---
status: accepted
date: 2026-10-30
decision-makers: Claudio (approved "Pode fazer a correção" after the first morning of real use)
---

# Any activity at the tray is analysed, not only a "dog in pad_zone"

## Context and Problem Statement

First morning of real use (2026-10-30), ground truth from the owner: pee 02:44, poop 06:53, pee about 08:24. The app counted only the 02:44 pee. Evidence (Frigate events, snapshots, recordings, the app's photos):

* 06:53 poop: Frigate labelled Ellie, squatting and seen from above, as **person** (74 %, box about 25,600 px² against about 145,600 px² for a real person). The app treated it as a person presence, asked only "was the pad replaced?" and missed the poop that was visible in its own after photo.
* 08:24 pee: she stood in front of the tray with her feet just outside `pad_zone`. Frigate saw a dog only in `tray_area` (1 s and 6 s tracks), never in `pad_zone`, so no visit was opened. The new stain is visible in the recordings (08:24:05 vs 08:24:50).
* Gemini itself was right whenever it received the photos: re-run on the three real cases, 3 of 3 correct.

## Considered Options

* Enlarge `pad_zone` in Frigate. Fixes 08:24 only; does not fix a dog labelled as person.
* Filter "person" detections by box size (a squatting dog is much smaller). Frigate would drop them, but then there is no track at all.
* Treat a dog in **`pad_zone` or `tray_area`** as at the tray, and ask every person presence (when the pad was not replaced) the visit question too; new pee or poop turns it into a visit.

## Decision Outcome

Chosen option: the third. `tray_area` is already `pad_zone` plus 20 % margin, so this is the "bigger zone" without touching Frigate, and the second question covers label flips. Visits keep the same before/after flow.

### Consequences

* Good, because the two misses of 2026-10-30 are caught (verified with the real photos and the real model).
* Good, because the app no longer depends on Frigate getting the label or the exact zone right; Gemini judges the photos either way.
* Bad, because more Gemini calls: one more per person presence (about 0.001 EUR each) and visits for dog passes in front of the tray. Expected about 1 euro cent a day.
* Bad, because a person presence with a new mark becomes a visit; if the owner spills something on the pad it may count. Unsure answers go to the review list.
* Bad, because a dog lying on the margin of `tray_area` (the rug edge) keeps a visit open; the 2-hour limit and the review list cover it.

### Confirmation

Tests reproduce both cases; the real-capture replay still yields one visit. The next mornings are compared with the owner's ground truth in validation.md.
