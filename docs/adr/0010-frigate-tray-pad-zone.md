---
status: accepted
date: 2026-10-29
decision-makers: Claudio (accepted by delegating the implementation; revisit if it surprises you)
---

# Extra Frigate zone tray_pad and the on_pad_s evidence

## Context and Problem Statement

pad_zone is bigger than the tray: Ellie standing on the floor beside it already counted as inside. The owner allowed changes to the Frigate config for more precise events.

## Considered Options

* Shrink pad_zone (it would change the visit trigger and lose the margin around the tray).
* Add a second zone tight around the tray and keep pad_zone as the visit trigger.

## Decision Outcome

Chosen option: a new zone **`tray_pad`** (dog only, tight polygon around the tray, `0.405,0.195,0.635,0.185,0.645,0.560,0.405,0.570`), validated with Frigate's own config API after a backup. The visit stores `on_pad_s`, the seconds she stood on the pad itself. It is evidence only; using it to skip the analysis of visits with 0 s is a later decision, taken with data.

### Consequences

* Good, because "on the pad" and "near the pad" can be told apart without touching the visit trigger.
* Bad, because the polygon is eyeballed from one frame and needs a real pass to confirm (the zone published no count yet).
