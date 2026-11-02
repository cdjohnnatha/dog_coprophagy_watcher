---
status: accepted
date: 2026-11-02
decision-makers: Claudio ("podes implementar assim" after seeing the stained photo still on the dashboard after a pad change)
---

# The big dashboard photo shows the clean pad after a pad change

## Context and Problem Statement

`image.ellie_last_pee_photo` showed the `after.jpg` of the last counted pee until the next pee. After the owner swapped the pad, the dashboard still showed the old, stained pad, which looked like the change had not been noticed.

## Decision

The entity (id unchanged, friendly name now "Foto do tapete") shows the latest of two events:

- the last counted pee (as before), or
- the current pad change, if it is newer: the `after.jpg` of the presence or visit that came with it (the clean pad), or, for a manual mark with no photo of its own, the newest reference frame taken **at or after** the mark. The owner may tap right after swapping, when the newest reference can still show the old pad, so it waits for the next minute's frame and keeps the previous photo meanwhile.

Undoing the change brings the previous photo back. With no pee and no change, the grey placeholder stays. The small thumbnail in the "Tapete" card stays as it was.

## Consequences

- After a change the photo of the last pee is no longer on the big card. It is still in the review list, `sensor.ellie_last_visit` and `labels.jsonl`.
- The key of the published photo is the visit id, or `pad:<change id>`, so each photo is sent to MQTT once.
- The card title in the dashboard files changed to "Foto do tapete"; the live dashboard needs that one name edited (or a repaste of `dashboard-ellie-limpo.yaml`).
