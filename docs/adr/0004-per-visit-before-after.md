---
status: accepted
date: 2026-10-29
decision-makers: Claudio
---

# Count per visit by comparing before/after of the tray area

## Context and Problem Statement

The pad already carries old stains and can be white or grey. "Is there pee on the pad?" stays true all day. What unit do we count, and what do we compare?

## Decision Drivers

* FR-006 (new change only), FR-002 (circling, re-entries), FR-003 (poop beside the tray), SC-006 (never double count).

## Considered Options

* Per visit, before vs after crop of the tray plus a margin, with a "during" frame of her posture.
* Continuous state classification of the pad ("clean / used").
* Posture-only (squat detection).

## Decision Outcome

Chosen option: **per visit, before/after of the analysis area (tray + 20 % margin) plus a "during" frame**. At most one pee per visit. Entries within 60 s are merged. The after frame is taken only once the area is clear of dog and person.

### Consequences

* Good, because old stains and pad colour cancel out, and poop beside the tray is in frame for later.
* Bad, because two pees in one visit count as one (owner can add manually), and the count lags the visit by about 1–2 minutes.

### Confirmation

Quickstart scenarios 1–6. SC-006 holds over the validation period.

## More Information

research.md R2, R4. data-model.md state machine.
