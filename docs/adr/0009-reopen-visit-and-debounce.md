---
status: accepted
date: 2026-10-29
decision-makers: Claudio (accepted by delegating the implementation; revisit if it surprises you)
---

# A re-entry before the after photo reopens the visit; every decision is debounced

## Context and Problem Statement

A walk-through recorded on the real camera showed that Frigate publishes zone counts only on change and that they flap (a person count went 1,0,1,0,1,0 within one second; pad_zone/dog went 1,0,1,0 in four seconds while tray_area/dog stayed 1; one person produced four tracked ids). The first design started a new visit whenever the dog re-entered after the merge gap.

## Considered Options

* New visit on every re-entry after the merge gap (two visits would share one after photo).
* Reopen the same visit while its after photo is not taken yet, and debounce every "gone" and "clear" decision.

## Decision Outcome

Chosen option: **reopen and debounce**. A re-entry while the visit is CLOSING or CAPTURING reopens it; the after photo needs 30 s with no dog and no person in tray_area; a person presence must last 10 s and ends after 30 s clear; only the plain count topics are used (never `/active`); a presence is measured to the moment its count returns to 0 (counts are not republished while unchanged). A replay of the real capture is part of the test suite.

### Consequences

* Good, because one before/after pair covers the whole stay and flapping cannot split or end it early.
* Bad, because a visit can be long if the area never clears (it ends as `unknown` after 5 minutes).
