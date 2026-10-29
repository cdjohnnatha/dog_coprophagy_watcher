---
status: accepted
date: 2026-10-29
decision-makers: Claudio
---

# Retire the pixel-heuristic watcher (v2)

## Context and Problem Statement

v2 detects pee/poop with ~20 hand-tuned thresholds on colour deltas, blob mass and areas, in ~2,800 lines. It is sensitive to lighting, pad colour and stains, and the owner doesn't trust it.

## Decision Drivers

* Owner: "muitos cálculos… suscetível a erros"; willing to start from scratch.
* FR-009 (≤ 6 tunables).

## Considered Options

* Rewrite (v3) and retire v2 after validation.
* Refactor v2 and swap the heuristics for an analyser.

## Decision Outcome

Chosen option: **rewrite, then retire v2** after v3 meets the success criteria for 14 consecutive days (spec US6). v2 code is archived on `legacy/v2` (ADR-0001). Its HA leftovers are removed in one commit, after a backup.

### Consequences

* Good, because there is a small, testable codebase and one source of truth.
* Bad, because v2's poop alerts stop at retirement until spec 004 delivers poop counting. The owner accepts this gap or retires v2 after 004.

### Confirmation

The retirement checklist (research R13) is done, and no `home/ellie/*` topics are published afterwards.

## More Information

research.md R1, R13.
