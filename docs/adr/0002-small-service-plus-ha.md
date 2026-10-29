---
status: accepted
date: 2026-10-29
decision-makers: Claudio
---

# Small service for logic, Home Assistant for UI and notifications

## Context and Problem Statement

The feature needs visits with a merge window, before/after capture, per-visit records with corrections, back-dated daily counts, per-analyser accuracy, dataset export, and later a local classifier in shadow mode. Where should this logic run?

## Decision Drivers

* Constitution VIII: prefer native → installed → HACS → custom (custom needs justification and the owner's go-ahead).
* The owner's pain with v2: many fragile calculations.
* FR-009 (≤ 6 tunables), FR-013 (back-dated manual add), FR-018a (per-analyser accuracy), FR-010a (shadow analysers).

## Considered Options

* HA-native only: automations, `ai_task`, counters, `utility_meter`, File notify.
* Small service (local app) + HA for entities, notifications and dashboard.
* LLM Vision (HACS) + HA helpers.

## Decision Outcome

Chosen option: **small service + HA**, because native building blocks can't back-date daily totals or compare analysers without one helper per metric, and Phase 2 needs a runtime for the classifier anyway. The service stays small: pure, unit-tested core (visit state machine, decision rule, counting), SQLite, MQTT discovery, target < 1,000 lines excluding tests. Notifications, the no-pee alert and the dashboard remain native HA.

### Consequences

* Good, because the logic is testable and in one place, and corrections are exact.
* Good, because swapping analysers (cloud → local) is a setting, not a rewrite.
* Bad, because it is custom code to maintain (mitigated by tests and these ADRs; the original size limit was not kept, see the update below).

### Update 2026-10-29 (measured after implementation)

The original target was a service core **under 1,000 lines**. The delivered service is **2,116 code lines** (3,129 with comments and docstrings), measured with blank lines, comments and docstrings removed. The old v2 had about 2,800 lines in total, so v3 is **not smaller**.

Why it grew: the scope grew with the owner's requests (labels and corrections, accuracy per analyser, dataset export, poop clips, retention, owner alerts for the Gemini credit, the whole pad change flow, image entities), and the entity catalogue (`discovery.py`, 262 lines) is data. The largest modules are `service.py` (490), `discovery.py` (262), `store.py` (219) and `pads_flow.py` (187).

What the decision still buys, and what it does not: v3 has **no numeric pixel heuristics** (v2 had about 20 thresholds that shifted with light and pad colour), every behaviour is covered by **260 tests** (including a replay of the real Frigate capture), and the pure modules (`visits`, `pads`, `decide`, `counting`, `accuracy`) have no I/O. It does **not** buy less code to read. The size limit is dropped as a goal; the maintenance mitigation is the tests and these ADRs.
### Confirmation

Line count and test coverage of the core checked in review. All entities have `unique_id`. Quickstart scenarios pass.

## More Information

research.md R1, R7, R8.
