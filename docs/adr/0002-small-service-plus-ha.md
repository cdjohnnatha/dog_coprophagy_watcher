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
* Bad, because it is custom code to maintain (mitigated by size limit, tests, and these ADRs).

### Confirmation

Line count and test coverage of the core checked in review. All entities have `unique_id`. Quickstart scenarios pass.

## More Information

research.md R1, R7, R8.
