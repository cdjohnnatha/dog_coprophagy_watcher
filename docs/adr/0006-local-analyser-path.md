---
status: proposed
date: 2026-10-29
decision-makers: Claudio
---

# Path to a local analyser on the Pi

## Context and Problem Statement

The goal is local analysis on the Raspberry Pi once enough labels exist (spec Clarifications, SC-010). Spec 005 (coprophagy prevention) also needs reactions within **seconds** while Ellie is still at the tray, which only a local model can deliver.

## Decision Drivers

* SC-010, FR-010a/b, Future Evolution 005.
* Pi 5 is arm64. Frigate docs list AVX/AVX2 (x86) for custom classification.
* Old stains: a single-crop "pad state" model can't tell new from old.

## Considered Options

* Frigate **object** classification of the dog's posture (trained in Frigate's UI on the Pi).
* Own before/after classifier trained on the Mac (PyTorch with the Apple Metal backend) → ONNX → `onnxruntime` in the v3 service.
* Frigate **state** classification of the tray crop.

## Decision Outcome

**Pending spike S1** (does Frigate 0.18.2 train/run custom classification on this Pi?) and ~300 labelled visits.
Current leaning: **posture classification in Frigate** if S1 passes (it also serves spec 005), **plus** the own before/after classifier if posture alone doesn't reach SC-001. State classification is **not** chosen (can't separate new from old stains).

### Consequences

* If S1 passes: the Mac may not be needed at all.
* If S1 fails: the Mac trains the own classifier (mac-handover.md Phase 2).

### Confirmation

Shadow accuracy ≥ SC-001 for 14 consecutive days before switching `select.ellie_deciding_analyser`.

## More Information

research.md R6, R10. mac-handover.md.
