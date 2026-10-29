---
status: accepted
date: 2026-10-29
decision-makers: Claudio (accepted by delegating the implementation; revisit if it surprises you)
---

# Tell the owner when the Gemini credit ends

## Context and Problem Statement

The Gemini key is on a pre-paid credit (about 4.5 EUR) with auto-reload off, so the credit is the spending cap. When it reaches zero the API answers HTTP 402 and visits would silently end as `unknown`. The owner asked to be notified.

## Considered Options

* Rely on Google's budget alert e-mail (it only warns and does not stop spending, and it is late).
* A hard alert from the app on HTTP 402/401/403 only.
* The hard alert plus an early warning from the estimated spend.

## Decision Outcome

Chosen option: **hard alert plus early warning**. The app publishes `ellie/alert` (at most once per 6 h per kind) on 402 (credit) and 401/403 (key); HA turns it into a phone and persistent notification. `ellie_v3_gemini_credit_low` warns at 80 % of `input_number.ellie_gemini_credit_eur` (total credit ever added, set by the owner) using the spend the app estimates from billed tokens and `usd_to_eur`.

### Consequences

* Good, because the owner learns about the problem in minutes, not after a week of unanalysed visits.
* Bad, because the spend is an estimate (price constants in `state.py`, a fixed exchange rate option): re-check both when the model changes.
