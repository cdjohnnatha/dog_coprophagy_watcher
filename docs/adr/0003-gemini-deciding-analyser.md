---
status: accepted
date: 2026-10-29
decision-makers: Claudio
---

# Gemini (paid tier) as the Phase 1 deciding analyser

## Context and Problem Statement

Each finished visit needs a pee / no pee / uncertain decision from before, after and during images, with no training data available yet.

## Decision Drivers

* Owner accepted sending photos to the cloud and chose Gemini (spec Clarifications).
* FR-010: cropped images only, paid tier (no training on our data), spending alert.
* Accuracy target SC-001 from day one.

## Considered Options

* Gemini API direct (`google-genai`, JSON response schema).
* HA `ai_task.generate_data` via the Google Generative AI integration.
* OpenAI API.

## Decision Outcome

Chosen option: **Gemini API called directly from the service**, with a pinned stable Flash-Lite model id, prompt versioned in the repo (`pee_v1`), response validated against `analysis-result.schema.json`, and a 2 € Cloud budget alert. Structured output and inline images are first-class in the Gemini API, while local-file attachments with `structure` in `ai_task` for Google are only community-reported. `ai_task` remains a possible alternative analyser behind the same interface.

### Consequences

* Good, because it needs no training and has the best expected accuracy at the start, for about 1–2 € per month (plan Scale/Scope).
* Bad, because it depends on the internet (visits go to `unknown` + review when offline) and the model id must be re-pinned when Google retires it.

### Confirmation

Validation-mode accuracy (SC-001) over 14 days. `sensor.ellie_analysis_calls_month` and the Cloud budget alert stay under 2 €.

### API and call settings (owner OK 2026-10-29)

* **Interactions API** (`client.interactions.create`, REST `POST /v1beta/interactions`), which the current docs recommend for new work. `generateContent` is still supported.
* **`store=false`** on every call. Without it, the server keeps each interaction for 55 days (paid tier).
* **Structured output** via `response_format = {type: "text", mime_type: "application/json", schema: …}`.
* **Per-image resolution**: before/after `high`, during `low`.
* **Thinking level**: `minimal`.
* **Model**: start with `gemini-3.5-flash-lite` (stable).
* **Spending protection**: Prepay $5 with no auto-reload, an AI Studio spend cap of $5/month, and a Cloud budget alert of 2 €. The alert is expected to fire in the first month and in busy months.

#### Pinned model

Checked on 2026-10-29 (T010):

* **Model id**: `gemini-3.5-flash-lite`. It appears in `GET /v1beta/models` for our key and is listed as stable on the models page.
* **Price** (pricing page): $0.30 per 1M input tokens (text/image), $2.50 per 1M output tokens.
* **Smoke test**: one full-frame JPEG (2560×1440, ~420 KB) + short prompt + JSON schema, `store: false`, `thinking_level: minimal`.
  * The request syntax as written in gemini-api-setup.md step 7 works (HTTP 200), and the JSON matches the schema.
  * **Usage**: input 1,128 tokens (image 1,100 at `high`, text 28); output 76–82 tokens; thought tokens 0. Output is far below the ~400 assumed in the estimate, so real cost per visit is about $0.0013 instead of $0.0020.
  * **Latency**: 34–44 s per call. This is close to the planned 60 s timeout.
  * **Availability**: the first attempts (both `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite`) failed with HTTP 503 `service_unavailable` ("high demand"). The key itself was valid: model list and text calls returned 200.
* **Consequences for the implementation (T032)**:
  * Treat 503/429 as retryable, with backoff (for example 15 s, 60 s, 180 s) inside the visit's 5-minute budget before `unknown`.
  * Raise the per-call timeout to 90 s.
  * Optionally fall back to `gemini-3.1-flash-lite` on 503. The analyser id records which model answered.
  * The model already noticed the old yellow stain on the pad ("mancha amarelada no centro"), which confirms that the before/after comparison (ADR-0004) is needed.

## More Information

research.md R5, [gemini-api-setup.md](../gemini-api-setup.md) (setup steps, cost estimate, sources).
