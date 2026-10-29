---
status: accepted
date: 2026-10-29
decision-makers: Claudio (accepted by delegating the implementation; revisit if it surprises you)
---

# Dashboard photos through MQTT image entities

## Context and Problem Statement

The review card and the last-pee card need to show the before/after photos. The app stores them under /media/ellie, and the first contract used /media/local/... URLs. A /media/local URL in a dashboard card needs a signed URL, so it would not render.

## Considered Options

* Keep /media/local URLs (works in phone notifications, not in cards).
* Copy photos into the HA `www` folder (served at /local/, **without authentication**).
* MQTT `image` entities fed with the JPEG bytes.

## Decision Outcome

Chosen option: **MQTT image entities** (`image.ellie_last_pee_photo`, `image.ellie_review_before`, `image.ellie_review_after`). The app publishes the bytes (retained) on `ellie/image/*` only when the photo changes; HA serves them with its own authentication. Notifications keep using the /media/local URL (companion app, unverified on iOS).

### Consequences

* Good, because photos stay private and need no extra mapped folder.
* Good, because they are published only on change (a few hundred KB, a few times a day).
* Bad, because the review card shows only the newest pending visit (the buttons answer that one).
