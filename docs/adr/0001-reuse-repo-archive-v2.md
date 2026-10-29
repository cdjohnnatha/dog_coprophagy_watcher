---
status: accepted
date: 2026-10-29
decision-makers: Claudio
---

# Reuse the existing repository, archive v2, ADRs in `docs/adr`

## Context and Problem Statement

The project restarts with a new purpose (pee counter, later poop/coprophagy). The existing repo `cdjohnnatha/dog_coprophagy_watcher` holds v2 (pixel heuristics), an unmerged `feat/llm` branch, and **15 uncommitted files** that exist only on the Home Assistant host. Where does v3 live, and where do ADRs live?

## Decision Drivers

* Keep history and nothing lost (constitution VII, Reversible by Default).
* One repo per deployable, with ADRs next to the code they constrain.
* The old MQTT password was committed (`config.yaml`, commit `d36354d`).

## Considered Options

* Reuse the repo: archive v2 on a branch and start a clean `main`.
* New repository `ellie-watcher`, old one archived read-only.
* Put the service inside the `/homeassistant` config repo.

## Decision Outcome

Chosen option: **reuse the repo**.
1. Commit the uncommitted v2 work to branch `legacy/v2`, tag `v2.4.3-final`, push. Leave `feat/llm` as is.
2. Start `main` from an empty tree for v3 (single "restart" commit that removes v2 files).
3. ADRs in MADR 4 under `docs/adr/`.
4. Optional: rename the repo to `ellie-watcher` (GitHub keeps redirects).
5. Rotate the Mosquitto password used by the old app.

Spec-kit artifacts and HA-side YAML stay in the `/homeassistant` repo.

### Consequences

* Good, because history, issues and datasets folders stay in one place.
* Good, because v2 keeps running from its own checkout during validation (`/local_apps/dog_coprophagy_watcher` on `legacy/v2`), while v3 runs from a second clone `/local_apps/ellie`.
* Bad, because two clones of one repo on the host can confuse. Mitigated by folder names and the README.

### Confirmation

`git branch -r` shows `legacy/v2`, the tag exists, `main` has no v2 files, `docs/adr/` contains these ADRs, and the old password no longer authenticates.

## More Information

research.md R11, R12, R13.
