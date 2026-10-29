# Architecture Decision Records

Format: [MADR 4](https://adr.github.io/madr/) (Markdown Architectural Decision Records). One decision per file, `NNNN-short-title.md`, never renumbered.

Status values: `proposed` → `accepted` | `rejected`; later `deprecated` or `superseded by NNNN`. A superseded ADR is not edited except for its status line and a link to the new one.

These ADRs were drafted with spec [003-ellie-pee-counter](https://github.com/cdjohnnatha/home-assistant-jetsons/blob/003-ellie-pee-counter/specs/003-ellie-pee-counter/spec.md) in the Home Assistant config repo and copied here (task T009). **This folder is the source of truth from now on.** The copy under `specs/003-ellie-pee-counter/adr/` is kept in sync until that feature is done, then it is replaced by a link.

| ADR | Title | Status |
|---|---|---|
| [0000](0000-template.md) | Template | n/a |
| [0001](0001-reuse-repo-archive-v2.md) | Reuse the existing repository, archive v2, ADRs in `docs/adr` | accepted (2026-10-29) |
| [0002](0002-small-service-plus-ha.md) | Small service for logic, Home Assistant for UI and notifications | accepted (2026-10-29) |
| [0003](0003-gemini-deciding-analyser.md) | Gemini (paid tier) as the Phase 1 deciding analyser | accepted (2026-10-29) |
| [0004](0004-per-visit-before-after.md) | Count per visit by comparing before/after of the tray area | accepted (2026-10-29) |
| [0005](0005-retire-pixel-heuristics.md) | Retire the pixel-heuristic watcher (v2) | accepted (2026-10-29) |
| [0006](0006-local-analyser-path.md) | Path to a local analyser on the Pi | proposed (pending spike S1) |
