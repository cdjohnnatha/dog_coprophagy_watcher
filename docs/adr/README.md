# Architecture Decision Records (drafts)

Format: [MADR 4](https://adr.github.io/madr/) (Markdown Architectural Decision Records). One decision per file, `NNNN-short-title.md`, never renumbered.

Copied to the app repo (`github.com/cdjohnnatha/dog_coprophagy_watcher`, `docs/adr/`) on 2026-10-29. New or changed ADRs are written **there**; update this copy until spec 003 is done.

Status values: `proposed` → `accepted` | `rejected`; later `deprecated` or `superseded by NNNN`. A superseded ADR is not edited except for its status line and a link to the new one.

These drafts belong to spec [003-ellie-pee-counter](../spec.md). The first implementation task moves them to the project repository under `docs/adr/` (ADR-0001). From then on, that folder is the source of truth.

| ADR | Title | Status |
|---|---|---|
| [0000](0000-template.md) | Template | n/a |
| [0001](0001-reuse-repo-archive-v2.md) | Reuse the existing repository, archive v2, ADRs in `docs/adr` | accepted (2026-10-29) |
| [0002](0002-small-service-plus-ha.md) | Small service for logic, Home Assistant for UI and notifications | accepted (2026-10-29) |
| [0003](0003-gemini-deciding-analyser.md) | Gemini (paid tier) as the Phase 1 deciding analyser | accepted (2026-10-29) |
| [0004](0004-per-visit-before-after.md) | Count per visit by comparing before/after of the tray area | accepted (2026-10-29) |
| [0005](0005-retire-pixel-heuristics.md) | Retire the pixel-heuristic watcher (v2) | accepted (2026-10-29) |
| [0006](0006-local-analyser-path.md) | Path to a local analyser on the Pi | proposed (pending spike S1) |
| [0007](0007-dashboard-photos-via-mqtt-image.md) | Dashboard photos through MQTT image entities | accepted (2026-10-29) |
| [0008](0008-gemini-credit-alerts.md) | Tell the owner when the Gemini credit ends | accepted (2026-10-29) |
| [0009](0009-reopen-visit-and-debounce.md) | A re-entry before the after photo reopens the visit; every decision is debounced | accepted (2026-10-29) |
| [0010](0010-frigate-tray-pad-zone.md) | Extra Frigate zone tray_pad and the on_pad_s evidence | accepted (2026-10-29) |
| [0011](0011-activity-at-the-tray-triggers-analysis.md) | Any activity at the tray is analysed, not only a "dog in pad_zone" | accepted (2026-10-30) |
