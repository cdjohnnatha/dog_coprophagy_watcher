# Project Structure (Runtime vs ML)

This repository is organized to clearly separate the Home Assistant runtime (production) from ML training code (experiments/artifacts).

## Top-level layout

```
dog_coprophagy_watcher/
├─ watcher/                    # Runtime package (what the add-on runs)
│  ├─ __init__.py
│  ├─ main.py                  # Entrypoint executed by run.sh
│  ├─ settings.py              # Pydantic settings (now loads thresholds.json if present)
│  ├─ app/
│  │  ├─ __init__.py
│  │  ├─ handlers.py           # MQTT handlers
│  │  └─ runner.py             # Wiring/DI, main loop
│  ├─ adapters/                # I/O adapters (Frigate, MQTT, CV, etc.)
│  └─ domain/                  # Business logic (FSM, heuristics, models, services)
│
├─ ml/                         # ML training & evaluation
│  ├─ data/                    # Dataset indexing, tf.data, video I/O, scripts
│  ├─ models/                  # Keras/TFLite models and losses/metrics
│  ├─ train/                   # Training CLI and pipeline
│  └─ eval/                    # Evaluation & prediction scripts
│
├─ scripts/                    # Back-compat wrappers (python -m scripts.* still works)
│
├─ docs/                       # Documentation
│  ├─ PROJECT_STRUCTURE.md     # This file
│  ├─ QUICK_REFERENCE.md       # Quick commands
│  ├─ ARCHITECTURE.md
│  ├─ MIGRATION.md
│  └─ RUNTIME_THRESHOLDS.md    # How thresholds.json is loaded in production
│
├─ datasets/                   # Local experiment outputs (ignored in Docker)
├─ assets/                     # Raw/normalized videos (ignored in Docker)
├─ artifacts/                  # Recommended: store exported models/thresholds here
│
├─ Dockerfile                  # Home Assistant image (copies only watcher/ + run.sh)
├─ config.yaml                 # Home Assistant add-on manifest
├─ run.sh                      # Add-on start script (loads options.json → env → watcher)
├─ .dockerignore               # Excludes ML/data from Docker context
└─ requirements.txt
```

## Runtime thresholds

The runtime reads thresholds from a JSON if present:
- Env var `THRESHOLDS_JSON` (highest precedence)
- `/data/thresholds.json` (add-on data share)
- `/config/ellie_thresholds.json` (HA config share)

Keys accepted:
```json
{ "poop_threshold": 0.80, "copro_threshold": 0.17 }
```

See `docs/RUNTIME_THRESHOLDS.md` for details.

## Backward compatibility

- Existing commands continue to work via wrappers:
  - `python -m scripts.train_from_folders ...` → forwards to `ml.train.cli`
  - `python -m scripts.snippets.eval_val ...` → forwards to `ml.eval.eval_val`

## Docker image size

The `.dockerignore` excludes ML data/artifacts. The Dockerfile only copies `watcher/` and `run.sh` to keep the add-on lean.

---

Last Updated: 2025-11-07  
Version: 3.0 (Runtime/ML split)

