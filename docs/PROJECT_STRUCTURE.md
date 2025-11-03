# Project Structure

## Directory Layout

```
dog_coprophagy_watcher/
│
├── 📄 README.md                    # Main project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 app.py                       # Entry point (compatibility shim)
├── 📄 main.py                      # Entry point (direct execution)
├── 📄 __main__.py                  # Entry point (module execution)
├── 📄 app_legacy.py                # Original monolithic code (preserved)
├── 📄 settings.py                  # Configuration (Pydantic Settings)
│
├── 📁 docs/                        # 📚 Documentation
│   ├── README.md                   # Documentation index
│   ├── ARCHITECTURE.md             # Architecture guide
│   ├── MIGRATION.md                # Migration guide
│   ├── QUICK_REFERENCE.md          # Developer quick reference
│   └── PROJECT_STRUCTURE.md        # This file
│
├── 📁 domain/                      # 🧠 Business Logic (Pure, No I/O)
│   ├── __init__.py
│   ├── models.py                   # Data models, DTOs, Enums
│   ├── heuristics.py               # Pure calculation functions
│   ├── fsm.py                      # Finite State Machine
│   └── services.py                 # Orchestration logic
│
├── 📁 adapters/                    # 🔌 I/O Operations
│   ├── __init__.py
│   ├── clock.py                    # Time abstraction (testable)
│   ├── cv_ops.py                   # OpenCV operations
│   ├── frigate_client.py           # Frigate API client
│   └── mqtt_client.py              # MQTT operations
│
├── 📁 app/                         # 🔧 Application Layer
│   ├── __init__.py
│   ├── handlers.py                 # MQTT message handlers
│   └── runner.py                   # Dependency injection & wiring
│
├── 📁 tests/                       # 🧪 Test Suite (optional)
│   ├── test_heuristics.py
│   ├── test_fsm.py
│   └── test_services.py
│
└── 📁 [other files]                # Docker, config, etc.
    ├── Dockerfile
    ├── config.yaml
    ├── build.yaml
    └── run.sh
```

## File Descriptions

### Root Level

| File | Purpose | Lines | Type |
|------|---------|-------|------|
| `README.md` | Main documentation, getting started | ~160 | Markdown |
| `requirements.txt` | Python package dependencies | ~10 | Text |
| `settings.py` | Centralized configuration with Pydantic | ~130 | Python |
| `app.py` | Compatibility entry point | ~20 | Python |
| `main.py` | Direct execution entry point | ~10 | Python |
| `__main__.py` | Module execution entry point | ~10 | Python |
| `app_legacy.py` | Original monolithic code (preserved) | ~15 | Python |

### Documentation (`docs/`)

| File | Purpose | Lines | Audience |
|------|---------|-------|----------|
| `README.md` | Documentation index and navigation | ~200 | All |
| `ARCHITECTURE.md` | Detailed architecture documentation | ~450 | Developers, Architects |
| `MIGRATION.md` | Migration guide from v1.0 to v2.0 | ~270 | DevOps, Maintainers |
| `QUICK_REFERENCE.md` | API reference and common tasks | ~320 | Developers |
| `PROJECT_STRUCTURE.md` | This file - project layout | ~100 | All |

### Domain Layer (`domain/`)

| File | Purpose | Lines | Complexity |
|------|---------|-------|------------|
| `models.py` | Data models, DTOs, Enums | ~150 | Low |
| `heuristics.py` | Pure calculation functions | ~200 | Medium |
| `fsm.py` | Finite State Machine | ~250 | Medium |
| `services.py` | Orchestration and workflows | ~400 | High |

**Total Domain Lines:** ~1000 (pure business logic)

### Adapters Layer (`adapters/`)

| File | Purpose | Lines | External Deps |
|------|---------|-------|---------------|
| `clock.py` | Time abstraction | ~80 | time |
| `cv_ops.py` | OpenCV operations | ~250 | cv2, numpy |
| `frigate_client.py` | Frigate API client | ~150 | requests |
| `mqtt_client.py` | MQTT operations | ~150 | paho-mqtt |

**Total Adapter Lines:** ~630 (I/O operations)

### Application Layer (`app/`)

| File | Purpose | Lines | Role |
|------|---------|-------|------|
| `handlers.py` | MQTT message handlers | ~120 | Request handling |
| `runner.py` | Dependency injection & wiring | ~100 | Bootstrap |

**Total Application Lines:** ~220 (wiring)

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│                     Entry Points                         │
│         (app.py, main.py, __main__.py)                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ app.runner  │
                  └──────┬──────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
  ┌──────────┐   ┌─────────────┐   ┌──────────┐
  │ handlers │   │   settings  │   │ adapters │
  └────┬─────┘   └─────────────┘   └────┬─────┘
       │                                  │
       └──────────┬──────────────────────┘
                  ▼
         ┌─────────────────┐
         │ domain.services │
         └────────┬────────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
     ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│  fsm    │  │heuristic│  │ models  │
└─────────┘  └─────────┘  └─────────┘
```

## Code Statistics

### By Layer

| Layer | Files | Lines | Percentage | Testability |
|-------|-------|-------|------------|-------------|
| Domain | 4 | ~1000 | 50% | ⭐⭐⭐⭐⭐ (Pure) |
| Adapters | 4 | ~630 | 32% | ⭐⭐⭐⭐ (Mockable) |
| Application | 2 | ~220 | 11% | ⭐⭐⭐ (Integration) |
| Settings | 1 | ~130 | 7% | ⭐⭐⭐⭐ (Config) |
| **Total** | **11** | **~1980** | **100%** | - |

### Comparison with Original

| Metric | Original (v1.0) | Refactored (v2.0) | Change |
|--------|-----------------|-------------------|--------|
| Files | 1 | 11 | +1000% |
| Total Lines | 772 | ~1980 | +156% |
| Testable Lines | ~0% | ~50% | ∞ |
| Cyclomatic Complexity | High | Low-Medium | ⬇️ |
| Maintainability Index | Low | High | ⬆️ |

## Import Structure

### External Dependencies

```python
# Core Python
import time
import json
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

# Third-party
import numpy as np
import cv2
import requests
import paho.mqtt.client as mqtt
from pydantic import Field
from pydantic_settings import BaseSettings
```

### Internal Imports (Example)

```python
# In domain/services.py
from .models import TrackState, DogDetection, EllieState
from .fsm import EllieFSM, Signal, Command
from . import heuristics
from ..settings import Settings

# In app/runner.py
from ..settings import load_settings
from ..adapters.clock import SystemClock
from ..adapters.frigate_client import FrigateClient
from ..adapters.mqtt_client import MQTTClient
from ..domain.services import EllieWatcherService
from .handlers import MQTTHandlers
```

## Testing Structure (Recommended)

```
tests/
├── unit/                           # Unit tests (fast, isolated)
│   ├── test_heuristics.py         # Pure function tests
│   ├── test_fsm.py                # State machine tests
│   └── test_models.py             # Model validation tests
│
├── integration/                    # Integration tests (with mocks)
│   ├── test_services.py           # Service orchestration
│   ├── test_cv_ops.py             # OpenCV operations
│   └── test_frigate_client.py     # API client tests
│
└── e2e/                           # End-to-end tests (slow)
    └── test_full_flow.py          # Complete detection flow
```

## Configuration Files

```
dog_coprophagy_watcher/
├── .env                           # Environment variables (local)
├── config.yaml                    # Home Assistant addon config
├── Dockerfile                     # Container definition
└── requirements.txt               # Python dependencies
```

## Key Design Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| **Settings** | `settings.py` | Centralized configuration |
| **Data Transfer Object** | `domain/models.py` | Data structures |
| **Pure Functions** | `domain/heuristics.py` | Testable calculations |
| **State Machine** | `domain/fsm.py` | State management |
| **Command Pattern** | `domain/fsm.py` | Decouple commands from execution |
| **Adapter Pattern** | `adapters/*` | Abstract external systems |
| **Dependency Injection** | `app/runner.py` | Loose coupling |
| **Repository Pattern** | `adapters/frigate_client.py` | Data access abstraction |

## Navigation Tips

### Finding Code

**"Where is the squat detection logic?"**
→ `domain/heuristics.py::score_squat()`

**"Where do we handle MQTT messages?"**
→ `app/handlers.py::on_message()`

**"Where is the main loop?"**
→ `app/runner.py::Application.run()`

**"Where are the state transitions?"**
→ `domain/fsm.py::EllieFSM.transition()`

**"Where do we fetch snapshots?"**
→ `adapters/frigate_client.py::fetch_snapshot()`

**"Where is the OpenCV blob detection?"**
→ `adapters/cv_ops.py::diff_blob()`

### Adding Features

**New detection heuristic:**
1. Add function to `domain/heuristics.py`
2. Add tests
3. Use in `domain/services.py`

**New external integration:**
1. Create adapter in `adapters/`
2. Inject in `app/runner.py`
3. Use in `domain/services.py`

**New state:**
1. Add to `EllieState` enum in `domain/models.py`
2. Add transition in `domain/fsm.py`
3. Handle in `domain/services.py`

## Documentation Files

All documentation is in the `docs/` folder:

- **README.md** - Documentation index
- **ARCHITECTURE.md** - Architecture deep dive
- **MIGRATION.md** - Migration guide
- **QUICK_REFERENCE.md** - API reference
- **PROJECT_STRUCTURE.md** - This file

## Version Control

```
.gitignore should include:
__pycache__/
*.pyc
*.pyo
.env
*.log
.vscode/
.idea/
```

---

**Last Updated:** 2025-11-03  
**Version:** 2.0 (Layered Architecture)

