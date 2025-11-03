# Ellie Watcher - Dog Coprophagy Detection System

A multi-layered Python application for monitoring dog behavior and detecting coprophagy (poop eating) using computer vision and Frigate NVR integration.

## Architecture

The application follows a clean, layered architecture with clear separation of concerns:

```
dog_coprophagy_watcher/
├── settings.py              # Environment configuration (Pydantic Settings)
├── domain/                  # Core business logic (pure, no I/O)
│   ├── models.py           # Data models and domain entities
│   ├── heuristics.py       # Pure scoring and detection functions
│   ├── fsm.py              # Finite State Machine for behavior tracking
│   └── services.py         # Orchestration and workflow logic
├── adapters/               # External integrations and I/O
│   ├── clock.py           # Time abstraction (testable)
│   ├── cv_ops.py          # OpenCV operations
│   ├── frigate_client.py  # Frigate API client
│   └── mqtt_client.py     # MQTT operations
├── app/                    # Application layer
│   ├── handlers.py        # MQTT message handlers
│   └── runner.py          # Dependency injection and wiring
├── main.py                # Entry point (direct execution)
└── __main__.py            # Entry point (module execution)
```

## Layers Explained

### 1. **Settings Layer** (`settings.py`)
- Centralizes all environment variables using Pydantic Settings
- Provides type-safe, validated configuration
- Immutable settings object

### 2. **Domain Layer** (`domain/`)
Pure business logic with no I/O dependencies:

- **`models.py`**: Data classes for entities (TrackState, PoopEvent, DogDetection, etc.)
- **`heuristics.py`**: Pure functions for scoring and detection
  - `score_squat()`: Calculate squat probability
  - `is_coprophagy()`: Determine if coprophagy occurred
  - `is_near_poop()`: Check proximity to poop
  - Shape and texture filtering functions
- **`fsm.py`**: Finite State Machine managing Ellie's states
  - States: IDLE, POSSIVEL_DEFECACAO, DEFECANDO, etc.
  - Signals: SQUAT_START, RESIDUE_CONFIRMED, etc.
  - Returns Commands for adapters to execute
- **`services.py`**: Orchestration service coordinating workflows
  - Manages confirmation windows
  - Coordinates monitoring threads
  - Translates FSM commands to adapter calls

### 3. **Adapters Layer** (`adapters/`)
External integrations and I/O operations:

- **`clock.py`**: Time abstraction (SystemClock, MockClock for testing)
- **`cv_ops.py`**: All OpenCV operations
  - Image decoding, ROI extraction
  - Blob detection and area calculation
  - Monochrome detection
- **`frigate_client.py`**: Frigate NVR API client
  - Snapshot fetching
  - Event searching and updating
  - URL generation
- **`mqtt_client.py`**: MQTT operations
  - Publishing state, events, alerts
  - Connection management
  - Debug logging

### 4. **Application Layer** (`app/`)
Wiring and request handling:

- **`handlers.py`**: MQTT message handlers
  - Transforms MQTT payloads into domain DTOs
  - Calls service methods
- **`runner.py`**: Application bootstrap
  - Dependency injection
  - Wiring all components
  - Main event loop

## Installation

```bash
pip install -r requirements.txt
```

## Running

### As a module:
```bash
python -m dog_coprophagy_watcher
```

### Direct execution:
```bash
python main.py
```

### Using the original entry point:
```bash
python app.py  # Still works, imports from new structure
```

## Environment Variables

All configuration is done via environment variables. See `settings.py` for the complete list. Key variables:

- `MQTT_HOST`, `MQTT_PORT`, `MQTT_USER`, `MQTT_PASS`
- `FRIGATE_BASE_URL`, `CAMERA_NAME`, `TOILET_ZONE`
- `SQUAT_SCORE_THRESH`, `SQUAT_MIN_DURATION_S`
- `ENABLE_DEBUG_WATCHER`

## Testing

The layered architecture makes testing easy:

```python
from adapters.clock import MockClock
from domain.heuristics import score_squat

# Test pure functions
score = score_squat(ratio=0.5, stationary=True, speed=0.1, ...)
assert score.is_squatting

# Test with mock clock
clock = MockClock(initial_time=1000.0)
service = EllieWatcherService(..., clock=clock)
clock.advance(10.0)  # Simulate time passing
```

## Benefits of This Architecture

1. **Testability**: Pure functions and injected dependencies make unit testing trivial
2. **Maintainability**: Clear separation of concerns, easy to locate and modify code
3. **Extensibility**: Add new adapters (e.g., different NVR systems) without touching domain logic
4. **Readability**: Each layer has a single responsibility
5. **Reusability**: Domain logic can be reused in different contexts (CLI, web service, etc.)

## Documentation

Comprehensive documentation is available in the `docs/` folder:

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Detailed architecture documentation with diagrams
- **[Migration Guide](docs/MIGRATION.md)** - Complete guide for migrating from the original monolithic code
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Developer quick reference for common tasks

## Migration from Original `app.py`

The original monolithic `app.py` has been refactored into this layered structure. See the [Migration Guide](docs/MIGRATION.md) for detailed instructions.

Quick steps:
1. Install dependencies: `pip install -r requirements.txt`
2. Update any imports if you have external code referencing the old structure
3. Run using one of the methods above

The original `app.py` can be kept as a compatibility shim if needed, or removed once migration is complete.
