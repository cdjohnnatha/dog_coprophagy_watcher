# Architecture Documentation

## Overview

The Ellie Watcher application follows a **clean, layered architecture** with strict separation of concerns. Each layer has a specific responsibility and dependencies flow in one direction (from outer to inner layers).

## Dependency Flow

```
┌─────────────────────────────────────────────────────────┐
│                     Entry Points                         │
│              (main.py, __main__.py, app.py)             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Application Layer                       │
│                   (app/runner.py)                        │
│  • Dependency Injection                                  │
│  • Component Wiring                                      │
│  • Main Event Loop                                       │
└────────┬────────────────────────────────────┬───────────┘
         │                                    │
         ▼                                    ▼
┌────────────────────┐            ┌──────────────────────┐
│   app/handlers.py  │            │  domain/services.py  │
│  • MQTT callbacks  │───────────▶│  • Orchestration     │
│  • DTO conversion  │            │  • Workflows         │
└────────────────────┘            │  • Business logic    │
                                  └──────────┬───────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
         ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
         │  domain/fsm.py   │    │domain/heuristics │    │ domain/models.py │
         │  • State machine │    │  • Pure functions│    │  • Data models   │
         │  • Transitions   │    │  • Calculations  │    │  • DTOs          │
         └──────────────────┘    └──────────────────┘    └──────────────────┘
                    ▲                        ▲                        ▲
                    │                        │                        │
                    └────────────────────────┴────────────────────────┘
                                      Domain Layer
                                   (Pure, No I/O)
                                             │
                    ┌────────────────────────┴────────────────────────┐
                    ▼                        ▼                        ▼
         ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
         │ adapters/        │    │ adapters/        │    │ adapters/        │
         │ mqtt_client.py   │    │ frigate_client.py│    │ cv_ops.py        │
         │  • MQTT pub/sub  │    │  • API calls     │    │  • OpenCV ops    │
         └──────────────────┘    └──────────────────┘    └──────────────────┘
                    │                        │                        │
                    └────────────────────────┴────────────────────────┘
                                    Adapters Layer
                                  (I/O Operations)
                                             │
                                             ▼
                              ┌──────────────────────────┐
                              │     External Systems     │
                              │  • MQTT Broker           │
                              │  • Frigate NVR           │
                              │  • File System           │
                              └──────────────────────────┘
```

## Layer Responsibilities

### 1. Settings Layer (`settings.py`)

**Purpose:** Centralized configuration management

**Responsibilities:**
- Load and validate environment variables
- Provide type-safe configuration access
- Define default values
- Generate MQTT topic names

**Dependencies:** None (only Pydantic)

**Example:**
```python
from settings import load_settings

settings = load_settings()
print(settings.camera_name)  # Type-safe access
```

### 2. Domain Layer (`domain/`)

**Purpose:** Core business logic (pure, no I/O)

#### `domain/models.py`
- Data classes and DTOs
- Type definitions
- Enums for states

**Key Models:**
- `TrackState`: Main state tracker
- `DogDetection`: Detection event data
- `PoopEvent`: Poop confirmation event
- `CoprophagyEvent`: Coprophagy detection event
- `EllieState`: State machine states (Enum)

#### `domain/heuristics.py`
- Pure calculation functions
- No side effects
- Easily testable

**Key Functions:**
- `score_squat()`: Calculate squat probability
- `is_coprophagy()`: Determine coprophagy occurrence
- `is_near_poop()`: Check proximity
- `passes_shape_filter()`: Geometric filtering
- `passes_texture_filter()`: Texture analysis

#### `domain/fsm.py`
- Finite State Machine
- State transitions
- Command generation

**States:**
- IDLE
- POSSIVEL_DEFECACAO
- DEFECANDO
- AGUARDANDO_CONFIRMACAO
- DEFECACAO_CONFIRMADA
- COPROPHAGIA_CONFIRMADA

**Signals:**
- SQUAT_START
- SQUAT_CONTINUE
- SQUAT_END
- RESIDUE_CONFIRMED
- RESIDUE_NOT_FOUND
- POOP_CLEANED
- COPROPHAGY_CONFIRMED

#### `domain/services.py`
- Orchestration logic
- Workflow coordination
- Thread management
- Command execution

**Key Methods:**
- `handle_dog_detection()`: Process dog detection
- `start_confirmation_window()`: Begin residue check
- `check_episode_timeout()`: Episode management

**Dependencies:** Domain models, heuristics, FSM, adapters (injected)

### 3. Adapters Layer (`adapters/`)

**Purpose:** External integrations and I/O operations

#### `adapters/clock.py`
- Time abstraction
- Testable time operations

**Implementations:**
- `SystemClock`: Real system time
- `MockClock`: For testing

#### `adapters/cv_ops.py`
- OpenCV operations
- Image processing
- Blob detection

**Key Functions:**
- `safe_roi()`: Safe ROI extraction
- `diff_blob()`: Find residue blobs
- `diff_area()`: Calculate total area
- `is_monochrome_roi()`: Detect IR mode

#### `adapters/frigate_client.py`
- Frigate API client
- Event management
- Snapshot fetching

**Key Methods:**
- `fetch_snapshot()`: Get latest image
- `find_nearby_dog_event()`: Search events
- `update_event_sub_label()`: Update event
- `get_event_urls()`: Generate URLs

#### `adapters/mqtt_client.py`
- MQTT operations
- Message publishing
- Connection management

**Key Methods:**
- `publish_state()`: Publish Ellie's state
- `publish_poop_present()`: Poop status
- `publish_coprophagy_risk()`: Risk alert
- `publish_coprophagy_event()`: Confirmation

**Dependencies:** Settings, external systems

### 4. Application Layer (`app/`)

**Purpose:** Wiring and request handling

#### `app/handlers.py`
- MQTT message handlers
- Payload parsing
- DTO conversion
- Service invocation

**Key Methods:**
- `on_connect()`: Connection handler
- `on_message()`: Message handler
- `_extract_bbox()`: Parse bounding box
- `_build_dog_detection()`: Create detection DTO

#### `app/runner.py`
- Application bootstrap
- Dependency injection
- Component wiring
- Main event loop

**Key Class:**
- `Application`: Main orchestrator
  - Initializes all components
  - Wires dependencies
  - Runs main loop

**Dependencies:** All layers

### 5. Entry Points

#### `main.py`
Direct execution: `python main.py`

#### `__main__.py`
Module execution: `python -m dog_coprophagy_watcher`

#### `app.py` (compatibility)
Legacy entry point: `python app.py`

## Data Flow

### Detection Flow

```
1. Frigate → MQTT Event
         ↓
2. MQTT Client receives message
         ↓
3. handlers.on_message() parses payload
         ↓
4. Creates DogDetection DTO
         ↓
5. service.handle_dog_detection()
         ↓
6. heuristics.score_squat() calculates score
         ↓
7. If squatting: FSM transition to POSSIVEL_DEFECACAO
         ↓
8. FSM returns Commands
         ↓
9. service executes Commands via adapters
         ↓
10. MQTT Client publishes state
```

### Confirmation Flow

```
1. Squat ends → FSM transition to AGUARDANDO_CONFIRMACAO
         ↓
2. service.start_confirmation_window()
         ↓
3. Thread spawned: _confirmation_window_worker()
         ↓
4. Loop: fetch snapshots via frigate_client
         ↓
5. cv_ops.diff_blob() detects residue
         ↓
6. If stable: FSM transition to DEFECACAO_CONFIRMADA
         ↓
7. FSM returns Commands
         ↓
8. service executes: publish events, start monitors
         ↓
9. Monitoring threads: _monitor_poop_present(), _monitor_coprophagy()
```

## Testing Strategy

### Unit Tests (Domain Layer)

```python
# Test pure functions
from domain.heuristics import score_squat

def test_squat_high_score():
    score = score_squat(
        ratio=0.5,
        stationary=True,
        speed=0.05,
        motionless_count=5,
        squat_thresh=0.65,
        motionless_min=3,
        speed_max_still=0.15,
    )
    assert score.is_squatting == True
    assert score.score > 0.65
```

### Integration Tests (Service Layer)

```python
# Test with mocked adapters
from adapters.clock import MockClock
from domain.services import EllieWatcherService

def test_confirmation_window():
    clock = MockClock()
    mock_frigate = MockFrigateClient()
    mock_mqtt = MockMQTTClient()
    
    service = EllieWatcherService(
        settings=test_settings,
        clock=clock,
        cv_ops=mock_cv_ops,
        frigate_client=mock_frigate,
        mqtt_client=mock_mqtt,
    )
    
    # Test workflow
    service.start_confirmation_window()
    clock.advance(20.0)
    
    assert mock_mqtt.published_states[-1] == "DEFECACAO_CONFIRMADA"
```

### End-to-End Tests

```python
# Test full application
def test_full_detection_flow():
    app = Application()
    # Simulate MQTT message
    # Verify state transitions
    # Check published events
```

## Extension Points

### Adding New Adapters

```python
# Example: Add Telegram notifications
class TelegramClient:
    def __init__(self, settings: Settings):
        self.bot_token = settings.telegram_bot_token
    
    def send_alert(self, message: str) -> None:
        # Implementation
        pass

# Wire in runner.py
telegram = TelegramClient(settings)
service = EllieWatcherService(..., telegram_client=telegram)
```

### Adding New Heuristics

```python
# domain/heuristics.py
def calculate_eating_probability(
    head_down_duration: float,
    mouth_movement: bool,
    proximity_to_poop: float,
) -> float:
    """Calculate probability dog is eating."""
    # Pure function logic
    return probability
```

### Adding New States

```python
# domain/fsm.py
class EllieState(str, Enum):
    # ... existing states
    EATING = "EATING"
    INVESTIGATING = "INVESTIGATING"

# Add transitions in EllieFSM class
```

## Performance Considerations

### Thread Management
- Confirmation window: 1 thread per episode
- Poop monitoring: 1 thread per confirmed poop
- Coprophagy monitoring: 1 thread per confirmed poop
- Main loop: 1 thread for episode timeout checks

### Memory Usage
- Baseline images stored in TrackState
- ROI images (not full frames)
- Minimal state retention

### Network Calls
- Snapshot fetching: Rate-limited by SNAPSHOT_FPS
- MQTT publishing: Non-blocking
- Frigate API: Cached where possible

## Security Considerations

1. **Credentials:** MQTT credentials via environment variables
2. **Network:** Internal network communication assumed
3. **Input Validation:** Pydantic validates all settings
4. **Error Handling:** Graceful degradation on failures

## Monitoring and Debugging

### Debug Mode
```bash
export ENABLE_DEBUG_WATCHER=true
python -m dog_coprophagy_watcher
```

### MQTT Debug Topic
```
home/ellie/debug/log
```

### Health Check
```
home/ellie/health
```

## Future Enhancements

1. **Metrics Collection:** Add Prometheus metrics
2. **Web Dashboard:** Real-time monitoring UI
3. **Machine Learning:** Replace heuristics with trained models
4. **Multi-Camera:** Support multiple cameras/zones
5. **Event Replay:** Record and replay events for testing
6. **Configuration UI:** Web-based configuration editor

## Conclusion

This architecture provides:
- ✅ Clear separation of concerns
- ✅ Easy testing at all levels
- ✅ Simple extension points
- ✅ Type safety throughout
- ✅ Maintainable codebase
- ✅ Production-ready structure

For implementation details, see the source code and inline documentation.

