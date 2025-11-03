# Quick Reference Guide

## Project Structure

```
dog_coprophagy_watcher/
├── settings.py                 # Configuration (Pydantic Settings)
├── domain/                     # Business Logic (Pure, No I/O)
│   ├── models.py              # Data models, DTOs, Enums
│   ├── heuristics.py          # Pure calculation functions
│   ├── fsm.py                 # State machine
│   └── services.py            # Orchestration logic
├── adapters/                   # I/O Operations
│   ├── clock.py               # Time abstraction
│   ├── cv_ops.py              # OpenCV operations
│   ├── frigate_client.py      # Frigate API client
│   └── mqtt_client.py         # MQTT operations
├── app/                        # Application Layer
│   ├── handlers.py            # MQTT message handlers
│   └── runner.py              # Dependency injection & wiring
├── main.py                     # Entry point (direct)
├── __main__.py                 # Entry point (module)
└── app.py                      # Entry point (compatibility)
```

## Running the Application

```bash
# Recommended (module execution)
python -m dog_coprophagy_watcher

# Alternative (direct execution)
python main.py

# Compatibility (uses new architecture)
python app.py
```

## Key Classes and Functions

### Settings (`settings.py`)
```python
from settings import load_settings

settings = load_settings()
# Access: settings.camera_name, settings.squat_score_thresh, etc.
```

### Domain Models (`domain/models.py`)
```python
from domain.models import (
    TrackState,      # Main state tracker
    DogDetection,    # Dog detection event
    PoopEvent,       # Poop confirmation
    CoprophagyEvent, # Coprophagy detection
    EllieState,      # State enum
    Blob,            # Detected blob
)
```

### Heuristics (`domain/heuristics.py`)
```python
from domain.heuristics import (
    score_squat,              # Calculate squat probability
    is_coprophagy,            # Determine coprophagy
    is_near_poop,             # Check proximity
    passes_shape_filter,      # Geometric filtering
    passes_texture_filter,    # Texture analysis
    distance_px,              # Calculate distance
    should_merge_episode,     # Episode merging logic
)
```

### FSM (`domain/fsm.py`)
```python
from domain.fsm import EllieFSM, Signal, Command

fsm = EllieFSM()
transition = fsm.transition(Signal.SQUAT_START, context={...})
# Returns: StateTransition with new_state and commands
```

### Service (`domain/services.py`)
```python
from domain.services import EllieWatcherService

service = EllieWatcherService(
    settings=settings,
    clock=clock,
    cv_ops=cv_ops,
    frigate_client=frigate,
    mqtt_client=mqtt,
)

# Main methods:
service.handle_dog_detection(detection)
service.handle_person_detection(timestamp)
service.check_episode_timeout()
```

### Adapters

#### Clock (`adapters/clock.py`)
```python
from adapters.clock import SystemClock, MockClock

clock = SystemClock()
# Methods: now(), now_iso(), sleep(s), monotonic()

# For testing:
mock_clock = MockClock(initial_time=1000.0)
mock_clock.advance(10.0)  # Simulate time passing
```

#### CV Operations (`adapters/cv_ops.py`)
```python
from adapters.cv_ops import (
    safe_roi,           # Safe ROI extraction
    diff_blob,          # Find residue blob
    diff_area,          # Calculate total area
    is_monochrome_roi,  # Detect IR mode
    decode_image,       # Decode image bytes
)
```

#### Frigate Client (`adapters/frigate_client.py`)
```python
from adapters.frigate_client import FrigateClient

frigate = FrigateClient(settings)
# Methods:
img = frigate.fetch_snapshot()
event_id = frigate.find_nearby_dog_event(timestamp)
frigate.update_event_sub_label(event_id, "poop")
urls = frigate.get_event_urls(event_id)
```

#### MQTT Client (`adapters/mqtt_client.py`)
```python
from adapters.mqtt_client import MQTTClient

mqtt = MQTTClient(settings, debug_enabled=True)
# Methods:
mqtt.publish_state("DEFECANDO")
mqtt.publish_poop_present(data)
mqtt.publish_coprophagy_risk(data)
mqtt.log("Debug message")
```

## State Machine

### States
- `IDLE` - No activity
- `POSSIVEL_DEFECACAO` - Squat detected
- `DEFECANDO` - Confirmed squatting
- `AGUARDANDO_CONFIRMACAO` - Waiting for residue
- `DEFECACAO_CONFIRMADA` - Poop confirmed
- `COPROPHAGIA_CONFIRMADA` - Coprophagy confirmed

### Signals
- `SQUAT_START` - Dog starts squatting
- `SQUAT_CONTINUE` - Squat maintained
- `SQUAT_END` - Dog stops squatting
- `RESIDUE_CONFIRMED` - Poop detected
- `RESIDUE_NOT_FOUND` - No poop found
- `POOP_CLEANED` - Poop removed
- `COPROPHAGY_CONFIRMED` - Coprophagy detected

## MQTT Topics

```python
# Input
frigate/events                    # Frigate detection events

# Output
home/ellie/state                  # Current state
home/ellie/poop_present           # Poop presence (retained)
home/ellie/poop_event             # Poop detection event
home/ellie/coprophagy_risk        # Risk alert
home/ellie/coprophagy_event       # Coprophagy confirmation
home/ellie/health                 # Health status
home/ellie/debug/log              # Debug logs (if enabled)
```

## Environment Variables

### Required
- `MQTT_HOST` - MQTT broker host
- `FRIGATE_BASE_URL` - Frigate API URL
- `CAMERA_NAME` - Camera to monitor
- `TOILET_ZONE` - Zone to monitor

### Optional (with defaults)
- `MQTT_PORT=1883`
- `MQTT_USER=""` - MQTT username
- `MQTT_PASS=""` - MQTT password
- `SQUAT_SCORE_THRESH=0.65` - Squat detection threshold
- `SQUAT_MIN_DURATION_S=5.0` - Minimum squat duration
- `RESIDUE_CONFIRM_WINDOW_S=20` - Confirmation window
- `ENABLE_DEBUG_WATCHER=false` - Enable debug logging

See `settings.py` for complete list.

## Testing

### Unit Test Example
```python
from domain.heuristics import score_squat

def test_squat_detection():
    score = score_squat(
        ratio=0.5,
        stationary=True,
        speed=0.1,
        motionless_count=5,
        squat_thresh=0.65,
        motionless_min=3,
        speed_max_still=0.15,
    )
    assert score.is_squatting == True
```

### Integration Test Example
```python
from adapters.clock import MockClock
from domain.services import EllieWatcherService

def test_service():
    clock = MockClock()
    service = EllieWatcherService(
        settings=test_settings,
        clock=clock,
        cv_ops=mock_cv_ops,
        frigate_client=mock_frigate,
        mqtt_client=mock_mqtt,
    )
    
    # Test workflow
    detection = DogDetection(...)
    service.handle_dog_detection(detection)
    
    # Verify behavior
    assert service.state.in_squat == True
```

## Common Tasks

### Add New Heuristic
1. Add pure function to `domain/heuristics.py`
2. Add unit tests
3. Use in `domain/services.py`

### Add New Adapter
1. Create new file in `adapters/`
2. Define interface/class
3. Inject in `app/runner.py`
4. Use in `domain/services.py`

### Add New State
1. Add to `EllieState` enum in `domain/models.py`
2. Add transition in `domain/fsm.py`
3. Handle in `domain/services.py`

### Debug Issues
1. Enable debug mode: `ENABLE_DEBUG_WATCHER=true`
2. Monitor MQTT topic: `home/ellie/debug/log`
3. Check logs for state transitions
4. Verify environment variables

## Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### Configuration Errors
```python
# Validate settings
python -c "from settings import load_settings; load_settings()"
```

### MQTT Connection Issues
```bash
# Test MQTT connection
mosquitto_sub -h $MQTT_HOST -p $MQTT_PORT -t "home/ellie/#" -v
```

### Frigate Connection Issues
```bash
# Test Frigate API
curl $FRIGATE_BASE_URL/api/version
```

## Performance Tips

1. **Adjust snapshot FPS:** Lower `SNAPSHOT_FPS` for slower systems
2. **Increase check interval:** Higher `CHECK_INTERVAL_S` reduces load
3. **Optimize ROI size:** Smaller ROIs process faster
4. **Monitor thread count:** Each poop spawns 2 monitoring threads

## Documentation

- `README.md` - Overview and getting started
- `ARCHITECTURE.md` - Detailed architecture documentation
- `MIGRATION.md` - Migration guide from original code
- `QUICK_REFERENCE.md` - This file

## Support

For issues or questions:
1. Check documentation
2. Enable debug mode
3. Review MQTT messages
4. Check environment variables
5. Verify Frigate connectivity

## Version History

- **v2.0** - Layered architecture refactor
- **v1.0** - Original monolithic implementation (see `app_legacy.py`)

