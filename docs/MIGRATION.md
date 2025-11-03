# Migration Guide: From Monolithic to Layered Architecture

## Overview

The original `app.py` (772 lines) has been refactored into a clean, multi-layered architecture following best practices. This guide explains the changes and how to migrate.

## What Changed?

### Before (Monolithic)
```
app.py (772 lines)
├── Environment variables scattered throughout
├── Helper functions mixed with business logic
├── MQTT callbacks with embedded logic
├── OpenCV operations inline
├── State management global
└── Difficult to test, maintain, extend
```

### After (Layered)
```
dog_coprophagy_watcher/
├── settings.py              # Centralized config
├── domain/                  # Pure business logic
│   ├── models.py
│   ├── heuristics.py
│   ├── fsm.py
│   └── services.py
├── adapters/               # I/O operations
│   ├── clock.py
│   ├── cv_ops.py
│   ├── frigate_client.py
│   └── mqtt_client.py
├── app/                    # Application wiring
│   ├── handlers.py
│   └── runner.py
└── main.py / __main__.py   # Entry points
```

## Migration Steps

### 1. Install Dependencies

```bash
cd addons/dog_coprophagy_watcher
pip install -r requirements.txt
```

### 2. Environment Variables

No changes needed! All environment variables work exactly as before. They're now centralized in `settings.py` with type validation via Pydantic.

### 3. Running the Application

**Option A: Module execution (recommended)**
```bash
python -m dog_coprophagy_watcher
```

**Option B: Direct execution**
```bash
python main.py
```

**Option C: Compatibility mode (uses new architecture)**
```bash
python app.py
```

### 4. Docker/Container Updates

Update your Dockerfile or docker-compose.yml:

**Before:**
```dockerfile
CMD ["python", "app.py"]
```

**After (any of these work):**
```dockerfile
CMD ["python", "-m", "dog_coprophagy_watcher"]
# or
CMD ["python", "main.py"]
# or (compatibility)
CMD ["python", "app.py"]
```

## Key Improvements

### 1. **Testability**

**Before:** Hard to test due to global state and mixed concerns
```python
# Difficult to test - requires full MQTT/Frigate setup
def on_message(client, userdata, msg):
    # 100+ lines of mixed logic
    ...
```

**After:** Pure functions and dependency injection
```python
# Easy to test - pure function
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

### 2. **Maintainability**

**Before:** Find where squat scoring happens? Search through 772 lines
**After:** `domain/heuristics.py` - clear, documented, single responsibility

### 3. **Extensibility**

**Before:** Want to support a different NVR? Rewrite everything
**After:** Create new adapter implementing same interface

```python
class MyNVRClient:
    def fetch_snapshot(self) -> np.ndarray:
        # Your implementation
        pass
    
    def find_nearby_dog_event(self, timestamp: float) -> Optional[str]:
        # Your implementation
        pass
```

### 4. **Type Safety**

**Before:** Runtime errors from typos or wrong types
```python
SQUAT_THR = float(os.getenv("SQUAT_SCORE_THRESH", "0.65"))  # Easy to mistype
```

**After:** Validated at startup with clear errors
```python
class Settings(BaseSettings):
    squat_score_thresh: float = Field(default=0.65, alias="SQUAT_SCORE_THRESH")
    # Pydantic validates type and provides clear error messages
```

## Code Mapping

Here's where original code moved to:

| Original Location | New Location | Notes |
|------------------|--------------|-------|
| Environment vars | `settings.py` | Now validated with Pydantic |
| `TrackState` dataclass | `domain/models.py` | Enhanced with more models |
| Squat scoring logic | `domain/heuristics.py::score_squat()` | Pure function |
| `residue_blob()` | `adapters/cv_ops.py::diff_blob()` | Cleaner interface |
| `fetch_snapshot()` | `adapters/frigate_client.py::fetch_snapshot()` | Part of client |
| `on_message()` | `app/handlers.py::on_message()` | Separated concerns |
| State machine logic | `domain/fsm.py` | Explicit FSM |
| Main loop | `app/runner.py::run()` | Clean bootstrap |

## Backwards Compatibility

The original `app.py` now imports and runs the new architecture:

```python
# app.py (new)
from app.runner import run

if __name__ == "__main__":
    run()
```

This means:
- ✅ Existing Docker containers work without changes
- ✅ Existing scripts work without changes
- ✅ Environment variables work without changes
- ✅ MQTT topics unchanged
- ✅ Behavior identical

## Testing the Migration

### 1. Verify Environment
```bash
# Check all required packages installed
pip list | grep -E "pydantic|opencv|paho-mqtt|requests|numpy"
```

### 2. Dry Run
```bash
# Test configuration loading
python -c "from settings import load_settings; s = load_settings(); print(f'Camera: {s.camera_name}')"
```

### 3. Full Run
```bash
# Run with debug enabled
export ENABLE_DEBUG_WATCHER=true
python -m dog_coprophagy_watcher
```

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'pydantic_settings'`

**Solution:**
```bash
pip install pydantic-settings
```

### Configuration Errors

**Error:** `ValidationError` on startup

**Solution:** Check environment variables match expected types. Pydantic will tell you exactly what's wrong.

### Behavior Differences

If you notice any behavioral differences from the original:
1. Check environment variables are set correctly
2. Enable debug mode: `ENABLE_DEBUG_WATCHER=true`
3. Compare MQTT messages with original
4. Report issues with specific examples

## Rollback Plan

If you need to rollback:

1. The original code is preserved in `app_legacy.py`
2. Copy it back: `cp app_legacy.py app.py`
3. Or use git: `git checkout app.py`

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Lines of code** | 772 in one file | ~2000 across organized modules |
| **Testability** | Difficult | Easy (pure functions) |
| **Maintainability** | Hard to navigate | Clear structure |
| **Type safety** | Runtime errors | Compile-time validation |
| **Extensibility** | Requires rewrites | Plugin architecture |
| **Documentation** | Inline comments | Docstrings + README |
| **Debugging** | Global state | Isolated components |

## Next Steps

1. ✅ Install dependencies
2. ✅ Test with existing environment
3. ✅ Monitor for 24-48 hours
4. ✅ Write unit tests for critical paths
5. ✅ Consider adding integration tests
6. ✅ Update deployment documentation

## Questions?

See `README.md` for architecture details and examples.

## Original Code

The original monolithic `app.py` is preserved in `app_legacy.py` for reference.

