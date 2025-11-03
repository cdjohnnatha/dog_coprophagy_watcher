# Ellie Watcher Documentation

Welcome to the Ellie Watcher documentation! This folder contains comprehensive guides for understanding, using, and extending the application.

## 📚 Documentation Index

### Getting Started

- **[Main README](../README.md)** - Project overview, installation, and basic usage
- **[Project Structure](PROJECT_STRUCTURE.md)** - Visual project layout and file organization
- **[Quick Reference](QUICK_REFERENCE.md)** - Developer quick reference for common tasks and APIs

### Architecture & Design

- **[Architecture Guide](ARCHITECTURE.md)** - Detailed architecture documentation
  - Layer responsibilities
  - Data flow diagrams
  - Component interactions
  - Extension points
  - Testing strategies

### Migration & Deployment

- **[Migration Guide](MIGRATION.md)** - Complete migration guide from monolithic to layered architecture
  - Step-by-step migration process
  - Code mapping (old → new)
  - Backwards compatibility
  - Troubleshooting
  - Rollback procedures

## 📖 Documentation Overview

### Architecture Guide (`ARCHITECTURE.md`)

**What you'll find:**
- Complete system architecture with visual diagrams
- Layer-by-layer breakdown of responsibilities
- Data flow through the application
- Testing strategies for each layer
- Performance considerations
- Security guidelines
- Future enhancement ideas

**Best for:**
- Understanding the overall system design
- Learning how components interact
- Planning new features
- Architectural decisions

### Migration Guide (`MIGRATION.md`)

**What you'll find:**
- Detailed comparison: before vs after
- Step-by-step migration instructions
- Code mapping from old to new structure
- Docker/container update instructions
- Testing procedures
- Troubleshooting common issues
- Rollback plan

**Best for:**
- Migrating from the original monolithic code
- Understanding what changed and why
- Deployment planning
- Troubleshooting migration issues

### Quick Reference (`QUICK_REFERENCE.md`)

**What you'll find:**
- Project structure at a glance
- Key classes and functions
- Common code patterns
- Environment variables
- MQTT topics
- Testing examples
- Common tasks (how-tos)
- Troubleshooting tips

**Best for:**
- Day-to-day development
- Quick API lookups
- Common task recipes
- Debugging reference

## 🎯 Quick Navigation

### I want to...

**Understand the project structure**
→ Start with [Project Structure](PROJECT_STRUCTURE.md)

**Understand the architecture**
→ Read [Architecture Guide](ARCHITECTURE.md)

**Migrate from old code**
→ Follow the [Migration Guide](MIGRATION.md)

**Look up an API or function**
→ Check [Quick Reference](QUICK_REFERENCE.md)

**Get started quickly**
→ See [Main README](../README.md)

**Find where code lives**
→ See [Project Structure](PROJECT_STRUCTURE.md) → Navigation Tips

**Add a new feature**
→ Read [Architecture Guide](ARCHITECTURE.md) → Extension Points

**Debug an issue**
→ Check [Quick Reference](QUICK_REFERENCE.md) → Troubleshooting

**Write tests**
→ See [Architecture Guide](ARCHITECTURE.md) → Testing Strategy

**Deploy to production**
→ Follow [Migration Guide](MIGRATION.md) → Migration Steps

## 📋 Additional Resources

### Source Code Documentation

The codebase itself contains extensive inline documentation:

```python
# All modules have docstrings
from domain.heuristics import score_squat
help(score_squat)  # Shows detailed documentation

# Classes and methods are documented
from domain.services import EllieWatcherService
help(EllieWatcherService)
```

### Configuration

- **Environment Variables**: See `settings.py` for complete list with types and defaults
- **MQTT Topics**: Documented in [Quick Reference](QUICK_REFERENCE.md)
- **State Machine**: States and transitions in [Architecture Guide](ARCHITECTURE.md)

### Examples

#### Testing Example
```python
# See ARCHITECTURE.md → Testing Strategy
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

#### Extension Example
```python
# See ARCHITECTURE.md → Extension Points
class MyCustomAdapter:
    def __init__(self, settings):
        self.settings = settings
    
    def do_something(self):
        # Your implementation
        pass

# Wire in app/runner.py
my_adapter = MyCustomAdapter(settings)
service = EllieWatcherService(..., my_adapter=my_adapter)
```

## 🔄 Documentation Updates

This documentation is maintained alongside the code. When making changes:

1. **Code changes** → Update relevant docs
2. **New features** → Add to Quick Reference
3. **Architecture changes** → Update Architecture Guide
4. **Breaking changes** → Update Migration Guide

## 📞 Support

If you can't find what you're looking for:

1. Check the relevant documentation file
2. Search the source code for examples
3. Enable debug mode: `ENABLE_DEBUG_WATCHER=true`
4. Review MQTT messages for runtime behavior

## 📝 Document Versions

- **v2.0** - Layered architecture documentation (current)
- **v1.0** - Original monolithic implementation (see `app_legacy.py`)

---

**Last Updated:** 2025-11-03  
**Documentation Version:** 2.0  
**Code Version:** 2.0 (Layered Architecture)

