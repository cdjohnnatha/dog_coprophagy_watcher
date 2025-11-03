"""
Compatibility shim for app.py - Redirects to new layered architecture.

The original monolithic app.py has been refactored into a clean, layered architecture.
This file now serves as a compatibility entry point that uses the new structure.

To run directly with the new architecture:
    python -m dog_coprophagy_watcher
or
    python main.py

See README.md for architecture details and migration guide.
"""

# Import and run from the new architecture
from app.runner import run

if __name__ == "__main__":
    run()