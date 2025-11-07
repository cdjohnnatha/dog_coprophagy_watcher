"""
Clock abstraction for time operations.
Allows easy mocking in tests.
"""
import time
from abc import ABC, abstractmethod


class Clock(ABC):
    """Abstract clock interface."""
    
    @abstractmethod
    def now(self) -> float:
        """Get current time as Unix timestamp."""
        pass
    
    @abstractmethod
    def now_iso(self) -> str:
        """Get current time as ISO 8601 string."""
        pass
    
    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """Sleep for specified seconds."""
        pass
    
    @abstractmethod
    def monotonic(self) -> float:
        """Get monotonic time (for intervals)."""
        pass


class SystemClock(Clock):
    """Real system clock implementation."""
    
    def now(self) -> float:
        """Get current time as Unix timestamp."""
        return time.time()
    
    def now_iso(self) -> str:
        """Get current time as ISO 8601 string (UTC)."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    def sleep(self, seconds: float) -> None:
        """Sleep for specified seconds."""
        time.sleep(seconds)
    
    def monotonic(self) -> float:
        """Get monotonic time (for intervals)."""
        return time.monotonic()


class MockClock(Clock):
    """Mock clock for testing."""
    
    def __init__(self, initial_time: float = 0.0):
        self._time = initial_time
        self._monotonic = 0.0
    
    def now(self) -> float:
        """Get current mock time."""
        return self._time
    
    def now_iso(self) -> str:
        """Get current mock time as ISO string."""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._time))
    
    def sleep(self, seconds: float) -> None:
        """Advance mock time."""
        self._time += seconds
        self._monotonic += seconds
    
    def monotonic(self) -> float:
        """Get monotonic mock time."""
        return self._monotonic
    
    def advance(self, seconds: float) -> None:
        """Manually advance time."""
        self._time += seconds
        self._monotonic += seconds
    
    def set_time(self, timestamp: float) -> None:
        """Set absolute time."""
        self._time = timestamp

