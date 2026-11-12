"""
AIRCONTROL — FPS COUNTER UTILITY
High-precision, thread-safe frame-rate and latency calculator with smooth windowing.
"""

import threading
import time
from collections import deque


class FPSCounter:
    """Thread-safe FPS and latency calculator with moving average smoothing."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._timestamps = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._last_time = time.perf_counter()
        self._current_fps = 0.0
        self._latency_ms = 0.0

    def tick(self) -> float:
        """Record a newly processed frame. Returns the current smoothed FPS."""
        now = time.perf_counter()
        with self._lock:
            dt = now - self._last_time
            self._last_time = now
            self._timestamps.append(now)

            if dt > 0:
                self._latency_ms = dt * 1000.0

            if len(self._timestamps) > 1:
                total_duration = self._timestamps[-1] - self._timestamps[0]
                if total_duration > 0:
                    self._current_fps = (len(self._timestamps) - 1) / total_duration
                else:
                    self._current_fps = 0.0
            else:
                self._current_fps = 0.0

            return self._current_fps

    @property
    def fps(self) -> float:
        """Get current smoothed frames per second."""
        with self._lock:
            # If no frame for more than 1.5 seconds, consider FPS 0
            if time.perf_counter() - self._last_time > 1.5:
                return 0.0
            return self._current_fps

    @property
    def latency_ms(self) -> float:
        """Get the latest inter-frame latency in milliseconds."""
        with self._lock:
            return self._latency_ms

    def reset(self) -> None:
        """Reset internal frame timings."""
        with self._lock:
            self._timestamps.clear()
            self._last_time = time.perf_counter()
            self._current_fps = 0.0
            self._latency_ms = 0.0

