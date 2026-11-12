"""
AIRCONTROL — DIAGNOSTICS MODULE (STEP 5)
High-precision instrumentation and timing diagnostics for worker threads and UI loop.
Allows real-time profiling of camera capture, hand tracking, gesture recognition,
desktop automation, and UI update latency.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Generator, Optional

from config import ENABLE_DIAGNOSTICS

logger = logging.getLogger("AirControl.Diagnostics")


class OperationTimer:
    """Lightweight timing context manager for measuring duration of critical operations."""

    def __init__(self, operation_name: str, threshold_ms: float = 0.0):
        self.operation_name = operation_name
        self.threshold_ms = threshold_ms
        self.start_time: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> OperationTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        if ENABLE_DIAGNOSTICS and self.duration_ms >= self.threshold_ms:
            thread_name = threading.current_thread().name
            timestamp = time.strftime("%H:%M:%S")
            logger.info(
                "[DIAGNOSTICS %s] [%s] %s took %.2f ms",
                timestamp,
                thread_name,
                self.operation_name,
                self.duration_ms,
            )


@contextmanager
def measure_time(operation_name: str, threshold_ms: float = 0.0) -> Generator[OperationTimer, None, None]:
    """Context manager for timing code blocks."""
    timer = OperationTimer(operation_name, threshold_ms)
    with timer:
        yield timer


def log_worker_status(workers_dict: dict[str, Optional[threading.Thread]]) -> None:
    """Log health and status of all worker threads."""
    if not ENABLE_DIAGNOSTICS:
        return
    status_parts = []
    for name, thread in workers_dict.items():
        if thread is None:
            status_parts.append(f"{name}: NONE")
        elif thread.is_alive():
            status_parts.append(f"{name}: ALIVE (id={thread.ident})")
        else:
            status_parts.append(f"{name}: DEAD")
    logger.info("[DIAGNOSTICS WORKERS] %s", " | ".join(status_parts))

