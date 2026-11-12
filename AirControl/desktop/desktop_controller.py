"""
AIRCONTROL — DESKTOP CONTROLLER MODULE (STAGE 4)
Main coordinator for Windows desktop automation. Receives gesture events,
manages safety states, emergency halts, and dispatches actions to CommandDispatcher
via a dedicated background Automation Worker thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, List, Optional

from config import AUTOMATION_QUEUE_MAXSIZE, DEFAULT_AUTOMATION_ENABLED, WORKER_SHUTDOWN_TIMEOUT
from desktop.command_config import DesktopCommand
from desktop.command_dispatcher import CommandDispatcher, CommandResult
from desktop.system_controls import SystemControls
from desktop.window_manager import WindowManager
from gestures.gesture_engine import GestureEvent
from utils.diagnostics import measure_time

logger = logging.getLogger("AirControl.DesktopController")


class DesktopController:
    """
    Unified manager for Stage 4 Windows Desktop Automation.
    Maintains safety guards, global toggle state, and handles incoming gesture events.
    Uses a dedicated background Automation Worker thread so desktop commands
    never block the vision engine or the Tkinter UI event loop.
    """

    def __init__(
        self,
        enabled: bool = DEFAULT_AUTOMATION_ENABLED,
        window_manager: Optional[WindowManager] = None,
        system_controls: Optional[SystemControls] = None,
        dispatcher: Optional[CommandDispatcher] = None,
        on_action: Optional[Callable[[CommandResult], None]] = None,
    ):
        self._enabled = enabled
        self._lock = threading.RLock()

        self.window_manager = window_manager or WindowManager()
        self.system_controls = system_controls or SystemControls()
        self.dispatcher = dispatcher or CommandDispatcher(
            window_manager=self.window_manager,
            system_controls=self.system_controls,
            on_command_executed=self._on_internal_command_executed,
        )
        self.on_action = on_action

        # Automation Worker Thread & Bounded Queues
        self._command_queue: queue.Queue[GestureEvent] = queue.Queue(maxsize=AUTOMATION_QUEUE_MAXSIZE)
        self._result_queue: queue.Queue[CommandResult] = queue.Queue(maxsize=30)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Telemetry State
        self._last_command: DesktopCommand = DesktopCommand.NONE
        self._last_description: str = "Desktop automation ready."
        self._last_gesture_name: str = "NONE"
        self._last_execution_time: float = 0.0
        self._total_commands_executed: int = 0

    # --------------------------------------------------------------------------
    # THREAD LIFECYCLE (AUTOMATION WORKER)
    # --------------------------------------------------------------------------
    @property
    def is_worker_running(self) -> bool:
        """True if the background automation worker thread is alive."""
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def start(self) -> bool:
        """Start the background automation worker thread."""
        with self._lock:
            if self.is_worker_running:
                return True
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._automation_worker,
                name="AirControl-AutomationWorker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("Automation worker thread started.")
            return True

    def stop(self, timeout: float = WORKER_SHUTDOWN_TIMEOUT) -> None:
        """Signal automation worker to stop and wait for termination."""
        with self._lock:
            self._stop_event.set()
            thread = self._worker_thread
            self._worker_thread = None

        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Automation worker thread did not terminate within timeout.")

        # Clear any residual commands in queue
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Automation worker stopped cleanly.")

    # --------------------------------------------------------------------------
    # PROPERTIES & SAFETY GUARDS
    # --------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, val: bool) -> None:
        with self._lock:
            self._enabled = val
            logger.info("Desktop Automation master state set to: %s", "ENABLED" if val else "DISABLED")

    @property
    def total_executed(self) -> int:
        with self._lock:
            return self._total_commands_executed

    @property
    def last_command_info(self) -> str:
        with self._lock:
            if self._last_command == DesktopCommand.NONE:
                return "STANDBY (No actions triggered)"
            return f"{self._last_command.value} via [{self._last_gesture_name}]"

    @property
    def last_description(self) -> str:
        with self._lock:
            return self._last_description

    def emergency_halt(self) -> None:
        """Immediately disengage automation and cancel any pending queued commands."""
        with self._lock:
            self._enabled = False
            # Drain command queue to cancel pending actions
            while not self._command_queue.empty():
                try:
                    self._command_queue.get_nowait()
                except queue.Empty:
                    break
            self.dispatcher.reset_cooldowns()
            self._last_description = "Emergency halt engaged. Automation disabled."
            logger.warning("SAFETY HALT: Desktop Automation disabled.")

    # --------------------------------------------------------------------------
    # EVENT HANDLING & WORKER LOOP
    # --------------------------------------------------------------------------
    def handle_gesture_event(self, event: GestureEvent, force_sync: bool = False) -> Optional[CommandResult]:
        """
        Primary callback receiving confirmed gesture events from GestureEngine.
        When the Automation Worker is active, this method queues the event non-blockingly
        and returns immediately, ensuring Vision Engine and UI are never blocked.
        If force_sync is True or the worker thread is not running, executes synchronously.
        """
        with self._lock:
            if not self._enabled:
                return None

        # Asynchronous dispatch via Automation Worker
        if self.is_worker_running and not force_sync:
            if self._command_queue.full():
                try:
                    self._command_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._command_queue.put_nowait(event)
            except queue.Full:
                pass
            return None

        # Synchronous execution fallback (used in unit tests or when worker inactive)
        return self._execute_gesture_event(event)

    def _execute_gesture_event(self, event: GestureEvent) -> Optional[CommandResult]:
        """Internal execution helper: routes event to CommandDispatcher."""
        with self._lock:
            if not self._enabled:
                return None

        with measure_time(f"AUTOMATION_EXECUTE [{event.gesture_name}]", threshold_ms=1.0):
            result = self.dispatcher.dispatch(
                gesture_name=event.gesture_name,
                confidence=event.confidence,
                hold_duration=event.hold_duration,
            )
        return result

    def _automation_worker(self) -> None:
        """Dedicated background worker consuming and executing queued desktop actions."""
        logger.info("Automation worker thread loop started.")
        while not self._stop_event.is_set():
            try:
                event = self._command_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break

            with self._lock:
                if not self._enabled:
                    continue

            result = self._execute_gesture_event(event)
            if result and result.success:
                # Place in result queue for safe main-thread UI consumption
                if self._result_queue.full():
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass
                try:
                    self._result_queue.put_nowait(result)
                except queue.Full:
                    pass

        logger.info("Automation worker thread loop finished.")

    def get_pending_results(self) -> List[CommandResult]:
        """Drains the result queue and returns all completed commands for the UI thread."""
        results: List[CommandResult] = []
        while not self._result_queue.empty():
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _on_internal_command_executed(self, result: CommandResult) -> None:
        """Internal callback updating controller state when an action succeeds."""
        with self._lock:
            self._last_command = result.command
            self._last_description = result.description
            self._last_gesture_name = result.gesture_name
            self._last_execution_time = result.timestamp
            self._total_commands_executed += 1

        if self.on_action:
            try:
                self.on_action(result)
            except Exception as e:
                logger.debug("on_action callback exception: %s", e)
