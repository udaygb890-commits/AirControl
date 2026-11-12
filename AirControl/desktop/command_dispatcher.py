"""
AIRCONTROL — COMMAND DISPATCHER MODULE (STAGE 4)
Maps incoming gesture events to desktop actions, validates safety thresholds,
enforces debouncing cooldowns, and dispatches to WindowManager or SystemControls.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from config import (
    GLOBAL_COMMAND_COOLDOWN_SEC,
    MIN_COMMAND_CONFIDENCE,
)
from desktop.command_config import (
    COMMAND_COOLDOWNS,
    COMMAND_DESCRIPTIONS,
    DEFAULT_GESTURE_COMMAND_MAP,
    HELD_REQUIRED_GESTURES,
    DesktopCommand,
)
from desktop.system_controls import SystemControls
from desktop.window_manager import WindowManager

logger = logging.getLogger("AirControl.CommandDispatcher")


@dataclass
class CommandResult:
    """Detailed outcome of a dispatched desktop automation action."""
    success: bool
    command: DesktopCommand
    description: str
    gesture_name: str
    reason: Optional[str] = None
    timestamp: float = 0.0


class CommandDispatcher:
    """
    Coordinates gesture-to-command translation, safety checks, cooldown timers,
    and delegating actions to WindowManager and SystemControls.
    """

    def __init__(
        self,
        window_manager: Optional[WindowManager] = None,
        system_controls: Optional[SystemControls] = None,
        gesture_map: Optional[Dict[str, DesktopCommand]] = None,
        on_command_executed: Optional[Callable[[CommandResult], None]] = None,
    ):
        self.window_manager = window_manager or WindowManager()
        self.system_controls = system_controls or SystemControls()
        self.gesture_map = gesture_map or dict(DEFAULT_GESTURE_COMMAND_MAP)
        self.on_command_executed = on_command_executed

        self._lock = threading.RLock()
        self._last_global_execution: float = 0.0
        self._last_command_executions: Dict[DesktopCommand, float] = {}
        self._command_history: List[CommandResult] = []

    def dispatch(
        self,
        gesture_name: str,
        confidence: float = 1.0,
        hold_duration: float = 0.0,
        force: bool = False,
    ) -> CommandResult:
        """
        Evaluate gesture against safety rules and execute corresponding desktop action.

        Args:
            gesture_name: Identifier of static, dynamic, or dual-hand gesture.
            confidence: Recognition confidence score (0.0 - 1.0).
            hold_duration: Continuous time gesture has been maintained in seconds.
            force: Bypass cooldown checks if True.

        Returns:
            CommandResult with execution status and details.
        """
        now = time.perf_counter()

        with self._lock:
            # 1. Lookup mapped desktop command
            command = self.gesture_map.get(gesture_name, DesktopCommand.NONE)
            if command == DesktopCommand.NONE:
                return CommandResult(
                    success=False,
                    command=DesktopCommand.NONE,
                    description="No command mapped",
                    gesture_name=gesture_name,
                    reason="Unmapped gesture",
                    timestamp=now,
                )

            # 2. Check confidence threshold
            if confidence < MIN_COMMAND_CONFIDENCE and not force:
                logger.debug(
                    "Command %s rejected: confidence %.2f < %.2f",
                    command.value,
                    confidence,
                    MIN_COMMAND_CONFIDENCE,
                )
                return CommandResult(
                    success=False,
                    command=command,
                    description=COMMAND_DESCRIPTIONS.get(command, command.value),
                    gesture_name=gesture_name,
                    reason=f"Confidence {confidence:.2f} below threshold",
                    timestamp=now,
                )

            # 3. Check sustained hold duration requirement if applicable
            required_hold = HELD_REQUIRED_GESTURES.get(gesture_name, 0.0)
            if required_hold > 0.0 and hold_duration < required_hold and not force:
                return CommandResult(
                    success=False,
                    command=command,
                    description=COMMAND_DESCRIPTIONS.get(command, command.value),
                    gesture_name=gesture_name,
                    reason=f"Hold duration {hold_duration:.2f}s < required {required_hold:.2f}s",
                    timestamp=now,
                )

            # 4. Check global cooldown
            if not force and (now - self._last_global_execution < GLOBAL_COMMAND_COOLDOWN_SEC):
                return CommandResult(
                    success=False,
                    command=command,
                    description=COMMAND_DESCRIPTIONS.get(command, command.value),
                    gesture_name=gesture_name,
                    reason="Global cooldown active",
                    timestamp=now,
                )

            # 5. Check per-command specific cooldown
            cmd_cooldown = COMMAND_COOLDOWNS.get(command, 0.50)
            last_cmd_time = self._last_command_executions.get(command, 0.0)
            if not force and (now - last_cmd_time < cmd_cooldown):
                return CommandResult(
                    success=False,
                    command=command,
                    description=COMMAND_DESCRIPTIONS.get(command, command.value),
                    gesture_name=gesture_name,
                    reason=f"Command cooldown active ({cmd_cooldown:.1f}s)",
                    timestamp=now,
                )

            # 6. Execute Desktop Command
            success = self._execute_action(command)
            desc = COMMAND_DESCRIPTIONS.get(command, command.value)

            if success:
                self._last_global_execution = now
                self._last_command_executions[command] = now

            result = CommandResult(
                success=success,
                command=command,
                description=desc,
                gesture_name=gesture_name,
                reason=None if success else "Execution returned failure",
                timestamp=now,
            )

            self._command_history.append(result)
            if len(self._command_history) > 50:
                self._command_history.pop(0)

            if success:
                logger.info("Executed Desktop Action: [%s] -> %s", gesture_name, desc)
                if self.on_command_executed:
                    try:
                        self.on_command_executed(result)
                    except Exception as e:
                        logger.debug("on_command_executed callback exception: %s", e)

            return result

    def _execute_action(self, command: DesktopCommand) -> bool:
        """Route command to target WindowManager or SystemControls implementation."""
        # Window Management Commands
        if command == DesktopCommand.MINIMIZE_ACTIVE_WINDOW:
            res = self.window_manager.minimize_active_window()
            if res:
                self.system_controls.play_sound("minimize")
            return res

        elif command == DesktopCommand.CLOSE_ACTIVE_WINDOW:
            res = self.window_manager.close_active_window()
            if res:
                self.system_controls.play_sound("close")
            return res

        elif command == DesktopCommand.TOGGLE_MAXIMIZE_RESTORE:
            res = self.window_manager.toggle_maximize_restore()
            if res:
                self.system_controls.play_sound("maximize")
            return res

        elif command == DesktopCommand.SWITCH_WINDOWS:
            res = self.window_manager.switch_windows()
            if res:
                self.system_controls.play_sound("switch")
            return res

        elif command == DesktopCommand.SHOW_DESKTOP:
            res = self.window_manager.show_desktop()
            if res:
                self.system_controls.play_sound("desktop")
            return res

        elif command == DesktopCommand.PREVIOUS_DESKTOP:
            res = self.window_manager.switch_virtual_desktop("previous")
            if res:
                self.system_controls.play_sound("virtual_desktop")
            return res

        elif command == DesktopCommand.NEXT_DESKTOP:
            res = self.window_manager.switch_virtual_desktop("next")
            if res:
                self.system_controls.play_sound("virtual_desktop")
            return res

        elif command == DesktopCommand.TASK_VIEW:
            res = self.window_manager.open_task_view()
            if res:
                self.system_controls.play_sound("desktop")
            return res

        # System Hardware Commands
        elif command == DesktopCommand.VOLUME_UP:
            return self.system_controls.volume_up(steps=2)

        elif command == DesktopCommand.VOLUME_DOWN:
            return self.system_controls.volume_down(steps=2)

        elif command == DesktopCommand.VOLUME_MUTE:
            return self.system_controls.volume_mute()

        elif command == DesktopCommand.SCREENSHOT:
            return self.system_controls.capture_screenshot()

        elif command == DesktopCommand.PLAY_PAUSE:
            return self.system_controls.media_play_pause()

        elif command == DesktopCommand.NEXT_TRACK:
            return self.system_controls.media_next_track()

        elif command == DesktopCommand.PREV_TRACK:
            return self.system_controls.media_prev_track()

        return False

    def reset_cooldowns(self) -> None:
        """Clear all cooldown timers (useful for unit testing)."""
        with self._lock:
            self._last_global_execution = 0.0
            self._last_command_executions.clear()

