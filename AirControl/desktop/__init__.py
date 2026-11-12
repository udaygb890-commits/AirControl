"""
AIRCONTROL — DESKTOP AUTOMATION SUBSYSTEM (STAGE 4)
Exports DesktopController, WindowManager, SystemControls, CommandDispatcher,
and DesktopCommand definitions.
"""

from desktop.command_config import (
    COMMAND_COOLDOWNS,
    COMMAND_DESCRIPTIONS,
    DEFAULT_GESTURE_COMMAND_MAP,
    DesktopCommand,
)
from desktop.command_dispatcher import (
    CommandDispatcher,
    CommandResult,
)
from desktop.desktop_controller import DesktopController
from desktop.system_controls import SystemControls
from desktop.window_manager import WindowManager

__all__ = [
    "DesktopCommand",
    "COMMAND_COOLDOWNS",
    "COMMAND_DESCRIPTIONS",
    "DEFAULT_GESTURE_COMMAND_MAP",
    "CommandDispatcher",
    "CommandResult",
    "WindowManager",
    "SystemControls",
    "DesktopController",
]

