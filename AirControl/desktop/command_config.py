"""
AIRCONTROL — COMMAND CONFIGURATION MODULE (STAGE 4)
Defines desktop automation commands, gesture-to-command mappings,
safety cooldowns, and confidence thresholds for Windows desktop control.
"""

from enum import Enum
from typing import Dict

from config import (
    COMMAND_CONFIRMATION_HOLD_SEC,
    DEFAULT_AUTOMATION_ENABLED,
    DEFAULT_AUTOMATION_SOUND,
    GLOBAL_COMMAND_COOLDOWN_SEC,
    MIN_COMMAND_CONFIDENCE,
    WINDOW_CLOSE_SAFETY_COOLDOWN,
)


class DesktopCommand(str, Enum):
    """Catalog of supported Windows desktop automation commands."""
    MINIMIZE_ACTIVE_WINDOW = "MINIMIZE_ACTIVE_WINDOW"
    CLOSE_ACTIVE_WINDOW = "CLOSE_ACTIVE_WINDOW"
    TOGGLE_MAXIMIZE_RESTORE = "TOGGLE_MAXIMIZE_RESTORE"
    SWITCH_WINDOWS = "SWITCH_WINDOWS"
    SHOW_DESKTOP = "SHOW_DESKTOP"
    PREVIOUS_DESKTOP = "PREVIOUS_DESKTOP"
    NEXT_DESKTOP = "NEXT_DESKTOP"
    VOLUME_UP = "VOLUME_UP"
    VOLUME_DOWN = "VOLUME_DOWN"
    VOLUME_MUTE = "VOLUME_MUTE"
    SCREENSHOT = "SCREENSHOT"
    TASK_VIEW = "TASK_VIEW"
    PLAY_PAUSE = "PLAY_PAUSE"
    NEXT_TRACK = "NEXT_TRACK"
    PREV_TRACK = "PREV_TRACK"
    NONE = "NONE"


# ------------------------------------------------------------------------------
# GESTURE TO DESKTOP COMMAND MAPPINGS
# Matches Stage 4 specifications:
# 1. THUMBS_DOWN     -> MINIMIZE_ACTIVE_WINDOW
# 2. TWO_HANDS_X     -> CLOSE_ACTIVE_WINDOW
# 3. THUMBS_UP       -> TOGGLE_MAXIMIZE_RESTORE
# 4. OK_SIGN/PINCH_OK-> SWITCH_WINDOWS
# 5. OPEN_PALM_HELD  -> SHOW_DESKTOP
# 6. SWIPE_LEFT      -> PREVIOUS_DESKTOP
# 7. SWIPE_RIGHT     -> NEXT_DESKTOP
# 8. SWIPE_UP        -> VOLUME_UP
# 9. SWIPE_DOWN      -> VOLUME_DOWN
# 10. ROCK_ON        -> SCREENSHOT
# 11. WAVE           -> TASK_VIEW
# 12. CALL_ME        -> PLAY_PAUSE
# ------------------------------------------------------------------------------
DEFAULT_GESTURE_COMMAND_MAP: Dict[str, DesktopCommand] = {
    # 1. Thumbs Down -> Minimize Active Window
    "THUMBS_DOWN": DesktopCommand.MINIMIZE_ACTIVE_WINDOW,

    # 2. Two Hands X -> Close Active Window (Safety Cooldown 1.5s)
    "TWO_HANDS_X": DesktopCommand.CLOSE_ACTIVE_WINDOW,

    # 3. Thumbs Up -> Toggle Maximize / Restore
    "THUMBS_UP": DesktopCommand.TOGGLE_MAXIMIZE_RESTORE,

    # 4. OK Sign (Pinch OK) -> Switch Windows (Alt+Tab)
    "PINCH_OK": DesktopCommand.SWITCH_WINDOWS,
    "OK_SIGN": DesktopCommand.SWITCH_WINDOWS,

    # 5. Open Palm Held -> Show Desktop (Win+D)
    "OPEN_PALM": DesktopCommand.SHOW_DESKTOP,
    "OPEN_PALM_HELD": DesktopCommand.SHOW_DESKTOP,

    # 6. Swipes -> Virtual Desktop & System Navigation
    "SWIPE_LEFT": DesktopCommand.PREVIOUS_DESKTOP,
    "SWIPE_RIGHT": DesktopCommand.NEXT_DESKTOP,
    "SWIPE_UP": DesktopCommand.VOLUME_UP,
    "SWIPE_DOWN": DesktopCommand.VOLUME_DOWN,

    # Additional Safe Controls
    "ROCK_ON": DesktopCommand.SCREENSHOT,
    "WAVE": DesktopCommand.TASK_VIEW,
    "CALL_ME": DesktopCommand.PLAY_PAUSE,
}


# Per-command cooldown timers in seconds to prevent accidental rapid re-triggering
COMMAND_COOLDOWNS: Dict[DesktopCommand, float] = {
    DesktopCommand.CLOSE_ACTIVE_WINDOW: WINDOW_CLOSE_SAFETY_COOLDOWN,  # 1.5s safety buffer
    DesktopCommand.MINIMIZE_ACTIVE_WINDOW: 0.80,
    DesktopCommand.TOGGLE_MAXIMIZE_RESTORE: 0.80,
    DesktopCommand.SWITCH_WINDOWS: 0.60,
    DesktopCommand.SHOW_DESKTOP: 0.90,
    DesktopCommand.PREVIOUS_DESKTOP: 0.45,
    DesktopCommand.NEXT_DESKTOP: 0.45,
    DesktopCommand.VOLUME_UP: 0.12,                                    # Responsive volume stepping
    DesktopCommand.VOLUME_DOWN: 0.12,
    DesktopCommand.VOLUME_MUTE: 0.80,
    DesktopCommand.SCREENSHOT: 1.20,
    DesktopCommand.TASK_VIEW: 0.80,
    DesktopCommand.PLAY_PAUSE: 0.70,
    DesktopCommand.NEXT_TRACK: 0.50,
    DesktopCommand.PREV_TRACK: 0.50,
}

# Gestures that strictly require sustained holding before triggering (prevents accidental trigger on transit)
HELD_REQUIRED_GESTURES = {
    "OPEN_PALM": COMMAND_CONFIRMATION_HOLD_SEC,
}

# Human-readable tactical descriptions for dashboard telemetry & logs
COMMAND_DESCRIPTIONS: Dict[DesktopCommand, str] = {
    DesktopCommand.MINIMIZE_ACTIVE_WINDOW: "Minimized active window",
    DesktopCommand.CLOSE_ACTIVE_WINDOW: "Closed active window (Alt+F4)",
    DesktopCommand.TOGGLE_MAXIMIZE_RESTORE: "Toggled window maximize / restore state",
    DesktopCommand.SWITCH_WINDOWS: "Switched active application window (Alt+Tab)",
    DesktopCommand.SHOW_DESKTOP: "Toggled Show Desktop (Win+D)",
    DesktopCommand.PREVIOUS_DESKTOP: "Switched to previous virtual desktop (Ctrl+Win+Left)",
    DesktopCommand.NEXT_DESKTOP: "Switched to next virtual desktop (Ctrl+Win+Right)",
    DesktopCommand.VOLUME_UP: "Increased system audio volume (+4%)",
    DesktopCommand.VOLUME_DOWN: "Decreased system audio volume (-4%)",
    DesktopCommand.VOLUME_MUTE: "Toggled audio mute state",
    DesktopCommand.SCREENSHOT: "Triggered Windows screen clip capture (Win+Shift+S)",
    DesktopCommand.TASK_VIEW: "Opened Windows Task View overview (Win+Tab)",
    DesktopCommand.PLAY_PAUSE: "Toggled media playback Play/Pause",
    DesktopCommand.NEXT_TRACK: "Skipped to next media track",
    DesktopCommand.PREV_TRACK: "Skipped to previous media track",
    DesktopCommand.NONE: "No action",
}

