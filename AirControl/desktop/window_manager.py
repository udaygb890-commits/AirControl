"""
AIRCONTROL — WINDOW MANAGER MODULE (STAGE 4)
Handles Windows OS window management, active window manipulation (minimize,
maximize, restore, close), application switching, and virtual desktop navigation.
"""

import ctypes
import logging
import sys
from typing import Optional

import pyautogui

logger = logging.getLogger("AirControl.WindowManager")

# Windows Win32 API handles
HAS_WIN32 = sys.platform == "win32"
if HAS_WIN32:
    try:
        user32 = ctypes.windll.user32
        SW_MAXIMIZE = 3
        SW_MINIMIZE = 6
        SW_RESTORE = 9
        WM_CLOSE = 0x0010
    except Exception as e:
        logger.debug("Win32 ctypes binding failed: %s", e)
        HAS_WIN32 = False


class WindowManager:
    """
    Executes Windows 11 window management actions safely via PyAutoGUI and Win32 API.
    """

    def __init__(self):
        # Configure PyAutoGUI for zero-latency automation
        pyautogui.PAUSE = 0.0
        pyautogui.FAILSAFE = False

    def get_active_window_title(self) -> str:
        """Retrieve the title bar text of the currently focused window."""
        if not HAS_WIN32:
            return "Active Window"
        try:
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return "Desktop"
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return "Active Window"
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value or "Active Window"
        except Exception as e:
            logger.debug("Error getting window title: %s", e)
            return "Active Window"

    def is_active_window_maximized(self) -> bool:
        """Check if current foreground window is currently maximized."""
        if not HAS_WIN32:
            return False
        try:
            hwnd = user32.GetForegroundWindow()
            if hwnd:
                return bool(user32.IsZoomed(hwnd))
        except Exception:
            pass
        return False

    def minimize_active_window(self) -> bool:
        """
        Minimize currently focused active window.
        Triggered by: THUMBS_DOWN
        """
        title = self.get_active_window_title()
        logger.info("Minimizing active window: '%s'", title)
        try:
            if HAS_WIN32:
                hwnd = user32.GetForegroundWindow()
                if hwnd:
                    user32.ShowWindow(hwnd, SW_MINIMIZE)
                    return True
            pyautogui.hotkey("win", "down")
            return True
        except Exception as e:
            logger.error("Failed to minimize active window: %s", e)
            return False

    def toggle_maximize_restore(self) -> bool:
        """
        Toggle maximize/restore state of the active window.
        Triggered by: THUMBS_UP
        """
        title = self.get_active_window_title()
        try:
            if HAS_WIN32:
                hwnd = user32.GetForegroundWindow()
                if hwnd:
                    if user32.IsZoomed(hwnd):
                        logger.info("Restoring active window: '%s'", title)
                        user32.ShowWindow(hwnd, SW_RESTORE)
                    else:
                        logger.info("Maximizing active window: '%s'", title)
                        user32.ShowWindow(hwnd, SW_MAXIMIZE)
                    return True

            # PyAutoGUI fallback: Win+Up maximizes
            pyautogui.hotkey("win", "up")
            return True
        except Exception as e:
            logger.error("Failed to toggle maximize/restore: %s", e)
            return False

    def close_active_window(self) -> bool:
        """
        Gracefully close the current foreground window.
        Triggered by: TWO_HANDS_X
        """
        title = self.get_active_window_title()
        logger.info("Closing active window: '%s'", title)
        try:
            pyautogui.hotkey("alt", "f4")
            return True
        except Exception as e:
            logger.error("Failed to close active window: %s", e)
            return False

    def switch_windows(self) -> bool:
        """
        Quickly cycle to the next active application window.
        Triggered by: OK_SIGN / PINCH_OK
        """
        logger.info("Switching application window (Alt+Tab)")
        try:
            pyautogui.hotkey("alt", "tab")
            return True
        except Exception as e:
            logger.error("Failed to switch windows: %s", e)
            return False

    def show_desktop(self) -> bool:
        """
        Toggle Show Desktop (minimize/restore all open windows).
        Triggered by: OPEN_PALM_HELD
        """
        logger.info("Toggling Show Desktop (Win+D)")
        try:
            pyautogui.hotkey("win", "d")
            return True
        except Exception as e:
            logger.error("Failed to toggle show desktop: %s", e)
            return False

    def switch_virtual_desktop(self, direction: str = "next") -> bool:
        """
        Switch to adjacent Windows 11 virtual desktop.
        Triggered by: SWIPE_LEFT (prev), SWIPE_RIGHT (next)
        """
        try:
            if direction in ("prev", "previous", "left"):
                logger.info("Switching to previous virtual desktop (Ctrl+Win+Left)")
                pyautogui.hotkey("ctrl", "win", "left")
            else:
                logger.info("Switching to next virtual desktop (Ctrl+Win+Right)")
                pyautogui.hotkey("ctrl", "win", "right")
            return True
        except Exception as e:
            logger.error("Failed to switch virtual desktop: %s", e)
            return False

    def open_task_view(self) -> bool:
        """
        Open Windows 11 Task View overview.
        Triggered by: WAVE
        """
        logger.info("Opening Windows Task View (Win+Tab)")
        try:
            pyautogui.hotkey("win", "tab")
            return True
        except Exception as e:
            logger.error("Failed to open Task View: %s", e)
            return False

