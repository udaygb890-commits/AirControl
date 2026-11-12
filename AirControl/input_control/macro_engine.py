"""
AIRCONTROL — DESKTOP MACRO ENGINE MODULE (STAGE 3)
Executes system-level Windows automation commands, media controls,
window snapping, and asynchronous cybernetic audio feedback.
"""

import logging
import threading
import time
from typing import Callable, Optional

import pyautogui

logger = logging.getLogger("AirControl.MacroEngine")

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class DesktopMacroEngine:
    """
    Executes Windows 11 system commands (volume, media, window management)
    with debouncing cooldowns and non-blocking futuristic audio feedback.
    """

    def __init__(
        self,
        audio_enabled: bool = True,
        cooldown_sec: float = 0.80,
        on_action: Optional[Callable[[str, str], None]] = None,
    ):
        self.audio_enabled = audio_enabled
        self.cooldown_sec = cooldown_sec
        self.on_action = on_action

        self._last_trigger_time: float = 0.0
        self._lock = threading.Lock()

    # --------------------------------------------------------------------------
    # ASYNCHRONOUS CYBERNETIC AUDIO FEEDBACK
    # --------------------------------------------------------------------------
    def play_sound(self, sound_type: str) -> None:
        """Play short futuristic synth blips on background thread without blocking."""
        if not self.audio_enabled or not HAS_WINSOUND:
            return

        def _sound_worker():
            try:
                if sound_type == "click":
                    winsound.Beep(1800, 20)
                elif sound_type == "right_click":
                    winsound.Beep(1400, 18)
                    winsound.Beep(2200, 22)
                elif sound_type == "clutch":
                    winsound.Beep(650, 35)
                elif sound_type == "macro":
                    winsound.Beep(2000, 25)
                    winsound.Beep(2600, 30)
                elif sound_type == "scroll":
                    winsound.Beep(1200, 10)
                elif sound_type == "toggle_on":
                    winsound.Beep(1500, 25)
                    winsound.Beep(2400, 35)
                elif sound_type == "toggle_off":
                    winsound.Beep(2000, 25)
                    winsound.Beep(1100, 35)
            except Exception as e:
                logger.debug("Audio playback exception: %s", e)

        threading.Thread(target=_sound_worker, daemon=True, name="AirControl-Audio").start()

    # --------------------------------------------------------------------------
    # DISPATCH SYSTEM COMMANDS WITH DEBOUNCING
    # --------------------------------------------------------------------------
    def trigger(self, action_name: str, force: bool = False) -> bool:
        """
        Execute an automation action if the cooldown window has elapsed.

        Args:
            action_name: Identifier for the macro action.
            force: Bypass cooldown timer if True (e.g. for volume steps).

        Returns:
            True if action was executed, False if suppressed by cooldown.
        """
        now = time.perf_counter()
        with self._lock:
            if not force and (now - self._last_trigger_time < self.cooldown_sec):
                return False
            self._last_trigger_time = now

        executed = False
        desc = ""

        try:
            if action_name == "SHOW_DESKTOP":
                pyautogui.hotkey("win", "d")
                desc = "Show Desktop (Win+D)"
                executed = True

            elif action_name == "TASK_VIEW":
                pyautogui.hotkey("win", "tab")
                desc = "Task View (Win+Tab)"
                executed = True

            elif action_name == "VOLUME_UP":
                pyautogui.press("volumeup")
                pyautogui.press("volumeup")
                desc = "Volume Up (+4%)"
                executed = True

            elif action_name == "VOLUME_DOWN":
                pyautogui.press("volumedown")
                pyautogui.press("volumedown")
                desc = "Volume Down (-4%)"
                executed = True

            elif action_name == "VOLUME_MUTE":
                pyautogui.press("volumemute")
                desc = "Toggle Volume Mute"
                executed = True

            elif action_name == "PLAY_PAUSE":
                pyautogui.press("playpause")
                desc = "Media Play/Pause"
                executed = True

            elif action_name == "NEXT_TRACK":
                pyautogui.press("nexttrack")
                desc = "Media Next Track"
                executed = True

            elif action_name == "PREV_TRACK":
                pyautogui.press("prevtrack")
                desc = "Media Previous Track"
                executed = True

            elif action_name == "SNAP_LEFT":
                pyautogui.hotkey("win", "left")
                desc = "Snap Window Left"
                executed = True

            elif action_name == "SNAP_RIGHT":
                pyautogui.hotkey("win", "right")
                desc = "Snap Window Right"
                executed = True

        except Exception as e:
            logger.error("Error executing macro action %s: %s", action_name, e)
            return False

        if executed:
            self.play_sound("macro")
            logger.info("Macro executed: %s (%s)", action_name, desc)
            if self.on_action:
                try:
                    self.on_action(action_name, desc)
                except Exception as e:
                    logger.debug("on_action callback error: %s", e)

        return executed

