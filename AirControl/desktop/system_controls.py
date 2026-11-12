"""
AIRCONTROL — SYSTEM CONTROLS MODULE (STAGE 4)
Handles Windows 11 system-level hardware controls (volume, mute, media playback,
screenshot capture) and non-blocking cybernetic audio feedback tones.
"""

import logging
import threading
from typing import Optional

import pyautogui

logger = logging.getLogger("AirControl.SystemControls")

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class SystemControls:
    """
    Executes Windows 11 system-level hardware commands with optional audio chimes.
    """

    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled = audio_enabled
        self._lock = threading.Lock()

    # --------------------------------------------------------------------------
    # ASYNCHRONOUS CYBERNETIC AUDIO FEEDBACK
    # --------------------------------------------------------------------------
    def play_sound(self, sound_type: str) -> None:
        """Play brief futuristic synth feedback on background thread."""
        if not self.audio_enabled or not HAS_WINSOUND:
            return

        def _worker():
            try:
                if sound_type == "minimize":
                    winsound.Beep(900, 30)
                    winsound.Beep(650, 40)
                elif sound_type == "maximize":
                    winsound.Beep(700, 30)
                    winsound.Beep(1100, 40)
                elif sound_type == "close":
                    winsound.Beep(1200, 35)
                    winsound.Beep(500, 50)
                elif sound_type == "switch":
                    winsound.Beep(1400, 25)
                    winsound.Beep(1800, 30)
                elif sound_type == "desktop":
                    winsound.Beep(800, 20)
                    winsound.Beep(1200, 25)
                    winsound.Beep(1600, 35)
                elif sound_type == "volume":
                    winsound.Beep(1500, 15)
                elif sound_type == "screenshot":
                    winsound.Beep(2200, 25)
                    winsound.Beep(2600, 35)
                elif sound_type == "virtual_desktop":
                    winsound.Beep(1000, 25)
                    winsound.Beep(1350, 30)
                elif sound_type == "media":
                    winsound.Beep(1700, 25)
                elif sound_type == "error":
                    winsound.Beep(450, 60)
            except Exception as e:
                logger.debug("Audio playback exception: %s", e)

        threading.Thread(target=_worker, daemon=True, name="AirControl-Audio").start()

    # --------------------------------------------------------------------------
    # VOLUME & AUDIO CONTROLS
    # --------------------------------------------------------------------------
    def volume_up(self, steps: int = 2) -> bool:
        """
        Increase system audio volume.
        Triggered by: SWIPE_UP
        """
        try:
            for _ in range(steps):
                pyautogui.press("volumeup")
            self.play_sound("volume")
            return True
        except Exception as e:
            logger.error("Failed to increase volume: %s", e)
            return False

    def volume_down(self, steps: int = 2) -> bool:
        """
        Decrease system audio volume.
        Triggered by: SWIPE_DOWN
        """
        try:
            for _ in range(steps):
                pyautogui.press("volumedown")
            self.play_sound("volume")
            return True
        except Exception as e:
            logger.error("Failed to decrease volume: %s", e)
            return False

    def volume_mute(self) -> bool:
        """Toggle system audio mute."""
        try:
            pyautogui.press("volumemute")
            self.play_sound("volume")
            return True
        except Exception as e:
            logger.error("Failed to toggle volume mute: %s", e)
            return False

    # --------------------------------------------------------------------------
    # SCREENSHOT CAPTURE
    # --------------------------------------------------------------------------
    def capture_screenshot(self) -> bool:
        """
        Trigger Windows Snipping Tool / interactive screen clipping.
        Triggered by: ROCK_ON
        """
        logger.info("Triggering Windows screen clip (Win+Shift+S)")
        try:
            pyautogui.hotkey("win", "shift", "s")
            self.play_sound("screenshot")
            return True
        except Exception as e:
            logger.error("Failed to capture screenshot: %s", e)
            return False

    # --------------------------------------------------------------------------
    # MEDIA PLAYBACK CONTROLS
    # --------------------------------------------------------------------------
    def media_play_pause(self) -> bool:
        """Toggle media Play/Pause."""
        try:
            pyautogui.press("playpause")
            self.play_sound("media")
            return True
        except Exception as e:
            logger.error("Failed to toggle media play/pause: %s", e)
            return False

    def media_next_track(self) -> bool:
        """Skip to next media track."""
        try:
            pyautogui.press("nexttrack")
            self.play_sound("media")
            return True
        except Exception as e:
            logger.error("Failed to skip media track: %s", e)
            return False

    def media_prev_track(self) -> bool:
        """Skip to previous media track."""
        try:
            pyautogui.press("prevtrack")
            self.play_sound("media")
            return True
        except Exception as e:
            logger.error("Failed to previous media track: %s", e)
            return False

