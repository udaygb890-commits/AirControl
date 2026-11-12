"""
AIRCONTROL — STAGE 1: THE VISION ENGINE
Main application entry point.

Initializes Windows DPI awareness, logging infrastructure,
vision subsystem, and launches the futuristic desktop dashboard.
"""

import ctypes
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import APP_NAME, APP_STAGE, APP_VERSION
from desktop import DesktopController
from gestures.gesture_engine import GestureEngine
from input_control.air_mouse import AirMouse
from interface.dashboard import AirControlDashboard
from vision.camera import CameraManager
from vision.hand_tracker import HandTracker


def setup_high_dpi_awareness() -> None:
    """Enable Windows 11 Per-Monitor DPI awareness for crisp vector rendering."""
    if sys.platform == "win32":
        try:
            # Per-monitor DPI awareness (Windows 8.1+)
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                # System DPI awareness fallback (Windows Vista+)
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def setup_logging() -> None:
    """Configure structured logging for AirControl."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    """Main execution entry point."""
    setup_high_dpi_awareness()
    setup_logging()

    logger = logging.getLogger("AirControl.Main")
    logger.info("Initializing %s (%s, v%s)...", APP_NAME, APP_STAGE, APP_VERSION)

    try:
        # Pre-verify and ensure MediaPipe model exists before UI launch
        hand_tracker = HandTracker()
        hand_tracker.ensure_model_available()

        # Instantiate Stage 4 Desktop Automation, Stage 3 Gesture Engine & Stage 2 Air Mouse
        gesture_engine = GestureEngine()
        air_mouse = AirMouse()
        desktop_controller = DesktopController()
        gesture_engine.add_listener(desktop_controller.handle_gesture_event)

        camera_manager = CameraManager(
            hand_tracker=hand_tracker,
            air_mouse=air_mouse,
            gesture_engine=gesture_engine,
        )

        # Launch Dashboard GUI
        app = AirControlDashboard(
            camera_manager=camera_manager,
            air_mouse=air_mouse,
            gesture_engine=gesture_engine,
            desktop_controller=desktop_controller,
        )
        app.mainloop()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Terminating AirControl.")
    except Exception as e:
        logger.exception("Fatal exception during AirControl execution: %s", e)
        sys.exit(1)
    finally:
        try:
            desktop_controller.stop(timeout=0.5)
            camera_manager.stop(timeout=0.5)
        except Exception:
            pass


if __name__ == "__main__":
    main()

