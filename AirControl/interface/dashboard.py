"""
AIRCONTROL — DASHBOARD INTERFACE MODULE (STAGE 2: RELATIVE AIR MOUSE)
Futuristic, high-contrast dark dashboard built with CustomTkinter.
Displays live camera stream, 21-point hand skeleton overlays, multi-hand telemetry,
real-time relative air-mouse metrics, hardware controls, and a live system event console.
"""

import datetime
import logging
import queue
import threading
import time
from typing import Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageDraw

from config import (
    APP_NAME,
    APP_STAGE,
    APP_SUBTITLE,
    APP_VERSION,
    COLOR_BG_CARD,
    COLOR_BG_HEADER,
    COLOR_BG_INNER,
    COLOR_BG_LOG,
    COLOR_BG_ROOT,
    COLOR_BLUE,
    COLOR_BORDER,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_RED,
    COLOR_TEXT_DIM,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    DEFAULT_AIR_MOUSE_ENABLED,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_MOUSE_DEADZONE,
    DEFAULT_MOUSE_SENSITIVITY_X,
    FONT_FAMILY,
    FONT_MONO,
    ICON_PATH,
    PRIMARY_MOUSE_HAND,
    WINDOW_HEIGHT,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_WIDTH,
)
from desktop import CommandResult, DesktopController
from gestures.gesture_engine import GestureEngine, GestureTelemetry
from hand3d import HandCalibrator, HandModel3D, SkeletonRenderer3D
from input_control.air_mouse import AirMouse, AirMouseTelemetry
from utils.diagnostics import measure_time
from vision.camera import CameraManager, FrameData
from vision.hand_tracker import HandData

logger = logging.getLogger("AirControl.Dashboard")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AirControlDashboard(ctk.CTk):
    """Futuristic desktop interface for AirControl Stage 4 Windows Desktop Automation."""

    def __init__(
        self,
        camera_manager: Optional[CameraManager] = None,
        air_mouse: Optional[AirMouse] = None,
        gesture_engine: Optional[GestureEngine] = None,
        desktop_controller: Optional[DesktopController] = None,
    ):
        super().__init__()

        self.air_mouse = air_mouse or AirMouse(enabled=DEFAULT_AIR_MOUSE_ENABLED)
        self.gesture_engine = (
            gesture_engine
            or getattr(camera_manager, "gesture_engine", None)
            or GestureEngine()
        )
        self.desktop_controller = desktop_controller or DesktopController()
        # Bind desktop controller to gesture engine events (avoiding duplicates)
        if self.desktop_controller.handle_gesture_event not in self.gesture_engine._listeners:
            self.gesture_engine.add_listener(self.desktop_controller.handle_gesture_event)
        self.desktop_controller.on_action = self._on_desktop_action_executed
        # Start background automation worker thread
        self.desktop_controller.start()

        self.camera_manager = camera_manager or CameraManager(
            air_mouse=self.air_mouse,
            gesture_engine=self.gesture_engine,
        )
        self.camera_manager.air_mouse = self.air_mouse
        self.camera_manager.gesture_engine = self.gesture_engine

        # Window Configuration
        self.title(f"{APP_NAME} — {APP_SUBTITLE} [{APP_STAGE}]")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.configure(fg_color=COLOR_BG_ROOT)

        # Set Window Icon
        if ICON_PATH.exists():
            try:
                self.iconbitmap(str(ICON_PATH))
            except Exception:
                pass

        # State Variables & Responsiveness Guards
        self._is_closing = False
        self._update_job: Optional[str] = None
        self._widget_cache: dict[int, tuple] = {}
        self._last_hand_count = -1
        self._last_mouse_state = "IDLE"
        self._last_applied_camera_status = "OFFLINE"
        self._last_dynamic_gesture_logged: Optional[str] = None
        self._last_dual_gesture_logged: Optional[str] = None
        self._standby_image: Optional[ctk.CTkImage] = None
        self._preview_size: Tuple[int, int] = (800, 450)

        # 3D Articulated Skeleton & Spatial Engine (v4.1)
        self.calibrator = HandCalibrator()
        self.skeleton_renderer_3d = SkeletonRenderer3D(width=340, height=220)
        self._current_3d_hand_model: Optional[HandModel3D] = None
        self._last_3d_render_time = 0.0

        # Build UI Architecture
        self._setup_grid()
        self._create_header()
        self._create_main_content()
        self._bind_events()

        # Generate Standby Display
        self._init_standby_display()

        # Append Initial Log Messages
        self.log_event("AirControl gesture intelligence initialized.")
        self.log_event(f"{APP_STAGE} active (v{APP_VERSION}).")
        self.log_event("MediaPipe Tasks Hand Landmarker ready (VIDEO mode).")
        self.log_event("Relative Air Mouse Engine loaded with exponential smoothing.")
        self.log_event("Stage 3 Gesture Engine ready: Static & dynamic motion recognition.")
        self.log_event("Stage 4 Desktop Automation ready: Windows manipulation active [F9].")
        self.log_event("Connecting to webcam sensor automatically...")

        # Automatically start camera on launch
        self.after(350, self._on_start_clicked)

        # Launch UI Refresh Loop (~60 Hz) with single scheduler guard
        self._schedule_next_update(delay_ms=20)

    def _schedule_next_update(self, delay_ms: int = 16) -> None:
        """Schedule the next UI update tick ensuring only one loop job exists."""
        if not self._is_closing:
            self._update_job = self.after(delay_ms, self._update_loop)

    def _set_widget_text(
        self,
        widget,
        text: str,
        text_color: Optional[str] = None,
        fg_color: Optional[str] = None,
    ) -> None:
        """
        Dirty-checking widget update: only reconfigures widget if text or colors changed.
        Eliminates expensive CustomTkinter canvas redraw lag and keeps UI perfectly responsive.
        """
        key = id(widget)
        current_state = (text, text_color, fg_color)
        if self._widget_cache.get(key) == current_state:
            return
        self._widget_cache[key] = current_state
        kwargs = {"text": text}
        if text_color is not None:
            kwargs["text_color"] = text_color
        if fg_color is not None:
            kwargs["fg_color"] = fg_color
        widget.configure(**kwargs)

    # ==========================================================================
    # LAYOUT & INITIALIZATION
    # ==========================================================================
    def _setup_grid(self) -> None:
        """Configure top-level grid layout."""
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Main Content
        self.grid_columnconfigure(0, weight=1)

    def _create_header(self) -> None:
        """Create sleek sci-fi header banner."""
        self.header_frame = ctk.CTkFrame(
            self,
            fg_color=COLOR_BG_HEADER,
            corner_radius=0,
            height=70,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        self.header_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 6))
        self.header_frame.grid_columnconfigure(0, weight=1)
        self.header_frame.grid_columnconfigure(1, weight=0)

        # Left Header Branding
        branding_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        branding_box.grid(row=0, column=0, sticky="w", padx=20, pady=10)

        self.logo_label = ctk.CTkLabel(
            branding_box,
            text=f"◈ {APP_NAME}",
            font=ctk.CTkFont(family=FONT_FAMILY, size=24, weight="bold"),
            text_color=COLOR_CYAN,
        )
        self.logo_label.pack(side="left", padx=(0, 14))

        title_divider = ctk.CTkLabel(
            branding_box,
            text="|",
            font=ctk.CTkFont(family=FONT_FAMILY, size=20),
            text_color=COLOR_BORDER,
        )
        title_divider.pack(side="left", padx=(0, 14))

        self.subtitle_label = ctk.CTkLabel(
            branding_box,
            text=f"{APP_SUBTITLE}  //  {APP_STAGE}",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="normal"),
            text_color=COLOR_TEXT_MUTED,
        )
        self.subtitle_label.pack(side="left")

        # Right Header Badges
        badges_box = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        badges_box.grid(row=0, column=1, sticky="e", padx=20, pady=10)

        self.status_pill = ctk.CTkLabel(
            badges_box,
            text="○ SYSTEM STANDBY",
            font=ctk.CTkFont(family=FONT_MONO, size=12, weight="bold"),
            text_color=COLOR_ORANGE,
            fg_color=COLOR_BG_INNER,
            corner_radius=12,
            padx=14,
            pady=4,
        )
        self.status_pill.pack(side="left", padx=(0, 12))

        version_pill = ctk.CTkLabel(
            badges_box,
            text=f"v{APP_VERSION}",
            font=ctk.CTkFont(family=FONT_MONO, size=12, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
            fg_color=COLOR_BG_CARD,
            corner_radius=12,
            padx=12,
            pady=4,
        )
        version_pill.pack(side="left")

    def _create_main_content(self) -> None:
        """Create split-screen content: Video Feed on left, Telemetry on right."""
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=3)  # Left Panel (Video Feed)
        content_frame.grid_columnconfigure(1, weight=2)  # Right Panel (Telemetry & Controls)

        self._build_left_panel(content_frame)
        self._build_right_panel(content_frame)

    # ==========================================================================
    # LEFT PANEL: LIVE WEBCAM & HUD PREVIEW
    # ==========================================================================
    def _build_left_panel(self, parent: ctk.CTkFrame) -> None:
        """Construct the live optical sensor viewport and action controls."""
        left_container = ctk.CTkFrame(
            parent,
            fg_color=COLOR_BG_CARD,
            corner_radius=14,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=0)
        left_container.grid_rowconfigure(0, weight=0)  # Panel Header
        left_container.grid_rowconfigure(1, weight=1)  # Video Display Frame
        left_container.grid_rowconfigure(2, weight=0)  # Telemetry Badges Strip
        left_container.grid_rowconfigure(3, weight=0)  # Action Buttons Bar
        left_container.grid_columnconfigure(0, weight=1)

        # 1. Panel Header
        header = ctk.CTkFrame(left_container, fg_color="transparent", height=40)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 6))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        left_title = ctk.CTkLabel(
            header,
            text="OPTICAL SENSOR VIEWPORT // LIVE HUD",
            font=ctk.CTkFont(family=FONT_MONO, size=13, weight="bold"),
            text_color=COLOR_CYAN,
        )
        left_title.grid(row=0, column=0, sticky="w")

        self.cam_badge = ctk.CTkLabel(
            header,
            text="SENSOR: CAMERA 0 [1280x720]",
            font=ctk.CTkFont(family=FONT_MONO, size=11),
            text_color=COLOR_TEXT_DIM,
        )
        self.cam_badge.grid(row=0, column=1, sticky="e")

        # 2. Video Preview Display
        self.video_container = ctk.CTkFrame(
            left_container,
            fg_color="#03060B",
            corner_radius=10,
            border_width=1,
            border_color="#18233C",
        )
        self.video_container.grid(row=1, column=0, sticky="nsew", padx=16, pady=6)
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_container, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # 3. Quick Telemetry Status Badges Strip
        status_strip = ctk.CTkFrame(left_container, fg_color="transparent", height=38)
        status_strip.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 8))
        status_strip.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.badge_cam_status = self._create_metric_chip(
            status_strip, 0, "CAMERA", "OFFLINE", COLOR_RED
        )
        self.badge_fps = self._create_metric_chip(
            status_strip, 1, "VIDEO FPS", "0.0 FPS", COLOR_CYAN
        )
        self.badge_mouse_state = self._create_metric_chip(
            status_strip, 2, "AIR MOUSE", "DISABLED", COLOR_TEXT_MUTED
        )
        self.badge_hands_detected = self._create_metric_chip(
            status_strip, 3, "HANDS DETECTED", "0 / 2", COLOR_TEXT_MUTED
        )

        # 4. Action Control Buttons Bar
        btn_bar = ctk.CTkFrame(left_container, fg_color="transparent")
        btn_bar.grid(row=3, column=0, sticky="ew", padx=16, pady=(4, 14))
        btn_bar.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.btn_start = ctk.CTkButton(
            btn_bar,
            text="▶  START CAMERA",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            fg_color=COLOR_CYAN,
            text_color="#000000",
            hover_color="#33F3FF",
            height=38,
            corner_radius=8,
            command=self._on_start_clicked,
        )
        self.btn_start.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.btn_stop = ctk.CTkButton(
            btn_bar,
            text="■  STOP CAMERA",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            fg_color=COLOR_RED,
            text_color="#FFFFFF",
            hover_color="#FF4D79",
            height=38,
            corner_radius=8,
            state="disabled",
            command=self._on_stop_clicked,
        )
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=4)

        self.btn_mouse_toggle = ctk.CTkButton(
            btn_bar,
            text="◈ AIR MOUSE [F8]",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            fg_color="#1E293B",
            text_color=COLOR_CYAN,
            border_width=1,
            border_color=COLOR_CYAN,
            hover_color="#0F172A",
            height=38,
            corner_radius=8,
            command=self._toggle_air_mouse,
        )
        self.btn_mouse_toggle.grid(row=0, column=2, sticky="ew", padx=4)

        self.btn_exit = ctk.CTkButton(
            btn_bar,
            text="✕  EXIT",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
            fg_color="#1E293B",
            text_color=COLOR_TEXT_MAIN,
            hover_color="#334155",
            height=38,
            corner_radius=8,
            command=self._on_exit_clicked,
        )
        self.btn_exit.grid(row=0, column=3, sticky="ew", padx=(4, 0))

    def _create_metric_chip(
        self, parent: ctk.CTkFrame, col: int, label_text: str, val_text: str, accent: str
    ) -> ctk.CTkLabel:
        """Helper to create a sleek metric readout chip."""
        chip = ctk.CTkFrame(parent, fg_color=COLOR_BG_INNER, corner_radius=8, height=36)
        chip.grid(row=0, column=col, sticky="ew", padx=3)
        chip.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            chip,
            text=label_text,
            font=ctk.CTkFont(family=FONT_MONO, size=9, weight="normal"),
            text_color=COLOR_TEXT_DIM,
        )
        title.pack(anchor="w", padx=8, pady=(2, 0))

        val = ctk.CTkLabel(
            chip,
            text=val_text,
            font=ctk.CTkFont(family=FONT_MONO, size=11, weight="bold"),
            text_color=accent,
        )
        val.pack(anchor="w", padx=8, pady=(0, 2))
        return val

    # ==========================================================================
    # RIGHT PANEL: AIR MOUSE ENGINE, TELEMETRY & HARDWARE CONTROLS
    # ==========================================================================
    def _build_right_panel(self, parent: ctk.CTkFrame) -> None:
        """Construct the right-hand analytics, dual-hand telemetry, and settings."""
        right_container = ctk.CTkScrollableFrame(
            parent,
            fg_color=COLOR_BG_CARD,
            corner_radius=14,
            border_width=1,
            border_color=COLOR_BORDER,
        )
        right_container.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=0)
        right_container.grid_columnconfigure(0, weight=1)

        # Panel Header
        r_header = ctk.CTkLabel(
            right_container,
            text="SYSTEM TELEMETRY & AIR MOUSE CONTROLS",
            font=ctk.CTkFont(family=FONT_MONO, size=13, weight="bold"),
            text_color=COLOR_CYAN,
        )
        r_header.pack(anchor="w", padx=12, pady=(12, 10))

        # CARD 1: AIR MOUSE ENGINE (STAGE 2)
        self._build_air_mouse_card(right_container)

        # CARD 2: 3D ARTICULATED SKELETON & SPATIAL ENGINE (STAGE 4.1)
        self._build_3d_spatial_card(right_container)

        # CARD 3: GESTURE INTELLIGENCE ENGINE (STAGE 3)
        self._build_gesture_intelligence_card(right_container)

        # CARD 3: WINDOWS DESKTOP AUTOMATION (STAGE 4)
        self._build_desktop_automation_card(right_container)

        # CARD 4: DETECTION ENGINE OVERVIEW
        self._build_engine_card(right_container)

        # CARD 3: RIGHT HAND TELEMETRY CARD (ELECTRIC CYAN)
        self._build_hand_card(right_container, hand_type="Right")

        # CARD 4: LEFT HAND TELEMETRY CARD (CYBER PURPLE)
        self._build_hand_card(right_container, hand_type="Left")

        # CARD 5: HARDWARE & VISION CONFIGURATION
        self._build_config_card(right_container)

        # CARD 6: SYSTEM EVENT CONSOLE
        self._build_event_log_card(right_container)

    # --------------------------------------------------------------------------
    # CARD 1: AIR MOUSE ENGINE & RELATIVE TRACKING (STAGE 2)
    # --------------------------------------------------------------------------
    def _build_air_mouse_card(self, parent: ctk.CTkFrame) -> None:
        """Air Mouse Control & Real-time relative displacement telemetry."""
        card = ctk.CTkFrame(
            parent,
            fg_color=COLOR_BG_INNER,
            corner_radius=10,
            border_width=1,
            border_color="#00D4FF",
        )
        card.pack(fill="x", padx=10, pady=5)
        card.grid_columnconfigure((0, 1), weight=1)

        # Header with Switch & State Badge
        h_row = ctk.CTkFrame(card, fg_color="transparent")
        h_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        h_row.grid_columnconfigure(0, weight=1)
        h_row.grid_columnconfigure(1, weight=0)

        card_title = ctk.CTkLabel(
            h_row,
            text="◈ RELATIVE AIR MOUSE ENGINE",
            font=ctk.CTkFont(family=FONT_MONO, size=12, weight="bold"),
            text_color=COLOR_CYAN,
        )
        card_title.grid(row=0, column=0, sticky="w")

        self.mouse_state_badge = ctk.CTkLabel(
            h_row,
            text="OFFLINE",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            text_color=COLOR_TEXT_DIM,
            fg_color="#0A0F1D",
            corner_radius=6,
            padx=10,
            pady=2,
        )
        self.mouse_state_badge.grid(row=0, column=1, sticky="e")

        # Master Switch & Hand Selector
        ctrl_row = ctk.CTkFrame(card, fg_color="transparent")
        ctrl_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 6))
        ctrl_row.grid_columnconfigure(0, weight=1)
        ctrl_row.grid_columnconfigure(1, weight=1)

        self.switch_air_mouse = ctk.CTkSwitch(
            ctrl_row,
            text="Engage Air Mouse [F8]",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
            progress_color=COLOR_CYAN,
            command=self._on_air_mouse_switch_toggled,
        )
        self.switch_air_mouse.grid(row=0, column=0, sticky="w")
        if self.air_mouse.enabled:
            self.switch_air_mouse.select()

        # Hand Selector Dropdown
        self.mouse_hand_selector = ctk.CTkOptionMenu(
            ctrl_row,
            values=["Right Hand (Default)", "Left Hand", "Any Hand"],
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            fg_color="#1E293B",
            button_color=COLOR_BLUE,
            dropdown_fg_color=COLOR_BG_CARD,
            dropdown_font=ctk.CTkFont(family=FONT_MONO, size=10),
            command=self._on_mouse_hand_select,
        )
        self.mouse_hand_selector.set("Right Hand (Default)")
        self.mouse_hand_selector.grid(row=0, column=1, sticky="e")

        # Real-time Movement Telemetry Readouts
        telemetry_box = ctk.CTkFrame(card, fg_color="#090E1A", corner_radius=6)
        telemetry_box.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        telemetry_box.grid_columnconfigure((0, 1), weight=1)

        self.lbl_mouse_delta = ctk.CTkLabel(
            telemetry_box,
            text="Rel Delta: dx: +0.000, dy: +0.000\nVelocity:  vx: +0.0 px, vy: +0.0 px",
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            text_color=COLOR_CYAN,
            justify="left",
        )
        self.lbl_mouse_delta.grid(row=0, column=0, sticky="w", padx=10, pady=6)

        self.lbl_mouse_pos = ctk.CTkLabel(
            telemetry_box,
            text="Cursor: (0, 0) px\nPinch: 100% | Drag: OFF\nClutch: READY",
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            text_color=COLOR_TEXT_MUTED,
            justify="left",
        )
        self.lbl_mouse_pos.grid(row=0, column=1, sticky="w", padx=10, pady=6)

        # Sensitivity & Deadzone Sliders
        sliders_box = ctk.CTkFrame(card, fg_color="transparent")
        sliders_box.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 10))
        sliders_box.grid_columnconfigure(1, weight=1)

        # Sensitivity Slider
        self.lbl_sens_val = ctk.CTkLabel(
            sliders_box,
            text=f"Sensitivity: {DEFAULT_MOUSE_SENSITIVITY_X:.1f}x",
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            text_color=COLOR_TEXT_MUTED,
            width=110,
            anchor="w",
        )
        self.lbl_sens_val.grid(row=0, column=0, sticky="w", pady=2)

        self.slider_sens = ctk.CTkSlider(
            sliders_box,
            from_=0.5,
            to=5.0,
            number_of_steps=45,
            progress_color=COLOR_CYAN,
            command=self._on_sensitivity_changed,
        )
        self.slider_sens.set(DEFAULT_MOUSE_SENSITIVITY_X)
        self.slider_sens.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)

        # Deadzone Slider
        self.lbl_deadzone_val = ctk.CTkLabel(
            sliders_box,
            text=f"Deadzone: {DEFAULT_MOUSE_DEADZONE:.4f}",
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            text_color=COLOR_TEXT_MUTED,
            width=110,
            anchor="w",
        )
        self.lbl_deadzone_val.grid(row=1, column=0, sticky="w", pady=2)

        self.slider_deadzone = ctk.CTkSlider(
            sliders_box,
            from_=0.001,
            to=0.012,
            number_of_steps=44,
            progress_color=COLOR_BLUE,
            command=self._on_deadzone_changed,
        )
        self.slider_deadzone.set(DEFAULT_MOUSE_DEADZONE)
        self.slider_deadzone.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=2)

    # --------------------------------------------------------------------------
    # CARD 2: 3D ARTICULATED SKELETON & SPATIAL ENGINE (STAGE 4.1)
    # --------------------------------------------------------------------------
    def _build_3d_spatial_card(self, parent: ctk.CTkFrame) -> None:
        """3D Articulated Skeleton Viewport, Camera Controls & Spatial Telemetry (v4.1)."""
        card = ctk.CTkFrame(
            parent,
            fg_color=COLOR_BG_INNER,
            corner_radius=10,
            border_width=1,
            border_color="#00E5FF",
        )
        card.pack(fill="x", padx=10, pady=5)
        card.grid_columnconfigure((0, 1), weight=1)

        # Header Row
        h_row = ctk.CTkFrame(card, fg_color="transparent")
        h_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        h_row.grid_columnconfigure(0, weight=1)
        h_row.grid_columnconfigure(1, weight=0)

        card_title = ctk.CTkLabel(
            h_row,
            text="◈ 3D ARTICULATED SKELETON (v4.1)",
            font=ctk.CTkFont(family=FONT_MONO, size=12, weight="bold"),
            text_color="#00E5FF",
        )
        card_title.grid(row=0, column=0, sticky="w")

        self.lbl_3d_mode_badge = ctk.CTkLabel(
            h_row,
            text="WORLD 3D ACTIVE",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            text_color=COLOR_GREEN,
            fg_color="#0A0F1D",
            corner_radius=6,
            padx=10,
            pady=2,
        )
        self.lbl_3d_mode_badge.grid(row=0, column=1, sticky="e")

        # 3D Spatial Cursor Switch & Mode Controls
        ctrl_row = ctk.CTkFrame(card, fg_color="transparent")
        ctrl_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 6))
        ctrl_row.grid_columnconfigure(0, weight=1)
        ctrl_row.grid_columnconfigure(1, weight=1)

        self.switch_spatial_3d = ctk.CTkSwitch(
            ctrl_row,
            text="3D Spatial Cursor Mode",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
            progress_color="#00E5FF",
            command=self._on_3d_cursor_toggle,
        )
        self.switch_spatial_3d.grid(row=0, column=0, sticky="w")

        btn_reset_view = ctk.CTkButton(
            ctrl_row,
            text="Reset 3D View",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            fg_color="#1E293B",
            hover_color="#334155",
            text_color=COLOR_CYAN,
            width=90,
            height=24,
            corner_radius=6,
            command=self._on_reset_3d_view,
        )
        btn_reset_view.grid(row=0, column=1, sticky="e")

        # 3D Viewport Frame
        vp_frame = ctk.CTkFrame(
            card,
            fg_color="#03060B",
            corner_radius=8,
            border_width=1,
            border_color="#18233C",
        )
        vp_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        vp_frame.grid_columnconfigure(0, weight=1)

        self.viewport_3d_label = ctk.CTkLabel(vp_frame, text="")
        self.viewport_3d_label.pack(fill="both", expand=True, padx=4, pady=4)

        # Initialize Viewport Standby Image
        initial_canvas = self.skeleton_renderer_3d.render(None)
        init_pil = Image.fromarray(initial_canvas)
        init_ctk = ctk.CTkImage(light_image=init_pil, dark_image=init_pil, size=(340, 220))
        self.viewport_3d_label.configure(image=init_ctk)
        self.viewport_3d_label.image = init_ctk

        # Orbit Sliders
        sliders_frame = ctk.CTkFrame(card, fg_color="transparent")
        sliders_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=2)
        sliders_frame.grid_columnconfigure((0, 1), weight=1)

        lbl_az = ctk.CTkLabel(sliders_frame, text="Orbit Azimuth / Elevation:", font=ctk.CTkFont(family=FONT_MONO, size=9), text_color=COLOR_TEXT_DIM)
        lbl_az.grid(row=0, column=0, sticky="w")

        self.slider_azimuth = ctk.CTkSlider(
            sliders_frame,
            from_=-180.0,
            to=180.0,
            number_of_steps=72,
            height=14,
            progress_color=COLOR_CYAN,
            command=self._on_orbit_slider_changed,
        )
        self.slider_azimuth.set(25.0)
        self.slider_azimuth.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(0, 4))

        self.slider_elevation = ctk.CTkSlider(
            sliders_frame,
            from_=-85.0,
            to=85.0,
            number_of_steps=34,
            height=14,
            progress_color="#FF00AA",
            command=self._on_orbit_slider_changed,
        )
        self.slider_elevation.set(-15.0)
        self.slider_elevation.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(0, 4))

        # Calibration Controls Row
        calib_row = ctk.CTkFrame(card, fg_color="transparent")
        calib_row.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 4))
        calib_row.grid_columnconfigure(0, weight=1)
        calib_row.grid_columnconfigure(1, weight=0)

        self.btn_calibrate = ctk.CTkButton(
            calib_row,
            text="Calibrate Hand Scale (Open Palm)",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            fg_color="#0D47A1",
            hover_color="#1565C0",
            text_color="#FFFFFF",
            height=26,
            corner_radius=6,
            command=self._on_calibrate_clicked,
        )
        self.btn_calibrate.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        btn_reset_calib = ctk.CTkButton(
            calib_row,
            text="Reset",
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            fg_color="#1E293B",
            hover_color="#334155",
            text_color=COLOR_TEXT_DIM,
            width=50,
            height=26,
            corner_radius=6,
            command=self._on_reset_calibration_clicked,
        )
        btn_reset_calib.grid(row=0, column=1, sticky="e")

        self.lbl_calibration_status = ctk.CTkLabel(
            card,
            text="Calibration: Default Hand Model (~8.5 cm palm)",
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_TEXT_DIM,
        )
        self.lbl_calibration_status.grid(row=5, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 4))

        # 3D Telemetry Readouts Box
        telemetry_box = ctk.CTkFrame(card, fg_color="#0A0F1D", corner_radius=6)
        telemetry_box.grid(row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 8))
        telemetry_box.grid_columnconfigure(0, weight=1)

        self.lbl_3d_orientation = ctk.CTkLabel(
            telemetry_box,
            text="Orientation: Pitch: 0° | Yaw: 0° | Roll: 0° | Facing: CAMERA",
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_CYAN,
            anchor="w",
            justify="left",
        )
        self.lbl_3d_orientation.pack(fill="x", padx=8, pady=(4, 1))

        self.lbl_3d_angles = ctk.CTkLabel(
            telemetry_box,
            text="3D Flexion:  Idx: --°  Mid: --°  Rng: --°  Pky: --°  Thb: --°",
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_TEXT_MAIN,
            anchor="w",
            justify="left",
        )
        self.lbl_3d_angles.pack(fill="x", padx=8, pady=1)

        self.lbl_3d_kinematics = ctk.CTkLabel(
            telemetry_box,
            text="Kinematics:  Velocity: (0.00, 0.00, 0.00) m/s | Scale: 1.00",
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
            justify="left",
        )
        self.lbl_3d_kinematics.pack(fill="x", padx=8, pady=(1, 4))

    # --------------------------------------------------------------------------
    # CARD 3: GESTURE INTELLIGENCE ENGINE (STAGE 3)
    # --------------------------------------------------------------------------
    def _build_gesture_intelligence_card(self, parent: ctk.CTkFrame) -> None:
        """Card displaying real-time static hand poses, dynamic gestures, and dual-hand states."""
        card = ctk.CTkFrame(
            parent,
            fg_color=COLOR_BG_INNER,
            corner_radius=10,
            border_width=1,
            border_color="#A855F7",
        )
        card.pack(fill="x", padx=10, pady=5)
        card.grid_columnconfigure((0, 1), weight=1)

        # Header with Switch & State Badge
        h_row = ctk.CTkFrame(card, fg_color="transparent")
        h_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        h_row.grid_columnconfigure(0, weight=1)
        h_row.grid_columnconfigure(1, weight=0)

        card_title = ctk.CTkLabel(
            h_row,
            text="◈ GESTURE INTELLIGENCE ENGINE",
            font=ctk.CTkFont(family=FONT_MONO, size=12, weight="bold"),
            text_color=COLOR_PURPLE,
        )
        card_title.grid(row=0, column=0, sticky="w")

        self.gesture_state_badge = ctk.CTkLabel(
            h_row,
            text="ACTIVE",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            text_color=COLOR_GREEN,
            fg_color="#0A0F1D",
            corner_radius=6,
            padx=10,
            pady=2,
        )
        self.gesture_state_badge.grid(row=0, column=1, sticky="e")

        # Toggle Switch Row
        ctrl_row = ctk.CTkFrame(card, fg_color="transparent")
        ctrl_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 6))
        ctrl_row.grid_columnconfigure(0, weight=1)

        self.switch_gesture_engine = ctk.CTkSwitch(
            ctrl_row,
            text="Enable Gesture Recognition Engine",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
            progress_color=COLOR_PURPLE,
            command=self._on_gesture_engine_switch_toggled,
        )
        self.switch_gesture_engine.grid(row=0, column=0, sticky="w")
        if self.gesture_engine.enabled:
            self.switch_gesture_engine.select()

        # Dual Hand Pose Display Box
        poses_box = ctk.CTkFrame(card, fg_color="#090E1A", corner_radius=6)
        poses_box.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        poses_box.grid_columnconfigure(0, weight=1)
        poses_box.grid_columnconfigure(1, weight=1)

        # Right Hand Pose Subcard
        r_box = ctk.CTkFrame(poses_box, fg_color="#0E1526", corner_radius=6)
        r_box.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        lbl_r_title = ctk.CTkLabel(
            r_box,
            text="RIGHT HAND POSE",
            font=ctk.CTkFont(family=FONT_MONO, size=9, weight="bold"),
            text_color=COLOR_CYAN,
        )
        lbl_r_title.pack(anchor="w", padx=8, pady=(4, 0))

        self.lbl_r_gesture = ctk.CTkLabel(
            r_box,
            text="NO_HAND",
            font=ctk.CTkFont(family=FONT_MONO, size=13, weight="bold"),
            text_color=COLOR_TEXT_MUTED,
        )
        self.lbl_r_gesture.pack(anchor="w", padx=8, pady=(2, 0))

        self.lbl_r_gesture_meta = ctk.CTkLabel(
            r_box,
            text="Conf: 0% | Hold: 0.0s",
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_TEXT_DIM,
        )
        self.lbl_r_gesture_meta.pack(anchor="w", padx=8, pady=(0, 4))

        # Left Hand Pose Subcard
        l_box = ctk.CTkFrame(poses_box, fg_color="#0E1526", corner_radius=6)
        l_box.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        lbl_l_title = ctk.CTkLabel(
            l_box,
            text="LEFT HAND POSE",
            font=ctk.CTkFont(family=FONT_MONO, size=9, weight="bold"),
            text_color=COLOR_PURPLE,
        )
        lbl_l_title.pack(anchor="w", padx=8, pady=(4, 0))

        self.lbl_l_gesture = ctk.CTkLabel(
            l_box,
            text="NO_HAND",
            font=ctk.CTkFont(family=FONT_MONO, size=13, weight="bold"),
            text_color=COLOR_TEXT_MUTED,
        )
        self.lbl_l_gesture.pack(anchor="w", padx=8, pady=(2, 0))

        self.lbl_l_gesture_meta = ctk.CTkLabel(
            l_box,
            text="Conf: 0% | Hold: 0.0s",
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_TEXT_DIM,
        )
        self.lbl_l_gesture_meta.pack(anchor="w", padx=8, pady=(0, 4))

        # Dynamic & Dual Motion Gesture Event Radar
        action_box = ctk.CTkFrame(card, fg_color="#090E1A", corner_radius=6)
        action_box.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 8))
        action_box.grid_columnconfigure(0, weight=1)

        self.lbl_dynamic_gesture = ctk.CTkLabel(
            action_box,
            text="Dynamic Motion: STANDBY (Ready for Swipes, Wave, Circle)",
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            text_color=COLOR_TEXT_MUTED,
            justify="left",
            anchor="w",
        )
        self.lbl_dynamic_gesture.pack(anchor="w", padx=8, pady=(4, 2))

        self.lbl_dual_gesture = ctk.CTkLabel(
            action_box,
            text="Dual Interaction: STANDBY (Dual Palms / Zoom / Box)",
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            text_color=COLOR_TEXT_DIM,
            justify="left",
            anchor="w",
        )
        self.lbl_dual_gesture.pack(anchor="w", padx=8, pady=(0, 4))

    # --------------------------------------------------------------------------
    # CARD 3: WINDOWS DESKTOP AUTOMATION ENGINE (STAGE 4)
    # --------------------------------------------------------------------------
    def _build_desktop_automation_card(self, parent: ctk.CTkFrame) -> None:
        """Desktop Automation: Windows management, application switching, and system controls."""
        card = ctk.CTkFrame(
            parent,
            fg_color=COLOR_BG_INNER,
            corner_radius=10,
            border_width=1,
            border_color="#00FFA3",
        )
        card.pack(fill="x", padx=10, pady=5)
        card.grid_columnconfigure((0, 1), weight=1)

        # Header with Switch & State Badge
        h_row = ctk.CTkFrame(card, fg_color="transparent")
        h_row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        h_row.grid_columnconfigure(0, weight=1)
        h_row.grid_columnconfigure(1, weight=0)

        card_title = ctk.CTkLabel(
            h_row,
            text="◈ WINDOWS DESKTOP AUTOMATION",
            font=ctk.CTkFont(family=FONT_MONO, size=12, weight="bold"),
            text_color=COLOR_GREEN,
        )
        card_title.grid(row=0, column=0, sticky="w")

        self.automation_state_badge = ctk.CTkLabel(
            h_row,
            text="ACTIVE",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            text_color=COLOR_GREEN,
            fg_color="#0A0F1D",
            corner_radius=6,
            padx=10,
            pady=2,
        )
        self.automation_state_badge.grid(row=0, column=1, sticky="e")

        # Master Toggle Switch Row
        ctrl_row = ctk.CTkFrame(card, fg_color="transparent")
        ctrl_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=(2, 6))
        ctrl_row.grid_columnconfigure(0, weight=1)

        self.switch_automation = ctk.CTkSwitch(
            ctrl_row,
            text="Engage Desktop Automation [F9]",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
            progress_color=COLOR_GREEN,
            command=self._on_automation_switch_toggled,
        )
        self.switch_automation.grid(row=0, column=0, sticky="w")
        if self.desktop_controller.enabled:
            self.switch_automation.select()

        # Last Command Executed Display
        status_box = ctk.CTkFrame(card, fg_color="#090E1A", corner_radius=6)
        status_box.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=4)
        status_box.grid_columnconfigure(0, weight=1)

        self.lbl_auto_last_action = ctk.CTkLabel(
            status_box,
            text="Last Action: STANDBY (Ready for Window & System Gestures)",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            text_color=COLOR_GREEN,
            anchor="w",
            justify="left",
        )
        self.lbl_auto_last_action.pack(anchor="w", padx=8, pady=(4, 2))

        self.lbl_auto_stats = ctk.CTkLabel(
            status_box,
            text="Executed: 0 actions | Safety Cooldown: ACTIVE | Audio: ON",
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        self.lbl_auto_stats.pack(anchor="w", padx=8, pady=(0, 4))

        # Quick Reference Gesture Mappings (Micro-chips)
        chips_frame = ctk.CTkFrame(card, fg_color="transparent")
        chips_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(4, 8))
        chips_frame.grid_columnconfigure((0, 1), weight=1)

        col1_text = (
            "• [THUMBS_DOWN]  -> Minimize Window\n"
            "• [TWO_HANDS_X]  -> Close Window (Alt+F4)\n"
            "• [THUMBS_UP]    -> Maximize / Restore\n"
            "• [OK_SIGN]      -> Switch Apps (Alt+Tab)"
        )
        col2_text = (
            "• [PALM_HELD]    -> Show Desktop (Win+D)\n"
            "• [SWIPE L / R]  -> Prev / Next Desktop\n"
            "• [SWIPE U / D]  -> Volume Up / Down\n"
            "• [ROCK_ON]      -> Screenshot (Win+Shift+S)"
        )

        lbl_chips_1 = ctk.CTkLabel(
            chips_frame,
            text=col1_text,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_TEXT_MUTED,
            justify="left",
            anchor="w",
        )
        lbl_chips_1.grid(row=0, column=0, sticky="w", padx=(0, 4))

        lbl_chips_2 = ctk.CTkLabel(
            chips_frame,
            text=col2_text,
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_TEXT_MUTED,
            justify="left",
            anchor="w",
        )
        lbl_chips_2.grid(row=0, column=1, sticky="w", padx=(4, 0))

    # --------------------------------------------------------------------------
    # CARD 4: DETECTION ENGINE OVERVIEW
    # --------------------------------------------------------------------------
    def _build_engine_card(self, parent: ctk.CTkFrame) -> None:
        """Card displaying Detection Engine properties and capacity."""
        card = ctk.CTkFrame(parent, fg_color=COLOR_BG_INNER, corner_radius=10, border_width=1, border_color=COLOR_BORDER)
        card.pack(fill="x", padx=10, pady=5)
        card.grid_columnconfigure((0, 1), weight=1)

        card_title = ctk.CTkLabel(
            card,
            text="◈ DETECTION ENGINE: MEDIAPIPE TASKS",
            font=ctk.CTkFont(family=FONT_MONO, size=11, weight="bold"),
            text_color=COLOR_BLUE,
        )
        card_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 4))

        info_left = ctk.CTkLabel(
            card,
            text="Mode: VIDEO  |  Confidence: 0.60\nLandmarks: 21 (3D Norm + Pixels)",
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            text_color=COLOR_TEXT_MUTED,
            justify="left",
        )
        info_left.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))

        meter_box = ctk.CTkFrame(card, fg_color="transparent")
        meter_box.grid(row=1, column=1, sticky="e", padx=12, pady=(0, 8))

        self.lbl_hands_count_large = ctk.CTkLabel(
            meter_box,
            text="0",
            font=ctk.CTkFont(family=FONT_MONO, size=24, weight="bold"),
            text_color=COLOR_CYAN,
        )
        self.lbl_hands_count_large.pack(side="left", padx=(0, 8))

        lbl_slash = ctk.CTkLabel(
            meter_box,
            text="/ 2 HANDS",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            text_color=COLOR_TEXT_DIM,
        )
        lbl_slash.pack(side="left")

    # --------------------------------------------------------------------------
    # CARDS 3 & 4: DEDICATED LEFT & RIGHT HAND TELEMETRY
    # --------------------------------------------------------------------------
    def _build_hand_card(self, parent: ctk.CTkFrame, hand_type: str) -> None:
        """Card presenting real-time telemetry for Right or Left hand."""
        card = ctk.CTkFrame(parent, fg_color=COLOR_BG_INNER, corner_radius=10, border_width=1, border_color=COLOR_BORDER)
        card.pack(fill="x", padx=10, pady=5)
        card.grid_columnconfigure(0, weight=1)

        accent_color = COLOR_CYAN if hand_type == "Right" else COLOR_PURPLE

        # Header Row
        h_row = ctk.CTkFrame(card, fg_color="transparent")
        h_row.pack(fill="x", padx=12, pady=(8, 4))

        title_lbl = ctk.CTkLabel(
            h_row,
            text=f"◈ {hand_type.upper()} HAND TELEMETRY",
            font=ctk.CTkFont(family=FONT_MONO, size=11, weight="bold"),
            text_color=accent_color,
        )
        title_lbl.pack(side="left")

        handedness_badge = ctk.CTkLabel(
            h_row,
            text="STANDBY",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            text_color=COLOR_TEXT_DIM,
            fg_color="#0A0F1D",
            corner_radius=6,
            padx=8,
            pady=1,
        )
        handedness_badge.pack(side="right")

        # Confidence Progress Bar
        conf_row = ctk.CTkFrame(card, fg_color="transparent")
        conf_row.pack(fill="x", padx=12, pady=(2, 4))

        conf_lbl = ctk.CTkLabel(
            conf_row,
            text="Confidence:",
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            text_color=COLOR_TEXT_MUTED,
        )
        conf_lbl.pack(side="left")

        conf_val = ctk.CTkLabel(
            conf_row,
            text="0.0%",
            font=ctk.CTkFont(family=FONT_MONO, size=10, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
        )
        conf_val.pack(side="right")

        conf_bar = ctk.CTkProgressBar(
            card,
            height=6,
            progress_color=accent_color,
            fg_color="#1A243B",
            corner_radius=3,
        )
        conf_bar.pack(fill="x", padx=12, pady=(0, 6))
        conf_bar.set(0.0)

        # Palm Coordinates Row
        palm_lbl = ctk.CTkLabel(
            card,
            text="Palm Center: (---, ---) px",
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            text_color=COLOR_TEXT_DIM,
            anchor="w",
        )
        palm_lbl.pack(fill="x", padx=12, pady=(0, 4))

        # 5 Finger Extension State Chips
        ft_frame = ctk.CTkFrame(card, fg_color="transparent")
        ft_frame.pack(fill="x", padx=12, pady=(0, 8))

        ft_labels = {}
        for tip_name in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            badge = ctk.CTkLabel(
                ft_frame,
                text=f"{tip_name[:1]}:--",
                font=ctk.CTkFont(family=FONT_MONO, size=9),
                text_color=COLOR_TEXT_DIM,
                fg_color="#0B101E",
                corner_radius=4,
                padx=5,
                pady=1,
            )
            badge.pack(side="left", padx=2)
            ft_labels[tip_name] = badge

        # Save component references
        if hand_type == "Right":
            self.right_badge = handedness_badge
            self.right_conf_val = conf_val
            self.right_conf_bar = conf_bar
            self.right_palm_lbl = palm_lbl
            self.right_ft_labels = ft_labels
        else:
            self.left_badge = handedness_badge
            self.left_conf_val = conf_val
            self.left_conf_bar = conf_bar
            self.left_palm_lbl = palm_lbl
            self.left_ft_labels = ft_labels

    # --------------------------------------------------------------------------
    # CARDS 5 & 6: CONFIGURATION & EVENT CONSOLE
    # --------------------------------------------------------------------------
    def _build_config_card(self, parent: ctk.CTkFrame) -> None:
        """Card containing camera selector, skeleton switch, and mirror switch."""
        card = ctk.CTkFrame(parent, fg_color=COLOR_BG_INNER, corner_radius=10, border_width=1, border_color=COLOR_BORDER)
        card.pack(fill="x", padx=10, pady=5)
        card.grid_columnconfigure((0, 1), weight=1)

        card_title = ctk.CTkLabel(
            card,
            text="◈ HARDWARE & HUD CONFIGURATION",
            font=ctk.CTkFont(family=FONT_MONO, size=11, weight="bold"),
            text_color=COLOR_TEXT_MAIN,
        )
        card_title.grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 6))

        # Camera Index Dropdown
        cam_lbl = ctk.CTkLabel(
            card,
            text="Camera Source:",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=COLOR_TEXT_MUTED,
        )
        cam_lbl.grid(row=1, column=0, sticky="w", padx=12, pady=4)

        self.cam_selector = ctk.CTkOptionMenu(
            card,
            values=["Camera 0 (Default)", "Camera 1", "Camera 2"],
            font=ctk.CTkFont(family=FONT_MONO, size=11),
            fg_color="#1E293B",
            button_color=COLOR_BLUE,
            button_hover_color=COLOR_CYAN,
            dropdown_fg_color=COLOR_BG_CARD,
            dropdown_font=ctk.CTkFont(family=FONT_MONO, size=11),
            command=self._on_camera_select,
        )
        self.cam_selector.set("Camera 0 (Default)")
        self.cam_selector.grid(row=1, column=1, sticky="e", padx=12, pady=4)

        # Switch 1: Hand Skeleton Overlay
        self.switch_skeleton = ctk.CTkSwitch(
            card,
            text="Render 21-Point Skeleton HUD",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=COLOR_TEXT_MAIN,
            progress_color=COLOR_CYAN,
            command=self._on_skeleton_toggle,
        )
        self.switch_skeleton.select()
        self.switch_skeleton.grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=4)

        # Switch 2: Mirror Selfie View
        self.switch_mirror = ctk.CTkSwitch(
            card,
            text="Mirror Optical Feed (Natural View)",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=COLOR_TEXT_MAIN,
            progress_color=COLOR_CYAN,
            command=self._on_mirror_toggle,
        )
        self.switch_mirror.select()
        self.switch_mirror.grid(row=3, column=0, columnspan=2, sticky="w", padx=12, pady=4)

        # Switch 3: Futuristic Telemetry Badges on Video
        self.switch_hud = ctk.CTkSwitch(
            card,
            text="Show Floating HUD Hand & Mouse Badges",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=COLOR_TEXT_MAIN,
            progress_color=COLOR_CYAN,
            command=self._on_hud_toggle,
        )
        self.switch_hud.select()
        self.switch_hud.grid(row=4, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 10))

    def _build_event_log_card(self, parent: ctk.CTkFrame) -> None:
        """Terminal-styled event log displaying system events in real time."""
        card = ctk.CTkFrame(parent, fg_color=COLOR_BG_INNER, corner_radius=10, border_width=1, border_color=COLOR_BORDER)
        card.pack(fill="x", padx=10, pady=(5, 12))
        card.grid_columnconfigure(0, weight=1)

        h_box = ctk.CTkFrame(card, fg_color="transparent")
        h_box.pack(fill="x", padx=12, pady=(8, 4))

        title = ctk.CTkLabel(
            h_box,
            text="◈ SYSTEM EVENT CONSOLE",
            font=ctk.CTkFont(family=FONT_MONO, size=11, weight="bold"),
            text_color=COLOR_TEXT_MUTED,
        )
        title.pack(side="left")

        btn_clear = ctk.CTkButton(
            h_box,
            text="CLEAR",
            font=ctk.CTkFont(family=FONT_MONO, size=9),
            fg_color="#1E293B",
            hover_color="#334155",
            width=50,
            height=20,
            corner_radius=4,
            command=self._on_clear_log,
        )
        btn_clear.pack(side="right")

        self.log_textbox = ctk.CTkTextbox(
            card,
            height=130,
            fg_color=COLOR_BG_LOG,
            text_color=COLOR_CYAN,
            font=ctk.CTkFont(family=FONT_MONO, size=10),
            corner_radius=6,
            border_width=1,
            border_color="#162035",
            wrap="none",
        )
        self.log_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.log_textbox.configure(state="disabled")

    # ==========================================================================
    # STANDBY DISPLAY GENERATION
    # ==========================================================================
    def _init_standby_display(self) -> None:
        """Create a futuristic cyber standby screen for when the camera is offline."""
        w, h = 800, 450
        img = Image.new("RGB", (w, h), color=(7, 11, 18))
        draw = ImageDraw.Draw(img)

        # Subtle cyber grid
        grid_step = 40
        for x in range(0, w, grid_step):
            draw.line([(x, 0), (x, h)], fill=(14, 22, 38), width=1)
        for y in range(0, h, grid_step):
            draw.line([(0, y), (w, y)], fill=(14, 22, 38), width=1)

        # Center Targeting Circles
        cx, cy = w // 2, h // 2
        for r, col, wid in [(160, (20, 35, 60), 1), (120, (0, 100, 180), 1), (80, (0, 240, 255), 2)]:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=wid)

        # Center Reticle Crosshairs
        draw.line([(cx - 190, cy), (cx - 90, cy)], fill=(0, 240, 255), width=2)
        draw.line([(cx + 90, cy), (cx + 190, cy)], fill=(0, 240, 255), width=2)
        draw.line([(cx, cy - 190), (cx, cy - 90)], fill=(0, 240, 255), width=2)
        draw.line([(cx, cy + 90), (cx, cy + 190)], fill=(0, 240, 255), width=2)

        # Corner HUD brackets
        b_len = 35
        for bx, by, dx, dy in [
            (25, 25, 1, 1),
            (w - 25, 25, -1, 1),
            (25, h - 25, 1, -1),
            (w - 25, h - 25, -1, -1),
        ]:
            draw.line([(bx, by), (bx + dx * b_len, by)], fill=(0, 240, 255), width=2)
            draw.line([(bx, by), (bx, by + dy * b_len)], fill=(0, 240, 255), width=2)

        # Center Text
        draw.text((cx, cy - 22), "AIRCONTROL  //  STAGE 2: RELATIVE AIR MOUSE", fill=(0, 240, 255), anchor="mm")
        draw.text((cx, cy + 5), "[ OPTICAL SENSOR ON STANDBY ]", fill=(255, 255, 255), anchor="mm")
        draw.text(
            (cx, cy + 32),
            "CLICK 'START CAMERA' OR PRESS [F8] TO ACTIVATE",
            fill=(148, 163, 184),
            anchor="mm",
        )

        self._standby_image = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.video_label.configure(image=self._standby_image)

    # ==========================================================================
    # EVENT LOGGING
    # ==========================================================================
    def log_event(self, text: str) -> None:
        """Append a timestamped message to the futuristic event console (Main UI Thread Safe)."""
        if threading.current_thread() != threading.main_thread():
            try:
                self.after(0, lambda: self.log_event(text))
            except Exception:
                pass
            return

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {text}\n"

        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", formatted)

        lines = int(self.log_textbox.index("end-1c").split(".")[0])
        if lines > 80:
            self.log_textbox.delete("1.0", "2.0")

        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def _on_clear_log(self) -> None:
        """Clear event console."""
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    # ==========================================================================
    # USER CONTROLS & AIR MOUSE CALLBACKS
    # ==========================================================================
    def _bind_events(self) -> None:
        """Bind window resize, closing, and Air Mouse keyboard shortcuts."""
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.video_container.bind("<Configure>", self._on_video_resize)
        # Hotkeys
        self.bind("<F8>", lambda e: self._toggle_air_mouse())
        self.bind("<F9>", lambda e: self._toggle_desktop_automation())
        self.bind("<Escape>", lambda e: self._emergency_halt())

    def _on_video_resize(self, event) -> None:
        """Dynamically adjust video display dimensions maintaining 16:9 aspect ratio."""
        avail_w = max(400, event.width - 12)
        avail_h = max(225, event.height - 12)

        ratio = 16.0 / 9.0
        if avail_w / avail_h > ratio:
            new_h = avail_h
            new_w = int(avail_h * ratio)
        else:
            new_w = avail_w
            new_h = int(avail_w / ratio)

        self._preview_size = (new_w, new_h)

    def _toggle_air_mouse(self) -> None:
        """Toggle Air Mouse active state via button or F8 key."""
        new_val = not self.air_mouse.enabled
        self.air_mouse.enabled = new_val
        if new_val:
            self.switch_air_mouse.select()
            self.btn_mouse_toggle.configure(fg_color=COLOR_CYAN, text_color="#000000")
            self.log_event("Air Mouse ENGAGED [F8]. Cursor steering active.")
        else:
            self.switch_air_mouse.deselect()
            self.btn_mouse_toggle.configure(fg_color="#1E293B", text_color=COLOR_CYAN)
            self.log_event("Air Mouse DISENGAGED [F8]. Cursor steering paused.")

    def _disable_air_mouse(self) -> None:
        """Safety shortcut: immediately disable cursor control."""
        if self.air_mouse.enabled:
            self.air_mouse.enabled = False
            self.switch_air_mouse.deselect()
            self.btn_mouse_toggle.configure(fg_color="#1E293B", text_color=COLOR_CYAN)
            self.log_event("Safety Halt: Air Mouse disengaged via [Escape].")

    def _on_air_mouse_switch_toggled(self) -> None:
        """Handle manual Air Mouse switch toggle."""
        val = bool(self.switch_air_mouse.get())
        self.air_mouse.enabled = val
        if val:
            self.btn_mouse_toggle.configure(fg_color=COLOR_CYAN, text_color="#000000")
            self.log_event("Air Mouse control enabled.")
        else:
            self.btn_mouse_toggle.configure(fg_color="#1E293B", text_color=COLOR_CYAN)
            self.log_event("Air Mouse control disabled.")

    def _on_mouse_hand_select(self, choice: str) -> None:
        """Change designated mouse control hand."""
        if "Right" in choice:
            self.air_mouse.primary_hand = "Right"
        elif "Left" in choice:
            self.air_mouse.primary_hand = "Left"
        else:
            self.air_mouse.primary_hand = "Any"
        self.log_event(f"Air Mouse primary hand set to: {self.air_mouse.primary_hand}")

    def _on_sensitivity_changed(self, val: float) -> None:
        """Update horizontal/vertical sensitivity multiplier."""
        self.air_mouse.set_sensitivity(val)
        self.lbl_sens_val.configure(text=f"Sensitivity: {val:.1f}x")

    def _on_deadzone_changed(self, val: float) -> None:
        """Update movement deadzone threshold."""
        self.air_mouse.set_deadzone(val)
        self.lbl_deadzone_val.configure(text=f"Deadzone: {val:.4f}")

    def _on_gesture_engine_switch_toggled(self) -> None:
        """Handle manual Gesture Recognition Engine switch toggle."""
        val = bool(self.switch_gesture_engine.get())
        self.gesture_engine.enabled = val
        if val:
            self.gesture_state_badge.configure(text="ACTIVE", text_color=COLOR_GREEN)
            self.log_event("Gesture Intelligence Engine enabled.")
        else:
            self.gesture_state_badge.configure(text="DISABLED", text_color=COLOR_TEXT_DIM)
            self.log_event("Gesture Intelligence Engine disabled.")

    def _toggle_desktop_automation(self) -> None:
        """Toggle Desktop Automation active state via button or F9 key."""
        new_val = not self.desktop_controller.enabled
        self.desktop_controller.enabled = new_val
        if new_val:
            self.switch_automation.select()
            self.automation_state_badge.configure(text="ACTIVE", text_color=COLOR_GREEN)
            self.log_event("Desktop Automation ENGAGED [F9]. Gestures mapped to Windows.")
        else:
            self.switch_automation.deselect()
            self.automation_state_badge.configure(text="DISABLED", text_color=COLOR_TEXT_DIM)
            self.log_event("Desktop Automation DISENGAGED [F9]. Window actions paused.")

    def _emergency_halt(self) -> None:
        """Emergency safety shortcut [Escape]: immediately halt cursor control and desktop automation."""
        self._disable_air_mouse()
        if self.desktop_controller.enabled:
            self.desktop_controller.emergency_halt()
            self.switch_automation.deselect()
            self.automation_state_badge.configure(text="HALTED", text_color=COLOR_RED)
            self.log_event("EMERGENCY SAFETY HALT [Escape]: All automation disengaged.")

    def _on_automation_switch_toggled(self) -> None:
        """Handle manual Desktop Automation switch toggle."""
        val = bool(self.switch_automation.get())
        self.desktop_controller.enabled = val
        if val:
            self.automation_state_badge.configure(text="ACTIVE", text_color=COLOR_GREEN)
            self.log_event("Desktop Automation enabled.")
        else:
            self.automation_state_badge.configure(text="DISABLED", text_color=COLOR_TEXT_DIM)
            self.log_event("Desktop Automation disabled.")

    def _on_desktop_action_executed(self, result: CommandResult) -> None:
        """Callback from DesktopController when a desktop action executes."""
        self.log_event(f"Desktop Action: {result.description} via [{result.gesture_name}]")

    # ==========================================================================
    # 3D SPATIAL ENGINE & VIEWPORT CONTROLS (STAGE 4.1)
    # ==========================================================================
    def _on_3d_cursor_toggle(self) -> None:
        """Toggle 3D Spatial Cursor Mode in AirMouse."""
        val = bool(self.switch_spatial_3d.get())
        self.air_mouse.spatial_3d_enabled = val
        self.log_event(f"3D Spatial Cursor Mode: {'ENABLED' if val else 'DISABLED'}")

    def _on_orbit_slider_changed(self, _val: float) -> None:
        """Update 3D viewport camera angles."""
        az = self.slider_azimuth.get()
        el = self.slider_elevation.get()
        self.skeleton_renderer_3d.set_camera_orbit(az, el)

    def _on_reset_3d_view(self) -> None:
        """Reset 3D camera angles."""
        self.slider_azimuth.set(25.0)
        self.slider_elevation.set(-15.0)
        self.skeleton_renderer_3d.reset_view()

    def _on_calibrate_clicked(self) -> None:
        """Start collecting user hand scale samples."""
        self.calibrator.start_calibration()
        self.lbl_calibration_status.configure(
            text="Hold an Open Palm towards camera to calibrate...",
            text_color=COLOR_ORANGE,
        )
        self.log_event("Hand calibration started: Hold open palm steady.")

    def _on_reset_calibration_clicked(self) -> None:
        """Reset calibration back to default."""
        self.calibrator.reset()
        self.lbl_calibration_status.configure(
            text="Calibration: Default Hand Model (~8.5 cm palm)",
            text_color=COLOR_TEXT_DIM,
        )
        self.log_event("Hand calibration reset to factory defaults.")

    def _update_3d_viewport(self, model: Optional[HandModel3D]) -> None:
        """Render 3D skeletal projection and update spatial telemetry (runs on UI thread)."""
        now = time.perf_counter()
        if now - self._last_3d_render_time < 0.030:  # ~33 FPS max for 3D viewport
            return
        self._last_3d_render_time = now

        # Calibration sampling
        if self.calibrator.is_collecting and model and model.is_valid:
            done = self.calibrator.add_sample(model.joints_raw)
            if done:
                p = self.calibrator.profile
                self.lbl_calibration_status.configure(
                    text=f"Calibrated: Palm {p.palm_scale * 100:.1f} cm ({p.sample_count} frames)",
                    text_color=COLOR_GREEN,
                )
                self.log_event(f"Hand calibration complete! User palm scale: {p.palm_scale * 100:.1f} cm.")
            else:
                count = len(self.calibrator._sample_palm_scales)
                self.lbl_calibration_status.configure(
                    text=f"Calibrating... {count}/{self.calibrator.target_samples} frames",
                    text_color=COLOR_ORANGE,
                )

        # Render 3D Wireframe Canvas
        rgb_canvas = self.skeleton_renderer_3d.render(model)
        pil_img = Image.fromarray(rgb_canvas)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(340, 220))
        self.viewport_3d_label.configure(image=ctk_img)
        self.viewport_3d_label.image = ctk_img

        # Telemetry Text
        if model and model.is_valid:
            ori = model.orientation
            self._set_widget_text(
                self.lbl_3d_orientation,
                f"Orientation: P: {ori.pitch_deg:+.0f}° | Y: {ori.yaw_deg:+.0f}° | R: {ori.roll_deg:+.0f}° | {ori.facing.value.replace('FACING_', '')}",
                COLOR_CYAN,
            )
            f = model.angles.fingers
            if f:
                idx_f = f.get('Index')
                mid_f = f.get('Middle')
                rng_f = f.get('Ring')
                pky_f = f.get('Pinky')
                thb_f = f.get('Thumb')
                idx_deg = idx_f.total_flexion_deg if idx_f else 0.0
                mid_deg = mid_f.total_flexion_deg if mid_f else 0.0
                rng_deg = rng_f.total_flexion_deg if rng_f else 0.0
                pky_deg = pky_f.total_flexion_deg if pky_f else 0.0
                thb_deg = thb_f.total_flexion_deg if thb_f else 0.0
                self._set_widget_text(
                    self.lbl_3d_angles,
                    f"3D Flexion:  Idx: {idx_deg:.0f}°  Mid: {mid_deg:.0f}°  Rng: {rng_deg:.0f}°  Pky: {pky_deg:.0f}°  Thb: {thb_deg:.0f}°",
                )
            sp = model.spatial
            scale_ratio = self.calibrator.get_scale_ratio(model.palm_scale)
            self._set_widget_text(
                self.lbl_3d_kinematics,
                f"Kinematics:  Vel: ({sp.velocity_3d[0]:+.2f}, {sp.velocity_3d[1]:+.2f}, {sp.velocity_3d[2]:+.2f}) m/s | Rel Scale: {scale_ratio:.2f}",
            )
        else:
            self._set_widget_text(
                self.lbl_3d_orientation,
                "Orientation: STANDBY (No 3D Hand Detected)",
                COLOR_TEXT_DIM,
            )
            self._set_widget_text(
                self.lbl_3d_angles,
                "3D Flexion:  Idx: --°  Mid: --°  Rng: --°  Pky: --°  Thb: --°",
                COLOR_TEXT_DIM,
            )
            self._set_widget_text(
                self.lbl_3d_kinematics,
                "Kinematics:  Velocity: (0.00, 0.00, 0.00) m/s | Scale: 1.00",
                COLOR_TEXT_DIM,
            )

    # ==========================================================================
    # CAMERA CONTROLS
    # ==========================================================================
    def _on_start_clicked(self) -> None:
        """User clicked 'Start Camera' button."""
        if self.camera_manager.is_running:
            return

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_pill.configure(text="● STARTING SENSOR...", text_color=COLOR_ORANGE)
        self.badge_cam_status.configure(text="STARTING...", text_color=COLOR_ORANGE)
        self.log_event(f"Engaging optical sensor on Camera {self.camera_manager.camera_index}...")

        # CameraManager.start() spawns its own background worker thread
        self.camera_manager.start()

    def _on_stop_clicked(self) -> None:
        """User clicked 'Stop Camera' button."""
        self.btn_stop.configure(state="disabled")
        self.btn_start.configure(state="normal")
        self.log_event("Disengaging optical sensor...")

        def _worker():
            self.camera_manager.stop()

        threading.Thread(target=_worker, daemon=True, name="AirControl-Stopper").start()

    def _post_stop_update(self) -> None:
        """Reset UI to standby following camera termination."""
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.status_pill.configure(text="○ SYSTEM STANDBY", text_color=COLOR_ORANGE)
        self.badge_cam_status.configure(text="OFFLINE", text_color=COLOR_RED)
        self.badge_fps.configure(text="0.0 FPS", text_color=COLOR_TEXT_DIM)
        self.badge_mouse_state.configure(text="STANDBY", text_color=COLOR_TEXT_DIM)
        self.badge_hands_detected.configure(text="0 / 2", text_color=COLOR_TEXT_DIM)
        self.lbl_hands_count_large.configure(text="0")

        self._reset_hand_card("Right")
        self._reset_hand_card("Left")

        self.lbl_r_gesture.configure(text="STANDBY", text_color=COLOR_TEXT_DIM)
        self.lbl_l_gesture.configure(text="STANDBY", text_color=COLOR_TEXT_DIM)
        self.lbl_r_gesture_meta.configure(text="Conf: 0% | Hold: 0.0s")
        self.lbl_l_gesture_meta.configure(text="Conf: 0% | Hold: 0.0s")
        self.lbl_dynamic_gesture.configure(text="Dynamic Motion: STANDBY", text_color=COLOR_TEXT_MUTED)
        self.lbl_dual_gesture.configure(text="Dual Interaction: STANDBY", text_color=COLOR_TEXT_DIM)

        if hasattr(self, "lbl_auto_last_action"):
            self.lbl_auto_last_action.configure(text="Last Action: STANDBY (Sensor Offline)", text_color=COLOR_TEXT_DIM)

        self._update_3d_viewport(None)

        if self._standby_image is not None:
            self.video_label.configure(image=self._standby_image)

        self.log_event("Optical sensor disengaged. System reverted to standby.")

    def _on_camera_select(self, choice: str) -> None:
        """User selected a different camera from the dropdown."""
        try:
            idx = int(choice.split()[1].strip("():"))
        except Exception:
            idx = 0

        self.log_event(f"Switching optical source to Camera {idx}...")
        self.cam_badge.configure(text=f"SENSOR: CAMERA {idx} [REQUESTED]")

        if self.camera_manager.is_running:
            def _restart_worker():
                self.camera_manager.restart(new_camera_index=idx)
            threading.Thread(target=_restart_worker, daemon=True, name="AirControl-CamSwitch").start()
            self.log_event(f"Restarting sensor on Camera {idx}...")
        else:
            self.camera_manager._camera_index = idx

    def _on_skeleton_toggle(self) -> None:
        """Toggle rendering of skeleton connections and landmarks."""
        val = bool(self.switch_skeleton.get())
        self.camera_manager.draw_skeleton = val
        self.log_event(f"Skeleton HUD overlay: {'ENABLED' if val else 'DISABLED'}")

    def _on_mirror_toggle(self) -> None:
        """Toggle horizontal frame mirroring."""
        val = bool(self.switch_mirror.get())
        self.camera_manager.mirror_feed = val
        self.log_event(f"Optical mirror mode: {'ENABLED' if val else 'DISABLED'}")

    def _on_hud_toggle(self) -> None:
        """Toggle floating hand label badges on video."""
        val = bool(self.switch_hud.get())
        self.camera_manager.draw_hud = val
        self.log_event(f"Floating HUD telemetry badges: {'ENABLED' if val else 'DISABLED'}")

    def _on_exit_clicked(self) -> None:
        """Exit button handler."""
        self.on_closing()

    def _apply_status_update(self, status: str) -> None:
        """Update UI badges based on background camera events (runs on main UI thread)."""
        if status == "ONLINE":
            w, h = self.camera_manager._actual_resolution
            cam_idx = self.camera_manager.camera_index
            self.badge_cam_status.configure(text=f"ONLINE [{w}x{h}]", text_color=COLOR_GREEN)
            self.cam_badge.configure(text=f"SENSOR: CAMERA {cam_idx} [{w}x{h}]")
            self.status_pill.configure(text="● TRACKING ENGAGED", text_color=COLOR_GREEN)
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.log_event(f"Optical sensor engaged: Camera {cam_idx} [{w}x{h}].")
        elif status == "OFFLINE":
            self._post_stop_update()
        elif status == "DISCONNECTED":
            self.badge_cam_status.configure(text="DISCONNECTED", text_color=COLOR_RED)
            self.status_pill.configure(text="✕ SENSOR LOST", text_color=COLOR_RED)
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.log_event("WARNING: Optical sensor disconnected or dropped frame stream.")
            self._post_stop_update()
        elif status.startswith("ERROR"):
            self.badge_cam_status.configure(text=status, text_color=COLOR_RED)
            self.status_pill.configure(text="✕ SENSOR ERROR", text_color=COLOR_RED)
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.log_event(f"ERROR: Optical sensor failure ({status}). Check camera permissions.")

    # ==========================================================================
    # MAIN UI REFRESH LOOP
    # ==========================================================================
    def _update_loop(self) -> None:
        """High-frequency UI update loop (~60 Hz). Protected against blocking and canvas redraw stalls."""
        if self._is_closing:
            return

        with measure_time("UI_TICK", threshold_ms=25.0):
            # Check camera status change safely on main UI thread
            curr_status = self.camera_manager.status
            if curr_status != self._last_applied_camera_status:
                self._last_applied_camera_status = curr_status
                self._apply_status_update(curr_status)

            # Drain any completed actions from the background Automation Worker thread
            if hasattr(self, "desktop_controller") and self.desktop_controller is not None:
                for res in self.desktop_controller.get_pending_results():
                    self.log_event(f"Desktop Action: {res.description} via [{res.gesture_name}]")

            try:
                if self.camera_manager.is_running:
                    frame_data: Optional[FrameData] = self.camera_manager.get_latest_frame()

                    if frame_data is not None:
                        # 1. Update Live Video Preview
                        self._render_video_frame(frame_data.display_frame)

                        # 2. Update Metric Chips (with dirty-checking)
                        self._set_widget_text(self.badge_fps, f"{frame_data.fps:.1f} FPS")

                        # 3. Update Hand Detection Counter
                        num_hands = frame_data.tracking_result.num_hands
                        hands_col = COLOR_GREEN if num_hands > 0 else COLOR_TEXT_MUTED
                        large_col = COLOR_GREEN if num_hands > 0 else COLOR_CYAN
                        self._set_widget_text(self.badge_hands_detected, f"{num_hands} / 2", hands_col)
                        self._set_widget_text(self.lbl_hands_count_large, str(num_hands), large_col)

                        # Log hand transitions
                        if num_hands != self._last_hand_count:
                            if num_hands > 0:
                                labels = [h.handedness for h in frame_data.tracking_result.hands]
                                self.log_event(f"Hand tracking locked: {num_hands} hand(s) [{', '.join(labels)}].")
                            else:
                                self.log_event("Optical tracking lost: No hands in frame.")
                            self._last_hand_count = num_hands

                        # 4. Update Dedicated Left / Right Hand Telemetry Cards
                        right_hand: Optional[HandData] = None
                        left_hand: Optional[HandData] = None

                        for h in frame_data.tracking_result.hands:
                            if h.handedness == "Right" and right_hand is None:
                                right_hand = h
                            elif h.handedness == "Left" and left_hand is None:
                                left_hand = h

                        if right_hand is not None:
                            self._update_hand_card("Right", right_hand)
                        else:
                            self._reset_hand_card("Right")

                        if left_hand is not None:
                            self._update_hand_card("Left", left_hand)
                        else:
                            self._reset_hand_card("Left")

                        # 5. Update Air Mouse Telemetry Card
                        mouse_tel: Optional[AirMouseTelemetry] = frame_data.air_mouse_telemetry or self.air_mouse.get_telemetry()
                        self._update_air_mouse_telemetry(mouse_tel)

                        # 6. Update Gesture Intelligence Card (Stage 3)
                        gesture_tel: Optional[GestureTelemetry] = frame_data.gesture_telemetry
                        self._update_gesture_intelligence_telemetry(gesture_tel)

                        # 7. Update Desktop Automation Card (Stage 4)
                        self._update_desktop_automation_telemetry()

                        # 8. Update 3D Articulated Hand Viewport & Spatial Telemetry (Stage 4.1)
                        primary_model: Optional[HandModel3D] = None
                        if right_hand and right_hand.hand_model_3d and right_hand.hand_model_3d.is_valid:
                            primary_model = right_hand.hand_model_3d
                        elif left_hand and left_hand.hand_model_3d and left_hand.hand_model_3d.is_valid:
                            primary_model = left_hand.hand_model_3d

                        self._update_3d_viewport(primary_model)

            except Exception as e:
                logger.error("Error in UI update loop: %s", e)

        # Schedule next tick (~16 ms -> 60 Hz) via single-scheduler guard
        self._schedule_next_update(delay_ms=16)

    def _render_video_frame(self, bgr_frame: np.ndarray) -> None:
        """Convert BGR OpenCV frame to CTkImage and update display."""
        target_w, target_h = self._preview_size
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(resized)

        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img

    # --------------------------------------------------------------------------
    # AIR MOUSE TELEMETRY UPDATE
    # --------------------------------------------------------------------------
    def _update_air_mouse_telemetry(self, tel: Optional[AirMouseTelemetry]) -> None:
        """Update Air Mouse indicators and metric readouts with dirty-checking."""
        if tel is None or not tel.enabled:
            self._set_widget_text(self.mouse_state_badge, "DISABLED", COLOR_TEXT_DIM, fg_color="#0A0F1D")
            self._set_widget_text(self.badge_mouse_state, "DISABLED", COLOR_TEXT_MUTED)
            self._set_widget_text(self.lbl_mouse_delta, "Air Mouse steering is currently disabled.")
            pos_text = f"Cursor: ({tel.cursor_x if tel else 0}, {tel.cursor_y if tel else 0}) px\nStatus: STANDBY [F8 to activate]"
            self._set_widget_text(self.lbl_mouse_pos, pos_text)
            return

        # Air Mouse is active
        state_color = COLOR_GREEN
        if tel.state in ("LEFT_CLICK", "RIGHT_CLICK"):
            state_color = COLOR_CYAN
        elif tel.state == "DRAGGING":
            state_color = COLOR_ORANGE
        elif tel.state == "CLUTCH":
            state_color = COLOR_PURPLE
        elif tel.state == "SCROLL":
            state_color = COLOR_BLUE

        self._set_widget_text(self.mouse_state_badge, tel.state, COLOR_BG_ROOT, fg_color=state_color)
        self._set_widget_text(self.badge_mouse_state, tel.state, state_color)
        self._set_widget_text(self.status_pill, f"● AIR MOUSE [{tel.state}]", COLOR_CYAN)

        # Log state transitions
        if tel.state != self._last_mouse_state:
            if tel.state in ("LEFT_CLICK", "RIGHT_CLICK", "DRAGGING", "SCROLL", "CLUTCH"):
                self.log_event(f"Air Mouse Gesture: {tel.details}")
            self._last_mouse_state = tel.state

        # Update numerical readouts
        delta_text = (
            f"Rel Delta: dx: {tel.raw_dx:+.4f}, dy: {tel.raw_dy:+.4f}\n"
            f"Velocity:  vx: {tel.smooth_vx:+.1f} px, vy: {tel.smooth_vy:+.1f} px"
        )
        self._set_widget_text(self.lbl_mouse_delta, delta_text)

        drag_str = "ACTIVE" if tel.is_dragging else "OFF"
        pos_text = (
            f"Cursor: ({tel.cursor_x}, {tel.cursor_y}) px\n"
            f"Pinch: {int((1.0 - min(1.0, tel.pinch_ratio))*100)}% | Drag: {drag_str}\n"
            f"Hand: {tel.hand_used or 'Searching...'}"
        )
        self._set_widget_text(self.lbl_mouse_pos, pos_text)

    # --------------------------------------------------------------------------
    # GESTURE INTELLIGENCE TELEMETRY UPDATE (STAGE 3)
    # --------------------------------------------------------------------------
    def _update_gesture_intelligence_telemetry(self, tel: Optional[GestureTelemetry]) -> None:
        """Update Gesture Intelligence UI widgets with dirty-checking."""
        if tel is None or not tel.enabled:
            self._set_widget_text(self.gesture_state_badge, "DISABLED", COLOR_TEXT_DIM)
            self._set_widget_text(self.lbl_r_gesture, "DISABLED", COLOR_TEXT_DIM)
            self._set_widget_text(self.lbl_l_gesture, "DISABLED", COLOR_TEXT_DIM)
            self._set_widget_text(self.lbl_r_gesture_meta, "Conf: 0% | Hold: 0.0s")
            self._set_widget_text(self.lbl_l_gesture_meta, "Conf: 0% | Hold: 0.0s")
            self._set_widget_text(self.lbl_dynamic_gesture, "Dynamic Motion: DISABLED", COLOR_TEXT_DIM)
            self._set_widget_text(self.lbl_dual_gesture, "Dual Interaction: DISABLED", COLOR_TEXT_DIM)
            return

        self._set_widget_text(self.gesture_state_badge, "ACTIVE", COLOR_GREEN)

        # Right Hand Pose Update
        r_state = tel.hands.get("Right")
        if r_state:
            pose_name = r_state.current_pose
            col = COLOR_CYAN if pose_name != "UNKNOWN" else COLOR_TEXT_MUTED
            self._set_widget_text(self.lbl_r_gesture, pose_name, col)
            conf_pct = int(r_state.confidence * 100)
            hold_str = f"{r_state.hold_duration_sec:.1f}s"
            if r_state.is_holding:
                hold_str += " [HOLD]"
            meta_text = f"Conf: {conf_pct}% | Hold: {hold_str} | {r_state.palm_direction}"
            self._set_widget_text(self.lbl_r_gesture_meta, meta_text)
        else:
            self._set_widget_text(self.lbl_r_gesture, "NO_HAND", COLOR_TEXT_DIM)
            self._set_widget_text(self.lbl_r_gesture_meta, "Conf: 0% | Hold: 0.0s")

        # Left Hand Pose Update
        l_state = tel.hands.get("Left")
        if l_state:
            pose_name = l_state.current_pose
            col = COLOR_PURPLE if pose_name != "UNKNOWN" else COLOR_TEXT_MUTED
            self._set_widget_text(self.lbl_l_gesture, pose_name, col)
            conf_pct = int(l_state.confidence * 100)
            hold_str = f"{l_state.hold_duration_sec:.1f}s"
            if l_state.is_holding:
                hold_str += " [HOLD]"
            meta_text = f"Conf: {conf_pct}% | Hold: {hold_str} | {l_state.palm_direction}"
            self._set_widget_text(self.lbl_l_gesture_meta, meta_text)
        else:
            self._set_widget_text(self.lbl_l_gesture, "NO_HAND", COLOR_TEXT_DIM)
            self._set_widget_text(self.lbl_l_gesture_meta, "Conf: 0% | Hold: 0.0s")

        # Dynamic Gesture Action
        dyn = tel.active_dynamic_gesture
        if dyn:
            hand_tag = tel.dynamic_gesture_hand or ""
            det = tel.dynamic_gesture_details or dyn
            dyn_text = f"◈ Motion: {dyn} [{hand_tag.upper()}] — {det}"
            self._set_widget_text(self.lbl_dynamic_gesture, dyn_text, COLOR_GREEN)
            if dyn != self._last_dynamic_gesture_logged:
                self.log_event(f"Dynamic Motion: {dyn} on {hand_tag} Hand ({det})")
                self._last_dynamic_gesture_logged = dyn
        else:
            self._set_widget_text(
                self.lbl_dynamic_gesture,
                "Dynamic Motion: STANDBY (Ready for Swipes, Wave, Circle)",
                COLOR_TEXT_MUTED,
            )
            self._last_dynamic_gesture_logged = None

        # Dual Hand Gesture Action
        dual = tel.active_dual_gesture
        if dual:
            det = tel.dual_gesture_details or dual
            dual_text = f"◈ Dual Interaction: {dual} — {det}"
            self._set_widget_text(self.lbl_dual_gesture, dual_text, COLOR_PURPLE)
            if dual != self._last_dual_gesture_logged:
                self.log_event(f"Dual Interaction: {dual} ({det})")
                self._last_dual_gesture_logged = dual
        else:
            self._set_widget_text(
                self.lbl_dual_gesture,
                "Dual Interaction: STANDBY (Dual Palms / Zoom / Box)",
                COLOR_TEXT_DIM,
            )
            self._last_dual_gesture_logged = None

    # --------------------------------------------------------------------------
    # DESKTOP AUTOMATION TELEMETRY UPDATE (STAGE 4)
    # --------------------------------------------------------------------------
    def _update_desktop_automation_telemetry(self) -> None:
        """Update Desktop Automation card widgets with dirty-checking."""
        if not hasattr(self, "lbl_auto_last_action") or not hasattr(self, "desktop_controller"):
            return

        if self.desktop_controller.enabled:
            self._set_widget_text(self.automation_state_badge, "ACTIVE", COLOR_GREEN)
            action_text = f"Last Action: {self.desktop_controller.last_description}"
            self._set_widget_text(self.lbl_auto_last_action, action_text, COLOR_GREEN)
            stats_text = f"Executed: {self.desktop_controller.total_executed} actions | Safety Cooldown: ACTIVE | Audio: ON"
            self._set_widget_text(self.lbl_auto_stats, stats_text)
        else:
            self._set_widget_text(self.automation_state_badge, "DISABLED", COLOR_TEXT_DIM)
            self._set_widget_text(self.lbl_auto_last_action, "Last Action: STANDBY (Automation Disengaged)", COLOR_TEXT_DIM)

    # --------------------------------------------------------------------------
    # HAND TELEMETRY CARD UPDATES
    # --------------------------------------------------------------------------
    def _update_hand_card(self, hand_type: str, hand: HandData) -> None:
        """Populate telemetry card with live hand measurements with dirty checking."""
        accent_color = COLOR_CYAN if hand_type == "Right" else COLOR_PURPLE

        badge = self.right_badge if hand_type == "Right" else self.left_badge
        conf_val = self.right_conf_val if hand_type == "Right" else self.left_conf_val
        conf_bar = self.right_conf_bar if hand_type == "Right" else self.left_conf_bar
        palm_lbl = self.right_palm_lbl if hand_type == "Right" else self.left_palm_lbl
        ft_labels = self.right_ft_labels if hand_type == "Right" else self.left_ft_labels

        # Badge
        self._set_widget_text(badge, f"{hand_type.upper()} ACTIVE", COLOR_BG_ROOT, fg_color=accent_color)

        # Confidence
        pct = int(hand.confidence * 100)
        self._set_widget_text(conf_val, f"{pct}%", accent_color)
        conf_bar.configure(progress_color=accent_color)
        conf_bar.set(hand.confidence)

        # Palm Center
        px, py = hand.palm_center
        self._set_widget_text(palm_lbl, f"Palm: ({px}, {py}) px", COLOR_TEXT_MAIN)

        # 5 Finger Extension State Chips
        for name, lbl in ft_labels.items():
            is_open = hand.finger_states.get(name, True)
            status_text = f"{name[:1]}:{'OPEN' if is_open else 'CURL'}"
            lbl_color = COLOR_GREEN if is_open else COLOR_TEXT_DIM
            chip_bg = "#18233C" if is_open else "#0B101E"
            self._set_widget_text(lbl, status_text, lbl_color, fg_color=chip_bg)

    def _reset_hand_card(self, hand_type: str) -> None:
        """Reset a hand card to its inactive/dimmed state with dirty checking."""
        badge = self.right_badge if hand_type == "Right" else self.left_badge
        conf_val = self.right_conf_val if hand_type == "Right" else self.left_conf_val
        conf_bar = self.right_conf_bar if hand_type == "Right" else self.left_conf_bar
        palm_lbl = self.right_palm_lbl if hand_type == "Right" else self.left_palm_lbl
        ft_labels = self.right_ft_labels if hand_type == "Right" else self.left_ft_labels

        self._set_widget_text(badge, "NOT DETECTED", COLOR_TEXT_DIM, fg_color="#0A0F1D")
        self._set_widget_text(conf_val, "0.0%", COLOR_TEXT_DIM)
        conf_bar.set(0.0)
        self._set_widget_text(palm_lbl, "Palm Center: (---, ---) px", COLOR_TEXT_DIM)
        for name, lbl in ft_labels.items():
            self._set_widget_text(lbl, f"{name[:1]}:--", COLOR_TEXT_DIM, fg_color="#0B101E")

    # ==========================================================================
    # CLEAN APPLICATION SHUTDOWN
    # ==========================================================================
    def on_closing(self) -> None:
        """Cleanly terminate camera, air mouse, vision tracker, and UI without hangs."""
        if self._is_closing:
            return

        self._is_closing = True
        logger.info("Application shutdown initiated...")

        # Cancel recurring update loop job
        if self._update_job is not None:
            try:
                self.after_cancel(self._update_job)
            except Exception:
                pass
            self._update_job = None

        try:
            # 1. Disable and reset Air Mouse
            if self.air_mouse:
                self.air_mouse.enabled = False
                self.air_mouse.reset()

            # 2. Stop automation worker thread
            if hasattr(self, "desktop_controller") and self.desktop_controller:
                self.desktop_controller.stop(timeout=0.5)

            # 3. Stop background camera & vision worker threads
            if self.camera_manager:
                self.camera_manager.stop(timeout=0.5)

            # 4. Release MediaPipe Tasks model
            if self.camera_manager and self.camera_manager.hand_tracker:
                self.camera_manager.hand_tracker.close()

        except Exception as e:
            logger.warning("Error during resource release on shutdown: %s", e)
        finally:
            self.destroy()
            logger.info("AirControl desktop application terminated cleanly.")
