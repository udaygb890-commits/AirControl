# ◈ AIRCONTROL v4.1 — 3D HAND TRACKING & SPATIAL GESTURE ENGINE

> **Futuristic, high-performance 3D hand tracking, articulated skeletal kinematics, relative & spatial air mouse, gesture intelligence, and native Windows 11 desktop automation system.**

[![Python](https://img.shields.io/badge/Python-3.11%2B%20(64--bit)-00F0FF?style=for-the-badge&logo=python&logoColor=black)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-0070F3?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![PyAutoGUI](https://img.shields.io/badge/PyAutoGUI-0.9.54%20Automation-00FFA3?style=for-the-badge)](https://pyautogui.readthedocs.io/)
[![Win32 API](https://img.shields.io/badge/Win32-ctypes%20user32-FF0055?style=for-the-badge)](https://learn.microsoft.com/en-us/windows/win32/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%20DirectShow-0070F3?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![CustomTkinter](https://img.shields.io/badge/CustomTkinter-Dark%20HUD-A855F7?style=for-the-badge)](https://github.com/TomSchimansky/CustomTkinter)
[![Tests](https://img.shields.io/badge/Tests-60%2F60%20Passing-00FFA3?style=for-the-badge)]()

---

## 1. Project Overview

**AirControl** is a futuristic, gesture-controlled desktop automation system designed for Windows 11. The core vision of AirControl is to allow users to navigate, manipulate, and control their Windows desktop environment completely hands-free through spatial gestures detected by a standard laptop webcam.

### What's New in v4.1 (3D Hand Tracking & Articulated Kinematics)
- **Articulated 3D Skeletal Hierarchy**: Moves beyond 2D pixel coordinates by utilizing MediaPipe's 21 3D world landmarks in real-world metric space. Computes 23 anatomical bone segments, 3D lengths, unit direction vectors, and kinematic parent-child hierarchies.
- **Biomechanical Joint Angles & Flexion**: True 3D angular flexion/extension for MCP, PIP, and DIP joints across all 5 digits ($0^\circ = \text{straight}$), thumb opposition angle across the palm plane, and inter-finger abduction/spread angles.
- **3D Palm Orientation & Normal Vector**: Derives unit normal vector $\vec{N}$ emerging perpendicular from the palm surface with handedness correction, pitch/yaw/roll Euler angles, and cardinal facing directions (`FACING_CAMERA`, `FACING_AWAY`, `FACING_UP`, etc.).
- **Scale- & Rotation-Invariant Canonical Frame**: Transforms hand coordinates into a local coordinate frame (Wrist at origin, Middle MCP along +Y normalized to 1.0, Palm Normal along +Z) that is 100% invariant to camera distance, tilt, and hand translation.
- **One-Euro Adaptive Filtering & Outlier Rejection**: Real-time jitter elimination at rest with near-zero lag during fast motion, combined with physical displacement spike clamping.
- **Interactive 3D Viewport**: Real-time perspective projection wireframe renderer with depth shading, 3D coordinate floor grid, orbit camera azimuth/elevation/zoom controls, and live spatial telemetry.
- **User Hand Calibration**: Interactive open-palm calibration routine to estimate personalized palm scale and finger segment lengths.
- **3D Spatial Cursor Mode**: Optional Air Mouse mode driven by physical centimeter hand displacement in 3D metric space.

---

## 2. System Architecture

```
AirControl/
│
├── main.py                     # Application entry point with DPI awareness & thread coordination
├── config.py                   # Central configuration (Camera, Air Mouse, Gestures, Automation)
├── requirements.txt            # Project dependencies (OpenCV, MediaPipe, CustomTkinter, PyAutoGUI)
├── test_aircontrol.py          # Complete unit & integration test suite (60 tests, 100% passing)
├── README.md                   # Comprehensive system documentation
│
├── hand3d/                     # STAGE 4.1: 3D ARTICULATED HAND & SPATIAL ENGINE
│   ├── __init__.py             # 3D module exports
│   ├── skeleton.py             # 21 joints, 23 bones, parent-child kinematic hierarchy
│   ├── joint_angles.py         # 3D MCP/PIP/DIP flexion, thumb opposition & abduction angles
│   ├── hand_orientation.py     # 3D palm normal vector, Euler angles (pitch/yaw/roll), facing
│   ├── spatial_features.py     # Canonical local frame, 3D fingertip distances, 3D velocity
│   ├── hand_filter.py          # One-Euro adaptive 3D temporal filter & outlier rejection
│   ├── calibration.py          # Open-palm reference pose & personalized scale calibrator
│   ├── hand_model.py           # Consolidated HandModel3D & Hand3DProcessor stream pipeline
│   └── renderer_3d.py          # Real-time perspective projection wireframe & joint renderer
│
├── desktop/                    # STAGE 4: WINDOWS DESKTOP AUTOMATION
│   ├── __init__.py             # Desktop automation exports
│   ├── command_config.py       # DesktopCommand enum, default gesture mappings & cooldown tables
│   ├── window_manager.py       # Win32 user32 & PyAutoGUI window manipulation engine
│   ├── system_controls.py      # Audio volume, screenshot capture & media playback controls
│   ├── command_dispatcher.py   # Cooldown tracking, hold duration validation & dispatch routing
│   └── desktop_controller.py   # High-level automation coordinator, safety state & telemetry
│
├── gestures/                   # STAGE 3: GESTURE INTELLIGENCE ENGINE
│   ├── __init__.py             # Gesture module exports
│   ├── gesture_classifier.py   # Scale-invariant static pose classification (12+ gestures)
│   └── gesture_engine.py       # Temporal trajectories, dynamic swipes, wave, circle & dual-hand
│
├── input_control/              # STAGE 2: RELATIVE AIR MOUSE ENGINE
│   ├── __init__.py             # Input control exports
│   ├── air_mouse.py            # AirMouse coordinator (Relative 2D & Spatial 3D cursor modes)
│   ├── cursor_smoother.py      # Exponential smoothing, deadzone & boundary clamping
│   ├── gesture_controller.py   # State machine for pinch, drag, right-click, scroll & clutch
│   └── macro_engine.py         # System command dispatch & cyber audio tones
│
├── vision/                     # STAGE 1: VISION SUBSYSTEM
│   ├── __init__.py             # Vision package exports
│   ├── camera.py               # Asynchronous CameraManager & OpenCV DirectShow capture
│   └── hand_tracker.py         # MediaPipe Tasks API, 3D world landmarks extraction & HUD
│
├── interface/                  # USER INTERFACE
│   ├── __init__.py             # Interface package exports
│   └── dashboard.py            # Futuristic CustomTkinter dark dashboard with 3D viewport
│
├── utils/                      # UTILITIES
│   ├── __init__.py             # Utilities package exports
│   ├── fps.py                  # Thread-safe sliding-window FPS & latency counter
│   └── diagnostics.py          # Real-time worker thread profiling & timing diagnostics
│
└── assets/                     # ASSETS
    ├── hand_landmarker.task    # Google MediaPipe Hand Landmarker model
    ├── icon.png                # Futuristic cyber application icon
    └── icon.ico                # Multi-resolution Windows application icon
```
│   ├── camera.py               # Asynchronous CameraManager & OpenCV DirectShow capture
│   └── hand_tracker.py         # MediaPipe Tasks API, landmark extraction & HUD rendering
│
├── interface/                  # USER INTERFACE
│   ├── __init__.py             # Interface package exports
│   └── dashboard.py            # Futuristic CustomTkinter dark desktop dashboard with HUD
│
├── utils/                      # UTILITIES
│   ├── __init__.py             # Utilities package exports
│   ├── fps.py                  # Thread-safe sliding-window FPS & latency counter
│   └── diagnostics.py          # Real-time worker thread profiling & timing diagnostics
│
└── assets/                     # ASSETS
    ├── hand_landmarker.task    # Google MediaPipe Hand Landmarker model
    ├── icon.png                # Futuristic cyber application icon
    └── icon.ico                # Multi-resolution Windows application icon
```

---

## 3. Technology Stack

| Component | Technology | Specification / Role |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ (64-bit) | Modern type-hinted Python with high performance |
| **Desktop Automation** | PyAutoGUI + `ctypes.windll.user32` | Direct Windows 11 mouse control, hotkeys, and Win32 window management |
| **Vision Engine** | Google MediaPipe Tasks API | `HandLandmarker` in `RunningMode.VIDEO` with dual-hand tracking |
| **Webcam Capture** | OpenCV (`cv2`) | DirectShow (`CAP_DSHOW`) capture at 1280×720 @ 30 FPS |
| **Numerical Processing** | NumPy | Coordinate transformations & vector calculations |
| **GUI Framework** | CustomTkinter | Modern dark theme, rounded cards, responsive vector widgets |
| **Frame Rendering** | Pillow (`PIL`) | High-DPI `CTkImage` rendering for UI viewport |
| **Threading & Safety** | `threading` + Win32 Events | Lock-free thread-safe queues and non-blocking command execution |

---

## 4. Stage 4: Gesture-to-Command Mappings

All recognized spatial gestures map directly to native Windows 11 desktop commands with built-in cooldowns and safety mechanisms:

| Gesture | Desktop Command | Windows Action / Keystroke | Cooldown / Safety |
| :--- | :--- | :--- | :--- |
| **`THUMBS_DOWN`** | `MINIMIZE_ACTIVE_WINDOW` | Minimizes foreground window (`Win + Down` / Win32) | 0.80s cooldown |
| **`TWO_HANDS_X`** | `CLOSE_ACTIVE_WINDOW` | Gracefully closes active window (`Alt + F4` / Win32) | **1.50s critical safety cooldown** |
| **`THUMBS_UP`** | `TOGGLE_MAXIMIZE_RESTORE` | Toggles maximize or restore on active window | 0.80s cooldown |
| **`OK_SIGN` (`PINCH_OK`)** | `SWITCH_WINDOWS` | Switches to next window application (`Alt + Tab`) | 0.40s cooldown |
| **`OPEN_PALM_HELD`** | `SHOW_DESKTOP` | Minimizes all windows to desktop (`Win + D`) | **Requires 0.35s continuous hold** |
| **`SWIPE_LEFT`** | `PREVIOUS_DESKTOP` | Switches to left virtual desktop (`Win + Ctrl + Left`) | 0.60s cooldown |
| **`SWIPE_RIGHT`** | `NEXT_DESKTOP` | Switches to right virtual desktop (`Win + Ctrl + Right`) | 0.60s cooldown |
| **`SWIPE_UP`** | `VOLUME_UP` | Increases master system volume (`Volume Up`) | 0.12s fast cooldown |
| **`SWIPE_DOWN`** | `VOLUME_DOWN` | Decreases master system volume (`Volume Down`) | 0.12s fast cooldown |
| **`ROCK_ON`** | `SCREENSHOT` | Takes full screenshot to Pictures (`Win + PrtScn`) | 1.00s cooldown + audio chirp |
| **`WAVE`** | `TASK_VIEW` | Opens Windows Task View / virtual desktops (`Win + Tab`) | 0.80s cooldown |
| **`CALL_ME`** | `PLAY_PAUSE` | Toggles media playback (`Play/Pause`) | 0.50s cooldown |

---

## 5. Safety & System Control Hotkeys

AirControl includes hardware-level emergency halts and individual subsystem toggles:

| Hotkey | Target Subsystem | Action |
| :--- | :--- | :--- |
| **`F8`** | **Relative Air Mouse** | Toggles gesture cursor steering on/off |
| **`F9`** | **Desktop Automation** | Toggles Windows command execution on/off |
| **`Escape`** | **Master Emergency Halt** | Instantly disables **BOTH** Air Mouse and Desktop Automation |

---

## 6. Stage 2 Relative Air Mouse Controls

When Air Mouse is engaged (`F8`), relative index fingertip tracking controls the Windows cursor with sub-pixel precision:

| Hand Action | Mouse Behavior | Mechanics |
| :--- | :--- | :--- |
| **Index Fingertip Movement** | Relative Cursor Steering | Frame-to-frame displacement $\Delta x, \Delta y$ with exponential smoothing and deadzone filtering |
| **Index-Middle Touch** (or Thumb Pinch) | Left Click / Drag | **Touching Middle finger with Index finger** ($d < 0.10$) triggers Left Click; holding touch engages drag |
| **Thumb-Middle Pinch** | Right Click | Secondary pinch threshold ($d < 0.070$) triggers context menu click |
| **Index + Middle Parallel** | Two-Finger Scroll | Vertical motion of adjacent fingers (with gap between them) scrolls documents smoothly |
| **Closed Fist** | Clutch / Freeze | Locks cursor position instantly to reposition hand without cursor drift |

---

## 7. Stage 3 Gesture Recognition Catalog

### Static Hand Poses
- **`OPEN_PALM`**: All 5 fingers extended straight ($c < 0.35$).
- **`FIST`**: All fingers curled tight ($c > 0.60$).
- **`POINTING`**: Index extended, Middle/Ring/Pinky curled.
- **`VICTORY`**: Index & Middle extended in V-shape.
- **`THUMBS_UP`**: 4 fingers curled, thumb extended upward.
- **`THUMBS_DOWN`**: 4 fingers curled, thumb extended downward.
- **`ROCK_ON`**: Index & Pinky extended, Middle & Ring curled.
- **`SPIDERMAN`**: Thumb, Index, Pinky extended, Middle & Ring curled.
- **`PINCH_OK`**: Thumb Tip (4) touches Index Tip (8), others extended.
- **`CALL_ME`**: Thumb and Pinky extended, middle three curled.
- **`THREE_FINGERS`**: Index, Middle, Ring extended.
- **`FOUR_FINGERS`**: All 4 fingers extended, Thumb tucked.

### Dynamic Motion Gestures
- **`SWIPE_LEFT` / `SWIPE_RIGHT`**: Rapid horizontal hand motion ($v_x \ge 0.32, \Delta x \ge 0.10$).
- **`SWIPE_UP` / `SWIPE_DOWN`**: Rapid vertical hand motion ($v_y \ge 0.32, \Delta y \ge 0.10$).
- **`WAVE`**: Rhythmic horizontal oscillation ($\ge 2$ reversals in Open Palm).
- **`CIRCLE_CW` / `CIRCLE_CCW`**: Clockwise or counter-clockwise circular trajectories ($\ge 270^\circ$).
- **`PUSH_TAP`**: Rapid forward z-axis displacement toward camera ($\Delta z < -0.075$).

### Dual-Hand Interactions
- **`TWO_HANDS_X`**: Crossed arms / wrists forming an X gesture (triggers safe active window close).
- **`DUAL_OPEN_PALMS`**: Both hands simultaneously displaying Open Palm.
- **`DUAL_FISTS`**: Both hands simultaneously clenched in Fists.
- **`PINCH_ZOOM_IN` / `PINCH_ZOOM_OUT`**: Distance between hands expanding or contracting.

---

## 8. Installation & Quick Start

### 1. Prerequisites
Ensure Python 3.11+ (64-bit) is installed on Windows 11.

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```

### 4. Run the Full Test Suite
```bash
python -m unittest test_aircontrol.py
```

---

## 9. User Interface Guide

### Left Panel (Sensor Viewport)
- **Live Video Display**: Displays 1280×720 mirrored camera feed with real-time skeleton HUD and cursor tracking reticles.
- **Metric Chips**:
  - `CAMERA`: Sensor status & negotiated resolution.
  - `VIDEO FPS`: Real-time frame rate (~30 FPS).
  - `AIR MOUSE`: Live mouse state (`IDLE`, `MOVING`, `LEFT_CLICK`, `DRAGGING`, `SCROLL`, `CLUTCH`, `DISABLED`).
  - `HANDS DETECTED`: Real-time hand count (`0 / 2`, `1 / 2`, `2 / 2`).
- **Action Buttons**:
  - `▶ START CAMERA`: Engages background capture and tracking pipeline.
  - `■ STOP CAMERA`: Disengages camera and cleanly releases hardware handles.
  - `◈ AIR MOUSE [F8]`: Master toggle for gesture cursor control.
  - `✕ EXIT`: Gracefully terminates all threads and closes application.

### Right Telemetry Panel
- **Card 1: Air Mouse Control Card**:
  - Master Switch: `Engage Air Mouse [F8]`
  - Real-time Displacement: $\Delta x, \Delta y$ and smoothed velocities $v_x, v_y$.
  - Current Screen Cursor Position: $(X, Y)$ in pixels.
  - Controlling Hand Dropdown: Select `Right Hand (Default)`, `Left Hand`, or `Any Hand`.
  - Sensitivity Slider: Adjustable from $0.5\times$ to $5.0\times$ with live readout.
  - Deadzone Slider: Adjustable from $0.001$ to $0.012$ with live readout.
- **Card 2: 3D Articulated Skeleton & Spatial Engine (v4.1)**:
  - Real-time 3D Perspective Projection Viewport with cyber dark background and floor grid.
  - Interactive Camera Orbit: Azimuth Slider (-180° to +180°), Elevation Slider (-85° to +85°), Reset View.
  - Calibration Controls: Open Palm hand dimension calibration and reset buttons.
  - 3D Telemetry: Live Palm Normal $[N_x, N_y, N_z]$, Euler Angles (Pitch, Yaw, Roll), Facing direction, 5-finger 3D flexion angles, and 3D velocity.
  - Switch: `3D Spatial Cursor Mode` for physical hand displacement control in metric space.
- **Card 3: Gesture Intelligence Card**:
  - Real-time Static Pose & Dynamic Gesture classification badges.
  - Hand trajectory meters, hold duration progress bar, and dual-hand interaction status.
- **Card 4: Windows Desktop Automation Card**:
  - Master Switch: `Enable Automation [F9]`
  - Automation State Badge: `ACTIVE` (Neon Green), `DISABLED` (Muted), or `COOLDOWN` (Amber).
  - Last Executed Command Banner: Displays command name, timestamp, and triggering gesture.
  - Execution Counter: Tracks total lifetime automation actions safely performed.
  - Quick Reference Mapping Chips: Color-coded chips displaying all active gesture shortcuts.
- **Card 5: Detection Engine Card**: Engine specifications, mode, and hand capacity counter.
- **Card 6 & 7: Dedicated Hand Telemetry Cards**: Styled in Electric Cyan (`#00F0FF`) and Cyber Purple (`#A855F7`), showing confidence, palm 3D coordinates, and 5 finger state chips.
- **Card 8: Hardware & HUD Controls**: Camera selector, skeleton switch, mirror switch, and floating labels switch.
- **Card 9: System Event Console**: Monospaced terminal logging timestamped events, state transitions, and gesture dispatches.

---

## 10. Automated Test Suite

The automated test suite (`test_aircontrol.py`) contains **60 test cases** covering all stages, multi-worker thread decoupling, and 3D spatial kinematics with a 100% pass rate:
- **`TestFPSCounter`** (3 tests): FPS sliding window, smoothed calculation, zero frames safety.
- **`TestHandTracker`** (5 tests): MediaPipe model download/existence, blank frame processing, landmark normalization, skeleton HUD rendering.
- **`TestCursorSmoother`** (5 tests): Dead-zone filtering, 4-way movement directionality, dynamic acceleration, speed clamping, and screen boundary clamping.
- **`TestGestureController`** (5 tests): Index-middle touch left click, pinch click, drag threshold, right click, scroll mode, and fist clutch.
- **`TestAirMouse`** (5 tests): Movement integration, smooth velocity propagation, tracking drop recovery, and re-acquisition safety.
- **`TestCameraManager`** (3 tests): Background capture thread lifecycle, clean camera release, and frame queue retrieval.
- **`TestGestureClassifier`** (4 tests): Landmark extraction, static poses (`OPEN_PALM`, `FIST`, `THUMBS_UP`, `THUMBS_DOWN`, `VICTORY`, `PINCH_OK`, etc.).
- **`TestGestureEngine`** (6 tests): Dynamic swipes (`SWIPE_LEFT`, `SWIPE_RIGHT`, `SWIPE_UP`, `SWIPE_DOWN`), `WAVE` oscillation, circular trajectories, and dual-hand `TWO_HANDS_X`.
- **`TestWindowManager`** (4 tests): Active window title retrieval, maximize/restore toggling, minimize, show desktop, and virtual desktop switching.
- **`TestSystemControls`** (2 tests): Volume up/down, screenshot capture, media playback controls.
- **`TestCommandDispatcher`** (2 tests): Gesture-to-command dispatch, confidence thresholding, hold verification, cooldown prevention.
- **`TestDesktopController`** (4 tests): Master automation switch, emergency halt (`Esc`), execution telemetry, asynchronous worker execution, and queue cancellation.
- **`TestThreadArchitecture`** (2 tests): Concurrent Camera Worker and Vision Worker execution, repeated camera start/stop cycles without thread leaks.
- **`TestHand3D`** (10 tests):
  - 23-bone articulated skeleton calculation and direction unit vectors.
  - True 3D flexion/extension angles for straight fingers ($<45^\circ$) vs curled fingers ($>60^\circ$).
  - 3D palm normal vector derivation and cardinal facing classification (`FACING_CAMERA`).
  - Canonical local coordinate transformation invariant to camera distance and tilt.
  - One-Euro adaptive temporal filtering and single-frame spike clamping.
  - Interactive Open-Palm hand dimension calibrator with scale ratio estimation.
  - Complete `Hand3DProcessor` kinematic pipeline.
  - `SkeletonRenderer3D` perspective wireframe generation, orbit camera controls, and standby canvas.
  - 3D Spatial Air Mouse cursor mode driven by real-world physical displacement.

```bash
Ran 60 tests in 35.147s

OK
```

---

## 11. Monocular Depth Estimation & Coordinate Systems

> [!NOTE]
> **Depth Estimation Characteristics**:
> Google MediaPipe's `hand_world_landmarks` predicts 3D coordinates in a metric Euclidean space (in meters) centered at the hand's approximate geometric center. This prediction is derived from a deep convolutional neural network trained on synthetic and annotated articulated hand poses rather than hardware stereo cameras or time-of-flight (ToF) LiDAR sensors.
> 
> In AirControl v4.1, absolute camera distance and scale variations are normalized by projecting the hand into a **canonical local coordinate frame** anchored at the Wrist $(0,0,0)$ and scaled by the baseline palm length $S = \|\text{Middle\_MCP} - \text{Wrist}\|$. This ensures that joint flexion angles, thumb opposition, and gesture classification remain reliable and scale-invariant across different user hand sizes and webcam viewing distances.
