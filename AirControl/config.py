"""
AIRCONTROL — CONFIGURATION MODULE
Central configuration for Vision Engine, Camera, MediaPipe Tasks, and UI Theme.
"""

from pathlib import Path

# ==============================================================================
# APPLICATION METADATA
# ==============================================================================
APP_NAME = "AIRCONTROL"
APP_SUBTITLE = "GESTURE INTELLIGENCE SYSTEM"
APP_STAGE = "STAGE 4: WINDOWS DESKTOP AUTOMATION"
APP_VERSION = "4.0.0"

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = ASSETS_DIR / MODEL_FILENAME
ICON_PATH = ASSETS_DIR / "icon.ico"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# ==============================================================================
# WEBCAM & VIDEO STREAM CONFIGURATION
# ==============================================================================
DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 30
DEFAULT_MIRROR_FEED = True
MAX_QUEUE_SIZE = 2

# Worker Architecture & Bounded Queues
CAMERA_QUEUE_MAXSIZE = 2
VISION_QUEUE_MAXSIZE = 2
AUTOMATION_QUEUE_MAXSIZE = 10
WORKER_SHUTDOWN_TIMEOUT = 1.0
ENABLE_DIAGNOSTICS = True

# ==============================================================================
# MEDIAPIPE HAND LANDMARKER CONFIGURATION
# ==============================================================================
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.50
MIN_PRESENCE_CONFIDENCE = 0.50
MIN_TRACKING_CONFIDENCE = 0.50

# Hand Landmark Skeleton Connections (21 Landmarks)
HAND_CONNECTIONS = frozenset([
    # Palm Base
    (0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (17, 18), (18, 19), (19, 20)
])

# Fingertip Indices: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)

LANDMARK_NAMES = [
    "WRIST",              # 0
    "THUMB_CMC",          # 1
    "THUMB_MCP",          # 2
    "THUMB_IP",           # 3
    "THUMB_TIP",          # 4
    "INDEX_FINGER_MCP",   # 5
    "INDEX_FINGER_PIP",   # 6
    "INDEX_FINGER_DIP",   # 7
    "INDEX_FINGER_TIP",   # 8
    "MIDDLE_FINGER_MCP",  # 9
    "MIDDLE_FINGER_PIP",  # 10
    "MIDDLE_FINGER_DIP",  # 11
    "MIDDLE_FINGER_TIP",  # 12
    "RING_FINGER_MCP",    # 13
    "RING_FINGER_PIP",    # 14
    "RING_FINGER_DIP",    # 15
    "RING_FINGER_TIP",    # 16
    "PINKY_MCP",          # 17
    "PINKY_PIP",          # 18
    "PINKY_DIP",          # 19
    "PINKY_TIP",          # 20
]

# ==============================================================================
# STAGE 2: RELATIVE AIR MOUSE CONFIGURATION
# ==============================================================================
DEFAULT_AIR_MOUSE_ENABLED = False
PRIMARY_MOUSE_HAND = "Right"       # "Right", "Left", or "Any"

# Relative Movement & Sensitivity
DEFAULT_MOUSE_SENSITIVITY_X = 2.4   # Multiplier for horizontal fingertip displacement
DEFAULT_MOUSE_SENSITIVITY_Y = 2.4   # Multiplier for vertical fingertip displacement
MIN_SENSITIVITY = 0.5
MAX_SENSITIVITY = 6.0

# Cursor Smoothing & Jitter Reduction
DEFAULT_MOUSE_DEADZONE = 0.0035     # Normalized movement deadzone (filters hand micro-tremor)
MIN_DEADZONE = 0.001
MAX_DEADZONE = 0.015
DEFAULT_SMOOTHING_FACTOR = 0.35    # EMA alpha (0.0 to 1.0; lower = smoother, higher = more responsive)
MOUSE_MAX_SPEED_PX = 85.0          # Max pixels moved per frame (speed clamp against tracking glitch)
MOUSE_ACCELERATION_ENABLED = True  # Non-linear velocity curve for precision + speed
MOUSE_ACCELERATION_POWER = 1.35

# Gesture Detection Thresholds
PINCH_CLICK_THRESHOLD = 0.065      # Normalized distance between Thumb (4) and Index (8)
PINCH_RELEASE_THRESHOLD = 0.085    # Hysteresis threshold to prevent click chatter
INDEX_MIDDLE_CLICK_THRESHOLD = 0.10 # Normalized distance between Index (8) and Middle (12)
INDEX_MIDDLE_RELEASE_THRESHOLD = 0.14 # Release threshold for index-middle touch
RIGHT_PINCH_THRESHOLD = 0.070      # Normalized distance between Thumb (4) and Middle (12)
RIGHT_PINCH_RELEASE_THRESHOLD = 0.090
DRAG_HOLD_DELAY_SEC = 0.25         # Holding pinch > 0.25s engages left mouse drag
SCROLL_SENSITIVITY = 25.0          # Scroll amount multiplier per frame
SCROLL_DEADZONE = 0.006            # Vertical threshold for two-finger scroll
CLUTCH_FIST_ENABLED = True         # Curling all fingers acts as clutch (freeze cursor)

# ==============================================================================
# STAGE 3: GESTURE INTELLIGENCE CONFIGURATION
# ==============================================================================
DEFAULT_GESTURE_ENGINE_ENABLED = True
GESTURE_STABILITY_FRAMES = 3        # Consecutive frames required to lock static pose
GESTURE_HOLD_THRESHOLD_SEC = 0.25   # Time holding pose to register continuous hold
GESTURE_COOLDOWN_SEC = 0.65         # Refractory period between dynamic gestures

# Dynamic Motion Gestures (Swipes, Wave, Circular)
SWIPE_VELOCITY_THRESHOLD = 0.32     # Normalized screen distance per second
SWIPE_MIN_DISTANCE = 0.10           # Minimum displacement to trigger swipe
SWIPE_MAX_TIME_SEC = 0.45           # Maximum duration of a swipe motion
WAVE_MIN_REVERSALS = 2              # Direction reversals to recognize wave
CIRCLE_MIN_POINTS = 8               # Landmark history points to detect circular path
CIRCLE_MIN_RADIUS = 0.035           # Normalized radius for circular gesture

# Dual-Hand Interaction
DUAL_HAND_GESTURES_ENABLED = True
PINCH_ZOOM_MIN_DELTA = 0.08         # Normalized delta between hands to trigger zoom event

# ==============================================================================
# STAGE 4: WINDOWS DESKTOP AUTOMATION CONFIGURATION
# ==============================================================================
DEFAULT_AUTOMATION_ENABLED = True
DEFAULT_AUTOMATION_SOUND = True
GLOBAL_COMMAND_COOLDOWN_SEC = 0.50  # Global minimum time between any two desktop commands
COMMAND_CONFIRMATION_HOLD_SEC = 0.35 # Hold duration required for sustained gestures (OPEN_PALM)
WINDOW_CLOSE_SAFETY_COOLDOWN = 1.50 # Extra cooldown to prevent multiple window closes
MIN_COMMAND_CONFIDENCE = 0.70       # Minimum gesture confidence to execute desktop action

# ==============================================================================
# UI THEME — FUTURISTIC SCI-FI / CYBER AESTHETIC
# ==============================================================================
WINDOW_WIDTH = 1380
WINDOW_HEIGHT = 860
WINDOW_MIN_WIDTH = 1180
WINDOW_MIN_HEIGHT = 740

# Color Palette
COLOR_BG_ROOT = "#070B12"          # Deep space obsidian
COLOR_BG_CARD = "#0E1526"          # Cyber panel card
COLOR_BG_INNER = "#131C31"         # Slightly elevated surface
COLOR_BG_HEADER = "#0A0F1D"        # Header background
COLOR_BG_LOG = "#060A10"           # Event log console black

# Accents
COLOR_CYAN = "#00F0FF"             # Neon electric cyan (Right Hand & Main Accent)
COLOR_BLUE = "#0070F3"             # Hyper electric blue
COLOR_PURPLE = "#A855F7"           # Subtle cyber purple (Left Hand accent)
COLOR_GREEN = "#00FFA3"            # Active / tracking engaged
COLOR_RED = "#FF2E63"              # Stop / error / camera off
COLOR_ORANGE = "#FF9900"           # Warning / standby
COLOR_BORDER = "#1E293B"           # Subtle border
COLOR_BORDER_GLOW = "#00D4FF"      # Glowing border accent

# Typography Colors
COLOR_TEXT_MAIN = "#F8FAFC"        # Crisp white
COLOR_TEXT_MUTED = "#94A3B8"       # Slate gray
COLOR_TEXT_DIM = "#64748B"         # Dim gray

# Font Settings
FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"

