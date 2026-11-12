"""
AIRCONTROL — UNIT & INTEGRATION TEST SUITE
Verifies FPS counter, HandTracker Tasks API, CameraManager,
CursorSmoother, GestureController, AirMouse, and Dashboard integration.
"""

import time
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from config import MODEL_PATH
from desktop import (
    CommandDispatcher,
    CommandResult,
    DesktopCommand,
    DesktopController,
    SystemControls,
    WindowManager,
)
from gestures.gesture_classifier import GestureClassifier, StaticGesture
from gestures.gesture_engine import (
    DualHandGesture,
    DynamicGesture,
    GestureEngine,
    GestureEvent as EngineGestureEvent,
    GestureTelemetry,
)
from hand3d import (
    CalibrationProfile,
    HandCalibrator,
    HandFilter3D,
    HandModel3D,
    Hand3DProcessor,
    HandSkeleton,
    JointAngleCalculator,
    OrientationEstimator,
    PalmFacing,
    SkeletonRenderer3D,
    SpatialFeatureExtractor,
)
from input_control.air_mouse import AirMouse
from input_control.cursor_smoother import CursorSmoother
from input_control.gesture_controller import GestureController, GestureEvent
from utils.fps import FPSCounter
from vision.camera import CameraManager
from vision.hand_tracker import HandTracker, HandData, HandTrackingResult


class TestFPSCounter(unittest.TestCase):
    """Test frame rate calculation and smoothing."""

    def test_fps_counter(self):
        fps_counter = FPSCounter(window_size=10)
        self.assertEqual(fps_counter.fps, 0.0)

        for _ in range(5):
            fps_counter.tick()
            time.sleep(0.01)

        fps = fps_counter.fps
        self.assertGreater(fps, 0.0)
        self.assertGreater(fps_counter.latency_ms, 0.0)

        fps_counter.reset()
        self.assertEqual(fps_counter.fps, 0.0)


class TestHandTracker(unittest.TestCase):
    """Test MediaPipe Hand Landmarker Tasks API initialization and processing."""

    @classmethod
    def setUpClass(cls):
        cls.tracker = HandTracker()
        cls.initialized = cls.tracker.initialize()

    @classmethod
    def tearDownClass(cls):
        cls.tracker.close()

    def test_model_exists(self):
        self.assertTrue(MODEL_PATH.exists())
        self.assertGreater(MODEL_PATH.stat().st_size, 1_000_000)

    def test_tracker_initialized(self):
        self.assertTrue(self.initialized)

    def test_process_blank_frame(self):
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = self.tracker.process_frame(blank, 1000)
        self.assertTrue(result.is_success)
        self.assertEqual(result.num_hands, 0)
        self.assertEqual(len(result.hands), 0)
        self.assertGreater(result.inference_time_ms, 0.0)

    def test_draw_landmarks_empty(self):
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = HandTrackingResult(hands=[], num_hands=0)
        out = self.tracker.draw_landmarks(blank, result)
        self.assertEqual(out.shape, (720, 1280, 3))

    def test_draw_landmarks_synthetic_hand(self):
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        norm_lms = [(0.5 + 0.01 * i, 0.5 + 0.01 * i, 0.0) for i in range(21)]
        pix_lms = [(int(x * 1280), int(y * 720)) for x, y, _ in norm_lms]
        ft = {
            "Thumb": pix_lms[4],
            "Index": pix_lms[8],
            "Middle": pix_lms[12],
            "Ring": pix_lms[16],
            "Pinky": pix_lms[20],
        }

        synthetic_hand = HandData(
            index=0,
            handedness="Right",
            confidence=0.98,
            landmarks_norm=norm_lms,
            landmarks_pixel=pix_lms,
            fingertips_pixel=ft,
            palm_center=pix_lms[0],
            bounding_box=(600, 300, 750, 450),
        )
        result = HandTrackingResult(hands=[synthetic_hand], num_hands=1)

        out = self.tracker.draw_landmarks(blank, result, draw_skeleton=True, draw_hud=True)
        self.assertEqual(out.shape, (720, 1280, 3))
        self.assertGreater(np.count_nonzero(out), 0)


class TestCursorSmoother(unittest.TestCase):
    """Test exponential smoothing, deadzone, speed clamps, and boundary limits."""

    def setUp(self):
        self.smoother = CursorSmoother(
            sensitivity_x=2.0,
            sensitivity_y=2.0,
            deadzone=0.005,
            alpha=0.5,
            max_speed_px=80.0,
        )

    def test_deadzone_filters_micro_tremor(self):
        # Micro displacement below deadzone threshold
        vx, vy = self.smoother.smooth(raw_dx=0.002, raw_dy=0.002, screen_w=1920, screen_h=1080)
        self.assertEqual(vx, 0.0)
        self.assertEqual(vy, 0.0)

    def test_directional_fidelity(self):
        # Move Hand Right (+dx) -> +vx
        vx_r, _ = self.smoother.smooth(raw_dx=0.02, raw_dy=0.0, screen_w=1920, screen_h=1080)
        self.assertGreater(vx_r, 0.0)
        self.smoother.reset()

        # Move Hand Left (-dx) -> -vx
        vx_l, _ = self.smoother.smooth(raw_dx=-0.02, raw_dy=0.0, screen_w=1920, screen_h=1080)
        self.assertLess(vx_l, 0.0)
        self.smoother.reset()

        # Move Hand Up (-dy) -> -vy (Screen Up)
        _, vy_u = self.smoother.smooth(raw_dx=0.0, raw_dy=-0.02, screen_w=1920, screen_h=1080)
        self.assertLess(vy_u, 0.0)
        self.smoother.reset()

        # Move Hand Down (+dy) -> +vy (Screen Down)
        _, vy_d = self.smoother.smooth(raw_dx=0.0, raw_dy=0.02, screen_w=1920, screen_h=1080)
        self.assertGreater(vy_d, 0.0)

    def test_speed_limit_clamp(self):
        # Extremely large tracking glitch displacement
        vx, vy = self.smoother.smooth(raw_dx=0.50, raw_dy=0.50, screen_w=1920, screen_h=1080)
        speed = np.hypot(vx, vy)
        self.assertLessEqual(speed, 80.01)

    def test_screen_boundary_clamping(self):
        # Clamp to right screen edge
        clamped_x, clamped_y = self.smoother.clamp_to_screen(
            current_x=1910, current_y=500, dx=50, dy=0, screen_w=1920, screen_h=1080
        )
        self.assertEqual(clamped_x, 1919)
        self.assertEqual(clamped_y, 500)

        # Clamp to top-left corner
        clamped_x, clamped_y = self.smoother.clamp_to_screen(
            current_x=10, current_y=10, dx=-50, dy=-50, screen_w=1920, screen_h=1080
        )
        self.assertEqual(clamped_x, 0)
        self.assertEqual(clamped_y, 0)


class TestGestureController(unittest.TestCase):
    """Test gesture classification: Pinch Left Click, Right Click, Scroll, Clutch."""

    def setUp(self):
        self.controller = GestureController()

    def _create_hand(self, norm_lms, handedness="Right"):
        pix_lms = [(int(x * 1280), int(y * 720)) for x, y, _ in norm_lms]
        return HandData(
            index=0,
            handedness=handedness,
            confidence=0.95,
            landmarks_norm=norm_lms,
            landmarks_pixel=pix_lms,
            fingertips_pixel={
                "Thumb": pix_lms[4], "Index": pix_lms[8], "Middle": pix_lms[12],
                "Ring": pix_lms[16], "Pinky": pix_lms[20],
            },
            palm_center=pix_lms[0],
            bounding_box=(0, 0, 100, 100),
        )

    def test_pinch_left_click(self):
        # Bring Thumb Tip (4) and Index Tip (8) to exact same coordinates
        lms = [(0.5, 0.5 + 0.01 * i, 0.0) for i in range(21)]
        lms[0] = (0.5, 0.8, 0.0)   # Wrist
        lms[9] = (0.5, 0.5, 0.0)   # Middle MCP (sets palm scale)
        lms[4] = (0.50, 0.40, 0.0) # Thumb Tip
        lms[8] = (0.51, 0.40, 0.0) # Index Tip touching thumb tip

        hand = self._create_hand(lms)
        event = self.controller.process(hand)
        self.assertEqual(event.state, "CLICK_LEFT")
        self.assertLess(event.pinch_ratio, 0.08)

    def test_index_middle_touch_left_click(self):
        # Bring Index Tip (8) and Middle Tip (12) together to touch
        lms = [(0.5, 0.5 + 0.01 * i, 0.0) for i in range(21)]
        lms[0] = (0.5, 0.8, 0.0)   # Wrist
        lms[9] = (0.5, 0.5, 0.0)   # Middle MCP (palm_scale = 0.3)
        lms[4] = (0.40, 0.50, 0.0) # Thumb out
        lms[8] = (0.50, 0.30, 0.0) # Index Tip
        lms[12] = (0.51, 0.30, 0.0) # Middle Tip touching Index Tip (dist = 0.01 / 0.3 = 0.033 < 0.10)
        lms[7] = (0.50, 0.38, 0.0)
        lms[11] = (0.51, 0.38, 0.0)

        hand = self._create_hand(lms)
        event = self.controller.process(hand)
        self.assertEqual(event.state, "CLICK_LEFT")
        self.assertIn("Index-Middle Touch", event.details)

    def test_middle_pinch_right_click(self):
        lms = [(0.5, 0.5 + 0.01 * i, 0.0) for i in range(21)]
        lms[0] = (0.5, 0.8, 0.0)
        lms[9] = (0.5, 0.5, 0.0)
        lms[4] = (0.50, 0.40, 0.0)  # Thumb Tip
        lms[8] = (0.60, 0.30, 0.0)  # Index Tip open
        lms[12] = (0.51, 0.40, 0.0) # Middle Tip touching thumb tip

        hand = self._create_hand(lms)
        event = self.controller.process(hand)
        self.assertEqual(event.state, "CLICK_RIGHT")

    def test_fist_clutch(self):
        # All fingertip distances closer to wrist than PIP joints (Fist pose)
        lms = [(0.5, 0.5, 0.0) for _ in range(21)]
        wrist = (0.5, 0.8, 0.0)
        lms[0] = wrist
        # PIPs extended outward
        lms[6] = (0.5, 0.6, 0.0)
        lms[10] = (0.5, 0.6, 0.0)
        lms[14] = (0.5, 0.6, 0.0)
        lms[18] = (0.5, 0.6, 0.0)
        # Tips curled back in towards wrist
        lms[4] = (0.5, 0.75, 0.0)
        lms[8] = (0.5, 0.72, 0.0)
        lms[12] = (0.5, 0.72, 0.0)
        lms[16] = (0.5, 0.72, 0.0)
        lms[20] = (0.5, 0.72, 0.0)

        hand = self._create_hand(lms)
        event = self.controller.process(hand)
        self.assertEqual(event.state, "CLUTCH")
        self.assertTrue(event.is_clutched)


class TestAirMouse(unittest.TestCase):
    """Test end-to-end Air Mouse relative tracking pipeline and state management."""

    def test_relative_tracking_flow(self):
        mouse = AirMouse(enabled=False)  # Test with automation disabled to avoid cursor interference
        self.assertFalse(mouse.enabled)

        # Create synthetic hand
        lms = [(0.5, 0.5 + 0.01 * i, 0.0) for i in range(21)]
        hand = HandData(
            index=0,
            handedness="Right",
            confidence=0.99,
            landmarks_norm=lms,
            landmarks_pixel=[(640, 360) for _ in range(21)],
            fingertips_pixel={},
            palm_center=(640, 360),
            bounding_box=(600, 300, 700, 400),
        )
        res = HandTrackingResult(hands=[hand], num_hands=1)

        # Tick 1: Acquired (Seeds initial reference position, zero displacement)
        tel1 = mouse.process_frame(res)
        self.assertEqual(tel1.state, "ACQUIRED")
        self.assertEqual(tel1.raw_dx, 0.0)
        self.assertEqual(tel1.raw_dy, 0.0)

        # Tick 2: Hand moves right (Index tip 8 moves from 0.50 to 0.53)
        hand.landmarks_norm[8] = (0.53, hand.landmarks_norm[8][1], 0.0)
        tel2 = mouse.process_frame(res)
        self.assertAlmostEqual(tel2.raw_dx, 0.03, places=3)
        self.assertGreater(tel2.smooth_vx, 0.0)

        # Tick 3: Tracking dropped (Hand leaves frame)
        res_empty = HandTrackingResult(hands=[], num_hands=0)
        tel3 = mouse.process_frame(res_empty)
        self.assertEqual(tel3.state, "NO_HAND")
        self.assertIsNone(mouse._prev_index_norm)

        # Tick 4: Hand re-appears in different position (Must not jump!)
        hand.landmarks_norm[8] = (0.20, 0.20, 0.0)
        tel4 = mouse.process_frame(res)
        self.assertEqual(tel4.state, "ACQUIRED")
        self.assertEqual(tel4.raw_dx, 0.0)
        self.assertEqual(tel4.smooth_vx, 0.0)


class TestCameraManager(unittest.TestCase):
    """Test CameraManager lifecycle and queue behavior."""

    def test_start_stop(self):
        cam = CameraManager()
        self.assertEqual(cam.status, "OFFLINE")

        started = cam.start()
        self.assertTrue(started)
        time.sleep(0.5)

        cam.stop(timeout=2.0)
        self.assertEqual(cam.status, "OFFLINE")


class TestGestureClassifier(unittest.TestCase):
    """Test Stage 3 static hand pose classification with synthetic landmark configurations."""

    @classmethod
    def setUpClass(cls):
        cls.clf = GestureClassifier()

    def _build_hand(
        self,
        wrist=(0.5, 0.7, 0.0),
        thumb_straight=True,
        index_straight=True,
        middle_straight=True,
        ring_straight=True,
        pinky_straight=True,
        thumb_pos=None,
        touch_thumb_index=False,
    ):
        lms = [(0.5, 0.5, 0.0) for _ in range(21)]
        lms[0] = wrist
        lms[9] = (wrist[0], wrist[1] - 0.20, wrist[2])

        def set_finger(mcp_idx, base_x, is_straight):
            mcp_y = wrist[1] - 0.20
            lms[mcp_idx] = (base_x, mcp_y, 0.0)
            if is_straight:
                lms[mcp_idx + 1] = (base_x, mcp_y - 0.08, 0.0)
                lms[mcp_idx + 2] = (base_x, mcp_y - 0.16, 0.0)
                lms[mcp_idx + 3] = (base_x, mcp_y - 0.24, 0.0)
            else:
                lms[mcp_idx + 1] = (base_x, mcp_y - 0.05, 0.0)
                lms[mcp_idx + 2] = (base_x, mcp_y - 0.02, 0.0)
                lms[mcp_idx + 3] = (base_x, mcp_y + 0.02, 0.0)

        set_finger(5, wrist[0] - 0.06, index_straight)
        set_finger(9, wrist[0], middle_straight)
        set_finger(13, wrist[0] + 0.06, ring_straight)
        set_finger(17, wrist[0] + 0.12, pinky_straight)

        if thumb_pos == "UP":
            lms[1] = (wrist[0] - 0.04, wrist[1] - 0.05, 0.0)
            lms[2] = (wrist[0] - 0.08, wrist[1] - 0.10, 0.0)
            lms[3] = (wrist[0] - 0.08, wrist[1] - 0.20, 0.0)
            lms[4] = (wrist[0] - 0.08, wrist[1] - 0.30, 0.0)
        elif thumb_pos == "DOWN":
            lms[1] = (wrist[0] - 0.04, wrist[1] + 0.02, 0.0)
            lms[2] = (wrist[0] - 0.08, wrist[1] + 0.06, 0.0)
            lms[3] = (wrist[0] - 0.08, wrist[1] + 0.14, 0.0)
            lms[4] = (wrist[0] - 0.08, wrist[1] + 0.22, 0.0)
        elif touch_thumb_index:
            # Thumb tip touches index tip in circle
            lms[1] = (wrist[0] - 0.04, wrist[1] - 0.05, 0.0)
            lms[2] = (wrist[0] - 0.07, wrist[1] - 0.12, 0.0)
            lms[3] = (wrist[0] - 0.06, wrist[1] - 0.20, 0.0)
            lms[4] = (wrist[0] - 0.06, wrist[1] - 0.24, 0.0)  # Touches index tip
            lms[8] = (wrist[0] - 0.06, wrist[1] - 0.24, 0.0)
        elif thumb_straight:
            lms[1] = (wrist[0] - 0.04, wrist[1] - 0.05, 0.0)
            lms[2] = (wrist[0] - 0.08, wrist[1] - 0.10, 0.0)
            lms[3] = (wrist[0] - 0.14, wrist[1] - 0.15, 0.0)
            lms[4] = (wrist[0] - 0.20, wrist[1] - 0.20, 0.0)
        else:
            lms[1] = (wrist[0] - 0.04, wrist[1] - 0.05, 0.0)
            lms[2] = (wrist[0] - 0.06, wrist[1] - 0.10, 0.0)
            lms[3] = (wrist[0] - 0.02, wrist[1] - 0.12, 0.0)
            lms[4] = (wrist[0], wrist[1] - 0.12, 0.0)

        return HandData(
            index=0,
            handedness="Right",
            confidence=0.98,
            landmarks_norm=lms,
            landmarks_pixel=[(int(x * 1280), int(y * 720)) for x, y, _ in lms],
            fingertips_pixel={},
            palm_center=(int(wrist[0] * 1280), int(wrist[1] * 720)),
            bounding_box=(100, 100, 300, 300),
            finger_states={},
        )

    def test_open_palm(self):
        hand = self._build_hand(
            thumb_straight=True, index_straight=True, middle_straight=True,
            ring_straight=True, pinky_straight=True
        )
        res = self.clf.classify(hand)
        self.assertEqual(res.name, StaticGesture.OPEN_PALM.value)
        self.assertGreater(res.confidence, 0.85)

    def test_fist(self):
        hand = self._build_hand(
            thumb_straight=False, index_straight=False, middle_straight=False,
            ring_straight=False, pinky_straight=False
        )
        res = self.clf.classify(hand)
        self.assertEqual(res.name, StaticGesture.FIST.value)
        self.assertGreater(res.confidence, 0.85)

    def test_pointing(self):
        hand = self._build_hand(
            thumb_straight=False, index_straight=True, middle_straight=False,
            ring_straight=False, pinky_straight=False
        )
        res = self.clf.classify(hand)
        self.assertEqual(res.name, StaticGesture.POINTING.value)

    def test_victory(self):
        hand = self._build_hand(
            thumb_straight=False, index_straight=True, middle_straight=True,
            ring_straight=False, pinky_straight=False
        )
        res = self.clf.classify(hand)
        self.assertEqual(res.name, StaticGesture.VICTORY.value)

    def test_thumbs_up(self):
        hand = self._build_hand(
            thumb_pos="UP", index_straight=False, middle_straight=False,
            ring_straight=False, pinky_straight=False
        )
        res = self.clf.classify(hand)
        self.assertEqual(res.name, StaticGesture.THUMBS_UP.value)

    def test_thumbs_down(self):
        hand = self._build_hand(
            thumb_pos="DOWN", index_straight=False, middle_straight=False,
            ring_straight=False, pinky_straight=False
        )
        res = self.clf.classify(hand)
        self.assertEqual(res.name, StaticGesture.THUMBS_DOWN.value)

    def test_rock_on(self):
        hand = self._build_hand(
            thumb_straight=False, index_straight=True, middle_straight=False,
            ring_straight=False, pinky_straight=True
        )
        res = self.clf.classify(hand)
        self.assertEqual(res.name, StaticGesture.ROCK_ON.value)

    def test_call_me(self):
        hand = self._build_hand(
            thumb_straight=True, index_straight=False, middle_straight=False,
            ring_straight=False, pinky_straight=True
        )
        res = self.clf.classify(hand)
        self.assertEqual(res.name, StaticGesture.CALL_ME.value)

    def test_insufficient_landmarks(self):
        hand = HandData(
            index=0,
            handedness="Right",
            confidence=0.5,
            landmarks_norm=[(0.0, 0.0, 0.0)] * 5,
            landmarks_pixel=[(0, 0)] * 5,
            fingertips_pixel={},
            palm_center=(0, 0),
            bounding_box=(0, 0, 0, 0),
        )
        res = self.clf.classify(hand)
        self.assertEqual(res.name, StaticGesture.UNKNOWN.value)


class TestGestureEngine(unittest.TestCase):
    """Test Stage 3 GestureEngine temporal stability, dynamic motions, and dual interactions."""

    def setUp(self):
        self.engine = GestureEngine(stability_frames=3, cooldown_sec=0.25)
        self.clf = GestureClassifier()

    def _make_hand_at(self, x: float, y: float, handedness: str = "Right") -> HandData:
        lms = [(x, y, 0.0) for _ in range(21)]
        lms[0] = (x, y, 0.0)
        lms[9] = (x, y - 0.20, 0.0)
        for idx, bx in [(5, x - 0.06), (9, x), (13, x + 0.06), (17, x + 0.12)]:
            lms[idx] = (bx, y - 0.20, 0.0)
            lms[idx + 1] = (bx, y - 0.28, 0.0)
            lms[idx + 2] = (bx, y - 0.36, 0.0)
            lms[idx + 3] = (bx, y - 0.44, 0.0)
        lms[1] = (x - 0.04, y - 0.05, 0.0)
        lms[2] = (x - 0.08, y - 0.10, 0.0)
        lms[3] = (x - 0.14, y - 0.15, 0.0)
        lms[4] = (x - 0.20, y - 0.20, 0.0)
        return HandData(
            index=0,
            handedness=handedness,
            confidence=0.99,
            landmarks_norm=lms,
            landmarks_pixel=[(int(px * 1280), int(py * 720)) for px, py, _ in lms],
            fingertips_pixel={},
            palm_center=(int(x * 1280), int(y * 720)),
            bounding_box=(100, 100, 300, 300),
            finger_states={},
        )

    def test_stability_and_holding(self):
        # Frame 1 & 2: UNKNOWN due to stability filter
        h = self._make_hand_at(0.5, 0.7)
        tel1 = self.engine.process(HandTrackingResult(hands=[h], num_hands=1))
        self.assertEqual(tel1.hands["Right"].current_pose, StaticGesture.UNKNOWN.value)

        tel2 = self.engine.process(HandTrackingResult(hands=[h], num_hands=1))
        self.assertEqual(tel2.hands["Right"].current_pose, StaticGesture.UNKNOWN.value)

        # Frame 3: Stability threshold reached -> locks into OPEN_PALM
        tel3 = self.engine.process(HandTrackingResult(hands=[h], num_hands=1))
        self.assertEqual(tel3.hands["Right"].current_pose, StaticGesture.OPEN_PALM.value)

        # Hold for > 0.3s
        time.sleep(0.30)
        tel4 = self.engine.process(HandTrackingResult(hands=[h], num_hands=1))
        self.assertTrue(tel4.hands["Right"].is_holding)
        self.assertGreaterEqual(tel4.hands["Right"].hold_duration_sec, 0.25)

    def test_dynamic_swipe_left(self):
        self.engine.reset()
        for x in [0.75, 0.68, 0.58, 0.46, 0.36, 0.30]:
            h = self._make_hand_at(x, 0.5)
            time.sleep(0.025)
            tel = self.engine.process(HandTrackingResult(hands=[h], num_hands=1))

        self.assertEqual(tel.active_dynamic_gesture, DynamicGesture.SWIPE_LEFT.value)

    def test_dynamic_swipe_right(self):
        self.engine.reset()
        for x in [0.25, 0.32, 0.42, 0.54, 0.66, 0.72]:
            h = self._make_hand_at(x, 0.5)
            time.sleep(0.025)
            tel = self.engine.process(HandTrackingResult(hands=[h], num_hands=1))

        self.assertEqual(tel.active_dynamic_gesture, DynamicGesture.SWIPE_RIGHT.value)

    def test_dynamic_swipe_up(self):
        self.engine.reset()
        # In image coords, y decreases as hand moves upward
        for y in [0.80, 0.72, 0.62, 0.50, 0.38, 0.32]:
            h = self._make_hand_at(0.5, y)
            time.sleep(0.025)
            tel = self.engine.process(HandTrackingResult(hands=[h], num_hands=1))

        self.assertEqual(tel.active_dynamic_gesture, DynamicGesture.SWIPE_UP.value)

    def test_dual_open_palms(self):
        self.engine.reset()
        # Feed 3 frames of two open palm hands
        for _ in range(3):
            h_r = self._make_hand_at(0.7, 0.5, "Right")
            h_l = self._make_hand_at(0.3, 0.5, "Left")
            tel = self.engine.process(HandTrackingResult(hands=[h_r, h_l], num_hands=2))

        self.assertEqual(tel.active_dual_gesture, DualHandGesture.DUAL_OPEN_PALMS.value)

    def test_dual_two_hands_x(self):
        self.engine.reset()
        # Create two hands where wrists are crossed in 'X' shape
        # Left wrist at (0.46, 0.65), pointing to (0.58, 0.40)
        # Right wrist at (0.54, 0.65), pointing to (0.42, 0.40)
        h_l = self._make_hand_at(0.46, 0.65, "Left")
        h_l.landmarks_norm[0] = (0.46, 0.65, 0.0)
        h_l.landmarks_norm[12] = (0.58, 0.40, 0.0)

        h_r = self._make_hand_at(0.54, 0.65, "Right")
        h_r.landmarks_norm[0] = (0.54, 0.65, 0.0)
        h_r.landmarks_norm[12] = (0.42, 0.40, 0.0)

        tel = self.engine.process(HandTrackingResult(hands=[h_l, h_r], num_hands=2))
        self.assertEqual(tel.active_dual_gesture, DualHandGesture.TWO_HANDS_X.value)


class TestWindowManager(unittest.TestCase):
    """Test Stage 4 WindowManager actions and hotkeys."""

    def setUp(self):
        self.wm = WindowManager()

    @patch("desktop.window_manager.pyautogui.hotkey")
    @patch("desktop.window_manager.HAS_WIN32", False)
    def test_minimize_active_window(self, mock_hotkey):
        res = self.wm.minimize_active_window()
        self.assertTrue(res)
        mock_hotkey.assert_called_with("win", "down")

    @patch("desktop.window_manager.pyautogui.hotkey")
    @patch("desktop.window_manager.HAS_WIN32", False)
    def test_close_active_window(self, mock_hotkey):
        res = self.wm.close_active_window()
        self.assertTrue(res)
        mock_hotkey.assert_called_with("alt", "f4")

    @patch("desktop.window_manager.pyautogui.hotkey")
    @patch("desktop.window_manager.HAS_WIN32", False)
    def test_toggle_maximize_restore(self, mock_hotkey):
        res = self.wm.toggle_maximize_restore()
        self.assertTrue(res)
        mock_hotkey.assert_called_with("win", "up")

    @patch("desktop.window_manager.pyautogui.hotkey")
    def test_switch_windows(self, mock_hotkey):
        res = self.wm.switch_windows()
        self.assertTrue(res)
        mock_hotkey.assert_called_with("alt", "tab")

    @patch("desktop.window_manager.pyautogui.hotkey")
    def test_show_desktop(self, mock_hotkey):
        res = self.wm.show_desktop()
        self.assertTrue(res)
        mock_hotkey.assert_called_with("win", "d")

    @patch("desktop.window_manager.pyautogui.hotkey")
    def test_switch_virtual_desktop(self, mock_hotkey):
        res_prev = self.wm.switch_virtual_desktop("previous")
        self.assertTrue(res_prev)
        mock_hotkey.assert_called_with("ctrl", "win", "left")

        res_next = self.wm.switch_virtual_desktop("next")
        self.assertTrue(res_next)
        mock_hotkey.assert_called_with("ctrl", "win", "right")


class TestSystemControls(unittest.TestCase):
    """Test Stage 4 SystemControls volume, screenshot, and media manipulation."""

    def setUp(self):
        self.sys_ctrl = SystemControls(audio_enabled=False)

    @patch("desktop.system_controls.pyautogui.press")
    def test_volume_controls(self, mock_press):
        self.assertTrue(self.sys_ctrl.volume_up(steps=2))
        self.assertEqual(mock_press.call_count, 2)
        mock_press.assert_called_with("volumeup")

        mock_press.reset_mock()
        self.assertTrue(self.sys_ctrl.volume_down(steps=1))
        mock_press.assert_called_with("volumedown")

    @patch("desktop.system_controls.pyautogui.hotkey")
    def test_screenshot(self, mock_hotkey):
        self.assertTrue(self.sys_ctrl.capture_screenshot())
        mock_hotkey.assert_called_with("win", "shift", "s")

    @patch("desktop.system_controls.pyautogui.press")
    def test_media_playback(self, mock_press):
        self.assertTrue(self.sys_ctrl.media_play_pause())
        mock_press.assert_called_with("playpause")


class TestCommandDispatcher(unittest.TestCase):
    """Test Stage 4 CommandDispatcher mappings, confidence checks, and cooldowns."""

    def setUp(self):
        self.mock_wm = MagicMock()
        self.mock_sys = MagicMock()
        self.dispatcher = CommandDispatcher(
            window_manager=self.mock_wm,
            system_controls=self.mock_sys,
        )

    def test_mapped_commands(self):
        # 1. THUMBS_DOWN -> MINIMIZE_ACTIVE_WINDOW
        res1 = self.dispatcher.dispatch("THUMBS_DOWN", confidence=0.95)
        self.assertTrue(res1.success)
        self.assertEqual(res1.command, DesktopCommand.MINIMIZE_ACTIVE_WINDOW)
        self.mock_wm.minimize_active_window.assert_called_once()

        self.dispatcher.reset_cooldowns()

        # 2. TWO_HANDS_X -> CLOSE_ACTIVE_WINDOW
        res2 = self.dispatcher.dispatch("TWO_HANDS_X", confidence=0.95)
        self.assertTrue(res2.success)
        self.assertEqual(res2.command, DesktopCommand.CLOSE_ACTIVE_WINDOW)
        self.mock_wm.close_active_window.assert_called_once()

        self.dispatcher.reset_cooldowns()

        # 3. THUMBS_UP -> TOGGLE_MAXIMIZE_RESTORE
        res3 = self.dispatcher.dispatch("THUMBS_UP", confidence=0.95)
        self.assertTrue(res3.success)
        self.assertEqual(res3.command, DesktopCommand.TOGGLE_MAXIMIZE_RESTORE)
        self.mock_wm.toggle_maximize_restore.assert_called_once()

        self.dispatcher.reset_cooldowns()

        # 4. OK_SIGN / PINCH_OK -> SWITCH_WINDOWS
        res4 = self.dispatcher.dispatch("OK_SIGN", confidence=0.95)
        self.assertTrue(res4.success)
        self.assertEqual(res4.command, DesktopCommand.SWITCH_WINDOWS)
        self.mock_wm.switch_windows.assert_called_once()

        self.dispatcher.reset_cooldowns()

        # 5. OPEN_PALM_HELD -> SHOW_DESKTOP
        res5 = self.dispatcher.dispatch("OPEN_PALM", confidence=0.95, hold_duration=0.40)
        self.assertTrue(res5.success)
        self.assertEqual(res5.command, DesktopCommand.SHOW_DESKTOP)
        self.mock_wm.show_desktop.assert_called_once()

        self.dispatcher.reset_cooldowns()

        # 6. SWIPE_LEFT -> PREVIOUS_DESKTOP
        res6 = self.dispatcher.dispatch("SWIPE_LEFT", confidence=0.92)
        self.assertTrue(res6.success)
        self.assertEqual(res6.command, DesktopCommand.PREVIOUS_DESKTOP)
        self.mock_wm.switch_virtual_desktop.assert_called_with("previous")

    def test_confidence_rejection(self):
        # Confidence < 0.70 is rejected
        res = self.dispatcher.dispatch("THUMBS_UP", confidence=0.55)
        self.assertFalse(res.success)
        self.assertIn("Confidence", res.reason)

    def test_held_required_rejection(self):
        # OPEN_PALM with insufficient hold duration (< 0.35s) rejected
        res = self.dispatcher.dispatch("OPEN_PALM", confidence=0.95, hold_duration=0.15)
        self.assertFalse(res.success)
        self.assertIn("Hold duration", res.reason)

    def test_cooldown_suppression(self):
        # First call succeeds
        res1 = self.dispatcher.dispatch("THUMBS_DOWN", confidence=0.95)
        self.assertTrue(res1.success)

        # Immediate repeat within cooldown window rejected
        res2 = self.dispatcher.dispatch("THUMBS_DOWN", confidence=0.95)
        self.assertFalse(res2.success)
        self.assertIn("cooldown active", res2.reason)


class TestDesktopController(unittest.TestCase):
    """Test Stage 4 DesktopController unified coordinator and safety halts."""

    def setUp(self):
        self.controller = DesktopController()
        self.controller.dispatcher.window_manager = MagicMock()
        self.controller.dispatcher.system_controls = MagicMock()

    def tearDown(self):
        self.controller.stop(timeout=0.5)

    def test_handle_gesture_event(self):
        evt = EngineGestureEvent(
            hand_label="Right",
            gesture_type="STATIC",
            gesture_name="THUMBS_DOWN",
            confidence=0.95,
            hold_duration=0.0,
            details="Thumbs down confirmed",
        )
        res = self.controller.handle_gesture_event(evt)
        self.assertIsNotNone(res)
        self.assertTrue(res.success)
        self.assertEqual(self.controller.total_executed, 1)

    def test_emergency_halt(self):
        self.controller.emergency_halt()
        self.assertFalse(self.controller.enabled)

        evt = EngineGestureEvent(
            hand_label="Right",
            gesture_type="STATIC",
            gesture_name="THUMBS_DOWN",
            confidence=0.95,
            hold_duration=0.0,
            details="Thumbs down",
        )
        res = self.controller.handle_gesture_event(evt)
        self.assertIsNone(res)
        self.assertEqual(self.controller.total_executed, 0)

    def test_async_worker_execution(self):
        self.controller.start()
        self.assertTrue(self.controller.is_worker_running)

        evt = EngineGestureEvent(
            hand_label="Right",
            gesture_type="STATIC",
            gesture_name="THUMBS_UP",
            confidence=0.95,
            hold_duration=0.0,
            details="Thumbs up",
        )
        # In async mode, handle_gesture_event enqueues and returns None immediately
        res = self.controller.handle_gesture_event(evt)
        self.assertIsNone(res)

        # Wait for worker thread to process from queue
        time.sleep(0.15)
        results = self.controller.get_pending_results()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].command, DesktopCommand.TOGGLE_MAXIMIZE_RESTORE)
        self.assertEqual(self.controller.total_executed, 1)

    def test_emergency_halt_cancels_queue(self):
        self.controller.start()
        evt = EngineGestureEvent(
            hand_label="Right",
            gesture_type="STATIC",
            gesture_name="THUMBS_DOWN",
            confidence=0.95,
            hold_duration=0.0,
            details="Thumbs down",
        )
        self.controller.handle_gesture_event(evt)
        self.controller.emergency_halt()
        self.assertFalse(self.controller.enabled)
        # Queue should be empty after emergency halt
        self.assertTrue(self.controller._command_queue.empty())


class TestThreadArchitecture(unittest.TestCase):
    """Test multi-worker thread decoupling (Camera, Vision, Automation)."""

    def test_camera_and_vision_workers_concurrent(self):
        cam = CameraManager()
        self.assertFalse(cam.is_running)

        cam.start()
        self.assertTrue(cam.is_running)
        self.assertIsNotNone(cam._camera_thread)
        self.assertIsNotNone(cam._vision_thread)
        self.assertTrue(cam._camera_thread.is_alive())
        self.assertTrue(cam._vision_thread.is_alive())
        self.assertEqual(cam._camera_thread.name, "AirControl-CameraWorker")
        self.assertEqual(cam._vision_thread.name, "AirControl-VisionWorker")

        cam.stop(timeout=1.0)
        self.assertFalse(cam.is_running)

    def test_repeated_camera_start_stop(self):
        cam = CameraManager()
        for _ in range(3):
            cam.start()
            self.assertTrue(cam.is_running)
            time.sleep(0.05)
            cam.stop(timeout=1.0)
            self.assertFalse(cam.is_running)


class TestHand3D(unittest.TestCase):
    """Unit tests for Stage 4.1 3D Hand Tracking, Articulated Kinematics and Spatial Engine."""

    def _make_open_palm_joints(self) -> list:
        """Create a synthetic 21-joint 3D open palm in metric coordinates."""
        pts = [(0.0, 0.0, 0.0)] * 21
        # Wrist at origin
        pts[0] = (0.0, 0.0, 0.0)
        # Thumb
        pts[1] = (-0.02, 0.02, 0.0)
        pts[2] = (-0.035, 0.04, 0.0)
        pts[3] = (-0.045, 0.06, 0.0)
        pts[4] = (-0.055, 0.08, 0.0)
        # Index
        pts[5] = (-0.025, 0.08, 0.0)
        pts[6] = (-0.027, 0.11, 0.0)
        pts[7] = (-0.028, 0.13, 0.0)
        pts[8] = (-0.029, 0.15, 0.0)
        # Middle
        pts[9] = (0.0, 0.085, 0.0)
        pts[10] = (0.0, 0.12, 0.0)
        pts[11] = (0.0, 0.145, 0.0)
        pts[12] = (0.0, 0.17, 0.0)
        # Ring
        pts[13] = (0.025, 0.08, 0.0)
        pts[14] = (0.027, 0.11, 0.0)
        pts[15] = (0.028, 0.13, 0.0)
        pts[16] = (0.029, 0.15, 0.0)
        # Pinky
        pts[17] = (0.045, 0.07, 0.0)
        pts[18] = (0.048, 0.095, 0.0)
        pts[19] = (0.050, 0.115, 0.0)
        pts[20] = (0.052, 0.13, 0.0)
        return pts

    def test_skeleton_bone_calculation(self):
        skeleton = HandSkeleton()
        joints = self._make_open_palm_joints()
        bones = skeleton.calculate_bones(joints)
        self.assertEqual(len(bones), 23)
        for bone in bones:
            self.assertGreater(bone.length, 0.0)
            dir_mag = (bone.direction[0]**2 + bone.direction[1]**2 + bone.direction[2]**2)**0.5
            self.assertAlmostEqual(dir_mag, 1.0, places=3)

    def test_joint_angles_straight_fingers(self):
        joints = self._make_open_palm_joints()
        angles = JointAngleCalculator.calculate_hand_angles(joints)
        self.assertEqual(len(angles.fingers), 5)
        # All fingers in open palm should be extended
        for fname in ("Index", "Middle", "Ring", "Pinky", "Thumb"):
            f = angles.fingers[fname]
            self.assertTrue(f.is_extended, f"{fname} should be extended")
            self.assertLess(f.total_flexion_deg, 45.0)

    def test_joint_angles_curled_finger(self):
        joints = self._make_open_palm_joints()
        # Bend index finger PIP & DIP by 90 degrees
        joints[6] = (-0.027, 0.11, 0.0)
        joints[7] = (-0.027, 0.11, 0.02)
        joints[8] = (-0.027, 0.09, 0.02)
        angles = JointAngleCalculator.calculate_hand_angles(joints)
        index_f = angles.fingers["Index"]
        self.assertGreater(index_f.total_flexion_deg, 60.0)

    def test_hand_orientation_normal_and_facing(self):
        joints = self._make_open_palm_joints()
        estimator = OrientationEstimator(is_right_hand=True)
        ori = estimator.estimate(joints)
        self.assertIsNotNone(ori.palm_normal)
        # Normal should be a unit vector
        norm_mag = sum(x**2 for x in ori.palm_normal)**0.5
        self.assertAlmostEqual(norm_mag, 1.0, places=3)
        self.assertIn(ori.facing, [PalmFacing.FACING_CAMERA, PalmFacing.FACING_AWAY])

    def test_spatial_features_canonical_frame(self):
        joints = self._make_open_palm_joints()
        extractor = SpatialFeatureExtractor()
        normal = (0.0, 0.0, 1.0)
        feat = extractor.extract(joints, normal)
        self.assertGreater(feat.palm_scale, 0.05)
        self.assertEqual(len(feat.canonical_joints), 21)
        # Wrist at origin
        self.assertAlmostEqual(feat.canonical_joints[0][0], 0.0, places=3)
        self.assertAlmostEqual(feat.canonical_joints[0][1], 0.0, places=3)
        # Fingertip distances
        self.assertGreater(feat.thumb_index_dist, 0.0)
        self.assertGreater(feat.index_middle_dist, 0.0)

    def test_hand_filter_3d_and_spike_rejection(self):
        hand_filter = HandFilter3D(max_jump_m=0.10)
        joints = self._make_open_palm_joints()
        filtered1 = hand_filter.filter_joints(joints, timestamp=1.0)
        self.assertEqual(len(filtered1), 21)

        # Introduce an anomalous 1-meter spike on index tip
        spiked_joints = list(joints)
        spiked_joints[8] = (1.0, 1.0, 1.0)
        filtered2 = hand_filter.filter_joints(spiked_joints, timestamp=1.033)
        # Index tip displacement should be clamped by max_jump
        dx = filtered2[8][0] - filtered1[8][0]
        dy = filtered2[8][1] - filtered1[8][1]
        dz = filtered2[8][2] - filtered1[8][2]
        dist = (dx*dx + dy*dy + dz*dz)**0.5
        self.assertLessEqual(dist, 0.11)

    def test_hand_calibrator(self):
        calibrator = HandCalibrator(target_samples=5)
        calibrator.start_calibration()
        self.assertTrue(calibrator.is_collecting)
        joints = self._make_open_palm_joints()
        for i in range(4):
            done = calibrator.add_sample(joints)
            self.assertFalse(done)
        # 5th sample completes calibration
        done = calibrator.add_sample(joints)
        self.assertTrue(done)
        self.assertTrue(calibrator.profile.is_calibrated)
        self.assertGreater(calibrator.profile.palm_scale, 0.05)
        # Ratio check
        ratio = calibrator.get_scale_ratio(calibrator.profile.palm_scale)
        self.assertAlmostEqual(ratio, 1.0, places=2)
        # Reset check
        calibrator.reset()
        self.assertFalse(calibrator.profile.is_calibrated)

    def test_hand_3d_processor_pipeline(self):
        processor = Hand3DProcessor(handedness="Right")
        joints = self._make_open_palm_joints()
        model = processor.update(joints, confidence=0.95)
        self.assertTrue(model.is_valid)
        self.assertEqual(model.handedness, "Right")
        self.assertTrue(model.is_finger_straight("Index"))
        self.assertFalse(model.is_finger_curled("Index"))
        self.assertGreater(model.get_pinch_distance("Index"), 0.0)
        self.assertEqual(len(model.bones), 23)

    def test_skeleton_renderer_3d(self):
        renderer = SkeletonRenderer3D(width=320, height=240)
        # Standby rendering
        img_standby = renderer.render(None)
        self.assertEqual(img_standby.shape, (240, 320, 3))
        # Active model rendering
        processor = Hand3DProcessor(handedness="Right")
        model = processor.update(self._make_open_palm_joints())
        img_model = renderer.render(model)
        self.assertEqual(img_model.shape, (240, 320, 3))
        # Camera orbit controls
        renderer.set_camera_orbit(45.0, -30.0, zoom=1.5)
        self.assertEqual(renderer.azimuth_deg, 45.0)
        self.assertEqual(renderer.elevation_deg, -30.0)
        self.assertEqual(renderer.zoom, 1.5)
        renderer.reset_view()
        self.assertEqual(renderer.azimuth_deg, 25.0)

    def test_spatial_3d_air_mouse_mode(self):
        mouse = AirMouse(enabled=True)
        mouse.spatial_3d_enabled = True
        self.assertTrue(mouse.spatial_3d_enabled)

        processor = Hand3DProcessor(handedness="Right")
        model1 = processor.update(self._make_open_palm_joints(), timestamp=1.0)
        hand1 = HandData(
            index=0,
            handedness="Right",
            confidence=0.95,
            landmarks_norm=[(0.5, 0.5, 0.0)]*21,
            landmarks_pixel=[(640, 360)]*21,
            fingertips_pixel={"Index": (640, 360)},
            palm_center=(640, 360),
            bounding_box=(600, 320, 680, 400),
            hand_model_3d=model1,
        )
        tracking_res1 = HandTrackingResult(hands=[hand1], num_hands=1)
        tel1 = mouse.process_frame(tracking_res1)
        self.assertTrue(tel1.is_spatial_3d)

        # Move hand physically by 2 cm in X
        joints2 = self._make_open_palm_joints()
        joints2 = [(p[0] + 0.02, p[1], p[2]) for p in joints2]
        model2 = processor.update(joints2, timestamp=1.033)
        hand2 = HandData(
            index=0,
            handedness="Right",
            confidence=0.95,
            landmarks_norm=[(0.52, 0.5, 0.0)]*21,
            landmarks_pixel=[(665, 360)]*21,
            fingertips_pixel={"Index": (665, 360)},
            palm_center=(665, 360),
            bounding_box=(625, 320, 705, 400),
            hand_model_3d=model2,
        )
        tracking_res2 = HandTrackingResult(hands=[hand2], num_hands=1)
        tel2 = mouse.process_frame(tracking_res2)
        self.assertTrue(tel2.is_spatial_3d)
        self.assertGreater(abs(tel2.raw_dx), 0.0)


if __name__ == "__main__":
    unittest.main()
