"""
AIRCONTROL — 3D SKELETAL WIREFRAME AND JOINT RENDERER
Renders an interactive real-time 3D perspective projection of the articulated
hand skeleton, joints, coordinate grid, and palm normal vector into an RGB image.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from hand3d.hand_model import HandModel3D
from hand3d.skeleton import (
    FINGER_JOINTS,
    JOINT_NAMES,
    JOINT_WRIST,
)

# Sci-Fi Cyberpunk Color Palette (BGR)
COLOR_GRID = (35, 40, 48)
COLOR_GRID_AXIS = (60, 75, 95)
COLOR_TEXT = (210, 225, 240)
COLOR_TEXT_DIM = (120, 135, 150)
COLOR_NORMAL_VEC = (0, 230, 255)    # Glowing Yellow/Cyan
COLOR_BONE_PALM = (180, 140, 50)    # Blue-Gray/Teal

FINGER_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Palm": (180, 140, 50),
    "Thumb": (255, 160, 50),     # Electric Blue/Cyan
    "Index": (50, 240, 255),     # Yellow-Gold
    "Middle": (50, 255, 160),    # Bright Emerald
    "Ring": (240, 120, 255),     # Neon Purple/Magenta
    "Pinky": (100, 180, 255),    # Coral/Orange
}


class SkeletonRenderer3D:
    """
    Renders 3D hand models with depth shading, 3D orbit camera controls,
    bone segments, joint spheres, and normal vectors onto an OpenCV/PIL canvas.
    """

    def __init__(self, width: int = 360, height: int = 280):
        self.width = width
        self.height = height

        # Camera Orbit Angles (degrees)
        self.azimuth_deg: float = 25.0    # Orbit left/right
        self.elevation_deg: float = -15.0 # Orbit up/down
        self.zoom: float = 1.0            # Camera distance multiplier
        self.camera_dist: float = 0.35    # Base camera distance in meters
        self.fov_focal: float = 400.0     # Perspective focal length

    def set_camera_orbit(self, azimuth_deg: float, elevation_deg: float, zoom: float = 1.0) -> None:
        """Set 3D orbit camera rotation and zoom."""
        self.azimuth_deg = azimuth_deg
        self.elevation_deg = elevation_deg
        self.zoom = max(0.4, min(3.0, zoom))

    def reset_view(self) -> None:
        """Reset camera to default isometric-style view."""
        self.azimuth_deg = 25.0
        self.elevation_deg = -15.0
        self.zoom = 1.0

    def _get_rotation_matrix(self) -> np.ndarray:
        """Calculate combined 3D rotation matrix for azimuth and elevation."""
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)

        # Yaw (Azimuth around Y)
        R_yaw = np.array([
            [math.cos(az), 0, math.sin(az)],
            [0, 1, 0],
            [-math.sin(az), 0, math.cos(az)],
        ], dtype=np.float32)

        # Pitch (Elevation around X)
        R_pitch = np.array([
            [1, 0, 0],
            [0, math.cos(el), -math.sin(el)],
            [0, math.sin(el), math.cos(el)],
        ], dtype=np.float32)

        return R_pitch @ R_yaw

    def _project_point(
        self,
        pt: Tuple[float, float, float],
        center: Tuple[float, float, float],
        R: np.ndarray,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Transform and project a 3D point (meters) into 2D screen coordinates.
        Returns (screen_x, screen_y, depth_z) or None if behind camera.
        """
        # Center relative to palm center
        rel = np.array([pt[0] - center[0], pt[1] - center[1], pt[2] - center[2]], dtype=np.float32)
        rotated = R @ rel

        cam_dist = self.camera_dist / self.zoom
        z_cam = cam_dist + rotated[2]

        if z_cam < 0.05:
            return None

        # Perspective projection
        f = self.fov_focal * self.zoom
        screen_x = int(self.width / 2 + (rotated[0] * f) / z_cam)
        screen_y = int(self.height / 2 - (rotated[1] * f) / z_cam)
        return (screen_x, screen_y, float(z_cam))

    def _draw_grid(self, canvas: np.ndarray, R: np.ndarray) -> None:
        """Draw subtle 3D floor perspective grid."""
        grid_size = 0.12  # meters
        steps = 4
        step_sz = grid_size / steps
        y_floor = -0.06

        lines_3d = []
        for i in range(-steps, steps + 1):
            offset = i * step_sz
            lines_3d.append(((offset, y_floor, -grid_size), (offset, y_floor, grid_size)))
            lines_3d.append(((-grid_size, y_floor, offset), (grid_size, y_floor, offset)))

        origin = (0.0, 0.0, 0.0)
        for p1, p2 in lines_3d:
            s1 = self._project_point(p1, origin, R)
            s2 = self._project_point(p2, origin, R)
            if s1 and s2:
                cv2.line(canvas, (s1[0], s1[1]), (s2[0], s2[1]), COLOR_GRID, 1, cv2.LINE_AA)

    def render(self, hand_model: Optional[HandModel3D]) -> np.ndarray:
        """
        Render the 3D hand model into an RGB image.
        Returns a (H, W, 3) uint8 numpy array in RGB format.
        """
        # Initialize dark cyber background
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (16, 20, 26)  # Dark slate background BGR

        R = self._get_rotation_matrix()
        self._draw_grid(canvas, R)

        if hand_model is None or not hand_model.is_valid or not hand_model.joints_filtered:
            # Standby message
            cv2.putText(
                canvas,
                "3D HAND TRACKER STANDBY",
                (self.width // 2 - 110, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_TEXT_DIM,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Present hand in camera view",
                (self.width // 2 - 95, self.height // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                COLOR_TEXT_DIM,
                1,
                cv2.LINE_AA,
            )
            return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        joints_3d = hand_model.joints_filtered
        center = hand_model.orientation.palm_center

        # Project all 21 joints
        projected: List[Optional[Tuple[int, int, float]]] = []
        for j in joints_3d:
            projected.append(self._project_point(j, center, R))

        # 1. Render Bone Segments
        for bone in hand_model.bones:
            p1 = projected[bone.start_id]
            p2 = projected[bone.end_id]

            if p1 is None or p2 is None:
                continue

            color = FINGER_COLORS_BGR.get(bone.finger_name, COLOR_BONE_PALM)
            avg_depth = (p1[2] + p2[2]) / 2.0
            depth_factor = max(0.4, min(1.3, 0.35 / max(0.1, avg_depth)))

            # Adjust line thickness & intensity based on depth
            thickness = max(1, int(round(2.5 * depth_factor)))
            b_color = tuple(int(np.clip(c * depth_factor, 0, 255)) for c in color)

            cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), b_color, thickness, cv2.LINE_AA)

        # 2. Render Joint Spheres
        for i, proj in enumerate(projected):
            if proj is None:
                continue

            x, y, depth = proj
            depth_factor = max(0.4, min(1.4, 0.35 / max(0.1, depth)))
            radius = max(2, int(round(4.0 * depth_factor)))

            # Special radius for fingertips & wrist
            if i in (4, 8, 12, 16, 20):
                radius = max(3, int(round(5.5 * depth_factor)))
                joint_color = (255, 255, 255)
            elif i == JOINT_WRIST:
                radius = max(3, int(round(6.0 * depth_factor)))
                joint_color = (0, 200, 255)
            else:
                joint_color = (210, 230, 250)

            # Outer glow / halo
            cv2.circle(canvas, (x, y), radius + 2, (40, 50, 60), 1, cv2.LINE_AA)
            # Solid core
            cv2.circle(canvas, (x, y), radius, joint_color, -1, cv2.LINE_AA)

        # 3. Render 3D Palm Normal Arrow
        normal = hand_model.orientation.palm_normal
        arrow_length = 0.045  # 4.5 cm in 3D
        norm_endpoint = (
            center[0] + normal[0] * arrow_length,
            center[1] + normal[1] * arrow_length,
            center[2] + normal[2] * arrow_length,
        )
        p_center = self._project_point(center, center, R)
        p_normal = self._project_point(norm_endpoint, center, R)

        if p_center and p_normal:
            cv2.arrowedLine(
                canvas,
                (p_center[0], p_center[1]),
                (p_normal[0], p_normal[1]),
                COLOR_NORMAL_VEC,
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )

        # 4. Telemetry Overlay
        ori = hand_model.orientation
        cv2.putText(
            canvas,
            f"{hand_model.handedness.upper()} HAND | 3D ARTICULATED",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )
        telemetry_line2 = (
            f"P:{ori.pitch_deg:+.0f}° Y:{ori.yaw_deg:+.0f}° R:{ori.roll_deg:+.0f}° | "
            f"{ori.facing.value.replace('FACING_', '')}"
        )
        cv2.putText(
            canvas,
            telemetry_line2,
            (10, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )
        telemetry_line3 = (
            f"Scale: {hand_model.palm_scale * 100:.1f}cm | Speed: {ori.angular_speed_deg_per_sec:.0f}°/s"
        )
        cv2.putText(
            canvas,
            telemetry_line3,
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            COLOR_TEXT_DIM,
            1,
            cv2.LINE_AA,
        )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

