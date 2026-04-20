from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def wrap_vec_rpy(rpy: np.ndarray) -> np.ndarray:
    out = np.array(rpy, dtype=float).reshape(3)
    out[0] = wrap_angle(out[0])
    out[1] = wrap_angle(out[1])
    out[2] = wrap_angle(out[2])
    return out


def angle_residual(meas: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Wrapped innovation meas - pred for roll/pitch/yaw.
    """
    y = np.array(meas, dtype=float).reshape(3) - np.array(pred, dtype=float).reshape(3)
    return wrap_vec_rpy(y)


def euler_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=float)

    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=float)

    rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    return rz @ ry @ rx


@dataclass
class AttitudeEKFConfig:
    """
    3-state attitude filter on [roll, pitch, yaw].

    imu_meas_std_rad:
        Noise on complementary-filter IMU roll/pitch/yaw.
    visual_meas_std_rad:
        Noise on visual roll/pitch/yaw.
    process_std_rad:
        Small random-walk process noise on the attitude state.
    """
    imu_meas_std_rad: float = 0.05
    visual_meas_std_rad: float = 0.02
    process_std_rad: float = 0.005


class AttitudeEKF:
    """
    Standalone attitude estimator.

    State:
        x = [roll, pitch, yaw]^T

    Intended usage:
        - call correct_with_imu_rpy() using /imu/rpy
        - call correct_with_visual_rpy() using visual attitude
        - future ROS node publishes self.rpy and/or self.rotation_matrix()
    """

    def __init__(self, config: Optional[AttitudeEKFConfig] = None) -> None:
        self.config = config or AttitudeEKFConfig()

        self.x = np.zeros(3, dtype=float)
        self.P = np.eye(3, dtype=float) * 0.25
        self.last_timestamp_s: Optional[float] = None
        self.initialized = False

    def initialize(self, rpy_rad: np.ndarray, timestamp_s: float) -> None:
        self.x = wrap_vec_rpy(rpy_rad)
        self.P = np.eye(3, dtype=float) * 0.05
        self.last_timestamp_s = float(timestamp_s)
        self.initialized = True

    def predict(self, timestamp_s: float) -> None:
        """
        Since the complementary filter already gives an absolute attitude estimate,
        we use a simple random-walk process model here.
        """
        timestamp_s = float(timestamp_s)

        if not self.initialized:
            self.last_timestamp_s = timestamp_s
            return

        if self.last_timestamp_s is None:
            self.last_timestamp_s = timestamp_s
            return

        dt = timestamp_s - self.last_timestamp_s
        self.last_timestamp_s = timestamp_s

        if dt <= 0.0:
            return

        q = (self.config.process_std_rad ** 2) * max(dt, 1e-3)
        Q = np.eye(3, dtype=float) * q

        self.P = self.P + Q
        self.x = wrap_vec_rpy(self.x)

    def _correct(self, z_rpy_rad: np.ndarray, meas_std_rad: float) -> None:
        z = wrap_vec_rpy(z_rpy_rad)

        H = np.eye(3, dtype=float)
        R = np.eye(3, dtype=float) * (meas_std_rad ** 2)

        y = angle_residual(z, self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x = wrap_vec_rpy(self.x)

        I = np.eye(3, dtype=float)
        self.P = (I - K @ H) @ self.P

    def correct_with_imu_rpy(self, timestamp_s: float, rpy_rad: np.ndarray) -> None:
        if not self.initialized:
            self.initialize(rpy_rad, timestamp_s)
            return

        self.predict(timestamp_s)
        self._correct(rpy_rad, self.config.imu_meas_std_rad)

    def correct_with_visual_rpy(self, timestamp_s: float, rpy_rad: np.ndarray) -> None:
        if not self.initialized:
            self.initialize(rpy_rad, timestamp_s)
            return

        self.predict(timestamp_s)
        self._correct(rpy_rad, self.config.visual_meas_std_rad)

    @property
    def rpy(self) -> np.ndarray:
        return self.x.copy()

    def rotation_matrix(self) -> np.ndarray:
        roll, pitch, yaw = self.x
        return euler_to_rotmat(roll, pitch, yaw)