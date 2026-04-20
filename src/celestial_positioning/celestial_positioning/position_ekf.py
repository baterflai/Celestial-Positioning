from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


GRAVITY_MPS2 = 9.80665


def gravity_world() -> np.ndarray:
    """
    World frame uses +Z up, so gravity points in -Z.
    """
    return np.array([0.0, 0.0, -GRAVITY_MPS2], dtype=float)


@dataclass
class PositionEKFConfig:
    """
    9-state position filter on [p, v, b_a].

    accel_process_std_mps2:
        Process noise on the linear acceleration input.
    accel_bias_walk_std_mps2:
        Random walk on accelerometer bias state.
    visual_pos_meas_std_m:
        Noise on visual position measurement.
    """
    accel_process_std_mps2: float = 0.8
    accel_bias_walk_std_mps2: float = 0.03
    visual_pos_meas_std_m: float = 2.0


class PositionEKF:
    """
    Standalone position estimator.

    State:
        x = [px, py, pz, vx, vy, vz, bax, bay, baz]^T

    Inputs each IMU step:
        - timestamp
        - raw body-frame accelerometer measurement
        - fused attitude rotation matrix R_wb

    Measurement updates:
        - visual position in world frame

    Notes:
        accel_world = R_wb @ accel_body - gravity_world()
        a_used = accel_world - accel_bias
    """

    def __init__(self, config: Optional[PositionEKFConfig] = None) -> None:
        self.config = config or PositionEKFConfig()

        self.x = np.zeros(9, dtype=float)
        self.P = np.eye(9, dtype=float) * 1.0
        self.last_timestamp_s: Optional[float] = None
        self.initialized = False

    def initialize(
        self,
        timestamp_s: float,
        position_m: Optional[np.ndarray] = None,
        velocity_mps: Optional[np.ndarray] = None,
        accel_bias_mps2: Optional[np.ndarray] = None,
    ) -> None:
        self.x[:] = 0.0
        if position_m is not None:
            self.x[0:3] = np.array(position_m, dtype=float).reshape(3)
        if velocity_mps is not None:
            self.x[3:6] = np.array(velocity_mps, dtype=float).reshape(3)
        if accel_bias_mps2 is not None:
            self.x[6:9] = np.array(accel_bias_mps2, dtype=float).reshape(3)

        self.P = np.eye(9, dtype=float)
        self.P[0:3, 0:3] *= 2.0
        self.P[3:6, 3:6] *= 1.0
        self.P[6:9, 6:9] *= 0.25

        self.last_timestamp_s = float(timestamp_s)
        self.initialized = True

    def predict(
        self,
        timestamp_s: float,
        accel_body_mps2: np.ndarray,
        R_wb: np.ndarray,
    ) -> None:
        """
        Propagate using raw IMU acceleration and fused attitude.
        """
        timestamp_s = float(timestamp_s)

        if not self.initialized:
            self.initialize(timestamp_s)
            return

        if self.last_timestamp_s is None:
            self.last_timestamp_s = timestamp_s
            return

        dt = timestamp_s - self.last_timestamp_s
        self.last_timestamp_s = timestamp_s

        if dt <= 0.0:
            return

        accel_body = np.array(accel_body_mps2, dtype=float).reshape(3)
        R_wb = np.array(R_wb, dtype=float).reshape(3, 3)

        accel_world = R_wb @ accel_body - gravity_world()
        bias = self.x[6:9].copy()
        a = accel_world - bias

        # State propagation
        p = self.x[0:3].copy()
        v = self.x[3:6].copy()

        self.x[0:3] = p + v * dt + 0.5 * a * (dt ** 2)
        self.x[3:6] = v + a * dt
        # bias is random walk, stays same in nominal propagation

        # Linearized covariance propagation
        F = np.eye(9, dtype=float)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * np.eye(3) * (dt ** 2)
        F[3:6, 6:9] = -np.eye(3) * dt

        G = np.zeros((9, 6), dtype=float)
        G[0:3, 0:3] = 0.5 * np.eye(3) * (dt ** 2)
        G[3:6, 0:3] = np.eye(3) * dt
        G[6:9, 3:6] = np.eye(3) * dt

        Qa = (self.config.accel_process_std_mps2 ** 2) * np.eye(3)
        Qb = (self.config.accel_bias_walk_std_mps2 ** 2) * np.eye(3)
        Q = np.zeros((6, 6), dtype=float)
        Q[0:3, 0:3] = Qa
        Q[3:6, 3:6] = Qb

        self.P = F @ self.P @ F.T + G @ Q @ G.T

    def correct_with_visual_position(
        self,
        timestamp_s: float,
        position_world_m: np.ndarray,
    ) -> None:
        """
        Correct using visual position measurement in world frame.
        """
        if not self.initialized:
            self.initialize(timestamp_s, position_m=position_world_m)
            return

        # Optional tiny predict sync if the visual measurement is newer
        if self.last_timestamp_s is not None and timestamp_s > self.last_timestamp_s:
            self.last_timestamp_s = float(timestamp_s)

        z = np.array(position_world_m, dtype=float).reshape(3)

        H = np.zeros((3, 9), dtype=float)
        H[:, 0:3] = np.eye(3)

        R = np.eye(3, dtype=float) * (self.config.visual_pos_meas_std_m ** 2)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(9, dtype=float)
        self.P = (I - K @ H) @ self.P

    @property
    def position_m(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def velocity_mps(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def accel_bias_mps2(self) -> np.ndarray:
        return self.x[6:9].copy()