from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def blend_angle(a: float, b: float, alpha: float) -> float:
    """
    Blend angle a toward angle b with wraparound handling.
    alpha = weight on a
    (1-alpha) = weight on b
    """
    err = wrap_angle(b - a)
    return wrap_angle(a + (1.0 - alpha) * err)


@dataclass
class ComplementaryFilterConfig:
    alpha_rp: float = 0.99
    alpha_yaw: float = 0.99
    use_mag_yaw: bool = True


class ComplementaryFilter:
    """
    Standalone complementary filter for roll, pitch, yaw.

    Inputs:
      - timestamp
      - accel (ax, ay, az) in m/s^2
      - gyro  (gx, gy, gz) in rad/s
      - optional mag (mx, my, mz)

    State:
      - roll, pitch, yaw in radians
    """

    def __init__(self, config: Optional[ComplementaryFilterConfig] = None) -> None:
        self.config = config or ComplementaryFilterConfig()

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.last_time: Optional[float] = None
        self.initialized = False

    def reset(self) -> None:
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_time = None
        self.initialized = False

    def get_rpy(self) -> Tuple[float, float, float]:
        return self.roll, self.pitch, self.yaw

    def update(
        self,
        timestamp: float,
        accel: Tuple[float, float, float],
        gyro: Tuple[float, float, float],
        mag: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        ax, ay, az = accel
        gx, gy, gz = gyro

        # Accelerometer-based tilt
        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.sqrt(ay * ay + az * az))

        if not self.initialized:
            self.roll = roll_acc
            self.pitch = pitch_acc

            if self.config.use_mag_yaw and mag is not None:
                self.yaw = self.compute_mag_yaw(self.roll, self.pitch, mag)
            else:
                self.yaw = 0.0

            self.last_time = timestamp
            self.initialized = True
            return self.get_rpy()

        dt = 0.0 if self.last_time is None else (timestamp - self.last_time)
        self.last_time = timestamp

        if dt <= 0.0:
            return self.get_rpy()

        # Gyro integration
        roll_gyro = self.roll + gx * dt
        pitch_gyro = self.pitch + gy * dt
        yaw_gyro = wrap_angle(self.yaw + gz * dt)

        # Roll/pitch complementary fusion
        self.roll = (
            self.config.alpha_rp * roll_gyro
            + (1.0 - self.config.alpha_rp) * roll_acc
        )
        self.pitch = (
            self.config.alpha_rp * pitch_gyro
            + (1.0 - self.config.alpha_rp) * pitch_acc
        )

        # Yaw complementary fusion
        if self.config.use_mag_yaw and mag is not None:
            yaw_mag = self.compute_mag_yaw(self.roll, self.pitch, mag)
            self.yaw = blend_angle(yaw_gyro, yaw_mag, self.config.alpha_yaw)
        else:
            self.yaw = yaw_gyro

        self.roll = wrap_angle(self.roll)
        self.pitch = wrap_angle(self.pitch)
        self.yaw = wrap_angle(self.yaw)

        return self.get_rpy()

    @staticmethod
    def compute_mag_yaw(
        roll: float,
        pitch: float,
        mag: Tuple[float, float, float],
    ) -> float:
        mx, my, mz = mag

        # Tilt compensation
        mx_h = mx * math.cos(pitch) + mz * math.sin(pitch)
        my_h = (
            mx * math.sin(roll) * math.sin(pitch)
            + my * math.cos(roll)
            - mz * math.sin(roll) * math.cos(pitch)
        )

        yaw = math.atan2(-my_h, mx_h)
        return wrap_angle(yaw)
