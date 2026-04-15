from __future__ import annotations

from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, MagneticField
from geometry_msgs.msg import Vector3Stamped

from complementary_filter import ComplementaryFilter, ComplementaryFilterConfig


class ImuComplementaryFilterNode(Node):
    """
    Thin ROS wrapper:
      - subscribes to /imu/data_raw and /imu/mag
      - pushes data into ComplementaryFilter
      - publishes /imu/rpy
    """

    def __init__(self) -> None:
        super().__init__("imu_complementary_filter_node")

        self.declare_parameter("imu_topic", "/imu/data_raw")
        self.declare_parameter("mag_topic", "/imu/mag")
        self.declare_parameter("rpy_topic", "/imu/rpy")

        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("alpha_rp", 0.98)
        self.declare_parameter("alpha_yaw", 0.98)
        self.declare_parameter("use_mag_yaw", True)

        imu_topic = str(self.get_parameter("imu_topic").value)
        mag_topic = str(self.get_parameter("mag_topic").value)
        rpy_topic = str(self.get_parameter("rpy_topic").value)

        self.frame_id = str(self.get_parameter("frame_id").value)

        cfg = ComplementaryFilterConfig(
            alpha_rp=float(self.get_parameter("alpha_rp").value),
            alpha_yaw=float(self.get_parameter("alpha_yaw").value),
            use_mag_yaw=bool(self.get_parameter("use_mag_yaw").value),
        )
        self.filter = ComplementaryFilter(cfg)

        self.last_mag: Optional[Tuple[float, float, float]] = None

        self.rpy_pub = self.create_publisher(Vector3Stamped, rpy_topic, 10)

        self.imu_sub = self.create_subscription(
            Imu, imu_topic, self.imu_callback, 100
        )
        self.mag_sub = self.create_subscription(
            MagneticField, mag_topic, self.mag_callback, 50
        )

        self.get_logger().info("IMU complementary filter node started")
        self.get_logger().info(f"Subscribing to {imu_topic} and {mag_topic}")
        self.get_logger().info(f"Publishing roll/pitch/yaw on {rpy_topic}")

    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def mag_callback(self, msg: MagneticField) -> None:
        self.last_mag = (
            float(msg.magnetic_field.x),
            float(msg.magnetic_field.y),
            float(msg.magnetic_field.z)*-1.0,
        )

    def imu_callback(self, msg: Imu) -> None:
        t = self.stamp_to_sec(msg.header.stamp)

        accel = (
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
        )
        gyro = (
            float(msg.angular_velocity.x),
            float(msg.angular_velocity.y),
            float(msg.angular_velocity.z),
        )

        roll, pitch, yaw = self.filter.update(
            timestamp=t,
            accel=accel,
            gyro=gyro,
            mag=self.last_mag,
        )

        out = Vector3Stamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.vector.x = roll
        out.vector.y = pitch
        out.vector.z = yaw
        self.rpy_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImuComplementaryFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
