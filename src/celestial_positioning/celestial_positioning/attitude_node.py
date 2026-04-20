from __future__ import annotations

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Header

from attitude_ekf import AttitudeEKF, AttitudeEKFConfig


class AttitudeNode(Node):
    """
    Thin ROS wrapper for AttitudeEKF.

    Inputs:
      - /imu/rpy       geometry_msgs/Vector3Stamped
      - /visual/rpy    geometry_msgs/Vector3Stamped

    Output:
      - /attitude/rpy  geometry_msgs/Vector3Stamped
    """

    def __init__(self) -> None:
        super().__init__("attitude_node")

        self.declare_parameter("imu_rpy_topic", "/imu/rpy")
        self.declare_parameter("visual_rpy_topic", "/visual/rpy")
        self.declare_parameter("attitude_topic", "/attitude/rpy")
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("imu_meas_std_rad", 0.05)
        self.declare_parameter("visual_meas_std_rad", 0.02)
        self.declare_parameter("process_std_rad", 0.005)

        imu_rpy_topic = str(self.get_parameter("imu_rpy_topic").value)
        visual_rpy_topic = str(self.get_parameter("visual_rpy_topic").value)
        attitude_topic = str(self.get_parameter("attitude_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        cfg = AttitudeEKFConfig(
            imu_meas_std_rad=float(self.get_parameter("imu_meas_std_rad").value),
            visual_meas_std_rad=float(self.get_parameter("visual_meas_std_rad").value),
            process_std_rad=float(self.get_parameter("process_std_rad").value),
        )
        self.ekf = AttitudeEKF(cfg)

        self.att_pub = self.create_publisher(Vector3Stamped, attitude_topic, 10)

        self.imu_sub = self.create_subscription(
            Vector3Stamped, imu_rpy_topic, self.imu_rpy_callback, 50
        )
        self.visual_sub = self.create_subscription(
            Vector3Stamped, visual_rpy_topic, self.visual_rpy_callback, 20
        )

        self.get_logger().info("Attitude node started")
        self.get_logger().info(f"Subscribing to {imu_rpy_topic} and {visual_rpy_topic}")
        self.get_logger().info(f"Publishing fused attitude on {attitude_topic}")

    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def publish_attitude(self, stamp) -> None:
        rpy = self.ekf.rpy

        msg = Vector3Stamped()
        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.vector.x = float(rpy[0])   # roll
        msg.vector.y = float(rpy[1])   # pitch
        msg.vector.z = float(rpy[2])   # yaw
        self.att_pub.publish(msg)

    def imu_rpy_callback(self, msg: Vector3Stamped) -> None:
        t = self.stamp_to_sec(msg.header.stamp)
        rpy = [msg.vector.x, msg.vector.y, msg.vector.z]

        self.ekf.correct_with_imu_rpy(t, rpy)
        self.publish_attitude(msg.header.stamp)

    def visual_rpy_callback(self, msg: Vector3Stamped) -> None:
        t = self.stamp_to_sec(msg.header.stamp)
        rpy = [msg.vector.x, msg.vector.y, msg.vector.z]

        self.ekf.correct_with_visual_rpy(t, rpy)
        self.publish_attitude(msg.header.stamp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AttitudeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()