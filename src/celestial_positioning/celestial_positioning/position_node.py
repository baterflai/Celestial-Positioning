from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3Stamped, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from position_ekf import PositionEKF, PositionEKFConfig


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


class PositionNode(Node):
    """
    Thin ROS wrapper for PositionEKF.

    Inputs:
      - /imu/data_raw      sensor_msgs/Imu
      - /attitude/rpy      geometry_msgs/Vector3Stamped
      - /visual/position   geometry_msgs/PointStamped

    Output:
      - /position/state    nav_msgs/Odometry
    """

    def __init__(self) -> None:
        super().__init__("position_node")

        self.declare_parameter("imu_topic", "/imu/data_raw")
        self.declare_parameter("attitude_topic", "/attitude/rpy")
        self.declare_parameter("visual_position_topic", "/visual/position")
        self.declare_parameter("position_topic", "/position/state")

        self.declare_parameter("world_frame", "map")
        self.declare_parameter("body_frame", "imu_link")

        self.declare_parameter("accel_process_std_mps2", 0.8)
        self.declare_parameter("accel_bias_walk_std_mps2", 0.03)
        self.declare_parameter("visual_pos_meas_std_m", 2.0)

        imu_topic = str(self.get_parameter("imu_topic").value)
        attitude_topic = str(self.get_parameter("attitude_topic").value)
        visual_position_topic = str(self.get_parameter("visual_position_topic").value)
        position_topic = str(self.get_parameter("position_topic").value)

        self.world_frame = str(self.get_parameter("world_frame").value)
        self.body_frame = str(self.get_parameter("body_frame").value)

        cfg = PositionEKFConfig(
            accel_process_std_mps2=float(
                self.get_parameter("accel_process_std_mps2").value
            ),
            accel_bias_walk_std_mps2=float(
                self.get_parameter("accel_bias_walk_std_mps2").value
            ),
            visual_pos_meas_std_m=float(
                self.get_parameter("visual_pos_meas_std_m").value
            ),
        )
        self.ekf = PositionEKF(cfg)

        self.latest_rpy = np.zeros(3, dtype=float)
        self.have_attitude = False

        self.pos_pub = self.create_publisher(Odometry, position_topic, 10)

        self.imu_sub = self.create_subscription(
            Imu, imu_topic, self.imu_callback, 100
        )
        self.att_sub = self.create_subscription(
            Vector3Stamped, attitude_topic, self.attitude_callback, 50
        )
        self.visual_pos_sub = self.create_subscription(
            PointStamped, visual_position_topic, self.visual_position_callback, 20
        )

        self.get_logger().info("Position node started")
        self.get_logger().info(
            f"Subscribing to {imu_topic}, {attitude_topic}, and {visual_position_topic}"
        )
        self.get_logger().info(f"Publishing position state on {position_topic}")

    @staticmethod
    def stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def publish_state(self, stamp) -> None:
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame
        msg.child_frame_id = self.body_frame

        p = self.ekf.position_m
        v = self.ekf.velocity_mps

        msg.pose.pose.position.x = float(p[0])
        msg.pose.pose.position.y = float(p[1])
        msg.pose.pose.position.z = float(p[2])

        # orientation left as identity here since this node is for position output
        msg.pose.pose.orientation.w = 1.0

        msg.twist.twist.linear.x = float(v[0])
        msg.twist.twist.linear.y = float(v[1])
        msg.twist.twist.linear.z = float(v[2])

        # simple diagonal covariance fill from EKF covariance
        msg.pose.covariance[0] = float(self.ekf.P[0, 0])
        msg.pose.covariance[7] = float(self.ekf.P[1, 1])
        msg.pose.covariance[14] = float(self.ekf.P[2, 2])

        msg.twist.covariance[0] = float(self.ekf.P[3, 3])
        msg.twist.covariance[7] = float(self.ekf.P[4, 4])
        msg.twist.covariance[14] = float(self.ekf.P[5, 5])

        self.pos_pub.publish(msg)

    def attitude_callback(self, msg: Vector3Stamped) -> None:
        self.latest_rpy[0] = float(msg.vector.x)
        self.latest_rpy[1] = float(msg.vector.y)
        self.latest_rpy[2] = float(msg.vector.z)
        self.have_attitude = True

    def imu_callback(self, msg: Imu) -> None:
        if not self.have_attitude:
            return

        t = self.stamp_to_sec(msg.header.stamp)

        accel_body = np.array([
            float(msg.linear_acceleration.x),
            float(msg.linear_acceleration.y),
            float(msg.linear_acceleration.z),
        ], dtype=float)

        R_wb = euler_to_rotmat(
            self.latest_rpy[0],
            self.latest_rpy[1],
            self.latest_rpy[2],
        )

        self.ekf.predict(
            timestamp_s=t,
            accel_body_mps2=accel_body,
            R_wb=R_wb,
        )

        self.publish_state(msg.header.stamp)

    def visual_position_callback(self, msg: PointStamped) -> None:
        t = self.stamp_to_sec(msg.header.stamp)

        pos = np.array([
            float(msg.point.x),
            float(msg.point.y),
            float(msg.point.z),
        ], dtype=float)

        self.ekf.correct_with_visual_position(
            timestamp_s=t,
            position_world_m=pos,
        )

        self.publish_state(msg.header.stamp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()