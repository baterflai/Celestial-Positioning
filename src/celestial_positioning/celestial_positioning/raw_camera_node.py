"""Raw Y10 (10-bit mono) camera publisher for IMX296LL.

Bypasses libcamera/ISP and reads directly from /dev/video0 via V4L2 to
preserve the sensor's native 10-bit depth and linear response. Publishes
sensor_msgs/Image with encoding 'mono16' where pixel values are in the
lower 10 bits (range 0-1023).
"""

import array
import subprocess

import numpy as np
import rclpy
from linuxpy.video.device import Device
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo

LINE_PERIOD_US = 9.26
SENSOR_HEIGHT_LINES = 1088
SENSOR_MIN_VBLANK = 30
SENSOR_MAX_VBLANK = 1047487


def set_sensor_controls(subdev_path, exposure_s, gain):
    """Set exposure and gain on the sensor subdevice."""
    exp_lines = max(1, min(
        int(round(exposure_s * 1e6 / LINE_PERIOD_US)),
        SENSOR_MAX_VBLANK))
    vblank = max(SENSOR_MIN_VBLANK,
                 min(exp_lines - SENSOR_HEIGHT_LINES + 20,
                     SENSOR_MAX_VBLANK))
    subprocess.run(
        ['v4l2-ctl', '-d', subdev_path,
         '--set-ctrl', f'vertical_blanking={vblank}'],
        capture_output=True, check=False)
    subprocess.run(
        ['v4l2-ctl', '-d', subdev_path,
         '--set-ctrl', f'exposure={exp_lines}'],
        capture_output=True, check=False)
    subprocess.run(
        ['v4l2-ctl', '-d', subdev_path,
         '--set-ctrl', f'analogue_gain={int(gain)}'],
        capture_output=True, check=False)
    return exp_lines


class RawCameraNode(Node):

    def __init__(self):
        super().__init__('raw_camera_node')

        self.declare_parameter('device_index', 0)
        self.declare_parameter('subdev_path', '/dev/v4l-subdev0')
        self.declare_parameter('width', 1456)
        self.declare_parameter('height', 1088)
        self.declare_parameter('frame_id', 'camera')
        self.declare_parameter('auto_exposure', True)
        self.declare_parameter('target_mean', 400.0)
        self.declare_parameter('exposure_s', 0.033)
        self.declare_parameter('gain', 0)
        self.declare_parameter('ae_min_exposure_s', 0.0001)
        self.declare_parameter('ae_max_exposure_s', 0.5)
        self.declare_parameter('ae_step_rate', 0.3)

        self.device_index = self.get_parameter('device_index').value
        self.subdev_path = self.get_parameter('subdev_path').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.frame_id = self.get_parameter('frame_id').value
        self.auto_exposure = self.get_parameter('auto_exposure').value
        self.target_mean = float(self.get_parameter('target_mean').value)
        self.exposure_s = float(self.get_parameter('exposure_s').value)
        self.gain = int(self.get_parameter('gain').value)
        self.ae_min_exp = float(self.get_parameter('ae_min_exposure_s').value)
        self.ae_max_exp = float(self.get_parameter('ae_max_exposure_s').value)
        self.ae_step_rate = float(self.get_parameter('ae_step_rate').value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.image_pub = self.create_publisher(
            Image, '/camera/image_raw', qos)
        self.info_pub = self.create_publisher(
            CameraInfo, '/camera/camera_info', qos)

        set_sensor_controls(
            self.subdev_path, self.exposure_s, self.gain)

        self.get_logger().info(
            f'Raw camera node started: {self.width}x{self.height} mono10, '
            f'auto_exposure={self.auto_exposure}, '
            f'initial exposure={self.exposure_s*1000:.1f}ms '
            f'gain={self.gain}')

    def run(self):
        """Capture loop — runs in the main thread, not an executor."""
        frame_count = 0
        with Device.from_id(self.device_index) as device:
            device.set_format(1, self.width, self.height, "Y10 ")
            self.get_logger().info(
                f'V4L2 device opened: Y10 {self.width}x{self.height}')

            for frame in device:
                if not rclpy.ok():
                    break

                now = self.get_clock().now().to_msg()
                data = array.array('B', frame.data)

                msg = Image()
                msg.header.stamp = now
                msg.header.frame_id = self.frame_id
                msg.height = self.height
                msg.width = self.width
                msg.encoding = 'mono16'
                msg.is_bigendian = 0
                msg.step = self.width * 2
                msg.data = data
                self.image_pub.publish(msg)

                info = CameraInfo()
                info.header = msg.header
                info.width = self.width
                info.height = self.height
                self.info_pub.publish(info)

                frame_count += 1

                if self.auto_exposure and frame_count % 10 == 0:
                    arr = np.frombuffer(bytes(data), dtype=np.uint16)
                    img_mean = float(arr.mean())
                    if img_mean < 1:
                        img_mean = 1
                    ratio = self.target_mean / img_mean
                    adjustment = ratio ** self.ae_step_rate
                    new_exp = self.exposure_s * adjustment
                    new_exp = max(self.ae_min_exp,
                                  min(self.ae_max_exp, new_exp))
                    if abs(new_exp - self.exposure_s) / self.exposure_s > 0.05:
                        self.exposure_s = new_exp
                        set_sensor_controls(
                            self.subdev_path, self.exposure_s, self.gain)

                if frame_count % 30 == 0:
                    arr = np.frombuffer(bytes(data), dtype=np.uint16)
                    self.get_logger().info(
                        f'Frame {frame_count}: '
                        f'exp={self.exposure_s*1000:.1f}ms '
                        f'gain={self.gain} '
                        f'mean={arr.mean():.0f} '
                        f'min={arr.min()} max={arr.max()}')


def main(args=None):
    rclpy.init(args=args)
    node = RawCameraNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
