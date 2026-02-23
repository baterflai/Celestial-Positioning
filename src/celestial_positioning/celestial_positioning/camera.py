import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('device', 0)

        rate = self.get_parameter('publish_rate').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value
        device = self.get_parameter('device').value

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().fatal(f'Failed to open camera device {device}')
            raise RuntimeError(f'Cannot open /dev/video{device}')

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.timer = self.create_timer(1.0 / rate, self._capture_and_publish)
        self.get_logger().info(
            f'Camera node started — publishing at {rate} Hz ({actual_w}x{actual_h})'
        )

    def _capture_and_publish(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Failed to capture frame')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'

        self.publisher_.publish(msg)

    def destroy_node(self):
        self.get_logger().info('Shutting down camera…')
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
