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

        rate = self.get_parameter('publish_rate').value
        width = self.get_parameter('width').value
        height = self.get_parameter('height').value

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        pipeline = (
            f'libcamerasrc ! '
            f'video/x-raw,width={width},height={height},framerate={int(rate)}/1 ! '
            f'videoconvert ! '
            f'video/x-raw,format=BGR ! '
            f'appsink drop=1'
        )
        self.get_logger().info(f'Opening camera with GStreamer pipeline: {pipeline}')

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().fatal('Failed to open camera via GStreamer/libcamera')
            raise RuntimeError('Cannot open camera — is gstreamer1.0-libcamera installed?')

        self.timer = self.create_timer(1.0 / rate, self._capture_and_publish)
        self.get_logger().info(
            f'Camera node started — publishing at {rate} Hz ({width}x{height})'
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
