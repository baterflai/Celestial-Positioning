import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from picamera2 import Picamera2


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

        self.camera = Picamera2()
        video_config = self.camera.create_video_configuration(
            main={'size': (width, height), 'format': 'RGB888'}
        )
        self.camera.configure(video_config)
        self.camera.start()

        self.timer = self.create_timer(1.0 / rate, self._capture_and_publish)
        self.get_logger().info(
            f'Camera node started — publishing at {rate} Hz ({width}x{height})'
        )

    def _capture_and_publish(self):
        frame = self.camera.capture_array('main')

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'

        self.publisher_.publish(msg)

    def destroy_node(self):
        self.get_logger().info('Shutting down camera…')
        self.camera.stop()
        self.camera.close()
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
