import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class FeatureExtractorNode(Node):

    def __init__(self):
        super().__init__('feature_extractor_node')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._image_callback,
            10,
        )
        self.frame_count = 0
        self.get_logger().info('Feature extractor node started — waiting for images…')

    def _image_callback(self, msg: Image):
        self.frame_count += 1
        if self.frame_count % 30 == 1:
            self.get_logger().info(
                f'Received frame #{self.frame_count}: '
                f'{msg.width}x{msg.height} ({msg.encoding})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = FeatureExtractorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
