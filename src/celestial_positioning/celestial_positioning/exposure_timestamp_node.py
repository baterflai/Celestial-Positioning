import os
import time
import threading

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header

try:
    import gpiod
    HAS_GPIOD = True
    # Detect gpiod API version: v1 exposes Chip, v2 exposes LineRequest
    GPIOD_V2 = hasattr(gpiod, 'LineSettings')
except ImportError:
    HAS_GPIOD = False
    GPIOD_V2 = False


class ExposureTimestampNode(Node):
    """Monitors the IMX296 STROBE GPIO line and publishes precise
    exposure-start and exposure-end timestamps.

    The STROBE pin on the IMX296 asserts high during sensor exposure. Rising
    edge = exposure start, falling edge = exposure end. Reading this with
    PPS-disciplined system time gives microsecond-accurate exposure timing.

    If no strobe is wired (GPIO not asserting), this node simply idles.
    """

    def __init__(self):
        super().__init__('exposure_timestamp_node')

        self.declare_parameter('gpio_chip', 'gpiochip0')
        self.declare_parameter('strobe_pin', 17)
        self.declare_parameter('frame_id', 'camera_strobe')

        chip_name = self.get_parameter('gpio_chip').value
        self.pin = self.get_parameter('strobe_pin').value
        self.frame_id = self.get_parameter('frame_id').value

        if not HAS_GPIOD:
            self.get_logger().warn(
                'gpiod not available — exposure timestamping disabled')
            return

        self.rising_pub = self.create_publisher(
            Header, '/camera/exposure_start', 50)
        self.falling_pub = self.create_publisher(
            Header, '/camera/exposure_end', 50)

        self._thread = threading.Thread(
            target=self._watch_loop,
            args=(chip_name,),
            daemon=True)
        self._stop = threading.Event()
        self._thread.start()

        self.get_logger().info(
            f'Exposure timestamp watcher on /dev/{chip_name} line {self.pin}. '
            f'Publishes /camera/exposure_start and /camera/exposure_end. '
            f'If strobe pin is not wired, no events will fire.')

    def _monotonic_to_realtime_offset_ns(self):
        """Return (realtime_ns - monotonic_ns). Used to convert gpiod edge
        timestamps (CLOCK_MONOTONIC) to CLOCK_REALTIME for ROS headers."""
        # Sample both clocks as close together as possible (3x for best of)
        samples = []
        for _ in range(3):
            rt = time.clock_gettime_ns(time.CLOCK_REALTIME)
            mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            rt2 = time.clock_gettime_ns(time.CLOCK_REALTIME)
            # Use the average of rt and rt2 to minimise skew between the two
            # clock reads
            samples.append(((rt + rt2) // 2) - mono)
        # Use median for robustness
        samples.sort()
        return samples[1]

    def _publish_edge(self, pub, ts_ns):
        hdr = Header()
        hdr.frame_id = self.frame_id
        hdr.stamp.sec = ts_ns // 1_000_000_000
        hdr.stamp.nanosec = ts_ns % 1_000_000_000
        pub.publish(hdr)

    def _watch_loop(self, chip_name):
        try:
            if GPIOD_V2:
                self._watch_v2(chip_name)
            else:
                self._watch_v1(chip_name)
        except Exception as e:
            self.get_logger().error(f'GPIO watcher crashed: {e}')

    def _watch_v2(self, chip_name):
        """gpiod 2.x API (newer systems)."""
        import gpiod
        from gpiod.line import Edge, Direction
        path = f'/dev/{chip_name}'
        with gpiod.request_lines(
                path,
                consumer='exposure_ts',
                config={self.pin: gpiod.LineSettings(
                    direction=Direction.INPUT,
                    edge_detection=Edge.BOTH)}) as req:
            while not self._stop.is_set():
                if req.wait_edge_events(timeout=1.0):
                    offset = self._monotonic_to_realtime_offset_ns()
                    for ev in req.read_edge_events():
                        # ev.timestamp_ns is CLOCK_MONOTONIC
                        rt_ns = ev.timestamp_ns + offset
                        if ev.event_type == ev.Type.RISING_EDGE:
                            self._publish_edge(self.rising_pub, rt_ns)
                        elif ev.event_type == ev.Type.FALLING_EDGE:
                            self._publish_edge(self.falling_pub, rt_ns)

    def _watch_v1(self, chip_name):
        """gpiod 1.x API (Ubuntu 22.04 default)."""
        import gpiod
        chip = gpiod.Chip(chip_name)
        line = chip.get_line(self.pin)
        line.request(
            consumer='exposure_ts',
            type=gpiod.LINE_REQ_EV_BOTH_EDGES)
        while not self._stop.is_set():
            if line.event_wait(sec=1):
                offset = self._monotonic_to_realtime_offset_ns()
                event = line.event_read()
                # event.sec + event.nsec is CLOCK_MONOTONIC
                mono_ns = event.sec * 1_000_000_000 + event.nsec
                rt_ns = mono_ns + offset
                if event.type == gpiod.LineEvent.RISING_EDGE:
                    self._publish_edge(self.rising_pub, rt_ns)
                elif event.type == gpiod.LineEvent.FALLING_EDGE:
                    self._publish_edge(self.falling_pub, rt_ns)
        line.release()
        chip.close()

    def destroy_node(self):
        if hasattr(self, '_stop'):
            self._stop.set()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExposureTimestampNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
