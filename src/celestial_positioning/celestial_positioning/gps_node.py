import struct
import time

import rclpy
import smbus2
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus


# NEO-M9N I2C defaults
NEO_M9N_ADDR = 0x42
NEO_M9N_DATA_REG = 0xFF
NEO_M9N_LEN_HI = 0xFD
NEO_M9N_LEN_LO = 0xFE

# UBX protocol constants
UBX_SYNC = b'\xb5\x62'
UBX_NAV_PVT_CLASS = 0x01
UBX_NAV_PVT_ID = 0x07

# UBX-NAV-PVT fix type mapping
FIX_TYPE_MAP = {
    0: NavSatStatus.STATUS_NO_FIX,      # no fix
    1: NavSatStatus.STATUS_NO_FIX,      # dead reckoning only
    2: NavSatStatus.STATUS_FIX,          # 2D
    3: NavSatStatus.STATUS_FIX,          # 3D
    4: NavSatStatus.STATUS_GBAS_FIX,    # GNSS + dead reckoning
    5: NavSatStatus.STATUS_NO_FIX,      # time only
}


def ubx_checksum(payload):
    """Compute UBX CK_A/CK_B over class+id+len+payload."""
    ck_a, ck_b = 0, 0
    for b in payload:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b


def ubx_poll_msg(cls, msg_id):
    """Build a UBX poll message (zero-length payload)."""
    length = 0
    header = struct.pack('<BBH', cls, msg_id, length)
    ck_a, ck_b = ubx_checksum(header)
    return b'\xb5\x62' + header + bytes([ck_a, ck_b])


class GpsNode(Node):

    def __init__(self):
        super().__init__('gps_node')

        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_addr', NEO_M9N_ADDR)
        self.declare_parameter('rate_hz', 1.0)
        self.declare_parameter('frame_id', 'gps_link')

        bus_num = self.get_parameter('i2c_bus').value
        self.addr = self.get_parameter('i2c_addr').value
        rate = self.get_parameter('rate_hz').value
        self.frame_id = self.get_parameter('frame_id').value

        self.fix_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)

        self.bus = smbus2.SMBus(bus_num)
        time.sleep(0.1)

        self._configure_m9n()

        self.timer = self.create_timer(1.0 / rate, self._poll_and_publish)
        self.get_logger().info(
            f'GPS node started on /dev/i2c-{bus_num} addr 0x{self.addr:02X} '
            f'at {rate} Hz')

    def _configure_m9n(self):
        """Enable UBX-NAV-PVT on I2C port at 1 Hz."""
        # UBX-CFG-MSG: enable NAV-PVT on I2C (port 0)
        # class=0x06, id=0x01, payload: msgClass, msgId, rate per port
        # For I2C (port 0), UART1 (port 1), etc. — set port 0 rate=1
        payload = struct.pack('BBBBBBBB',
                              UBX_NAV_PVT_CLASS, UBX_NAV_PVT_ID,
                              1, 0, 0, 0, 0, 0)  # I2C=1, others=0
        header = struct.pack('<BBH', 0x06, 0x01, len(payload))
        body = header + payload
        ck_a, ck_b = ubx_checksum(body)
        msg = b'\xb5\x62' + body + bytes([ck_a, ck_b])

        wr = smbus2.i2c_msg.write(self.addr, list(msg))
        self.bus.i2c_rdwr(wr)
        time.sleep(0.1)
        self.get_logger().info('Configured NEO-M9N for UBX-NAV-PVT on I2C')

    def _read_available(self):
        """Read all available bytes from the NEO-M9N."""
        hi = self.bus.read_byte_data(self.addr, NEO_M9N_LEN_HI)
        lo = self.bus.read_byte_data(self.addr, NEO_M9N_LEN_LO)
        avail = (hi << 8) | lo
        if avail == 0:
            return b''

        data = b''
        while avail > 0:
            chunk = min(avail, 32)
            rd = smbus2.i2c_msg.read(self.addr, chunk)
            self.bus.i2c_rdwr(rd)
            data += bytes(rd)
            avail -= chunk

        return data

    def _parse_ubx_nav_pvt(self, payload):
        """Parse a UBX-NAV-PVT payload (92 bytes) into a NavSatFix."""
        if len(payload) < 92:
            return None

        # Unpack key fields from the 92-byte payload
        (iTOW, year, month, day, hour, minute, second, valid,
         tAcc, nano, fixType, flags, flags2, numSV,
         lon, lat, height, hMSL,
         hAcc, vAcc) = struct.unpack_from('<IHBBBBBBIiBBBBiiiiII', payload, 0)

        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.status.status = FIX_TYPE_MAP.get(
            fixType, NavSatStatus.STATUS_NO_FIX)
        msg.status.service = NavSatStatus.SERVICE_GPS

        msg.latitude = lat * 1e-7
        msg.longitude = lon * 1e-7
        msg.altitude = hMSL * 1e-3  # mm to m

        # Horizontal and vertical accuracy in m
        h_acc_m = hAcc * 1e-3
        v_acc_m = vAcc * 1e-3
        msg.position_covariance = [
            h_acc_m ** 2, 0.0, 0.0,
            0.0, h_acc_m ** 2, 0.0,
            0.0, 0.0, v_acc_m ** 2,
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.get_logger().info(
            f'Fix: type={fixType} sats={numSV} '
            f'lat={msg.latitude:.7f} lon={msg.longitude:.7f} '
            f'alt={msg.altitude:.1f}m hAcc={h_acc_m:.1f}m',
            throttle_duration_sec=5)

        return msg

    def _poll_and_publish(self):
        try:
            data = self._read_available()
        except OSError as e:
            self.get_logger().warn(
                f'I2C read error: {e}', throttle_duration_sec=5)
            return

        if not data:
            # Send a poll request for NAV-PVT
            poll = ubx_poll_msg(UBX_NAV_PVT_CLASS, UBX_NAV_PVT_ID)
            try:
                wr = smbus2.i2c_msg.write(self.addr, list(poll))
                self.bus.i2c_rdwr(wr)
            except OSError:
                pass
            return

        # Scan for UBX frames in the data
        idx = 0
        while idx < len(data) - 5:
            if data[idx] == 0xB5 and data[idx + 1] == 0x62:
                cls = data[idx + 2]
                msg_id = data[idx + 3]
                length = struct.unpack_from('<H', data, idx + 4)[0]
                frame_end = idx + 6 + length + 2  # +2 for checksum

                if frame_end > len(data):
                    break

                if cls == UBX_NAV_PVT_CLASS and msg_id == UBX_NAV_PVT_ID:
                    payload = data[idx + 6:idx + 6 + length]
                    fix_msg = self._parse_ubx_nav_pvt(payload)
                    if fix_msg:
                        self.fix_pub.publish(fix_msg)

                idx = frame_end
            else:
                idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = GpsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.bus.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
