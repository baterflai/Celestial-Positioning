import struct
import time
import calendar

import rclpy
import smbus2
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus


NEO_M9N_ADDR = 0x42
NEO_M9N_DATA_REG = 0xFF
NEO_M9N_LEN_HI = 0xFD
NEO_M9N_LEN_LO = 0xFE

UBX_SYNC = b'\xb5\x62'
UBX_NAV_PVT_CLASS = 0x01
UBX_NAV_PVT_ID = 0x07
UBX_CFG_MSG_CLASS = 0x06
UBX_CFG_MSG_ID = 0x01

# valid flags in UBX-NAV-PVT
VALID_DATE = 0x01
VALID_TIME = 0x02
VALID_FULLY_RESOLVED = 0x04

FIX_TYPE_MAP = {
    0: NavSatStatus.STATUS_NO_FIX,
    1: NavSatStatus.STATUS_NO_FIX,
    2: NavSatStatus.STATUS_FIX,
    3: NavSatStatus.STATUS_FIX,
    4: NavSatStatus.STATUS_GBAS_FIX,
    5: NavSatStatus.STATUS_NO_FIX,
}

# chrony / ntpd SHM refclock segment layout (from chrony source)
# Key = 0x4e545030 + unit (where "NTP0" = 0x4e545030)
import sysv_ipc
import ctypes


class ShmTime(ctypes.Structure):
    """Layout matching chrony/ntpd SHM refclock driver."""
    _fields_ = [
        ("mode", ctypes.c_int),
        ("count", ctypes.c_int),  # incremented around updates
        ("clockTimeStampSec", ctypes.c_long),
        ("clockTimeStampUSec", ctypes.c_int),
        ("receiveTimeStampSec", ctypes.c_long),
        ("receiveTimeStampUSec", ctypes.c_int),
        ("leap", ctypes.c_int),
        ("precision", ctypes.c_int),
        ("nsamples", ctypes.c_int),
        ("valid", ctypes.c_int),
        ("clockTimeStampNSec", ctypes.c_uint),
        ("receiveTimeStampNSec", ctypes.c_uint),
        ("_dummy", ctypes.c_int * 8),
    ]


def ubx_checksum(payload):
    ck_a, ck_b = 0, 0
    for b in payload:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return ck_a, ck_b


def ubx_poll_msg(cls, msg_id):
    length = 0
    header = struct.pack('<BBH', cls, msg_id, length)
    ck_a, ck_b = ubx_checksum(header)
    return b'\xb5\x62' + header + bytes([ck_a, ck_b])


class ChronyShmWriter:
    """Writes GPS timestamps to a chrony/ntpd SHM refclock segment."""

    def __init__(self, unit=0, logger=None):
        self.unit = unit
        self.logger = logger
        key = 0x4e545030 + unit  # "NTP" + unit digit
        try:
            self.shm = sysv_ipc.SharedMemory(
                key, flags=sysv_ipc.IPC_CREAT,
                mode=0o666, size=ctypes.sizeof(ShmTime))
            self.attached = True
            if logger:
                logger.info(
                    f'Chrony SHM unit {unit} attached (key 0x{key:08x})')
        except sysv_ipc.Error as e:
            self.attached = False
            if logger:
                logger.warn(f'Failed to attach chrony SHM: {e}')

    def update(self, gps_time_unix_ns, receive_time_unix_ns):
        """Publish a GPS time sample to chrony.

        Args:
            gps_time_unix_ns: GPS-derived Unix timestamp in nanoseconds
            receive_time_unix_ns: Local clock time when GPS sample arrived
        """
        if not self.attached:
            return
        buf = self.shm.read(byte_count=ctypes.sizeof(ShmTime))
        shm_time = ShmTime.from_buffer_copy(buf)

        # Chrony's update protocol: increment count, write fields, increment
        # count again. Odd count signals "update in progress."
        shm_time.mode = 1  # we provide both clock and receive times
        shm_time.count += 1

        gps_sec = gps_time_unix_ns // 1_000_000_000
        gps_nsec = gps_time_unix_ns % 1_000_000_000
        rx_sec = receive_time_unix_ns // 1_000_000_000
        rx_nsec = receive_time_unix_ns % 1_000_000_000

        shm_time.clockTimeStampSec = gps_sec
        shm_time.clockTimeStampUSec = gps_nsec // 1000
        shm_time.clockTimeStampNSec = gps_nsec
        shm_time.receiveTimeStampSec = rx_sec
        shm_time.receiveTimeStampUSec = rx_nsec // 1000
        shm_time.receiveTimeStampNSec = rx_nsec
        shm_time.leap = 0
        shm_time.precision = -10  # ~1ms precision (2^-10 sec)
        shm_time.valid = 1
        shm_time.count += 1

        self.shm.write(bytes(shm_time))


class GpsNode(Node):

    def __init__(self):
        super().__init__('gps_node')

        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_addr', NEO_M9N_ADDR)
        self.declare_parameter('rate_hz', 1.0)
        self.declare_parameter('frame_id', 'gps_link')
        self.declare_parameter('chrony_shm_unit', 0)
        self.declare_parameter('use_gps_time_in_header', True)

        bus_num = self.get_parameter('i2c_bus').value
        self.addr = self.get_parameter('i2c_addr').value
        rate = self.get_parameter('rate_hz').value
        self.frame_id = self.get_parameter('frame_id').value
        shm_unit = self.get_parameter('chrony_shm_unit').value
        self.use_gps_time = self.get_parameter('use_gps_time_in_header').value

        self.fix_pub = self.create_publisher(NavSatFix, '/gps/fix', 10)

        self.bus = smbus2.SMBus(bus_num)
        time.sleep(0.1)

        self._configure_m9n()

        self.shm = ChronyShmWriter(unit=shm_unit, logger=self.get_logger())

        self.timer = self.create_timer(1.0 / rate, self._poll_and_publish)
        self.get_logger().info(
            f'GPS node started on /dev/i2c-{bus_num} addr 0x{self.addr:02X} '
            f'at {rate} Hz, chrony_shm_unit={shm_unit}, '
            f'use_gps_time={self.use_gps_time}')

    def _configure_m9n(self):
        """Enable UBX-NAV-PVT on I2C port at 1 Hz."""
        payload = struct.pack('BBBBBBBB',
                              UBX_NAV_PVT_CLASS, UBX_NAV_PVT_ID,
                              1, 0, 0, 0, 0, 0)
        header = struct.pack('<BBH', UBX_CFG_MSG_CLASS, UBX_CFG_MSG_ID,
                             len(payload))
        body = header + payload
        ck_a, ck_b = ubx_checksum(body)
        msg = b'\xb5\x62' + body + bytes([ck_a, ck_b])

        wr = smbus2.i2c_msg.write(self.addr, list(msg))
        self.bus.i2c_rdwr(wr)
        time.sleep(0.1)
        self.get_logger().info('Configured NEO-M9N for UBX-NAV-PVT on I2C')

    def _read_available(self):
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

    def _gps_time_to_unix_ns(self, year, month, day, hour, minute, second,
                             nano, valid):
        """Convert UBX-NAV-PVT date/time fields to Unix nanoseconds."""
        if not (valid & (VALID_DATE | VALID_TIME)):
            return None
        if year < 2020 or year > 2100:
            return None
        try:
            ts = calendar.timegm(
                (year, month, day, hour, minute, second, 0, 0, 0))
        except (ValueError, OverflowError):
            return None
        # `nano` is signed int32 of nanoseconds past "second" boundary.
        # Can be negative (time is actually slightly before the second).
        total_ns = ts * 1_000_000_000 + int(nano)
        return total_ns

    def _parse_ubx_nav_pvt(self, payload, receive_time_ns):
        if len(payload) < 92:
            return None

        (iTOW, year, month, day, hour, minute, second, valid,
         tAcc, nano, fixType, flags, flags2, numSV,
         lon, lat, height, hMSL,
         hAcc, vAcc) = struct.unpack_from(
             '<IHBBBBBBIiBBBBiiiiII', payload, 0)

        msg = NavSatFix()

        # GPS-derived UTC time for the header when available AND time is valid
        # AND we have a fix. Otherwise use local clock (which chrony is
        # hopefully disciplining from GPS anyway).
        gps_time_ns = self._gps_time_to_unix_ns(
            year, month, day, hour, minute, second, nano, valid)

        has_fix = fixType in (2, 3, 4)
        use_gps_stamp = (self.use_gps_time and gps_time_ns is not None
                          and has_fix)

        if use_gps_stamp:
            msg.header.stamp.sec = gps_time_ns // 1_000_000_000
            msg.header.stamp.nanosec = gps_time_ns % 1_000_000_000
        else:
            now = self.get_clock().now().to_msg()
            msg.header.stamp = now

        msg.header.frame_id = self.frame_id

        msg.status.status = FIX_TYPE_MAP.get(
            fixType, NavSatStatus.STATUS_NO_FIX)
        msg.status.service = NavSatStatus.SERVICE_GPS

        msg.latitude = lat * 1e-7
        msg.longitude = lon * 1e-7
        msg.altitude = hMSL * 1e-3

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
            f'alt={msg.altitude:.1f}m hAcc={h_acc_m:.1f}m '
            f'tAcc={tAcc/1e6:.3f}ms valid=0x{valid:02x}',
            throttle_duration_sec=5)

        # Return parsed message + time info so the caller can pick the
        # newest one to feed chrony (older messages in a batched read are
        # stale and would be rejected as outliers)
        return msg, gps_time_ns, has_fix

    def _poll_and_publish(self):
        try:
            data = self._read_available()
        except OSError as e:
            self.get_logger().warn(
                f'I2C read error: {e}', throttle_duration_sec=5)
            return

        # Capture receive timestamp as close to the I2C read as possible
        receive_time_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)

        if not data:
            poll = ubx_poll_msg(UBX_NAV_PVT_CLASS, UBX_NAV_PVT_ID)
            try:
                wr = smbus2.i2c_msg.write(self.addr, list(poll))
                self.bus.i2c_rdwr(wr)
            except OSError:
                pass
            return

        latest_gps_ns = None
        n_parsed = 0
        idx = 0
        while idx < len(data) - 5:
            if data[idx] == 0xB5 and data[idx + 1] == 0x62:
                cls = data[idx + 2]
                msg_id = data[idx + 3]
                length = struct.unpack_from('<H', data, idx + 4)[0]
                frame_end = idx + 6 + length + 2

                if frame_end > len(data):
                    break

                if cls == UBX_NAV_PVT_CLASS and msg_id == UBX_NAV_PVT_ID:
                    payload = data[idx + 6:idx + 6 + length]
                    result = self._parse_ubx_nav_pvt(
                        payload, receive_time_ns)
                    if result:
                        fix_msg, gps_ns, has_fix = result
                        self.fix_pub.publish(fix_msg)
                        n_parsed += 1
                        if has_fix and gps_ns is not None:
                            latest_gps_ns = gps_ns

                idx = frame_end
            else:
                idx += 1

        # Feed chrony only the MOST RECENT fix from this read batch.
        # Older messages in the batch are stale and would corrupt chrony's
        # filtered estimate.
        if latest_gps_ns is not None:
            # Adjust receive time by an estimate of I2C read latency. With
            # 34kB of buffered data at ~100kbit/s I2C, the data we just
            # finished reading was generated progressively over the read
            # window. The newest message arrived at the END of the read.
            # receive_time_ns (captured just after the read) is correct.
            try:
                self.shm.update(latest_gps_ns, receive_time_ns)
                offset_ms = (latest_gps_ns - receive_time_ns) / 1e6
                self.get_logger().debug(
                    f'SHM update: {n_parsed} msgs batched, '
                    f'GPS-local offset={offset_ms:+.1f}ms',
                    throttle_duration_sec=10)
            except Exception as e:
                self.get_logger().warn(
                    f'Chrony SHM update failed: {e}',
                    throttle_duration_sec=30)


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
