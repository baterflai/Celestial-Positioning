import struct
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField
import smbus2


# ISM330DHCX registers
ISM330DHCX_ADDR = 0x6B
ISM330DHCX_WHO_AM_I = 0x0F
ISM330DHCX_CTRL1_XL = 0x10  # Accelerometer ODR & scale
ISM330DHCX_CTRL2_G = 0x11   # Gyroscope ODR & scale
ISM330DHCX_STATUS = 0x1E
ISM330DHCX_OUTX_L_G = 0x22  # Gyro data start
ISM330DHCX_OUTX_L_A = 0x28  # Accel data start

# MMC5983MA registers
MMC5983MA_ADDR = 0x30
MMC5983MA_XOUT_0 = 0x00
MMC5983MA_STATUS = 0x08
MMC5983MA_CTRL0 = 0x09
MMC5983MA_CTRL1 = 0x0A
MMC5983MA_CTRL2 = 0x0B
MMC5983MA_WHO_AM_I = 0x2F

# Scale factors
ACCEL_SCALE_4G = 4.0 * 9.80665 / 32768.0   # m/s^2 per LSB at ±4g
GYRO_SCALE_500DPS = 500.0 / 32768.0          # deg/s per LSB at ±500dps
DEG_TO_RAD = 3.14159265358979 / 180.0
MAG_SCALE = 1.0 / 16384.0                    # Gauss per LSB (18-bit, ±8G)
GAUSS_TO_TESLA = 1e-4


class ImuNode(Node):

    def __init__(self):
        super().__init__('imu_node')

        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('frame_id', 'imu_link')

        bus_num = self.get_parameter('i2c_bus').value
        rate = self.get_parameter('rate_hz').value
        self.frame_id = self.get_parameter('frame_id').value

        self.imu_pub = self.create_publisher(Imu, '/imu/data_raw', 10)
        self.mag_pub = self.create_publisher(
            MagneticField, '/imu/mag', 10)

        self.bus = smbus2.SMBus(bus_num)
        time.sleep(0.1)

        self._init_ism330dhcx()
        self._init_mmc5983ma()

        self.timer = self.create_timer(1.0 / rate, self._read_and_publish)
        self.get_logger().info(
            f'IMU node started on /dev/i2c-{bus_num} at {rate} Hz')

    def _init_ism330dhcx(self):
        who = self.bus.read_byte_data(ISM330DHCX_ADDR, ISM330DHCX_WHO_AM_I)
        if who != 0x6B:
            self.get_logger().error(
                f'ISM330DHCX WHO_AM_I: expected 0x6B, got 0x{who:02X}')
            raise RuntimeError('ISM330DHCX not found')
        self.get_logger().info(f'ISM330DHCX detected (WHO_AM_I=0x{who:02X})')

        # Accelerometer: 104 Hz ODR, ±4g
        self.bus.write_byte_data(ISM330DHCX_ADDR, ISM330DHCX_CTRL1_XL, 0x48)
        # Gyroscope: 104 Hz ODR, ±500 dps
        self.bus.write_byte_data(ISM330DHCX_ADDR, ISM330DHCX_CTRL2_G, 0x44)

    def _init_mmc5983ma(self):
        try:
            who = self.bus.read_byte_data(
                MMC5983MA_ADDR, MMC5983MA_WHO_AM_I)
            if who != 0x30:
                self.get_logger().warn(
                    f'MMC5983MA WHO_AM_I: expected 0x30, got 0x{who:02X}')
                self.mag_available = False
                return
            self.get_logger().info(
                f'MMC5983MA detected (WHO_AM_I=0x{who:02X})')

            # Software reset
            self.bus.write_byte_data(MMC5983MA_ADDR, MMC5983MA_CTRL1, 0x80)
            time.sleep(0.05)

            # Continuous mode, 100 Hz (Cmm_en=1 | Cm_freq=101)
            self.bus.write_byte_data(MMC5983MA_ADDR, MMC5983MA_CTRL2, 0x0D)
            self.mag_available = True

        except OSError:
            self.get_logger().warn(
                'MMC5983MA not found, magnetometer disabled')
            self.mag_available = False

    def _read_6_bytes(self, addr, reg):
        data = self.bus.read_i2c_block_data(addr, reg, 6)
        x = struct.unpack_from('<h', bytes(data), 0)[0]
        y = struct.unpack_from('<h', bytes(data), 2)[0]
        z = struct.unpack_from('<h', bytes(data), 4)[0]
        return x, y, z

    def _read_mag(self):
        data = self.bus.read_i2c_block_data(
            MMC5983MA_ADDR, MMC5983MA_XOUT_0, 7)
        # 18-bit output: high byte, low byte, and 2 MSBs from register 6
        x = (data[0] << 10) | (data[1] << 2) | ((data[6] >> 6) & 0x03)
        y = (data[2] << 10) | (data[3] << 2) | ((data[6] >> 4) & 0x03)
        z = (data[4] << 10) | (data[5] << 2) | ((data[6] >> 2) & 0x03)
        # Center at zero (18-bit unsigned, midpoint = 131072)
        x -= 131072
        y -= 131072
        z -= 131072
        return x, y, z

    def _read_and_publish(self):
        now = self.get_clock().now().to_msg()

        try:
            ax, ay, az = self._read_6_bytes(
                ISM330DHCX_ADDR, ISM330DHCX_OUTX_L_A)
            gx, gy, gz = self._read_6_bytes(
                ISM330DHCX_ADDR, ISM330DHCX_OUTX_L_G)
        except OSError as e:
            self.get_logger().warn(
                f'I2C read error: {e}', throttle_duration_sec=5)
            return

        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = self.frame_id

        imu_msg.linear_acceleration.x = ax * ACCEL_SCALE_4G
        imu_msg.linear_acceleration.y = ay * ACCEL_SCALE_4G
        imu_msg.linear_acceleration.z = az * ACCEL_SCALE_4G

        imu_msg.angular_velocity.x = gx * GYRO_SCALE_500DPS * DEG_TO_RAD
        imu_msg.angular_velocity.y = gy * GYRO_SCALE_500DPS * DEG_TO_RAD
        imu_msg.angular_velocity.z = gz * GYRO_SCALE_500DPS * DEG_TO_RAD

        # No orientation estimate from raw data
        imu_msg.orientation_covariance[0] = -1.0

        self.imu_pub.publish(imu_msg)

        if self.mag_available:
            try:
                mx, my, mz = self._read_mag()
                mag_msg = MagneticField()
                mag_msg.header.stamp = now
                mag_msg.header.frame_id = self.frame_id
                mag_msg.magnetic_field.x = mx * MAG_SCALE * GAUSS_TO_TESLA
                mag_msg.magnetic_field.y = my * MAG_SCALE * GAUSS_TO_TESLA
                mag_msg.magnetic_field.z = mz * MAG_SCALE * GAUSS_TO_TESLA
                self.mag_pub.publish(mag_msg)
            except OSError:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
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
