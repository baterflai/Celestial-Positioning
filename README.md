# Celestial Positioning

Visual-based celestial positioning system running on Raspberry Pi 4 with an IMX477 camera and SparkFun 9DoF ISM330DHCX IMU.

## Hardware

- **Raspberry Pi 4**
- **IMX477 HQ Camera** — connected via CSI ribbon cable
- **SparkFun 9DoF ISM330DHCX IMU** — connected via I2C1 (SDA → Pin 3, SCL → Pin 5, 3V3 → Pin 1, GND → Pin 6)

## Nodes

### `camera_node` (from `camera_ros`)

Publishes raw camera frames from the IMX477.

| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 1280x720 RGB888 frames |

### `camera_calibrator_node`

Web-based camera calibration tool. Detects a checkerboard pattern (7x9 inner corners) and computes intrinsic camera parameters. Serves a UI at `http://<pi-ip>:8080`.

| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Subscribed — raw camera input |

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `web_port` | `8080` | HTTP port for calibration UI |
| `pattern_columns` | `7` | Inner corners horizontally |
| `pattern_rows` | `9` | Inner corners vertically |
| `square_size` | `0.020` | Square size in meters |
| `detection_scale` | `0.5` | Downscale factor for detection |
| `save_path` | `~/camera_calibration.yaml` | Output calibration file path |

### `imu_node`

Reads the ISM330DHCX (accelerometer + gyroscope) and MMC5983MA (magnetometer) over I2C and publishes raw sensor data.

| Topic | Type | Rate | Description |
|---|---|---|---|
| `/imu/data_raw` | `sensor_msgs/Imu` | 100 Hz | Accelerometer + gyroscope |
| `/imu/mag` | `sensor_msgs/MagneticField` | 100 Hz | Magnetometer |

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `i2c_bus` | `1` | I2C bus number |
| `rate_hz` | `100.0` | Publishing rate in Hz |
| `frame_id` | `imu_link` | TF frame ID for messages |

### `feature_extractor_node`

Extracts visual features from camera frames for celestial positioning.

| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Subscribed — raw camera input |

## Topic Message Formats

### `sensor_msgs/Imu` (`/imu/data_raw`)

```yaml
header:
  stamp:
    sec: <uint32>
    nanosec: <uint32>
  frame_id: "imu_link"
orientation:                    # Not estimated (covariance[0] = -1)
  x: 0.0
  y: 0.0
  z: 0.0
  w: 1.0
orientation_covariance: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
angular_velocity:               # rad/s, ±500 dps range
  x: <float64>
  y: <float64>
  z: <float64>
angular_velocity_covariance: [0.0, ...]
linear_acceleration:            # m/s², ±4g range
  x: <float64>
  y: <float64>
  z: <float64>
linear_acceleration_covariance: [0.0, ...]
```

### `sensor_msgs/MagneticField` (`/imu/mag`)

```yaml
header:
  stamp:
    sec: <uint32>
    nanosec: <uint32>
  frame_id: "imu_link"
magnetic_field:                 # Tesla
  x: <float64>
  y: <float64>
  z: <float64>
magnetic_field_covariance: [0.0, ...]
```

### `sensor_msgs/Image` (`/camera/image_raw`)

```yaml
header:
  stamp:
    sec: <uint32>
    nanosec: <uint32>
  frame_id: "camera"
height: 720
width: 1280
encoding: "rgb8"
is_bigendian: 0
step: 3840                      # width * 3 channels
data: <uint8[]>                 # 2764800 bytes
```

## Launch Files

### `calibration_launch.py`

Launches camera and calibrator nodes for camera intrinsic calibration.

```bash
ros2 launch celestial_positioning calibration_launch.py
```

### `system_launch.py`

Main system launch file.

```bash
ros2 launch celestial_positioning system_launch.py
```

## Build

```bash
cd ~/Celestial-Positioning
colcon build --packages-select celestial_positioning
source install/setup.bash
```

## Running Individual Nodes

### IMU Node

```bash
ros2 run celestial_positioning imu_node
```

With custom parameters:

```bash
ros2 run celestial_positioning imu_node --ros-args -p rate_hz:=50.0 -p i2c_bus:=1
```

## Echoing Topics

```bash
# IMU accelerometer + gyroscope
ros2 topic echo /imu/data_raw

# Magnetometer
ros2 topic echo /imu/mag

# Camera frames
ros2 topic echo /camera/image_raw

# List all active topics
ros2 topic list

# Check publish rate
ros2 topic hz /imu/data_raw
```
