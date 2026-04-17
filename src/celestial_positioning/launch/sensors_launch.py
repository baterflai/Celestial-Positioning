import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Always-on sensor publishers: IMU, camera, exposure timestamps.

    GPS is run separately by celestial_gps.service because it also feeds
    chrony via SHM and must be up regardless of whether sensors are publishing.
    """

    cal_file_arg = DeclareLaunchArgument(
        'calibration_file',
        default_value=os.path.expanduser(
            '~/ros2_ws/src/celestial_positioning/config/imu_calibration.yaml'),
        description='Path to IMU calibration YAML file',
    )

    imu_node = Node(
        package='celestial_positioning',
        executable='imu_node',
        name='imu_node',
        parameters=[{
            'i2c_bus': 1,
            'rate_hz': 100.0,
            'frame_id': 'imu_link',
        }],
        output='screen',
    )

    exposure_ts_node = Node(
        package='celestial_positioning',
        executable='exposure_timestamp_node',
        name='exposure_timestamp_node',
        parameters=[{
            'gpio_chip': 'gpiochip0',
            'strobe_pin': 17,
            'frame_id': 'camera_strobe',
        }],
        output='screen',
    )

    camera_node = Node(
        package='celestial_positioning',
        executable='raw_camera_node',
        name='camera',
        parameters=[{
            'width': 1456,
            'height': 1088,
            'frame_id': 'camera',
            'auto_exposure': True,
            'target_mean': 400.0,
            'exposure_s': 0.033,
            'gain': 0,
        }],
        output='screen',
    )

    return LaunchDescription([
        imu_node,
        exposure_ts_node,
        camera_node,
    ])
