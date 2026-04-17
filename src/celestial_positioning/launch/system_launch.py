import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    width_arg = DeclareLaunchArgument(
        'width',
        default_value='1920',
        description='Capture width in pixels',
    )
    height_arg = DeclareLaunchArgument(
        'height',
        default_value='1080',
        description='Capture height in pixels',
    )
    cal_file_arg = DeclareLaunchArgument(
        'calibration_file',
        default_value=os.path.expanduser('~/ros2_ws/src/celestial_positioning/config/imu_calibration.yaml'),
        description='Path to IMU calibration YAML file',
    )

    camera_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='camera',
        parameters=[{
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
            'format': 'BGR888',
        }],
        output='screen',
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

    feature_extractor_node = Node(
        package='celestial_positioning',
        executable='feature_extractor_node',
        name='feature_extractor_node',
        output='screen',
    )

    return LaunchDescription([
        width_arg,
        height_arg,
        camera_node,
        imu_node,
        feature_extractor_node,
    ])
