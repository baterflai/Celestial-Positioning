import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.expanduser(f'~/logs/{timestamp}')

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
            'calibration_file': LaunchConfiguration('calibration_file'),
        }],
        output='screen',
    )

    gps_node = Node(
        package='celestial_positioning',
        executable='gps_node',
        name='gps_node',
        parameters=[{
            'i2c_bus': 1,
            'rate_hz': 1.0,
            'frame_id': 'gps_link',
        }],
        output='screen',
    )

    camera_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='camera',
        parameters=[{
            'width': 1456,
            'height': 1088,
        }],
        output='screen',
    )

    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--storage', 'mcap',
            '--output', log_dir,
            '/imu/data_raw',
            '/imu/mag',
            '/gps/fix',
            '/camera/image_raw',
        ],
        output='screen',
    )

    return LaunchDescription([
        cal_file_arg,
        imu_node,
        gps_node,
        camera_node,
        bag_record,
    ])
