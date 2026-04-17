import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    """Bag-only recorder. Subscribes to topics published by the
    celestial_sensors and celestial_gps services which run at boot."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.expanduser(f'~/logs/{timestamp}')

    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '--storage', 'mcap',
            '--compression-mode', 'file',
            '--compression-format', 'zstd',
            '--output', log_dir,
            '/imu/data_raw',
            '/imu/mag',
            '/gps/fix',
            '/camera/image_raw',
        ],
        output='screen',
    )

    return LaunchDescription([bag_record])
