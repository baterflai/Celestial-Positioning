from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='celestial_positioning',
            executable='gps_node',
            name='gps_node',
            parameters=[{
                'i2c_bus': 1,
                'rate_hz': 5.0,
                'frame_id': 'gps_link',
                'chrony_shm_unit': 0,
                'use_gps_time_in_header': True,
            }],
            output='screen',
        ),
    ])
