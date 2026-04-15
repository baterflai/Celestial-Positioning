from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port_arg = DeclareLaunchArgument(
        'web_port', default_value='8082',
        description='HTTP port for the magnetometer calibration web UI',
    )
    min_samples_arg = DeclareLaunchArgument(
        'min_samples', default_value='200',
        description='Minimum samples required before calibration',
    )
    save_path_arg = DeclareLaunchArgument(
        'save_path', default_value='~/imu_calibration.yaml',
        description='Path to save calibration YAML',
    )

    imu_node = Node(
        package='celestial_positioning',
        executable='imu_node',
        name='imu_node',
        output='screen',
    )

    calibrator_node = Node(
        package='celestial_positioning',
        executable='mag_calibrator_node',
        name='mag_calibrator_node',
        parameters=[{
            'web_port': LaunchConfiguration('web_port'),
            'min_samples': LaunchConfiguration('min_samples'),
            'save_path': LaunchConfiguration('save_path'),
        }],
        output='screen',
    )

    return LaunchDescription([
        port_arg,
        min_samples_arg,
        save_path_arg,
        imu_node,
        calibrator_node,
    ])
