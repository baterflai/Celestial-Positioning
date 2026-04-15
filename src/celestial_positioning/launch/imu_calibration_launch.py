from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port_arg = DeclareLaunchArgument(
        'web_port', default_value='8081',
        description='HTTP port for the IMU calibration web UI',
    )
    threshold_arg = DeclareLaunchArgument(
        'stationarity_threshold', default_value='0.005',
        description='Accel magnitude variance threshold for stationarity',
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
        executable='imu_calibrator_node',
        name='imu_calibrator_node',
        parameters=[{
            'web_port': LaunchConfiguration('web_port'),
            'stationarity_threshold': LaunchConfiguration(
                'stationarity_threshold'),
            'save_path': LaunchConfiguration('save_path'),
        }],
        output='screen',
    )

    return LaunchDescription([
        port_arg,
        threshold_arg,
        save_path_arg,
        imu_node,
        calibrator_node,
    ])
