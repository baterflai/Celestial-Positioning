from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    width_arg = DeclareLaunchArgument(
        'width', default_value='1280',
        description='Capture width in pixels',
    )
    height_arg = DeclareLaunchArgument(
        'height', default_value='720',
        description='Capture height in pixels',
    )
    port_arg = DeclareLaunchArgument(
        'web_port', default_value='8080',
        description='HTTP port for the calibration web UI',
    )
    pattern_cols_arg = DeclareLaunchArgument(
        'pattern_columns', default_value='7',
        description='Inner corner count horizontally (squares - 1)',
    )
    pattern_rows_arg = DeclareLaunchArgument(
        'pattern_rows', default_value='9',
        description='Inner corner count vertically (squares - 1)',
    )
    square_size_arg = DeclareLaunchArgument(
        'square_size', default_value='0.020',
        description='Checkerboard square size in meters',
    )
    detection_scale_arg = DeclareLaunchArgument(
        'detection_scale', default_value='0.5',
        description='Scale factor for corner detection (0.25-1.0)',
    )

    camera_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='camera',
        parameters=[{
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
            'format': 'RGB888',
        }],
        output='screen',
    )

    calibrator_node = Node(
        package='celestial_positioning',
        executable='camera_calibrator_node',
        name='camera_calibrator_node',
        parameters=[{
            'web_port': LaunchConfiguration('web_port'),
            'pattern_columns': LaunchConfiguration('pattern_columns'),
            'pattern_rows': LaunchConfiguration('pattern_rows'),
            'square_size': LaunchConfiguration('square_size'),
            'detection_scale': LaunchConfiguration('detection_scale'),
        }],
        output='screen',
    )

    return LaunchDescription([
        width_arg,
        height_arg,
        port_arg,
        pattern_cols_arg,
        pattern_rows_arg,
        square_size_arg,
        detection_scale_arg,
        camera_node,
        calibrator_node,
    ])
