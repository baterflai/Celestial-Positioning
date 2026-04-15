from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    width_arg = DeclareLaunchArgument(
        'width', default_value='1456',
        description='Capture width in pixels',
    )
    height_arg = DeclareLaunchArgument(
        'height', default_value='1088',
        description='Capture height in pixels',
    )
    port_arg = DeclareLaunchArgument(
        'web_port', default_value='8080',
        description='HTTP port for the calibration web UI',
    )
    grid_cols_arg = DeclareLaunchArgument(
        'grid_cols', default_value='4',
        description='AprilGrid columns (number of tags)',
    )
    grid_rows_arg = DeclareLaunchArgument(
        'grid_rows', default_value='6',
        description='AprilGrid rows (number of tags)',
    )
    tag_size_arg = DeclareLaunchArgument(
        'tag_size', default_value='0.030',
        description='AprilTag size in meters',
    )
    tag_spacing_arg = DeclareLaunchArgument(
        'tag_spacing', default_value='0.009',
        description='Gap between tags in meters',
    )

    camera_node = Node(
        package='camera_ros',
        executable='camera_node',
        name='camera',
        parameters=[{
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
        }],
        output='screen',
    )

    calibrator_node = Node(
        package='celestial_positioning',
        executable='camera_calibrator_node',
        name='camera_calibrator_node',
        parameters=[{
            'web_port': LaunchConfiguration('web_port'),
            'grid_cols': LaunchConfiguration('grid_cols'),
            'grid_rows': LaunchConfiguration('grid_rows'),
            'tag_size': LaunchConfiguration('tag_size'),
            'tag_spacing': LaunchConfiguration('tag_spacing'),
        }],
        output='screen',
    )

    return LaunchDescription([
        width_arg,
        height_arg,
        port_arg,
        grid_cols_arg,
        grid_rows_arg,
        tag_size_arg,
        tag_spacing_arg,
        camera_node,
        calibrator_node,
    ])
