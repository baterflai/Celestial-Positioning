from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='5.0',
        description='Camera capture and publish rate in Hz',
    )
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
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='0',
        description='V4L2 camera device index (/dev/videoN)',
    )

    camera_node = Node(
        package='celestial_positioning',
        executable='camera_node',
        name='camera_node',
        parameters=[{
            'publish_rate': LaunchConfiguration('publish_rate'),
            'width': LaunchConfiguration('width'),
            'height': LaunchConfiguration('height'),
            'device': LaunchConfiguration('device'),
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
        publish_rate_arg,
        width_arg,
        height_arg,
        device_arg,
        camera_node,
        feature_extractor_node,
    ])
