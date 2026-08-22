"""Standalone left-camera Hailo-8 YOLO person/car test for a Raspberry Pi."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    hef_default = PathJoinSubstitution([
        '/usr', 'share', 'hailo-models', 'yolov8s_h8.hef',
    ])
    python_default = PathJoinSubstitution([
        EnvironmentVariable('HOME'), '.venvs', 'camera-yolo', 'bin', 'python',
    ])
    camera_index = LaunchConfiguration('camera_index')
    hef_path = LaunchConfiguration('hef_path')
    python_executable = LaunchConfiguration('python_executable')
    return LaunchDescription([
        DeclareLaunchArgument('camera_index', default_value='0'),
        DeclareLaunchArgument('hef_path', default_value=hef_default),
        DeclareLaunchArgument('python_executable', default_value=python_default),
        DeclareLaunchArgument('confidence_threshold', default_value='0.40'),
        Node(
            package='camera_ros',
            executable='camera_node',
            name='left_camera',
            namespace='hailo_yolo_test',
            parameters=[{
                'camera': camera_index,
                'sensor_mode': '1640:1232',
                'width': 640,
                'height': 480,
                'format': 'YUYV',
                'orientation': 180,
                'FrameDurationLimits': [100000, 100000],
                'AeEnable': True,
                'frame_id': 'hailo_yolo_left_optical_frame',
            }],
            remappings=[
                ('~/image_raw', 'image_raw'),
                ('~/image_raw/compressed', 'image_raw/compressed'),
                ('~/camera_info', 'camera_info'),
            ],
            output='screen',
        ),
        Node(
            package='hailo_yolo_detector',
            executable='hailo_detector',
            name='detector',
            namespace='hailo_yolo_test',
            prefix=[python_executable],
            parameters=[{
                'image_topic': 'image_raw',
                'hef_path': hef_path,
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'allowed_classes': [0, 2],
                'enable_debug_image': True,
            }],
            output='screen',
        ),
    ])
