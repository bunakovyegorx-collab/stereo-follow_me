"""Standalone left-camera YOLO person/car test for a Raspberry Pi."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    model_default = PathJoinSubstitution([
        EnvironmentVariable('HOME'), '.local', 'share', 'camera-yolo', 'models',
        'yolo26n_ncnn_model',
    ])
    python_default = PathJoinSubstitution([
        EnvironmentVariable('HOME'), '.venvs', 'camera-yolo', 'bin', 'python',
    ])
    camera_index = LaunchConfiguration('camera_index')
    model_path = LaunchConfiguration('model_path')
    python_executable = LaunchConfiguration('python_executable')
    return LaunchDescription([
        DeclareLaunchArgument('camera_index', default_value='0'),
        DeclareLaunchArgument('model_path', default_value=model_default),
        DeclareLaunchArgument('python_executable', default_value=python_default),
        DeclareLaunchArgument('confidence_threshold', default_value='0.40'),
        DeclareLaunchArgument('imgsz', default_value='320'),
        Node(
            package='camera_ros',
            executable='camera_node',
            name='left_camera',
            namespace='yolo_test',
            parameters=[{
                'camera': camera_index,
                'sensor_mode': '1640:1232',
                'width': 640,
                'height': 480,
                'format': 'YUYV',
                'orientation': 180,
                'FrameDurationLimits': [100000, 100000],
                'AeEnable': True,
                'frame_id': 'yolo_left_optical_frame',
            }],
            remappings=[
                ('~/image_raw', 'image_raw'),
                ('~/image_raw/compressed', 'image_raw/compressed'),
                ('~/camera_info', 'camera_info'),
            ],
            output='screen',
        ),
        Node(
            package='yolo_person_car',
            executable='detector',
            name='detector',
            namespace='yolo_test',
            prefix=[python_executable],
            parameters=[{
                'image_topic': 'image_raw',
                'model_path': model_path,
                'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                'imgsz': LaunchConfiguration('imgsz'),
                'allowed_classes': [0, 2],
                'enable_debug_image': True,
            }],
            output='screen',
        ),
    ])
