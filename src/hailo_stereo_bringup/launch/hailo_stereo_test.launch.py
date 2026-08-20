"""Standalone hardware-validation launch for the Hailo StereoNet pilot.

Runs the existing CPU SGBM stereo pipeline (which owns the cameras) plus a
new Hailo-8 StereoNet node in parallel, both visualized side by side in
RViz. This intentionally does NOT touch the production
person_range.launch.py pipeline -- see docs/HAILO_OFFLOAD_PLAN.md and the
approved plan for the design rationale.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    hef_default = PathJoinSubstitution([
        EnvironmentVariable('HOME'), '.local', 'share', 'hailo-models', 'stereonet.hef',
    ])
    python_default = PathJoinSubstitution([
        EnvironmentVariable('HOME'), '.venvs', 'camera-yolo', 'bin', 'python',
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'profile', default_value='person_range',
            description='Camera/SGBM profile passed through to stereo_sgbm_light.launch.py.',
        ),
        DeclareLaunchArgument('hef_path', default_value=hef_default),
        DeclareLaunchArgument('python_executable', default_value=python_default),
        DeclareLaunchArgument(
            'display', default_value=':0',
            description='X11 DISPLAY for rviz2 (physical HDMI monitor on the Pi).',
        ),
        DeclareLaunchArgument(
            'launch_rviz', default_value='True',
            description='Start this package\'s own dual-view rviz2 (kept'
                        ' distinct from the included launch file\'s "rviz"'
                        ' argument to avoid a LaunchConfiguration name clash).',
        ),

        # Owns the cameras + CPU SGBM baseline + its own disparity_viz
        # instance (/stereo/disparity_viz). Do NOT also include
        # stereo_cameras.launch.py -- this file already starts both
        # camera_ros nodes internally, and a physical camera only supports
        # one streaming client at a time.
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('drone_stereo_bringup'), '/launch/stereo_sgbm_light.launch.py',
            ]),
            launch_arguments={
                'profile': LaunchConfiguration('profile'),
                'rviz': 'False',
                'disparity_viz': 'True',
                'combined_mode': 'True',
            }.items(),
        ),

        Node(
            package='hailo_stereo_bringup',
            executable='stereonet',
            name='stereonet',
            namespace='stereo',
            prefix=['nice -n 0 taskset -c 3 ', LaunchConfiguration('python_executable')],
            parameters=[{'hef_path': LaunchConfiguration('hef_path')}],
            output='screen',
        ),

        # Second disparity_viz instance for the Hailo topic. disparity_viz
        # subscribes/publishes hardcoded absolute topic names, so remap
        # those exact strings rather than relative names.
        Node(
            package='drone_stereo_bringup',
            executable='disparity_viz',
            name='disparity_viz_hailo',
            remappings=[
                ('/stereo/disparity', '/stereo/disparity_hailo'),
                ('/stereo/disparity_viz', '/stereo/disparity_hailo_viz'),
            ],
            prefix=['nice -n 10 taskset -c 3'],
            output='screen',
        ),

        TimerAction(period=3.0, actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            additional_env={
                'DISPLAY': LaunchConfiguration('display'),
                'LIBGL_ALWAYS_SOFTWARE': '1',
            },
            arguments=[
                '-d',
                [FindPackageShare('hailo_stereo_bringup'), '/rviz/disparity_dual.rviz'],
            ],
            condition=IfCondition(LaunchConfiguration('launch_rviz')),
            output='screen',
        )]),
    ])
