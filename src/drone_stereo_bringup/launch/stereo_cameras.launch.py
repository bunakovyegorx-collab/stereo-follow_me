from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from drone_stereo_bringup.profiles import get_camera_profile


def camera_node(profile, namespace, camera_index, frame_id):
    parameters = dict(profile)
    parameters.update(camera=camera_index, frame_id=frame_id)
    return Node(
        package='camera_ros',
        executable='camera_node',
        name='camera_node',
        namespace=namespace,
        parameters=[parameters],
        remappings=[
            ('~/image_raw', 'image_raw'),
            ('~/image_raw/compressed', 'image_raw/compressed'),
            ('~/camera_info', 'camera_info'),
        ],
    )


def launch_setup(context):
    profile = get_camera_profile(
        LaunchConfiguration('profile').perform(context)
    )
    return [
        camera_node(
            profile, 'stereo/left', 0, 'stereo_left_optical_frame'
        ),
        camera_node(
            profile, 'stereo/right', 1, 'stereo_right_optical_frame'
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'profile', default_value='quality',
            choices=[
                'quality', 'realtime', 'live_10fps',
                'theoretical_maximum', 'maximum_sgbm',
            ],
        ),
        OpaqueFunction(function=launch_setup),
    ])
