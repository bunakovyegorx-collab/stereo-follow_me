from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

from drone_stereo_bringup.calibration import scale_calibration
from drone_stereo_bringup.profiles import get_camera_profile
from drone_stereo_bringup.profiles import load_named_stereo_profile


CALIBRATION_DIRECTORY = Path.home() / '.ros/camera_info'
SOURCE_CALIBRATIONS = {
    'left': CALIBRATION_DIRECTORY / (
        'imx219__base_axi_pcie_1000120000_rp1_i2c_88000_imx219_10_'
        '1640x1232_1640x1232_RGGB_PISP_COMP1_RAW.yaml'
    ),
    'right': CALIBRATION_DIRECTORY / (
        'imx219__base_axi_pcie_1000120000_rp1_i2c_80000_imx219_10_'
        '1640x1232_1640x1232_RGGB_PISP_COMP1_RAW.yaml'
    ),
}
SCALED_CALIBRATIONS = {
    side: CALIBRATION_DIRECTORY / f'stereo_{side}_maximum_sgbm.yaml'
    for side in ('left', 'right')
}
LIVE_CALIBRATIONS = {
    'left': CALIBRATION_DIRECTORY / (
        'imx219__base_axi_pcie_1000120000_rp1_i2c_88000_imx219_10_'
        '640x480_1640x1232_RGGB_PISP_COMP1_RAW.yaml'
    ),
    'right': CALIBRATION_DIRECTORY / (
        'imx219__base_axi_pcie_1000120000_rp1_i2c_80000_imx219_10_'
        '640x480_1640x1232_RGGB_PISP_COMP1_RAW.yaml'
    ),
}
SCALED_LIVE_CALIBRATIONS = {
    side: CALIBRATION_DIRECTORY / f'stereo_{side}_live_320x240.yaml'
    for side in ('left', 'right')
}


def prepare_calibration(side):
    source_path = SOURCE_CALIBRATIONS[side]
    if not source_path.is_file():
        raise FileNotFoundError(
            f'missing source calibration for {side} camera: {source_path}'
        )
    source = yaml.safe_load(source_path.read_text())
    scaled = scale_calibration(
        source,
        width=3280,
        height=2464,
        camera_name=source['camera_name'].replace(
            '1640x1232_1640x1232', '3280x2464_3280x2464',
        ),
    )
    target_path = SCALED_CALIBRATIONS[side]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(scaled, sort_keys=False))
    return target_path


def prepare_live_calibration(side):
    source_path = LIVE_CALIBRATIONS[side]
    if not source_path.is_file():
        raise FileNotFoundError(
            f'missing 640x480 source calibration for {side}: {source_path}'
        )
    source = yaml.safe_load(source_path.read_text())
    scaled = scale_calibration(
        source,
        width=320,
        height=240,
        camera_name=source['camera_name'].replace(
            '640x480_1640x1232', '320x240_1640x1232',
        ),
    )
    target_path = SCALED_LIVE_CALIBRATIONS[side]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(scaled, sort_keys=False))
    return target_path


def profile_calibrations(profile_name):
    if profile_name == 'maximum_sgbm':
        return {
            side: prepare_calibration(side) for side in ('left', 'right')
        }
    if profile_name in ('live_10fps', 'foxglove_live', 'person_range'):
        return {
            side: prepare_live_calibration(side)
            for side in ('left', 'right')
        }
    raise ValueError(f'unsupported SGBM launch profile: {profile_name}')


def camera_node(side, camera_index, calibration_url, profile):
    parameters = dict(profile)
    parameters.update({
        'camera': camera_index,
        'frame_id': f'stereo_{side}_optical_frame',
        'camera_info_url': calibration_url,
    })
    return Node(
        package='camera_ros',
        executable='camera_node',
        name='camera_node',
        namespace=f'stereo/{side}',
        parameters=[parameters],
        remappings=[
            ('~/image_raw', 'image_raw'),
            ('~/image_raw/compressed', 'image_raw/compressed'),
            ('~/camera_info', 'camera_info'),
        ],
        prefix=['nice -n 10'],
        output='screen',
    )


def mono_components(side):
    namespace = f'stereo/{side}'
    return [
        ComposableNode(
            package='image_proc',
            plugin='image_proc::DebayerNode',
            name='debayer_node',
            namespace=namespace,
            extra_arguments=[{'use_intra_process_comms': True}],
        ),
        ComposableNode(
            package='image_proc',
            plugin='image_proc::RectifyNode',
            name='rectify_mono_node',
            namespace=namespace,
            remappings=[
                ('image', 'image_mono'),
                ('image_rect', 'image_rect'),
            ],
            extra_arguments=[{'use_intra_process_comms': True}],
        ),
    ]


def launch_setup(context):
    profile_name = LaunchConfiguration('profile').perform(context)
    profile = get_camera_profile(profile_name)
    calibrations = profile_calibrations(profile_name)
    expected_size = (int(profile['width']), int(profile['height']))
    stereo_profile = load_named_stereo_profile(
        profile_name, calibrations['right'], expected_size=expected_size,
    )

    cameras = [
        camera_node('left', 0, calibrations['left'].as_uri(), profile),
        camera_node('right', 1, calibrations['right'].as_uri(), profile),
    ]
    components = [
        *mono_components('left'),
        *mono_components('right'),
        ComposableNode(
            package='stereo_image_proc',
            plugin='stereo_image_proc::DisparityNode',
            name='disparity_node',
            namespace='stereo',
            parameters=[{
                'approximate_sync': True,
                **{
                    name: value for name, value in stereo_profile.items()
                    if name not in ('use_color', 'avoid_point_cloud_padding')
                },
            }],
            remappings=[
                ('left/image_rect', 'left/image_rect'),
                ('left/camera_info', 'left/camera_info'),
                ('right/image_rect', 'right/image_rect'),
                ('right/camera_info', 'right/camera_info'),
            ],
            extra_arguments=[{'use_intra_process_comms': True}],
        ),
    ]

    combined_mode = LaunchConfiguration('combined_mode').perform(
        context
    ).lower() in ('1', 'true', 'yes', 'on')
    container_prefix = (
        ['nice -n 10 taskset -c 0-2']
        if profile_name == 'maximum_sgbm'
        else (
            ['nice -n 0 taskset -c 0-2']
            if combined_mode
            else ['nice -n 0 taskset -c 0-3']
        )
    )
    container = ComposableNodeContainer(
        package='rclcpp_components',
        executable='component_container_mt',
        name='stereo_sgbm_light_container',
        namespace='',
        composable_node_descriptions=components,
        prefix=container_prefix,
        output='screen',
    )
    disparity_viz = Node(
        package='drone_stereo_bringup',
        executable='disparity_viz',
        name='disparity_viz',
        output='screen',
        prefix=['nice -n 10 taskset -c 3'],
        condition=IfCondition(LaunchConfiguration('disparity_viz')),
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        additional_env={
            # RViz працює через XWayland на Pi; software GL усуває короткі
            # чорні кадри V3D/XWayland, не змінюючи обробку SGBM.
            'LIBGL_ALWAYS_SOFTWARE': '1',
        },
        arguments=[
            '-d',
            [FindPackageShare('drone_stereo_bringup'),
             '/rviz/disparity_light.rviz'],
        ],
        condition=IfCondition(LaunchConfiguration('rviz')),
        output='screen',
    )
    actions = [
        *cameras,
        container,
    ]
    if profile_name == 'foxglove_live':
        actions.append(Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{
                'address': '0.0.0.0',
                'port': 8765,
                # Один кадр disparity має близько 307 kB. Глибина QoS 1
                # не накопичує застарілі кадри, якщо Mac або мережа відстає.
                'min_qos_depth': 1,
                'max_qos_depth': 1,
                'num_threads': 2,
                # При проблемах мережі скидати старі повідомлення приблизно
                # через 0.1 с потоку, а не показувати їх із довгою затримкою.
                'send_buffer_limit': 1000000,
                'use_compression': False,
                'best_effort_qos_topic_whitelist': [
                    '^/stereo/disparity$',
                    '^/stereo/left/image_rect$',
                ],
                # Не виставляти назовні усі сирі camera topics. Foxglove
                # отримує disparity, калібрування, лівий rectified image для
                # забарвлення хмари, TF і діагностику Raspberry Pi.
                'topic_whitelist': [
                    '^/stereo/disparity$',
                    '^/stereo/left/camera_info$',
                    '^/stereo/left/image_rect$',
                    '^/tf$',
                    '^/tf_static$',
                    '^/rosout$',
                    '^/foxglove_bridge/client_count$',
                ],
                'service_whitelist': ['(?!)'],
                'param_whitelist': ['(?!)'],
                'client_topic_whitelist': ['(?!)'],
                'capabilities': ['connectionGraph'],
                'publish_client_count': True,
            }],
            output='screen',
        ))
    else:
        actions.extend([
            disparity_viz,
            TimerAction(period=3.0, actions=[rviz]),
        ])
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'profile',
            default_value='live_10fps',
            choices=[
                'live_10fps', 'foxglove_live', 'person_range',
                'maximum_sgbm',
            ],
            description='Camera and SGBM performance profile.',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='True',
            description='Start RViz with the disparity image display.',
        ),
        DeclareLaunchArgument(
            'disparity_viz',
            default_value='True',
            description='Publish the mono8 disparity visualization.',
        ),
        DeclareLaunchArgument(
            'combined_mode',
            default_value='False',
            description='Reserve CPU core 3 for detector/fusion nodes.',
        ),
        OpaqueFunction(function=launch_setup),
    ])
