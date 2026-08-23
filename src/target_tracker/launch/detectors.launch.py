"""Run only the geometric detectors, on top of an already-running pipeline.

Deliberately standalone: start person_range.launch.py first, then this, so the
detectors can be restarted and re-tuned without disturbing the stereo/YOLO
stack that feeds them.

    ros2 launch target_tracker detectors.launch.py
    ros2 launch target_tracker detectors.launch.py enable_udepth:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    # Substitutions resolve to strings, so the value_type is spelled out --
    # the node declares these as bool/int and would reject a bare 'true'.
    enable_cluster = ParameterValue(
        LaunchConfiguration('enable_cluster'), value_type=bool,
    )
    enable_udepth = ParameterValue(
        LaunchConfiguration('enable_udepth'), value_type=bool,
    )
    stride = ParameterValue(LaunchConfiguration('stride'), value_type=int)

    return LaunchDescription([
        DeclareLaunchArgument('enable_cluster', default_value='true'),
        DeclareLaunchArgument('enable_udepth', default_value='true'),
        DeclareLaunchArgument('stride', default_value='3'),

        Node(
            package='target_tracker',
            executable='detectors',
            name='detectors',
            namespace='detectors',
            # Core 3 already carries YOLO and fusion, and cores 0-2 carry SGBM
            # and depth. Core 2 is the least contended of those, and nice 10
            # keeps this experiment from ever starving the live pipeline.
            prefix=['nice -n 10 taskset -c 2 '],
            parameters=[{
                'depth_topic': '/stereo/depth',
                'camera_info_topic': '/stereo/left/camera_info',
                'output_frame': 'stereo_left_up',
                'optical_frame': 'stereo_left_optical_frame',
                'camera_height': 0.35,
                'enable_cluster': enable_cluster,
                'enable_udepth': enable_udepth,
                'stride': stride,
                'z_min': 1.2,
                'z_max': 5.5,
                'voxel_resolution': 0.08,
                'min_points_per_voxel': 4,
                'ground_z': 0.15,
                'min_cluster_points': 25,
                'min_cluster_voxels': 4,
                'udepth_bins': 64,
                'udepth_col_scale': 0.5,
                'udepth_min_count': 6,
                'udepth_min_run': 8,
                'publish_umap_jpeg': True,
                'umap_jpeg_fps': 2.0,
                'umap_jpeg_quality': 80,
            }],
            output='screen',
        ),
    ])
