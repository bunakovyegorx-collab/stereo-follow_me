from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.substitutions import FindPackageShare

from drone_stereo_bringup.profiles import get_camera_profile
from drone_stereo_bringup.profiles import load_named_stereo_profile
from drone_stereo_bringup.profiles import load_stereo_profile


CALIBRATION_DIRECTORY = Path.home() / '.ros/camera_info'
RIGHT_CALIBRATIONS = {
    'quality': CALIBRATION_DIRECTORY / (
        'imx219__base_axi_pcie_1000120000_rp1_i2c_80000_imx219_10_'
        '1640x1232_1640x1232_RGGB_PISP_COMP1_RAW.yaml'
    ),
    'realtime': CALIBRATION_DIRECTORY / (
        'imx219__base_axi_pcie_1000120000_rp1_i2c_80000_imx219_10_'
        '640x480_1640x1232_RGGB_PISP_COMP1_RAW.yaml'
    ),
    # Same 640x480 geometry as realtime; StereoBM params instead of SGBM.
    'realtime_bm': CALIBRATION_DIRECTORY / (
        'imx219__base_axi_pcie_1000120000_rp1_i2c_80000_imx219_10_'
        '640x480_1640x1232_RGGB_PISP_COMP1_RAW.yaml'
    ),
    'cpu_stereobm_baseline': CALIBRATION_DIRECTORY / (
        'imx219__base_axi_pcie_1000120000_rp1_i2c_80000_imx219_10_'
        '640x480_1640x1232_RGGB_PISP_COMP1_RAW.yaml'
    ),
}

# Camera geometry profile used for each stereo profile name.
CAMERA_PROFILE_FOR_STEREO = {
    'quality': 'quality',
    'realtime': 'realtime',
    'realtime_bm': 'realtime',
    'cpu_stereobm_baseline': 'realtime',
}


# Перетворює Python-значення параметра на рядок для передачі у вкладений
# launch-файл. Булеві значення повертає як 'True' або 'False', а всі інші
# значення перетворює стандартною функцією str().
def launch_value(value):
    if isinstance(value, bool):
        return 'True' if value else 'False'
    return str(value)


# Читає вибраний launch-профіль із поточного контексту, перевіряє відповідність
# роздільності калібрування, готує параметри stereo_image_proc, створює
# контейнер composable nodes і повертає список дій, які потрібно запустити.
def launch_setup(context):
    profile_name = LaunchConfiguration('profile').perform(context)
    camera_name = CAMERA_PROFILE_FOR_STEREO[profile_name]
    camera_profile = get_camera_profile(camera_name)
    expected_size = (
        int(camera_profile['width']), int(camera_profile['height'])
    )
    if profile_name in ('realtime_bm', 'cpu_stereobm_baseline'):
        profile = load_named_stereo_profile(
            profile_name,
            RIGHT_CALIBRATIONS[profile_name],
            expected_size=expected_size,
        )
    else:
        profile = load_stereo_profile(
            RIGHT_CALIBRATIONS[profile_name], expected_size=expected_size,
        )
    container_name = 'stereo_image_proc_container'
    arguments = {
        'namespace': 'stereo',
        'left_namespace': 'left',
        'right_namespace': 'right',
        'approximate_sync': 'True',
        'container': container_name,
    }
    arguments.update({
        name: launch_value(value)
        for name, value in profile.items()
    })
    container = ComposableNodeContainer(
        package='rclcpp_components',
        executable='component_container',
        name=container_name,
        namespace='',
        prefix=['nice -n 10 taskset -c 0-2'],
        output='screen',
    )
    return [
        container,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('stereo_image_proc'),
                    'launch',
                    'stereo_image_proc.launch.py',
                ])
            ),
            launch_arguments=arguments.items(),
        )
    ]


# Створює головний LaunchDescription: оголошує аргумент profile з варіантами
# quality/realtime та відкладає побудову решти пайплайну до launch_setup(),
# коли значення аргументу profile вже буде відоме.
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'profile', default_value='quality',
            choices=[
                'quality',
                'realtime',
                'realtime_bm',
                'cpu_stereobm_baseline',
            ],
        ),
        OpaqueFunction(function=launch_setup),
    ])
