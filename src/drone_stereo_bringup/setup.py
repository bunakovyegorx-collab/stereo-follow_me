from glob import glob
from setuptools import setup

package_name = 'drone_stereo_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'disparity_viz = '
            'drone_stereo_bringup.disparity_viz_node:main',
            'disparity_jpeg = '
            'drone_stereo_bringup.disparity_jpeg_node:main',
            'stereo_point_cloud_preview = '
            'drone_stereo_bringup.point_cloud_preview_node:main',
        ],
    },
    zip_safe=True,
    maintainer='qqqq',
    maintainer_email='qqqq@example.com',
    description='Bringup launch files for the dual IMX219 stereo camera pipeline.',
    license='MIT',
)
