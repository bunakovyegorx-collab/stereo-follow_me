from glob import glob

from setuptools import setup


package_name = 'hailo_stereo_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qqqq',
    maintainer_email='qqqq@example.com',
    description='Hailo-8 StereoNet parallel depth pilot, for comparison against CPU SGBM.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'stereonet = hailo_stereo_bringup.stereonet_node:main',
        ],
    },
)
