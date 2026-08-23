from glob import glob

from setuptools import setup


package_name = 'target_tracker'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qqqq',
    maintainer_email='qqqq@example.com',
    description='Geometric obstacle detectors (voxel clustering + U-depth).',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detectors = target_tracker.detectors_node:main',
        ],
    },
)
