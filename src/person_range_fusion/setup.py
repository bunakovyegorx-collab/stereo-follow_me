from glob import glob

from setuptools import setup


package_name = 'person_range_fusion'

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
    description='Fuse YOLO person detections with stereo disparity.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'fusion = person_range_fusion.fusion_node:main',
            'pi_diagnostics = person_range_fusion.diagnostics_node:main',
            'depth = person_range_fusion.depth_node:main',
        ],
    },
)
