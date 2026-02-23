import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'celestial_positioning'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*_launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='developer',
    maintainer_email='todo@todo.com',
    description='Visual-based celestial positioning system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'feature_extractor_node = celestial_positioning.feature_extractor:main',
        ],
    },
)
