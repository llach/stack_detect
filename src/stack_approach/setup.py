"""Setup script for ROS package"""
from setuptools import setup
import os
from glob import glob

PACKAGE_NAME = "stack_approach"

setup(
    name=PACKAGE_NAME,
    version="0.0.1",
    packages=[PACKAGE_NAME],
    data_files=[
        ("share/" + PACKAGE_NAME, ["package.xml"]),
        (os.path.join('share', PACKAGE_NAME, 'launch'), glob('launch/*.py')),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Luca Lach",
    maintainer_email="luca.michael.lach@upc.edu",
    description="Movements for unstacking",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        'console_scripts': [
            '2d_table_grasp = stack_approach.2d_table_grasp:main',
            '3d_shelf_grasp = stack_approach.3d_shelf_grasp:main',
            'dino_approach = stack_approach.dino_approach:main',
            'papp = stack_approach.papp:main',
            'primitive = stack_approach.primitive:main',
            'collect_cloud_action = stack_approach.collect_cloud_action:main',
            'record_cloud = stack_approach.record_cloud:main',
            'move_arm_service = stack_approach.move_arm_service:main',
            'cloud_pose_vary = stack_approach.cloud_pose_vary:main',
            'gripper = stack_approach.gripper_service:main',
            'blue_gripper = stack_approach.blue_gripper_service:main',
            'roller_gripper = stack_approach.roller_gripper_service_v2:main',
            'bag_opening = stack_approach.bag_opening:main',
            'bag_opening_perc = stack_approach.bag_opening_perc:main',
            'primitive_test = stack_approach.primitive_test:main',
            'ft_publisher = stack_approach.ft_publisher:main',
            'get_js = stack_approach.get_js:main',
        ],
    },
)
