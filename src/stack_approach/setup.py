"""Setup script for ROS package"""
from setuptools import setup

PACKAGE_NAME = "stack_approach"

setup(
    name=PACKAGE_NAME,
    version="0.0.1",
    packages=[PACKAGE_NAME],
    data_files=[
        ("share/" + PACKAGE_NAME, ["package.xml"]),
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
        ],
    },
)
