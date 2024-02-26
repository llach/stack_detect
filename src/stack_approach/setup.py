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
            'viewer = stack_approach.viewer:main',
            'cloud = stack_approach.cloud_grasp:main',
            'grasp = stack_approach.3d_grasp:main',
            'insert = stack_approach.insert:main',
        ],
    },
)
