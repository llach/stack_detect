"""Setup script for ROS package"""
from setuptools import setup

PACKAGE_NAME = "stack_detect"

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
    description="Detect uppermost item on a stack",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        'console_scripts': [
            'cloud_normals = stack_detect.cloud_normals:main',
            'towel = stack_detect.towel_detector:main',
            'sam = stack_detect.sam_detect:main',
            'dino = stack_detect.dino_detect:main',
            'sam2_dino_srv = stack_detect.sam2_dino_srv:main',
            'sam_gpe = stack_detect.sam_gpe:main',
            'collect_classify = stack_detect.collect_classify:main',
        ],
    },
)
