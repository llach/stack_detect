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
        "console_scripts": [
            "talker = py_pubsub.publisher_member_function:main",
            "listener = py_pubsub.subscriber_member_function:main",
        ],
    },
)
