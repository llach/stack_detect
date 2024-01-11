"""Setup script for ROS package"""
from setuptools import setup

PACKAGE_NAME = "py_pubsub"

setup(
    name=PACKAGE_NAME,
    version="0.0.0",
    packages=[PACKAGE_NAME],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + PACKAGE_NAME]),
        ("share/" + PACKAGE_NAME, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="External projects",
    maintainer_email="external-projects-XD@ocado.com",
    description="Sample ROS app",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "talker = py_pubsub.publisher_member_function:main",
            "listener = py_pubsub.subscriber_member_function:main",
        ],
    },
)
