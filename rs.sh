#!/bin/bash

ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true config_file:=/home/ros/repos/iri_ur5e_description/src/iri_ur5e_description/config/l515.yaml
