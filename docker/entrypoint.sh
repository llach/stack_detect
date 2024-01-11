#!/bin/bash
# Basic entrypoint for ROS / Colcon Docker containers

# Source ROS 2
source /opt/ros/${ROS_DISTRO}/setup.bash
echo "Sourced ROS 2 ${ROS_DISTRO}"

# Source the overlay workspace, if built
if [ -f /ws/install/setup.bash ]
then
  source /ws/install/setup.bash
  echo "Sourced workspace"
fi

# Configure and run everything that you need here
export food=waffle
echo "Starting the image" 

# Execute the command passed into this entrypoint
exec "$@"
