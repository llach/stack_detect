ARG ROS_DISTRO=humble

FROM ros:${ROS_DISTRO}-ros-base

ENV ROS_DISTRO=${ROS_DISTRO}
SHELL ["/bin/bash", "-c"]

WORKDIR /ws

# Dev container arguments
ARG USERNAME=ros
ARG UID=1000
ARG GID=${UID} 

# Create new user and home directory
# hadolint ignore=DL3004
RUN groupadd --gid $GID $USERNAME \
 && useradd --uid ${GID} --gid ${UID} --create-home ${USERNAME} \
 && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME} \
 && chown -R ${UID}:${GID} /home/${USERNAME}

 # Set the ownership of the overlay workspace to the new user
RUN chown -R ${UID}:${GID} /ws/

USER ${USERNAME}

COPY . /ws/

# Install ROS dependencies
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && rosdep update && rosdep install --from-paths src -y --ignore-src && colcon build --symlink-install

# Set up the entrypoint
COPY /docker/entrypoint.sh /
  
ENTRYPOINT [ "/entrypoint.sh" ]
