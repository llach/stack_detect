#!/bin/bash

REPOSITORY_NAME="$(basename "$(dirname -- "$( readlink -f -- "$0"; )")")"


DOCKER_VOLUMES="
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--volume="${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority" \
"

DOCKER_ARGS=${DOCKER_VOLUMES}

# Run the command
docker run -it --rm \
                --net=host \
                --ipc=host \
                --hostname="$(hostname)" \
                --privileged  \
                ${DOCKER_ARGS} \
                ${REPOSITORY_NAME} bash
