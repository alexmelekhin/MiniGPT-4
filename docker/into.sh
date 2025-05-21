#!/bin/bash

docker exec --user docker_minigpt4 -it ${USER}_minigpt4 \
    /bin/bash -c "cd /home/docker_minigpt4; echo ${USER}_minigpt4 container; echo ; /bin/bash"
