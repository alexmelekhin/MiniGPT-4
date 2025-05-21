#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD"/"$1"
    fi
}

ARCH=`uname -m`
if [ $ARCH == "x86_64" ]; then
    if command -v nvidia-smi &> /dev/null; then
        DEVICE=cuda
        ARGS="--ipc host --gpus all"
    else
        echo "${orange}CPU-only${reset_color} build is not supported yet"
        exit 1
    fi
else
    echo "${orange}${ARCH}${reset_color} architecture is not supported"
    exit 1
fi

if [ ! -d $DATASETS_DIR ]; then
    echo "Error: DATASETS_DIR=$DATASETS_DIR is not an existing directory."
    exit 1
fi

PROJECT_ROOT_DIR=$(cd ./"`dirname $0`"/.. || exit; pwd)

echo "Running on ${orange}${ARCH}${reset_color} with ${orange}${DEVICE}${reset_color}"

docker run -it -d --rm \
    $ARGS \
    --privileged \
    --name ${USER}_minigpt4 \
    --net host \
    -v $PROJECT_ROOT_DIR:/home/docker_minigpt4/MiniGPT-4:rw \
    minigpt4

docker exec --user root \
    ${USER}_minigpt4 bash -c "\
        sed -i 's/^#*\s*Port\s\+22/Port 2224/' /etc/ssh/sshd_config && \
        sed -i 's/^#*\s*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
        sed -i 's/^#*\s*ChallengeResponseAuthentication.*/ChallengeResponseAuthentication yes/' /etc/ssh/sshd_config && \
        sed -i 's/^#*\s*UsePAM.*/UsePAM yes/' /etc/ssh/sshd_config && \
        mkdir -p /var/run/sshd && \
        /etc/init.d/ssh restart \
    "
