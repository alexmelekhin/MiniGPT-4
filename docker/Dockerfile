FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    git \
    build-essential \
    openssh-server \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# add user and his password
ENV USER=docker_minigpt4
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}

# create some directories for mounting volumes
RUN mkdir MiniGPT-4 && chown -R ${UID}:${GID} /home/${USER}

# generate SSH host keys and create run directory
RUN mkdir -p /var/run/sshd \
    && ssh-keygen -A

USER ${UID}:${GID}

ENV PATH=/opt/conda/bin:$PATH
