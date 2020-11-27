# Based on mujoco-py's Dockerfile, but with the following changes:
# - Slightly changed nvidia stuff.
# - Uses Conda Python 3.7 instead of Python 3.6.
# The Conda bits are based on https://hub.docker.com/r/continuumio/miniconda3/dockerfile
FROM nvidia/cuda:10.1-cudnn8-runtime-ubuntu18.04

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6 \
    libegl1-mesa  \
    xvfb \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Conda and make it the default Python
ENV PATH /opt/conda/bin:$PATH
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/conda.sh || true \
  && bash /root/conda.sh -b -p /opt/conda || true \
  && rm /root/conda.sh
RUN conda update -n base -c defaults conda \
  && conda install -c anaconda python=3.6 \
  && conda clean -ay

# This is useful for making the X server work (but will break if the X server is
# not started on display :0)
ENV DISPLAY=:0
