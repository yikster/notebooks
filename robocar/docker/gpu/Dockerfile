FROM nvidia/cuda:9.0-runtime-ubuntu16.04

MAINTAINER Amazon AI

ENV DEBIAN_FRONTEND noninteractive
# packaging dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        dh-make \
        fakeroot \
        cuda-command-line-tools-9-0 \
        cuda-cublas-dev-9-0 \
        cuda-cudart-dev-9-0 \
        cuda-cufft-dev-9-0 \
        cuda-curand-dev-9-0 \
        cuda-cusolver-dev-9-0 \
        cuda-cusparse-dev-9-0 \        
        libcudnn7=7.1.4.18-1+cuda9.0 \
        libcudnn7-dev=7.1.4.18-1+cuda9.0 \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        build-essential \
        nvidia-modprobe \
        python3-virtualenv \
        python3-wheel \
        curl \
        git \
        devscripts \
        python3 \
        python3-pip \        
        python3-setuptools \
        lsb-release && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip3 --no-cache-dir install -U numpy

ARG TENSORFLOW_VERSION=1.8.0
ARG TENSORFLOW_DEVICE=gpu
ARG TENSORFLOW_APPEND=_gpu
RUN pip3 --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow${TENSORFLOW_APPEND}-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl

ARG KERAS_VERSION=2.1.5
ENV KERAS_BACKEND=tensorflow
RUN pip3 --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION} && pip3 install pandas Pillow opencv-python boto3

# quick test and dump package lists
RUN python3 -c "import tensorflow; print(tensorflow.__version__)" \
 && dpkg-query -l > /dpkg-query-l.txt \
 && pip3 freeze > /pip3-freeze.txt


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY app /opt/program
WORKDIR /opt/program


