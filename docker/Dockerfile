#####
#
# Docker file to build container with Keras
#

FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

MAINTAINER Hidy Chiu, Wei-Yi Cheng

RUN apt-get update

RUN apt-get install -y build-essential cmake pkg-config wget zip unzip curl
RUN apt-get install -y python3.5-dev

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.5 get-pip.py && \
    rm get-pip.py

RUN apt-get install -y git-core

RUN pip3 install numpy keras pandas h5py versioneer && \
    pip3 install tensorflow-gpu

# Data science tools
RUN pip3 install jupyter scikit-learn matplotlib fancyimpute

# Jupyter cell theme
RUN pip3 install jupyter_contrib_nbextensions && \
    pip3 install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master && \
    jupyter contrib nbextension install

RUN mkdir -p $(jupyter --data-dir)/nbextensions && \
    cd $(jupyter --data-dir)/nbextensions && \
    mkdir jupyter_themes && cd jupyter_themes && \
    wget https://raw.githubusercontent.com/merqurio/jupyter_themes/master/theme_selector.js && \
    cd ../ && jupyter nbextension enable jupyter_themes/theme_selector

RUN rm -rf /root/.cache/pip/*

# install boto3
RUN pip3 install boto3


# install iceburger
RUN cd /root && git clone https://github.com/Hidysabc/iceburger.git && \
    cd iceburger && pip install -e .

# Clear apt downloads
RUN rm -rf /var/apt/lists/*

EXPOSE 6006

EXPOSE 8888

WORKDIR /root/iceburger/
