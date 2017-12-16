#####
#
# Docker file to build container with Keras
#

FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04


MAINTAINER Hidy Chiu, Wei-Yi Cheng

RUN apt-get update

# Install opencv
RUN apt-get install -y build-essential cmake pkg-config wget zip unzip curl

RUN apt-get install -y libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev && \
    apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev && \
    apt-get install -y libxvidcore-dev libx264-dev && \
    apt-get install -y libgtk-3-dev && \
    apt-get install -y libatlas-base-dev gfortran && \
    apt-get install -y python3.5-dev && \
    apt-get install -y libcupti-dev

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.5 get-pip.py && \
    rm get-pip.py

RUN cd /opt && wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip && \
    unzip opencv.zip

RUN cd /opt && wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip && \
    unzip opencv_contrib.zip

RUN apt-get install -y git-core

RUN pip3 install numpy keras pandas h5py versioneer scikit-image && \
    pip3 install boto3 && \
    pip3 install tensorflow-gpu && \
    rm -rf /root/.cache/pip/*

RUN cd /opt/opencv-3.1.0/ && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D INSTALL_C_EXAMPLES=OFF \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-3.1.0/modules \
          -D PYTHON_EXECUTABLE=/usr/bin/python3.5 \
          -D BUILD_EXAMPLES=ON ..

RUN cd /opt/opencv-3.1.0/build && make -j4 && make clean && make && make install && ldconfig

RUN ln -s /usr/loca/lib/python3.5/dist-packages/cv2.cpython-35m-x86_64-linux-gnu.so \
    /usr/local/lib/python3.5/dist-packages/cv2.so


# install iceburger
RUN cd /root && git clone https://github.com/Hidysabc/iceburger.git && \
    cd iceburger && pip3 install -e .

# Clear apt downloads
RUN rm -rf /var/apt/lists/*

RUN cd /opt && rm -rf opencv.zip opencv_contrib.zip opencv-3.1.0 opencv_contrib-3.1.0

WORKDIR /root/iceburger/
