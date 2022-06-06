FROM jupyter/scipy-notebook:8ee3ea7a590a
RUN pip install pip --upgrade
RUN pip3 install tensorflow tensorflow-gpu
RUN pip install sklearn
RUN pip install numpy

USER root
RUN apt-get update && apt-get install -y jq
ARG NUM_CORES=2

## Install packages
RUN apt-get update -qq && \
    apt-get install -y --force-yes --fix-missing --no-install-recommends \
    # Basics
    wget curl build-essential checkinstall unzip autoconf automake libtool g++ yasm \
    # OpenCV Prerequisites
    cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev \
    # Extra packages for OpenCV
    libtiff5-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev \
    libv4l-dev v4l-utils libmp3lame-dev \
    libboost-all-dev libeigen3-dev
WORKDIR /tmp
RUN git clone --depth 1 git://source.ffmpeg.org/ffmpeg && \
  cd ffmpeg && \
  ./configure --enable-shared --disable-static --enable-nonfree && \
  make -j ${NUM_CORES} && \
  make install && \
  make clean

# Build OpenCV 3.x
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/opencv/opencv.git
RUN git clone --depth 1 https://github.com/opencv/opencv_contrib.git
RUN mkdir -p opencv/build && \
    cd opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_TBB=ON \
          -D BUILD_PYTHON_SUPPORT=ON \
          -D WITH_V4L=ON \
          -D INSTALL_C_EXAMPLES=ON \
          -D INSTALL_PYTHON_EXAMPLES=ON \
          -D BUILD_EXAMPLES=ON \
          -D BUILD_DOCS=ON \
          -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
          -D WITH_XIMEA=YES \
#          -D WITH_QT=YES \
          -D WITH_FFMPEG=YES \
          -D WITH_PVAPI=YES \
          -D WITH_GSTREAMER=YES \
          -D WITH_TIFF=YES \
          -D WITH_OPENCL=YES \
          -D PYTHON2_EXECUTABLE=/opt/conda/envs/python2/bin/python \
          -D PYTHON2_INCLUDE_DIR=/opt/conda/envs/python2/include/python2.7 \
          -D PYTHON2_LIBRARIES=/opt/conda/envs/python2/lib/libpython2.7.so \
          -D PYTHON2_PACKAGES_PATH=/opt/conda/envs/python2/lib/python2.7/site-packages \
          -D PYTHON2_NUMPY_INCLUDE_DIRS=/opt/conda/envs/python2/lib/python2.7/site-packages/numpy/core/include/ \
          -D BUILD_opencv_python3=ON \
          -D PYTHON3_EXECUTABLE=/opt/conda/bin/python \
          -D PYTHON3_INCLUDE_DIR=/opt/conda/include/python3.5m/ \
          -D PYTHON3_LIBRARY=/opt/conda/lib/libpython3.so \
          -D PYTHON_LIBRARY=/opt/conda/lib/libpython3.so \
          -D PYTHON3_PACKAGES_PATH=/opt/conda/lib/python3.5/site-packages \
          -D PYTHON3_NUMPY_INCLUDE_DIRS=/opt/conda/lib/python3.5/site-packages/numpy/core/include/ \
          .. && \
          make -j ${NUM_CORES} && \
          make install && \
          make clean && \
          sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf'

## Additional modules
# GSL 2.1
WORKDIR /tmp
RUN curl -L ftp://ftp.gnu.org/gnu/gsl/gsl-2.1.tar.gz | tar zxf - && \
    cd gsl-2.1 && \
    ./configure && \
    make -j ${NUM_CORES} && \
    make install && \
    make clean

# VLFeat
WORKDIR /usr/local/src
RUN wget http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz && \
    tar zxvf vlfeat-0.9.20-bin.tar.gz && \
    rm vlfeat-0.9.20-bin.tar.gz
ENV VLFEAT_DIR=/usr/local/src/vlfeat-0.9.20

# fix libdc1394 error: Failed to initialize libdc1394
RUN ln /dev/null /dev/raw1394
RUN echo 'KERNEL=="raw1394", GROUP="video"' > /etc/udev/rules.d/raw1394.rules

# cleanup package manager
RUN ldconfig && \
    apt-get remove --purge -y curl build-essential checkinstall cmake && \
    apt-get autoclean && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

## Misc
ENV LC_ALL=C

## Switch back to jupyter user
USER jovyan
WORKDIR /home/jovyan/work

## Additional python modules
#RUN /opt/conda/envs/python2/bin/pip install imutils minieigen
#RUN /opt/conda/bin/pip install imutils minieigen

RUN pip install opencv-python
RUN pip install mediapipe
RUN mkdir model MP_Data processed_data results


ENV MP_Data=/home/jovyan/MP_Data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RESULTS_DIR=/home/jovyan/results



COPY MP_Data ./MP_Data
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py

