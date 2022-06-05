FROM jupyter/scipy-notebook

RUN pip install tensorflow==2.5.0 sklearn
RUN pip install numpy
 
USER root
RUN apt-get update && apt-get install -y jq
# Install packages required for compiling opencv
RUN apt-get -y install build-essential cmake pkg-config wget
# Install packages providing support for several image formats
RUN apt-get -y install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
# Install gtk (GUI components in opencv rely on gtk)
RUN apt-get -y install libgtk-3-dev
# Install additional packages optimizing opencv
RUN apt-get -y install libatlas-base-dev gfortran

RUN mkdir model MP_Data processed_data results


ENV MP_Data=/home/jovyan/MP_Data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RESULTS_DIR=/home/jovyan/results



COPY MP_Data ./MP_Data
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py

