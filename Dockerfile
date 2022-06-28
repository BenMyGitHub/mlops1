FROM jupyter/scipy-notebook

RUN pip install tensorflow sklearn
RUN pip install numpy

USER root
RUN apt-get update && apt-get install -y jq


RUN mkdir model MP_Data processed_data results


ENV MP_Data=/home/jovyan/MP_Data
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV MODEL_DIR=/home/jovyan/model
ENV RESULTS_DIR=/home/jovyan/results



COPY MP_Data ./MP_Data
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
