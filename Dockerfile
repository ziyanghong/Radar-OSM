#FROM ros:kinetic-perception



#FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
#FROM nvidia/cuda:7.5-cudnn6-runtime-ubuntu14.04
# FROM 5463965/ubuntu16.04-cuda8.0-cudnn7
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu16.04

SHELL ["/bin/bash", "-c"]
RUN apt update

RUN apt install llvm-8 libspatialindex-dev libboost-python-dev libboost-serialization-dev libopenmpi-dev vim curl qt4-default python-dev python-setuptools python-pip -y
# RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
# RUN python get-pip.py


RUN pip install pip==20.3.4


RUN pip install numpy==1.16.6

# RUN pip3 install --upgrade pip3 setuptools
#RUN export LLVM_CONFIG=$(which llvm-config-3.9)
#RUN pip install enum34==0.9.23
RUN pip install llvmlite==0.31.0
#RUN pip install numba==0.16.0
RUN pip install numba==0.47.0

RUN pip install pyparsing==2.0.3
RUN pip install scikit-learn
RUN pip install scipy==1.2.0
RUN pip install networkx==1.1
RUN pip install cycler==0.10
RUN pip install matplotlib==2.1.2
RUN pip install Pillow==2.2.2
RUN pip install mpi4py==1.3.1
RUN pip install pydevd-pycharm~=212.5457.59
RUN pip install osmium==2.15.0
RUN pip install opencv-python==3.3.1.11
RUN pip install utm
RUN pip install Rtree==0.9.1
#RUN pip install Cython==0.29.33

# install ceres Dependencies

CMD [ "bash" ]
