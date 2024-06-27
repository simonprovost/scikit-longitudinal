#ARG CONDA_VER=2023.09-0
#ARG OS_TYPE
#
#FROM ubuntu:latest AS builder
#
## ==========================
## Dockerfile for Scikit Longitudinal Project
## System: Linux under Python 3.9.8 lightweight image
## Python: 3.9.8
## ==========================
#
#RUN echo "==========================\nStage 1: The Build Process\n=========================="
#
## -----------------------------------
## 🛠 System-level Setup and Libraries 🛠
## -----------------------------------
#RUN apt-get update && \
#    apt-get install -y software-properties-common && \
#    add-apt-repository ppa:deadsnakes/ppa && \
#    apt-get update && \
#    apt-get install -y \
#    libomp-dev \
#    build-essential \
#    wget \
#    curl \
#    python3.9 \
#    python3.9-venv \
#    python3.9-dev \
#    gcc \
#    g++ \
#    libc-dev && \
#    apt-get remove -y python3-pip && \
#    rm -rf /var/lib/apt/lists/*
#
## ------------------------
## 🛠 Compiler Configurations 🛠
## ------------------------
#ENV CC=gcc
#ENV CXX=g++
#ENV CPPFLAGS="-I/usr/local/include"
#ENV CFLAGS="-Wall"
#ENV CXXFLAGS="-Wall"
#ENV LDFLAGS="-L/usr/local/lib"
#
## -------------------
## 🛠 Python Utilities 🛠
## -------------------
#RUN echo "🛠 Python Utilities 🛠"
#RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py
#RUN python3.9 -m pip install --upgrade setuptools wheel
#RUN python3.9 -m pip install pdm
#
## ---------------------------
## 📦 Python Dependency Setup 📦
## ---------------------------
#COPY pyproject.toml pdm.lock /scikit_longitudinal/
#WORKDIR /scikit_longitudinal
#RUN mkdir __pypackages__
#
#FROM ubuntu:latest
#
#RUN echo "==========================\nStage 2: The Run-Time Setup\n=========================="
#
## -----------------------------------
## 🛠 System-level Setup and Libraries 🛠
## -----------------------------------
#RUN echo "🛠 System-level Setup and Libraries 🛠"
#RUN apt-get update && \
#    apt-get install -y software-properties-common && \
#    add-apt-repository ppa:deadsnakes/ppa && \
#    apt-get update && \
#    apt-get install -y \
#    libomp-dev \
#    build-essential \
#    wget \
#    curl \
#    libc-dev \
#    python3.9 \
#    python3.9-venv \
#    python3.9-dev \
#    gcc \
#    g++ \
#    dpkg \
#    libc6:arm64 && \
#    apt-get remove -y python3-pip && \
#    rm -rf /var/lib/apt/lists/*
#
## -------------------------
## 🐍 Anaconda Installation 🐍
## -------------------------
#RUN echo "🐍 Anaconda Installation 🐍"
#ARG CONDA_VER
#ARG OS_TYPE
#RUN if [ -z "${OS_TYPE}" ]; then echo "OS_TYPE argument not provided"; exit 1; fi
#RUN wget -q "https://repo.anaconda.com/archive/Anaconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh" -O ~/Anaconda.sh
#RUN bash ~/Anaconda.sh -b -p /anaconda
#RUN rm ~/Anaconda.sh
#ENV PATH=/anaconda/bin:${PATH}
#RUN conda update --quiet -y conda
#
## ------------------------
## 🛠 Compiler Configurations 🛠
## ------------------------
#RUN echo "🛠 Compiler Configurations 🛠"
#ENV CC=gcc
#ENV CXX=g++
#ENV CPPFLAGS="-I/usr/local/include"
#ENV CFLAGS="-Wall"
#ENV CXXFLAGS="-Wall"
#ENV LDFLAGS="-L/usr/local/lib"
#
## ---------------------------
## 🐍 Python Environment Setup 🐍
## ---------------------------
#RUN echo "🐍 Python Environment Setup 🐍"
#ENV PYTHONPATH=/scikit_longitudinal/pkgs
#
## ----------------------
## 📦 Project File Setup 📦
## ----------------------
#RUN echo "📦 Project File Setup 📦"
#COPY pyproject.toml pdm.lock /scikit_longitudinal/
#COPY scikit_longitudinal/ /scikit_longitudinal/scikit_longitudinal/
#COPY scikit-learn/ /scikit_longitudinal/scikit-learn/
#COPY data/ /scikit_longitudinal/data/
#COPY scripts/ /scikit_longitudinal/scripts/
#COPY README.md .coveragerc /scripts/linux/docker_scikit_longitudinal_installs.sh /scripts/linux/docker_start_pdm_env.sh /scikit_longitudinal/
#
## -------------------------------
## 🚀 Scikit Longitudinal Installation 🚀
## -------------------------------
#RUN echo "🚀 Scikit Longitudinal Installation 🚀"
#WORKDIR /scikit_longitudinal
#RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py
#RUN python3.9 -m pip install pdm
#ENV PDM_IN_ENV=in-project
#RUN chmod +x /scikit_longitudinal/scripts/linux/docker_scikit_longitudinal_installs.sh /scikit_longitudinal/scripts/linux/docker_start_pdm_env.sh
#RUN /scikit_longitudinal/scripts/linux/docker_scikit_learn_tree_build.sh
### -------------------------------
### 🛠 Build Wheel for Scikit-Learn 🛠
### -------------------------------
##RUN echo "🛠 Build Wheel for Scikit-Learn 🛠"
##WORKDIR /scikit_longitudinal/scikit-learn
##RUN export PDM_IN_ENV=in-project && \
##    eval "$(pdm venv activate $PDM_IN_ENV)" && \
##    pip install numpy==1.23.3 && \
##    pip install scipy==1.10.1 && \
##    python3.9 setup.py bdist_wheel

ARG CONDA_VER=2023.09-0
ARG OS_TYPE=x86_64  # Ensure the OS_TYPE is set to x86_64

FROM ubuntu:latest AS builder

# ==========================
# Dockerfile for Scikit Longitudinal Project
# System: Linux under Python 3.9.8 lightweight image
# Python: 3.9.8
# ==========================

RUN echo "==========================\nStage 1: The Build Process\n=========================="

# -----------------------------------
# 🛠 System-level Setup and Libraries 🛠
# -----------------------------------
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    libomp-dev \
    build-essential \
    wget \
    curl \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    gcc \
    g++ \
    libc-dev && \
    apt-get remove -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# ------------------------
# 🛠 Compiler Configurations 🛠
# ------------------------
ENV CC=gcc
ENV CXX=g++
ENV CPPFLAGS="-I/usr/local/include"
ENV CFLAGS="-Wall"
ENV CXXFLAGS="-Wall"
ENV LDFLAGS="-L/usr/local/lib"

# -------------------
# 🛠 Python Utilities 🛠
# -------------------
RUN echo "🛠 Python Utilities 🛠"
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py
RUN python3.9 -m pip install --upgrade setuptools wheel
RUN python3.9 -m pip install pdm

# ---------------------------
# 📦 Python Dependency Setup 📦
# ---------------------------
COPY pyproject.toml pdm.lock /scikit_longitudinal/
WORKDIR /scikit_longitudinal
RUN mkdir __pypackages__

FROM ubuntu:latest

RUN echo "==========================\nStage 2: The Run-Time Setup\n=========================="

# -----------------------------------
# 🛠 System-level Setup and Libraries 🛠
# -----------------------------------
RUN echo "🛠 System-level Setup and Libraries 🛠"
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    libomp-dev \
    build-essential \
    wget \
    curl \
    libc-dev \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    gcc \
    g++ \
    dpkg && \
    apt-get remove -y python3-pip && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# 🐍 Anaconda Installation 🐍
# -------------------------
RUN echo "🐍 Anaconda Installation 🐍"
ARG CONDA_VER
ARG OS_TYPE=x86_64
RUN wget -q "https://repo.anaconda.com/archive/Anaconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh" -O ~/Anaconda.sh
RUN bash ~/Anaconda.sh -b -p /anaconda
RUN rm ~/Anaconda.sh
ENV PATH=/anaconda/bin:${PATH}
RUN conda update --quiet -y conda

# ------------------------
# 🛠 Compiler Configurations 🛠
# ------------------------
RUN echo "🛠 Compiler Configurations 🛠"
ENV CC=gcc
ENV CXX=g++
ENV CPPFLAGS="-I/usr/local/include"
ENV CFLAGS="-Wall"
ENV CXXFLAGS="-Wall"
ENV LDFLAGS="-L/usr/local/lib"

# ---------------------------
# 🐍 Python Environment Setup 🐍
# ---------------------------
RUN echo "🐍 Python Environment Setup 🐍"
ENV PYTHONPATH=/scikit_longitudinal/pkgs

# ----------------------
# 📦 Project File Setup 📦
# ----------------------
RUN echo "📦 Project File Setup 📦"
COPY pyproject.toml pdm.lock /scikit_longitudinal/
COPY scikit_longitudinal/ /scikit_longitudinal/scikit_longitudinal/
COPY scikit-learn/ /scikit_longitudinal/scikit-learn/
COPY data/ /scikit_longitudinal/data/
COPY scripts/ /scikit_longitudinal/scripts/
COPY README.md .coveragerc /scripts/linux/docker_scikit_longitudinal_installs.sh /scripts/linux/docker_start_pdm_env.sh /scikit_longitudinal/

# -------------------------------
# 🚀 Scikit Longitudinal Installation 🚀
# -------------------------------
RUN echo "🚀 Scikit Longitudinal Installation 🚀"
WORKDIR /scikit_longitudinal
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.9 get-pip.py && rm get-pip.py
RUN python3.9 -m pip install pdm
ENV PDM_IN_ENV=in-project
RUN chmod +x /scikit_longitudinal/scripts/linux/docker_scikit_longitudinal_installs.sh /scikit_longitudinal/scripts/linux/docker_start_pdm_env.sh
RUN /scikit_longitudinal/scripts/linux/docker_scikit_longitudinal_installs.sh

# -------------------------------
# 🛠 Build Wheel for Scikit-Learn 🛠
# -------------------------------
RUN echo "🛠 Build Wheel for Scikit-Learn 🛠"
WORKDIR /scikit_longitudinal/scikit-learn
RUN export PDM_IN_ENV=in-project && \
    eval "$(pdm venv activate $PDM_IN_ENV)" && \
    python3.9 setup.py bdist_wheel