ARG CONDA_VER=2023.09-0
ARG OS_TYPE=aarch64

FROM python:3.9.8 AS builder

# ==========================
# Dockerfile for Scikit Longitudinal Project
# System: Linux under Python 3.9.8 lightweight image
# Python: 3.9.8
# ==========================

RUN echo "==========================\nStage 1: The Build Process\n=========================="

# -----------------------------------
# 🛠 System-level Setup and Libraries 🛠
# -----------------------------------
RUN apt-get update && apt-get install -y libomp-dev

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
RUN pip install -U pip setuptools wheel
RUN pip install pdm

# ---------------------------
# 📦 Python Dependency Setup 📦
# ---------------------------
COPY pyproject.toml pdm.lock /scikit_longitudinal/
WORKDIR /scikit_longitudinal
RUN mkdir __pypackages__

FROM python:3.9
RUN echo "==========================\nStage 2: The Run-Time Setup\n=========================="

# -----------------------------------
# 🛠 System-level Setup and Libraries 🛠
# -----------------------------------
RUN echo "🛠 System-level Setup and Libraries 🛠"
RUN apt-get update && apt-get install -y libomp-dev build-essential wget curl libc-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# 🐍 Anaconda Installation 🐍
# -------------------------
RUN echo "🐍 Anaconda Installation 🐍"
ARG CONDA_VER
ARG OS_TYPE
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
RUN pip install pdm
ENV PDM_IN_ENV=in-project
RUN chmod +x /scikit_longitudinal/scripts/linux/docker_scikit_longitudinal_installs.sh /scikit_longitudinal/scripts/linux/docker_start_pdm_env.sh
RUN /scikit_longitudinal/scripts/linux/docker_scikit_longitudinal_installs.sh
