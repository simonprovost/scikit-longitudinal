ARG PYTHON_BASE=3.9-slim
FROM python:$PYTHON_BASE AS builder

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

RUN apt-get update && apt-get install -y build-essential libomp-dev libc-dev && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV CC=gcc
ENV CXX=g++
ENV CPPFLAGS="-I/usr/local/include"
ENV CFLAGS="-Wall"
ENV CXXFLAGS="-Wall"
ENV LDFLAGS="-L/usr/local/lib"

RUN pip install -U pip setuptools wheel

COPY pyproject.toml pdm.lock /scikit_longitudinal/
COPY scikit_longitudinal/ /scikit_longitudinal/scikit_longitudinal/
COPY data/ /scikit_longitudinal/data/
COPY scripts/ /scikit_longitudinal/scripts/
COPY .env README.md .coveragerc /scikit_longitudinal/

WORKDIR /scikit_longitudinal
RUN pdm install --check --with :all --no-editable

FROM python:$PYTHON_BASE

COPY --from=builder /scikit_longitudinal/.venv/ /scikit_longitudinal/.venv
ENV PATH="/scikit_longitudinal/.venv/bin:$PATH"
COPY pyproject.toml pdm.lock /scikit_longitudinal/
COPY scikit_longitudinal/ /scikit_longitudinal/scikit_longitudinal/
COPY data/ /scikit_longitudinal/data/
COPY scripts/ /scikit_longitudinal/scripts/
COPY .env README.md .coveragerc /scikit_longitudinal/

WORKDIR /scikit_longitudinal
CMD ["/bin/bash"]