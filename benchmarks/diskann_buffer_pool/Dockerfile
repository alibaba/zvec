FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    libaio1t64 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir \
      "cmake>=3.26,<4" \
      ninja==1.11.1.1 \
      numpy \
      pybind11==3.0 \
      scikit-build-core \
      setuptools_scm

ENV PATH=/opt/venv/bin:${PATH}
WORKDIR /work
