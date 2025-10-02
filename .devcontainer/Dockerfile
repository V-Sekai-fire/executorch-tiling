# ExecutorTorch Tiled Inference Dev Container
# Ubuntu 24.04 base image (works on both ARM64 and x86_64)

FROM ubuntu:24.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    just \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install uv (modern Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set working directory
WORKDIR /workspace

# Note: Dependencies (torch, executorch, torchvision) are installed via postCreateCommand
# to ensure they're installed fresh when the container is created
