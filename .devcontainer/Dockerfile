# TileLang Hello World Dev Container
# Ubuntu 22.04 base image (works on both ARM64 and x86_64)

FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Set up environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    sudo \
    zsh \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment and activate it
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install just command runner
RUN wget -qO - 'https://proget.makedeb.org/debian-feeds/prebuilt-mpr.pub' | gpg --dearmor | tee /usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg 1> /dev/null && \
    echo "deb [arch=all,$(dpkg --print-architecture) signed-by=/usr/share/keyrings/prebuilt-mpr-archive-keyring.gpg] https://proget.makedeb.org prebuilt-mpr jammy" | tee /etc/apt/sources.list.d/prebuilt-mpr.list && \
    apt-get update && \
    apt-get install -y just && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Note: Dependencies (torch, numpy, tilelang) are installed via postCreateCommand
# to ensure they're installed fresh when the container is created

CMD ["/bin/bash"]
