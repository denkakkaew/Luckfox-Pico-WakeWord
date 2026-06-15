# Luckfox-Pico-WakeWord dev environment.
#
# rknn-toolkit2 is x86_64-only, so this image is pinned to linux/amd64 (see
# devcontainer.json runArgs). TensorFlow < 2.16 needs Python <= 3.11, so we base
# on the 3.10 devcontainer image, which also ships a non-root `vscode` user.
FROM mcr.microsoft.com/devcontainers/python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# Toolchain + runtime deps:
#   build-essential, make  -> cross-compile the board app (kws)
#   libgl1, libglib2.0-0   -> shared libs pulled in by rknn-toolkit2 (opencv)
#   alsa-utils             -> arecord, handy for inspecting/replaying WAV locally
#   curl, gnupg            -> add the NodeSource repo (Node for Claude Code)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        make \
        git \
        wget \
        curl \
        gnupg \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        alsa-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Node.js + Claude Code CLI, so you can run Claude inside the container.
# Auth/config live in ~/.claude, persisted via a volume (see devcontainer.json),
# so you only log in once.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python deps for training + RKNN conversion (see requirements.txt).
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
