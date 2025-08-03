# Retro-PEFT-Adapters Development Container
FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        vim \
        htop \
        tmux \
        tree \
        jq \
        sqlite3 \
        redis-tools \
        postgresql-client \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA if available (for GPU support)
RUN if [ -f /usr/local/cuda/bin/nvcc ]; then \
        echo "CUDA found, configuring GPU support"; \
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/vscode/.bashrc; \
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /home/vscode/.bashrc; \
    fi

# Install Python dependencies for development
RUN pip install --upgrade pip setuptools wheel

# Install core ML libraries
RUN pip install \
    torch \
    transformers \
    accelerate \
    peft \
    datasets \
    sentence-transformers \
    scikit-learn \
    numpy \
    scipy

# Install vector database clients
RUN pip install \
    faiss-cpu \
    qdrant-client \
    weaviate-client

# Install development tools
RUN pip install \
    jupyter \
    ipykernel \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-mock \
    pre-commit \
    sphinx \
    sphinx-rtd-theme \
    myst-parser

# Install optional dependencies
RUN pip install \
    redis \
    h5py \
    psutil \
    tensorboard \
    wandb \
    mlflow

# Create workspace directories
RUN mkdir -p /workspaces/retro-peft-adapters/{models,cache,logs,indices,data}

# Set up git safe directory
RUN git config --global --add safe.directory /workspaces/retro-peft-adapters

# Configure shell
RUN echo 'alias ll="ls -la"' >> /home/vscode/.bashrc \
    && echo 'alias la="ls -la"' >> /home/vscode/.bashrc \
    && echo 'export EDITOR=vim' >> /home/vscode/.bashrc \
    && echo 'export PYTHONPATH=/workspaces/retro-peft-adapters/src:$PYTHONPATH' >> /home/vscode/.bashrc

# Switch to vscode user
USER vscode

# Set working directory
WORKDIR /workspaces/retro-peft-adapters