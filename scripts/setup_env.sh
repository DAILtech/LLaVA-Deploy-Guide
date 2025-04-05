#!/bin/bash
# This script attempts to set up the system environment for LLaVA-Deploy-Guide.
# It checks for NVIDIA driver, CUDA Toolkit, Miniconda, and git-lfs, and installs them if needed.
# Note: This script is designed for Ubuntu systems and requires sudo for certain installations (e.g., drivers).

set -e

echo "=== LLaVA-Deploy-Tutorial Environment Setup Script ==="

# 1. Check NVIDIA GPU and driver
if command -v nvidia-smi &> /dev/null; then
    echo "[Info] NVIDIA GPU detected. Driver version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
else
    echo "[Warning] NVIDIA GPU driver not found. Attempting to install the latest NVIDIA driver..."
    echo "This may require a reboot once completed."
    sleep 2
    sudo apt-get update
    # Install NVIDIA drivers using ubuntu-drivers (this will install appropriate driver for the system)
    sudo ubuntu-drivers autoinstall || echo "[Error] Automatic NVIDIA driver installation failed. Please install drivers manually."
fi

# 2. Check CUDA Toolkit (nvcc)
if command -v nvcc &> /dev/null; then
    echo "[Info] CUDA Toolkit is installed. nvcc version: $(nvcc --version | grep release)"
else
    echo "[Warning] CUDA Toolkit not found. Installing CUDA Toolkit 11.8..."
    # Add NVIDIA package repository for CUDA 11.8
    CUDA_PKG=cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    wget -q https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/${CUDA_PKG}
    sudo dpkg -i ${CUDA_PKG} && sudo apt-key add /var/cuda-repo-ubuntu2004-11-8-local/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-11-8
    echo "[Info] CUDA Toolkit 11.8 installed."
fi

# 3. Install Miniconda (if conda is not present)
if command -v conda &> /dev/null; then
    echo "[Info] Conda is already installed: $(conda --version)"
else
    echo "[Info] Miniconda not found. Installing Miniconda3 (silent mode)..."
    CONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh
    wget -q https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}
    bash ${CONDA_INSTALLER} -b -p $HOME/miniconda
    source $HOME/miniconda/bin/activate
    conda init bash
    echo "[Info] Miniconda installed at $HOME/miniconda. Please restart your shell or run 'source ~/.bashrc' to activate conda."
fi

# 4. Install git-lfs (if not present)
if command -v git-lfs &> /dev/null; then
    echo "[Info] git-lfs is already installed."
else
    echo "[Info] Installing git-lfs..."
    sudo apt-get update
    sudo apt-get install -y git-lfs
    git lfs install
fi

echo "=== System environment setup complete. ==="
echo "Next steps: Create the Python environment and install dependencies (see README)."
