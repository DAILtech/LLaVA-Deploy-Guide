# LLaVA-Deploy-Guide

**Chinese version is provided below.**

## Introduction 
LLaVA-Deploy-Guide is an open-source project that provides a step-by-step tutorial for deploying the **LLaVA (Large Language and Vision Assistant)** multi-modal model. This project demonstrates how to set up the environment, download pre-trained LLaVA model weights, and run inference through both a command-line interface (CLI) and a web-based UI. By following this guide, users can quickly get started with LLaVA 1.5 and 1.6 models (7B and 13B variants) for image-question answering and multi-modal chatbot applications.

## Features
- **Easy Setup:** Streamlined installation with Conda or pip, and helper scripts to automatically prepare environment (NVIDIA drivers, CUDA Toolkit, etc.).
- **Model Download:** Convenient script to download LLaVA 1.5/1.6 model weights (7B or 13B), with support for Hugging Face download acceleration via mirror.
- **Multiple Interfaces:** Run the model either through an interactive CLI or a Gradio Web UI, both with 4-bit quantization enabled by default for lower VRAM usage.
- **Extensibility:** Modular utilities (in `llava_deploy/utils.py`) for image preprocessing, model loading, and text prompt formatting, making it easier to integrate LLaVA into other applications.
- **Examples and Docs:** Provided example images and prompts for quick testing, and a detailed performance guide for hardware requirements and optimization tips.

## Environment Requirements
- **Operating System:** Ubuntu 20.04 (or compatible Linux). Windows is not officially tested (WSL2 is an alternative) and MacOS support is limited (no GPU acceleration).
- **Hardware:** NVIDIA GPU with CUDA capability. For LLaVA-1.5/1.6 models, we recommend at least **8 GB GPU VRAM** for the 7B model (with 4-bit quantization) and **16 GB VRAM** for the 13B model. Multiple GPUs can be used for larger models if needed.
- **NVIDIA Drivers:** NVIDIA driver supporting CUDA 11.8. Verify by running `nvidia-smi`. If not installed, see `scripts/setup_env.sh` which can assist in driver installation.
- **CUDA Toolkit:** CUDA 11.8 is recommended (if using PyTorch with CUDA 11.8). The toolkit is optional for runtime (PyTorch binaries include necessary CUDA libraries), but required if compiling any CUDA kernels.
- **Python:** Python 3.8+ (tested on 3.9/3.10). Using Conda is recommended for ease of environment management.
- **Others:** Git and **Git LFS** (Large File Storage) are required to fetch model weights from Hugging Face. Ensure `git lfs install` is run after installing Git LFS. An internet connection is needed for downloading models.

## Installation
You can set up the project either using **Conda** (with the provided `environment.yml`) or using **pip** with `requirements.txt`. Before installation, optionally run the automated environment setup script to ensure system dependencies are in place:
- *Optional:* Run `bash scripts/setup_env.sh` to automatically install system requirements (NVIDIA driver, CUDA, Miniconda, git-lfs). This script is intended for Ubuntu and requires sudo for driver installation. You can also perform these steps manually as described in Environment Requirements.
1. **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/LLaVA-Deploy-Tutorial.git  
    cd LLaVA-Deploy-Tutorial
    ```
2. **Conda Environment (Recommended):** Create a Conda environment with the necessary packages.
    ```bash
    conda env create -f environment.yml  
    conda activate llava_deploy
    ```
   This will install Python, PyTorch 2.x (with CUDA 11.8 support), Hugging Face Transformers, Gradio, and other dependencies.
4. **(Alternative) Pip Environment:** Ensure you have Python 3.8+ installed, then install packages via pip (preferably in a virtual environment).  
    python3 -m venv venv                  # optional: create virtual environment  
    source venv/bin/activate             # activate the virtual environment  
    pip install -U pip                   # upgrade pip  
    pip install -r requirements.txt
   *Note:* For GPU support, make sure to install the correct PyTorch wheel with CUDA (for example, `torch==2.0.1+cu118`). See [PyTorch documentation](https://pytorch.org/get-started/locally/) for more details if the default `torch` installation does not use GPU.
5. **Download Model Weights:** (See next section for details.) You will need to download LLaVA model weights separately, as they are not included in this repo.
6. **Verify Installation:** After installing dependencies and downloading a model, you can run a quick test:  
    python scripts/run_cli.py --model llava-1.5-7b --image examples/images/demo1.jpg --question "What is in this image?"
   If everything is set up correctly, the model will load and output an answer about the demo image.

## Model Download 
The LLaVA model weights are not distributed with this repository due to their size. Use the provided script to download the desired version of LLaVA:
- **Available Models:** LLaVA-1.5 (7B and 13B) and LLaVA-1.6 (7B and 13B) with Vicuna backend. Ensure you have sufficient VRAM for the model you choose (see Environment Requirements).
- **Download Script:** Run `scripts/download_model.sh` with the model name. For example:  
    # Download LLaVA 1.5 7B model  
    bash scripts/download_model.sh llava-1.5-7b  
  This will create a directory under `models/` (which is git-ignored) and download all necessary weight files there (it may be several GBs). If you have Git LFS installed, the script uses `git clone` from Hugging Face Hub.
- **Using Hugging Face Mirror:** If you are in a region with slow access to huggingface.co, you can use the `--hf-mirror` flag to download from the [hf-mirror](https://hf-mirror.com) site. For example:  
    bash scripts/download_model.sh llava-1.5-13b --hf-mirror  
  The script will replace the download URLs to use the mirror. Alternatively, you can set the environment variable `HF_ENDPOINT=https://hf-mirror.com` before running the script for the same effect.
- **Hugging Face Access:** The LLaVA weights are hosted on Hugging Face and may require you to accept the model license (since they are based on LLaMA/Vicuna). If the download script fails due to permission, make sure:
  1. You have a Hugging Face account and have accepted the usage terms for the LLaVA model repositories.
  2. You have run `huggingface-cli login` (if using Hugging Face CLI) or set `HUGGINGFACE_HUB_TOKEN` environment variable with your token.
- **Manual Download:** As an alternative, you can manually download the model from the Hugging Face web UI or using `huggingface_hub` Python library, then place the files under the `models/` directory (e.g., `models/llava-1.5-7b/`).

## Usage 
Once the environment is set up and model weights are downloaded, you can run LLaVA in two ways: through the CLI for quick interaction or through a web interface for a richer experience.

### CLI Interactive Mode 
The CLI allows you to chat with the model via the terminal. Use the `run_cli.py` script:
```bash
python scripts/run_cli.py --model llava-1.5-7b --image path/to/your_image.jpg

### 
