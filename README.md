# LLaVA-Deploy-Tutorial

**Chinese version is provided below. （中文版本请往下看）**

## Introduction (项目简介)
LLaVA-Deploy-Tutorial is an open-source project that provides a step-by-step tutorial for deploying the **LLaVA (Large Language and Vision Assistant)** multi-modal model. This project demonstrates how to set up the environment, download pre-trained LLaVA model weights, and run inference through both a command-line interface (CLI) and a web-based UI. By following this guide, users can quickly get started with LLaVA 1.5 and 1.6 models (7B and 13B variants) for image-question answering and multi-modal chatbot applications.

LLaVA-Deploy-Tutorial 是一个开源项目，提供部署 **LLaVA (Large Language and Vision Assistant)** 多模态模型的分步教程。本项目演示如何配置运行环境、下载预训练的 LLaVA 模型权重，以及通过命令行界面 (CLI) 和基于 Web 的界面进行推理。按照本教程，用户可以快速使用 LLaVA 1.5 和 1.6 系列模型（7B 和 13B 参数量）实现图像问答和多模态聊天功能。

## Features (功能)
- **Easy Setup:** Streamlined installation with Conda or pip, and helper scripts to automatically prepare environment (NVIDIA drivers, CUDA Toolkit, etc.).
- **Model Download:** Convenient script to download LLaVA 1.5/1.6 model weights (7B or 13B), with support for Hugging Face download acceleration via mirror.
- **Multiple Interfaces:** Run the model either through an interactive CLI or a Gradio Web UI, both with 4-bit quantization enabled by default for lower VRAM usage.
- **Extensibility:** Modular utilities (in `llava_deploy/utils.py`) for image preprocessing, model loading, and text prompt formatting, making it easier to integrate LLaVA into other applications.
- **Examples and Docs:** Provided example images and prompts for quick testing, and a detailed performance guide for hardware requirements and optimization tips.

- **简单部署：** 使用 Conda 或 pip 快速安装，附带辅助脚本自动准备环境（NVIDIA 驱动、CUDA 工具包等）。
- **模型下载：** 提供脚本方便下载 LLaVA 1.5/1.6 模型权重（7B 或 13B），支持通过镜像加速 Hugging Face 模型下载。
- **多种接口：** 可通过交互式命令行或 Gradio Web UI 运行模型，两种方式默认启用4-bit量化以降低显存占用。
- **模块化设计：** 在 `llava_deploy/utils.py` 中提供图像预处理、模型加载、提示格式化等工具函数，方便将 LLAVA 集成到其他应用。
- **示例与文档：** 提供示例图像和问题用于快速测试，并附有详细的性能指南，说明硬件需求和优化技巧。

## Environment Requirements (环境要求)
- **Operating System:** Ubuntu 20.04 (or compatible Linux). Windows is not officially tested (WSL2 is an alternative) and MacOS support is limited (no GPU acceleration).
- **Hardware:** NVIDIA GPU with CUDA capability. For LLaVA-1.5/1.6 models, we recommend at least **8 GB GPU VRAM** for the 7B model (with 4-bit quantization) and **16 GB VRAM** for the 13B model. Multiple GPUs can be used for larger models if needed.
- **NVIDIA Drivers:** NVIDIA driver supporting CUDA 11.8. Verify by running `nvidia-smi`. If not installed, see `scripts/setup_env.sh` which can assist in driver installation.
- **CUDA Toolkit:** CUDA 11.8 is recommended (if using PyTorch with CUDA 11.8). The toolkit is optional for runtime (PyTorch binaries include necessary CUDA libraries), but required if compiling any CUDA kernels.
- **Python:** Python 3.8+ (tested on 3.9/3.10). Using Conda is recommended for ease of environment management.
- **Others:** Git and **Git LFS** (Large File Storage) are required to fetch model weights from Hugging Face. Ensure `git lfs install` is run after installing Git LFS. An internet connection is needed for downloading models.

- **操作系统：** Ubuntu 20.04（或兼容的 Linux 发行版）。Windows 未经官方测试（可考虑使用 WSL2），MacOS 支持有限（无 GPU 加速）。
- **硬件：** 配备 NVIDIA GPU 且支持 CUDA。对于 LLaVA-1.5/1.6 模型，建议 7B 模型显存至少 **8GB**（在4-bit量化下），13B 模型显存至少 **16GB**。如果显存不足，可使用多张 GPU 配合加载更大的模型。
- **NVIDIA 驱动：** 安装支持 CUDA 11.8 的 NVIDIA 驱动。可通过运行 `nvidia-smi` 检查是否安装。如未安装，可参考 `scripts/setup_env.sh` 脚本来辅助安装驱动。
- **CUDA 工具包：** 推荐安装 CUDA 11.8（若使用与 CUDA 11.8 兼容的 PyTorch）。运行时可不安装 CUDA 工具包（PyTorch 二进制包含必要的 CUDA 库），但若需编译 CUDA 内核则需要安装。
- **Python：** Python 3.8 及以上（在 3.9/3.10 环境下测试通过）。建议使用 Conda 管理 Python 环境。
- **其他：** 需要安装 Git 及 **Git LFS**（大文件存储）用于从 Hugging Face 获取模型权重。确保安装 Git LFS 后执行过 `git lfs install`。下载模型需要网络连接支持。

## Installation (安装步骤)
You can set up the project either using **Conda** (with the provided `environment.yml`) or using **pip** with `requirements.txt`. Before installation, optionally run the automated environment setup script to ensure system dependencies are in place:
- *Optional:* Run `bash scripts/setup_env.sh` to automatically install system requirements (NVIDIA driver, CUDA, Miniconda, git-lfs). This script is intended for Ubuntu and requires sudo for driver installation. You can also perform these steps manually as described in Environment Requirements.
1. **Clone the repository:**
    git clone https://github.com/YourUsername/LLaVA-Deploy-Tutorial.git  
    cd LLaVA-Deploy-Tutorial
2. **Conda Environment (Recommended):** Create a Conda environment with the necessary packages.  
    conda env create -f environment.yml  
    conda activate llava_deploy
   This will install Python, PyTorch 2.x (with CUDA 11.8 support), Hugging Face Transformers, Gradio, and other dependencies.
3. **(Alternative) Pip Environment:** Ensure you have Python 3.8+ installed, then install packages via pip (preferably in a virtual environment).  
    python3 -m venv venv                  # optional: create virtual environment  
    source venv/bin/activate             # activate the virtual environment  
    pip install -U pip                   # upgrade pip  
    pip install -r requirements.txt
   *Note:* For GPU support, make sure to install the correct PyTorch wheel with CUDA (for example, `torch==2.0.1+cu118`). See [PyTorch documentation](https://pytorch.org/get-started/locally/) for more details if the default `torch` installation does not use GPU.
4. **Download Model Weights:** (See next section for details.) You will need to download LLaVA model weights separately, as they are not included in this repo.
5. **Verify Installation:** After installing dependencies and downloading a model, you can run a quick test:  
    python scripts/run_cli.py --model llava-1.5-7b --image examples/images/demo1.jpg --question "What is in this image?"
   If everything is set up correctly, the model will load and output an answer about the demo image.

您可以使用 **Conda**（通过提供的 `environment.yml`）或 **pip**（通过 `requirements.txt`）来安装本项目。在正式安装之前，可选地运行自动环境配置脚本，以确保系统依赖满足要求：
- *可选：* 运行 `bash scripts/setup_env.sh` 自动安装系统依赖（NVIDIA 驱动、CUDA、Miniconda、git-lfs）。该脚本适用于 Ubuntu 系统，安装驱动需 sudo 权限。您也可以根据环境要求章节手动完成这些步骤。
1. **克隆仓库：**
    git clone https://github.com/YourUsername/LLaVA-Deploy-Tutorial.git  
    cd LLaVA-Deploy-Tutorial
2. **Conda 环境（推荐）：** 使用 Conda 创建包含必要依赖的环境。  
    conda env create -f environment.yml  
    conda activate llava_deploy
   以上命令将安装 Python、PyTorch 2.x（支持 CUDA 11.8）、Hugging Face Transformers、Gradio 以及其他依赖项。
3. **（可选）pip 环境：** 确保已安装 Python 3.8+，然后使用 pip 安装所需包（建议在虚拟环境中执行）。  
    python3 -m venv venv                  # 可选：创建虚拟环境  
    source venv/bin/activate             # 激活虚拟环境  
    pip install -U pip                   # 升级 pip  
    pip install -r requirements.txt
   *注意：* 若需 GPU 支持，请确保安装正确的带 CUDA 的 PyTorch 版本（例如 `torch==2.0.1+cu118`）。如果默认安装的 `torch` 不能使用 GPU，请参考 [PyTorch 官方文档](https://pytorch.org/get-started/locally/) 了解安装支持 CUDA 的版本。
4. **下载模型权重：**（详细步骤见下一节）。您需要单独下载 LLaVA 模型权重，因为它们未包含在本仓库中。
5. **验证安装：** 安装依赖并下载模型后，可以运行快速测试：  
    python scripts/run_cli.py --model llava-1.5-7b --image examples/images/demo1.jpg --question "这张图片中有什么？"
   如果环境配置正确，模型将被加载，并针对示例图像输出相应的答案。

## Model Download (模型下载)
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

LLaVA 模型权重文件较大，未随仓库提供。请使用提供的脚本下载所需版本的模型：
- **可用模型：** 提供 LLaVA-1.5（7B 和 13B）和 LLaVA-1.6（7B 和 13B，使用 Vicuna 作为基座）模型可供选择。请选择与您硬件条件相匹配的模型（显存需求见环境要求）。
- **下载脚本：** 运行 `scripts/download_model.sh` 并指定模型名称，例如：  
    # 下载 LLaVA 1.5 7B 模型  
    bash scripts/download_model.sh llava-1.5-7b  
  脚本将在 `models/` 目录下创建对应子目录（该目录已在.gitignore中忽略），并下载所有必要的权重文件（总计可能达到数 GB）。确保已安装 Git LFS，脚本将使用 `git clone` 从 Hugging Face Hub 获取模型。
- **使用 Hugging Face 镜像：** 如果直接访问 huggingface.co 较慢，可以使用 `--hf-mirror` 参数从 [hf-mirror](https://hf-mirror.com) 镜像站下载。例如：  
    bash scripts/download_model.sh llava-1.5-13b --hf-mirror  
  该脚本将自动将下载地址替换为镜像站。或者，您也可以在运行脚本前设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`，效果相同。
- **访问权限：** LLaVA 权重托管在 Hugging Face，上游模型基于 LLaMA/Vicuna，下载前可能需要您同意模型许可。如果下载脚本提示权限错误，请确保：
  1. 您已有 Hugging Face 帐号，并在模型页面接受了其使用协议。
  2. 您已执行 `huggingface-cli login` 登录 Hugging Face CLI，或设置了环境变量 `HUGGINGFACE_HUB_TOKEN`（包含您的访问令牌）。
- **手动下载：** 您也可以选择手动从 Hugging Face 网站下载模型文件（或使用 Python 的 `huggingface_hub`工具）。下载完毕后，将模型文件放置到 `models/` 目录下相应子文件夹中（例如 `models/llava-1.5-7b/`）。

## Usage (使用方法)
Once the environment is set up and model weights are downloaded, you can run LLaVA in two ways: through the CLI for quick interaction or through a web interface for a richer experience.

环境配置完成并下载模型后，可以通过以下两种方式运行 LLaVA 模型：使用命令行界面进行快速交互，或使用 Web 界面获得更丰富的体验。

### CLI Interactive Mode (命令行交互)
The CLI allows you to chat with the model via the terminal. Use the `run_cli.py` script:
```bash
python scripts/run_cli.py --model llava-1.5-7b --image path/to/your_image.jpg
