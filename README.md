# LLaVA-Deploy-Guide

**Chinese version is provided in `README_zh.md`.**

## Introduction 
LLaVA-Deploy-Guide is an open-source project that provides a step-by-step tutorial for deploying the **LLaVA (Large Language and Vision Assistant)** multi-modal model. This project demonstrates how to set up the environment, download pre-trained LLaVA model weights, and run inference through both a command-line interface (CLI) and a web-based UI. By following this guide, users can quickly get started with LLaVA 1.5 and 1.6 models (7B and 13B variants) for image-question answering and multi-modal chatbot applications.  
_If you find this project helpful, please consider giving us a Star ⭐️. Your support means a lot to our team._

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
    git clone https://github.com/DAILtech/LLaVA-Deploy-Guide.git  
    cd LLaVA-Deploy-Guide
    ```
2. **Conda Environment (Recommended):**  
   Create a Conda environment with the necessary packages.
    ```bash
    conda env create -f environment.yml  
    conda activate llava_deploy
    ```
   This will install Python, PyTorch 2.x (with CUDA 11.8 support), Hugging Face Transformers, Gradio, and other dependencies. Replaced by `environment_zh.yml` in China.
3. **(Alternative) Pip Environment:**  
   Ensure you have Python 3.8+ installed, then install packages via pip (preferably in a virtual environment).
    ```bash
    python3 -m venv venv                  # optional: create virtual environment  
    source venv/bin/activate             # activate the virtual environment  
    pip install -U pip                   # upgrade pip  
    pip install -r requirements.txt
     ```
   *Note:* For GPU support, make sure to install the correct PyTorch wheel with CUDA (for example, `torch==2.0.1+cu118`). See [PyTorch documentation](https://pytorch.org/get-started/locally/) for more details if the default `torch` installation does not use GPU.
4. **Download Model Weights:** (See next section for details.)   
   You will need to download LLaVA model weights separately, as they are not included in this repo.
5. **Verify Installation:**  
   After installing dependencies and downloading a model, you can run a quick test:
    ```bash
    python scripts/run_cli.py --model llava-1.5-7b --image examples/images/demo1.jpg --question "What is in this image?"
     ```
   If everything is set up correctly, the model will load and output an answer about the demo image.

## Model Download  
The LLaVA model weights are not distributed with this repository due to their size. Use the provided script to download the desired version of LLaVA:
- **Available Models:** LLaVA-1.5 (7B and 13B) and LLaVA-1.6 (7B and 13B) with Vicuna backend. Ensure you have sufficient VRAM for the model you choose (see Environment Requirements).
- **Download Script:** Run `scripts/download_model.sh` with the model name. For example: **Download LLaVA 1.5 7B model：**  
1. **Install LFS:**
Make sure that LFS is installed, installation instructions:
```bash
sudo apt-get update
sudo apt-get install git-lfs
```
Then initial Git LFS:
```bash
git lfs install
```
Output:  
2. **Download models:**
```bash
bash scripts/download_model.sh llava-1.5-7b
```
      This will create a directory under `models/` (which is git-ignored) and download all necessary weight files there (it may be several GBs). If you have Git LFS installed, the script uses `git clone` from Hugging Face Hub.
3. **Using Hugging Face Mirror:** If you are in a region with slow access to huggingface.co, you can use the `--hf-mirror` flag to download from the [hf-mirror](https://hf-mirror.com) site. For example:  
     `bash scripts/download_model.sh llava-1.5-13b --hf-mirror`
      The script will replace the download URLs to use the mirror. Alternatively, you can set the environment variable `HF_ENDPOINT=https://hf-mirror.com` before running the script for the same effect.
4. **Hugging Face Access:** The LLaVA weights are hosted on Hugging Face and may require you to accept the model license (since they are based on LLaMA/Vicuna). If the download script fails due to permission,     make sure:
      1. You have a Hugging Face account and have accepted the usage terms for the LLaVA model repositories.
      2. You have run `huggingface-cli login` (if using Hugging Face CLI) or set `HUGGINGFACE_HUB_TOKEN` environment variable with your token.
5. **Manual Download:** As an alternative, you can manually download the model from the Hugging Face web UI or using `huggingface_hub` Python library, then place the files under the `models/` directory (e.g., `models/llava-1.5-7b/`).

## Usage 
Once the environment is set up and model weights are downloaded, you can run LLaVA in two ways: through the CLI for quick interaction or through a web interface for a richer experience.

### CLI Interactive Mode 
The CLI allows you to chat with the model via the terminal. Use the `run_cli.py` script:
```bash
python scripts/run_cli.py --model llava-1.5-7b --image path/to/your_image.jpg
```
By default, if you do not provide a `--question` argument, the script will launch an interactive session. You will be prompted to enter a question after the image is loaded. For example: $ python scripts/run_cli.py --model llava-1.5-7b --image examples/images/demo1.jpg Loaded model llava-1.5-7b (4-bit quantized). Image 'examples/images/demo1.jpg' loaded. You can now ask questions about the image. Question: **What is the animal in this image?** Answer: This image shows a cat. Question: **what is it doing?** Answer: It is resting. Question: **exit** 
In the above session, the user asked an English question and a Chinese question about the same image, and the model answered accordingly. Type `exit` (or press `Ctrl+C`) to quit the interactive mode. You can also specify a one-time question via the `--question` argument for a single-turn inference:

```bash
python scripts/run_cli.py --model llava-1.5-7b \\
       --image examples/images/demo2.png \\
       --question "Describe the scene in this image."
```
(The --model parameter accepts either a model name like "llava-1.5-7b" or a path to a local model directory. The script automatically enables 4-bit quantization for efficiency.)
### Web UI (Gradio)
The web UI provides an easy-to-use interface for uploading images and asking questions. Launch it by running:
```bash
python scripts/run_webui.py --model llava-1.5-7b
```
This will start a local web server and display a Gradio interface. By default, the server runs on `http://localhost:7860`. You should see a web page that allows you to upload an image and enter a question. Click "Submit" to get the model's answer.
- The Gradio UI supports both English and Chinese inputs. It also includes a few example image+question pairs (loaded from the examples/ directory) that you can try with one click.
- The model is loaded with 4-bit quantization in the backend, just like in CLI mode. The first time you ask a question, there might be a delay as the model initializes on the GPU. Subsequent questions on the same image will be faster.
- You can ask multiple questions about the same image. To switch to a different image, upload a new image file.
- To stop the web server, press Ctrl+C in the terminal where it's running.

## License
This project is released under the MIT License. Note that the LLaVA model weights are subject to their own licenses (e.g., LLaMA and Vicuna licenses) – please review and comply with those when using the models.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Documentation and Support 
For detailed performance information and hardware recommendations, see the performance guide. If you encounter issues with this tutorial, please feel free to open an issue on GitHub or reach out to the maintainers. We welcome contributions to improve this project.
