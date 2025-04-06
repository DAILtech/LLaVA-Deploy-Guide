# LLaVA-Deploy-Guide 中文版

## 项目简介
LLaVA-Deploy-Guide 是一个开源教程项目，旨在指导新手开发者部署 **LLaVA (Large Language and Vision Assistant)** 多模态大模型。通过本教程，您可以学习如何配置环境、下载 LLaVA 模型权重，并通过命令行界面或 Web 界面运行模型，实现对图像提问并获取回答的功能。支持 LLaVA 1.5 和 1.6 版本（7B 和 13B 模型），帮助您快速上手图像问答和多模态对话应用。  
_如果觉得本项目有帮助，请点点⭐️支持一下，我们团队非常感谢！🤗_

## 功能
- **简单部署：** 使用 Conda 或 pip 快速安装，附带辅助脚本自动准备环境（NVIDIA 驱动、CUDA 工具包等）。
- **模型下载：** 提供脚本方便下载 LLaVA 1.5/1.6 模型权重（7B 或 13B），支持通过镜像加速 Hugging Face 模型下载。
- **多种接口：** 可通过交互式命令行或 Gradio Web UI 运行模型，两种方式默认启用4-bit量化以降低显存占用。
- **模块化设计：** 在 `llava_deploy/utils.py` 中提供图像预处理、模型加载、提示格式化等工具函数，方便将 LLAVA 集成到其他应用。
- **示例与文档：** 提供示例图像和问题用于快速测试，并附有详细的性能指南，说明硬件需求和优化技巧。

## 环境需求
- **操作系统：** Ubuntu 20.04 或其他 Linux 发行版。Windows 用户可考虑使用 WSL2，MacOS 仅有限支持（无 GPU 加速）。
- **硬件：** NVIDIA GPU（支持 CUDA）。7B 模型建议至少 **8GB** 显存（4-bit 量化模式下），13B 模型建议至少 **16GB** 显存。如显存不足，可使用多块 GPU 加载模型。
- **NVIDIA 驱动：** 安装支持 CUDA 11.8 的 NVIDIA 驱动（运行 `nvidia-smi` 检查）。如未安装驱动，可参考脚本 `scripts/setup_env.sh` 协助安装。
- **CUDA 工具包：** 推荐安装 CUDA Toolkit 11.8。尽管使用 PyTorch 不一定需要单独安装 CUDA 工具包，但在编译部分模块时可能需要。
- **Python：** Python 3.8 或更高版本（开发中使用 3.9/3.10 测试）。建议使用 Conda 管理 Python 环境。
- **其他依赖：** 需安装 Git 和 **Git LFS**（大文件存储）用于下载模型权重。确保安装 Git LFS 后执行了 `git lfs install`。下载模型需要网络连接。

## 安装步骤
1. **克隆项目：**

   执行以下命令获取代码：
   ```bash
    git clone https://github.com/DAILtech/LLaVA-Deploy-Guide.git  
    cd LLaVA-Deploy-Guide
   ```
   运行结果示例：  
   ![image](https://github.com/user-attachments/assets/564839e1-9708-473c-bd99-f424e4cf4273)

2. **Conda 创建环境（推荐）：**
  
   使用 Conda 根据 `environment.yml` 创建环境并安装依赖：
   ```bash
    conda env create -f environment.yml  
    conda activate llava_deploy
   ```
   该步骤将安装 Python、PyTorch 2.x（CUDA 11.8）、Transformers、Gradio 等所需库。
   如果在国内加载速度过慢，可将`environment.tml`替换为`environment_zh.tml`，使用国内镜像源下载环境配置文件。  
   运行结果示例：  
   ![image](https://github.com/user-attachments/assets/258a57ae-9439-4121-888b-d6009440155a)

3. **或使用 pip 安装：**

   确保系统已有 Python 3.8+，可选择使用虚拟环境，然后通过 pip 安装：
   ```bash
    python3 -m venv venv                  # （可选）创建虚拟环境  
    source venv/bin/activate             # 激活虚拟环境  
    pip install -U pip                   # 升级 pip  
    pip install -r requirements.txt
   ```
   *注意：* 如果需要 GPU 加速，请安装匹配 CUDA 版本的 PyTorch （例如通过 `pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html`）。默认直接使用 `pip install torch` 可能安装无 CUDA 支持的版本，请根据 PyTorch 官方指南选择正确版本。
   
4. **下载模型：**

   模型权重需另行下载（见下文“模型下载”部分）。您可以运行脚本获取所需模型。
5. **测试验证：**

    安装完依赖并下载模型后，运行示例命令测试是否成功：
    `python -m scripts/run_cli.py --model llava-1.5-7b --image examples/images/demo1.jpg --question "这张图片中有什么？"`
   若一切配置正确，模型将加载并对示例图片给出合理的描述答案。

## 模型下载
由于模型体积较大，本仓库未直接提供 LLaVA 模型权重。请按照以下步骤获取：
- **可选模型：**
   LLaVA-1.5 和 LLaVA-1.6 的 7B/13B 权重。请选择与您GPU显存相匹配的模型版本。  
1.  **LFS安装**：确保已安装Git LFS，安装指令：
   ```bash
   sudo apt-get update
   sudo apt-get install git-lfs
   ```  
   安装后初始化Git LFS：  
   ```bash
   git lfs install
   ```  
   运行结果示例：  
   ![image](https://github.com/user-attachments/assets/33243f7c-adb4-4dde-b631-a640e1269ad2)  

2.  **使用下载脚本：**   
运行 `scripts/download_model.sh` 并指定模型版本名称。  
例如下载 LLaVA-1.5-7B 模型：   
```bash
    bash scripts/download_model.sh llava-1.5-7b    
```   
  运行后将在 `models/` 目录下创建对应子文件夹，并下载模型文件到其中（`models/` 已添加到 .gitignore）。脚本将通过 `git clone` 从 Hugging Face 获取权重。  
  运行结果示例：
  ![image](https://github.com/user-attachments/assets/f193ed59-6a72-42bf-9fea-5a8760cae16b)  
  
3. **镜像加速：**  
如果在国内直接下载速度缓慢，可以添加参数 `--hf-mirror` 使用 Hugging Face 国内镜像源：   
```bash
    bash scripts/download_model.sh llava-1.5-7b --hf-mirror
```    
  脚本将使用 hf-mirror 加速下载。或手动设置环境变量 `HF_ENDPOINT=https://hf-mirror.com` 后再运行脚本。  
  
4. **权限提示：**   
如果脚本提示没有权限下载，请确认在 Hugging Face 对应模型页面接受了使用协议，并确保已登录 Hugging Face（例如运行过 `huggingface-cli login`）。必要时，可在脚本中配置 Hugging Face Token 或手动下载模型后放置到 `models/`目录。

6. **手动下载：**   
您也可以从 Hugging Face 网站手动下载所有模型文件，然后将它们置于 `models/<模型名称>/` 文件夹下（例如 `models/llava-1.5-7b/`）。下载时建议使用支持断点续传的工具加速。

## 使用方法  
模型准备就绪后，您可以通过命令行或 Web 界面两种方式与模型交互：

### 命令行交互
运行 `run_cli.py` 脚本，可进入与模型对话的命令行界面。例如：
```bash
python scripts/run_cli.py --model llava-1.5-7b --image path/to/your_image.jpg
```
如果不使用 `--question` 参数，脚本将进入交互模式。载入图像后，您可以在提示符下输入关于该图像的问题。输入 `exit` 或按 `Ctrl+C` 可退出。  
示例： $ `python scripts/run_cli.py --model llava-1.5-7b --image examples/images/demo1.jpg`  
已加载 llava-1.5-7b 模型（4-bit 量化）。 图像 'examples/images/demo1.jpg' 已加载，可开始提问。  
Question: **这张图片中的动物是什么？**  
Answer: 这是一只猫。  
Question: **它正在做什么**   
Answer: 它正躺着休息。  
Question: **exit**  
如上例，您可以在同一次会话中对同一图像连续提问（支持中英混合）。也可通过加入 `--question` 参数直接询问单个问题：
```bash
python scripts/run_cli.py --model llava-1.5-7b \\
       --image examples/images/demo2.png \\
       --question "请描述这张图片中的场景。"
```
提示：--model 参数既可使用预设名称（如 "llava-1.5-7b"），也可指向本地模型路径。CLI 模式默认使用 4-bit 量化加载模型以减少显存占用。
## Web界面
运行 run_webui.py 脚本可启动基于 Gradio 的 Web 图形界面。例如：
```bash
python scripts/run_webui.py --model llava-1.5-7b
```
成功后，终端会显示本地访问链接（默认为 http://localhost:7860）。打开该链接即可看到 Web 界面，在其中上传图片并输入问题，点击“Submit”提交即可得到回答。
- Web UI 支持中文和英文提问，并内置了示例供一键测试（使用 examples/ 目录下的示例图片和问题）。
- 模型在后端以 4-bit 量化方式运行。初次提问可能略有延迟（加载模型所致），之后对同一图像的提问会更快。
- 您可以对上传的同一图片连续提问多个问题；如需更换图片，上传新图片即可。
- 如需停止服务器，在终端按下 `Ctrl+C` 即可停止 Gradio 服务。

## 文档和支持
更多性能测试数据和硬件建议，请参阅docs/performance.md 性能指南。如在使用中遇到问题，欢迎提交 issue 与我们交流。我们也欢迎您贡献代码或建议，共同改进本项目。
