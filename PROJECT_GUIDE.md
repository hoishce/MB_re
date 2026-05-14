# MockingBird 本地工程说明

本仓库基于 MockingBird，并加入了本地音频清洗、下载、切片、转写、数据集构建和训练辅助脚本。它适合上传到 GitHub 的内容是代码、配置、运行脚本和说明文档；大文件与本地产物默认不提交。

## 主要目录

- `models/`：encoder、synthesizer、vocoder、PPG 等核心模型代码。
- `control/cli/`：训练和预处理命令行入口。
- `demo/`：本地自动化流水线、下载、清洗、转写和推理示例。
- `tools/`：数据准备、校验、批处理、CUDA 诊断、训练启动等工具脚本。
- `requirements-*.txt`：按用途拆分的依赖文件。
- `run_*.bat`、`tools/run_synth_train.ps1`：Windows 快速启动脚本。
- `TRAINING_GUIDE.md`：数据准备与训练流程说明。

## 环境建议

不同音频工具对 PyTorch、CUDA、NumPy 和音频依赖的要求不同，建议按用途使用独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-main.txt
```

高级音频清洗环境：

```powershell
pip install -r requirements-advanced.txt
```

音源分离环境：

```powershell
pip install -r requirements-audio-separator.txt
```

Demucs 旧版兼容环境：

```powershell
pip install -r requirements-demucs-legacy.txt
```

如需 GPU，请按本机 CUDA 版本单独安装匹配的 `torch`、`torchaudio` 和 `torchvision`。

## 常用入口

Web 程序：

```powershell
python web.py
```

原工具箱：

```powershell
python demo_toolbox.py -d <datasets_root>
```

本地数据流水线：

```powershell
python demo/pipeline.py
python demo/pipeline.py --url <video_or_audio_url>
python demo/pipeline.py --keyword "<keyword>" --n 5
```

数据校验：

```powershell
python tools/validate_dataset.py <dataset_or_synth_dir>
```

## GitHub 提交范围

建议提交：

- Python 源码、PowerShell/BAT 启动脚本。
- `requirements-*.txt`、Docker 相关文件、VS Code 通用配置。
- `README*.md`、`TRAINING_GUIDE.md`、本文件等说明文档。

不建议提交：

- `.env`、`.venv*`、`venv/`、`env/`。
- `saved_models/`、`pipeline_temp/`、`mockingbird_dataset/`、`dataset_build/`、`datasets/`、`references/`、`base/`、`temp_audio/`。
- `.pt`、`.pth`、`.ckpt`、`.onnx`、`.wav`、`.mp3`、`.jsonl` 等模型、音频和中间数据。

这些忽略规则已经写入 `.gitignore`，避免误把本地大文件或敏感配置推到 GitHub。
