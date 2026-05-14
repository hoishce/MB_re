# 数据准备与训练指南

这份指南记录本仓库中用于 MockingBird 语音克隆训练的本地流程。仓库只提交代码、脚本和说明文档；音频、数据集、模型权重和流水线输出请放在被 `.gitignore` 忽略的目录中。

## 训练目标

使用单说话人语音数据微调或训练 MockingBird 的 synthesizer，并配合 encoder、vocoder 或 HiFi-GAN 完成推理。

## 数据要求

- 推荐音频格式：16 kHz、单声道、WAV。
- 推荐切片长度：1 到 15 秒，尽量避免长静音、重叠说话和明显背景音乐。
- 推荐数据量：10 到 30 分钟可用于快速试验；1 小时左右可以得到基础音色；10 小时以上更适合稳定训练。
- 每条音频需要准确文本，文件名与文本记录必须一一对应。

## 数据准备

长音频带时间戳文本时，可以使用本仓库的辅助脚本切分并生成训练目录：

```powershell
python tools/prepare_and_preprocess.py `
  --audio <converted_wav> `
  --transcript <timestamped_txt> `
  --out-dataset <datasets/minidataset> `
  --out-synth <saved_models/synth_minidataset> `
  --encoder-ckpt <encoder_ckpt>
```

生成后的 synthesizer 训练目录通常包含：

```text
saved_models/synth_minidataset/
  audio/
  mels/
  embeds/
  train.txt
```

## 数据校验

训练前建议运行校验脚本，检查 `train.txt`、音频文件和预处理产物是否完整：

```powershell
python tools/validate_dataset.py saved_models/synth_minidataset
python tools/validate_dataset.py datasets/minidataset
```

## 启动训练

PowerShell 入口：

```powershell
.\tools\run_synth_train.ps1 `
  -run_id myrun `
  -syn_dir saved_models/synth_minidataset `
  -models_dir saved_models/synth_finetune
```

Python 入口：

```powershell
python train.py --type synth <run_id> <syn_dir> -m <models_dir>
```

更多训练参数请参考 `control/cli/synthesizer_train.py`。

## 小数据微调建议

- 从较小学习率开始，先做短轮次试验，确认 loss、样本音频和文本对齐正常。
- 少量数据时优先冻结部分层，只训练 decoder、postnet 等后段模块。
- 训练产物、日志和权重建议放在 `saved_models/`，该目录默认不会提交到 GitHub。

## 推理验证

训练过程中保存的 checkpoint 可以用于快速推理验证：

```powershell
python demo/mockingbird-01.py --syn <path_to_new_ckpt>
```

如果只想跑 Web 或工具箱入口，请参考 `README-CN.md` 中的启动说明。
