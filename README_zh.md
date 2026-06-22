# PCASTNet

PCASTNet 是面向跨机器小样本故障诊断样本生成的物理约束自适应风格迁移网络。

本目录是 CWRU -> BJTU 实验的 reset 版 demo 工程。工程目标是保持实验口径清晰：
固定 CWT 数据、固定样本数量、一个 Python 入口、阶段化输出、日志和产物集中保存在同一个
run 目录下。

## 方法口径

PCASTNet 处理的问题是：被监测机器只有少量带标签故障样本，而参考机器有更多历史数据。方法
不是从随机噪声生成样本，而是把参考机器中的故障内容迁移到被监测机器的机器风格上。

论文中的语义定义如下：

- content：跨机器应保持一致的故障判别性谱结构；
- style：由机器结构、传感器、载荷、转速等差异带来的机器特有时频外观；
- physical consistency：生成的 CWT 样本应保持被监测机器的频带能量分布。

流程基于 CWT 时频图，包含：

1. 基于 VGG 的谱特征编码器，使用 AdaIN 的 normalized VGG 权重初始化；
2. Adaptive Style Normalization 模块，用于内容和风格融合；
3. 解码器，用于重建目标机器风格下的 CWT 样本；
4. 多目标损失：

```text
L = 1 * L_content + 10 * L_style + 1 * L_energy
```

其中 band energy loss 是物理约束项，用来约束频率方向上的能量分布。

## Demo 实验口径

当前配置对应 CWRU -> BJTU 小样本实验：

| 角色 | 数据集 | 数量 | 用途 |
| --- | --- | ---: | --- |
| reference/content train | CWRU | 500 | 作为生成阶段的内容来源 |
| monitored/style train | BJTU | 50 | encoder 训练、风格迁移、分类器真实样本 |
| monitored/style valid | BJTU | 100 | 验证 |
| monitored/style test | BJTU | 500 | 最终测试 |

encoder 训练只使用 50 张 BJTU monitored train 样本，并在内存中按 8:2 划分训练/验证集。
不会在 `data/` 下生成任何 split 数据集。

## 项目结构

```text
reset/
  configs/
    experiments/
      cwru_bjtu.json
  data/
    datasets/
      machines/
        CWRU/cwts/
        BJTU/cwts/
  demo.py
  src/
    pcastnet/
    models/
    STC.py
    data_loader.py
    function.py
    sampler.py
    stc_*.py
  experiments/
  requirements.txt
  pyproject.toml
```

生成样本和训练模型都属于 run 输出，保存在 `experiments/`，不作为固定数据放在 `data/`。

## 环境

本 demo 是 PyTorch/CUDA 实验。部分检查可以在 CPU 上运行，但完整的 encoder、
style-transfer、generate、classifier 流程应使用 NVIDIA GPU 运行。

推荐环境口径如下：

| 组件 | 要求 |
| --- | --- |
| 操作系统 | Windows 或 Linux |
| Python | 3.10 或更高 |
| GPU | 支持 CUDA PyTorch 的 NVIDIA GPU |
| 主要框架 | PyTorch + torchvision |
| 其他依赖 | numpy、pillow、matplotlib、tqdm、tensorboardX、scikit-learn |

当前最终 demo 在本机验证过的环境为：

| 组件 | 本机值 |
| --- | --- |
| Python | 3.12.8, Anaconda |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| PyTorch 报告的 CUDA runtime | 12.8 |
| GPU | NVIDIA GeForce RTX 4070 Ti |
| NVIDIA driver | 576.52 |

这些版本不是硬性锁死要求。关键要求是完整训练前
`torch.cuda.is_available()` 必须返回 `True`。

### 创建环境

使用 conda：

```powershell
conda create -n pcastnet python=3.12 -y
conda activate pcastnet
```

PyTorch 请按本机显卡驱动从官方命令安装。CUDA 12.x 环境示例：

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

然后安装项目和其余依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果已有可用环境，只需要补装缺失依赖：

```powershell
python -m pip install -e .
```

### 验证 GPU 和依赖版本

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi
```

如果第一条命令输出 `False`，需要先修正 PyTorch/CUDA 安装，再启动完整 demo。

## AdaIN VGG 权重

encoder 使用 AdaIN 的 normalized VGG 初始化。项目期望路径为：

```text
src/models/vgg/vgg.pth
```

下载命令：

```bash
mkdir -p src/models/vgg
wget -c https://s3.amazonaws.com/xunhuang-public/adain/vgg_normalised.t7 -O src/models/vgg/vgg.pth
```

上游文件名是 `.t7`，本项目保存为 `vgg.pth`，因为代码按 VGG encoder checkpoint 读取。

## 检查数据和配置

```powershell
python demo.py --dry-run --config configs/experiments/cwru_bjtu.json
```

dry-run 会检查：

- CWRU/BJTU CWT 数据路径；
- 固定样本数量；
- AdaIN VGG 权重路径；
- 输出路径；
- encoder 内存划分策略。

## 运行完整 demo

只使用 `demo.py` 作为 demo 入口：

```powershell
python demo.py --stage all --config configs/experiments/cwru_bjtu.json
```

也可以单独运行某个阶段：

```powershell
python demo.py --stage encoder --config configs/experiments/cwru_bjtu.json
python demo.py --stage style-transfer --config configs/experiments/cwru_bjtu.json
python demo.py --stage generate --config configs/experiments/cwru_bjtu.json
python demo.py --stage classifier --config configs/experiments/cwru_bjtu.json
```

长时间训练会把阶段输出写入 run 目录下的 `train.log`。

## 输出结构

完整运行会生成：

```text
experiments/pcastnet_cwru_bjtu_<timestamp>/
  config.json
  effective_config.json
  meta.json
  status.json
  encoder_manifest.json
  pipeline_manifest.json
  train.log
  flow1_CWRU-BJTU_encoder_c/
    *_encoder.pth.tar
    *_classifier.pth.tar
    *_confusion_matrix_*.png
  flow2_CWRU-BJTU_st/
    *_decoder.pth.tar
    *_adailn.pth.tar
    *_encoder.pth.tar
    *_iter_*.png
  generated/
    train/
    dataset_info.json
  downstream_cnn_CWRU-BJTU_c/
    *_confusion_matrix_*.png
    *_confusion_matrix_*.npy
    *_predictions_*.npz
    *_iter_*.pth.tar
```

encoder 不再复制到第二个 canonical checkpoint。encoder 阶段会写
`encoder_manifest.json`，其中的 `selected_encoder_path` 就是后续阶段直接读取的真实训练产物。

## 指标文件

分类器阶段会保存：

- 混淆矩阵图片：`*_confusion_matrix_*.png`；
- 混淆矩阵原始数组：`*_confusion_matrix_*.npy`；
- labels、predictions、softmax probabilities、accuracy、F1、AUC：
  `*_predictions_*.npz`。

AUC 使用 softmax 概率计算 macro one-vs-rest ROC-AUC。

## 注意事项

- 当前 demo 固定为论文口径下的 CWRU -> BJTU 实验。
- 不使用 `content-valid` 和 `content-test`。
- `data/` 只保留 demo 所需 CWT 固定数据。
- encoder 训练不落地 split 数据集，只做运行时内存划分。

## License

发布或分发代码和数据前，请先查看 [LICENSE.md](LICENSE.md)。
