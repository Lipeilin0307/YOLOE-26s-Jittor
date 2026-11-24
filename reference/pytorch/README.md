# YOLOE-26s 的 PyTorch 四阶段训练

这个工程为后续的 Jittor 对照提供 PyTorch 基线。四阶段按 YOLO26 论文附录 S6 组织。TP 从 Objects365-v1 预训练的闭集检测器开始，VP、PF、SEG 分别读取本次训练得到的同一个 TP best.pt。四个阶段的检测与分割监督均来自 COCO 2017。[论文附录 S6](https://arxiv.org/html/2606.03748v1#A6) · [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)

| 阶段 | 训练内容 | 验证口径 |
|---|---|---|
| TP | MobileCLIP2 编码类别名，训练检测器和 RepRTA 文本对齐模块 | COCO 80 类框 AP |
| VP | 通过训练标注框生成视觉提示，更新 SAVPE 和 cv4 对比头 | 用训练集构建视觉类别原型，在留出图上测框 AP |
| PF | 将所有训练目标合为 object，更新分类塔末端投影 | 单类目标框 AP；另将 TP 词表与 PF 提案层合成为 LRPC 权重，测无需输入提示的 80 类框 AP |
| SEG | 载入 TP 检测权重，训练 cv5 掩膜系数和 proto 原型分支 | 文本提示下的框 AP 与实例掩膜 AP |

## 数据与实验设置

| 配置 | 训练图 | 验证图 | TP / VP / PF / SEG 轮数 | 默认输入尺寸 |
|---|---:|---:|---|---:|
| smoke | 200 | 200 | 1 / 1 / 1 / 1 | 320 |
| full | 20,000 | 5,000 | 30 / 10 / 10 / 10 | 640 |

`splits/` 随工程附带原定的 2 万张训练清单和 5,000 张验证清单，与仓库根目录公开的划分一致。`train200.txt` 和 `val200.txt` 固定了冒烟实验的图像。训练样本从已到位且属于 2 万张清单的图片中选取，并覆盖全部 80 类。验证样本从原定 val2017 清单独立抽取。数据准备会检查子集成员关系，并输出清单哈希。检测路径使用 COCO 框，分割路径使用实例多边形，跳过 crowd 标注，多片多边形按 Ultralytics 转换器合并。

两种配置都使用 COCO 类别名与 COCO 人工标注。论文使用的 grounding 文本、教师伪标签、4,585 类词表与 LVIS 零样本评估不在本实验范围内。单卡默认 batch 为 8，冒烟配置使用 FP32，关闭 Mosaic 与学习率预热，每个 batch 执行一次优化。完整配置也默认使用 FP32。两种配置均验证 YOLO26 的端到端分支。

## 环境与权重

实测环境为 WSL2、Python 3.10、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、Ultralytics 8.4.216 和一张 RTX 5090。下面的命令从本工程目录执行，适用于 Linux 或 WSL 的 Bash。

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

ASSETS=assets
mkdir -p "$ASSETS"
for file in yolo26s-objv1-150.pt mobileclip2_b.ts; do
  test -f "$ASSETS/$file" || curl -fL --retry 3 -o "$ASSETS/$file" \
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/$file"
done
```

## 运行

`COCO_ROOT` 下需要有 `annotations/instances_train2017.json`、`annotations/instances_val2017.json` 和对应的图像目录。若图片单独存放，可追加 `--image-root`，指向包含 `train2017/` 或 `val2017/` 的父目录。

```bash
COCO_ROOT=/data/coco2017
DATA=data/smoke200
RUN=runs/yoloe26s-smoke

# 已经准备好的 DATA 可直接用于下面两个入口
python prepare.py --profile smoke --coco-root "$COCO_ROOT" --out "$DATA"
python train.py --data "$DATA" --assets "$ASSETS" --out "$RUN"
python verify.py --data "$DATA" --assets "$ASSETS" --out "$RUN"
```

以上命令从本目录执行。运行目录保存每阶段的权重和重新加载后的验证结果。

切换到 2 万张时，将数据准备命令改为 `--profile full`，为 `DATA` 与 `RUN` 使用新目录，并提供 2 万张训练图与 5,000 张验证图。训练入口从数据目录读取配置与轮数。准备数据时优先使用硬链接，跨文件系统时复制图片。已有 TP 后可以用 `--stages VP PF SEG` 单独启动尚未执行的分支。训练入口会保护已有的阶段目录。

## 产物与验收

- 每个阶段的 `weights/best.pt`、`weights/last.pt`、`results.csv` 和 `args.yaml` 分别记录权重、逐轮指标和配置。
- `verified.json` 记录父权重及 best 权重的 SHA-256、实际更新的参数名、优化步数、训练损失与保存后验证指标。训练期间检查冻结参数保持一致。
- PF 的 `weights/prompt-free-coco80.pt` 提供完整的无提示类别检索推理。PF 的 `best.pt` 对应单类 object 训练头。
- `verify.py` 在单独进程读取四组 last 和 best 权重，核对父权重哈希，分别验证四个 best 及 PF 的 80 类导出权重。结果写入运行目录的 `summary.json`。所有 AP 以 0–1 为单位；后续框架对照应固定清单、输入尺寸、提示来源、词表与推理分支。
