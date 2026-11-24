# 实验记录与对照

以下命令均从仓库根目录执行。

### PF 单类检测的 P5 训练对照

PF 的一对一分支用物体分数与框的重合度选择正样本。默认配方将偏置的预热起始学习率设为 `0.1`。在 COCO 20k 对照中，可以单独降低步长为 32 的 P5 单类输出偏置的预热学习率。正式数据配置包含预热，小样本 smoke 配置跳过预热。

```bash
mkdir -p runs/pf-slow-bias/TP
cp runs/coco20k/TP/best.npz runs/pf-slow-bias/TP/best.npz
python train.py --device cpu --init init --data data/coco20k --out runs/pf-slow-bias \
  --stages PF --batch 8 --pf-p5-bias-warmup-lr 0.01
```

另一项独立对照使用 `--pf-p5-assignment tp-vocab`。它在训练时以同一份 TP 父权重对 80 个训练类别的最高分参与 P5 正样本分配，单类物体头继续接受分类损失的梯度。验证和推理使用训练好的单类头。两项设置分别在各自的 PF 运行目录中记录，正式 COCO 20k 命令会跑 PF 的全部 10 轮。

在同一份修订版 TP 权重上，PF 第 1 轮使用 20,000 张训练图，共 2,500 批、1,369 次更新。按训练时的 COCO 5,000 张矩形验证口径，原配方的单类框 AP50–95 为 40.292，词表引导为 44.823。两组对照共享冻结特征，分别用 Jittor 训练检测头、重载整模并验证。仅将 P5 输出偏置预热起始学习率降到 `0.01` 时，独立的纯 Jittor 训练入口得到 44.917。固定 48 张图上，旧、新 TP 的 P5 正样本位置从第 325 批起持续分歧，从第 364 批起新 TP 原配方给参考格点左邻的分数持续偏高。这组结果覆盖 10 轮配方的第 1 轮。[实验配置、权重校验与分数轨迹](../verification/yoloe26-pf-epoch1.json)

同一 TP 父权重在 A800 上使用 `--pf-p5-bias-warmup-lr 0.01` 完成 10 轮 PF 训练，共 4,497 次更新。按逐轮验证选中第 4 轮权重，独立重载后在 COCO val2017 的 5,000 张图上测得单类框 AP50–95 为 45.046，原配方为 41.385，配对 PyTorch 基线为 45.248。COCO 80 类检索 AP50–95 分别为 41.117、41.120 和 41.195。[A800 十轮训练对照及权重校验值](../verification/yoloe26-pf-slow-bias-coco20k.json)

### 早期迁移验证（历史权重）

2026-11-22 在 A800 80 GB 的独立虚拟环境实测。该环境中 `torch`、`torchvision` 和 `ultralytics` 均未安装。机器上的历史基线环境用于提供对照结果。所有记录见 [A800 验证结果](../verification/a800-native.json)。

| 检查 | 实测结果 |
| --- | --- |
| 原始 COCO → 两种标签 | 200 训练图及 200 验证图，两份清单和 800 个标签文件与旧流程逐字节相同 |
| 640 像素增强 | 6 个固定随机输入覆盖启用和关闭 Mosaic，图像、类别、框、掩码、视觉提示和文本索引通过对照 |
| 四阶段训练 | 各更新 25 步。TP / VP / PF / SEG 分别有 394 / 62 / 12 / 68 个可训练张量改变，全部成功保存，在新进程重载 best 后验证通过 |
| 同一份正式 TP 权重，COCO val2017 全部 5,000 张 | 原生验证器框 mAP50–95 为 41.245724，旧验证器为 41.245723，按 0–100 计 |
| 开放词汇 | LVIS 全部 1,203 个类别的文本向量与参考编码器最大绝对差 3.28×10⁻⁷。中英文、标点及超长文本的分词对照通过 |
| 新提示检测 | COCO 图像 000000039769 上输入 `tabby cat`，置信度 0.372，输出原图坐标检测框 |
| LVIS minival | 先通过 12 图预检，再用正式 TP 权重独立编码、推理和计分全部 4,809 张图。同一份 Jittor 权重在独立版得到 12.849377 Fixed AP，旧版为 12.849376。单独训练的 PyTorch 基线为 13.273002 |

上表记录开发期间的历史验收。正式独立重训和最终发布权重的成绩列在[主 README](../README.md)。`check_native.py` 可以直接读取旧实验目录的 `verified.json`，重新计算相同检查点的指标。

### LVIS minival 与双框架对照

`lvis_tp.py` 在原实验目录布局下使用作者的 4,809 张 minival 图像、1,203 个类别、640 输入、端到端 one-to-one 分支，每图最多 1,000 个候选，置信度阈值 0.001，整个数据集每类保留最多 10,000 个预测，调用 LVIS API 计算 bbox Fixed AP。图像与 COCO 20k 训练集重合 0 张，与选择最佳权重所用的 COCO 验证集重合 4,809 张。输入文件需放入 `lvis/inputs/`。脚本会核对原始文件校验和。[作者清单](https://raw.githubusercontent.com/THU-MIG/yoloe/main/tools/lvis/minival.txt) · [类别词表](https://raw.githubusercontent.com/THU-MIG/yoloe/main/ultralytics/cfg/datasets/lvis.yaml) · [标注](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json)

```bash
python lvis_tp.py prepare --base "$EXPERIMENT"
python lvis_tp.py infer --base "$EXPERIMENT" --engine jittor --smoke
python lvis_tp.py score --base "$EXPERIMENT" --engine jittor --smoke
```

去掉 `--smoke` 即使用全部 minival 图像。默认布局从 `jittor/full/TP` 读取原始正式权重，从 `jittor/init` 读取配套初始化，从 `work/yoloe-runtime/assets` 读取冻结编码器。已发布结果的复测应使用独立实验目录保存新的文本缓存和预测。双框架对照的 PyTorch 分支使用其自己的基线环境。

### LVIS 同义提示对照

这组推理实验固定 COCO 20k 独立版 TP 最佳权重，用同一份 LVIS 清单比较首名称提示与同义提示。先将每个类别的名称去重，用冻结的 MobileCLIP2-B 编码别名，再将各名称的单位向量等权平均、归一化，最后通过原模型的 ReP-RTA 文本分支。462 类有不同别名，另外 741 类直接复用基线缓存中的向量。

在已准备好原生首名称缓存的实验上，可以重复以下步骤。`JT` 和 `SCORE` 分别指向 Jittor 推理和 LVIS 计分环境的 Python。

```bash
inputs="$EXPERIMENT/lvis/inputs"
output="$EXPERIMENT/lvis-synonyms"
"$JT" build_synonym_cache.py --manifest "$inputs/manifest.json" --vocab "$inputs/lvis.yaml" \
  --encoder mobileclip2_b.ts --output "$output/alias-text.npy" --device cpu
"$JT" lvis_tp.py infer --base "$EXPERIMENT" --engine jittor --device cpu --workers 3 \
  --text-cache "$output/alias-text.npy" --output "$output/results"
"$SCORE" lvis_tp.py score --base "$EXPERIMENT" --engine jittor --output "$output/results"
```

在 4,809 张 minival 图上，这份权重的 bbox Fixed AP 从 13.2837 降至 13.0052。实际有标注的 315 个多名称类别平均下降 0.7216 个 AP 点，489 个单名称类别平均上升 0.0069 个点。完整指标、分组和权重校验值见[同义提示对照结果](../verification/yoloe26-tp-lvis-synonyms.json)。

### 使用同一份官方 26s 权重验证迁移精度

使用 [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) 发布的 `yoloe-26s-seg.pt` 和 `mobileclip2_b.ts`。导出工具核验两份权重的 SHA-256，逐个对照检测分支的参数名、形状和类型。PyTorch 评测会打开公开权重的 one-to-one 检测头并从分割输出中提取检测框。Jittor 从同一份权重恢复检测和提示分支。两个环境分别编码 1,203 个类别。

以下 Bash 命令在项目目录执行。`PT`、`JT`、`SCORE` 分别是参考、原生和 LVIS 计分环境的 Python 路径，`ASSETS` 是官方权重目录，`LVIS` 包含上述三份基准文件，`VAL` 指向 COCO `val2017` JPEG 目录，`RUN` 是本次实验的输出目录。

```bash
"$PT" official_tp_export.py --weights "$ASSETS/yoloe-26s-seg.pt" --init init \
    --text-model "$ASSETS/mobileclip2_b.ts" --lvis-yaml "$LVIS/lvis.yaml" --out "$RUN/export"
args=(--assets-dir "$LVIS" --images-dir "$VAL" --manifest "$RUN/manifest.json" --output "$RUN/results")
"$PT" lvis_tp.py prepare "${args[@]}" --native-text
"$JT" text_encoder.py --device cpu --names "$RUN/manifest.json" \
    --weights "$ASSETS/mobileclip2_b.ts" --out "$RUN/jittor-text.npy"
"$PT" lvis_tp.py infer "${args[@]}" --engine pytorch --device cuda:0 \
    --weights "$ASSETS/yoloe-26s-seg.pt" --text-cache "$RUN/export/pytorch-text.npy"
"$JT" lvis_tp.py infer "${args[@]}" --engine jittor --device cpu --workers 3 --init-dir init \
    --weights "$RUN/export/official-tp.npz" --text-cache "$RUN/jittor-text.npy"
"$SCORE" lvis_tp.py score "${args[@]}" --engine pytorch --weights "$ASSETS/yoloe-26s-seg.pt"
"$SCORE" lvis_tp.py score "${args[@]}" --engine jittor --weights "$RUN/export/official-tp.npz"
"$SCORE" lvis_tp.py compare "${args[@]}"
```

在 4,809 张 LVIS minival 图上，公开权重重新测得 PyTorch 30.220828、Jittor 30.220837 bbox Fixed AP。运行时可在 `results/full/comparison.json` 查看双框架差值，本文对应结果含权重、词表及预测文件校验和。[本地同权重复测](../verification/yoloe26-official-lvis.json)

论文中 26s 的参照值为 29.9 bbox AP。用 COCO 20k 重训的 TP 权重有单独的实验记录。[YOLO26 技术报告表 S12](https://arxiv.org/html/2606.03748v1) · [历史 COCO 20k 权重的 LVIS 结果](../verification/lvis-original-comparison.json)

训练对照中，`train.py --device cpu` 可在本地运行同一训练入口。修订版按官方 MuSGD 参数组选择正交化方向，SEG 的 `proto.cv3` 使用常规学习率，CPU 训练使用最近偶数 BF16 舍入，并提高 BN 统计和梯度范数的累加精度。重新训练时使用新输出目录，以便保留历史权重和结果。单步梯度、实际参数增量和固定梯度实验分别见 [26s 真实训练单步](../verification/yoloe26-cpu-tp-step.json) 与 [MuSGD 消融](../verification/yoloe26-cpu-muon-ablation.json)。

数据增强和几何代码沿用 Ultralytics 8.4.216 的 NumPy/OpenCV 实现，适用 [AGPL-3.0](../LICENSE)。分词器与词表来自 OpenAI CLIP，见 [MIT 许可](../LICENSE-CLIP)。方法和评测来源见 [YOLOE 论文](https://arxiv.org/abs/2503.07465)、[Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/)、[作者的 Fixed AP 评估脚本](https://github.com/THU-MIG/yoloe/blob/main/tools/eval_fixed_ap.py) 及 [Jittor](https://github.com/Jittor/jittor)。
