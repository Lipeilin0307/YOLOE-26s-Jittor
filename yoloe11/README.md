# YOLOE-11s 文本提示检测

本目录复用项目的 Jittor 图算子、COCO 加载器和 AP 评估器，支持 11s 的单分支检测头。分类使用原版 MobileCLIP-B(LT) 和 RepRTA，框回归使用每边 16 档分布及 DFL，验证使用按类别的 NMS。初始化来自 Ultralytics v8.4.0 发布的未融合 `yoloe-11s-seg.pt`，导出时提取其中的检测和文本提示参数。支持范围为 TP 检测。

以下命令从本项目根目录 `YOLOE-26s-Jittor` 执行。CPU 和 CUDA 由 `--device` 选择，本次验收在本地 WSL 的 CPU 上完成。分辨率 320，batch 4，固定种子 20261121。正式权重和文本模型可从[官方发布页](https://github.com/ultralytics/assets/releases/tag/v8.4.0)下载，文件 SHA-256 在导出时核验。

### 一次性转换与参考输出

`REF` 指向安装 Ultralytics 8.4.216、PyTorch 和 OpenAI CLIP 的参考 Python，`JT` 指向安装本项目依赖的独立 Jittor Python。独立训练和推理均使用 NumPy 权重。示例使用主 README 的 prepare.py --profile smoke 生成的 200 张训练图和 200 张验证图。

```bash
ASSETS=assets/yoloe11
INIT=runs/yoloe11/init
DATA=data/smoke200
OUT=runs/YOLOE-11s-Jittor-TP-smoke
PARITY=verification/yoloe11-parity.json
mkdir -p "$ASSETS"
for name in yoloe-11s-seg.pt mobileclip_blt.ts; do
  test -f "$ASSETS/$name" || curl -fL -o "$ASSETS/$name" "https://github.com/ultralytics/assets/releases/download/v8.4.0/$name"
done
"$REF" -m yoloe11.export --assets "$ASSETS" --data "$DATA" --out "$INIT"
"$JT" -m yoloe11.parity --init "$INIT" --report "$PARITY"
```

参考模型在两张真实 COCO 图片上生成不同尺寸和词表的输出。验证逐项对齐文本向量、RepRTA、P3/P4/P5、多档回归和分类 logits、所有解码框，以及置信度阈值 0.001、IoU 阈值 0.7 后的多标签 NMS 结果。COCO 训练缓存由 Jittor 重新编码全部 80 个类别并与参考输出逐项比较。

### 单轮训练与独立重载

```bash
"$JT" -m yoloe11.train --data "$DATA" --init "$INIT" --parity "$PARITY" --out "$OUT"
"$JT" -m yoloe11.train --data "$DATA" --init "$INIT" --parity "$PARITY" --out "$OUT" --verify-only
```

训练使用 SGD、动量 0.9、学习率 0.0005 和梯度范数上限 10。程序检查图像网络、回归分支、分类分支和文本投影均有参数变化。首次运行写入 `tp-epoch1.npz` 和 `epoch1.json`。第二条命令在新进程校验检查点哈希，重新计算 200 张验证图的指标并写入 `reloaded.json`。本实验检验单轮微调和保存重载，正式精度需用更大验证集和训练设置评估。

### 自定义类别推理

```bash
"$JT" -m yoloe11.predict --init "$INIT" --weights "$OUT/tp-epoch1.npz" \
  --image "$DATA/detect/images/val/000000000139.jpg" --classes chair person "potted plant"
```

输出包含类别名称、置信度和原图像素坐标。省略 `--weights` 使用官方初始检测权重。

### 本地验收结果

两张图分别使用不同的文本词表，NMS 后得到 63 和 46 个检测框。两个实现的类别和顺序逐项一致，坐标最大绝对差为 0.000214 像素，置信度最大绝对差为 0.00000161。[逐项对齐记录](../verification/yoloe11-parity.json)

固定 200 张训练图跑完 1 轮，共更新 50 步，271 个可训练张量中有 266 个发生变化。表中两列使用同一行的权重、另 200 张固定验证图和同一套检测指标，数值为 mAP50–95，满分 100。

| 权重 | 官方 PyTorch 实现 | Jittor 实现 |
| --- | ---: | ---: |
| 官方预训练权重 | 35.07715 | 35.07718 |
| Jittor 训练一轮的权重 | 23.19321 | 23.19320 |

Jittor 在新进程重载后的精确率、召回率、mAP50 和 mAP50–95 均与保存前完全相同。本次微调后 mAP50–95 下降 11.884 个点，两种实现均复现了降幅。[训练与复测记录](../verification/yoloe11-smoke.json)

在同一验证集上复测官方实现，可运行以下两条命令。

```bash
"$REF" -m yoloe11.check_validation reference --init "$INIT" --assets "$ASSETS" --data "$DATA" --out "$OUT/reference-init.json"
"$REF" -m yoloe11.check_validation reference --init "$INIT" --assets "$ASSETS" --data "$DATA" --weights "$OUT/tp-epoch1.npz" --out "$OUT/reference-epoch1.json"
```

### 官方 PyTorch 单轮训练对照

使用 Ultralytics 8.4.216 的 YOLOE 模型和 `v8DetectionLoss` 计算 TP 检测损失。输入由上次实验所用的 Jittor 环境先生成一次，各框架逐批读取相同的图像、检测标签和文本提示顺序。两个训练过程都使用固定的文本编码和 271 个可训练张量。SGD 的学习率为 0.0005，动量为 0.9，梯度范数上限为 10，每个 batch 更新一次，没有权重衰减、warmup 或 EMA。验证统一使用 200 张独立图像。

| 训练实现 | 平均 box loss | 平均 cls loss | 平均 DFL | 单轮 mAP50 | 单轮 mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch | 1.32785 | 1.82741 | 1.19338 | 42.5285 | 30.1198 |
| Jittor | 1.34973 | 1.86765 | 1.20180 | 32.9246 | 23.1932 |

从约 35.0772 的共同起点出发，PyTorch 和 Jittor 分别下降 4.9573 和 11.8840 个点。固定各自单轮训练后的参数，清空 BN 统计量，在同一官方模型中用训练集的 50 个 batch 重新估计均值与方差后，两组权重分别得到 30.7601 和 29.2000 AP。固定预训练参数、按原动量更新 BN，两种框架分别得到 34.2491 和 34.2510 AP。

在首批使用相同框分布 logits、分类 logits 和训练标签时，两套 loss 对这两组 logits 的梯度相对 L2 误差都小于 0.000003。单独比较各框架端到端反传时，全模型梯度相对 L2 误差为 1.11%。这些结果说明本次验证精度对 BN 统计量敏感。剩余差距的具体来源还需要逐步比较更新轨迹。[逐步 loss、验证与诊断记录](../verification/yoloe11-training-comparison.json)

```bash
REPLAY=runs/yoloe11/training-replay
REF_OUT=runs/YOLOE-11s-PyTorch-TP-smoke
"$JT" -m yoloe11.training_replay --data "$DATA" --run "$OUT/epoch1.json" --out "$REPLAY"
"$REF" -m yoloe11.train_reference --assets "$ASSETS" --init "$INIT" --replay "$REPLAY" --jittor-run "$OUT/epoch1.json" --out "$REF_OUT"
"$REF" -m yoloe11.check_bn --assets "$ASSETS" --init "$INIT" --replay "$REPLAY" --pytorch "$REF_OUT/tp-epoch1.pt" --jittor "$OUT/tp-epoch1.npz" --out "$REF_OUT/bn-recalibration.json"
"$JT" -m yoloe11.check_bn --native-frozen --init "$INIT" --replay "$REPLAY" --out "$REF_OUT/native-bn-only.json"
"$REF" -m yoloe11.check_training_step reference --init "$INIT" --assets "$ASSETS" --replay "$REPLAY" --fixture "$REF_OUT/reference-first-step.npz"
"$JT" -m yoloe11.check_training_step jittor --init "$INIT" --replay "$REPLAY" --fixture "$REF_OUT/reference-first-step.npz" --report "$REF_OUT/training-probe.json"
```

`tp-epoch1.pt` 是检测和文本提示分支的 PyTorch state dict。独立加载时先构建官方预训练模型，再用 `load_state_dict(..., strict=False)` 加载这份权重。BN 诊断只操作内存中的模型副本。

### CPU 训练逐步对齐

11s 在 CPU 训练时使用 float64 累加 BN 的中心矩和反向归约，以及全模型梯度的平方和。图像除以 255 后舍入到 float32。卷积、可训练参数、BN 缓冲区、SGD 动量和检查点仍为 float32。旧版检查点在 `--verify-only` 时沿用其原来的图像归一化方式。

以下命令沿用上面的 `JT`、`REF`、`INIT`、`ASSETS`、`REPLAY`、`DATA` 和 `PARITY`，从同一官方权重重新训练。跟踪命令额外保存约 5 GiB 本地临时参考数组。对照只在诊断进程中读取这些数组。

```bash
ALIGNED=runs/YOLOE-11s-Jittor-TP-aligned
"$JT" -m yoloe11.train --data "$DATA" --init "$INIT" --parity "$PARITY" --out "$ALIGNED"
"$JT" -m yoloe11.train --data "$DATA" --init "$INIT" --parity "$PARITY" --out "$ALIGNED" --verify-only

CACHE=$(mktemp -d)
TRACE=(--init "$INIT" --replay "$REPLAY" --cache "$CACHE")
export OPENBLAS_NUM_THREADS=1
"$REF" -m yoloe11.trace_training reference "${TRACE[@]}" --assets "$ASSETS" --shared-start --report "$CACHE/reference.json"
"$JT" -m yoloe11.trace_training jittor "${TRACE[@]}" --legacy-bn --legacy-input --legacy-clip --report "$CACHE/old-free.json"
"$JT" -m yoloe11.trace_training jittor "${TRACE[@]}" --stable-clip --report "$CACHE/new-free.json"
"$JT" -m yoloe11.trace_training jittor "${TRACE[@]}" --stable-clip --shared-start --report "$CACHE/new-controlled.json"
"$JT" -m yoloe11.trace_training jittor "${TRACE[@]}" --shared-start --reference-gradients --legacy-clip --report "$CACHE/old-optimizer.json"
"$JT" -m yoloe11.trace_training jittor "${TRACE[@]}" --stable-clip --shared-start --reference-gradients --report "$CACHE/new-optimizer.json"
"$REF" -m yoloe11.trace_layers reference "${TRACE[@]}" --assets "$ASSETS"
"$JT" -m yoloe11.trace_layers jittor "${TRACE[@]}" --report "$CACHE/new-layers.json"
```

连续训练追踪每一步裁剪前的梯度、实际参数增量、BN 缓冲区及其当步增量。`--shared-start` 在第 1、2、3、4、5、10、20、50 步分别加载该步官方的参数、BN 缓冲区和动量，检验每一步自身的误差。`--reference-gradients` 进一步固定裁剪前的梯度，用于单独检验裁剪和 SGD。逐层实验将官方算子实际收到的输入和上游梯度交给原生算子，再比较输出和输入梯度。读取中间张量会改变 Jittor 的算子融合，因此最终 AP 以普通训练入口独立运行的检查点为准。

首批输入的像素归一化修正后与参考逐值一致。首层 BN 在相同输入上的输出相对 L2 误差从 1.30e-4 降到 5.25e-8。对 8 个相同起点取中位数，全模型梯度误差从 0.844% 降到 0.00496%，实际参数增量误差从 0.883% 降到 0.00964%，BN 当步增量误差从 0.0460% 降到 0.000939%。

直接从官方初始化独立训练一轮，官方 PyTorch、原版 Jittor、只修 BN 的 Jittor、完整修复的 Jittor 分别为 30.120、23.193、18.813、18.575 mAP50–95。修复版检查点独立重载得到相同指标。仅在第 1 步结束后把官方模型的首层卷积权重和相应动量替换为修复版 Jittor 的数值，余下 49 步完全交给 PyTorch，结果为 20.613。此处首层权重最大绝对改变量相对于原官方结果仅为 7.67e-8。该对照证明本次训练对首步差异敏感。

第 3 步在相同起点上比较时，SPPF (`model.9`) 第二次最大池化有 3 个输出选择了不同位置。同一池化输入产生的索引完全相同。在诊断中固定为官方的反传位置，整网梯度相对误差从 0.128% 降至 0.00234%。另一次诊断固定单轮权重，并仅用 200 张训练图重估 BN，官方 PyTorch 与修复版 Jittor 权重分别得到 30.760 和 29.360 mAP50–95。这些重估结果单独记录，提交的检查点来自原训练流程。[对齐报告](../verification/yoloe11-alignment.json) · [50 步逐层数值](../verification/yoloe11-layer-traces.json) · [误差曲线](../verification/yoloe11-training-drift.png)

```bash
"$JT" -m yoloe11.stem_control capture --init "$INIT" --replay "$REPLAY" --stem "$CACHE/stem.npz"
"$REF" -m yoloe11.stem_control roll-out --init "$INIT" --assets "$ASSETS" --replay "$REPLAY" --stem "$CACHE/stem.npz" --report "$CACHE/stem-control.json"
"$REF" -m yoloe11.check_pool reference "${TRACE[@]}" --assets "$ASSETS" --fixture "$CACHE/pool-step3.npz"
"$JT" -m yoloe11.check_pool jittor "${TRACE[@]}" --fixture "$CACHE/pool-step3.npz" --out "$CACHE/pool-step3.json"
```

### 两种学习率的配对验收

固定 200 张训练图和 200 张验证图、官方初始化、数据增强、batch 4、320 像素、50 次全网 SGD 更新。学习率分别为 0.0005 与 0.00005，每组在种子 20261121、20261122、20261123 上各跑一次。种子控制训练样本次序和增强，验证图及处理方式保持一致。同一个种子下，每个框架、每种学习率使用相同的训练张量。主指标为单轮原始检查点的检测 mAP50–95。

本次工程筛查预设每一对的绝对差距不超过 1 个 AP 点，同时检查骨干、回归分支、分类分支与文本投影的参数更新。三个种子共享一个验证子集，因此验收范围是这套 200 张单轮实验。结果同时报告原始预训练 AP、仅更新 BN 的 AP 和可训练参数实际变化量。

```bash
RUNS=runs/YOLOE-11s-TP-paired
PREV_J=runs/YOLOE-11s-Jittor-TP-aligned
PREV_P=runs/YOLOE-11s-PyTorch-TP-smoke
for seed in 20261121 20261122 20261123; do
  replay=runs/yoloe11/paired/replay-$seed
  "$JT" -m yoloe11.training_replay --data "$DATA" --run "$PREV_J/epoch1.json" --out "$replay" --seed "$seed"
  for lr in 0.00005 0.0005; do
    if [[ "$seed" == 20261121 && "$lr" == 0.0005 ]]; then continue; fi
    pair="$RUNS/$seed/lr-$lr"
    "$JT" -m yoloe11.train --data "$DATA" --init "$INIT" --parity "$PARITY" --out "$pair/jittor" --lr "$lr" --seed "$seed"
    "$REF" -m yoloe11.train_reference --assets "$ASSETS" --init "$INIT" --replay "$replay" --jittor-run "$pair/jittor/epoch1.json" --out "$pair/pytorch"
  done
done
"$REF" -m yoloe11.paired_report --init "$INIT" --runs "$RUNS" --previous-native "$PREV_J" --previous-reference "$PREV_P" --report verification/yoloe11-paired-accuracy.json
```

种子 20261121、学习率 0.0005 的一对复用已保存的 PyTorch 和修复版 Jittor 结果。其余五对分别独立训练。汇总程序核对数据、权重哈希、参数变化和全部六对原始指标。

| 学习率 | PyTorch 平均 AP | Jittor 平均 AP | 配对绝对差均值 | 最大绝对差 | 差距在 1 AP 内 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.0005 | 24.720 | 21.059 | 5.987 | 11.545 | 0/3 |
| 0.00005 | 32.605 | 32.558 | 0.369 | 0.527 | 3/3 |

两种学习率下每次训练均更新了 271 个可训练张量中的 266 个。较低学习率通过了本次单轮框架对齐筛查，两边平均 AP 均低于预训练起点的 35.077。原学习率与更大规模训练需独立评价。[各个种子的 AP、训练损失、更新幅度和数据哈希](../verification/yoloe11-paired-accuracy.json)
