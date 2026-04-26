# Multimodal Time-series Fusion Pipeline 总结

## 1. 项目目标

本项目当前目标是做材料类别分类。每个样本来自一条双通道时序信号：

```text
signal shape = [2, 1024]
channel 0 = high
channel 1 = low
class = cloth / leather / metal / wood
```

当前方法不是严格意义上的多传感器多模态，而是把同一条时序信号构造成两种视图：

```text
1. 原始时序视图：signal
2. 曲线图像视图：image
```

因此更准确的描述是：

```text
时序信号 + 时序图像编码的多视图联合分类
```

当前代码已经完成基础分类 pipeline，diffuser 还没有实现，目前只作为后续数据增强路线规划。

## 2. 当前整体 Pipeline

当前已经跑通的流程如下：

```text
yaml config
-> MultiModalDataset
-> DataLoader
-> MultiModalClassifier
-> CrossEntropyLoss
-> AdamW
-> train / validate
-> best.ckpt
```

训练入口支持：

```text
python test_train.py -c mtf/config/config.yaml
```

训练脚本当前包含：

- 命令行读取配置文件
- yaml 配置加载
- 随机种子固定
- device 选择
- train / val 数据切分
- 多 epoch 训练
- tqdm batch 进度条
- train loss / train accuracy
- val loss / val accuracy
- 按 `val_loss` 保存 `best.ckpt`
- 训练结束 summary 打印

## 3. 数据 Pipeline

数据目录结构当前按类别组织：

```text
data/train_data/
  cloth/
    cloth_xxx.xlsx
  leather/
    leather_xxx.xlsx
  metal/
    metal_xxx.xlsx
  wood/
    wood_xxx.xlsx
```

每个 Excel 样本中需要包含对应类别的两列：

```text
{class_name}_high
{class_name}_low
```

例如 cloth 类样本需要：

```text
cloth_high
cloth_low
```

Dataset 内部处理流程：

```text
读取 Excel
-> 取 high / low 两列
-> dropna
-> 重采样到 sequence_length=1024
-> per-sample 标准化
-> 渲染成 3 通道曲线图
-> 返回 signal / image / label
```

单个样本输出：

```text
signal: [2, 1024]
image:  [3, 224, 224]
label:  int
```

一个 batch 输出：

```text
signal: [B, 2, 1024]
image:  [B, 3, 224, 224]
label:  [B]
```

图像视图的三个通道当前为：

```text
channel 0 = high 曲线
channel 1 = low 曲线
channel 2 = high - low 差值曲线
```

这保证 image 仍然来自原始 signal，不引入类别文字、标题、坐标轴等泄漏信息。

## 4. 网络 Pipeline

当前模型是 `MultiModalClassifier`，不是 diffusion 模型。

整体结构：

```text
signal -> TimeEncoder  -> feature_ts
image  -> ImageEncoder -> feature_img

concat(feature_ts, feature_img)
-> FusionClassifier
-> logits
```

最终输出：

```text
logits: [B, 4]
```

其中 4 对应四个材料类别。

## 5. TimeEncoder 结构

`TimeEncoder` 处理原始双通道时序信号。

输入：

```text
[B, 2, 1024]
```

结构：

```text
Conv1d(2 -> 32, kernel_size=7, stride=2)
-> BatchNorm1d
-> ReLU
-> ResidualBlock1D(32 -> 64)
-> ResidualBlock1D(64 -> 128)
-> ResidualBlock1D(128 -> 256)
-> AdaptiveAvgPool1d(1)
-> squeeze
-> Linear(256 -> 128)
-> ReLU
-> Dropout
```

输出：

```text
feature_ts: [B, 128]
```

这一分支学习的是数值时序特征，例如局部变化、幅值关系、斜率趋势和 high / low 通道关系。

## 6. ImageEncoder 结构

`ImageEncoder` 处理由时序信号渲染得到的曲线图像。

输入：

```text
[B, 3, 224, 224]
```

结构：

```text
Conv2d(3 -> 32)
-> BatchNorm2d
-> ReLU
-> MaxPool2d

Conv2d(32 -> 64)
-> BatchNorm2d
-> ReLU
-> MaxPool2d

Conv2d(64 -> 128)
-> BatchNorm2d
-> ReLU
-> MaxPool2d

Conv2d(128 -> 128)
-> BatchNorm2d
-> ReLU
-> AdaptiveAvgPool2d(1, 1)
-> flatten
-> Linear(128 -> 128)
-> ReLU
-> Dropout
```

输出：

```text
feature_img: [B, 128]
```

这一分支学习的是曲线图像中的形态特征，例如整体轮廓、峰谷位置、曲线趋势和 high-low 差值结构。

## 7. FusionClassifier 结构

当前采用 late fusion，也就是两个 encoder 各自提取完特征后再融合。

输入：

```text
feature_ts:  [B, 128]
feature_img: [B, 128]
```

融合：

```text
concat([feature_ts, feature_img], dim=1)
-> [B, 256]
```

分类头：

```text
Linear(256 -> 128)
-> ReLU
-> Dropout
-> Linear(128 -> 64)
-> ReLU
-> Dropout
-> Linear(64 -> 4)
```

输出：

```text
logits: [B, 4]
```

训练损失：

```text
CrossEntropyLoss(logits, label)
```

## 8. 当前训练 Pipeline

训练时，一个 batch 的流动过程是：

```text
batch["signal"] -> device
batch["image"]  -> device
batch["label"]  -> device

logits = model(signal, image)
loss = CrossEntropyLoss(logits, label)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

每个 epoch：

```text
model.train()
-> 遍历 train_loader
-> 统计 train_loss / train_acc
-> model.eval()
-> torch.no_grad()
-> 遍历 val_loader
-> 统计 val_loss / val_acc
-> 如果 val_loss 更低，保存 best.ckpt
```

当前 accuracy 已经按样本数统计：

```text
accuracy = total_correct / total_samples
```

当前 loss 也按样本数加权统计：

```text
avg_loss = sum(batch_loss * batch_size) / total_samples
```

## 9. 当前模型不是 Diffuser

当前模型是监督分类模型，不包含 diffusion 训练过程。

当前没有：

- timestep embedding
- noise scheduler
- forward diffusion
- reverse denoising
- noise prediction loss
- sampling denoise loop

所以现在的模型应该被理解为：

```text
ResNet1D + LightCNN + Late Fusion 分类 baseline
```

而不是 diffuser。

## 10. Diffuser 数据增强规划

如果后续新增 diffuser，建议作为数据增强模块放在 classifier 前面，而不是直接插入当前分类网络内部。

推荐数据流：

```text
raw signal
-> augmentation / diffuser
-> augmented signal
-> render image
-> MultiModalClassifier
```

优先增强对象：

```text
signal: [2, 1024]
```

原因是 image 本来就是由 signal 渲染出来的。先增强 signal，再重新生成 image，可以保证两个视图之间仍然一致。

不建议第一步直接做完整 DDPM。更稳的路线是：

```text
阶段 1：稳定当前分类 baseline
阶段 2：传统 signal augmentation
阶段 3：轻量 conditional diffusion / diffusion-lite
阶段 4：将 synthetic signal 混入训练集
```

## 11. 建议实验顺序

### 11.1 Baseline 稳定性

先确认当前 fusion 分类模型可以稳定训练：

```text
train_loss
train_acc
val_loss
val_acc
best.ckpt
```

### 11.2 单分支消融

建议做三组对照：

```text
1. TimeEncoder only
2. ImageEncoder only
3. TimeEncoder + ImageEncoder late fusion
```

如果 fusion 没有优于单时序分支，说明当前图像视图可能贡献有限，或者图像编码方式需要调整。

### 11.3 图像编码对照

当前图像是无坐标轴曲线图。后续可以对比：

```text
1. 当前曲线图
2. GAF
3. Recurrence Plot
4. STFT / CWT
```

第一阶段可以只做：

```text
当前曲线图 vs GAF
```

### 11.4 数据增强对照

先比较传统增强，再比较 diffuser：

```text
A1: 只用真实/传统增强数据
A2: 传统增强 + diffusion 生成数据
A3: 直接传统增强到相同数量
```

如果 A2 不优于 A3，说明 diffuser 暂时没有提供额外价值。

## 12. 当前注意事项

1. 当前 image 是 signal 的另一种表示，不是独立传感器数据。
2. 图像生成时不能包含类别文字、图例、标题、文件名等信息。
3. `batch_size` 较小时，BatchNorm 统计可能不稳定。
4. 测试集最好来自真实独立采样，不能完全依赖同一条曲线的增强样本。
5. diffusion 生成数据不应该完全替代真实或传统增强数据。
6. 当前最重要的是先建立可靠 baseline，再逐步增加复杂模块。

## 13. 推荐落地顺序

当前最推荐的实现顺序：

```text
1. 固定当前 ResNet1D + LightCNN + Late Fusion baseline
2. 跑通完整 train / val / best.ckpt
3. 做 time-only / image-only / fusion 消融
4. 加入传统 signal augmentation
5. 尝试更结构化的图像编码方式
6. 最后再加入 diffuser 数据增强
```

这样每一步的收益和风险都可以单独观察，不会因为一次加入太多模块而难以 debug。
