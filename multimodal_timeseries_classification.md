# 基于时序信号与时序图像编码的多模态联合分类

## 1. 研究目标

当前数据形式为每个样本一条双通道时序信号：

```text
sample shape = (1024, 2)
channel 1 = high
channel 2 = low
class = cloth / leather / metal / wood
```

项目目标是：在每类真实原始样本较少的情况下，先用小幅度噪声增强模拟真实采样扰动，再用 diffusion 思想扩展样本数量，最后结合时序信号和时序图像编码训练分类器。

整体思想不是让图像提供新的物理信息，而是把同一条时序曲线转换成另一种结构表达，让模型同时学习：

```text
时序数值特征：局部变化、斜率、幅值、通道关系
图像形态特征：整体轮廓、峰谷位置、曲线趋势、形状结构
```

因此，该方法更准确地称为：

```text
时序信号 + 时序图像编码的多视图 / 多模态联合分类
```

而不是严格意义上的 co-training。

## 2. 总体技术路线

推荐流程如下：

```text
每类 1 条真实双通道曲线
  -> 传统 augment 生成每类约 20 条样本
  -> diffusion-lite 学习扰动分布
  -> 扩展到每类约 200 条样本
  -> 每个样本生成两种视图：
       1. 原始双通道时序张量
       2. 时序图像编码
  -> 双分支网络联合训练
  -> 与单时序模型、单图像模型做横向对比
```

建议第一阶段不要直接扩散到每类 2000 条。先做：

```text
augment: 每类 20 条
diffusion: 每类补充到 200 条
```

这样可以降低生成数据主导训练集的风险，也更容易判断 diffusion 和图像视图是否真的带来提升。

## 3. 数据增强阶段

### 3.1 传统 augment

当前 `augment_dataset.py` 的逻辑可以视为真实采样扰动模拟：

```text
增强样本 = 原始曲线 + 逐点随机噪声 + 平滑漂移噪声
```

该阶段的作用是生成少量“近似真实采样环境下的扰动样本”。

推荐配置：

```bash
python datatset/augment_dataset.py data.xlsx -num 20 --output-dir data_aug20
```

输出结构：

```text
data_aug20/
  cloth/cloth_001.xlsx ... cloth_020.xlsx
  leather/leather_001.xlsx ... leather_020.xlsx
  metal/metal_001.xlsx ... metal_020.xlsx
  wood/wood_001.xlsx ... wood_020.xlsx
```

### 3.2 Diffusion-lite 扩增

由于每类真实样本很少，不建议一开始使用大型完整 DDPM。第一版建议使用轻量条件 diffusion / denoising diffusion-lite。

训练目标：

```text
x0 = augment 后的真实风格样本
t = 随机噪声等级
noise = 随机高斯噪声
xt = sqrt(alpha_bar[t]) * x0 + sqrt(1 - alpha_bar[t]) * noise

model(xt, t, class_label) -> predicted_noise
loss = MSE(predicted_noise, noise)
```

生成目标：

```text
指定类别 label
从随机噪声或带噪模板曲线开始
逐步去噪
生成新的同类别双通道曲线
```

推荐第一版输出规模：

```text
每类最终约 200 条
= 20 条 augment 样本 + 180 条 diffusion 样本
```

## 4. 时序图像编码设计

### 4.1 不建议直接使用普通 matplotlib 曲线图作为唯一图像输入

普通曲线图可能包含以下非数据因素：

```text
坐标轴
标题
图例
类别文字
线条颜色
留白
坐标缩放差异
```

这些因素可能让图像模型学到绘图格式，而不是材料信号特征。

如果使用普通折线图，必须统一规则：

```text
不显示标题
不显示类别名
不显示坐标轴文字
不显示图例
统一 y 轴范围
统一图像尺寸
统一线宽
统一颜色策略
```

### 4.2 推荐的图像编码方式

更推荐把时间序列转换为结构化图像，例如：

```text
1. Gramian Angular Field, GAF
2. Recurrence Plot, RP
3. Markov Transition Field, MTF
4. STFT 频谱图
5. CWT 小波时频图
```

第一版可以优先选择两种实现成本较低的方式：

```text
方案 A：无坐标轴双通道曲线图
方案 B：GAF / RP 图像编码
```

双通道样本可以构造成 3 通道图像：

```text
image channel 1 = high 通道图像编码
image channel 2 = low 通道图像编码
image channel 3 = high - low 差值图像编码
```

这样图像 encoder 可以同时看到单通道形态和双通道差异关系。

## 5. 多模态联合分类框架

每个样本同时包含两个输入：

```text
time_series: (2, 1024)
image_view:  (3, H, W)
label:       int
```

模型结构：

```text
time_series
  -> time encoder
  -> feature_ts

image_view
  -> image encoder
  -> feature_img

concat(feature_ts, feature_img)
  -> fusion MLP
  -> classifier
  -> class logits
```

基础损失：

```text
loss_cls = CrossEntropy(logits, label)
```

可选辅助损失：

```text
loss_align = ContrastiveLoss(feature_ts, feature_img)
loss = loss_cls + lambda_align * loss_align
```

第一版建议先只使用分类损失，确认融合结构有效后，再加入对比约束。

## 6. Base 网络 1：ResNet1D + CNN 图像分支 Late Fusion

### 6.1 网络定位

这是最稳妥的 baseline。结构简单、训练稳定，适合先验证多模态融合是否有效。

### 6.2 输入

```text
时序输入: (batch, 2, 1024)
图像输入: (batch, 3, 224, 224)
类别输出: (batch, 4)
```

### 6.3 时序分支

使用轻量 ResNet1D：

```text
Conv1d(2 -> 32, kernel_size=7, stride=2)
BatchNorm1d
ReLU

ResidualBlock1D(32 -> 64)
ResidualBlock1D(64 -> 128)
ResidualBlock1D(128 -> 256)

GlobalAveragePooling1D
Linear(256 -> 128)
```

输出：

```text
feature_ts: (batch, 128)
```

### 6.4 图像分支

第一版可以使用轻量 CNN，而不是大型预训练模型：

```text
Conv2d(3 -> 32, kernel_size=3)
BatchNorm2d
ReLU
MaxPool2d

Conv2d(32 -> 64, kernel_size=3)
BatchNorm2d
ReLU
MaxPool2d

Conv2d(64 -> 128, kernel_size=3)
BatchNorm2d
ReLU
AdaptiveAvgPool2d(1)

Linear(128 -> 128)
```

输出：

```text
feature_img: (batch, 128)
```

### 6.5 融合分类头

```text
feature = concat(feature_ts, feature_img)
feature shape = (batch, 256)

Linear(256 -> 128)
ReLU
Dropout(0.3)
Linear(128 -> 4)
```

### 6.6 优点

```text
结构清楚
参数量较小
容易训练
容易做消融实验
适合作为第一版 baseline
```

### 6.7 风险

```text
图像分支可能学习到图像绘制格式
late fusion 对两个模态的交互建模较弱
如果数据很少，图像分支仍可能过拟合
```

## 7. Base 网络 2：TCN 时序分支 + 图像 Encoder + 对比对齐融合

### 7.1 网络定位

这个 baseline 比 Base 网络 1 更强调时序长程依赖和跨模态一致性。适合验证：

```text
时序特征和图像特征是否能形成互补
同一样本的两个视图是否能在特征空间中对齐
```

### 7.2 输入

```text
时序输入: (batch, 2, 1024)
图像输入: (batch, 3, 224, 224)
类别输出: (batch, 4)
```

### 7.3 时序分支：TCN

TCN 使用膨胀卷积扩大感受野，适合处理长序列。

```text
Conv1d(2 -> 64, kernel_size=3, dilation=1)
TCNBlock(64, dilation=1)
TCNBlock(64, dilation=2)
TCNBlock(64, dilation=4)
TCNBlock(64, dilation=8)
TCNBlock(64, dilation=16)

GlobalAveragePooling1D
Linear(64 -> 128)
```

输出：

```text
feature_ts: (batch, 128)
```

### 7.4 图像分支

图像分支可继续使用轻量 CNN，保证和 Base 网络 1 的对比主要来自时序分支和训练目标差异。

```text
LightCNN(image_view) -> feature_img
feature_img: (batch, 128)
```

### 7.5 对比对齐头

将两个模态投影到同一特征空间：

```text
proj_ts = MLP(feature_ts)   -> (batch, 64)
proj_img = MLP(feature_img) -> (batch, 64)
```

辅助目标：

```text
同一个样本的 proj_ts 与 proj_img 距离更近
不同样本，尤其不同类别样本的距离更远
```

可用简化版 InfoNCE：

```text
loss_align = InfoNCE(proj_ts, proj_img)
```

最终损失：

```text
loss = loss_cls + lambda_align * loss_align
```

第一版推荐：

```text
lambda_align = 0.05 或 0.1
```

### 7.6 融合分类头

```text
feature = concat(feature_ts, feature_img, abs(feature_ts - feature_img))
feature shape = (batch, 384)

Linear(384 -> 128)
ReLU
Dropout(0.3)
Linear(128 -> 4)
```

### 7.7 优点

```text
TCN 对长序列建模能力更强
对比对齐能约束两个视图学习一致语义
比简单拼接更能利用双视图关系
```

### 7.8 风险

```text
训练更复杂
lambda_align 需要调参
小数据下对比学习可能不稳定
如果图像视图质量不高，对齐损失可能反而干扰分类
```

## 8. 两个 Base 网络横向对比

| 维度 | Base 网络 1：ResNet1D + LightCNN Late Fusion | Base 网络 2：TCN + LightCNN + 对比对齐 |
|---|---|---|
| 主要目的 | 验证多模态拼接是否有效 | 验证长程时序建模与跨模态对齐是否有效 |
| 时序建模 | ResNet1D，稳定、通用 | TCN，适合长序列依赖 |
| 图像建模 | LightCNN | LightCNN |
| 融合方式 | concat 后 MLP | concat + 差值特征 + 对比对齐 |
| 损失函数 | CrossEntropy | CrossEntropy + InfoNCE |
| 训练难度 | 低 | 中 |
| 参数敏感性 | 较低 | 较高 |
| 过拟合风险 | 中 | 中到高 |
| 适合作用 | 第一版主 baseline | 第二版增强 baseline |

## 9. 实验对照设计

为了判断每个模块是否真的有效，建议至少做以下对照：

### 9.1 数据增强对照

```text
A1: 只用 augment 20 条/类
A2: augment 20 条/类 + diffusion 180 条/类
A3: 直接传统 augment 到 200 条/类
```

观察：

```text
A2 是否优于 A1
A2 是否优于 A3
```

如果 A2 没有优于 A3，说明 diffusion 暂时没有提供额外价值。

### 9.2 模型结构对照

```text
B1: 只用时序分支
B2: 只用图像分支
B3: 时序 + 图像 late fusion
B4: 时序 + 图像 + 对比对齐
```

观察：

```text
B3 是否优于 B1 和 B2
B4 是否优于 B3
```

如果 B3 不优于 B1，说明图像视图可能没有贡献，或者图像编码方式需要调整。

### 9.3 图像编码方式对照

```text
C1: 无坐标轴双通道曲线图
C2: GAF 图像
C3: Recurrence Plot 图像
C4: STFT / CWT 时频图
```

第一版建议先做：

```text
C1 vs C2
```

## 10. 评价指标

分类任务建议记录：

```text
accuracy
macro precision
macro recall
macro F1
confusion matrix
```

由于类别数较少，必须看混淆矩阵，确认模型是否只对某几类表现好。

同时建议记录：

```text
训练集准确率
验证集准确率
测试集准确率
训练/验证 loss 曲线
```

如果出现：

```text
训练集准确率很高
测试集准确率明显较低
```

说明模型可能过拟合了 augment / diffusion 的生成分布。

## 11. 关键注意事项

### 11.1 测试集必须尽量独立

最理想情况是测试集来自真实独立采样，而不是用同一个 augment 脚本从同一条原始曲线生成。

否则测试结果可能虚高。

### 11.2 图像不能泄漏类别信息

生成图像时必须避免：

```text
标题中出现类别名
文件名写入图像
图例使用类别名
不同类别使用固定不同颜色
不同类别 y 轴范围不同
```

否则图像分支可能学到非信号信息。

### 11.3 diffusion 数据不要完全替代 augment 数据

推荐训练集构成：

```text
每类 20 条 augment 样本
每类 180 条 diffusion 样本
```

而不是只保留 diffusion 生成样本。

### 11.4 先建立可解释 baseline

推荐实现顺序：

```text
1. 只用时序数据训练 ResNet1D
2. 加入图像分支做 Late Fusion
3. 更换图像编码方式
4. 加入 diffusion 扩增
5. 加入对比对齐损失
```

这样每一步的收益和风险都能被单独评估。

## 12. 推荐第一版落地方案

第一版最小可行系统：

```text
数据：
  augment 到每类 20 条
  diffusion 扩到每类 200 条

图像：
  先使用无坐标轴、统一尺度的双通道曲线图

模型：
  Base 网络 1：ResNet1D + LightCNN Late Fusion

对照：
  只用时序 ResNet1D
  只用图像 LightCNN
  时序 + 图像融合
```

第二版增强系统：

```text
图像：
  尝试 GAF / RP 图像编码

模型：
  Base 网络 2：TCN + LightCNN + 对比对齐融合

对照：
  Late Fusion vs 对比对齐
  曲线图 vs GAF/RP
```

如果第二版没有明显优于第一版，说明当前任务中时序原始信号已经包含了主要可分信息，复杂多模态结构不一定必要。

