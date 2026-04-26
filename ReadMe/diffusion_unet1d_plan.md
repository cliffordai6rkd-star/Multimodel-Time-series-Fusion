# Diffusion 1D U-Net 思路说明

## 1. 为什么考虑把 Denoiser 改成 U-Net

当前 diffusion 的 denoiser 主要是几层 `Conv1d`：

```text
x_t + timestep + label
-> Conv1d
-> 条件融合
-> 多层 Conv1d
-> pred_noise
```

这个结构可以跑通 DDPM 的基本训练流程，但它主要在原始长度 `1024` 上做局部卷积。对于 cloth 这类信号，真实曲线有明显的全局结构：

```text
低值区 -> 快速上升 -> 平台区 -> 快速下降 -> 低值区
```

普通浅层卷积容易学到局部抖动和数值范围，但不容易学到“什么时候上升、什么时候下降、平台持续多久”这种较长距离结构。

U-Net 的优势是：

```text
下采样：逐步压缩时间长度，让模型看到更大的时间范围
中间层：在低分辨率下学习全局结构
上采样：逐步恢复到 1024 长度
跳跃连接：把浅层细节传回解码阶段
```

因此 1D U-Net 更适合做时序 diffusion 的噪声预测网络。

## 2. 当前任务中的 U-Net 是什么

这里的 U-Net 不是图像中的 2D U-Net，而是 1D U-Net。

输入仍然是双通道时序：

```text
x_t: [B, 2, 1024]
t:   [B]
y:   [B]
```

输出仍然是预测噪声：

```text
pred_noise: [B, 2, 1024]
```

它替代的是现在的 `Denoiser`，不是替代整个 diffusion 流程。

训练目标仍然不变：

```text
noise = torch.randn_like(x0)
x_t = add_noise(x0, noise, t)
pred_noise = denoiser(x_t, t, label)
loss = MSE(pred_noise, noise)
```

也就是说，U-Net 只负责更强地完成：

```text
denoiser(x_t, t, label) -> pred_noise
```

## 3. 关键名词解释

### 3.1 下采样

下采样就是把时间长度变短。

例如：

```text
[B, 64, 1024] -> [B, 64, 512]
[B, 128, 512] -> [B, 128, 256]
[B, 256, 256] -> [B, 256, 128]
```

它的作用是让模型在更短的序列上看更大的范围。

可以类比为：

```text
原图看细节
缩略图看整体轮廓
```

在 1D 时序里，下采样常见做法有：

```text
stride=2 的 Conv1d
MaxPool1d
AvgPool1d
```

后续实现时，推荐先用 `stride=2 的 Conv1d`，因为它可以一边降采样，一边学习特征。

### 3.2 上采样

上采样就是把时间长度恢复变长。

例如：

```text
[B, 256, 128] -> [B, 256, 256]
[B, 128, 256] -> [B, 128, 512]
[B, 64, 512]  -> [B, 64, 1024]
```

它的作用是把低分辨率下学到的全局结构恢复到原始长度。

常见做法有：

```text
插值上采样 + Conv1d
ConvTranspose1d
```

后续实现时，推荐先用：

```text
Upsample(scale_factor=2) + Conv1d
```

这样 shape 更容易控制。

### 3.3 编码器

编码器是 U-Net 的左半边，负责逐步下采样。

它把原始信号变成更抽象、更低分辨率的特征：

```text
[B, 2, 1024]
-> [B, 64, 1024]
-> [B, 128, 512]
-> [B, 256, 256]
-> [B, 256, 128]
```

编码器越往后，时间长度越短，通道数通常越多。

### 3.4 解码器

解码器是 U-Net 的右半边，负责逐步上采样。

它把低分辨率特征恢复成原始长度：

```text
[B, 256, 128]
-> [B, 256, 256]
-> [B, 128, 512]
-> [B, 64, 1024]
-> [B, 2, 1024]
```

解码器最终输出的不是分类结果，而是：

```text
pred_noise
```

### 3.5 跳跃连接

跳跃连接也叫 skip connection。

它把编码器早期的特征直接传给解码器对应层。

原因是：

```text
下采样会压缩信息，可能丢掉细节
跳跃连接可以把细节补回来
```

例如：

```text
encoder 得到 skip2: [B, 128, 256]
decoder 上采样得到: [B, 256, 256]
拼接后: [B, 384, 256]
再用 ConvBlock 融合
```

对于你的时序数据，跳跃连接有助于保留局部波动细节。

### 3.6 bottleneck

bottleneck 是 U-Net 中间最窄的位置。

例如：

```text
[B, 256, 128]
```

它的时间长度最短，但通道数较多。这里适合学习更全局的结构，比如：

```text
平台从哪里开始
平台在哪里结束
整体趋势是什么
```

### 3.7 条件信息

你的 diffusion 是 conditional diffusion，因为 denoiser 不只看 `x_t`，还看：

```text
timestep t
label y
```

其中：

```text
t 告诉模型当前噪声强度
label 告诉模型要生成哪一类数据
```

当前做法是：

```text
time_embed(t) + label_embed(y)
```

得到条件向量后，可以注入到 U-Net 的多个层中。

第一版可以先简单一些：

```text
在每个 ConvBlock 前，把条件向量投影到当前通道数，然后加到 feature 上
```

## 4. 建议的第一版 1D U-Net 结构

第一版不要做太复杂，目标是先跑通并验证能否过拟合。

推荐结构：

```text
输入:
x_t [B, 2, 1024]

条件:
t_embed + label_embed -> cond

Encoder:
block1: [B, 2, 1024]   -> [B, 64, 1024]
down1:  [B, 64, 1024]  -> [B, 64, 512]

block2: [B, 64, 512]   -> [B, 128, 512]
down2:  [B, 128, 512]  -> [B, 128, 256]

block3: [B, 128, 256]  -> [B, 256, 256]
down3:  [B, 256, 256]  -> [B, 256, 128]

Bottleneck:
middle: [B, 256, 128]  -> [B, 256, 128]

Decoder:
up3:    [B, 256, 128]  -> [B, 256, 256]
concat skip3
block_up3 -> [B, 128, 256]

up2:    [B, 128, 256]  -> [B, 128, 512]
concat skip2
block_up2 -> [B, 64, 512]

up1:    [B, 64, 512]   -> [B, 64, 1024]
concat skip1
block_up1 -> [B, 64, 1024]

输出:
Conv1d: [B, 64, 1024] -> [B, 2, 1024]
```

## 5. 与当前 Denoiser 的关系

当前 Denoiser 可以保留，作为 baseline。

后续建议新增一个模型，而不是直接覆盖：

```text
Denoiser          普通 Conv baseline
DenoiseUNet1D     更强的 U-Net denoiser
```

这样可以对比：

```text
相同数据
相同训练配置
不同 denoiser 结构
```

观察：

```text
loss 是否更低
生成曲线是否出现平台结构
是否更像真实 cloth / leather / metal / wood
```

## 6. 过拟合实验建议

在正式追求泛化之前，建议先做过拟合实验。

目标：

```text
证明 U-Net 至少能学会训练集的结构
```

推荐配置：

```text
num_steps: 100
base_channels: 64
channel_mults: [1, 2, 4]
num_epochs: 500
batch_size: 4
lr: 1e-4
```

观察指标：

```text
1. diffusion loss 是否明显低于当前 Conv Denoiser
2. 生成曲线是否出现低值区、平台区、下降区
3. 同一类别生成结果是否保持该类别的大体形状
```

如果 U-Net 仍然无法在小数据上过拟合，说明还需要检查：

```text
采样公式
归一化/反归一化
训练数据质量
条件 label 是否正确
```

## 7. 后续实现顺序

建议后续按下面顺序实现，不要一次写完整模型：

```text
1. 实现 ConvBlock1D，并验证输入输出 shape
2. 实现 DownBlock1D，并验证长度 1024 -> 512
3. 实现 UpBlock1D，并验证长度 512 -> 1024
4. 组合成最小 U-Net，不加条件，先跑 shape
5. 加入 timestep embedding
6. 加入 label embedding
7. 接入 train_diffusion.py
8. 做一轮小数据过拟合实验
9. 接入 sample_diffuser.py 生成并可视化
```

这样每一步都能单独 debug，不容易在复杂结构里迷路。

