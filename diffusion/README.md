# Diffusion 数据增强数学说明

## 1. 在本项目中 diffusion 做什么

本项目当前主任务是材料分类。分类模型输入为：

```text
signal: [B, 2, 1024]
image:  [B, 3, 224, 224]
label:  [B]
```

其中 `image` 是由 `signal` 渲染得到的曲线图。因此如果要用 diffusion 做数据增强，推荐优先增强原始时序信号：

```text
真实 signal
-> diffusion 生成 synthetic signal
-> 再渲染成 image
-> 加入分类训练集
```

也就是说，diffusion 模块不直接替代分类器，而是放在分类器之前，用来生成更多合理的时序样本。

## 2. diffusion 的两个阶段

Diffusion 通常分成两个过程：

```text
forward diffusion:  干净数据 -> 逐步加噪
reverse diffusion: 噪声 -> 逐步去噪生成数据
```

当前 `noise_scheduler.py` 实现的是第一部分：

```text
x0 -> xt
```

其中：

```text
x0 = 干净的真实 signal
xt = 第 t 个噪声等级下的带噪 signal
```

## 3. 变量含义

### x0

`x0` 是原始干净数据。

在本项目中：

```text
x0 shape = [B, 2, 1024]
```

含义：

```text
B    = batch size
2    = high / low 两个通道
1024 = 时序长度
```

### noise

`noise` 是随机高斯噪声，形状和 `x0` 相同：

```text
noise shape = [B, 2, 1024]
```

通常由下面的逻辑生成：

```text
noise = torch.randn_like(x0)
```

### t

`t` 是 timestep，也可以理解成噪声等级。

如果：

```text
num_steps = 100
```

那么：

```text
t = 0, 1, 2, ..., 99
```

直观理解：

```text
t 越小，噪声越轻
t 越大，噪声越重
```

## 4. beta, alpha, alpha_bar

### beta

`beta_t` 表示第 `t` 步加噪声的强度。

通常设置一个起点和终点：

```text
beta_start = 1e-4
beta_end   = 0.02
```

然后线性生成：

```text
betas = [beta_0, beta_1, ..., beta_T]
```

可以直观理解为：

```text
beta_start = 最开始每一步加多少噪声
beta_end   = 最后每一步加多少噪声
```

第一版建议使用经典默认值：

```text
num_steps: 100
beta_start: 1e-4
beta_end: 0.02
```

### alpha

`alpha_t` 表示第 `t` 步保留多少原始信号。

它由 `beta_t` 决定：

```text
alpha_t = 1 - beta_t
```

例如：

```text
beta_t = 0.0001
alpha_t = 0.9999
```

表示这一小步几乎保留全部信号。

如果：

```text
beta_t = 0.02
alpha_t = 0.98
```

表示这一小步噪声影响更强。

### alpha_bar

`alpha_bar_t` 是从第 0 步到第 t 步累计保留的信号比例。

公式：

```text
alpha_bar_t = alpha_0 * alpha_1 * ... * alpha_t
```

在代码里通常是：

```text
alpha_bars = torch.cumprod(alphas, dim=0)
```

直观理解：

```text
alpha_t     = 单步保留比例
alpha_bar_t = 从 0 到 t 的累计保留比例
```

随着 `t` 增大：

```text
alpha_bar_t 越来越小
```

也就是：

```text
原始信号越来越少
随机噪声越来越多
```

## 5. 前向加噪公式

Diffusion 的前向加噪公式是：

```text
xt = sqrt(alpha_bar_t) * x0
     + sqrt(1 - alpha_bar_t) * noise
```

这句话可以理解成：

```text
xt = 一部分原始信号 + 一部分随机噪声
```

其中：

```text
sqrt(alpha_bar_t)
```

控制保留多少原始信号。

```text
sqrt(1 - alpha_bar_t)
```

控制加入多少随机噪声。

## 6. 为什么要开根号

这是 diffusion 里最容易困惑的地方。

关键点是：

```text
我们希望控制的是方差/能量比例，而不是普通数值比例。
```

假设：

```text
x0 的方差约为 1
noise 的方差约为 1
```

如果：

```text
xt = a * x0 + b * noise
```

那么 `xt` 的方差大约是：

```text
Var(xt) = a^2 * Var(x0) + b^2 * Var(noise)
```

因为一个随机变量乘上系数 `a` 后，方差会乘以：

```text
a^2
```

如果我们希望：

```text
原始信号占 alpha_bar_t 的能量比例
噪声占 1 - alpha_bar_t 的能量比例
```

就需要：

```text
a^2 = alpha_bar_t
b^2 = 1 - alpha_bar_t
```

所以：

```text
a = sqrt(alpha_bar_t)
b = sqrt(1 - alpha_bar_t)
```

这就是为什么公式里要开根号。

## 7. 不开根号会怎样

如果直接写成：

```text
xt = alpha_bar_t * x0 + (1 - alpha_bar_t) * noise
```

那么实际方差比例会变成：

```text
alpha_bar_t^2
(1 - alpha_bar_t)^2
```

这会导致整体尺度变小，噪声和信号的能量比例也不是我们想要的比例。

举例：

```text
alpha_bar_t = 0.5
```

如果不开根号：

```text
xt = 0.5 * x0 + 0.5 * noise
```

方差约为：

```text
0.5^2 + 0.5^2 = 0.5
```

整体能量变小。

如果开根号：

```text
xt = sqrt(0.5) * x0 + sqrt(0.5) * noise
```

方差约为：

```text
0.5 + 0.5 = 1.0
```

整体尺度更稳定。

一句话总结：

```text
开根号是因为权重乘在数据上，而我们真正想控制的是方差/能量比例。
```

## 8. t 不同时 xt 的样子

当 `t` 很小：

```text
alpha_bar_t 接近 1
sqrt(alpha_bar_t) 接近 1
sqrt(1 - alpha_bar_t) 接近 0
```

所以：

```text
xt 几乎就是 x0
```

当 `t` 很大：

```text
alpha_bar_t 变小
sqrt(alpha_bar_t) 变小
sqrt(1 - alpha_bar_t) 变大
```

所以：

```text
xt 越来越接近随机噪声
```

## 9. denoise model 学什么

训练 diffusion model 时，不是直接让模型预测干净信号 `x0`。

更常见的做法是让模型预测加入的噪声：

```text
model(xt, t, label) -> pred_noise
```

训练目标：

```text
pred_noise 接近真实 noise
```

损失函数：

```text
loss = MSE(pred_noise, noise)
```

训练流程：

```text
1. 从数据集中取真实 signal，记作 x0
2. 随机采样 timestep t
3. 随机生成 noise
4. 用 NoiseScheduler 得到 xt
5. 把 xt, t, label 输入 denoise model
6. 模型输出 pred_noise
7. 用 MSELoss(pred_noise, noise) 训练
```

## 10. 生成数据时发生什么

训练好 denoise model 后，可以从纯噪声开始生成样本。

大致过程：

```text
x_T = random noise
for t = T, T-1, ..., 1:
    pred_noise = model(x_t, t, label)
    x_{t-1} = 根据 pred_noise 去掉一部分噪声

得到 synthetic signal
```

最终输出：

```text
synthetic signal: [2, 1024]
```

再把它保存成 Excel：

```text
{class_name}_high
{class_name}_low
```

这样现有 `MultiModalDataset` 就可以直接读取这些增强样本。

## 11. 本项目建议路线

第一阶段不要急着追求完整复杂 diffusion。建议按下面顺序做：

```text
1. NoiseScheduler.add_noise 跑通
2. DenoiseModel forward 跑通
3. train_diffuser.py 跑通一轮训练
4. 观察 pred_noise 和 noise 的 MSE 是否下降
5. sample_diffuser.py 生成少量 synthetic signal
6. 保存为 Excel
7. 用分类器验证增强数据是否提升 val accuracy / val loss
```

推荐第一版配置：

```yaml
num_steps: 100
beta_start: 1e-4
beta_end: 0.02
```

如果后续发现最后一步噪声不够强，可以适当增大：

```text
beta_end -> 0.03
```

如果发现信号过早被噪声破坏，可以减小：

```text
beta_end -> 0.01
```

但第一版建议先使用经典默认值，优先把 pipeline 跑通。
