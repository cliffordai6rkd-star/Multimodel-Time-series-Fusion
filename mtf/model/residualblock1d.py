"""一维残差块。

这个模块是 TimeEncoder 的基础组件，输入输出都是一维时序特征：
    x: (batch, channels, sequence_length)
"""

import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Conv1D 残差块。

    参数说明：
    - in_channels：输入通道数。
    - out_channels：输出通道数。
    - stride：第一层卷积的步长，用于下采样时间长度。
    - kernel_size：卷积核大小，默认 3。
    - use_batch_norm：是否使用 BatchNorm1d。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel_size=3,
        use_batch_norm=True,
    ):
        super().__init__()

        padding = kernel_size // 2
        bias = not use_batch_norm

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                ),
                nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out
