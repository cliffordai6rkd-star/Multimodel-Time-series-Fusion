"""双通道时序信号编码器。

TimeEncoder 负责把输入形状为 (batch, 2, 1024) 的 high/low 双通道曲线，
编码成固定长度特征向量，供后续多模态融合分类器使用。

配置方式：
    config = {
        "input_channels": 2,
        "stem_channels": 32,
        "stem_kernel_size": 7,
        "stem_stride": 2,
        "block_channels": [64, 128, 256],
        "block_strides": [2, 2, 2],
        "block_kernel_size": 3,
        "feature_dim": 128,
        "dropout": 0.2,
        "use_batch_norm": True,
    }
    encoder = TimeEncoder(config)

也支持把 stem / block 写成嵌套配置：
    config = {
        "input_channels": 2,
        "stem": {"channels": 32, "kernel_size": 7, "stride": 2},
        "blocks": {"channels": [64, 128, 256], "strides": [2, 2, 2], "kernel_size": 3},
        "feature_dim": 128,
    }
"""

import torch.nn as nn

from mtf_fusion.model.residualblock1d import ResidualBlock1D


def _config_get(config, key, default=None, group=None, group_key=None):
    """读取配置项，并兼容 flat config 与嵌套 config。

    例如下面两种写法都能读到 stem_channels：
        {"stem_channels": 32}
        {"stem": {"channels": 32}}
    """
    if config is None:
        return default

    if key in config:
        return config.get(key, default)

    if group is not None:
        group_config = config.get(group, {})
        if isinstance(group_config, dict):
            return group_config.get(group_key or key, default)

    return default


class TimeEncoder(nn.Module):
    """基于 ResNet1D 的时序编码器。

    输入：
        x: (batch, input_channels, sequence_length)

    输出：
        feature: (batch, feature_dim)
    """

    def __init__(self, config=None):
        super().__init__()

        self.config = config or {}

        self.input_channels = _config_get(self.config, "input_channels", 2)
        self.stem_channels = _config_get(
            self.config,
            "stem_channels",
            32,
            group="stem",
            group_key="channels",
        )
        self.stem_kernel_size = _config_get(
            self.config,
            "stem_kernel_size",
            7,
            group="stem",
            group_key="kernel_size",
        )
        self.stem_stride = _config_get(
            self.config,
            "stem_stride",
            2,
            group="stem",
            group_key="stride",
        )

        self.block_channels = list(
            _config_get(
                self.config,
                "block_channels",
                [64, 128, 256],
                group="blocks",
                group_key="channels",
            )
        )
        self.block_strides = list(
            _config_get(
                self.config,
                "block_strides",
                [2, 2, 2],
                group="blocks",
                group_key="strides",
            )
        )
        self.block_kernel_size = _config_get(
            self.config,
            "block_kernel_size",
            3,
            group="blocks",
            group_key="kernel_size",
        )

        self.feature_dim = _config_get(self.config, "feature_dim", 128)
        self.dropout = _config_get(self.config, "dropout", 0.2)
        self.use_batch_norm = _config_get(self.config, "use_batch_norm", True)

        self._validate_config()

        stem_padding = self.stem_kernel_size // 2
        stem_bias = not self.use_batch_norm
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.input_channels,
                self.stem_channels,
                kernel_size=self.stem_kernel_size,
                stride=self.stem_stride,
                padding=stem_padding,
                bias=stem_bias,
            ),
            nn.BatchNorm1d(self.stem_channels) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        layers = []
        in_channels = self.stem_channels
        for out_channels, stride in zip(self.block_channels, self.block_strides):
            layers.append(
                ResidualBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    kernel_size=self.block_kernel_size,
                    use_batch_norm=self.use_batch_norm,
                )
            )
            in_channels = out_channels

        self.layers = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )

    def _validate_config(self):
        """检查配置是否合法，避免 forward 时才暴露 shape 错误。"""
        if len(self.block_channels) != len(self.block_strides):
            raise ValueError("block_channels 和 block_strides 长度必须一致。")

        if not self.block_channels:
            raise ValueError("block_channels 不能为空。")

        if self.input_channels < 1:
            raise ValueError("input_channels 必须大于 0。")

        if self.stem_channels < 1:
            raise ValueError("stem_channels 必须大于 0。")

        if self.feature_dim < 1:
            raise ValueError("feature_dim 必须大于 0。")

        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")

    def forward(self, x, return_shapes=False):
        """执行前向传播。

        return_shapes=True 时额外返回每一层输出形状，方便你调参时检查网络结构。
        """
        shapes = {}

        if return_shapes:
            shapes["input"] = tuple(x.shape)

        x = self.stem(x)
        if return_shapes:
            shapes["stem"] = tuple(x.shape)

        for layer_index, layer in enumerate(self.layers, start=1):
            x = layer(x)
            if return_shapes:
                shapes[f"layer{layer_index}"] = tuple(x.shape)

        x = self.pool(x)
        if return_shapes:
            shapes["pool"] = tuple(x.shape)

        x = x.squeeze(-1)
        if return_shapes:
            shapes["flatten"] = tuple(x.shape)

        feature = self.proj(x)
        if return_shapes:
            shapes["feature"] = tuple(feature.shape)
            return feature, shapes

        return feature
