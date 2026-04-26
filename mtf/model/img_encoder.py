# ImageEncoder 负责把由时序曲线转换得到的图像视图编码成固定长度特征。
# 所有可调结构参数都从 config 读取，方便后续横向对比。


import torch
import torch.nn as nn


def _get(config, key, default):
    if config is None:
        return default
    return config.get(key, default)


class ImageEncoder(nn.Module):

        # input :image: (batch, input_channels, height, width)
        # out_put: feature: (batch, feature_dim)

    def __init__(self, config=None):
        super().__init__()

        self.config = config or {}
        self.input_channels = _get(self.config, "input_channels", 3)
        self.channels = list(_get(self.config, "channels", [32, 64, 128, 128]))
        self.kernel_size = _get(self.config, "kernel_size", 3)
        self.pool_every = list(_get(self.config, "pool_every", [True, True, True, False]))
        self.feature_dim = _get(self.config, "feature_dim", 128)
        self.dropout = _get(self.config, "dropout", 0.2)
        self.use_batch_norm = _get(self.config, "use_batch_norm", True)

        self._validate_config()

        padding = self.kernel_size // 2
        bias = not self.use_batch_norm
        layers = []
        in_channels = self.input_channels
        for out_channels, use_pool in zip(self.channels, self.pool_every):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    padding=padding,
                    bias=bias,
                )
            )
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if use_pool:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )

    def _validate_config(self):
        if self.input_channels < 1:
            raise ValueError("image_encoder.input_channels must > 0")
        if not self.channels:
            raise ValueError("image_encoder.channels is empty")
        if len(self.channels) != len(self.pool_every):
            raise ValueError("image_encoder.channels and pool_every must have same lenth")
        if self.feature_dim < 1:
            raise ValueError("image_encoder.feature_dim must > 0")
        if not 0 <= self.dropout < 1:
            raise ValueError("image_encoder.dropout must between [0, 1)")

    def forward(self, image, return_shapes=False):
        shapes = {}
        if return_shapes:
            shapes["input"] = tuple(image.shape)

        x = image
        block_index = 1
        for layer in self.features:
            x = layer(x)
            if return_shapes and isinstance(layer, nn.MaxPool2d):
                shapes[f"pool{block_index}"] = tuple(x.shape)
                block_index += 1

        if return_shapes:
            shapes["features"] = tuple(x.shape)

        x = torch.flatten(x, start_dim=1)
        if return_shapes:
            shapes["flatten"] = tuple(x.shape)

        feature = self.proj(x)
        if return_shapes:
            shapes["feature"] = tuple(feature.shape)
            return feature, shapes

        return feature
