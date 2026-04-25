"""多模态分类模型。

模型由三部分组成：
1. TimeEncoder：编码原始双通道时序信号。
2. ImageEncoder：编码由时序曲线转换得到的图像视图。
3. FusionClassifier：融合两个特征并输出类别 logits。
"""

import torch
import torch.nn as nn

from mtf.model.img_encoder import ImageEncoder
from mtf.model.time_encoder import TimeEncoder


def _get(config, key, default):
    if config is None:
        return default
    return config.get(key, default)


class FusionClassifier(nn.Module):
    """Late Fusion 分类头。"""

    def __init__(self, config=None):
        super().__init__()

        self.config = config or {}
        self.time_feature_dim = _get(self.config, "time_feature_dim", 128)
        self.image_feature_dim = _get(self.config, "image_feature_dim", 128)
        self.hidden_dims = list(_get(self.config, "hidden_dims", [128, 64]))
        self.dropouts = list(_get(self.config, "dropouts", [0.3, 0.2]))
        self.num_classes = _get(self.config, "num_classes", 4)

        if len(self.hidden_dims) != len(self.dropouts):
            raise ValueError("fusion.hidden_dims 和 fusion.dropouts 长度必须一致。")

        layers = []
        in_dim = self.time_feature_dim + self.image_feature_dim
        for hidden_dim, dropout in zip(self.hidden_dims, self.dropouts):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, feature_ts, feature_img):
        feature = torch.cat([feature_ts, feature_img], dim=1)
        logits = self.classifier(feature)
        return logits


class MultiModalClassifier(nn.Module):
    """时序 + 图像多模态分类器。"""

    def __init__(self, config=None):
        super().__init__()

        self.config = config or {}
        model_config = self.config.get("model", self.config)

        num_classes = model_config.get("num_classes", 4)
        time_config = model_config.get("time_encoder", {})
        image_config = model_config.get("image_encoder", {})
        fusion_config = dict(model_config.get("fusion", {}))

        self.time_encoder = TimeEncoder(time_config)
        self.image_encoder = ImageEncoder(image_config)

        fusion_config.setdefault("time_feature_dim", self.time_encoder.feature_dim)
        fusion_config.setdefault("image_feature_dim", self.image_encoder.feature_dim)
        fusion_config.setdefault("num_classes", num_classes)
        self.classifier = FusionClassifier(fusion_config)

    def forward(self, signal, image, return_features=False):
        feature_ts = self.time_encoder(signal)
        feature_img = self.image_encoder(image)
        logits = self.classifier(feature_ts, feature_img)

        if return_features:
            return {
                "logits": logits,
                "feature_ts": feature_ts,
                "feature_img": feature_img,
            }

        return logits
