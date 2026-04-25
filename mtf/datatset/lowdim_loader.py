"""多模态数据集读取模块。

每个样本从单个 Excel 文件读取，返回：
    signal: (2, sequence_length)
    image:  (3, image_size, image_size)
    label:  int

图像视图由时序信号直接栅格化生成，不包含标题、坐标轴、图例或类别名，
避免图像分支学习到非信号信息。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _get(config, key, default):
    if config is None:
        return default
    return config.get(key, default)


def resample_signal(signal: np.ndarray, target_length: int) -> np.ndarray:
    """把 (N, 2) 信号重采样到 (target_length, 2)。"""
    if len(signal) == target_length:
        return signal.astype(np.float32)

    source_x = np.linspace(0.0, 1.0, len(signal))
    target_x = np.linspace(0.0, 1.0, target_length)
    high = np.interp(target_x, source_x, signal[:, 0])
    low = np.interp(target_x, source_x, signal[:, 1])
    return np.column_stack([high, low]).astype(np.float32)


def normalize_signal(signal: np.ndarray, mode: str) -> np.ndarray:
    """标准化时序信号。

    none：不做标准化。
    per_sample：每个样本、每个通道单独做 z-score。
    """
    if mode == "none":
        return signal.astype(np.float32)

    if mode != "per_sample":
        raise ValueError(f"未知 signal_normalize 模式: {mode}")

    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return ((signal - mean) / std).astype(np.float32)


def _normalize_to_unit(values: np.ndarray) -> np.ndarray:
    min_value = float(values.min())
    max_value = float(values.max())
    value_range = max(max_value - min_value, 1e-8)
    return (values - min_value) / value_range


def _draw_curve(values: np.ndarray, image_size: int, line_width: int) -> np.ndarray:
    """把一维曲线画成单通道灰度图，返回 (H, W)。"""
    values = _normalize_to_unit(values)
    x_positions = np.linspace(0, image_size - 1, len(values))
    y_positions = (1.0 - values) * (image_size - 1)

    image = np.zeros((image_size, image_size), dtype=np.float32)
    radius = max(0, int(line_width) // 2)

    for index in range(len(values) - 1):
        x0, y0 = x_positions[index], y_positions[index]
        x1, y1 = x_positions[index + 1], y_positions[index + 1]
        steps = max(int(abs(x1 - x0)), int(abs(y1 - y0)), 1) + 1
        xs = np.linspace(x0, x1, steps).round().astype(int)
        ys = np.linspace(y0, y1, steps).round().astype(int)
        xs = np.clip(xs, 0, image_size - 1)
        ys = np.clip(ys, 0, image_size - 1)
        for x, y in zip(xs, ys):
            image[
                max(0, y - radius) : min(image_size, y + radius + 1),
                max(0, x - radius) : min(image_size, x + radius + 1),
            ] = 1.0

    return image


def render_curve_image(
    signal: np.ndarray,
    image_size: int,
    line_width: int,
    normalize: bool,
) -> np.ndarray:
    """把双通道时序信号转换成 3 通道图像。

    channel 0：high 曲线
    channel 1：low 曲线
    channel 2：high - low 差值曲线
    """
    high = signal[:, 0]
    low = signal[:, 1]
    diff = high - low
    image = np.stack(
        [
            _draw_curve(high, image_size, line_width),
            _draw_curve(low, image_size, line_width),
            _draw_curve(diff, image_size, line_width),
        ],
        axis=0,
    )

    if normalize:
        image = image * 2.0 - 1.0

    return image.astype(np.float32)


class MultiModalDataset(Dataset):
    """读取 data/<class>/<class>_xxx.xlsx 格式的数据集。"""

    def __init__(self, config=None):
        self.config = config or {}
        self.data_dir = Path(_get(self.config, "data_dir", "data")).expanduser().resolve()
        self.class_names = list(
            _get(self.config, "class_names", ["cloth", "leather", "metal", "wood"])
        )
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.sequence_length = int(_get(self.config, "sequence_length", 1024))
        self.signal_normalize = _get(self.config, "signal_normalize", "per_sample")
        self.image_size = int(_get(self.config, "image_size", 224))
        self.image_line_width = int(_get(self.config, "image_line_width", 1))
        self.image_normalize = bool(_get(self.config, "image_normalize", True))

        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        self.samples = []
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            for file_path in sorted(class_dir.glob("*.xlsx")):
                self.samples.append(
                    {
                        "path": file_path,
                        "label": self.class_to_idx[class_name],
                        "class_name": class_name,
                    }
                )

        if not self.samples:
            raise ValueError(f"没有在 {self.data_dir} 中找到 xlsx 样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        class_name = item["class_name"]
        df = pd.read_excel(item["path"])

        high_column = f"{class_name}_high"
        low_column = f"{class_name}_low"
        required = [high_column, low_column]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{item['path']} 缺少列: {missing}")

        raw_signal = df[required].dropna().to_numpy(dtype=np.float32)
        signal = resample_signal(raw_signal, self.sequence_length)
        signal = normalize_signal(signal, self.signal_normalize)
        image = render_curve_image(
            signal=signal,
            image_size=self.image_size,
            line_width=self.image_line_width,
            normalize=self.image_normalize,
        )

        return {
            "signal": torch.from_numpy(signal.T),  # (2, sequence_length)
            "image": torch.from_numpy(image),  # (3, image_size, image_size)
            "label": torch.tensor(item["label"], dtype=torch.long),
            "path": str(item["path"]),
            "class_name": class_name,
        }
