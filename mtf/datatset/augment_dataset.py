"""双通道时间序列数据增强工具。

用途：
- 从原始总表 Excel 中读取每个类别的 xxx_high / xxx_low 双通道数据。
- 将每个类别的曲线重采样到统一长度。
- 在原始曲线上加入很小的随机点噪声和平滑漂移噪声，生成更多样本。
- 按类别导出为单样本 Excel，供训练或后续可视化使用。

常用命令：
python datatset/augment_dataset.py data.xlsx -num 200 --output-dir data

输入格式约定：
- 必须有时间列，默认列名是 time_s，可通过 --time-column 修改。
- 每个类别必须成对出现：xxx_high 和 xxx_low。
- 默认会先过滤孤立异常点，再做重采样和随机增强，避免把异常尖点复制到所有增强样本中。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。

    关键参数：
    - excel_path：原始总表 Excel。
    - -num/--num：每个类别最终生成多少个样本，包含 1 个原始样本。
    - --target-length：每条曲线重采样后的统一长度。
    - --noise-scale：逐点随机噪声强度。
    - --drift-scale：平滑漂移噪声强度。
    - --outlier-window：异常点过滤的局部窗口大小。
    - --outlier-threshold：偏离局部趋势多少倍 MAD 才判定为异常点。
    - --output-dir：增强样本输出目录。
    """
    parser = argparse.ArgumentParser(
        description="基于极小幅度噪声为双通道时间序列做数据扩充，并按类别导出为单样本 Excel。"
    )
    parser.add_argument("excel_path", nargs="?", default="data.xlsx", help="输入 Excel 文件")
    parser.add_argument("--time-column", default="time_s", help="时间列名")
    parser.add_argument(
        "-num",
        "--num",
        type=int,
        default=20,
        help="每个类别生成的样本数，包含原始样本，例如 -num 200",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=1024,
        help="统一重采样后的序列长度",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.001,
        help="逐点噪声强度，相对于通道范围的比例，默认千分之一",
    )
    parser.add_argument(
        "--drift-scale",
        type=float,
        default=0.0005,
        help="平滑漂移强度，相对于通道范围的比例，默认万分之五",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--no-outlier-filter",
        action="store_true",
        help="关闭异常点过滤；默认会先过滤孤立异常点再增强",
    )
    parser.add_argument(
        "--outlier-window",
        type=int,
        default=21,
        help="异常点过滤的局部中值窗口大小，建议使用奇数，默认 21",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=25.0,
        help="异常点阈值，数值越小过滤越强，默认 25.0 倍局部 MAD",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="输出目录，默认创建 data 文件夹",
    )
    return parser


def discover_classes(df: pd.DataFrame) -> list[str]:
    """从总表列名中自动发现类别。

    只有同时存在 `类别_high` 和 `类别_low` 两列时，才认为这是一个有效类别。
    返回值按字母顺序排序，保证每次运行的标签顺序稳定。
    """
    classes: list[str] = []
    for column in df.columns:
        if not isinstance(column, str) or not column.endswith("_high"):
            continue
        class_name = column[: -len("_high")]
        if f"{class_name}_low" in df.columns:
            classes.append(class_name)
    return sorted(classes)


def normalize_window_size(window_size: int, signal_length: int) -> int:
    """把用户给定的窗口大小修正为合法的奇数窗口。

    中值滤波需要在局部窗口内估计曲线趋势。窗口太小会把正常抖动也当成趋势，
    窗口太大则可能抹掉真实峰谷。默认 21 对 4000 点原始曲线是比较保守的选择。
    """
    if signal_length < 3:
        return 1

    window_size = max(3, int(window_size))
    window_size = min(window_size, signal_length if signal_length % 2 == 1 else signal_length - 1)
    if window_size % 2 == 0:
        window_size -= 1
    return max(1, window_size)


def rolling_median(values: np.ndarray, window_size: int) -> np.ndarray:
    """计算一维数组的局部中值曲线。

    使用 pandas 的 rolling median，边界位置允许使用较少点数，避免首尾出现 NaN。
    """
    return (
        pd.Series(values)
        .rolling(window=window_size, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def replace_masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把被标记为异常的点替换为邻近正常点的线性插值结果。

    如果整条曲线都被标记为异常，则直接返回原值，避免生成全 NaN 或无意义结果。
    """
    if not np.any(mask):
        return values

    indices = np.arange(len(values))
    valid_indices = indices[~mask]
    if len(valid_indices) == 0:
        return values

    cleaned = values.copy()
    cleaned[mask] = np.interp(indices[mask], valid_indices, values[~mask])
    return cleaned


def filter_signal_outliers(
    signal: np.ndarray,
    window_size: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """过滤双通道曲线中的孤立异常点。

    方法：
    1. 对每个通道计算局部中值曲线，作为该位置附近的正常趋势。
    2. 计算原始值与局部中值的偏差 residual。
    3. 用 MAD（Median Absolute Deviation，中位数绝对偏差）估计偏差尺度。
    4. 偏差超过 threshold * MAD 的点视作异常点。
    5. 异常点不删除，而是用邻近正常点线性插值替换。

    返回值：
    - cleaned_signal：过滤后的双通道曲线。
    - outlier_mask：形状同 signal 的布尔数组，True 表示该点曾被判定为异常。
    """
    if threshold <= 0:
        raise ValueError("--outlier-threshold 必须大于 0。")

    window_size = normalize_window_size(window_size, len(signal))
    cleaned_channels: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for channel_index in range(signal.shape[1]):
        values = signal[:, channel_index]
        trend = rolling_median(values, window_size)
        residual = values - trend
        mad = np.median(np.abs(residual - np.median(residual)))

        # 1.4826 * MAD 近似高斯分布标准差；兜底值避免完全平滑曲线除零。
        robust_sigma = max(1.4826 * mad, np.ptp(values) * 1e-6, 1e-12)
        mask = np.abs(residual) > threshold * robust_sigma

        cleaned_channels.append(replace_masked_values(values, mask))
        masks.append(mask)

    return np.column_stack(cleaned_channels), np.column_stack(masks)


def load_signal_pair(
    df: pd.DataFrame,
    class_name: str,
    time_column: str,
    filter_outliers: bool,
    outlier_window: int,
    outlier_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """读取某个类别的一组 high/low 双通道曲线。

    返回值：
    - time_values：有效时间点数组。
    - signal：形状为 (有效点数, 2) 的数组，第 0 列 high，第 1 列 low。
    """
    high_column = f"{class_name}_high"
    low_column = f"{class_name}_low"
    required = [time_column, high_column, low_column]
    # 只保留时间、高通道、低通道都不为空的行。
    valid = df[required].notna().all(axis=1)
    subset = df.loc[valid, required]

    time_values = subset[time_column].to_numpy(dtype=float)
    signal = subset[[high_column, low_column]].to_numpy(dtype=float)
    if len(time_values) < 8:
        raise ValueError(f"{class_name} 的有效点数太少。")

    if filter_outliers:
        signal, outlier_mask = filter_signal_outliers(
            signal=signal,
            window_size=outlier_window,
            threshold=outlier_threshold,
        )
        high_count = int(outlier_mask[:, 0].sum())
        low_count = int(outlier_mask[:, 1].sum())
        print(
            f"{class_name}: 已过滤异常点 high={high_count}, low={low_count} "
            f"(window={normalize_window_size(outlier_window, len(signal))}, "
            f"threshold={outlier_threshold})"
        )

    return time_values, signal


def resample_signal(
    time_values: np.ndarray,
    signal: np.ndarray,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把原始曲线插值到固定长度。

    不同类别或原始文件的有效点数可能不同，模型训练通常要求输入长度一致，
    所以这里用线性插值统一成 target_length 个点。
    """
    target_time = np.linspace(time_values[0], time_values[-1], target_length)
    high = np.interp(target_time, time_values, signal[:, 0])
    low = np.interp(target_time, time_values, signal[:, 1])
    return target_time, np.column_stack([high, low])


def smooth_noise(
    rng: np.random.Generator,
    length: int,
    scale: float,
    control_points: int = 8,
) -> np.ndarray:
    """生成一条平滑漂移噪声。

    做法是先生成少量控制点上的随机值，再插值到完整长度。
    这种噪声比逐点白噪声更平滑，模拟缓慢漂移。
    """
    base_x = np.linspace(0.0, 1.0, control_points)
    full_x = np.linspace(0.0, 1.0, length)
    base_y = rng.normal(0.0, scale, size=control_points)
    return np.interp(full_x, base_x, base_y)


def augment_sample(
    sample: np.ndarray,
    rng: np.random.Generator,
    noise_scale: float,
    drift_scale: float,
) -> np.ndarray:
    """对一个双通道样本做轻量增强。

    增强包含两部分：
    - point_noise：每个点独立的小随机扰动。
    - drift_noise：沿时间方向缓慢变化的平滑扰动。

    最后会用 clip 限制增强结果，避免噪声把曲线推到原始范围太远之外。
    """
    channel_range = np.ptp(sample, axis=0)
    # 如果某个通道几乎没有变化，使用 1.0 作为安全范围，避免除零或噪声为 NaN。
    safe_range = np.where(channel_range > 1e-12, channel_range, 1.0)

    point_noise = rng.normal(0.0, noise_scale, size=sample.shape) * safe_range
    drift_noise = np.column_stack(
        [
            smooth_noise(rng, len(sample), scale=drift_scale * safe_range[0]),
            smooth_noise(rng, len(sample), scale=drift_scale * safe_range[1]),
        ]
    )

    augmented = sample + point_noise + drift_noise

    # 限制增强曲线的上下界：原始最小/最大值外再放宽 0.2%。
    min_values = np.min(sample, axis=0) - safe_range * 0.002
    max_values = np.max(sample, axis=0) + safe_range * 0.002
    return np.clip(augmented, min_values, max_values)


def build_dataset(
    df: pd.DataFrame,
    time_column: str,
    samples_per_class: int,
    target_length: int,
    rng: np.random.Generator,
    noise_scale: float,
    drift_scale: float,
    filter_outliers: bool,
    outlier_window: int,
    outlier_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """根据原始总表生成完整增强数据集。

    返回值：
    - x_data：形状为 (样本数, 序列长度, 2) 的双通道数据。
    - y_data：每个样本对应的整数类别标签。
    - class_names：标签编号到类别名的映射。
    - sample_names：每个样本的内部名称。
    - time_axis：重采样后的参考时间轴。
    """
    class_names = discover_classes(df)
    if not class_names:
        raise ValueError("没有识别到 xxx_high / xxx_low 格式的类别。")

    x_samples: list[np.ndarray] = []
    y_labels: list[int] = []
    sample_names: list[str] = []
    time_axis_reference: np.ndarray | None = None

    for class_index, class_name in enumerate(class_names):
        time_values, signal = load_signal_pair(
            df=df,
            class_name=class_name,
            time_column=time_column,
            filter_outliers=filter_outliers,
            outlier_window=outlier_window,
            outlier_threshold=outlier_threshold,
        )
        time_axis, base_sample = resample_signal(time_values, signal, target_length)

        # 每个类别先放入 1 个原始样本，再生成 num-1 个增强样本。
        if time_axis_reference is None:
            time_axis_reference = time_axis

        x_samples.append(base_sample)
        y_labels.append(class_index)
        sample_names.append(f"{class_name}_original")

        for sample_id in range(samples_per_class - 1):
            augmented = augment_sample(
                base_sample,
                rng,
                noise_scale=noise_scale,
                drift_scale=drift_scale,
            )
            x_samples.append(augmented)
            y_labels.append(class_index)
            sample_names.append(f"{class_name}_aug_{sample_id + 1:03d}")

    if time_axis_reference is None:
        raise ValueError("未能生成时间轴。")

    return (
        np.stack(x_samples),
        np.asarray(y_labels, dtype=np.int64),
        np.asarray(class_names),
        np.asarray(sample_names),
        time_axis_reference,
    )


def save_sample_excels(
    output_dir: Path,
    x_data: np.ndarray,
    class_names: np.ndarray,
    sample_names: np.ndarray,
    y_data: np.ndarray,
) -> list[Path]:
    """把增强后的样本按类别保存为单独 Excel 文件。

    输出结构：
    data/
      cloth/cloth_001.xlsx
      cloth/cloth_002.xlsx
      leather/leather_001.xlsx

    注意：每次写入某个类别前，会清理该类别目录下旧的 xlsx 文件。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []
    class_counters: dict[str, int] = {str(class_name): 0 for class_name in class_names}
    prepared_dirs: set[str] = set()

    for sample_index, sample_name in enumerate(sample_names):
        label_index = int(y_data[sample_index])
        class_name = str(class_names[label_index])
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        if class_name not in prepared_dirs:
            # 避免新旧样本混在一起，首次进入某类别时清空旧 Excel。
            for old_file in class_dir.glob("*.xlsx"):
                old_file.unlink()
            for lock_file in class_dir.glob(".~lock.*.xlsx#"):
                lock_file.unlink()
            prepared_dirs.add(class_name)

        class_counters[class_name] += 1
        file_path = class_dir / f"{class_name}_{class_counters[class_name]:03d}.xlsx"
        sample_df = pd.DataFrame(
            {
                f"{class_name}_high": x_data[sample_index, :, 0],
                f"{class_name}_low": x_data[sample_index, :, 1],
            }
        )
        sample_df.to_excel(file_path, index=False)
        written_files.append(file_path)

    return written_files


def main() -> None:
    """脚本入口：读取 Excel、生成增强数据、导出样本文件。"""
    parser = build_parser()
    args = parser.parse_args()

    # 基础参数校验，尽早给出清晰错误信息。
    excel_path = Path(args.excel_path).expanduser().resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {excel_path}")
    if args.num < 1:
        raise ValueError("-num 至少为 1。")
    if args.target_length < 16:
        raise ValueError("--target-length 至少为 16。")
    if args.noise_scale < 0 or args.drift_scale < 0:
        raise ValueError("--noise-scale 和 --drift-scale 必须为非负数。")
    if args.outlier_window < 1:
        raise ValueError("--outlier-window 至少为 1。")
    if args.outlier_threshold <= 0:
        raise ValueError("--outlier-threshold 必须大于 0。")

    # 读取原始 Excel，并清理空列和 pandas 自动生成的 Unnamed 列。
    df = pd.read_excel(excel_path)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

    # 固定随机种子后，同样参数可以复现同样的增强结果。
    rng = np.random.default_rng(args.seed)
    x_data, y_data, class_names, sample_names, time_axis = build_dataset(
        df=df,
        time_column=args.time_column,
        samples_per_class=args.num,
        target_length=args.target_length,
        rng=rng,
        noise_scale=args.noise_scale,
        drift_scale=args.drift_scale,
        filter_outliers=not args.no_outlier_filter,
        outlier_window=args.outlier_window,
        outlier_threshold=args.outlier_threshold,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    written_files = save_sample_excels(
        output_dir,
        x_data,
        class_names,
        sample_names,
        y_data,
    )

    print(f"类别: {class_names.tolist()}")
    print(f"X 形状: {x_data.shape}，格式为 (样本数, 序列长度, 通道数)")
    print(f"y 形状: {y_data.shape}")
    print(f"噪声比例: noise_scale={args.noise_scale}, drift_scale={args.drift_scale}")
    print(
        "异常点过滤: "
        f"{'关闭' if args.no_outlier_filter else '开启'}"
        f"，window={args.outlier_window}, threshold={args.outlier_threshold}"
    )
    print(f"输出目录: {output_dir}")
    print(f"已写入文件数: {len(written_files)}")


if __name__ == "__main__":
    main()
