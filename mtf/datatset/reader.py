"""数据读取与可视化工具。

这个脚本有两种使用方式：
1. 可视化单个总表 Excel：
   python datatset/reader.py data.xlsx --x-column time_s

2. 批量可视化 data 目录下的单样本 Excel：
   python datatset/reader.py --data-dir data --batch-size 20

数据格式约定：
- 双通道类别列必须命名为 xxx_high / xxx_low，例如 cloth_high / cloth_low。
- 单样本 Excel 通常只包含一种类别的 high/low 两列。
"""

from __future__ import annotations

import argparse
import keyword
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def normalize_column_name(column_name: str, used_names: set[str]) -> str:
    """把 Excel 列名转换成安全的 Python 变量名。

    例如 `Cloth High` 会变成 `cloth_high`。如果列名重复，会自动追加后缀，
    避免生成重复变量名。
    """
    normalized = re.sub(r"\W+", "_", str(column_name).strip()).strip("_").lower()

    # 处理空列名、数字开头、Python 关键字等不能直接当变量名的情况。
    if not normalized:
        normalized = "column"
    if normalized[0].isdigit():
        normalized = f"col_{normalized}"
    if keyword.iskeyword(normalized):
        normalized = f"{normalized}_col"

    candidate = normalized
    suffix = 1
    while candidate in used_names:
        suffix += 1
        candidate = f"{normalized}_{suffix}"

    used_names.add(candidate)
    return candidate


def read_excel_columns(
    file_path: Path,
    sheet_name: str | int | None = 0,
    drop_empty_columns: bool = True,
    drop_unnamed_columns: bool = True,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """读取 Excel，并把每一列转换成列表。

    返回值：
    - df：清理后的完整表格，供后续按类别拆分和绘图使用。
    - column_arrays：以规范化列名为 key、列数据列表为 value 的字典。
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Excel 经常会带全空列或 pandas 自动生成的 Unnamed 列，这里默认丢弃。
    if drop_empty_columns:
        df = df.dropna(axis=1, how="all")
    if drop_unnamed_columns:
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

    column_arrays: dict[str, list] = {}
    used_names: set[str] = set()

    for column in df.columns:
        variable_name = normalize_column_name(str(column), used_names)
        column_arrays[variable_name] = df[column].dropna().tolist()

    return df, column_arrays


def assign_column_variables(column_arrays: dict[str, list], namespace: dict) -> None:
    """把列数组写入指定命名空间。

    当前脚本把它写入 globals()，这样运行后可得到 cloth_high、cloth_low
    之类的全局变量，方便调试或交互式查看。
    """
    namespace.update(column_arrays)


def build_class_tables(
    df: pd.DataFrame,
    x_column: str | None = None,
) -> dict[str, pd.DataFrame]:
    """把总表按 `xxx_high / xxx_low` 拆成多个类别表。

    每个类别表都包含三列：
    - x 轴列：用户指定的时间列，或默认 row_index。
    - high：该类别高通道数据。
    - low：该类别低通道数据。
    """
    if df.empty:
        raise ValueError("Excel 文件中没有可处理数据。")

    if x_column and x_column not in df.columns:
        raise ValueError(f"x 轴列 `{x_column}` 不存在，可选列: {list(df.columns)}")

    if x_column:
        x_values = pd.to_numeric(df[x_column], errors="coerce")
        x_label = x_column
    else:
        x_values = pd.Series(range(len(df)))
        x_label = "row_index"

    class_tables: dict[str, pd.DataFrame] = {}
    for column in df.columns:
        # 只有同时存在 xxx_high 和 xxx_low 的列组才会被当作一个类别。
        if column == x_column or not isinstance(column, str) or not column.endswith("_high"):
            continue

        class_name = column[: -len("_high")]
        low_column = f"{class_name}_low"
        if low_column not in df.columns:
            continue

        table = pd.DataFrame(
            {
                x_label: x_values,
                "high": pd.to_numeric(df[column], errors="coerce"),
                "low": pd.to_numeric(df[low_column], errors="coerce"),
            }
        ).dropna()
        if table.empty:
            continue

        class_tables[class_name] = table

    if not class_tables:
        raise ValueError("没有识别到形如 xxx_high / xxx_low 的双通道类别数据。")

    return class_tables


def plot_class_tables(
    class_tables: dict[str, pd.DataFrame],
    output_path: Path,
    x_label: str,
    show_plot: bool = False,
) -> list[str]:
    """把一个总表中的各类别 high/low 曲线画到同一张图片中。

    默认布局是 2x2，适合当前 cloth/leather/metal/wood 四类数据。
    """
    class_names = list(class_tables.keys())
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False, sharey=False)
    flat_axes = axes.flatten()

    for axis, class_name in zip(flat_axes, class_names):
        table = class_tables[class_name]
        axis.plot(table[x_label], table["high"], label="high", linewidth=0.6)
        axis.plot(table[x_label], table["low"], label="low", linewidth=0.6)
        axis.set_title(class_name)
        axis.set_xlabel(x_label)
        axis.set_ylabel("value")
        axis.grid(alpha=0.25)
        axis.legend()

    for axis in flat_axes[len(class_names) :]:
        axis.axis("off")

    figure.suptitle("Class-wise Dual-channel Visualization")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close(figure)

    return class_names


def iter_sample_excel_files(data_dir: Path) -> dict[str, list[Path]]:
    """扫描 data 目录，按类别收集单样本 Excel 文件。

    期望目录结构：
    data/
      cloth/cloth_001.xlsx
      leather/leather_001.xlsx
      metal/metal_001.xlsx
      wood/wood_001.xlsx
    """
    class_files: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        excel_files = sorted(class_dir.glob("*.xlsx"))
        if excel_files:
            class_files[class_dir.name] = excel_files

    if not class_files:
        raise ValueError(f"没有在 {data_dir} 下找到类别子目录和 xlsx 文件。")

    return class_files


def read_sample_table(file_path: Path, sheet_name: str | int | None = 0) -> tuple[str, pd.DataFrame]:
    """读取一个单样本 Excel，并返回类别名和 high/low 表格。

    单样本文件应只包含一组双通道数据，例如 cloth_high / cloth_low。
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
    class_tables = build_class_tables(df)

    if len(class_tables) != 1:
        raise ValueError(
            f"{file_path} 中识别到 {len(class_tables)} 组双通道数据，单样本文件应只有一组。"
        )

    class_name, table = next(iter(class_tables.items()))
    return class_name, table


def chunked(items: list[Path], size: int) -> list[list[Path]]:
    """把文件列表按固定数量切分，用于“每 N 个样本保存一张图”。"""
    return [items[index : index + size] for index in range(0, len(items), size)]


def plot_sample_batch(
    class_name: str,
    files: list[Path],
    output_path: Path,
    sheet_name: str | int | None = 0,
) -> None:
    """把同一类别的一批样本画到一张图片中。

    每个样本占一个子图；每个子图里有 high 和 low 两条曲线。
    默认每行 5 个子图，所以 batch_size=20 时会形成 4x5 布局。
    """
    columns = 5
    rows = math.ceil(len(files) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 2.4))
    flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, file_path in zip(flat_axes, files):
        # x_column 通常是 row_index；如果样本文件以后带时间列，也能自动使用第一列。
        detected_class, table = read_sample_table(file_path, sheet_name=sheet_name)
        x_column = table.columns[0]
        axis.plot(table[x_column], table["high"], label="high", linewidth=0.6)
        axis.plot(table[x_column], table["low"], label="low", linewidth=0.6)
        axis.set_title(f"{detected_class}: {file_path.stem}", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.25)

    for axis in flat_axes[len(files) :]:
        axis.axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="upper right")
    figure.suptitle(f"{class_name} samples", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_data_dir_batches(
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
    sheet_name: str | int | None = 0,
) -> list[Path]:
    """批量可视化 data 目录。

    输出规则：
    - 按类别分别建目录。
    - 每 batch_size 个样本保存成一张 PNG。
    - 文件名使用样本编号范围，例如 cloth_001_020.png。
    """
    if batch_size < 1:
        raise ValueError("--batch-size 至少为 1。")

    written_files: list[Path] = []
    for class_name, files in iter_sample_excel_files(data_dir).items():
        for batch_index, batch_files in enumerate(chunked(files, batch_size), start=1):
            start_number = (batch_index - 1) * batch_size + 1
            end_number = start_number + len(batch_files) - 1
            output_path = (
                output_dir
                / class_name
                / f"{class_name}_{start_number:03d}_{end_number:03d}.png"
            )
            plot_sample_batch(
                class_name=class_name,
                files=batch_files,
                output_path=output_path,
                sheet_name=sheet_name,
            )
            written_files.append(output_path)

    return written_files


def build_parser() -> argparse.ArgumentParser:
    """定义命令行参数。

    常用命令：
    - 单文件可视化：python datatset/reader.py data.xlsx --x-column time_s
    - 批量可视化：python datatset/reader.py --data-dir data --batch-size 20
    """
    parser = argparse.ArgumentParser(
        description="从 Excel 读取各列数据，将列名映射为数组变量，并绘制到同一张图中。"
    )
    parser.add_argument("excel_path", nargs="?", default="data.xlsx", help="Excel 文件路径")
    parser.add_argument("--sheet", default=0, help="sheet 名称或索引，默认第 1 个 sheet")
    parser.add_argument(
        "--x-column",
        default=None,
        help="指定作为 x 轴的列名；未指定时使用行号作为 x 轴",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出图片路径，默认保存在 Excel 同目录下",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="保存图片后同时显示图表窗口",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="批量可视化 data 目录；目录下应按类别存放单样本 xlsx",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="批量模式下每张图片包含的样本图数量，默认 20",
    )
    parser.add_argument(
        "--batch-output-dir",
        default=None,
        help="批量图片输出目录，默认保存到 data_visualization",
    )
    return parser


def resolve_excel_path(raw_path: str) -> Path:
    """兼容 `excel_path=xxx.xlsx` 这种传参写法，并返回绝对路径。"""
    if raw_path.startswith("excel_path="):
        raw_path = raw_path.split("=", 1)[1]
    return Path(raw_path).expanduser().resolve()


def main() -> None:
    """脚本入口。

    如果传入 --data-dir，执行批量样本可视化；否则执行单个 Excel 总表可视化。
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.data_dir:
        # 批量模式：读取 data/<类别>/*.xlsx，每 batch_size 个样本合成一张图。
        data_dir = Path(args.data_dir).expanduser().resolve()
        if not data_dir.exists():
            raise FileNotFoundError(f"data 目录不存在: {data_dir}")
        output_dir = (
            Path(args.batch_output_dir).expanduser().resolve()
            if args.batch_output_dir
            else data_dir.with_name(f"{data_dir.name}_visualization")
        )
        written_files = plot_data_dir_batches(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=args.batch_size,
            sheet_name=args.sheet,
        )
        print(f"已批量读取目录: {data_dir}")
        print(f"每张图片包含样本数: {args.batch_size}")
        print(f"批量图表输出目录: {output_dir}")
        print(f"已写入图片数: {len(written_files)}")
        return

    # 单文件模式：读取一个总表 Excel，把所有类别画到一张图中。
    excel_path = resolve_excel_path(args.excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel 文件不存在: {excel_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else excel_path.with_name(f"{excel_path.stem}_all_columns.png")
    )

    x_label = args.x_column or "row_index"
    df, column_arrays = read_excel_columns(excel_path, sheet_name=args.sheet)
    assign_column_variables(column_arrays, globals())
    class_tables = build_class_tables(
        df=df,
        x_column=args.x_column,
    )
    plotted_classes = plot_class_tables(
        class_tables=class_tables,
        output_path=output_path,
        x_label=x_label,
        show_plot=args.show,
    )

    print(f"已读取文件: {excel_path}")
    print("已创建数组变量:")
    for variable_name, values in column_arrays.items():
        preview = values[:5]
        print(f"  {variable_name}: 长度={len(values)}, 前5项={preview}")

    print("已创建分类表格:")
    for class_name, table in class_tables.items():
        variable_name = f"{class_name}_table"
        globals()[variable_name] = table
        print(f"  {variable_name}: 形状={table.shape}, 列={table.columns.tolist()}")

    print(f"已绘制类别: {plotted_classes}")
    print(f"图表已保存到: {output_path}")


if __name__ == "__main__":
    main()
