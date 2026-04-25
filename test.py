"""NumPy 维度转换示例。

这个文件只是一个最小测试脚本，用来演示如何把形状数组反转。
例如图像形状常见写法是 (C, H, W)，反转后会得到 (W, H, C)。
"""

import numpy as np
import logging as log

# 设置日志级别为 INFO，这样 log.info(...) 会输出到终端。
log.basicConfig(level=log.INFO)

# 示例：假设 img 表示一个 3 通道、480x480 的图像形状。
img = np.array([3, 480, 480])

# 使用 [::-1] 反转数组顺序，并转成 tuple。
resize_img = tuple(img[::-1])

# 输出结果：(480, 480, 3)
log.info(f"{resize_img}")
