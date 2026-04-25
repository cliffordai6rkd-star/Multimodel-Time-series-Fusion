# 训练容器基础镜像。
# 当前 Dockerfile 只是在已有镜像 dp-hirol:train 的基础上启动环境；
# 如果后续需要安装 Python 依赖、复制代码或设置工作目录，可以继续在这里追加 RUN/COPY/WORKDIR。
FROM docker-for-train:v1

# pandas 读取 .xlsx 训练样本需要 openpyxl。
RUN python -m pip install openpyxl
