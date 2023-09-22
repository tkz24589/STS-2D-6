# 设置基础映像
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
COPY . /app
WORKDIR /app
# 切换ubuntu源
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get clean

# 更新软件包列表并安装所需的依赖项
RUN apt-get update && apt-get install -y \
    software-properties-common \
    libgl1-mesa-glx \
    && add-apt-repository ppa:deadsnakes/ppa

# 安装Python 3.9
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils

RUN rm /usr/bin/python3

# # 创建符号链接以使用python3和pip3命令
RUN ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/python3.9 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/local/bin/python

# 验证Python安装
RUN python3 --version

# 安装pip
RUN apt-get install -y python3-pip

# 验证pip安装
RUN pip3 --version

# 安装python依赖
RUN python3 -m pip --no-cache-dir install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
