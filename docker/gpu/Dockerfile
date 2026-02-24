# 第一阶段：构建环境
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 as builder

WORKDIR /app

# 安装Python 3.10及相关工具
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip python3.10-dev python3.10-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 复制requirements.txt
COPY tts-project-gpu/requirements.txt .

# 创建并激活虚拟环境
RUN python3.10 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --no-cache-dir --timeout=120 -r requirements.txt && \
    pip cache purge

# 第二阶段：运行环境
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 安装必要的运行时依赖
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境和应用文件
COPY --from=builder /app/venv /app/venv

# 创建必要的目录
RUN mkdir -p app models app/static app/templates

# 复制应用文件
COPY tts-project-gpu/app/ app/

# 复制models目录
COPY models/ models/

EXPOSE 8000

# 激活虚拟环境并运行应用
CMD ["bash", "-c", ". /app/venv/bin/activate && python3.10 app/main.py"]
