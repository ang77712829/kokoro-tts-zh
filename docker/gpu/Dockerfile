FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 创建必要的目录
RUN mkdir -p app models app/static app/templates

# 复制应用文件（从构建上下文根目录）
COPY tts-project-gpu/app/ app/
COPY tts-project-gpu/requirements.txt .

# 复制models目录
COPY models/ models/

# 安装Python 3.10及相关工具（关键修复：添加python3.10-venv）
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip python3.10-dev python3.10-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 创建并激活虚拟环境
RUN python3.10 -m venv /app/venv && \
    . /app/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python3.10", "app/main.py"]
