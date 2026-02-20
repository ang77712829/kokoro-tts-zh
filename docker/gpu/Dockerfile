FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 创建必要的目录
RUN mkdir -p app models

# 复制应用文件
COPY app/ app/
COPY requirements.txt .

# 尝试复制models目录（如果存在）
COPY --chown=root:root models/ models/ 2>/dev/null || true

RUN apt-get update && apt-get install -y python3 python3-pip python3-dev
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3", "app/main.py"]