FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 创建必要的目录
RUN mkdir -p app models

# 复制应用文件（从构建上下文根目录）
COPY tts-project-gpu/app/ app/
COPY tts-project-gpu/requirements.txt .

# 复制models目录
COPY models/ models/

# 安装Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3.10-pip python3.10-dev

# 设置Python 3.10为默认
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.10 1

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3", "app/main.py"]