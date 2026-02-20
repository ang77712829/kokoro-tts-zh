FROM python:3.10-slim

WORKDIR /app

# 创建必要的目录
RUN mkdir -p app models

# 复制应用文件（从构建上下文根目录）
COPY tts-project-cpu/app/ app/
COPY tts-project-cpu/requirements.txt .

# 复制models目录
COPY models/ models/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app/main.py"]