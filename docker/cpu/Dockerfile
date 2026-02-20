FROM python:3.9-slim

WORKDIR /app

# 创建必要的目录
RUN mkdir -p app models

# 复制应用文件
COPY app/ app/
COPY requirements.txt .

# 尝试复制models目录（如果存在）
COPY --chown=root:root models/ models/ 2>/dev/null || true

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app/main.py"]