FROM python:3.9-slim

WORKDIR /app

COPY app/ app/
COPY requirements.txt .
COPY models/ models/ 2>/dev/null || mkdir -p models/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app/main.py"]