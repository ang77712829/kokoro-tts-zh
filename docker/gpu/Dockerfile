FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

COPY app/ app/
COPY requirements.txt .
COPY models/ models/ 2>/dev/null || mkdir -p models/

RUN apt-get update && apt-get install -y python3 python3-pip python3-dev
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3", "app/main.py"]