FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /app/backend
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libffi-dev \
    liblapack-dev \
    llvm-15 \
    llvm-15-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LLVM_CONFIG=/usr/bin/llvm-config-15

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install spotdl

EXPOSE 5000
EXPOSE 5001

COPY . .