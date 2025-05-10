FROM python:3.10-slim

WORKDIR /app/backend

RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    libffi-dev \
    python3-dev \
    liblapack-dev \
    llvm-15 \
    llvm-15-dev \
    && apt-get clean

ENV LLVM_CONFIG=/usr/bin/llvm-config-15

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install spotdl

COPY . .