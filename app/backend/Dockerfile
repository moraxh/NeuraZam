FROM python:alpine

WORKDIR /app/backend

RUN apk add --no-cache ffmpeg

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install spotdl

COPY . .