# Stage 1: Builder - Install heavy dependencies
FROM python:3.12 AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final lightweight image
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

COPY requirements.txt .
COPY dvc.yaml dvc.lock ./
COPY src/ ./src/
COPY config/ ./config/
COPY .dvc/ ./.dvc/

RUN mkdir -p data/raw data/prepared data/models

ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

CMD ["bash"]
