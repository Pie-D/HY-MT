FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV TORCH_HOME=/models
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 hymt

WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    fastapi \
    uvicorn

COPY app.py /app/app.py

# 🔴 FIX ở đây
RUN mkdir -p /models && \
    chown -R hymt:hymt /app /models

USER hymt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
