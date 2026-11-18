FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# CPU-only torch keeps the image small; SAM ViT-B runs in ~1-2s per photo on CPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir transformers opencv-python-headless pillow
ENV PYTHONUNBUFFERED=1
WORKDIR /lab/sam
