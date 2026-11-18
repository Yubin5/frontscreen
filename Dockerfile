# API server image (api_server.py, rectify.py, screen_ocr.py); the screen detector/SAM runs in sam/ (see run_api.sh)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# pinned to the tested versions: paddlepaddle 3.3.x crashes on CPU (oneDNN), and OpenCV 5 changed the line
# segment detector's output shape
RUN pip install --no-cache-dir "paddlepaddle==3.2.2" "paddleocr==3.7.0" "paddlex==3.7.2" \
        "opencv-contrib-python==4.10.0.84" "numpy==2.3.5"
# OCR models are downloaded on first start into /root/.paddlex (run_api.sh mounts cache/paddlex there)
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True PYTHONUNBUFFERED=1
WORKDIR /lab
