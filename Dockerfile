# API server image (api_server.py, rectify.py): OpenCV + NumPy only; the models run in sam/ (see run_api.sh)
FROM python:3.11-slim
# pinned to the tested versions (OpenCV 5 changed the line segment detector's output shape)
RUN pip install --no-cache-dir numpy==2.3.5 opencv-python-headless==4.10.0.84
ENV PYTHONUNBUFFERED=1
WORKDIR /lab
