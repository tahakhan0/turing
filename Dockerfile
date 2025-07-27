# Use an official Python runtime as a parent image (pinned to specific SHA256)
# python:3.10.11-slim
FROM python@sha256:fd86924ba14682eb11a3c244f60a35b5dfe3267cbf26d883fb5c14813ce926f1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for face_recognition, OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install specific versions of pip and uv for reproducibility
RUN pip install --no-cache-dir pip==23.3.2 uv==0.1.18

# Create a directory for the YOLO model
RUN mkdir -p /app/models

# Download the YOLOv8n model with integrity check
RUN wget -O /app/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt && \
    echo "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95  /app/models/yolov8n.pt" | sha256sum -c

# Copy the requirements file and install Python packages using uv
COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt

# Copy the application code from the src directory into the container
COPY ./src /app/src

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]