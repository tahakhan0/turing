# Use an official Python runtime as a parent image
FROM python:3.10.11-slim

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

# Install uv, a fast Python package installer
RUN pip install --upgrade pip uv

# Create a directory for the YOLO model
RUN mkdir -p /app/models

# Download the YOLOv8n model into the models directory
RUN wget -O /app/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Copy the requirements file and install Python packages using uv
COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt

# Copy the application code from the src directory into the container
COPY ./src /app/src

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
