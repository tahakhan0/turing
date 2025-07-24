# Use an official Python runtime as a parent image (pinned to specific SHA256)
# python:3.10.11-slim
FROM python@sha256:fd86924ba14682eb11a3c244f60a35b5dfe3267cbf26d883fb5c14813ce926f1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for face_recognition, OpenCV, segmentation and other libraries
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
    libgomp1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install specific versions of pip and uv for reproducibility
RUN pip install --no-cache-dir pip==23.3.2 uv==0.1.18

# Create directories for models
RUN mkdir -p /app/models /app/segmentation_models

# Download the YOLOv8n model with integrity check
RUN wget -O /app/models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt && \
    echo "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95  /app/models/yolov8n.pt" | sha256sum -c

# Download segmentation model weights with integrity checks
RUN wget -O /app/segmentation_models/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth && \
    echo "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799  /app/segmentation_models/groundingdino_swint_ogc.pth" | sha256sum -c

RUN wget -O /app/segmentation_models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && \
    echo "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e  /app/segmentation_models/sam_vit_h_4b8939.pth" | sha256sum -c

# Clone Grounded-SAM repository for dependencies
RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git /app/Grounded-Segment-Anything

# Copy the requirements file and install Python packages using uv
COPY requirements.txt .
RUN uv pip install --no-cache --system -r requirements.txt

# Install Grounded-SAM dependencies
RUN cd /app/Grounded-Segment-Anything && \
    uv pip install --no-cache --system -e . && \
    uv pip install --no-cache --system -e segment_anything && \
    uv pip install --no-cache --system -e GroundingDINO

# Set Python path for Grounded-SAM imports
ENV PYTHONPATH="/app/Grounded-Segment-Anything:/app/Grounded-Segment-Anything/GroundingDINO:$PYTHONPATH"

# Copy the application code from the src directory into the container
COPY ./src /app/src

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
